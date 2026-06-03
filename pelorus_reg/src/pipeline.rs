use std::sync::{Arc, Mutex};

use crate::{ActiveBackend, Registration, registration::RegistrationState};
use anyhow::anyhow;
use either::Either;
use nalgebra::{Isometry3, Translation3, UnitQuaternion};
use pelorus_core::{
    ImuMsg, ImuPosePrediction, ImuPredictor, ImuPredictorUndistort, KinematicModel, LidarVendor,
    LioConfig, OdomMsg, PointXYZIT, PrelioProcessedCloud,
};
use prelio::{LocatedBufferView, Prelio};
use spole::{IVoxHashMap, IVoxNearbyType};

#[cfg(feature = "cuda")]
const CUDA_CLOUD_BUFFER_PIPELINE_DEPTH: usize = 2;

#[cfg(feature = "cuda")]
struct CudaCloudUploadBuffers {
    buffers: [Option<LocatedBufferView<PointXYZIT>>; CUDA_CLOUD_BUFFER_PIPELINE_DEPTH],
    next_slot: usize,
}

#[cfg(feature = "cuda")]
impl Default for CudaCloudUploadBuffers {
    fn default() -> Self {
        Self {
            buffers: [None, None],
            next_slot: 0,
        }
    }
}

#[cfg(feature = "cuda")]
impl CudaCloudUploadBuffers {
    fn next_buffer_mut(
        &mut self,
        stream: &pelorus_cuda::Stream,
        len: usize,
    ) -> &mut LocatedBufferView<PointXYZIT> {
        let slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % CUDA_CLOUD_BUFFER_PIPELINE_DEPTH;

        let cloud_view = self.buffers[slot]
            .get_or_insert_with(|| LocatedBufferView::alloc_points_cuda(stream, len));
        cloud_view.realloc_if_needed_cuda(stream, len);
        cloud_view
    }
}

pub struct Pipeline {
    map: Arc<Mutex<RegistrationState>>,
    ext_lidar_to_imu: Isometry3<f64>,
    pub config: LioConfig,
    active_backend: ActiveBackend,
    #[cfg(feature = "cuda")]
    cuda_cloud_buffers: Mutex<CudaCloudUploadBuffers>,
    #[cfg(feature = "metal")]
    metal_cloud_buffer: Mutex<Option<LocatedBufferView<PointXYZIT>>>,
    #[cfg(feature = "vulkan")]
    vulkan_cloud_buffer: Mutex<Option<LocatedBufferView<PointXYZIT>>>,
}

/// Result from cloud registration and mapping
///
/// # Frame Conventions
/// - **World Frame (W)**: Fixed global reference frame, aligned with IMU orientation at
///   initialization. Z-axis points upward (anti-gravity).
/// - **IMU Frame (I)**: Body-fixed frame attached to the IMU sensor
/// - **Lidar Frame (L)**: Sensor-fixed frame attached to the LiDAR
#[derive(Debug, Clone)]
pub struct PelorusOutput {
    pub scan_id: u64,
    pub timestamp: ros_pointcloud2::ros::TimeMsg,
    pub pose: Isometry3<f64>, // in
    pub pose_covariance: [f64; 36],
    pub linear_velocity: nalgebra::Vector3<f64>,
    pub angular_velocity: nalgebra::Vector3<f64>,
    pub twist_covariance: [f64; 36],
    pub cloud: Vec<PointXYZIT>,
    pub frame_id: String,
}

impl Pipeline {
    fn ensure_backend_map_initialized(&self) {
        match &self.active_backend {
            ActiveBackend::Cpu => {}
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(stream) => {
                if pelorus_cuda::get_map_size() == 0 {
                    pelorus_cuda::init_map(
                        stream,
                        self.config.map_capacity,
                        self.config.map_resolution as f32,
                        self.config.dist_update_sensitivity,
                        (self.config.map_resolution as f32) / 4.0,
                        pelorus_cuda::NeighborType::Neighbor26,
                    );
                }
            }
            #[cfg(feature = "metal")]
            ActiveBackend::Metal(stream) => {
                if pelorus_metal::get_map_size() == 0 {
                    pelorus_metal::init_map(
                        stream,
                        self.config.map_capacity,
                        self.config.map_resolution as f32,
                        self.config.dist_update_sensitivity,
                        (self.config.map_resolution as f32) / 4.0,
                        pelorus_metal::NeighborType::Neighbor26,
                    );
                }
            }
            #[cfg(feature = "vulkan")]
            ActiveBackend::Vulkan(stream) => {
                if pelorus_vulkan::get_map_size() == 0 {
                    pelorus_vulkan::init_map(
                        stream,
                        self.config.map_capacity,
                        self.config.map_resolution as f32,
                        self.config.dist_update_sensitivity,
                        (self.config.map_resolution as f32) / 4.0,
                        pelorus_vulkan::NeighborType::Neighbor26,
                    );
                }
            }
        }
    }

    pub fn new(config: &LioConfig) -> Self {
        let ext_lidar_to_imu = Isometry3::from_parts(
            Translation3::new(
                config.imu_params.initial_translation_lidar_to_imu.x,
                config.imu_params.initial_translation_lidar_to_imu.y,
                config.imu_params.initial_translation_lidar_to_imu.z,
            ),
            UnitQuaternion::from_matrix(&config.imu_params.initial_rotation_lidar_to_imu),
        );

        let ivox = IVoxHashMap::new(
            config.map_resolution as f32,
            config.map_capacity,
            IVoxNearbyType::Nearby26,
            config.dist_update_sensitivity as f32,
        );

        let map = RegistrationState::new(
            ivox,
            &config.imu_params,
            &ext_lidar_to_imu,
            &config.base_to_lidar,
            config.prelio_params.imu.gravity,
            config.lidar_odom_cov,
            config.ekf_max_iterations,
            None,
        );

        Self {
            map: Arc::new(Mutex::new(map)),
            ext_lidar_to_imu,
            config: config.clone(),
            active_backend: ActiveBackend::detect(config),
            #[cfg(feature = "cuda")]
            cuda_cloud_buffers: Mutex::new(CudaCloudUploadBuffers::default()),
            #[cfg(feature = "metal")]
            metal_cloud_buffer: Mutex::new(None),
            #[cfg(feature = "vulkan")]
            vulkan_cloud_buffer: Mutex::new(None),
        }
    }

    pub fn map(&self) -> Arc<Mutex<RegistrationState>> {
        Arc::clone(&self.map)
    }

    pub fn set_base_to_lidar(&mut self, base_to_lidar: Isometry3<f64>) {
        self.config.base_to_lidar = base_to_lidar;
        self.map.lock().unwrap().set_base_to_lidar(base_to_lidar);
    }

    pub fn ext(&self) -> Isometry3<f64> {
        self.ext_lidar_to_imu
    }

    pub fn lidar_vendor(&self) -> LidarVendor {
        self.config.prelio_params.lidar_vendor
    }

    pub fn kinematic_model(&self) -> KinematicModel {
        self.config.imu_params.kinematic_model
    }

    pub fn build_init_callback(&self) -> Arc<Mutex<dyn FnMut(&[ImuMsg]) + Send + Sync>> {
        let map = self.map();
        Arc::new(Mutex::new(move |msg: &[ImuMsg]| {
            let mut map = map.lock().unwrap();
            let _ = map.init_imu_ekf(&msg);
        }))
    }

    pub fn build_undistort(&self) -> Box<ImuPredictorUndistort> {
        let map = self.map();
        let kinematic_model = self.config.imu_params.kinematic_model;
        let constraint_std = self.config.imu_params.kinematic_constraint_std;
        let lidar_vendor = self.config.prelio_params.lidar_vendor;

        Box::new(
            move |imu_msgs: &[ImuMsg],
                  odom_msgs: &[OdomMsg],
                  last_cloud_end_time,
                  cloud_begin_time,
                  cloud_end_time,
                  cloud,
                  cloud_passthrough| {
                let mut map = map.lock().unwrap();
                map.undistort(
                    &imu_msgs,
                    odom_msgs,
                    kinematic_model,
                    constraint_std,
                    last_cloud_end_time,
                    cloud_begin_time,
                    cloud_end_time,
                    cloud,
                    cloud_passthrough,
                    Some(lidar_vendor),
                );
                let pose = map.get_state().pose;

                Either::Right(pose)
            },
        )
    }

    pub fn build_init_guess(&self) -> Box<ImuPredictor> {
        let map = self.map();

        Box::new(move |_, _, _, _| {
            let ekfstate = map.lock().unwrap().get_state();
            ImuPosePrediction {
                pose: ekfstate.pose,
                linear_velocity: ekfstate.linear_velocity,
                angular_velocity: nalgebra::Vector3::default(),
            }
        })
    }

    pub fn register_processed_cloud(
        &self,
        mut processed: PrelioProcessedCloud,
        ready_for_next: Option<&std::sync::mpsc::Sender<()>>,
    ) -> anyhow::Result<Option<PelorusOutput>> {
        self.ensure_backend_map_initialized();

        let mut eval = processed.runtime_eval;
        let lidar_timestamp = prelio::timestamp_to_secs(&processed.end_of_frame_time);

        #[cfg(feature = "cuda")]
        if let ActiveBackend::Cuda(stream) = &self.active_backend {
            let register_result = {
                let mut cuda_cloud = self
                    .cuda_cloud_buffers
                    .lock()
                    .expect("cuda cloud buffers mutex poisoned");
                // Same-stream ordering keeps each buffer safe while previous frame kernels run.
                let cloud_view = cuda_cloud.next_buffer_mut(stream, processed.cloud.len());
                cloud_view.upload_from_slice_cuda(stream, &processed.cloud);

                let mut map = self.map.lock().unwrap();
                match map.register_and_map(
                    processed.should_map_only,
                    self.ext_lidar_to_imu,
                    cloud_view,
                    lidar_timestamp,
                    processed.scan_id as usize,
                    ready_for_next,
                    &mut eval,
                    &self.active_backend,
                ) {
                    Ok(Some(res)) => {
                        let state = map.get_state();
                        let (pose_cov_map, _lv_unused) = map.get_odometry_covariances();
                        let twist_cov_map = map.get_twist_covariance_6x6();
                        Some((res, state, pose_cov_map, twist_cov_map))
                    }
                    Ok(None) => None,
                    Err(e) => return Err(e),
                }
            };

            let Some((registration, state, pose_cov_map, twist_cov_map)) = register_result else {
                return Ok(None);
            };

            let cloud = match processed.passthrough {
                Some(cloud) => cloud,
                None => processed.cloud,
            };

            return Ok(Some(PelorusOutput {
                scan_id: processed.scan_id,
                timestamp: processed.end_of_frame_time,
                pose: registration.pose,
                pose_covariance: pose_cov_map,
                linear_velocity: state.linear_velocity,
                angular_velocity: processed.end_of_frame_twist.angular,
                twist_covariance: twist_cov_map,
                cloud,
                frame_id: processed.header.frame_id,
            }));
        }

        #[cfg(feature = "metal")]
        if let ActiveBackend::Metal(stream) = &self.active_backend {
            let register_result = {
                // Avoid overwriting while previous frame kernels may still be reading.
                stream.sync();
                let mut metal_cloud = self
                    .metal_cloud_buffer
                    .lock()
                    .expect("metal cloud buffer mutex poisoned");
                let cloud_view = metal_cloud.get_or_insert_with(|| {
                    LocatedBufferView::alloc_points_metal(stream, processed.cloud.len())
                });
                cloud_view.realloc_if_needed_metal(stream, processed.cloud.len());
                cloud_view.upload_from_slice_metal(stream, &processed.cloud);

                let mut map = self.map.lock().unwrap();
                match map.register_and_map(
                    processed.should_map_only,
                    self.ext_lidar_to_imu,
                    cloud_view,
                    lidar_timestamp,
                    processed.scan_id as usize,
                    ready_for_next,
                    &mut eval,
                    &self.active_backend,
                ) {
                    Ok(Some(res)) => {
                        let state = map.get_state();
                        let (pose_cov_map, _lv_unused) = map.get_odometry_covariances();
                        let twist_cov_map = map.get_twist_covariance_6x6();
                        Some((res, state, pose_cov_map, twist_cov_map))
                    }
                    Ok(None) => None,
                    Err(e) => return Err(e),
                }
            };

            let Some((registration, state, pose_cov_map, twist_cov_map)) = register_result else {
                return Ok(None);
            };

            let cloud = match processed.passthrough {
                Some(cloud) => cloud,
                None => processed.cloud,
            };

            return Ok(Some(PelorusOutput {
                scan_id: processed.scan_id,
                timestamp: processed.end_of_frame_time,
                pose: registration.pose,
                pose_covariance: pose_cov_map,
                linear_velocity: state.linear_velocity,
                angular_velocity: processed.end_of_frame_twist.angular,
                twist_covariance: twist_cov_map,
                cloud,
                frame_id: processed.header.frame_id,
            }));
        }

        #[cfg(feature = "vulkan")]
        if let ActiveBackend::Vulkan(stream) = &self.active_backend {
            let register_result = {
                // Avoid overwriting while previous frame kernels may still be reading.
                stream.sync();
                let mut vulkan_cloud = self
                    .vulkan_cloud_buffer
                    .lock()
                    .expect("vulkan cloud buffer mutex poisoned");
                let cloud_view = vulkan_cloud.get_or_insert_with(|| {
                    LocatedBufferView::alloc_points_vulkan(stream, processed.cloud.len())
                });
                cloud_view.realloc_if_needed_vulkan(stream, processed.cloud.len());
                cloud_view.upload_from_slice_vulkan(stream, &processed.cloud);

                let mut map = self.map.lock().unwrap();
                match map.register_and_map(
                    processed.should_map_only,
                    self.ext_lidar_to_imu,
                    cloud_view,
                    lidar_timestamp,
                    processed.scan_id as usize,
                    ready_for_next,
                    &mut eval,
                    &self.active_backend,
                ) {
                    Ok(Some(res)) => {
                        let state = map.get_state();
                        let (pose_cov_map, _lv_unused) = map.get_odometry_covariances();
                        let twist_cov_map = map.get_twist_covariance_6x6();
                        Some((res, state, pose_cov_map, twist_cov_map))
                    }
                    Ok(None) => None,
                    Err(e) => return Err(e),
                }
            };

            let Some((registration, state, pose_cov_map, twist_cov_map)) = register_result else {
                return Ok(None);
            };

            let cloud = match processed.passthrough {
                Some(cloud) => cloud,
                None => processed.cloud,
            };

            return Ok(Some(PelorusOutput {
                scan_id: processed.scan_id,
                timestamp: processed.end_of_frame_time,
                pose: registration.pose,
                pose_covariance: pose_cov_map,
                linear_velocity: state.linear_velocity,
                angular_velocity: processed.end_of_frame_twist.angular,
                twist_covariance: twist_cov_map,
                cloud,
                frame_id: processed.header.frame_id,
            }));
        }

        let cloud_view = LocatedBufferView::Ram(std::mem::take(&mut processed.cloud));

        let register_result = {
            let mut map = self.map.lock().unwrap();
            match map.register_and_map(
                processed.should_map_only,
                self.ext_lidar_to_imu,
                &cloud_view,
                lidar_timestamp,
                processed.scan_id as usize,
                ready_for_next,
                &mut eval,
                &self.active_backend,
            ) {
                Ok(Some(res)) => {
                    let state = map.get_state();
                    let (pose_cov_map, _lv_unused) = map.get_odometry_covariances();
                    let twist_cov_map = map.get_twist_covariance_6x6();
                    Some((res, state, pose_cov_map, twist_cov_map))
                }
                Ok(None) => None,
                Err(e) => return Err(e),
            }
        };

        let Some((registration, state, pose_cov_map, twist_cov_map)) = register_result else {
            return Ok(None);
        };

        let cloud = match processed.passthrough {
            Some(cloud) => cloud,
            None => match cloud_view {
                LocatedBufferView::Ram(cloud) => cloud,
                #[allow(unreachable_patterns)]
                _ => processed.cloud,
            },
        };

        Ok(Some(PelorusOutput {
            scan_id: processed.scan_id,
            timestamp: processed.end_of_frame_time,
            pose: registration.pose,
            pose_covariance: pose_cov_map,
            linear_velocity: state.linear_velocity,
            angular_velocity: processed.end_of_frame_twist.angular,
            twist_covariance: twist_cov_map,
            cloud,
            frame_id: processed.header.frame_id,
        }))
    }

    pub fn create_prelio(
        &self,
        ready_for_next: Option<std::sync::mpsc::Receiver<()>>,
    ) -> anyhow::Result<(Prelio, std::sync::mpsc::Receiver<PrelioProcessedCloud>)> {
        let params = &self.config.prelio_params;
        let gravity = params.imu.gravity;
        let madgwick_filter = params.imu.madgwick_setting;
        let imu_calib = params.imu.calibration.clone();
        let undistort = Some(self.build_undistort());
        let init_guess = Some(self.build_init_guess());

        let mut prelio = Prelio::create(
            gravity,
            madgwick_filter,
            &imu_calib,
            params,
            undistort,
            init_guess,
            params.lidar_vendor,
            ready_for_next,
            false,
        )?;

        let rx = prelio
            .get_chan()
            .ok_or_else(|| anyhow!("Prelio channel is not available"))?;

        Ok((prelio, rx))
    }
}

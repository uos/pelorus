use std::sync::{Arc, Mutex};
#[cfg(feature = "processing-time-export")]
use pelorus_core::Instant;

use crate::{
    ActiveBackend, Registration,
    registration::{RegistrationSettings, RegistrationState},
};
use either::Either;
use nalgebra::{Isometry3, Translation3, UnitQuaternion};
use pelorus_core::{
    ImuMsg, ImuPosePrediction, ImuPredictor, ImuPredictorUndistort, KinematicModel, LidarVendor,
    LioConfig, OdomMsg, PointXYZIT, SyncProcessedCloud,
};
use pelorus_localmap::{IVoxHashMap, IVoxNearbyType};
use pelorus_sync::{
    ImuInitCallback, ImuSyncMode, LocatedBufferView, SteppedSynchronizer, SteppedSynchronizerHooks,
};
#[cfg(feature = "threads")]
use pelorus_sync::{Synchronizer, SynchronizerHooks};

/// How densely the constant-twist model is sampled into IMU messages. It only
/// has to be fine enough that the rotation between two samples stays small.
const IMU_SYNTHESIS_RATE: f64 = 200.0;

#[cfg(feature = "cuda")]
const CUDA_CLOUD_BUFFER_PIPELINE_DEPTH: usize = 2;

#[cfg(feature = "cuda")]
#[derive(Default)]
struct CudaCloudUploadBuffers {
    buffers: [Option<LocatedBufferView<PointXYZIT>>; CUDA_CLOUD_BUFFER_PIPELINE_DEPTH],
    next_slot: usize,
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
/// - **Lidar Frame (L)**: Sensor-fixed frame attached to the Lidar
#[derive(Debug, Clone)]
pub struct PelorusOutput {
    pub scan_id: u64,
    pub timestamp: ros_pointcloud2::ros::TimeMsg,
    pub pose: Isometry3<f64>, // in
    pub pose_covariance: [f64; 36],
    pub linear_velocity: nalgebra::Vector3<f64>,
    pub angular_velocity: nalgebra::Vector3<f64>,
    pub accel_bias: nalgebra::Vector3<f64>,
    pub gyro_bias: nalgebra::Vector3<f64>,
    pub twist_covariance: [f64; 36],
    pub cloud: Vec<PointXYZIT>,
    pub frame_id: String,
    #[cfg(feature = "processing-time-export")]
    pub processing_time_ns: u128,
}

impl Pipeline {
    fn ensure_backend_map_initialized(&self) {
        match &self.active_backend {
            ActiveBackend::Cpu => {}
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(stream) => {
                if pelorus_cuda::get_map_size() == 0 {
                    pelorus_cuda::init_map_with_mode(
                        stream,
                        self.config.map_capacity,
                        self.config.map_resolution as f32,
                        self.config.dist_update_sensitivity,
                        (self.config.map_resolution as f32) / 4.0,
                        pelorus_cuda::NeighborType::Neighbor26,
                        self.config.map_multiresolution,
                    );
                }
            }
            #[cfg(feature = "metal")]
            ActiveBackend::Metal(stream) => {
                if pelorus_metal::get_map_size() == 0 {
                    pelorus_metal::init_map_with_mode(
                        stream,
                        self.config.map_capacity,
                        self.config.map_resolution as f32,
                        self.config.dist_update_sensitivity,
                        (self.config.map_resolution as f32) / 4.0,
                        pelorus_metal::NeighborType::Neighbor26,
                        self.config.map_multiresolution,
                    );
                }
            }
            #[cfg(feature = "vulkan")]
            ActiveBackend::Vulkan(stream) => {
                if pelorus_vulkan::get_map_size() == 0 {
                    pelorus_vulkan::init_map_with_mode(
                        stream,
                        self.config.map_capacity,
                        self.config.map_resolution as f32,
                        self.config.dist_update_sensitivity,
                        (self.config.map_resolution as f32) / 4.0,
                        pelorus_vulkan::NeighborType::Neighbor26,
                        self.config.map_multiresolution,
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

        let mut ivox = IVoxHashMap::new(
            config.map_resolution as f32,
            config.map_capacity,
            IVoxNearbyType::Nearby26,
            config.dist_update_sensitivity,
        );
        if config.map_multiresolution {
            ivox.enable_multiresolution(config.map_capacity, config.dist_update_sensitivity);
        }

        let mut map = RegistrationState::new(
            ivox,
            &config.imu_params,
            &ext_lidar_to_imu,
            &config.base_to_lidar,
            RegistrationSettings {
                gravity: config.sync_params.imu.gravity,
                laser_point_noise_cov: config.lidar_odom_cov,
                ekf_max_iterations: config.ekf_max_iterations,
                parallelism_decision_threshold_stddevs: config
                    .sync_params
                    .parallelism_decision_threshold_stddevs,
                parallelism_decision_samples_per_mode: config
                    .sync_params
                    .parallelism_decision_samples_per_mode,
                parallelism_execution_policy: config.sync_params.parallelism_execution_policy,
            },
            None,
        );

        if config.imu_params.lidar_only {
            let synthetic = crate::lidar_only::SyntheticImu::new(
                config.sync_params.imu.gravity,
                IMU_SYNTHESIS_RATE,
            );
            let _ = map.init_imu_ekf(&synthetic.initialization_buffer(2));
        }

        Self {
            map: Arc::new(Mutex::new(map)),
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
        self.map.lock().unwrap().get_state().lidar_imu_extrinsic
    }

    /// Drop the GPU-resident buffers this pipeline and its registration state own.
    ///
    /// Backend teardown destroys the device, and every child object has to be gone
    /// before that. These two buffers outlive the map and downsample state, so they
    /// have to be released explicitly. Safe to call more than once. The pipeline
    /// reallocates them on the next scan.
    pub fn release_gpu_buffers(&mut self) {
        #[cfg(feature = "vulkan")]
        {
            *self
                .vulkan_cloud_buffer
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        }
        self.map
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .planes_gpu_buffer = None;
    }

    pub fn lidar_vendor(&self) -> LidarVendor {
        self.config.sync_params.lidar_vendor
    }

    pub fn kinematic_model(&self) -> KinematicModel {
        self.config.imu_params.kinematic_model
    }

    pub fn build_init_callback(&self) -> ImuInitCallback {
        let map = self.map();
        let lidar_only = self.config.imu_params.lidar_only;
        Arc::new(Mutex::new(move |msg: &[ImuMsg]| {
            if lidar_only {
                return;
            }
            let mut map = map.lock().unwrap();
            let _ = map.init_imu_ekf(msg);
        }))
    }

    pub fn build_undistort(&self) -> Box<ImuPredictorUndistort> {
        let map = self.map();
        let kinematic_model = self.config.imu_params.kinematic_model;
        let constraint_std = self.config.imu_params.kinematic_constraint_std;
        let lidar_vendor = self.config.sync_params.lidar_vendor;
        let lidar_only = self.config.imu_params.lidar_only;
        let synthetic = Mutex::new(crate::lidar_only::SyntheticImu::new(
            self.config.sync_params.imu.gravity,
            IMU_SYNTHESIS_RATE,
        ));

        Box::new(
            move |imu_msgs: &[ImuMsg],
                  odom_msgs: &[OdomMsg],
                  last_cloud_end_time,
                  cloud_begin_time,
                  cloud_end_time,
                  cloud,
                  cloud_passthrough| {
                let mut map = map.lock().unwrap();
                // Feed synthetic samples through the normal estimation path.
                let generated;
                let imu_msgs: &[ImuMsg] = if lidar_only {
                    let pose = map.get_state().pose;
                    let (messages, linear_velocity) = synthetic.lock().unwrap().generate(
                        &pose,
                        last_cloud_end_time,
                        cloud_end_time,
                    );
                    if let Some(linear_velocity) = linear_velocity {
                        map.set_linear_velocity(linear_velocity);
                    }
                    generated = messages;
                    &generated
                } else {
                    imu_msgs
                };
                map.undistort(
                    crate::UndistortScan {
                        imu_msgs,
                        odom_msgs,
                        last_cloud_end_time,
                        cloud_begin_time,
                        cloud_end_time,
                        cloud,
                        cloud_passthrough,
                        lidar_vendor: Some(lidar_vendor),
                    },
                    kinematic_model,
                    constraint_std,
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
        mut processed: SyncProcessedCloud,
        ready_for_next: Option<&std::sync::mpsc::Sender<u64>>,
    ) -> anyhow::Result<Option<PelorusOutput>> {
        #[cfg(feature = "processing-time-export")]
        let processing_started = Instant::now();
        self.ensure_backend_map_initialized();

        let mut eval = processed.runtime_eval;
        let lidar_timestamp = pelorus_sync::timestamp_to_secs(&processed.end_of_frame_time);

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
                let ext = map.get_state().lidar_imu_extrinsic;
                match map.register_and_map(
                    crate::RegistrationScan {
                        cloud: cloud_view,
                        imu_transform: ext,
                        lidar_timestamp,
                        scan_id: processed.scan_id as usize,
                        map_only: processed.should_map_only,
                    },
                    ready_for_next,
                    &mut eval,
                    &self.active_backend,
                ) {
                    Ok(Some(res)) => {
                        let state = map.get_state();
                        let pose_cov_map = map.get_output_pose_covariance(&Isometry3::identity());
                        let twist_cov_map = map.get_output_twist_covariance(&res.pose);
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
                accel_bias: state.accel_bias,
                gyro_bias: state.gyro_bias,
                twist_covariance: twist_cov_map,
                cloud,
                frame_id: processed.header.frame_id,
                #[cfg(feature = "processing-time-export")]
                processing_time_ns: processing_started.elapsed().as_nanos(),
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
                let ext = map.get_state().lidar_imu_extrinsic;
                match map.register_and_map(
                    crate::RegistrationScan {
                        cloud: cloud_view,
                        imu_transform: ext,
                        lidar_timestamp,
                        scan_id: processed.scan_id as usize,
                        map_only: processed.should_map_only,
                    },
                    ready_for_next,
                    &mut eval,
                    &self.active_backend,
                ) {
                    Ok(Some(res)) => {
                        let state = map.get_state();
                        let pose_cov_map = map.get_output_pose_covariance(&Isometry3::identity());
                        let twist_cov_map = map.get_output_twist_covariance(&res.pose);
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
                accel_bias: state.accel_bias,
                gyro_bias: state.gyro_bias,
                twist_covariance: twist_cov_map,
                cloud,
                frame_id: processed.header.frame_id,
                #[cfg(feature = "processing-time-export")]
                processing_time_ns: processing_started.elapsed().as_nanos(),
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
                let ext = map.get_state().lidar_imu_extrinsic;
                match map.register_and_map(
                    crate::RegistrationScan {
                        cloud: cloud_view,
                        imu_transform: ext,
                        lidar_timestamp,
                        scan_id: processed.scan_id as usize,
                        map_only: processed.should_map_only,
                    },
                    ready_for_next,
                    &mut eval,
                    &self.active_backend,
                ) {
                    Ok(Some(res)) => {
                        let state = map.get_state();
                        let pose_cov_map = map.get_output_pose_covariance(&Isometry3::identity());
                        let twist_cov_map = map.get_output_twist_covariance(&res.pose);
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
                accel_bias: state.accel_bias,
                gyro_bias: state.gyro_bias,
                twist_covariance: twist_cov_map,
                cloud,
                frame_id: processed.header.frame_id,
                #[cfg(feature = "processing-time-export")]
                processing_time_ns: processing_started.elapsed().as_nanos(),
            }));
        }

        let cloud_view = LocatedBufferView::Ram(std::mem::take(&mut processed.cloud));

        let register_result = {
            let mut map = self.map.lock().unwrap();
            let ext = map.get_state().lidar_imu_extrinsic;
            match map.register_and_map(
                crate::RegistrationScan {
                    cloud: &cloud_view,
                    imu_transform: ext,
                    lidar_timestamp,
                    scan_id: processed.scan_id as usize,
                    map_only: processed.should_map_only,
                },
                ready_for_next,
                &mut eval,
                &self.active_backend,
            ) {
                Ok(Some(res)) => {
                    let state = map.get_state();
                    let pose_cov_map = map.get_output_pose_covariance(&Isometry3::identity());
                    let twist_cov_map = map.get_output_twist_covariance(&res.pose);
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
            accel_bias: state.accel_bias,
            gyro_bias: state.gyro_bias,
            twist_covariance: twist_cov_map,
            cloud,
            frame_id: processed.header.frame_id,
            #[cfg(feature = "processing-time-export")]
            processing_time_ns: processing_started.elapsed().as_nanos(),
        }))
    }

    fn imu_sync_mode(&self) -> ImuSyncMode {
        if self.config.imu_params.lidar_only {
            ImuSyncMode::Disabled { lossless: false }
        } else if self.config.sync_params.lidar_as_sync {
            ImuSyncMode::LidarDriven
        } else {
            ImuSyncMode::Continuous
        }
    }

    /// Synchronizer that runs on its own threads and hands each synchronized
    /// scan to the returned channel.
    #[cfg(feature = "threads")]
    pub fn create_sync(
        &self,
        ready_for_next: Option<std::sync::mpsc::Receiver<u64>>,
    ) -> anyhow::Result<(Synchronizer, std::sync::mpsc::Receiver<SyncProcessedCloud>)> {
        let params = &self.config.sync_params;
        let gravity = params.imu.gravity;
        let madgwick_filter = params.imu.madgwick_setting;
        let imu_calib = params.imu.calibration.clone();
        let undistort = Some(self.build_undistort());
        let init_guess = Some(self.build_init_guess());
        let imu_sync_mode = self.imu_sync_mode();
        let mut sync = Synchronizer::create(
            gravity,
            madgwick_filter,
            &imu_calib,
            params,
            params.lidar_vendor,
            imu_sync_mode,
            SynchronizerHooks {
                imu_pose_prediction_undistort: undistort,
                imu_pose_prediction_init_guess: init_guess,
                ready_for_next,
                ..Default::default()
            },
        )?;

        let rx = sync
            .get_chan()
            .ok_or_else(|| anyhow::anyhow!("Synchronizer channel is not available"))?;

        Ok((sync, rx))
    }

    /// Synchronizer that runs on the calling thread. Register every scan it
    /// returns before asking it for the next one.
    pub fn create_stepped_sync(&self) -> anyhow::Result<SteppedSynchronizer> {
        let params = &self.config.sync_params;
        SteppedSynchronizer::create(
            params,
            params.lidar_vendor,
            self.imu_sync_mode(),
            SteppedSynchronizerHooks {
                imu_pose_prediction_undistort: Some(self.build_undistort()),
                imu_pose_prediction_init_guess: Some(self.build_init_guess()),
                ..Default::default()
            },
        )
    }
}

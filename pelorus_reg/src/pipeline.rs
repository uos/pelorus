use std::sync::{Arc, Mutex};

use anyhow::anyhow;
use either::Either;
use nalgebra::{Isometry3, Translation3, UnitQuaternion};
use pelorus_core::{
    ImuMsg, ImuPosePrediction, ImuPredictor, ImuPredictorUndistort, KinematicModel, LidarVendor,
    LioConfig, LivoxMid360Point, LivoxPoint, OdomMsg, OusterPoint, PointXYZIT,
    PrelioProcessedCloud, RobosensePoint, VelodynePoint,
};
use prelio::Prelio;
use spole::IVoxNearbyType;

use crate::{Registration, registration::RegistrationState};

pub struct Pipeline {
    map: Arc<Mutex<RegistrationState>>,
    ext_lidar_to_imu: Isometry3<f64>,
    pub config: LioConfig,
    #[cfg(feature = "cuda")]
    cuda_stream: pelorus_cuda::Stream,
    #[cfg(feature = "metal")]
    metal_stream: pelorus_metal::Stream,
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
    pub fn new(config: &LioConfig) -> Self {
        let ext_lidar_to_imu = Isometry3::from_parts(
            Translation3::new(
                config.imu_params.initial_translation_lidar_to_imu.x,
                config.imu_params.initial_translation_lidar_to_imu.y,
                config.imu_params.initial_translation_lidar_to_imu.z,
            ),
            UnitQuaternion::from_matrix(&config.imu_params.initial_rotation_lidar_to_imu),
        );

        let map = RegistrationState::new(
            config.map_resolution,
            config.map_capacity,
            config.prelio_params.imu.gravity,
            &config.imu_params,
            config.lidar_odom_cov,
            config.neighbor_max_range,
            ext_lidar_to_imu,
            config.base_to_lidar,
            IVoxNearbyType::Nearby26,
            config.dist_update_sensitivity,
            config.ekf_max_iterations,
        );

        Self {
            map: Arc::new(Mutex::new(map)),
            ext_lidar_to_imu,
            config: config.clone(),
            #[cfg(feature = "cuda")]
            cuda_stream: pelorus_cuda::Stream::new(),
            #[cfg(feature = "metal")]
            metal_stream: pelorus_metal::Stream::new(),
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
        processed: PrelioProcessedCloud,
        ready_for_next: Option<&std::sync::mpsc::Sender<()>>,
    ) -> Option<PelorusOutput> {
        let mut eval = processed.runtime_eval;
        let lidar_timestamp = prelio::timestamp_to_secs(&processed.end_of_frame_time);
        let cloud_view = prelio::LocatedBufferView::Ram(processed.cloud.clone()); // TODO this must somehow work without this clone

        let register_result = {
            let mut map = self.map.lock().unwrap();
            match map.register_and_map(
                processed.should_map_only,
                self.ext_lidar_to_imu,
                &cloud_view,
                lidar_timestamp,
                ready_for_next,
                &mut eval,
                #[cfg(feature = "cuda")]
                &self.cuda_stream,
                #[cfg(feature = "metal")]
                &self.metal_stream,
            ) {
                Ok(Some(res)) => {
                    let state = map.get_state();
                    let (pose_cov_map, _lv_unused) = map.get_odometry_covariances();
                    let twist_cov_map = map.get_twist_covariance_6x6();
                    Some((res, state, pose_cov_map, twist_cov_map))
                }
                _ => None,
            }
        }?;

        let (registration, state, pose_cov_map, twist_cov_map) = register_result;

        Some(PelorusOutput {
            scan_id: processed.scan_id,
            timestamp: processed.end_of_frame_time,
            pose: registration.pose,
            pose_covariance: pose_cov_map,
            linear_velocity: state.linear_velocity,
            angular_velocity: processed.end_of_frame_twist.angular,
            twist_covariance: twist_cov_map,
            cloud: processed.passthrough.unwrap_or(processed.cloud),
            frame_id: processed.header.frame_id,
        })
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

        let mut prelio = match params.lidar_vendor {
            LidarVendor::Ouster => Prelio::create::<5, OusterPoint>(
                gravity,
                madgwick_filter,
                &imu_calib,
                params,
                undistort,
                init_guess,
                Some(params.lidar_vendor),
                ready_for_next,
            ),
            LidarVendor::Velodyne => Prelio::create::<5, VelodynePoint>(
                gravity,
                madgwick_filter,
                &imu_calib,
                params,
                undistort,
                init_guess,
                Some(params.lidar_vendor),
                ready_for_next,
            ),
            LidarVendor::Livox => Prelio::create::<5, LivoxPoint>(
                gravity,
                madgwick_filter,
                &imu_calib,
                params,
                undistort,
                init_guess,
                Some(params.lidar_vendor),
                ready_for_next,
            ),
            LidarVendor::Robosense => Prelio::create::<5, RobosensePoint>(
                gravity,
                madgwick_filter,
                &imu_calib,
                params,
                undistort,
                init_guess,
                Some(params.lidar_vendor),
                ready_for_next,
            ),
            LidarVendor::LivoxMid360 => Prelio::create::<5, LivoxMid360Point>(
                gravity,
                madgwick_filter,
                &imu_calib,
                params,
                undistort,
                init_guess,
                Some(params.lidar_vendor),
                ready_for_next,
            ),
        }?;

        let rx = prelio
            .get_chan()
            .ok_or_else(|| anyhow!("Prelio channel is not available"))?;

        Ok((prelio, rx))
    }
}

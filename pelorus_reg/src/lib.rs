use nalgebra::*;
use pelorus_core::{
    ImuMsg, KinematicModel, LidarVendor, OdomMsg, PointXYZIT, timing::RuntimeEvaluator,
};
use prelio::LocatedBufferView;
use ros_pointcloud2::prelude::*;

pub mod backend;
pub use backend::ActiveBackend;
pub mod iekf;
pub mod imu;
pub mod pipeline;
pub mod registration;
use anyhow::Result;
pub use spole::*;

/// Result from cloud registration and mapping
///
/// # Frame Conventions
/// - **World Frame (W)**: Fixed global reference frame, aligned with IMU orientation at
///   initialization. Z-axis points upward (anti-gravity).
/// - **IMU Frame (I)**: Body-fixed frame attached to the IMU sensor
/// - **Lidar Frame (L)**: Sensor-fixed frame attached to the LiDAR
#[derive(Clone, Default, Debug)]
pub struct CloudRegisterResult {
    /// Pose of the lidar in world frame (T_W_L)
    pub pose: Isometry3<f64>,
    /// Pose of the base link in world frame (T_W_B)
    pub base_pose: Isometry3<f64>,
    pub avg_iterations: Option<u32>,
}

/// IMU-based EKF state prediction
///
/// # Frame Conventions
/// - **World Frame (W)**: Fixed global reference frame, aligned with IMU at initialization,
///   Z-axis points upward (anti-gravity)
/// - **IMU Frame (I)**: Body-fixed frame attached to the IMU sensor
/// - **Lidar Frame (L)**: Sensor-fixed frame attached to the LiDAR
#[derive(Clone, Debug, PartialEq)]
pub struct ImuEkfPrediction {
    /// Pose of the IMU in world frame
    pub pose: Isometry3<f64>,
    /// Linear velocity in world frame
    pub linear_velocity: Vector3<f64>,
    /// Angular velocity (bias-corrected) in body frame
    pub angular_velocity: Vector3<f64>,
    /// Extrinsic calibration from lidar to IMU (T_I_L)
    pub lidar_imu_extrinsic: Isometry3<f64>,
    /// Accelerometer bias estimate
    pub accel_bias: Vector3<f64>,
    /// Gyroscope bias estimate
    pub gyro_bias: Vector3<f64>,
}

impl Default for ImuEkfPrediction {
    fn default() -> Self {
        ImuEkfPrediction {
            pose: Isometry3::identity(),
            linear_velocity: Vector3::zeros(),
            angular_velocity: Vector3::zeros(),
            lidar_imu_extrinsic: Isometry3::identity(),
            accel_bias: Vector3::zeros(),
            gyro_bias: Vector3::zeros(),
        }
    }
}

pub trait Registration {
    fn register_and_map(
        &mut self,
        map_only: bool,
        imu_transform: Isometry3<f64>,
        cloud: &LocatedBufferView<PointXYZIT>,
        lidar_timestamp: f64,
        scan_id: usize,
        ready_for_next: Option<&std::sync::mpsc::Sender<()>>,
        eval: &mut RuntimeEvaluator,
        backend: &ActiveBackend,
    ) -> Result<Option<CloudRegisterResult>>;
    fn requested_register_and_map_cloud_type(&self) -> LocatedBufferView<PointXYZ>;

    fn imu_ekf_predict(&mut self, acc: [f64; 3], gyro: [f64; 3], dt: f64) -> ImuEkfPrediction;
    fn init_imu_ekf(&mut self, message_buffer: &[ImuMsg]) -> Vector3<f64>;
    fn set_base_to_lidar(&mut self, base_to_lidar: Isometry3<f64>);
    fn get_state(&self) -> ImuEkfPrediction;
    fn undistort(
        &mut self,
        imu_msgs: &[ImuMsg],
        odom_msgs: &[OdomMsg],
        kinematic: KinematicModel,
        constraint_std: f64,
        last_cloud_end_time: f64,
        cloud_begin_time: f64,
        cloud_end_time: f64,
        cloud: &mut Vec<PointXYZIT>,
        cloud_passthrough: Option<&mut Vec<PointXYZIT>>,
        lidar_vendor: Option<LidarVendor>,
    );
}

use nalgebra::*;
use prelio::{LocatedBufferView, timing::RuntimeEvaluator};
use ros_pointcloud2::prelude::*;

pub mod imu;
pub mod registration;
pub mod sophus_ekf;
use anyhow::Result;
pub use spole::*;

#[derive(Clone, Default, Debug)]
pub struct CloudRegisterResult {
    pub pose: Isometry3<f64>,
    pub avg_iterations: Option<u32>,
    // pub registered_cloud: Option<Vec<PointXYZINormal>>,
}

#[derive(Clone, Default, Debug, PartialEq)]
pub struct ImuMsg {
    pub timestamp_sec: f64,
    pub orientation: UnitQuaternion<f64>,
    pub orientation_covariance: [f64; 9],
    pub angular_velocity: Vector3<f64>,
    pub angular_velocity_covariance: [f64; 9],
    pub linear_acceleration: Vector3<f64>,
    pub linear_acceleration_covariance: [f64; 9],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImuEkfPrediction {
    pub pose: Isometry3<f64>,
    pub linear_velocity: Vector3<f64>,
    pub lidar_imu_extrinsic: Isometry3<f64>,
}

impl Default for ImuEkfPrediction {
    fn default() -> Self {
        ImuEkfPrediction {
            pose: Isometry3::identity(),
            linear_velocity: Vector3::zeros(),
            lidar_imu_extrinsic: Isometry3::identity(),
        }
    }
}

pub trait Registration {
    fn register_and_map(
        &mut self,
        map_only: bool,
        cloud: &LocatedBufferView<PointXYZI>,
        ready_for_next: Option<&std::sync::mpsc::Sender<()>>,
        eval: &mut RuntimeEvaluator,
        #[cfg(feature = "cuda")] stream: &cuda::Stream,
    ) -> Result<CloudRegisterResult>;
    fn requested_register_and_map_cloud_type(&self) -> LocatedBufferView<PointXYZ>;

    fn imu_ekf_predict(&mut self, acc: [f64; 3], gyro: [f64; 3], dt: f64) -> ImuEkfPrediction;
    fn init_imu_ekf(&mut self, message_buffer: &[ImuMsg]) -> Vector3<f64>;
    fn get_state(&self) -> ImuEkfPrediction;
    fn undistort(
        &mut self,
        imu_msgs: &[ImuMsg],
        last_cloud_end_time: f64,
        cloud_begin_time: f64,
        cloud_end_time: f64,
        cloud: &[prelio::PointXYZIT],
        lidar_vendor: Option<prelio::LidarVendor>,
    ) -> Vec<PointXYZI>;
}

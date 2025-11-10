use nalgebra::*;
use pelorus_core::{
    ImuMsg, KinematicModel, LidarVendor, OdomMsg, PointXYZIT, timing::RuntimeEvaluator,
};
use prelio::LocatedBufferView;
use ros_pointcloud2::prelude::*;

pub mod imu;
pub mod pipeline;
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
        imu_transform: Isometry3<f64>,
        cloud: &LocatedBufferView<PointXYZI>,
        lidar_timestamp: f64,
        ready_for_next: Option<&std::sync::mpsc::Sender<()>>,
        eval: &mut RuntimeEvaluator,
        #[cfg(feature = "cuda")] stream: &cuda::Stream,
    ) -> Result<Option<CloudRegisterResult>>;
    fn requested_register_and_map_cloud_type(&self) -> LocatedBufferView<PointXYZ>;

    fn imu_ekf_predict(&mut self, acc: [f64; 3], gyro: [f64; 3], dt: f64) -> ImuEkfPrediction;
    fn init_imu_ekf(&mut self, message_buffer: &[ImuMsg]) -> Vector3<f64>;
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
        cloud: &[PointXYZIT],
        lidar_vendor: Option<LidarVendor>,
    ) -> Vec<PointXYZI>;
}

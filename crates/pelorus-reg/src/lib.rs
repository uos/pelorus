use nalgebra::*;
use ros_pointcloud2::prelude::*;

mod imu;
mod localmap;
pub mod sfasterlio;
mod sophus_ekf;

#[derive(Clone, Default, Debug, PartialEq)]
pub struct CloudRegisterResult {
    pub pose: Isometry3<f64>,
    pub avg_iterations: Option<u32>,
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
    fn register_and_map(&mut self, cloud: &[PointXYZ]) -> anyhow::Result<CloudRegisterResult>;
    fn imu_ekf_predict(&mut self, acc: [f64; 3], gyro: [f64; 3], dt: f64) -> ImuEkfPrediction;
    fn init_imu_ekf(&mut self, message_buffer: &[ImuMsg]);
}

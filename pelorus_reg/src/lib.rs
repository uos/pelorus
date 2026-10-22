use nalgebra::*;
use pelorus_core::{
    ImuMsg, KinematicModel, LidarVendor, OdomMsg, PointXYZIT, timing::RuntimeEvaluator,
};
use pelorus_sync::LocatedBufferView;
use ros_pointcloud2::prelude::*;

pub mod backend;
pub mod lidar_only;
pub use backend::ActiveBackend;
pub mod eqf;
pub mod imu;
pub mod lio;
pub mod pipeline;
#[cfg(feature = "threads")]
pub mod realtime;
pub mod registration;
use anyhow::Result;
pub use pelorus_localmap::*;

/// Result from cloud registration and mapping
///
/// # Frame Conventions
/// - **World Frame (W)**: Fixed global reference frame, aligned with IMU orientation at
///   initialization. Z-axis points upward (anti-gravity).
/// - **IMU Frame (I)**: Body-fixed frame attached to the IMU sensor
/// - **Lidar Frame (L)**: Sensor-fixed frame attached to the Lidar
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
/// - **Lidar Frame (L)**: Sensor-fixed frame attached to the Lidar
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

impl ImuEkfPrediction {
    /// Linear and angular velocity of a frame rigidly attached to the IMU,
    /// expressed in that frame. `output_pose` is the frame's pose in the world
    /// frame (T_W_O) at the same instant as this state. The linear velocity
    /// includes the lever arm from the IMU to the frame's origin.
    pub fn body_twist(&self, output_pose: &Isometry3<f64>) -> (Vector3<f64>, Vector3<f64>) {
        let angular_velocity_world = self.pose.rotation * self.angular_velocity;
        let lever_arm_world = output_pose.translation.vector - self.pose.translation.vector;
        let linear_velocity_world =
            self.linear_velocity + angular_velocity_world.cross(&lever_arm_world);
        let world_to_output = output_pose.rotation.inverse();
        (
            world_to_output * linear_velocity_world,
            world_to_output * angular_velocity_world,
        )
    }
}

/// One preprocessed scan handed to [`Registration::register_and_map`].
pub struct RegistrationScan<'a> {
    pub cloud: &'a LocatedBufferView<PointXYZIT>,
    pub imu_transform: Isometry3<f64>,
    pub lidar_timestamp: f64,
    pub scan_id: usize,
    pub map_only: bool,
}

/// One scan and the IMU and odometry messages spanning it, handed to
/// [`Registration::undistort`].
pub struct UndistortScan<'a> {
    pub imu_msgs: &'a [ImuMsg],
    pub odom_msgs: &'a [OdomMsg],
    pub last_cloud_end_time: f64,
    pub cloud_begin_time: f64,
    pub cloud_end_time: f64,
    pub cloud: &'a mut Vec<PointXYZIT>,
    pub cloud_passthrough: Option<&'a mut Vec<PointXYZIT>>,
    pub lidar_vendor: Option<LidarVendor>,
}

pub trait Registration {
    fn register_and_map(
        &mut self,
        scan: RegistrationScan<'_>,
        ready_for_next: Option<&std::sync::mpsc::Sender<u64>>,
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
        scan: UndistortScan<'_>,
        kinematic: KinematicModel,
        constraint_std: f64,
    );
}

/// Register a session that may use a GPU backend.
///
/// Pair with [`backend_session_release`] around the lifetime of an embedder handle.
/// Without it the process can exit with a live Vulkan instance, and the driver then
/// tears itself down from an `atexit` handler while its own worker threads are still
/// running. A no-op for backends that do not need it.
pub fn backend_session_retain() {
    #[cfg(feature = "vulkan")]
    pelorus_vulkan::session_retain();
}

/// End a session registered with [`backend_session_retain`], releasing process-wide
/// GPU resources once the last session is gone.
pub fn backend_session_release() {
    #[cfg(feature = "vulkan")]
    pelorus_vulkan::session_release();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_twist_rotates_into_a_yawed_output_frame_and_adds_the_lever_arm() {
        // IMU at the origin, base 1 m ahead of it along world x and yawed 90
        // degrees, as with a Lidar mounted sideways on the robot.
        let imu_pose = Isometry3::identity();
        let base_pose = Isometry3::from_parts(
            Translation3::new(1.0, 0.0, 0.0),
            UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
        );
        let state = ImuEkfPrediction {
            pose: imu_pose,
            linear_velocity: Vector3::new(1.0, 0.0, 0.0),
            angular_velocity: Vector3::new(0.1, 0.0, 0.5),
            ..Default::default()
        };

        let (linear, angular) = state.body_twist(&base_pose);

        // World omega (0.1, 0, 0.5) seen from a frame yawed +90 degrees.
        assert!((angular - Vector3::new(0.0, -0.1, 0.5)).norm() < 1e-12);
        // World velocity (1, 0, 0) + omega x (1, 0, 0) = (1, 0.5, 0), seen
        // from the same frame.
        assert!((linear - Vector3::new(0.5, -1.0, 0.0)).norm() < 1e-12);
    }
}

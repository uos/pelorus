//! Library entry for Pelorus.
//!
//! Exposes reusable building blocks so the binary and C FFI can share the
//! LiDAR-Inertial Odometry pipeline.

#[cfg(feature = "evaluate")]
pub mod evaluate;
pub mod mapping;
#[cfg(feature = "rat")]
pub mod output_tui;
pub mod stats_tui;

// pub fn rpcl2_time_to_r2r_time(t: &ros_pointcloud2::ros::TimeMsg) -> TimeMsg {
//     TimeMsg {
//         sec: t.sec,
//         nanosec: t.nanosec,
//     }
// }

// pub fn imu_msg_to_pelorus_reg(m: &ImuMsg) -> pelorus_reg::ImuMsg {
//     pelorus_reg::ImuMsg {
//         timestamp_sec: prelio_io::prelio::timestamp_to_secs(&m.stamp),
//         orientation: UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
//             m.orientation[0],
//             m.orientation[1],
//             m.orientation[2],
//             m.orientation[3],
//         )),
//         angular_velocity: Vector3::new(
//             m.angular_velocity[0],
//             m.angular_velocity[1],
//             m.angular_velocity[2],
//         ),
//         linear_acceleration: Vector3::new(
//             m.linear_acceleration[0],
//             m.linear_acceleration[1],
//             m.linear_acceleration[2],
//         ),
//         orientation_covariance: m.orientation_covariance,
//         angular_velocity_covariance: m.angular_velocity_covariance,
//         linear_acceleration_covariance: m.linear_acceleration_covariance,
//     }
// }

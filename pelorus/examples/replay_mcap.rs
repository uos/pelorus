//! Replay a ROS 2 MCAP bag through an [`Odometer`] and write the trajectory in TUM format.
//!
//! Every message is fed in the order of the file, which is the log-time order of bags
//! written by rosbag2.
//!
//! ```sh
//! cargo run --release -p pelorus_sdk --example replay_mcap -- \
//!     config.rl bag.mcap /livox/lidar /livox/imu trajectory.tum
//! ```

use std::io::Write;

use anyhow::Context;
use pelorus::{ImuMeasurement, Odometer};
use ros_pointcloud2::prelude::*;
use ros2_interfaces_jazzy_serde::sensor_msgs::msg::{Imu, PointCloud2};

impl_pointcloud2_for_ros2_interfaces_jazzy_serde!();

/// ROS messages are f64. The odometer takes [`pelorus::Real`].
fn real<const N: usize>(values: [f64; N]) -> [pelorus::Real; N] {
    values.map(pelorus_core::real)
}

fn imu_measurement(imu: Imu) -> ImuMeasurement {
    let (o, w, a) = (
        imu.orientation,
        imu.angular_velocity,
        imu.linear_acceleration,
    );
    ImuMeasurement {
        timestamp_sec: imu.header.stamp.sec,
        timestamp_nanosec: imu.header.stamp.nanosec,
        orientation: real([o.w, o.x, o.y, o.z]),
        orientation_covariance: real(imu.orientation_covariance),
        angular_velocity: real([w.x, w.y, w.z]),
        angular_velocity_covariance: real(imu.angular_velocity_covariance),
        linear_acceleration: real([a.x, a.y, a.z]),
        linear_acceleration_covariance: real(imu.linear_acceleration_covariance),
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let [_, config, bag, lidar_topic, imu_topic, out] = args.as_slice() else {
        anyhow::bail!(
            "usage: replay_mcap <config.rl> <bag.mcap> <lidar topic> <imu topic> <out.tum>"
        );
    };

    let mut odometer = Odometer::new(Some(config))?;
    let data = std::fs::read(bag).with_context(|| format!("reading {bag}"))?;
    let mut trajectory = std::io::BufWriter::new(std::fs::File::create(out)?);
    let mut poses = 0_usize;

    for message in mcap::MessageStream::new(&data)? {
        let message = message?;
        let topic = message.channel.topic.as_str();
        if topic == imu_topic {
            let imu: Imu = cdr::deserialize(&message.data)?;
            odometer.add_imu(imu_measurement(imu));
        } else if topic == lidar_topic {
            let cloud: PointCloud2 = cdr::deserialize(&message.data)?;
            odometer.add_pointcloud2(impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(cloud));
        } else {
            continue;
        }
        while let Some(scan) = odometer.next_scan() {
            let (sec, nanosec) = scan.timestamp();
            let [x, y, z] = scan.position();
            let [qw, qx, qy, qz] = scan.orientation();
            writeln!(
                trajectory,
                "{sec}.{nanosec:09} {x} {y} {z} {qx} {qy} {qz} {qw}"
            )?;
            poses += 1;
        }
    }
    trajectory.flush()?;
    println!("{poses} poses written to {out}");
    Ok(())
}

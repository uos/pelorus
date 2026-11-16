//! Turn a ROS 2 MCAP bag and a `.rl` configuration into an input stream.

use std::error::Error;
use std::io::{self, Write};

use pelorus::PointCloud2Msg;
use pelorus_core::adaptive_execution::ExecutionPolicy;
use ros_pointcloud2::{Denseness, Endian};
use ros2_interfaces_jazzy_serde::sensor_msgs::msg::{Imu, PointCloud2};

use crate::stream::{CLOUD, IMU, MAGIC};

ros_pointcloud2::impl_pointcloud2_for_ros2_interfaces_jazzy_serde!();

/// Channel metadata that marks clouds marina stored compressed. The schema
/// stays PointCloud2 and the payload is a cloudini `CompressedPointCloud2`.
const CODEC_KEY: &str = "marina.pointcloud.codec";
const CLOUDINI_CODEC: &str = "cloudini";

/// The resolved configuration of `config`, postcard-encoded.
fn resolve(config: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    // The std build resolves the `.rl` file. Both builds then read the same LioConfig.
    let mut config = pelorus::load_config(Some(config))?;
    // Both builds run on one thread, so neither sums in a parallel order.
    config.sync_params.parallelism_execution_policy = ExecutionPolicy::SingleCore;
    Ok(postcard::to_allocvec(&config)?)
}

/// Write the resolved configuration of `config` in the precision of this
/// build, for `run` of a build in the same precision.
pub fn config(config: &str, out: &str) -> Result<(), Box<dyn Error>> {
    std::fs::write(out, resolve(config)?)?;
    Ok(())
}

/// `-` is the standard output, so a stream can go straight into `run`.
pub fn convert(
    config: &str,
    bag: &str,
    lidar_topic: &str,
    imu_topic: &str,
    out: &str,
) -> Result<(), Box<dyn Error>> {
    let config = resolve(config)?;
    // The bag is read as it streams, so its size does not matter.
    let file = std::fs::File::open(bag).map_err(|e| format!("reading {bag}: {e}"))?;
    let messages = mcap::io::MessageReader::new(io::BufReader::with_capacity(1 << 20, file));
    let out: Box<dyn Write> = if out == "-" {
        Box::new(io::stdout().lock())
    } else {
        Box::new(std::fs::File::create(out)?)
    };
    let mut out = io::BufWriter::with_capacity(1 << 20, out);
    out.write_all(MAGIC)?;
    write_bytes(&mut out, &config)?;

    let (mut imus, mut clouds) = (0_usize, 0_usize);
    for message in messages {
        let message = message?;
        let topic = message.channel.topic.as_str();
        if topic == imu_topic {
            let imu = cdr::deserialize(&message.data)
                .map_err(|e| format!("decoding an IMU message on {topic}: {e}"))?;
            write_imu(&mut out, imu)?;
            imus += 1;
        } else if topic == lidar_topic {
            let compressed = message
                .channel
                .metadata
                .get(CODEC_KEY)
                .is_some_and(|codec| codec.starts_with(CLOUDINI_CODEC));
            let cloud = if compressed {
                let cloud: cloudini::ros::CompressedPointCloud2 =
                    cdr::deserialize(&message.data)
                        .map_err(|e| format!("decoding a compressed cloud on {topic}: {e}"))?;
                cloud.decompress()?
            } else {
                let cloud: PointCloud2 = cdr::deserialize(&message.data)
                    .map_err(|e| format!("decoding a cloud on {topic}: {e}"))?;
                impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(cloud)
            };
            write_cloud(&mut out, &cloud)?;
            clouds += 1;
        }
    }
    out.flush()?;
    if clouds == 0 || imus == 0 {
        return Err(format!(
            "found {clouds} clouds on {lidar_topic} and {imus} IMU messages on {imu_topic}"
        )
        .into());
    }
    eprintln!("{clouds} clouds and {imus} IMU messages");
    Ok(())
}

fn write_imu(out: &mut impl Write, imu: Imu) -> io::Result<()> {
    let (o, w, a) = (
        imu.orientation,
        imu.angular_velocity,
        imu.linear_acceleration,
    );
    out.write_all(&[IMU])?;
    out.write_all(&imu.header.stamp.sec.to_le_bytes())?;
    out.write_all(&imu.header.stamp.nanosec.to_le_bytes())?;
    for value in [o.w, o.x, o.y, o.z]
        .iter()
        .chain(&imu.orientation_covariance)
        .chain(&[w.x, w.y, w.z])
        .chain(&imu.angular_velocity_covariance)
        .chain(&[a.x, a.y, a.z])
        .chain(&imu.linear_acceleration_covariance)
    {
        out.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_cloud(out: &mut impl Write, cloud: &PointCloud2Msg) -> io::Result<()> {
    out.write_all(&[CLOUD])?;
    out.write_all(&cloud.header.stamp.sec.to_le_bytes())?;
    out.write_all(&cloud.header.stamp.nanosec.to_le_bytes())?;
    write_bytes(out, cloud.header.frame_id.as_bytes())?;
    out.write_all(&cloud.dimensions.width.to_le_bytes())?;
    out.write_all(&cloud.dimensions.height.to_le_bytes())?;
    out.write_all(&(cloud.fields.len() as u32).to_le_bytes())?;
    for field in &cloud.fields {
        write_bytes(out, field.name.as_bytes())?;
        out.write_all(&field.offset.to_le_bytes())?;
        out.write_all(&[field.datatype])?;
        out.write_all(&field.count.to_le_bytes())?;
    }
    out.write_all(&[u8::from(matches!(cloud.endian, Endian::Big))])?;
    out.write_all(&cloud.point_step.to_le_bytes())?;
    out.write_all(&cloud.row_step.to_le_bytes())?;
    write_bytes(out, &cloud.data)?;
    out.write_all(&[u8::from(matches!(cloud.dense, Denseness::Dense))])
}

fn write_bytes(out: &mut impl Write, bytes: &[u8]) -> io::Result<()> {
    out.write_all(&(bytes.len() as u32).to_le_bytes())?;
    out.write_all(bytes)
}

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use anyhow::{Context, Result};
use mcap::records::MessageHeader;
use ros2_interfaces_jazzy_serde::{
    builtin_interfaces::msg::Time,
    nav_msgs::msg::Odometry,
    sensor_msgs::msg::{Imu, PointCloud2},
    tf2_msgs::msg::TFMessage,
};

const POINTCLOUD2_SCHEMA: &str = r#"std_msgs/Header header
uint32 height
uint32 width
sensor_msgs/PointField[] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] data
bool is_dense
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: sensor_msgs/PointField
uint8 INT8=1
uint8 UINT8=2
uint8 INT16=3
uint8 UINT16=4
uint8 INT32=5
uint8 UINT32=6
uint8 FLOAT32=7
uint8 FLOAT64=8
string name
uint32 offset
uint8 datatype
uint32 count
"#;

const ODOMETRY_SCHEMA: &str = r#"std_msgs/Header header
string child_frame_id
geometry_msgs/PoseWithCovariance pose
geometry_msgs/TwistWithCovariance twist
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: geometry_msgs/PoseWithCovariance
geometry_msgs/Pose pose
float64[36] covariance
================================================================================
MSG: geometry_msgs/Pose
geometry_msgs/Point position
geometry_msgs/Quaternion orientation
================================================================================
MSG: geometry_msgs/Point
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
================================================================================
MSG: geometry_msgs/TwistWithCovariance
geometry_msgs/Twist twist
float64[36] covariance
================================================================================
MSG: geometry_msgs/Twist
geometry_msgs/Vector3 linear
geometry_msgs/Vector3 angular
================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
"#;

const TF_SCHEMA: &str = r#"geometry_msgs/TransformStamped[] transforms
================================================================================
MSG: geometry_msgs/TransformStamped
std_msgs/Header header
string child_frame_id
geometry_msgs/Transform transform
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: geometry_msgs/Transform
geometry_msgs/Vector3 translation
geometry_msgs/Quaternion rotation
================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
"#;

const IMU_SCHEMA: &str = r#"std_msgs/Header header
geometry_msgs/Quaternion orientation
float64[9] orientation_covariance
geometry_msgs/Vector3 angular_velocity
float64[9] angular_velocity_covariance
geometry_msgs/Vector3 linear_acceleration
float64[9] linear_acceleration_covariance
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
"#;

#[derive(Debug, Clone)]
pub struct McapTopics {
    pub cloud: String,
    pub odom: String,
    pub tf: String,
    pub filtered_imu: String,
}

pub struct OfflineMcapWriter<W: Write + Seek> {
    writer: mcap::Writer<W>,
    cloud_channel: u16,
    odom_channel: u16,
    tf_channel: u16,
    filtered_imu_channel: u16,
    cloud_seq: u32,
    odom_seq: u32,
    tf_seq: u32,
    filtered_imu_seq: u32,
}

impl OfflineMcapWriter<BufWriter<File>> {
    pub fn create(path: &Path, topics: &McapTopics) -> Result<Self> {
        let file = File::create(path)
            .with_context(|| format!("Failed to create MCAP output '{}'", path.display()))?;
        Self::new(BufWriter::new(file), topics)
    }
}

impl<W: Write + Seek> OfflineMcapWriter<W> {
    pub fn new(writer: W, topics: &McapTopics) -> Result<Self> {
        let mut writer = mcap::WriteOptions::new()
            .profile("ros2")
            .library("pelorus")
            .create(writer)?;

        let cloud_schema = writer.add_schema(
            "sensor_msgs/msg/PointCloud2",
            "ros2msg",
            POINTCLOUD2_SCHEMA.as_bytes(),
        )?;
        let odom_schema = writer.add_schema(
            "nav_msgs/msg/Odometry",
            "ros2msg",
            ODOMETRY_SCHEMA.as_bytes(),
        )?;
        let tf_schema =
            writer.add_schema("tf2_msgs/msg/TFMessage", "ros2msg", TF_SCHEMA.as_bytes())?;
        let imu_schema =
            writer.add_schema("sensor_msgs/msg/Imu", "ros2msg", IMU_SCHEMA.as_bytes())?;

        let metadata = BTreeMap::new();
        let cloud_channel = writer.add_channel(cloud_schema, &topics.cloud, "cdr", &metadata)?;
        let odom_channel = writer.add_channel(odom_schema, &topics.odom, "cdr", &metadata)?;
        let tf_channel = writer.add_channel(tf_schema, &topics.tf, "cdr", &metadata)?;
        let filtered_imu_channel =
            writer.add_channel(imu_schema, &topics.filtered_imu, "cdr", &metadata)?;

        Ok(Self {
            writer,
            cloud_channel,
            odom_channel,
            tf_channel,
            filtered_imu_channel,
            cloud_seq: 0,
            odom_seq: 0,
            tf_seq: 0,
            filtered_imu_seq: 0,
        })
    }

    pub fn write_cloud(&mut self, msg: &PointCloud2) -> Result<()> {
        self.cloud_seq = self.cloud_seq.wrapping_add(1);
        write_msg(
            &mut self.writer,
            self.cloud_channel,
            self.cloud_seq,
            stamp_to_nanos(&msg.header.stamp),
            msg,
        )
    }

    pub fn write_odom(&mut self, msg: &Odometry) -> Result<()> {
        self.odom_seq = self.odom_seq.wrapping_add(1);
        write_msg(
            &mut self.writer,
            self.odom_channel,
            self.odom_seq,
            stamp_to_nanos(&msg.header.stamp),
            msg,
        )
    }

    pub fn write_tf(&mut self, msg: &TFMessage) -> Result<()> {
        self.tf_seq = self.tf_seq.wrapping_add(1);
        let time = msg
            .transforms
            .first()
            .map(|tf| stamp_to_nanos(&tf.header.stamp))
            .unwrap_or(0);
        write_msg(&mut self.writer, self.tf_channel, self.tf_seq, time, msg)
    }

    pub fn write_filtered_imu(&mut self, msg: &Imu) -> Result<()> {
        self.filtered_imu_seq = self.filtered_imu_seq.wrapping_add(1);
        write_msg(
            &mut self.writer,
            self.filtered_imu_channel,
            self.filtered_imu_seq,
            stamp_to_nanos(&msg.header.stamp),
            msg,
        )
    }

    pub fn finish(mut self) -> Result<()> {
        self.writer.finish()?;
        Ok(())
    }
}

fn write_msg<W, T>(
    writer: &mut mcap::Writer<W>,
    channel_id: u16,
    sequence: u32,
    time: u64,
    msg: &T,
) -> Result<()>
where
    W: Write + Seek,
    T: serde::Serialize,
{
    let bytes = cdr::serialize::<_, _, cdr::CdrLe>(msg, cdr::Infinite)?;
    writer.write_to_known_channel(
        &MessageHeader {
            channel_id,
            sequence,
            log_time: time,
            publish_time: time,
        },
        &bytes,
    )?;
    Ok(())
}

fn stamp_to_nanos(stamp: &Time) -> u64 {
    let sec = stamp.sec.max(0) as u64;
    sec.saturating_mul(1_000_000_000)
        .saturating_add(stamp.nanosec as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ros2_interfaces_jazzy_serde::{
        geometry_msgs::msg::TransformStamped, std_msgs::msg::Header, tf2_msgs::msg::TFMessage,
    };

    #[test]
    fn writes_configured_channels_and_timestamps() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "pelorus_mcap_output_test_{}_{}.mcap",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));

        let topics = McapTopics {
            cloud: "/cloud_out".to_string(),
            odom: "/odom_out".to_string(),
            tf: "/tf_out".to_string(),
            filtered_imu: "/imu_out".to_string(),
        };

        let mut writer = OfflineMcapWriter::create(&path, &topics)?;

        let stamp = Time {
            sec: 12,
            nanosec: 34,
        };
        let header = Header {
            stamp: stamp.clone(),
            frame_id: "lidar".to_string(),
        };

        let mut cloud = PointCloud2::default();
        cloud.header = header.clone();
        let mut odom = Odometry::default();
        odom.header = header.clone();
        let tf = TFMessage {
            transforms: vec![TransformStamped {
                header: header.clone(),
                child_frame_id: "base_link".to_string(),
                ..TransformStamped::default()
            }],
        };
        let mut imu = Imu::default();
        imu.header = header;

        writer.write_cloud(&cloud)?;
        writer.write_odom(&odom)?;
        writer.write_tf(&tf)?;
        writer.write_filtered_imu(&imu)?;
        writer.finish()?;

        let data = std::fs::read(&path)?;
        let _ = std::fs::remove_file(&path);

        let mut seen = BTreeMap::<String, u64>::new();
        for message in mcap::MessageStream::new(&data)? {
            let message = message?;
            seen.insert(message.channel.topic.clone(), message.log_time);
        }

        let expected_time = 12_000_000_034;
        assert_eq!(seen.get("/cloud_out"), Some(&expected_time));
        assert_eq!(seen.get("/odom_out"), Some(&expected_time));
        assert_eq!(seen.get("/tf_out"), Some(&expected_time));
        assert_eq!(seen.get("/imu_out"), Some(&expected_time));

        Ok(())
    }
}

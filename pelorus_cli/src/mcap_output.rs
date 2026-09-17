use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use mcap::records::MessageHeader;
use ros2_interfaces_jazzy_serde::{
    builtin_interfaces::msg::Time,
    nav_msgs::msg::{Odometry, Path as NavPath},
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

const PATH_SCHEMA: &str = r#"std_msgs/Header header
geometry_msgs/PoseStamped[] poses
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: geometry_msgs/PoseStamped
std_msgs/Header header
geometry_msgs/Pose pose
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

/// A ROS 2 message type together with the concatenated `.msg` definition that
/// MCAP consumers need in order to decode it.
#[derive(Debug, Clone, Copy)]
pub struct MsgSchema {
    pub name: &'static str,
    pub definition: &'static str,
}

pub const POINTCLOUD2_MSG: MsgSchema = MsgSchema {
    name: "sensor_msgs/msg/PointCloud2",
    definition: POINTCLOUD2_SCHEMA,
};
pub const ODOMETRY_MSG: MsgSchema = MsgSchema {
    name: "nav_msgs/msg/Odometry",
    definition: ODOMETRY_SCHEMA,
};
pub const PATH_MSG: MsgSchema = MsgSchema {
    name: "nav_msgs/msg/Path",
    definition: PATH_SCHEMA,
};
pub const TF_MSG: MsgSchema = MsgSchema {
    name: "tf2_msgs/msg/TFMessage",
    definition: TF_SCHEMA,
};
pub const IMU_MSG: MsgSchema = MsgSchema {
    name: "sensor_msgs/msg/Imu",
    definition: IMU_SCHEMA,
};

/// Handle for a topic registered with [`McapWriter::add_topic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopicId(usize);

struct TopicEntry {
    name: String,
    ros_type: &'static str,
    channel: u16,
    sequence: u32,
    message_count: u64,
}

/// Writes CDR-encoded ROS 2 messages to an MCAP file, optionally accompanied by
/// `metadata.yaml` so the result opens as a ROS 2 bag directory.
///
/// Topics are registered up front so metadata includes idle channels and remains
/// reproducible.
pub struct McapWriter<W: Write + Seek> {
    writer: mcap::Writer<W>,
    mcap_path: Option<PathBuf>,
    rosbag_metadata_path: Option<PathBuf>,
    topics: Vec<TopicEntry>,
    schemas: BTreeMap<&'static str, u16>,
    start_time: Option<u64>,
    end_time: Option<u64>,
}

impl McapWriter<BufWriter<File>> {
    pub fn create(path: &Path, rosbag_metadata_path: Option<PathBuf>) -> Result<Self> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Failed to create MCAP output directory '{}'",
                    parent.display()
                )
            })?;
        }
        let file = File::create(path)
            .with_context(|| format!("Failed to create MCAP output '{}'", path.display()))?;
        let mut writer = Self::new(BufWriter::new(file))?;
        writer.mcap_path = Some(path.to_path_buf());
        writer.rosbag_metadata_path = rosbag_metadata_path;
        Ok(writer)
    }
}

impl<W: Write + Seek> McapWriter<W> {
    pub fn new(writer: W) -> Result<Self> {
        let writer = mcap::WriteOptions::new()
            .profile("ros2")
            .library("pelorus")
            .create(writer)?;
        Ok(Self {
            writer,
            mcap_path: None,
            rosbag_metadata_path: None,
            topics: Vec::new(),
            schemas: BTreeMap::new(),
            start_time: None,
            end_time: None,
        })
    }

    /// Registers a topic and returns the handle used to write to it.
    ///
    /// Registration order is the order topics appear in `metadata.yaml`.
    pub fn add_topic(&mut self, topic: &str, schema: MsgSchema) -> Result<TopicId> {
        let schema_id = match self.schemas.get(schema.name) {
            Some(id) => *id,
            None => {
                let id =
                    self.writer
                        .add_schema(schema.name, "ros2msg", schema.definition.as_bytes())?;
                self.schemas.insert(schema.name, id);
                id
            }
        };
        let channel = self
            .writer
            .add_channel(schema_id, topic, "cdr", &BTreeMap::new())?;
        self.topics.push(TopicEntry {
            name: topic.to_string(),
            ros_type: schema.name,
            channel,
            sequence: 0,
            message_count: 0,
        });
        Ok(TopicId(self.topics.len() - 1))
    }

    pub fn write<T: serde::Serialize>(
        &mut self,
        topic: TopicId,
        msg: &T,
        log_time: u64,
    ) -> Result<()> {
        let entry = self
            .topics
            .get_mut(topic.0)
            .context("MCAP topic handle does not belong to this writer")?;
        entry.sequence = entry.sequence.wrapping_add(1);
        entry.message_count += 1;
        let bytes = cdr::serialize::<_, _, cdr::CdrLe>(msg, cdr::Infinite)?;
        self.writer.write_to_known_channel(
            &MessageHeader {
                channel_id: entry.channel,
                sequence: entry.sequence,
                log_time,
                publish_time: log_time,
            },
            &bytes,
        )?;
        self.start_time = Some(
            self.start_time
                .map_or(log_time, |start| start.min(log_time)),
        );
        self.end_time = Some(self.end_time.map_or(log_time, |end| end.max(log_time)));
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        self.writer.finish()?;
        if let Some(metadata_path) = self.rosbag_metadata_path.as_ref() {
            let mcap_path = self
                .mcap_path
                .as_deref()
                .context("Cannot write ROS bag metadata without an MCAP output path")?;
            write_rosbag_metadata(metadata_path, mcap_path, &self)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct McapTopics {
    pub cloud: String,
    pub odom: String,
    pub path: String,
    pub tf: String,
    /// Omitted for Lidar-only output, including its channel and schema.
    pub filtered_imu: Option<String>,
}

/// The Pelorus odometry output bag: registered clouds, odometry, path, TF, and
/// the filtered IMU stream.
pub struct McapOutputWriter<W: Write + Seek> {
    inner: McapWriter<W>,
    cloud: TopicId,
    odom: TopicId,
    path: TopicId,
    tf: TopicId,
    filtered_imu: Option<TopicId>,
}

impl McapOutputWriter<BufWriter<File>> {
    pub fn create(
        path: &Path,
        topics: &McapTopics,
        rosbag_metadata_path: Option<PathBuf>,
    ) -> Result<Self> {
        Self::from_writer(McapWriter::create(path, rosbag_metadata_path)?, topics)
    }
}

impl<W: Write + Seek> McapOutputWriter<W> {
    pub fn new(writer: W, topics: &McapTopics) -> Result<Self> {
        Self::from_writer(McapWriter::new(writer)?, topics)
    }

    fn from_writer(mut inner: McapWriter<W>, topics: &McapTopics) -> Result<Self> {
        let cloud = inner.add_topic(&topics.cloud, POINTCLOUD2_MSG)?;
        let odom = inner.add_topic(&topics.odom, ODOMETRY_MSG)?;
        let path = inner.add_topic(&topics.path, PATH_MSG)?;
        let tf = inner.add_topic(&topics.tf, TF_MSG)?;
        let filtered_imu = topics
            .filtered_imu
            .as_deref()
            .map(|topic| inner.add_topic(topic, IMU_MSG))
            .transpose()?;
        Ok(Self {
            inner,
            cloud,
            odom,
            path,
            tf,
            filtered_imu,
        })
    }

    pub fn write_cloud(&mut self, msg: &PointCloud2) -> Result<()> {
        let time = stamp_to_nanos(&msg.header.stamp);
        self.inner.write(self.cloud, msg, time)
    }

    pub fn write_odom(&mut self, msg: &Odometry) -> Result<()> {
        let time = stamp_to_nanos(&msg.header.stamp);
        self.inner.write(self.odom, msg, time)
    }

    pub fn write_path(&mut self, msg: &NavPath) -> Result<()> {
        let time = stamp_to_nanos(&msg.header.stamp);
        self.inner.write(self.path, msg, time)
    }

    pub fn write_tf(&mut self, msg: &TFMessage) -> Result<()> {
        let time = msg
            .transforms
            .first()
            .map(|tf| stamp_to_nanos(&tf.header.stamp))
            .unwrap_or(0);
        self.inner.write(self.tf, msg, time)
    }

    pub fn write_filtered_imu(&mut self, msg: &Imu) -> Result<()> {
        let Some(topic) = self.filtered_imu else {
            return Ok(());
        };
        let time = stamp_to_nanos(&msg.header.stamp);
        self.inner.write(topic, msg, time)
    }

    pub fn finish(self) -> Result<()> {
        self.inner.finish()
    }
}

/// Converts a backend-neutral point cloud into the CDR-serializable ROS message
/// this writer stores.
///
/// `PointCloud2Msg` keeps recording independent of the pub/sub backend.
pub fn cloud_msg_to_ros(cloud: ros_pointcloud2::PointCloud2Msg) -> PointCloud2 {
    use ros2_interfaces_jazzy_serde::sensor_msgs::msg::PointField;
    use ros2_interfaces_jazzy_serde::std_msgs::msg::Header;

    PointCloud2 {
        header: Header {
            stamp: Time {
                sec: cloud.header.stamp.sec,
                nanosec: cloud.header.stamp.nanosec,
            },
            frame_id: cloud.header.frame_id,
        },
        height: cloud.dimensions.height,
        width: cloud.dimensions.width,
        fields: cloud
            .fields
            .iter()
            .map(|field| PointField {
                name: field.name.to_string(),
                offset: field.offset,
                datatype: field.datatype,
                count: field.count,
            })
            .collect(),
        is_bigendian: matches!(cloud.endian, ros_pointcloud2::Endian::Big),
        point_step: cloud.point_step,
        row_step: cloud.row_step,
        data: cloud.data,
        is_dense: matches!(cloud.dense, ros_pointcloud2::Denseness::Dense),
    }
}

/// Converts a Pelorus IMU sample into the CDR-serializable ROS message this
/// writer stores.
pub fn imu_msg_to_ros(msg: &pelorus_core::ImuMsg) -> Imu {
    use ros2_interfaces_jazzy_serde::geometry_msgs::msg::{Quaternion, Vector3};
    use ros2_interfaces_jazzy_serde::std_msgs::msg::Header;

    Imu {
        header: Header {
            stamp: Time {
                sec: msg.stamp.sec,
                nanosec: msg.stamp.nanosec,
            },
            frame_id: msg.frame_id.clone(),
        },
        orientation: Quaternion {
            x: msg.orientation[1],
            y: msg.orientation[2],
            z: msg.orientation[3],
            w: msg.orientation[0],
        },
        orientation_covariance: msg.orientation_covariance,
        angular_velocity: Vector3 {
            x: msg.angular_velocity[0],
            y: msg.angular_velocity[1],
            z: msg.angular_velocity[2],
        },
        angular_velocity_covariance: msg.angular_velocity_covariance,
        linear_acceleration: Vector3 {
            x: msg.linear_acceleration[0],
            y: msg.linear_acceleration[1],
            z: msg.linear_acceleration[2],
        },
        linear_acceleration_covariance: msg.linear_acceleration_covariance,
    }
}

pub fn stamp_to_nanos(stamp: &Time) -> u64 {
    let sec = stamp.sec.max(0) as u64;
    sec.saturating_mul(1_000_000_000)
        .saturating_add(stamp.nanosec as u64)
}

fn write_rosbag_metadata<W: Write + Seek>(
    metadata_path: &Path,
    mcap_path: &Path,
    writer: &McapWriter<W>,
) -> Result<()> {
    let relative_mcap_path = metadata_path
        .parent()
        .and_then(|parent| mcap_path.strip_prefix(parent).ok())
        .unwrap_or(mcap_path);
    let relative_mcap_path = relative_mcap_path.to_string_lossy();
    let start_time = writer.start_time.unwrap_or(0);
    let duration = writer
        .end_time
        .zip(writer.start_time)
        .map(|(end, start)| end.saturating_sub(start))
        .unwrap_or(0);
    let message_count: u64 = writer.topics.iter().map(|topic| topic.message_count).sum();

    let mut metadata = String::new();
    metadata.push_str("rosbag2_bagfile_information:\n");
    metadata.push_str("  version: 9\n");
    metadata.push_str("  storage_identifier: mcap\n");
    metadata.push_str(&format!("  duration:\n    nanoseconds: {duration}\n"));
    metadata.push_str(&format!(
        "  starting_time:\n    nanoseconds_since_epoch: {start_time}\n"
    ));
    metadata.push_str(&format!("  message_count: {message_count}\n"));
    metadata.push_str("  topics_with_message_count:\n");
    for topic in &writer.topics {
        append_topic_metadata(
            &mut metadata,
            &topic.name,
            topic.ros_type,
            topic.message_count,
        );
    }
    metadata.push_str("  compression_format: \"\"\n");
    metadata.push_str("  compression_mode: \"\"\n");
    metadata.push_str("  relative_file_paths:\n");
    metadata.push_str(&format!("    - {}\n", yaml_quote(&relative_mcap_path)));
    metadata.push_str("  files:\n");
    metadata.push_str(&format!(
        "    - path: {}\n",
        yaml_quote(&relative_mcap_path)
    ));
    metadata.push_str(&format!(
        "      starting_time:\n        nanoseconds_since_epoch: {start_time}\n"
    ));
    metadata.push_str(&format!(
        "      duration:\n        nanoseconds: {duration}\n"
    ));
    metadata.push_str(&format!("      message_count: {message_count}\n"));
    metadata.push_str("  custom_data: {}\n");
    metadata.push_str("  ros_distro: jazzy\n");

    if let Some(parent) = metadata_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "Failed to create ROS bag metadata directory '{}'",
                parent.display()
            )
        })?;
    }
    std::fs::write(metadata_path, metadata).with_context(|| {
        format!(
            "Failed to write ROS bag metadata '{}'",
            metadata_path.display()
        )
    })
}

fn append_topic_metadata(metadata: &mut String, name: &str, ty: &str, message_count: u64) {
    metadata.push_str("    - topic_metadata:\n");
    metadata.push_str(&format!("        name: {}\n", yaml_quote(name)));
    metadata.push_str(&format!("        type: {}\n", yaml_quote(ty)));
    metadata.push_str("        serialization_format: cdr\n");
    metadata.push_str("        offered_qos_profiles: []\n");
    metadata.push_str("        type_description_hash: \"\"\n");
    metadata.push_str(&format!("      message_count: {message_count}\n"));
}

fn yaml_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ros2_interfaces_jazzy_serde::{
        geometry_msgs::msg::TransformStamped, std_msgs::msg::Header, tf2_msgs::msg::TFMessage,
    };

    #[test]
    fn lidar_only_output_has_no_imu_messages_channel_or_schema() -> Result<()> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let path = std::env::temp_dir().join(format!("pelorus_mcap_lidar_only_{nonce}.mcap"));
        let metadata_path =
            std::env::temp_dir().join(format!("pelorus_mcap_lidar_only_{nonce}.yaml"));
        let topics = McapTopics {
            cloud: "/cloud_out".to_string(),
            odom: "/odom_out".to_string(),
            path: "/path_out".to_string(),
            tf: "/tf_out".to_string(),
            filtered_imu: None,
        };
        let mut writer = McapOutputWriter::create(&path, &topics, Some(metadata_path.clone()))?;
        // Even an unexpected IMU sample must not create a channel or message.
        writer.write_filtered_imu(&Imu::default())?;
        writer.write_odom(&Odometry::default())?;
        writer.finish()?;

        let data = std::fs::read(&path)?;
        let metadata = std::fs::read_to_string(&metadata_path)?;
        std::fs::remove_file(&path)?;
        std::fs::remove_file(&metadata_path)?;
        let summary = mcap::Summary::read(&data)?.expect("MCAP summary");
        assert_eq!(summary.channels.len(), 4);
        assert!(
            summary
                .schemas
                .values()
                .all(|schema| schema.name != IMU_MSG.name)
        );
        assert!(!metadata.contains(IMU_MSG.name));
        let messages =
            mcap::MessageStream::new(&data)?.collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].channel.topic, "/odom_out");
        Ok(())
    }

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
            path: "/path_out".to_string(),
            tf: "/tf_out".to_string(),
            filtered_imu: Some("/imu_out".to_string()),
        };

        let mut writer = McapOutputWriter::create(&path, &topics, None)?;

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
        let mut path_msg = NavPath::default();
        path_msg.header = header.clone();
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
        writer.write_path(&path_msg)?;
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
        assert_eq!(seen.get("/path_out"), Some(&expected_time));
        assert_eq!(seen.get("/tf_out"), Some(&expected_time));
        assert_eq!(seen.get("/imu_out"), Some(&expected_time));

        Ok(())
    }

    #[test]
    fn odometry_metadata_still_lists_every_offered_topic() -> Result<()> {
        let nonce = nonce()?;
        let path = std::env::temp_dir().join(format!("pelorus_mcap_offered_{nonce}.mcap"));
        let metadata_path = std::env::temp_dir().join(format!("pelorus_mcap_offered_{nonce}.yaml"));

        let topics = McapTopics {
            cloud: "/cloud_out".to_string(),
            odom: "/odom_out".to_string(),
            path: "/path_out".to_string(),
            tf: "/tf_out".to_string(),
            filtered_imu: Some("/imu_out".to_string()),
        };
        let mut writer = McapOutputWriter::create(&path, &topics, Some(metadata_path.clone()))?;
        // Only one topic carries traffic.
        let mut odom = Odometry::default();
        odom.header.stamp = Time { sec: 5, nanosec: 0 };
        writer.write_odom(&odom)?;
        writer.finish()?;

        let metadata = std::fs::read_to_string(&metadata_path)?;
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&metadata_path);

        for topic in [
            "/cloud_out",
            "/odom_out",
            "/path_out",
            "/tf_out",
            "/imu_out",
        ] {
            assert!(
                metadata.contains(&format!("name: '{topic}'")),
                "metadata dropped {topic}"
            );
        }
        // Registration order decides the listing order.
        let cloud_at = metadata.find("name: '/cloud_out'").expect("cloud");
        let imu_at = metadata.find("name: '/imu_out'").expect("imu");
        assert!(cloud_at < imu_at);
        Ok(())
    }

    /// A raw run records only the sensors it drives.
    #[test]
    fn a_raw_bag_holds_only_the_topics_it_registered() -> Result<()> {
        let nonce = nonce()?;
        let path = std::env::temp_dir().join(format!("pelorus_mcap_raw_{nonce}.mcap"));
        let metadata_path = std::env::temp_dir().join(format!("pelorus_mcap_raw_{nonce}.yaml"));

        let mut writer = McapWriter::create(&path, Some(metadata_path.clone()))?;
        let imu_topic = writer.add_topic("/ouster/imu", IMU_MSG)?;

        let sample = pelorus_core::ImuMsg {
            stamp: ros_pointcloud2::ros::TimeMsg {
                sec: 7,
                nanosec: 500,
            },
            frame_id: "os_imu".to_string(),
            orientation: [1.0, 0.0, 0.0, 0.0],
            angular_velocity: [0.1, 0.2, 0.3],
            linear_acceleration: [0.0, 0.0, 9.81],
            ..Default::default()
        };
        let msg = imu_msg_to_ros(&sample);
        let time = stamp_to_nanos(&msg.header.stamp);
        writer.write(imu_topic, &msg, time)?;
        writer.finish()?;

        let data = std::fs::read(&path)?;
        let metadata = std::fs::read_to_string(&metadata_path)?;
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&metadata_path);

        let topics: Vec<String> = mcap::MessageStream::new(&data)?
            .map(|message| message.map(|message| message.channel.topic.clone()))
            .collect::<std::result::Result<_, _>>()?;
        assert_eq!(topics, vec!["/ouster/imu".to_string()]);

        assert!(metadata.contains("name: '/ouster/imu'"));
        assert!(metadata.contains("type: 'sensor_msgs/msg/Imu'"));
        // No point-cloud channel was registered, so none may be advertised.
        assert!(!metadata.contains("sensor_msgs/msg/PointCloud2"));
        assert!(metadata.contains("message_count: 1"));

        // The bag is stamped from the sensor clock, not the wall clock.
        assert!(metadata.contains("nanoseconds_since_epoch: 7000000500"));
        Ok(())
    }

    /// The IMU reaches the bag exactly as the driver reported it. `ImuMsg`
    /// stores the quaternion `wxyz`. The ROS message stores it `xyzw`.
    #[test]
    fn imu_conversion_preserves_quaternion_order() {
        let sample = pelorus_core::ImuMsg {
            orientation: [0.1, 0.2, 0.3, 0.4],
            ..Default::default()
        };
        let msg = imu_msg_to_ros(&sample);
        assert_eq!(msg.orientation.w, 0.1);
        assert_eq!(msg.orientation.x, 0.2);
        assert_eq!(msg.orientation.y, 0.3);
        assert_eq!(msg.orientation.z, 0.4);
    }

    fn nonce() -> Result<String> {
        Ok(format!(
            "{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ))
    }

    #[test]
    fn writes_rosbag_metadata_when_configured() -> Result<()> {
        let nonce = format!(
            "{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        );
        let path = std::env::temp_dir().join(format!("pelorus_mcap_output_test_{nonce}.mcap"));
        let metadata_path =
            std::env::temp_dir().join(format!("pelorus_mcap_output_metadata_test_{nonce}.yaml"));

        let topics = McapTopics {
            cloud: "/cloud_out".to_string(),
            odom: "/odom_out".to_string(),
            path: "/path_out".to_string(),
            tf: "/tf_out".to_string(),
            filtered_imu: Some("/imu_out".to_string()),
        };

        let mut writer = McapOutputWriter::create(&path, &topics, Some(metadata_path.clone()))?;

        let stamp = Time {
            sec: 12,
            nanosec: 34,
        };
        let header = Header {
            stamp,
            frame_id: "lidar".to_string(),
        };
        let mut cloud = PointCloud2::default();
        cloud.header = header;

        writer.write_cloud(&cloud)?;
        writer.finish()?;

        let metadata = std::fs::read_to_string(&metadata_path)?;
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&metadata_path);

        assert!(metadata.contains("rosbag2_bagfile_information:"));
        assert!(metadata.contains("storage_identifier: mcap"));
        assert!(metadata.contains("message_count: 1"));
        assert!(metadata.contains("name: '/cloud_out'"));
        assert!(metadata.contains("type: 'sensor_msgs/msg/PointCloud2'"));
        assert!(metadata.contains("offered_qos_profiles: []"));
        assert!(metadata.contains("type_description_hash: \"\""));
        assert!(metadata.contains("custom_data: {}"));
        assert!(metadata.contains("ros_distro: jazzy"));
        assert!(metadata.contains("starting_time:\n    nanoseconds_since_epoch: 12000000034"));

        Ok(())
    }
}

#![warn(clippy::std_instead_of_alloc, clippy::std_instead_of_core)]

extern crate alloc;

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
use alloc::collections::VecDeque;
use alloc::sync::Arc;
#[allow(unused_imports)]
use anyhow::Error;
use anyhow::anyhow;
use clap::{CommandFactory, ValueEnum};
use clap_complete::{Shell, generate};
use core::error::Error as CoreError;
#[cfg(feature = "rat")]
use core::panic::AssertUnwindSafe;
use core::sync::atomic::{AtomicBool, Ordering};
use core::time::Duration;
#[cfg(feature = "rat")]
use marina::{Marina, ProgressReporter, ResolveResult, WriterProgress};
#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;
use pelorus_core::LioConfig;
use pelorus_core::config::{ast_to_config, ast_to_config_with_user_usage};
#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
use pelorus_core::timestamp_to_secs;
#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
use pelorus_core::timing::RuntimeEvaluator;
use pelorus_reg::ActiveBackend;
#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
use pelorus_reg::pipeline::Pipeline;
#[cfg(feature = "r2r")]
use prelio_io::impl_r2r;
#[cfg(feature = "mt_pubsub")]
use prelio_io::impl_ros2_interfaces_jazzy_rkyv;
#[cfg(feature = "ros2-client")]
use ros2_interfaces_jazzy_serde::{
    builtin_interfaces::msg::Time,
    geometry_msgs::msg::{Point, PoseStamped, Quaternion, Transform, TransformStamped, Vector3},
    nav_msgs::msg::{Odometry, Path},
    sensor_msgs::msg::PointCloud2,
    std_msgs::msg::Header,
};
#[cfg(feature = "rat")]
use std::fs;
#[cfg(feature = "rat")]
use std::process::{Command as ProcessCommand, Stdio};
#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
const PATH_LIMIT: usize = 10_000;
#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
const MCAP_PATH_LIMIT: usize = 512;
#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
const MCAP_PATH_MIN_PERIOD_NS: u64 = 1_000_000_000;
const DEFAULT_OFFLINE_PREFETCH_SCANS: usize = 4;

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
use ros_pointcloud2::prelude::*;

#[global_allocator]
#[cfg(feature = "mimalloc")]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{Parser, Subcommand};
#[allow(unused_imports)]
use futures_lite::future;
use log::{debug, error, info, trace, warn};
#[cfg(feature = "mt_pubsub")]
use mt_pubsub::Node;
#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
use nalgebra::{Isometry3, Translation3, UnitQuaternion};
#[cfg(feature = "r2r")]
#[allow(unused_imports)]
use r2r::{
    QosProfile,
    builtin_interfaces::msg::Time,
    geometry_msgs::msg::{Point, PoseStamped, Quaternion, Transform, TransformStamped, Vector3},
    nav_msgs::msg::{Odometry, Path},
    sensor_msgs::msg::PointCloud2,
    std_msgs::msg::Header,
    visualization_msgs::msg::Marker,
};
#[cfg(feature = "rerun")]
use rerun::external::glam;
#[cfg(feature = "ros2-client")]
use ros2_client::{NodeName, NodeOptions};
#[cfg(feature = "mt_pubsub")]
#[allow(unused_imports)]
use ros2_interfaces_jazzy_rkyv::{
    builtin_interfaces::msg::Time,
    geometry_msgs::msg::{Point, PoseStamped, Quaternion, Transform, TransformStamped, Vector3},
    nav_msgs::msg::{Odometry, Path},
    sensor_msgs::msg::PointCloud2,
    std_msgs::msg::{Header, String as RosString},
    visualization_msgs::msg::Marker,
};
use std::io::IsTerminal;
#[cfg(feature = "rat")]
use std::io::Write;

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
use prelio_io::prelio::LocatedBufferView;
#[cfg(feature = "ros2-client")]
use prelio_io::tf::TfBroadcaster;
#[cfg(feature = "mt_pubsub")]
use prelio_io::tf_lh::TfBroadcaster;
#[cfg(feature = "r2r")]
use prelio_io::tf_r2r::TfBroadcaster;

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
use pelorus_reg::Registration;
#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
use prelio_io::PrelioNode;
use std::path::PathBuf;

use pelorus_cli::mcap_output::{McapTopics, OfflineMcapWriter};
use pelorus_cli::stats_tui::{
    SharedStats, StatsTui, init_logger, set_tui_active, update_log_level,
};
#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "rat"
))]
use std::time::UNIX_EPOCH;

#[allow(dead_code)]
fn lerp_color(c1: (f32, f32, f32), c2: (f32, f32, f32), t: f32) -> (f32, f32, f32) {
    (
        c1.0 * (1.0 - t) + c2.0 * t,
        c1.1 * (1.0 - t) + c2.1 * t,
        c1.2 * (1.0 - t) + c2.2 * t,
    )
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Clone, Subcommand)]
#[command()]
pub enum Command {
    /// Spawn the server process and wait for input using the compiled data provider
    Online(ConfigArgs),

    /// Use a Bagfile directly and run synchronously
    #[cfg(feature = "rat")]
    Offline(OfflineArgs),

    /// Create or validate configuration files
    #[command(subcommand)]
    Config(ConfigCommand),

    /// Generate shell completions
    /// Example usage: `pelorus completions bash > pelorus_completions.sh`
    Completions {
        /// The shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

#[derive(Debug, Clone, Parser)]
pub struct OfflineArgs {
    /// Optional path to a custom '.rl' configuration file.
    /// If not provided, only the default configuration will be used
    #[arg(short, long)]
    pub config_path: Option<PathBuf>,

    /// Local bagfile path (file or directory), or a marina tag to resolve and optionally pull.
    pub bagfile_path: PathBuf,

    /// Path to the Minot binary
    #[arg(short, long, default_value = "minot")]
    pub minot_path: PathBuf,

    /// Export a TUM trajectory file.
    /// If not provided, trajectory is not exported.
    /// If provided without value, automatically generates a filename.
    /// If provided with a path, exports to that exact location.
    #[arg(
        short,
        long,
        value_name = "PATH",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    pub export_trajectory: Option<String>,

    /// Force CPU-only processing regardless of available GPU backends.
    #[arg(long, default_value_t = false)]
    pub cpu_only: bool,

    /// Answer yes to overwrite prompts.
    #[arg(short = 'y', long, default_value_t = false)]
    pub yes: bool,

    /// Number of offline scans to request ahead from Minot.
    ///
    /// Registration and publishing remain gated one scan at a time; this only
    /// keeps the preprocessing side fed while the current scan is registering.
    #[arg(long, default_value_t = DEFAULT_OFFLINE_PREFETCH_SCANS)]
    pub prefetch_scans: usize,

    /// Write Pelorus offline outputs to this MCAP bagfile.
    #[arg(long, value_name = "PATH", conflicts_with = "output_rosbag")]
    pub output_mcap: Option<PathBuf>,

    /// Also write ROS 2 bag metadata.yaml for the MCAP output.
    #[arg(long, value_name = "PATH", requires = "output_mcap")]
    pub output_rosbag_metadata: Option<PathBuf>,

    /// Write Pelorus offline outputs as a ROS 2 MCAP bag directory.
    #[arg(long, value_name = "DIR", conflicts_with_all = ["output_mcap", "output_rosbag_metadata"])]
    pub output_rosbag: Option<PathBuf>,

    /// MCAP topic for undistorted PointCloud2 output.
    #[arg(long, default_value = "/registered")]
    pub mcap_cloud_topic: String,

    /// MCAP topic for Pelorus odometry output.
    #[arg(long, default_value = "/odom")]
    pub mcap_odom_topic: String,

    /// MCAP topic for Pelorus path output.
    #[arg(long, default_value = "/path")]
    pub mcap_path_topic: String,

    /// MCAP topic for dynamic TF output.
    #[arg(long, default_value = "/tf")]
    pub mcap_tf_topic: String,

    /// MCAP topic for filtered IMU output. Defaults to the configured filtered IMU topic.
    #[arg(long)]
    pub mcap_filtered_imu_topic: Option<String>,
}

#[derive(Debug, Clone, Parser)]
pub struct ConfigArgs {
    /// Optional path to a custom '.rl' configuration file to overwrite defaults.
    /// If not provided, only the default configuration will be used
    #[arg(short, long)]
    pub config_path: Option<PathBuf>,

    /// Export a TUM trajectory file.
    /// If not provided, trajectory is not exported.
    /// If provided without value, automatically generates a filename.
    /// If provided with a path, exports to that exact location.
    #[arg(
        short,
        long,
        value_name = "PATH",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    pub export_trajectory: Option<String>,

    /// Publish tick_end checkpoint using ratlib from Minot.
    #[arg(short, long)]
    pub rat: bool,

    /// Force CPU-only processing regardless of available GPU backends.
    #[arg(long, default_value_t = false)]
    pub cpu_only: bool,
}

#[derive(Debug, Clone, Subcommand)]
pub enum ConfigCommand {
    /// Generate a configuration. Choose 'recommended' for recommended settings or 'full' for the complete default config.
    Generate(GenerateArgs),
    /// Compiles and validates a given configuration file against the defaults
    Verify(TestArgs),
    /// Listen to ROS2 TF and topic frame IDs, then write the discovered transforms to a config file
    #[cfg(feature = "r2r")]
    TfAutoRos2(TfAutoRos2Args),
}

#[cfg(feature = "r2r")]
#[derive(Debug, Clone, Parser)]
pub struct TfAutoRos2Args {
    /// Output configuration file path
    pub config: PathBuf,

    /// Seconds to wait for TF data and topic messages
    #[arg(long, default_value_t = 1.0)]
    pub timeout: f64,

    /// Optional path to a custom '.rl' configuration file to read topics from.
    /// If not provided, the default configuration values are used
    #[arg(short, long)]
    pub config_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Parser)]
pub struct TestArgs {
    /// The path of the configuration file to test. Use '-' to read from stdin
    #[arg(default_value = "pelorus.rl")]
    pub config_path: PathBuf,
}

#[derive(Debug, Clone, Parser)]
pub struct GenerateArgs {
    /// Which configuration to generate: 'recommended' or 'full'
    #[arg(value_enum)]
    pub kind: GenerateKind,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum GenerateKind {
    Recommended,
    Full,
}

#[derive(Debug, Clone)]
struct OfflineMcapConfig {
    path: PathBuf,
    rosbag_metadata_path: Option<PathBuf>,
    topics: McapTopics,
}

fn normalize_mcap_topic(topic: impl Into<String>) -> String {
    let topic = topic.into();
    if topic.starts_with('/') {
        topic
    } else {
        format!("/{topic}")
    }
}

fn rosbag_mcap_path(rosbag_path: &std::path::Path) -> PathBuf {
    let bag_name = rosbag_path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("output");
    rosbag_path.join(format!("{bag_name}.mcap"))
}

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
fn copy_array<const N: usize>(values: &[f64]) -> [f64; N] {
    let mut out = [0.0; N];
    out.copy_from_slice(&values[..N]);
    out
}

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
fn serde_time(time: &Time) -> ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time {
    ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time {
        sec: time.sec,
        nanosec: time.nanosec,
    }
}

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
fn serde_header(header: &Header) -> ros2_interfaces_jazzy_serde::std_msgs::msg::Header {
    ros2_interfaces_jazzy_serde::std_msgs::msg::Header {
        stamp: serde_time(&header.stamp),
        frame_id: header.frame_id.clone(),
    }
}

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
fn serde_vector3(vector: &Vector3) -> ros2_interfaces_jazzy_serde::geometry_msgs::msg::Vector3 {
    ros2_interfaces_jazzy_serde::geometry_msgs::msg::Vector3 {
        x: vector.x,
        y: vector.y,
        z: vector.z,
    }
}

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
fn serde_quaternion(
    quat: &Quaternion,
) -> ros2_interfaces_jazzy_serde::geometry_msgs::msg::Quaternion {
    ros2_interfaces_jazzy_serde::geometry_msgs::msg::Quaternion {
        x: quat.x,
        y: quat.y,
        z: quat.z,
        w: quat.w,
    }
}

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
fn serde_point(point: &Point) -> ros2_interfaces_jazzy_serde::geometry_msgs::msg::Point {
    ros2_interfaces_jazzy_serde::geometry_msgs::msg::Point {
        x: point.x,
        y: point.y,
        z: point.z,
    }
}

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
fn serde_cloud(cloud: &PointCloud2) -> ros2_interfaces_jazzy_serde::sensor_msgs::msg::PointCloud2 {
    ros2_interfaces_jazzy_serde::sensor_msgs::msg::PointCloud2 {
        header: serde_header(&cloud.header),
        height: cloud.height,
        width: cloud.width,
        fields: cloud
            .fields
            .iter()
            .map(
                |field| ros2_interfaces_jazzy_serde::sensor_msgs::msg::PointField {
                    name: field.name.clone(),
                    offset: field.offset,
                    datatype: field.datatype,
                    count: field.count,
                },
            )
            .collect(),
        is_bigendian: cloud.is_bigendian,
        point_step: cloud.point_step,
        row_step: cloud.row_step,
        data: cloud.data.clone(),
        is_dense: cloud.is_dense,
    }
}

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
fn serde_odom(odom: &Odometry) -> ros2_interfaces_jazzy_serde::nav_msgs::msg::Odometry {
    ros2_interfaces_jazzy_serde::nav_msgs::msg::Odometry {
        header: serde_header(&odom.header),
        child_frame_id: odom.child_frame_id.clone(),
        pose: ros2_interfaces_jazzy_serde::geometry_msgs::msg::PoseWithCovariance {
            pose: ros2_interfaces_jazzy_serde::geometry_msgs::msg::Pose {
                position: serde_point(&odom.pose.pose.position),
                orientation: serde_quaternion(&odom.pose.pose.orientation),
            },
            covariance: copy_array::<36>(&odom.pose.covariance[..]),
        },
        twist: ros2_interfaces_jazzy_serde::geometry_msgs::msg::TwistWithCovariance {
            twist: ros2_interfaces_jazzy_serde::geometry_msgs::msg::Twist {
                linear: serde_vector3(&odom.twist.twist.linear),
                angular: serde_vector3(&odom.twist.twist.angular),
            },
            covariance: copy_array::<36>(&odom.twist.covariance[..]),
        },
    }
}

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
fn serde_pose_stamped(
    pose: &PoseStamped,
) -> ros2_interfaces_jazzy_serde::geometry_msgs::msg::PoseStamped {
    ros2_interfaces_jazzy_serde::geometry_msgs::msg::PoseStamped {
        header: serde_header(&pose.header),
        pose: ros2_interfaces_jazzy_serde::geometry_msgs::msg::Pose {
            position: serde_point(&pose.pose.position),
            orientation: serde_quaternion(&pose.pose.orientation),
        },
    }
}

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
fn serde_path(path: &Path) -> ros2_interfaces_jazzy_serde::nav_msgs::msg::Path {
    ros2_interfaces_jazzy_serde::nav_msgs::msg::Path {
        header: serde_header(&path.header),
        poses: path.poses.iter().map(serde_pose_stamped).collect(),
    }
}

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
fn serde_tf_message(
    tf: &TransformStamped,
) -> ros2_interfaces_jazzy_serde::tf2_msgs::msg::TFMessage {
    ros2_interfaces_jazzy_serde::tf2_msgs::msg::TFMessage {
        transforms: vec![
            ros2_interfaces_jazzy_serde::geometry_msgs::msg::TransformStamped {
                header: serde_header(&tf.header),
                child_frame_id: tf.child_frame_id.clone(),
                transform: ros2_interfaces_jazzy_serde::geometry_msgs::msg::Transform {
                    translation: serde_vector3(&tf.transform.translation),
                    rotation: serde_quaternion(&tf.transform.rotation),
                },
            },
        ],
    }
}

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
fn serde_imu_msg(msg: &pelorus_core::ImuMsg) -> ros2_interfaces_jazzy_serde::sensor_msgs::msg::Imu {
    ros2_interfaces_jazzy_serde::sensor_msgs::msg::Imu {
        header: ros2_interfaces_jazzy_serde::std_msgs::msg::Header {
            stamp: ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time {
                sec: msg.stamp.sec,
                nanosec: msg.stamp.nanosec,
            },
            frame_id: msg.frame_id.clone(),
        },
        orientation: ros2_interfaces_jazzy_serde::geometry_msgs::msg::Quaternion {
            x: msg.orientation[1],
            y: msg.orientation[2],
            z: msg.orientation[3],
            w: msg.orientation[0],
        },
        orientation_covariance: msg.orientation_covariance,
        angular_velocity: ros2_interfaces_jazzy_serde::geometry_msgs::msg::Vector3 {
            x: msg.angular_velocity[0],
            y: msg.angular_velocity[1],
            z: msg.angular_velocity[2],
        },
        angular_velocity_covariance: msg.angular_velocity_covariance,
        linear_acceleration: ros2_interfaces_jazzy_serde::geometry_msgs::msg::Vector3 {
            x: msg.linear_acceleration[0],
            y: msg.linear_acceleration[1],
            z: msg.linear_acceleration[2],
        },
        linear_acceleration_covariance: msg.linear_acceleration_covariance,
    }
}

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
fn drain_filtered_imu_mcap(
    writer: &mut Option<OfflineMcapWriter<std::io::BufWriter<std::fs::File>>>,
    rx: &mut Option<std::sync::mpsc::Receiver<pelorus_core::ImuMsg>>,
) -> anyhow::Result<()> {
    let (Some(writer), Some(rx)) = (writer.as_mut(), rx.as_mut()) else {
        return Ok(());
    };

    while let Ok(msg) = rx.try_recv() {
        writer.write_filtered_imu(&serde_imu_msg(&msg))?;
    }
    Ok(())
}

fn lio(
    config: LioConfig,
    export_trajectory: Option<String>,
    synchronous_mode: bool,
) -> Result<(), Box<dyn CoreError>> {
    lio_with_ready_signal(
        config,
        None,
        export_trajectory,
        synchronous_mode,
        None,
        None,
        None,
        1,
        None,
    )
}

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "rat"
))]
fn resolve_export_trajectory_path(export_trajectory: Option<String>) -> Option<String> {
    match export_trajectory {
        Some(path) if path.is_empty() => {
            let name = petname::petname(2, "_").unwrap_or_else(|| "unnamed".to_string());
            Some(format!(
                "pelorus_trajectory_{}_{}.tum",
                std::time::SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("time goes forward")
                    .as_secs(),
                name
            ))
        }
        other => other,
    }
}

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
fn release_minot_tick(
    shutdown_flag: Option<&Arc<AtomicBool>>,
    #[allow(unused_variables)] synchronous_mode: bool,
) {
    release_minot_ticks(shutdown_flag, synchronous_mode, 1);
}

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
fn release_minot_ticks(
    shutdown_flag: Option<&Arc<AtomicBool>>,
    #[allow(unused_variables)] synchronous_mode: bool,
    #[allow(unused_variables)] count: usize,
) {
    if shutdown_flag
        .map(|flag| flag.load(Ordering::Relaxed))
        .unwrap_or(false)
    {
        return;
    }

    #[cfg(feature = "rat")]
    if synchronous_mode {
        static MINOT_MONO_START: std::sync::OnceLock<std::time::Instant> =
            std::sync::OnceLock::new();
        for _ in 0..count {
            let now_ns = MINOT_MONO_START
                .get_or_init(std::time::Instant::now)
                .elapsed()
                .as_nanos() as f64;
            let mut ts = rat::NetArray::from(nalgebra::DMatrix::<f64>::from_element(1, 1, now_ns));
            rat::bacon("pelorus:sd_end", &mut ts, rat::VariableType::F64).unwrap();
            let mut tick = String::new();
            rat::bacon("tick_end", &mut tick, rat::VariableType::StaticOnly).unwrap();
        }
    }
}

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
fn lio_with_ready_signal(
    config: LioConfig,
    ready_signal: Option<std::sync::mpsc::Sender<()>>,
    export_trajectory: Option<String>,
    #[allow(unused_variables)] synchronous_mode: bool,
    shared_stats: Option<Arc<SharedStats>>,
    // Optional flag that, when set to true, will cause the LIO loop to exit
    shutdown_flag: Option<Arc<AtomicBool>>,
    // Optional blocking shutdown signal for pipeline teardown mode.
    shutdown_request_rx: Option<std::sync::mpsc::Receiver<OfflineShutdownRequest>>,
    offline_prefetch_scans: usize,
    output_mcap: Option<OfflineMcapConfig>,
) -> Result<(), Box<dyn CoreError>> {
    let export_trajectory = resolve_export_trajectory_path(export_trajectory);
    if let Some(path) = export_trajectory.as_ref() {
        warn!("Trajectory export target: {}", path);
    }
    #[cfg(any(feature = "mt_pubsub", feature = "rat"))]
    let rt = if cfg!(feature = "mt_pubsub") || (synchronous_mode && cfg!(feature = "rat")) {
        Some(Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap(),
        ))
    } else {
        None
    };

    #[cfg(not(feature = "mt_pubsub"))]
    #[allow(unused_variables)]
    let namespace = "/pelorus";
    #[cfg(feature = "ros2-client")]
    let ctx = ros2_client::Context::new()?;
    #[cfg(feature = "ros2-client")]
    let mut node = ctx.new_node(
        NodeName::new(namespace, "pelorus_node")?,
        NodeOptions::new().enable_rosout(true),
    )?;
    #[cfg(feature = "mt_pubsub")]
    let node = Arc::new(
        rt.as_ref()
            .expect("rt always exists with mt_pubsub feat")
            .block_on(async {
                Node::create(mt_pubsub::NodeConfig {
                    name: "pelorus_node".to_owned(),
                    mode: mt_pubsub::Qos::Reliable,
                    coord_mode: mt_pubsub::CoordMode::AutoStart,
                })
                .await
                .unwrap()
            }),
    );

    #[cfg(feature = "rat")]
    if synchronous_mode {
        if let Some(rt) = rt.as_ref() {
            rat::init("pelorus", None, Some(Arc::clone(rt))).unwrap();
        }
    }

    #[cfg(feature = "r2r")]
    let ctx = r2r::Context::create()?;
    #[cfg(feature = "r2r")]
    let mut node = r2r::Node::create(ctx, "pelorus", "pelorus")?;

    #[cfg(not(feature = "mt_pubsub"))]
    let tf_broadcaster = TfBroadcaster::new(&mut node);
    #[cfg(feature = "mt_pubsub")]
    let tf_broadcaster = Arc::new(TfBroadcaster::new(
        &node,
        rt.as_ref()
            .expect("rt always exists with mt_pubsub feat")
            .clone(),
    ));

    let mut pipeline = Pipeline::new(&config);
    let map = pipeline.map();
    let ext = pipeline.ext();
    let _odom_ext = Isometry3::from_parts(
        nalgebra::Translation3::new(0.0, 0.0, 0.0),
        UnitQuaternion::from_matrix(&config.prelio_params.wheel_to_imu_rotation),
    );

    // Set lidar vendor in shared stats if present
    if let Some(ref stats) = shared_stats {
        stats.set_lidar_vendor(config.prelio_params.lidar_vendor);
    }

    #[cfg(feature = "rerun")]
    let rec = {
        let rerun_enabled = std::env::var("RERUN_ENABLED")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);

        if rerun_enabled {
            let rerun_server_addr =
                std::env::var("RERUN_SERVER_ADDR").unwrap_or("127.0.0.1:9876".to_string());

            // First check if the server is reachable via TCP
            let is_reachable = std::net::TcpStream::connect_timeout(
                &rerun_server_addr
                    .parse()
                    .unwrap_or_else(|_| std::net::SocketAddr::from(([127, 0, 0, 1], 9876))),
                Duration::from_millis(500),
            )
            .is_ok();

            if !is_reachable {
                info!(
                    "Rerun server not reachable at {}. Visualization disabled.",
                    rerun_server_addr
                );
                None
            } else {
                // Server is reachable, try to connect
                match rerun::RecordingStreamBuilder::new("pelorus")
                    .connect_grpc_opts(format!("rerun+http://{}/proxy", rerun_server_addr))
                {
                    Ok(stream) => {
                        info!("Connected to Rerun at {}", rerun_server_addr);
                        Some(Arc::new(stream))
                    }
                    Err(e) => {
                        warn!(
                            "Could not connect to Rerun at {}: {}. Visualization disabled.",
                            rerun_server_addr, e
                        );
                        None
                    }
                }
            }
        } else {
            info!("Rerun disabled via RERUN_ENABLED=false");
            None
        }
    };

    #[cfg(not(feature = "rerun"))]
    let _rerun_not_enabled = ();

    let (ready_for_next_tx, ready_for_next_rx) = std::sync::mpsc::channel();
    let minot_replay_mode = synchronous_mode && shutdown_request_rx.is_some();
    let (filtered_imu_tx, filtered_imu_rx) = if output_mcap.is_some() {
        let (tx, rx) = std::sync::mpsc::channel();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    let (lifis_node, synced_rx) = PrelioNode::new(
        #[cfg(not(feature = "mt_pubsub"))]
        &mut node,
        #[cfg(feature = "mt_pubsub")]
        &node,
        #[cfg(feature = "mt_pubsub")]
        rt.as_ref()
            .expect("rt always exists with mt_pubsub feat")
            .clone(),
        &config.prelio_params,
        &config.prelio_node_params,
        Some(pipeline.build_init_callback()),
        Some(pipeline.build_undistort()),
        Some(pipeline.build_init_guess()),
        #[cfg(feature = "rerun")]
        rec.clone(),
        Some(ready_for_next_rx),
        filtered_imu_tx,
    )?;
    let lifis_node = Arc::new(lifis_node);
    let drain_on_shutdown = Arc::new(AtomicBool::new(false));
    if let Some(stop_rx) = shutdown_request_rx {
        let node_for_stop = Arc::clone(&lifis_node);
        let drain_on_shutdown_for_stop = Arc::clone(&drain_on_shutdown);
        let shutdown_flag_for_stop = shutdown_flag.clone();
        std::thread::spawn(move || {
            if let Ok(request) = stop_rx.recv() {
                match request {
                    OfflineShutdownRequest::DrainAfterEndOfBag => {
                        drain_on_shutdown_for_stop.store(true, Ordering::SeqCst);
                    }
                    OfflineShutdownRequest::Immediate => {
                        node_for_stop.request_shutdown();
                        if let Some(flag) = shutdown_flag_for_stop.as_ref() {
                            flag.store(true, Ordering::SeqCst);
                        }
                    }
                }
            }
        });
    }

    // #[cfg(feature = "ros2-client")]
    // {
    //     #[cfg(all(feature = "map", not(feature = "no-viz")))]
    //     {
    //         let map_publisher = node
    //             .create_publisher::<PointCloud2>(
    //                 &node
    //                     .create_topic(
    //                         &ros2_client::Name::new(&namespace, "map").unwrap(),
    //                         ros2_client::MessageTypeName::new("sensor_msgs", "PointCloud2"),
    //                         &Qos::default().into(),
    //                     )
    //                     .unwrap(),
    //                 None,
    //             )
    //             .unwrap();
    //     }
    // }
    #[cfg(feature = "ros2-client")]
    let cloud_publisher = node
        .create_publisher(
            &node
                .create_topic(
                    &ros2_client::Name::new(&namespace, "registered").unwrap(),
                    ros2_client::MessageTypeName::new("sensor_msgs", "PointCloud2"),
                    &prelio_io::qos_to_ros2(config.registered_cloud_qos),
                )
                .unwrap(),
            Some(prelio_io::qos_to_ros2(config.registered_cloud_qos)),
        )
        .unwrap();
    #[cfg(any(feature = "ros2-client"))]
    let odom_publisher = node
        .create_publisher(
            &node
                .create_topic(
                    &ros2_client::Name::new(&namespace, "odom").unwrap(),
                    ros2_client::MessageTypeName::new("nav_msgs", "Odometry"),
                    &prelio_io::qos_to_ros2(config.odom_qos),
                )
                .unwrap(),
            Some(prelio_io::qos_to_ros2(config.odom_qos)),
        )
        .unwrap();
    #[cfg(all(not(feature = "no-viz"), feature = "ros2-client"))]
    let lidar_path_publisher = node
        .create_publisher(
            &node
                .create_topic(
                    &ros2_client::Name::new(&namespace, "path").unwrap(),
                    ros2_client::MessageTypeName::new("nav_msgs", "Path"),
                    &prelio_io::qos_to_ros2(config.path_qos),
                )
                .unwrap(),
            Some(prelio_io::qos_to_ros2(config.path_qos)),
        )
        .unwrap();
    #[cfg(all(not(feature = "no-viz"), feature = "ros2-client"))]
    let imu_odom_publisher = node
        .create_publisher(
            &node
                .create_topic(
                    &ros2_client::Name::new(&namespace, "imu_odom").unwrap(),
                    ros2_client::MessageTypeName::new("nav_msgs", "Odometry"),
                    &prelio_io::qos_to_ros2(config.imu_odom_qos),
                )
                .unwrap(),
            Some(prelio_io::qos_to_ros2(config.imu_odom_qos)),
        )
        .unwrap();

    // #[cfg(all(not(feature = "no-viz"), feature = "ros2-client"))]
    // let imu_path_publisher = node
    //     .create_publisher(
    //         &node
    //             .create_topic(
    //                 &ros2_client::Name::new(&namespace, "imu_path").unwrap(),
    //                 ros2_client::MessageTypeName::new("nav_msgs", "Path"),
    //                 &config.imu_path_qos.into(),
    //             )
    //             .unwrap(),
    //         None,
    //     )
    //     .unwrap();
    // #[cfg(all(
    //     not(feature = "no-viz"),
    //     feature = "ros2-client",
    //     not(feature = "rerun")
    // ))]
    // let registered_visual_publisher = node
    //     .create_publisher::<Marker>(
    //         &node
    //             .create_topic(
    //                 &ros2_client::Name::new(&format!("{namespace}/viz"), "registered").unwrap(),
    //                 ros2_client::MessageTypeName::new("visualization_msgs", "Marker"),
    //                 &ros2_client::DEFAULT_PUBLISHER_QOS,
    //             )
    //             .unwrap(),
    //         None,
    //     )
    //     .unwrap();
    // #[cfg(all(not(feature = "no-viz"), feature = "map", feature = "r2r"))]
    // let map_publisher = node.create_publisher::<PointCloud2>("map", QosProfile::default())?;
    // #[cfg(feature = "mt_pubsub")]
    // let map_publisher = rt.block_on(async {
    //     node.create_publisher::<PointCloud2>("map".to_owned())
    //         .await
    //         .unwrap()
    // });
    #[cfg(feature = "r2r")]
    let cloud_publisher =
        node.create_publisher::<PointCloud2>("registered", config.registered_cloud_qos.into())?;
    #[cfg(feature = "mt_pubsub")]
    let cloud_publisher = rt
        .as_ref()
        .expect("rt always exists with mt_pubsub feat")
        .block_on(async {
            node.create_publisher::<PointCloud2>("registered".to_owned(), mt_pubsub::Qos::Reliable)
                .await
                .unwrap()
        });
    #[cfg(feature = "r2r")]
    let odom_publisher = node.create_publisher::<Odometry>("odom", config.odom_qos.into())?;
    #[cfg(feature = "mt_pubsub")]
    let odom_publisher = rt
        .as_ref()
        .expect("rt always exists with mt_pubsub feat")
        .block_on(async {
            node.create_publisher::<Odometry>("odom".to_owned(), mt_pubsub::Qos::Reliable)
                .await
                .unwrap()
        });
    #[cfg(all(not(feature = "no-viz"), feature = "r2r"))]
    let lidar_path_publisher = node.create_publisher::<Path>("path", config.path_qos.into())?;
    #[cfg(all(not(feature = "no-viz"), feature = "mt_pubsub"))]
    let lidar_path_publisher = rt
        .as_ref()
        .expect("rt always exists with mt_pubsub feat")
        .block_on(async {
            node.create_publisher::<Path>("path".to_owned(), mt_pubsub::Qos::Reliable)
                .await
                .unwrap()
        });
    #[cfg(all(not(feature = "no-viz"), feature = "r2r"))]
    let imu_odom_publisher =
        node.create_publisher::<Odometry>("imu_odom", config.imu_odom_qos.into())?;
    #[cfg(all(not(feature = "no-viz"), feature = "mt_pubsub"))]
    let imu_odom_publisher = rt
        .as_ref()
        .expect("rt always exists with mt_pubsub feat")
        .block_on(async {
            node.create_publisher::<Odometry>("imu_odom".to_owned(), mt_pubsub::Qos::Reliable)
                .await
                .unwrap()
        });
    #[cfg(feature = "mt_pubsub")]
    let stats_publisher = rt
        .as_ref()
        .expect("rt always exists with mt_pubsub feat")
        .block_on(async {
            node.create_publisher::<RosString>("pelorus_stats".to_owned(), mt_pubsub::Qos::Reliable)
                .await
                .unwrap()
        });
    // #[cfg(all(not(feature = "no-viz"), feature = "r2r"))]
    // let imu_path_publisher =
    //     node.create_publisher::<Path>("imu_path", config.imu_path_qos.into())?;
    // #[cfg(all(not(feature = "no-viz"), feature = "mt_pubsub"))]
    // let imu_path_publisher = rt
    //     .as_ref()
    //     .expect("rt always exists with mt_pubsub feat")
    //     .block_on(async {
    //         node.create_publisher::<Path>("imu_path".to_owned())
    //             .await
    //             .unwrap()
    //     });
    // #[cfg(all(
    //     not(feature = "no-viz"),
    //     feature = "map",
    //     feature = "r2r",
    //     not(feature = "rerun")
    // ))]
    // let registered_visual_publisher =
    // node.create_publisher::<Marker>("registered_viz", r2r::QosProfile::default())?;

    let mut t_world = nalgebra::Isometry3::identity();
    let mut t_last_current = nalgebra::Isometry3::identity();
    #[cfg(feature = "mt_pubsub")]
    let mut last_stats_pub = std::time::Instant::now();

    // Use VecDeque for efficient removal from front (O(1) instead of O(n))
    let mut lidar_path_poses: VecDeque<PoseStamped> = VecDeque::new();
    let lidar_path_frame_id = config.prelio_params.map_frame.clone();
    let mut last_mcap_path_write_time: Option<u64> = None;

    // let mut imu_path_poses: VecDeque<PoseStamped> = VecDeque::new();
    // let imu_path_frame_id = config.prelio_params.map_frame.clone();

    let mut last_scan_time = None;

    #[cfg(not(feature = "no-viz"))]
    let path_update_translate_threshold = 0.05; // default 0.15 meters - TODO config
    #[cfg(not(feature = "no-viz"))]
    let path_update_rotate_threshold = 0.1; // radians (~5.7 degrees) - TODO config

    let mut eval = RuntimeEvaluator::new();

    let debug_render_interval = if config.debug_fps == 0 {
        None
    } else {
        Some(Duration::from_secs_f64(1.0 / config.debug_fps as f64))
    };
    let mut last_render = std::time::Instant::now();

    // Clone shutdown_flag for the processing thread; original is used for the spinner
    let processing_shutdown_flag = shutdown_flag.clone();

    let export_trajectory_for_worker = export_trajectory.clone();
    let processing_handle = std::thread::spawn(move || -> anyhow::Result<()> {
        let mut filtered_imu_rx = filtered_imu_rx;
        let mut mcap_writer = match output_mcap {
            Some(output) => Some(OfflineMcapWriter::create(
                &output.path,
                &output.topics,
                output.rosbag_metadata_path,
            )?),
            None => None,
        };
        // Initialize Prelio prediction
        let mut ready_for_next_tx = Some(ready_for_next_tx);
        let mut registration_ready_tx = None;
        let mut ready_relay_handle = None;
        if minot_replay_mode {
            let (relay_tx, relay_rx) = std::sync::mpsc::channel();
            registration_ready_tx = Some(relay_tx);
            let prelio_ready_tx = ready_for_next_tx
                .as_ref()
                .expect("prelio ready sender must exist")
                .clone();
            let relay_shutdown_flag = processing_shutdown_flag.clone();
            let relay_drain_on_shutdown = Arc::clone(&drain_on_shutdown);
            let prefetch_scans = offline_prefetch_scans.max(1);
            ready_relay_handle = Some(std::thread::spawn(move || {
                let mut requested_prefill = false;
                while relay_rx.recv().is_ok() {
                    if prelio_ready_tx.send(()).is_err() {
                        break;
                    }
                    if relay_drain_on_shutdown.load(Ordering::SeqCst)
                        || relay_shutdown_flag
                            .as_ref()
                            .map(|flag| flag.load(Ordering::Relaxed))
                            .unwrap_or(false)
                    {
                        continue;
                    }
                    let request_count = if requested_prefill {
                        1
                    } else {
                        requested_prefill = true;
                        prefetch_scans
                    };
                    release_minot_ticks(
                        relay_shutdown_flag.as_ref(),
                        synchronous_mode,
                        request_count,
                    );
                }
            }));
        }
        if let Some(ready) = ready_for_next_tx.as_ref() {
            if let Err(e) = ready.send(()) {
                error!("failed to send initial prelio ready signal: {e}");
            }
        }
        let mut lifis_node = Some(lifis_node);

        let active_backend = ActiveBackend::detect(&config);
        if let Some(ref stats) = shared_stats {
            set_backend_from_runtime(stats, &config, &active_backend);
        }

        // TODO avoid casting f32, maybe res needs f64?
        match &active_backend {
            ActiveBackend::Cpu => {}
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(stream) => pelorus_cuda::init_map(
                stream,
                config.map_capacity,
                config.map_resolution as f32,
                config.dist_update_sensitivity,
                config.neighbor_max_range as f32,
                pelorus_cuda::NeighborType::Neighbor26,
            ),
            #[cfg(feature = "metal")]
            ActiveBackend::Metal(stream) => pelorus_metal::init_map(
                stream,
                config.map_capacity,
                config.map_resolution as f32,
                config.dist_update_sensitivity,
                config.neighbor_max_range as f32,
                pelorus_metal::NeighborType::Neighbor26,
            ),
            #[cfg(feature = "vulkan")]
            ActiveBackend::Vulkan(stream) => pelorus_vulkan::init_map(
                stream,
                config.map_capacity,
                config.map_resolution as f32,
                config.dist_update_sensitivity,
                config.neighbor_max_range as f32,
                pelorus_vulkan::NeighborType::Neighbor26,
            ),
        }

        let mut trajectory_file = match export_trajectory_for_worker {
            Some(path_str) => {
                log::info!("Saving trajectory to {}", &path_str);
                match std::fs::File::create(&path_str) {
                    Ok(f) => Some(f),
                    Err(e) => {
                        log::error!("Failed to create trajectory file {}: {}", &path_str, e);
                        None
                    }
                }
            }
            None => {
                // No export
                None
            }
        };

        if let Some(signal) = ready_signal {
            let _ = signal.send(());
        }

        let mut static_tf_published = false;

        #[cfg(feature = "cuda")]
        let mut persistent_cuda_cloud_bufs: Option<
            [LocatedBufferView<pelorus_core::PointXYZIT>; 2],
        > = if let ActiveBackend::Cuda(stream) = &active_backend {
            Some([
                LocatedBufferView::alloc_points_cuda(stream, 100_000),
                LocatedBufferView::alloc_points_cuda(stream, 100_000),
            ])
        } else {
            None
        };
        #[cfg(feature = "cuda")]
        let mut persistent_cuda_cloud_buf_idx = 0usize;

        // Reusable buffer for the point cloud on GPU
        #[allow(unused_mut)]
        let mut persistent_cloud_buf = match &active_backend {
            ActiveBackend::Cpu => LocatedBufferView::Ram(Vec::new()),
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(_) => LocatedBufferView::Ram(Vec::new()),
            #[cfg(feature = "metal")]
            ActiveBackend::Metal(stream) => LocatedBufferView::Metal {
                buffer: pelorus_metal::MetalBuffer::new(
                    stream,
                    size_of::<pelorus_core::PointXYZIT>(),
                    100_000,
                ),
            },
            #[cfg(feature = "vulkan")]
            ActiveBackend::Vulkan(stream) => LocatedBufferView::Vulkan {
                buffer: pelorus_vulkan::VulkanBuffer::new(
                    stream,
                    size_of::<pelorus_core::PointXYZIT>(),
                    100_000,
                ),
            },
        };

        // Main processing loop. Use recv_timeout when a shutdown flag is provided so we can
        // check for shutdown requests periodically. If no shutdown flag is provided, block
        // on recv() as before.
        let mut draining_after_end_of_bag = false;
        loop {
            let lifis_tick = match processing_shutdown_flag.as_ref() {
                Some(flag) => {
                    let shutdown_requested = flag.load(std::sync::atomic::Ordering::Relaxed);
                    let drain_requested =
                        drain_on_shutdown.load(std::sync::atomic::Ordering::SeqCst);
                    if shutdown_requested || drain_requested {
                        if drain_requested {
                            if !draining_after_end_of_bag {
                                draining_after_end_of_bag = true;
                                info!(
                                    "Shutdown requested after end-of-bag hint; draining queued scans"
                                );
                            }
                            match synced_rx.recv_timeout(Duration::from_millis(50)) {
                                Ok(t) => t,
                                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                                | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                                    if let Some(ref lifis_node) = lifis_node {
                                        lifis_node.request_shutdown();
                                    }
                                    // Drop sync gate and node ownership so all producer chains can close.
                                    drop(registration_ready_tx.take());
                                    drop(ready_for_next_tx.take());
                                    drop(lifis_node.take());
                                    info!(
                                        "Shutdown requested after end-of-bag hint, queue drained, exiting processing loop"
                                    );
                                    break;
                                }
                            }
                        } else {
                            if let Some(ref lifis_node) = lifis_node {
                                lifis_node.request_shutdown();
                            }
                            // Drop sync gate and node ownership so all producer chains can close.
                            drop(registration_ready_tx.take());
                            drop(ready_for_next_tx.take());
                            drop(lifis_node.take());
                            info!("Shutdown requested, exiting processing loop");
                            break;
                        }
                    } else {
                        // Block with a short timeout so we can still respond promptly to shutdown signals.
                        match synced_rx.recv_timeout(Duration::from_millis(50)) {
                            Ok(t) => t,
                            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                                info!("synced_rx disconnected, exiting processing loop");
                                break;
                            }
                        }
                    }
                }
                None => match synced_rx.recv() {
                    Ok(t) => t,
                    Err(_) => break,
                },
            };
            // Record scan arrival for measuring actual sensor rate
            if let Some(ref stats) = shared_stats {
                stats.record_scan();
            }

            eval.merge(&lifis_tick.runtime_eval);
            drain_filtered_imu_mcap(&mut mcap_writer, &mut filtered_imu_rx)?;

            match &active_backend {
                ActiveBackend::Cpu => {}
                #[cfg(feature = "cuda")]
                ActiveBackend::Cuda(stream) => {
                    let cuda_cloud_bufs = persistent_cuda_cloud_bufs
                        .as_mut()
                        .expect("missing persistent CUDA cloud buffers");
                    let current_slot = persistent_cuda_cloud_buf_idx;
                    persistent_cuda_cloud_buf_idx =
                        (persistent_cuda_cloud_buf_idx + 1) % cuda_cloud_bufs.len();

                    // Uploads and kernels are enqueued on the same stream, so ping-pong slots
                    // can be reused without forcing a host sync here.
                    let cuda_cloud = &mut cuda_cloud_bufs[current_slot];
                    cuda_cloud.realloc_if_needed_cuda(stream, lifis_tick.deskewed_cloud.len());
                    cuda_cloud.upload_from_slice_cuda(stream, lifis_tick.deskewed_cloud.as_slice());
                }
                #[cfg(feature = "metal")]
                ActiveBackend::Metal(stream) => {
                    // Metal uses shared memory; sync before overwriting to avoid races.
                    stream.sync();
                    persistent_cloud_buf
                        .realloc_if_needed_metal(stream, lifis_tick.deskewed_cloud.len());
                    persistent_cloud_buf
                        .upload_from_slice_metal(stream, lifis_tick.deskewed_cloud.as_slice());
                }
                #[cfg(feature = "vulkan")]
                ActiveBackend::Vulkan(stream) => {
                    stream.sync();
                    persistent_cloud_buf
                        .realloc_if_needed_vulkan(stream, lifis_tick.deskewed_cloud.len());
                    persistent_cloud_buf
                        .upload_from_slice_vulkan(stream, lifis_tick.deskewed_cloud.as_slice());
                }
            }

            let ram_cloud;
            #[allow(unreachable_patterns)]
            let cloud_for_reg: &LocatedBufferView<pelorus_core::PointXYZIT> = match &active_backend
            {
                ActiveBackend::Cpu => {
                    ram_cloud = LocatedBufferView::Ram(lifis_tick.deskewed_cloud);
                    &ram_cloud
                }
                #[cfg(feature = "cuda")]
                ActiveBackend::Cuda(_) => {
                    let cuda_cloud_bufs = persistent_cuda_cloud_bufs
                        .as_ref()
                        .expect("missing persistent CUDA cloud buffers");
                    let last_slot = (persistent_cuda_cloud_buf_idx + cuda_cloud_bufs.len() - 1)
                        % cuda_cloud_bufs.len();
                    &cuda_cloud_bufs[last_slot]
                }
                _ => &persistent_cloud_buf,
            };

            let cloud_out = lifis_tick.deskewed_cloud_passthrough;

            let orig_frame = lifis_tick.cloud_header.frame_id.clone();
            let orig_time = lifis_tick.cloud_header.stamp;
            let current_cloud_time = timestamp_to_secs(&lifis_tick.odom_time);
            if let Some(existing_last_scan_time) = last_scan_time {
                if current_cloud_time < existing_last_scan_time {
                    let mut map = map.lock().unwrap();
                    map.clear_localmap(&active_backend);

                    info!("Resetting map");
                    match &active_backend {
                        ActiveBackend::Cpu => {}
                        #[cfg(feature = "cuda")]
                        ActiveBackend::Cuda(stream) => pelorus_cuda::init_map(
                            stream,
                            config.map_capacity,
                            config.map_resolution as f32,
                            config.dist_update_sensitivity,
                            config.neighbor_max_range as f32,
                            pelorus_cuda::NeighborType::Neighbor26,
                        ),
                        #[cfg(feature = "metal")]
                        ActiveBackend::Metal(stream) => pelorus_metal::init_map(
                            stream,
                            config.map_capacity,
                            config.map_resolution as f32,
                            config.dist_update_sensitivity,
                            config.neighbor_max_range as f32,
                            pelorus_metal::NeighborType::Neighbor26,
                        ),
                        #[cfg(feature = "vulkan")]
                        ActiveBackend::Vulkan(stream) => pelorus_vulkan::init_map(
                            stream,
                            config.map_capacity,
                            config.map_resolution as f32,
                            config.dist_update_sensitivity,
                            config.neighbor_max_range as f32,
                            pelorus_vulkan::NeighborType::Neighbor26,
                        ),
                    };
                    t_world = nalgebra::Isometry3::identity();
                    t_last_current = nalgebra::Isometry3::identity();
                }
            }

            last_scan_time = Some(timestamp_to_secs(&orig_time));

            // use the last difference if there is no odom from lifis
            let init_guess_lidar = if lifis_tick.imu_odom.pose.pose.position.x == 0.0
                && lifis_tick.imu_odom.pose.pose.position.y == 0.0
                && lifis_tick.imu_odom.pose.pose.position.z == 0.0
            {
                // lifis did not return a orientation (for example when not using IMU at all), use the last difference as guess
                if lifis_tick.imu_odom.pose.pose.orientation.w == 1.0
                    && lifis_tick.imu_odom.pose.pose.orientation.x == 0.0
                    && lifis_tick.imu_odom.pose.pose.orientation.y == 0.0
                    && lifis_tick.imu_odom.pose.pose.orientation.z == 0.0
                {
                    t_world * t_last_current
                } else {
                    // there is an orientation calculated by lifis for undeskewing the cloud, so we can use that
                    let initial_guess_rotation =
                        nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                            lifis_tick.imu_odom.pose.pose.orientation.w,
                            lifis_tick.imu_odom.pose.pose.orientation.x,
                            lifis_tick.imu_odom.pose.pose.orientation.y,
                            lifis_tick.imu_odom.pose.pose.orientation.z,
                        ));
                    let initial_guess_lidar_local = nalgebra::Isometry3::from_parts(
                        t_last_current.translation,
                        initial_guess_rotation,
                    );
                    t_world * initial_guess_lidar_local
                }
            } else {
                let initial_guess_translation = Translation3::new(
                    lifis_tick.imu_odom.pose.pose.position.x,
                    lifis_tick.imu_odom.pose.pose.position.y,
                    lifis_tick.imu_odom.pose.pose.position.z,
                );

                let initial_guess_rotation =
                    nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                        lifis_tick.imu_odom.pose.pose.orientation.w,
                        lifis_tick.imu_odom.pose.pose.orientation.x,
                        lifis_tick.imu_odom.pose.pose.orientation.y,
                        lifis_tick.imu_odom.pose.pose.orientation.z,
                    ));

                // init_guess from lifis is IMU pose T_W_I. Convert to LiDAR pose T_W_L.
                nalgebra::Isometry3::from_parts(initial_guess_translation, initial_guess_rotation)
                    * ext
            };

            let registration_result = {
                let mut map = map.lock().unwrap();
                let ready_for_registration = if minot_replay_mode {
                    None
                } else {
                    ready_for_next_tx.as_ref()
                };
                match map.register_and_map(
                    lifis_tick.map_only,
                    init_guess_lidar,
                    cloud_for_reg,
                    current_cloud_time,
                    lifis_tick.scan_id as usize,
                    ready_for_registration,
                    &mut eval,
                    &active_backend,
                ) {
                    Ok(Some(pose)) => Ok(Some((pose, map.get_state()))),
                    Ok(None) => Ok(None),
                    Err(err) => Err(err),
                }
            };

            let (register_pose, ekf_state) = match registration_result {
                Ok(Some(result)) => result,
                Ok(None) => {
                    drain_filtered_imu_mcap(&mut mcap_writer, &mut filtered_imu_rx)?;
                    if minot_replay_mode {
                        if let Some(ready) = registration_ready_tx.as_ref() {
                            let _ = ready.send(());
                        }
                    } else {
                        release_minot_tick(processing_shutdown_flag.as_ref(), synchronous_mode);
                    }
                    continue;
                } // skip outputting trivial solutions on initialization
                Err(err) => {
                    error!("{err}");
                    drain_filtered_imu_mcap(&mut mcap_writer, &mut filtered_imu_rx)?;
                    if minot_replay_mode {
                        if let Some(ready) = registration_ready_tx.as_ref() {
                            let _ = ready.send(());
                        }
                    } else {
                        release_minot_tick(processing_shutdown_flag.as_ref(), synchronous_mode);
                    }
                    continue;
                }
            };

            // --- Adapting Coordinate Systems ---
            #[cfg(feature = "rerun")]
            if let Some(ref rec) = rec {
                rec.set_timestamp_secs_since_epoch("sensor", current_cloud_time);
            }

            #[cfg(feature = "rerun")]
            let ros_to_rerun = Isometry3::<f64>::identity(); // placeholder for other visualizations if needed

            let old_t_world = t_world;

            let child_frame = config
                .base_link_frame
                .as_deref()
                .unwrap_or(&orig_frame)
                .to_string();

            if !static_tf_published && config.base_link_frame.is_some() {
                if config.base_to_lidar_tf_override {
                    if let Some(ref lifis_node) = lifis_node {
                        match lifis_node.tf_listener.lookup_transform(
                            &child_frame,
                            &orig_frame,
                            Time {
                                sec: orig_time.sec,
                                nanosec: orig_time.nanosec,
                            },
                        ) {
                            Ok(tf) => {
                                #[cfg(feature = "r2r")]
                                let iso = prelio_io::tf_r2r::isometry_from_transform(&tf.transform);
                                #[cfg(feature = "ros2-client")]
                                let iso = prelio_io::tf::isometry_from_transform(&tf.transform);
                                #[cfg(feature = "mt_pubsub")]
                                let iso = prelio_io::tf_lh::isometry_from_transform(&tf.transform);

                                info!(
                                    "Updated base_link to lidar transform from TF: translation={:?}, rotation={:?}",
                                    iso.translation.vector, iso.rotation
                                );
                                // Update the local config and the registration's internal transform
                                pipeline.set_base_to_lidar(iso);
                            }
                            Err(e) => {
                                error!(
                                    "Failed to lookup base_link to lidar transform from TF: {}. Using manual config.",
                                    e
                                );
                            }
                        }
                    }
                }
                static_tf_published = true;
            }

            t_world = register_pose.base_pose;

            // transformation from the last pose to the current pose in the local frame
            t_last_current = old_t_world.inverse() * t_world;

            let mut pelorus_odom = Odometry::default();
            pelorus_odom.header.stamp = Time {
                sec: lifis_tick.odom_time.sec,
                nanosec: lifis_tick.odom_time.nanosec,
            };
            pelorus_odom.header.frame_id = config.prelio_params.map_frame.clone();
            pelorus_odom.child_frame_id = child_frame.clone();
            pelorus_odom.pose.pose.position.x = t_world.translation.vector.x;
            pelorus_odom.pose.pose.position.y = t_world.translation.vector.y;
            pelorus_odom.pose.pose.position.z = t_world.translation.vector.z;
            pelorus_odom.pose.pose.orientation.x = t_world.rotation.i;
            pelorus_odom.pose.pose.orientation.y = t_world.rotation.j;
            pelorus_odom.pose.pose.orientation.z = t_world.rotation.k;
            pelorus_odom.pose.pose.orientation.w = t_world.rotation.w;

            let lin_vel_body = t_world.rotation.inverse() * ekf_state.linear_velocity;
            pelorus_odom.twist.twist.linear.x = lin_vel_body.x;
            pelorus_odom.twist.twist.linear.y = lin_vel_body.y;
            pelorus_odom.twist.twist.linear.z = lin_vel_body.z;

            let r_b_l = config.base_to_lidar.rotation.inverse();
            let r_l_i = ekf_state.lidar_imu_extrinsic.rotation.inverse();
            let ang_vel_body = (r_b_l * r_l_i) * ekf_state.angular_velocity;
            pelorus_odom.twist.twist.angular.x = ang_vel_body.x;
            pelorus_odom.twist.twist.angular.y = ang_vel_body.y;
            pelorus_odom.twist.twist.angular.z = ang_vel_body.z;

            {
                // TODO covariances probably not in the right frame right now and ignores extrinsic rotation
                let (pose_cov_iekf, twist_cov_iekf) = {
                    let map = pipeline.map();
                    let guard = map.lock().unwrap();
                    let (pose_cov, _lv_cov_unused) = guard.get_odometry_covariances();
                    let tw_cov = guard.get_twist_covariance_6x6();
                    (pose_cov, tw_cov)
                };

                #[cfg(not(feature = "r2r"))]
                {
                    pelorus_odom.pose.covariance = pose_cov_iekf;
                    pelorus_odom.twist.covariance = twist_cov_iekf;
                }

                #[cfg(feature = "r2r")]
                {
                    pelorus_odom.pose.covariance = pose_cov_iekf.to_vec();
                    pelorus_odom.twist.covariance = twist_cov_iekf.to_vec();
                }
            }

            // publish paths but only add new poses if the robot moved enough
            #[cfg(not(feature = "no-viz"))]
            let changed_lidar_path = {
                match lidar_path_poses.back() {
                    Some(last_pose) => {
                        let last_pose = last_pose.pose.clone();
                        let last_pose_position = last_pose.position;
                        let current_pose_position = pelorus_odom.pose.pose.position.clone();

                        let distance = nalgebra::distance(
                            &nalgebra::Point3::new(
                                last_pose_position.x,
                                last_pose_position.y,
                                last_pose_position.z,
                            ),
                            &nalgebra::Point3::new(
                                current_pose_position.x,
                                current_pose_position.y,
                                current_pose_position.z,
                            ),
                        );

                        // Check rotational difference
                        let last_quat = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                            last_pose.orientation.w,
                            last_pose.orientation.x,
                            last_pose.orientation.y,
                            last_pose.orientation.z,
                        ));
                        let current_quat =
                            UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                                pelorus_odom.pose.pose.orientation.w,
                                pelorus_odom.pose.pose.orientation.x,
                                pelorus_odom.pose.pose.orientation.y,
                                pelorus_odom.pose.pose.orientation.z,
                            ));
                        let rotation_angle = last_quat.angle_to(&current_quat);

                        if distance > path_update_translate_threshold
                            || rotation_angle > path_update_rotate_threshold
                        {
                            lidar_path_poses.push_back(PoseStamped {
                                pose: pelorus_odom.pose.pose.clone(),
                                header: pelorus_odom.header.clone(),
                            });
                            if lidar_path_poses.len() > PATH_LIMIT {
                                lidar_path_poses.pop_front();
                            }
                            true
                        } else {
                            false
                        }
                    }
                    None => {
                        lidar_path_poses.push_back(PoseStamped {
                            pose: pelorus_odom.pose.pose.clone(),
                            header: pelorus_odom.header.clone(),
                        });
                        if lidar_path_poses.len() > PATH_LIMIT {
                            lidar_path_poses.pop_front();
                        }
                        true
                    }
                }
            };

            let mut cloud: PointCloud2 = {
                match (cloud_out, cloud_for_reg) {
                    (None, LocatedBufferView::Ram(items)) => {
                        let msg = PointCloud2Msg::try_from_vec(items.to_vec()).expect("msg");
                        #[cfg(feature = "ros2-client")]
                        let out =
                            prelio_io::impl_ros2_interfaces_jazzy_serde::from_pointcloud2_msg(msg);
                        #[cfg(feature = "mt_pubsub")]
                        let out = impl_ros2_interfaces_jazzy_rkyv::from_pointcloud2_msg(msg);
                        #[cfg(feature = "r2r")]
                        let out = impl_r2r::from_pointcloud2_msg(msg);

                        out
                    }
                    // ignore downsampled one if we have the raw
                    (Some(items), _) => {
                        let msg = PointCloud2Msg::try_from_vec(items.to_vec()).expect("msg");
                        #[cfg(feature = "ros2-client")]
                        let out =
                            prelio_io::impl_ros2_interfaces_jazzy_serde::from_pointcloud2_msg(msg);
                        #[cfg(feature = "mt_pubsub")]
                        let out = impl_ros2_interfaces_jazzy_rkyv::from_pointcloud2_msg(msg);
                        #[cfg(feature = "r2r")]
                        let out = impl_r2r::from_pointcloud2_msg(msg);

                        out
                    }
                    #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
                    (None, gpu_cloud) => {
                        let buff = match (gpu_cloud, &active_backend) {
                            #[cfg(feature = "cuda")]
                            (LocatedBufferView::Cuda { buffer }, ActiveBackend::Cuda(stream)) => {
                                buffer.into_cloud(stream)
                            }
                            #[cfg(feature = "metal")]
                            (LocatedBufferView::Metal { buffer }, ActiveBackend::Metal(stream)) => {
                                buffer.into_cloud(stream)
                            }
                            #[cfg(feature = "vulkan")]
                            (
                                LocatedBufferView::Vulkan { buffer },
                                ActiveBackend::Vulkan(stream),
                            ) => buffer.into_cloud(stream),
                            _ => unreachable!("buffer/backend mismatch"),
                        };
                        let msg = ros_pointcloud2::PointCloud2Msg::try_from_vec(buff).unwrap();
                        #[cfg(feature = "ros2-client")]
                        let out =
                            prelio_io::impl_ros2_interfaces_jazzy_serde::from_pointcloud2_msg(msg);
                        #[cfg(feature = "mt_pubsub")]
                        let out = impl_ros2_interfaces_jazzy_rkyv::from_pointcloud2_msg(msg);
                        #[cfg(feature = "r2r")]
                        let out = impl_r2r::from_pointcloud2_msg(msg);

                        out
                    }
                }
            };

            cloud.header.frame_id = orig_frame.clone();
            cloud.header.stamp = Time {
                sec: lifis_tick.odom_time.sec,
                nanosec: lifis_tick.odom_time.nanosec,
            };
            if let Some(mut trajectory_file) = trajectory_file.as_mut() {
                let t = &t_world.translation.vector;
                let q = t_world.rotation.quaternion();

                // <time in seconds> <translation_x> <translation_y> <translation_z> <quaternion_x> <quaternion_y> <quaternion_z> <quaternion_w>
                // Format the string to write to the file
                let content_to_write = format!(
                    "{} {} {} {} {} {} {} {}\n",
                    current_cloud_time, t.x, t.y, t.z, q.i, q.j, q.k, q.w
                );

                std::io::Write::write_all(&mut trajectory_file, content_to_write.as_bytes())
                    .unwrap();
            }

            #[cfg(all(feature = "map", all(feature = "rerun", feature = "mt_pubsub")))]
            {
                eval.start("pub_localmap").unwrap();
                #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
                let total_map = map.lock().unwrap().get_entire_map();
                #[cfg(feature = "cuda")]
                let total_map = pelorus_cuda::get_map();
                #[cfg(feature = "metal")]
                let total_map = pelorus_metal::get_map();
                #[cfg(feature = "vulkan")]
                let total_map = pelorus_vulkan::get_map();
                #[cfg(feature = "rerun")]
                if let Some(ref rec) = rec {
                    let points = total_map.iter().map(|v| {
                        let v = ros_to_rerun.cast().transform_point(v);
                        glam::Vec3::new(v.x as f32, v.y as f32, v.z as f32)
                    });

                    if let Err(err) = rec.log(
                        "world/map",
                        &rerun::Points3D::update_fields().with_positions(points),
                    ) {
                        error!("Error sending to rerun {err}");
                    }
                }
                #[cfg(all(
                    not(feature = "no-viz"),
                    not(feature = "rerun"),
                    not(feature = "mt_pubsub") // currently no use of the total map in mt_pubsub
                ))]
                {
                    let mut total_map = PointCloud2Msg::try_from_vec(
                        total_map
                            .iter()
                            .map(|p| PointXYZ::new(p.x, p.y, p.z))
                            .collect(),
                    )
                    .unwrap();
                    total_map.header.frame_id = config.prelio_params.map_frame.clone();
                    let msg: PointCloud2 = total_map.into();
                    map_publisher.publish(&msg).expect("Failed to publish map");
                }
                eval.stop("pub_localmap").unwrap();
            }

            // -- TF --
            let tf = TransformStamped {
                header: Header {
                    frame_id: config.prelio_params.map_frame.clone(),
                    stamp: Time {
                        sec: lifis_tick.odom_time.sec,
                        nanosec: lifis_tick.odom_time.nanosec,
                    },
                },
                child_frame_id: child_frame,
                transform: Transform {
                    translation: Vector3 {
                        x: t_world.translation.vector.x as f64,
                        y: t_world.translation.vector.y as f64,
                        z: t_world.translation.vector.z as f64,
                    },
                    rotation: Quaternion {
                        x: t_world.rotation.quaternion().i as f64,
                        y: t_world.rotation.quaternion().j as f64,
                        z: t_world.rotation.quaternion().k as f64,
                        w: t_world.rotation.quaternion().w as f64,
                    },
                },
            };
            let tf_for_mcap = tf.clone();
            #[cfg(feature = "mt_pubsub")]
            {
                let inner_tf_broad = Arc::clone(&tf_broadcaster);
                std::thread::spawn(move || {
                    future::block_on(async move {
                        if let Err(err) = inner_tf_broad.send_transform(tf).await {
                            error!("Failed to send transform: {err}");
                        }
                    });
                });
            }
            #[cfg(not(feature = "mt_pubsub"))]
            tf_broadcaster
                .send_transform(tf)
                .expect("Failed to send transform");
            // -- --

            if let Some(writer) = mcap_writer.as_mut() {
                writer.write_cloud(&serde_cloud(&cloud))?;
                writer.write_odom(&serde_odom(&pelorus_odom))?;
                #[cfg(not(feature = "no-viz"))]
                if changed_lidar_path {
                    let path_time = (pelorus_odom.header.stamp.sec.max(0) as u64)
                        .saturating_mul(1_000_000_000)
                        .saturating_add(pelorus_odom.header.stamp.nanosec as u64);
                    let should_write_path = last_mcap_path_write_time
                        .map(|last| path_time.saturating_sub(last) >= MCAP_PATH_MIN_PERIOD_NS)
                        .unwrap_or(true);
                    if should_write_path {
                        last_mcap_path_write_time = Some(path_time);
                        let skip = lidar_path_poses.len().saturating_sub(MCAP_PATH_LIMIT);
                        let lidar_path = Path {
                            header: Header {
                                frame_id: lidar_path_frame_id.clone(),
                                stamp: pelorus_odom.header.stamp.clone(),
                            },
                            poses: lidar_path_poses.iter().skip(skip).cloned().collect(),
                        };
                        writer.write_path(&serde_path(&lidar_path))?;
                    }
                }
                writer.write_tf(&serde_tf_message(&tf_for_mcap))?;
            }
            drain_filtered_imu_mcap(&mut mcap_writer, &mut filtered_imu_rx)?;
            if minot_replay_mode {
                if let Some(ready) = registration_ready_tx.as_ref() {
                    let _ = ready.send(());
                }
            }

            #[cfg(feature = "rerun")]
            if let Some(ref rec) = rec {
                let t_world_rerun = ros_to_rerun.cast() * t_world;
                let pose = rerun::Transform3D::from_translation(glam::Vec3::new(
                    t_world_rerun.translation.x as f32,
                    t_world_rerun.translation.y as f32,
                    t_world_rerun.translation.z as f32,
                ))
                .with_quaternion(glam::Quat::from_xyzw(
                    t_world_rerun.rotation.quaternion().i as f32,
                    t_world_rerun.rotation.quaternion().j as f32,
                    t_world_rerun.rotation.quaternion().k as f32,
                    t_world_rerun.rotation.quaternion().w as f32,
                ));

                if let Err(err) = rec.log("world/lidar", &pose) {
                    error!("Error sending to rerun {err}");
                }
            }

            let mut imu_odom = pelorus_odom.clone();
            imu_odom.pose.pose.position.x = init_guess_lidar.translation.vector.x;
            imu_odom.pose.pose.position.y = init_guess_lidar.translation.vector.y;
            imu_odom.pose.pose.position.z = init_guess_lidar.translation.vector.z;
            imu_odom.pose.pose.orientation.x = init_guess_lidar.rotation.quaternion().i;
            imu_odom.pose.pose.orientation.y = init_guess_lidar.rotation.quaternion().j;
            imu_odom.pose.pose.orientation.z = init_guess_lidar.rotation.quaternion().k;
            imu_odom.pose.pose.orientation.w = init_guess_lidar.rotation.quaternion().w;
            imu_odom.pose.covariance = pelorus_odom.pose.covariance.clone();
            imu_odom.twist.covariance = pelorus_odom.twist.covariance.clone();

            // #[cfg(not(feature = "no-viz"))]
            // imu_path_poses.push_back(PoseStamped {
            //     pose: imu_odom.pose.pose.clone(),
            //     header: imu_odom.header.clone(),
            // });
            // #[cfg(not(feature = "no-viz"))]
            // if imu_path_poses.len() > PATH_LIMIT {
            //     imu_path_poses.pop_front();
            // }

            #[cfg(feature = "rerun")]
            if let Some(ref rec) = rec {
                #[cfg(feature = "ros2-client")]
                let ps = impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(cloud.clone());
                #[cfg(feature = "mt_pubsub")]
                let ps = impl_ros2_interfaces_jazzy_rkyv::to_pointcloud2_msg(cloud.clone());
                #[cfg(feature = "r2r")]
                let ps = impl_r2r::to_pointcloud2_msg(cloud.clone());

                let gradient = colorous::TURBO;
                let (positions, colors): (Vec<_>, Vec<_>) = ps
                    .try_into_iter()
                    .unwrap()
                    .map(|v: PointXYZI| {
                        let color = gradient.eval_continuous(v.intensity as f64 / 255.0);
                        let v = if config.rerun_undistorted_static_world {
                            register_pose.pose.cast() * nalgebra::Point3::new(v.x, v.y, v.z)
                        } else {
                            nalgebra::Point3::new(v.x, v.y, v.z)
                        };
                        let pos = glam::Vec3::new(v.x as f32, v.y as f32, v.z as f32);
                        let color_rgb = [color.r, color.g, color.b];

                        (pos, color_rgb)
                    })
                    .unzip();

                let topic = if config.rerun_undistorted_static_world {
                    format!("undistorted_{:?}", current_cloud_time)
                } else {
                    "world/lidar/undistorted".to_string()
                };

                if let Err(err) = rec.log(
                    topic,
                    &rerun::Points3D::update_fields()
                        .with_positions(positions)
                        .with_colors(colors)
                        .with_radii([rerun::Radius::new_ui_points(1.25)]), // for video scene_points: 0.025
                ) {
                    error!("Error sending to rerun {err}");
                }
            }

            #[cfg(feature = "mt_pubsub")]
            rt.as_ref()
                .expect("rt always exists with mt_pubsub feat")
                .block_on(async {
                    cloud_publisher
                        .publish(&cloud)
                        .await
                        .expect("Failed to publish cloud");

                    odom_publisher
                        .publish(&pelorus_odom)
                        .await
                        .expect("Failed to publish odom");
                });
            #[cfg(not(feature = "mt_pubsub"))]
            {
                cloud_publisher
                    .publish(&cloud)
                    .expect("Failed to publish cloud");

                odom_publisher
                    .publish(&pelorus_odom)
                    .expect("Failed to publish odom");
            }

            #[cfg(not(feature = "no-viz"))]
            if changed_lidar_path {
                #[cfg(all(not(feature = "no-viz"), feature = "rerun"))]
                if let Some(ref rec) = rec {
                    // Visualize the lidar path as a line strip
                    let lidar_path_strip =
                        rerun::LineStrip3D::from_iter(lidar_path_poses.iter().map(|p| {
                            glam::Vec3::new(
                                p.pose.position.x as f32,
                                p.pose.position.y as f32,
                                p.pose.position.z as f32,
                            )
                        }));
                    let lidar_path_strips =
                        rerun::LineStrips3D::new(std::iter::once(lidar_path_strip))
                            .with_colors([rerun::Color::from_rgb(255, 128, 0)]); // Orange color for lidar path

                    if let Err(err) = rec.log("world/lidar_path", &lidar_path_strips) {
                        error!("Error sending lidar path to rerun {err}");
                    }
                }
                #[cfg(all(not(feature = "no-viz"), not(feature = "mt_pubsub")))]
                {
                    let lidar_path = Path {
                        header: Header {
                            frame_id: lidar_path_frame_id.clone(),
                            stamp: pelorus_odom.header.stamp.clone(),
                        },
                        poses: lidar_path_poses.iter().cloned().collect(),
                    };
                    lidar_path_publisher
                        .publish(&lidar_path)
                        .expect("Failed to publish path");
                }
            }

            #[cfg(all(not(feature = "no-viz"), not(feature = "mt_pubsub")))]
            imu_odom_publisher
                .publish(&imu_odom)
                .expect("Failed to publish imu odom");

            // #[cfg(all(not(feature = "no-viz"), not(feature = "mt_pubsub")))]
            // imu_path_publisher
            //     .publish(&imu_path)
            //     .expect("Failed to publish imu path"); // TODO this will spam a lot, maybe also throttle like lidar_path

            #[cfg(all(not(feature = "no-viz"), feature = "mt_pubsub"))]
            let pub_lidar_path = Path {
                header: Header {
                    frame_id: lidar_path_frame_id.clone(),
                    stamp: pelorus_odom.header.stamp.clone(),
                },
                poses: lidar_path_poses.iter().cloned().collect(),
            };
            // #[cfg(all(not(feature = "no-viz"), feature = "mt_pubsub"))]
            // let pub_imu_path = Path {
            //     header: Header {
            //         frame_id: imu_path_frame_id.clone(),
            //         stamp: imu_odom.header.stamp.clone(),
            //     },
            //     poses: imu_path_poses.iter().cloned().collect(),
            // };
            #[cfg(all(not(feature = "no-viz"), feature = "mt_pubsub"))]
            rt.as_ref()
                .expect("rt always exists with mt_pubsub feat")
                .block_on(async {
                    lidar_path_publisher
                        .publish(&pub_lidar_path)
                        .await
                        .expect("Failed to publish path");

                    imu_odom_publisher
                        .publish(&imu_odom)
                        .await
                        .expect("Failed to publish imu odom");

                    // imu_path_publisher
                    //     .publish(&pub_imu_path)
                    //     .await
                    //     .expect("Failed to publish imu path"); // TODO this will spam a lot, maybe also throttle like lidar_path
                });
            // #[cfg(feature = "rerun")]
            // if let Some(ref rec) = rec {
            //     let imu_path_strip =
            //         rerun::LineStrip3D::from_iter(imu_path_poses.iter().map(|p| {
            //             glam::Vec3::new(
            //                 p.pose.position.x as f32,
            //                 p.pose.position.y as f32,
            //                 p.pose.position.z as f32,
            //             )
            //         }));
            //     let imu_path_strips = rerun::LineStrips3D::new(std::iter::once(imu_path_strip));
            //     if let Err(err) = rec.log("imu_path", &imu_path_strips) {
            //         error!("Error sending to rerun {err}");
            //     }
            // }

            let iteration_time_ms = eval.get_sum_of_top_level_times_ms();

            const NAME: &str = "Pelorus";

            // Update shared stats if TUI is active
            if let Some(ref stats) = shared_stats {
                let pos = t_world.translation.vector;
                let q = t_world.rotation.quaternion();
                let lv = ekf_state.linear_velocity;
                let av = ekf_state.angular_velocity;

                stats.update_pose(
                    [pos.x, pos.y, pos.z],
                    [q.w, q.i, q.j, q.k],
                    [lv.x, lv.y, lv.z],
                    [av.x, av.y, av.z],
                );

                // Update bias values
                let ab = ekf_state.accel_bias;
                let gb = ekf_state.gyro_bias;
                stats.update_bias([ab.x, ab.y, ab.z], [gb.x, gb.y, gb.z]);

                // Record computation time for this scan iteration
                stats.record_comp_time(iteration_time_ms);
            }

            #[cfg(feature = "mt_pubsub")]
            if let Some(ref stats) = shared_stats {
                if last_stats_pub.elapsed() >= Duration::from_millis(250) {
                    last_stats_pub = std::time::Instant::now();
                    let snapshot = stats.get();
                    let payload = format!(
                        "elapsed={:.2};lidar_hz={:.2};proc_hz={:.2};rt={:.2};cpu={:.1};ram={:.1};comp_ms={:.2};comp_avg={:.2};pos={:.3},{:.3},{:.3};ori={:.5},{:.5},{:.5},{:.5};vel={:.3},{:.3},{:.3};ang={:.4},{:.4},{:.4}",
                        snapshot.elapsed_secs,
                        snapshot.measured_sensor_rate_hz,
                        snapshot.processing_freq_hz,
                        snapshot.real_time_factor,
                        snapshot.cpu_percent,
                        snapshot.ram_mb,
                        snapshot.last_comp_time_ms,
                        snapshot.avg_comp_time_ms,
                        snapshot.position[0],
                        snapshot.position[1],
                        snapshot.position[2],
                        snapshot.orientation[0],
                        snapshot.orientation[1],
                        snapshot.orientation[2],
                        snapshot.orientation[3],
                        snapshot.linear_velocity[0],
                        snapshot.linear_velocity[1],
                        snapshot.linear_velocity[2],
                        snapshot.angular_velocity[0],
                        snapshot.angular_velocity[1],
                        snapshot.angular_velocity[2],
                    );
                    let msg = RosString { data: payload };
                    let _ = rt
                        .as_ref()
                        .expect("rt always exists with mt_pubsub feat")
                        .block_on(async { stats_publisher.publish(&msg).await });
                }
            }

            // Debug output (non-TUI timing metrics)
            if let Some(debug_interval) = debug_render_interval {
                if last_render.elapsed() > debug_interval {
                    last_render = std::time::Instant::now();
                    // Only print debug output if TUI is not running
                    // TUI runs when show_tui is true AND we're in a terminal
                    let tui_is_running = config.show_tui && std::io::stdout().is_terminal();
                    if !tui_is_running {
                        println!(
                            "{}[2J{}[1;L{}\n{}",
                            27 as char,
                            27 as char,
                            &NAME,
                            eval.render()
                        );
                    }
                }
            }

            if !minot_replay_mode {
                release_minot_tick(processing_shutdown_flag.as_ref(), synchronous_mode);
            }
        }

        drop(registration_ready_tx.take());
        drop(ready_for_next_tx.take());
        if let Some(handle) = ready_relay_handle.take() {
            let _ = handle.join();
        }

        // Ensure backend work is quiesced and map state is released before GPU
        // buffers/contexts in this thread are dropped.
        active_backend.stream_sync();
        match &active_backend {
            ActiveBackend::Cpu => {}
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(_) => pelorus_cuda::cleanup_map(),
            #[cfg(feature = "metal")]
            ActiveBackend::Metal(_) => pelorus_metal::cleanup_map(),
            #[cfg(feature = "vulkan")]
            ActiveBackend::Vulkan(_) => pelorus_vulkan::cleanup_map(),
        }
        drain_filtered_imu_mcap(&mut mcap_writer, &mut filtered_imu_rx)?;
        if let Some(writer) = mcap_writer.take() {
            writer.finish()?;
        }
        info!("--- END lifis tick ---");
        Ok(())
    });

    #[cfg(any(feature = "mt_pubsub", feature = "r2r"))]
    let spinner_shutdown_flag = shutdown_flag.clone();

    #[cfg(feature = "mt_pubsub")]
    {
        // Check shutdown flag periodically instead of parking forever
        loop {
            if processing_handle.is_finished() {
                info!("Processing thread finished in mt_pubsub spinner");
                break;
            }
            if let Some(ref flag) = spinner_shutdown_flag {
                if flag.load(Ordering::Relaxed) {
                    info!("Shutdown flag detected in mt_pubsub spinner");
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        return match processing_handle.join() {
            Ok(Ok(())) => Ok(()),
            Ok(Err(err)) => Err(err.into()),
            Err(_) => Err(anyhow!("LIO processing thread panicked").into()),
        };
    }

    #[cfg(feature = "ros2-client")]
    {
        // ros2-client doesn't have a simple way to break the spinner, so we run it in a thread
        // and check the shutdown flag periodically
        let spinner_shutdown = shutdown_flag.clone();
        let _spinner_handle = std::thread::spawn(move || {
            future::block_on(async {
                match node.spinner().unwrap().spin().await {
                    Err(e) => {
                        error!("Node did not exit successfully {e}");
                    }
                    Ok(_) => {}
                }
            });
        });

        // Monitor shutdown flag and wait for processing to complete
        loop {
            if let Some(ref flag) = spinner_shutdown {
                if flag.load(Ordering::Relaxed) {
                    info!("Shutdown flag detected, waiting for processing thread");
                    break;
                }
            }
            if processing_handle.is_finished() {
                info!("Processing thread finished");
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }

        return match processing_handle.join() {
            Ok(Ok(())) => Ok(()),
            Ok(Err(err)) => Err(err.into()),
            Err(_) => Err(anyhow!("LIO processing thread panicked").into()),
        };
    }

    #[cfg(feature = "r2r")]
    {
        let spin_timeout = Duration::from_millis(10);
        loop {
            node.spin_once(spin_timeout.clone());
            if let Some(ref flag) = spinner_shutdown_flag {
                if flag.load(Ordering::Relaxed) {
                    info!("Shutdown flag detected in r2r spinner");
                    break;
                }
            }
            if processing_handle.is_finished() {
                info!("Processing thread finished");
                break;
            }
        }
        return match processing_handle.join() {
            Ok(Ok(())) => Ok(()),
            Ok(Err(err)) => Err(err.into()),
            Err(_) => Err(anyhow!("LIO processing thread panicked").into()),
        };
    }

    #[cfg(not(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r")))]
    {
        match processing_handle.join() {
            Ok(Ok(())) => Ok(()),
            Ok(Err(err)) => Err(err.into()),
            Err(_) => Err(anyhow!("LIO processing thread panicked").into()),
        }
    }
}

#[cfg(not(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r")))]
fn lio_with_ready_signal(
    _config: LioConfig,
    _ready_signal: Option<std::sync::mpsc::Sender<()>>,
    _export_trajectory: Option<String>,
    _synchronous_mode: bool,
    _shared_stats: Option<Arc<SharedStats>>,
    _shutdown_flag: Option<Arc<AtomicBool>>,
    _shutdown_request_rx: Option<std::sync::mpsc::Receiver<OfflineShutdownRequest>>,
    _offline_prefetch_scans: usize,
    _output_mcap: Option<OfflineMcapConfig>,
) -> Result<(), Box<dyn CoreError>> {
    Err(anyhow!(
        "No IO backend is enabled. Rebuild with one of: offline, mt_pubsub, ros2-client, or r2r."
    )
    .into())
}

#[allow(dead_code)]
fn log_minot_line(line: &str) {
    if let Some(rest) = line.strip_prefix("[ERROR] ") {
        error!("[Minot] {}", rest.trim());
    } else if let Some(rest) = line.strip_prefix("[WARN] ") {
        warn!("[Minot] {}", rest.trim());
    } else if let Some(rest) = line.strip_prefix("[DEBUG] ") {
        debug!("[Minot] {}", rest.trim());
    } else if let Some(rest) = line.strip_prefix("[TRACE] ") {
        trace!("[Minot] {}", rest.trim());
    } else if let Some(rest) = line.strip_prefix("[INFO] ") {
        info!("[Minot] {}", rest.trim());
    } else {
        info!("[Minot] {}", line);
    }
}

#[cfg(feature = "rat")]
#[derive(Debug)]
enum OfflineRuntimeEvent {
    TuiShutdownRequested,
    MinotReady,
    MinotEndOfBag,
    MinotExited(std::io::Result<std::process::ExitStatus>),
    LioExited(Result<(), String>),
}

#[cfg_attr(not(feature = "rat"), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
enum OfflineShutdownRequest {
    Immediate,
    DrainAfterEndOfBag,
}

#[cfg(feature = "rat")]
fn signal_process(pid: u32, signal: libc::c_int) -> std::io::Result<()> {
    let rc = unsafe { libc::kill(pid as i32, signal) };
    if rc == 0 {
        return Ok(());
    }

    let err = std::io::Error::last_os_error();
    if err.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(err)
    }
}

fn cpu_backend_label(config: &LioConfig) -> String {
    let cores = effective_cpu_cores(config);
    let unit = if cores == 1 { "core" } else { "cores" };
    format!("CPU ({cores} {unit})")
}

#[cfg(feature = "rayon")]
fn effective_cpu_cores(config: &LioConfig) -> usize {
    if config.num_threads > 0 {
        config.num_threads
    } else {
        num_cpus::get()
    }
}

#[cfg(not(feature = "rayon"))]
fn effective_cpu_cores(_config: &LioConfig) -> usize {
    1
}

fn set_backend_from_config(shared_stats: &SharedStats, config: &LioConfig) {
    let (backend, accelerator, details) = match ActiveBackend::detect(config) {
        ActiveBackend::Cpu => (cpu_backend_label(config), None, None),
        #[cfg(feature = "cuda")]
        ActiveBackend::Cuda(_) => (
            cpu_backend_label(config),
            Some("CUDA".to_string()),
            pelorus_cuda::get_cuda_metadata().map(|m| format_cuda_metadata_details(&m)),
        ),
        #[cfg(feature = "metal")]
        ActiveBackend::Metal(_) => (
            cpu_backend_label(config),
            Some("Metal".to_string()),
            None, // Metal metadata not implemented
        ),
        #[cfg(feature = "vulkan")]
        ActiveBackend::Vulkan(_) => (
            cpu_backend_label(config),
            Some("Vulkan".to_string()),
            pelorus_vulkan::get_vulkan_metadata().map(|m| format_vulkan_metadata_details(&m)),
        ),
    };

    shared_stats.set_backend_info(backend, accelerator, details);
}

#[cfg(feature = "cuda")]
fn format_cuda_metadata_details(metadata: &pelorus_cuda::CudaMetadata) -> String {
    let primary = metadata
        .runtime_version
        .or(metadata.driver_version)
        .map(|v| v.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let maybe_driver_suffix = match (metadata.runtime_version, metadata.driver_version) {
        (Some(runtime), Some(driver)) if runtime != driver => format!(" (driver {})", driver),
        _ => String::new(),
    };

    format!(
        "v{}{} - {}",
        primary, maybe_driver_suffix, metadata.device_name
    )
}

#[cfg(feature = "vulkan")]
fn format_vulkan_metadata_details(metadata: &pelorus_vulkan::VulkanMetadata) -> String {
    format!("v{} - {}", metadata.api_version, metadata.device_name)
}

#[cfg(all(test, feature = "cuda"))]
mod backend_details_tests {
    use super::format_cuda_metadata_details;

    #[test]
    fn cuda_details_show_runtime_only_when_runtime_equals_driver() {
        let details = format_cuda_metadata_details(&pelorus_cuda::CudaMetadata {
            device_name: "NVIDIA GeForce RTX 4060 Ti".to_string(),
            runtime_version: Some(pelorus_cuda::CudaSemanticVersion {
                major: 13,
                minor: 2,
                patch: 0,
            }),
            driver_version: Some(pelorus_cuda::CudaSemanticVersion {
                major: 13,
                minor: 2,
                patch: 0,
            }),
        });
        assert_eq!(details, "v13.2.0 - NVIDIA GeForce RTX 4060 Ti");
    }

    #[test]
    fn cuda_details_append_driver_when_versions_differ() {
        let details = format_cuda_metadata_details(&pelorus_cuda::CudaMetadata {
            device_name: "NVIDIA RTX".to_string(),
            runtime_version: Some(pelorus_cuda::CudaSemanticVersion {
                major: 12,
                minor: 4,
                patch: 1,
            }),
            driver_version: Some(pelorus_cuda::CudaSemanticVersion {
                major: 13,
                minor: 0,
                patch: 0,
            }),
        });
        assert_eq!(details, "v12.4.1 (driver 13.0.0) - NVIDIA RTX");
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod vulkan_backend_details_tests {
    use super::format_vulkan_metadata_details;

    #[test]
    fn vulkan_details_show_api_version_and_gpu_name() {
        let details = format_vulkan_metadata_details(&pelorus_vulkan::VulkanMetadata {
            device_name: "NVIDIA Vulkan GPU".to_string(),
            api_version: pelorus_vulkan::VulkanVersion {
                major: 1,
                minor: 3,
                patch: 0,
            },
            driver_version: pelorus_vulkan::VulkanVersion {
                major: 550,
                minor: 40,
                patch: 0,
            },
            driver_version_raw: 0,
        });
        assert_eq!(details, "v1.3.0 - NVIDIA Vulkan GPU");
    }
}

#[cfg(any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"))]
fn set_backend_from_runtime(
    shared_stats: &SharedStats,
    config: &LioConfig,
    active: &ActiveBackend,
) {
    let backend = cpu_backend_label(config);
    let (accelerator, details) = match active {
        ActiveBackend::Cpu => (None, None),
        #[cfg(feature = "cuda")]
        ActiveBackend::Cuda(_) => (
            Some("CUDA".to_string()),
            pelorus_cuda::get_cuda_metadata().map(|m| format_cuda_metadata_details(&m)),
        ),
        #[cfg(feature = "metal")]
        ActiveBackend::Metal(_) => (Some("Metal".to_string()), None),
        #[cfg(feature = "vulkan")]
        ActiveBackend::Vulkan(_) => (
            Some("Vulkan".to_string()),
            pelorus_vulkan::get_vulkan_metadata().map(|m| format_vulkan_metadata_details(&m)),
        ),
    };

    shared_stats.set_backend_info(backend, accelerator, details);
}

#[cfg(feature = "rat")]
fn canonicalize_bagfile_path(path: &std::path::Path) -> anyhow::Result<PathBuf> {
    path.canonicalize().map_err(|e| {
        anyhow!(
            "Failed to canonicalize bagfile path '{}': {}",
            path.display(),
            e
        )
    })
}

#[cfg(feature = "rat")]
fn confirm_marina_pull(input: &str, bag: &str, registry: &str) -> anyhow::Result<bool> {
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return Err(anyhow!(
            "Bagfile input '{}' is only available remotely as '{}' in registry '{}'. \
This command is non-interactive, so pull it first with `marina pull {}` and retry.",
            input,
            bag,
            registry,
            bag
        ));
    }

    print!(
        "Bagfile input '{}' is not local. Pull '{}' from registry '{}' now? [y/N]: ",
        input, bag, registry
    );
    std::io::stdout()
        .flush()
        .map_err(|e| anyhow!("Failed to flush prompt output: {}", e))?;

    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(|e| anyhow!("Failed to read pull confirmation: {}", e))?;

    match answer.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" => Ok(true),
        "" | "n" | "no" => Ok(false),
        _ => Err(anyhow!(
            "Invalid response '{}'. Please answer with 'y' or 'n'.",
            answer.trim()
        )),
    }
}

#[cfg(feature = "rat")]
fn confirm_output_overwrite(
    label: &str,
    path: &std::path::Path,
    assume_yes: bool,
) -> anyhow::Result<bool> {
    if !path.exists() {
        return Ok(true);
    }
    if assume_yes {
        return Ok(true);
    }

    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return Err(anyhow!(
            "{} '{}' already exists. This command is non-interactive, so refusing to overwrite it.",
            label,
            path.display()
        ));
    }

    print!(
        "{} '{}' already exists. Overwrite it? [y/N]: ",
        label,
        path.display()
    );
    std::io::stdout()
        .flush()
        .map_err(|e| anyhow!("Failed to flush prompt output: {}", e))?;

    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(|e| anyhow!("Failed to read overwrite confirmation: {}", e))?;

    match answer.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" => Ok(true),
        "" | "n" | "no" => Ok(false),
        _ => Err(anyhow!(
            "Invalid response '{}'. Please answer with 'y' or 'n'.",
            answer.trim()
        )),
    }
}

#[cfg(feature = "rat")]
fn resolve_offline_bagfile_path(input: &std::path::Path) -> anyhow::Result<PathBuf> {
    if input.exists() {
        if input.is_dir() || input.is_file() {
            return canonicalize_bagfile_path(input);
        }
        return Err(anyhow!(
            "Bagfile path '{}' exists but is neither a regular file nor a directory.",
            input.display()
        ));
    }

    let input_ref = input.to_string_lossy().to_string();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow!("Failed to initialize async runtime for marina: {}", e))?;

    let resolved_path = rt.block_on(async {
        let mut marina =
            Marina::load().map_err(|e| anyhow!("Failed to initialize marina: {}", e))?;
        let resolved = marina.resolve_target(&input_ref, None).await.map_err(|e| {
            anyhow!(
                "Requested path or marina tag '{}' does not exist locally and could not be resolved \
with the current marina registries: {}",
                input_ref,
                e
            )
        })?;

        match resolved {
            ResolveResult::LocalPath(path) | ResolveResult::Cached(path) => Ok(path),
            ResolveResult::RemoteAvailable { bag, registry, .. } => {
                let bag_display = bag.to_string();
                let registry_name = registry.to_string();

                if !confirm_marina_pull(&input_ref, &bag_display, &registry_name)? {
                    return Err(anyhow!(
                        "Bagfile input '{}' is remote-only ('{}' in '{}'). Pull it first with \
`marina pull {}` and retry.",
                        input_ref,
                        bag_display,
                        registry_name,
                        bag_display
                    ));
                }

                let mut stdout = std::io::stdout();
                let mut sink = WriterProgress::new(&mut stdout);
                let mut reporter = ProgressReporter::new(&mut sink);
                marina
                    .pull_exact_with_progress(&bag, Some(registry_name.as_str()), &mut reporter)
                    .await
                    .map_err(|e| {
                        anyhow!(
                            "Failed to pull marina tag '{}' from registry '{}': {}",
                            bag_display,
                            registry_name,
                            e
                        )
                    })
            }
            ResolveResult::Ambiguous { candidates } => Err(anyhow!(
                "Requested path or marina tag '{}' is ambiguous across {} registries. \
Please specify a local path or resolve/pull with an explicit registry.",
                input_ref,
                candidates.len()
            )),
        }
    })?;

    canonicalize_bagfile_path(resolved_path.as_path())
}

fn main() -> Result<(), Box<dyn CoreError>> {
    // Initialize our custom logger that works with both TUI and console
    init_logger();

    let args = Args::parse();

    const DEFAULT_CONFIG_STR: &str = include_str!("../../default.rl");

    match args.command {
        Command::Completions { shell } => {
            let mut cmd = Args::command();
            generate(
                shell,
                &mut cmd,
                "pelorus".to_owned(),
                &mut std::io::stdout(),
            );
            Ok(())
        }
        /* `run` wrapper removed; `online`/`offline` are top-level commands */
        Command::Online(config_args) => {
            let default_ast = ratslang::compile_code(DEFAULT_CONFIG_STR)
                .map_err(|e| anyhow!("Failed to compile embedded default config: {}", e))?;

            let user_ast = if let Some(config_path) = config_args.config_path {
                let path = std::path::Path::new(&config_path);
                if !path.exists() {
                    return Err(anyhow!(
                        "Error: Configuration file not found at '{}'",
                        config_path.to_string_lossy()
                    )
                    .into());
                }
                ratslang::compile_file(&path.to_path_buf(), None, None).map_err(|e| {
                    anyhow!(
                        "Failed to compile user config '{}': {}",
                        config_path.to_string_lossy(),
                        e
                    )
                })?
            } else {
                ratslang::compile_code("")?
            };

            let mut config = ast_to_config(default_ast, user_ast)?;
            if config_args.cpu_only {
                config.cpu_only = true;
            }

            if config.num_threads > 0 {
                #[cfg(feature = "rayon")]
                if let Err(e) = rayon::ThreadPoolBuilder::new()
                    .num_threads(config.num_threads)
                    .build_global()
                {
                    warn!(
                        "Failed to configure rayon thread pool to {} threads: {e}",
                        config.num_threads
                    );
                }
                #[cfg(not(feature = "rayon"))]
                warn!(
                    "Ignoring num_threads={}; thread-pool configuration requires the rayon feature.",
                    config.num_threads
                );
            }

            // Update log level after config has set RUST_LOG
            update_log_level();

            let mut show_tui = config.show_tui;

            if show_tui && !std::io::stdout().is_terminal() {
                show_tui = false;
            }

            if !show_tui {
                // TUI won't be used — restore stderr logging suppressed by init_logger
                set_tui_active(false);
            }

            if show_tui {
                let shared_stats = Arc::new(SharedStats::new());
                shared_stats.set_mode("Online");
                set_backend_from_config(&shared_stats, &config);
                let stats_for_tui = shared_stats.clone();

                shared_stats.set_target_lidar_rate(10.0); // will be updated when first scan arrives

                let (tui_stop_tx, tui_stop_rx) = std::sync::mpsc::channel();
                let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel();

                let tui_thread = std::thread::spawn(move || {
                    let mut tui = StatsTui::new(stats_for_tui)
                        .with_stop_receiver(tui_stop_rx)
                        .with_shutdown_sender(shutdown_tx);
                    let _ = tui.run();
                });
                let lio_config = config.clone();
                let lio_export = config_args.export_trajectory.clone();
                let lio_sync = config_args.rat;
                let lio_stats = Some(shared_stats);

                let lio_shutdown_flag = Arc::new(AtomicBool::new(false));
                let lio_shutdown_flag_for_thread = lio_shutdown_flag.clone();

                let lio_handle = std::thread::spawn(move || -> Result<(), String> {
                    lio_with_ready_signal(
                        lio_config,
                        None,
                        lio_export,
                        lio_sync,
                        lio_stats,
                        Some(lio_shutdown_flag_for_thread),
                        None,
                        1,
                        None,
                    )
                    .map_err(|e| e.to_string())
                });

                // Wait for either TUI shutdown or LIO completion
                loop {
                    if shutdown_rx.try_recv().is_ok() {
                        // TUI requested shutdown -> request LIO to stop
                        lio_shutdown_flag.store(true, Ordering::SeqCst);
                        info!("received shutdown from TUI");
                        break;
                    }
                    if lio_handle.is_finished() {
                        // LIO terminated -> request TUI to stop
                        let _ = tui_stop_tx.send(());
                        info!("lio finished, stopping tui");
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }

                // Ensure both threads are asked to stop
                let _ = tui_stop_tx.send(());
                lio_shutdown_flag.store(true, Ordering::SeqCst);

                // Join TUI thread first (should be quick)
                let tui_join_result = tui_thread.join();
                if tui_join_result.is_err() {
                    warn!("TUI thread panicked");
                }

                // Get LIO result
                match lio_handle.join() {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(e)) => Err(anyhow!("{}", e).into()),
                    Err(_) => Err(anyhow!("LIO thread panicked").into()),
                }
            } else {
                lio(config, config_args.export_trajectory, config_args.rat)
            }
        }
        #[cfg(feature = "rat")]
        Command::Offline(offline_args) => {
            let shared_stats = Arc::new(SharedStats::new());
            shared_stats.set_mode("Offline");
            let stats_for_tui = shared_stats.clone();
            // Set default target rate, will be updated by measured rate
            shared_stats.set_target_lidar_rate(10.0);

            let default_ast = ratslang::compile_code(DEFAULT_CONFIG_STR)
                .map_err(|e| anyhow!("Failed to compile embedded default config: {}", e))?;

            let user_ast = if let Some(config_path) = offline_args.config_path {
                let path = std::path::Path::new(&config_path);
                if !path.exists() {
                    return Err(anyhow!(
                        "Error: Configuration file not found at '{}'",
                        config_path.to_string_lossy()
                    )
                    .into());
                }
                ratslang::compile_file(&path.to_path_buf(), None, None).map_err(|e| {
                    anyhow!(
                        "Failed to compile user config '{}': {}",
                        config_path.to_string_lossy(),
                        e
                    )
                })?
            } else {
                ratslang::compile_code("")?
            };

            let config = {
                let mut config = ast_to_config(default_ast, user_ast)?;
                config.prelio_node_params.synchronous_mode = true;
                if offline_args.cpu_only {
                    config.cpu_only = true;
                }
                config
            };

            let output_mcap_config = offline_args
                .output_mcap
                .clone()
                .or_else(|| {
                    offline_args
                        .output_rosbag
                        .as_ref()
                        .map(|path| rosbag_mcap_path(path))
                })
                .map(|path| {
                    let filtered_imu_topic = offline_args
                        .mcap_filtered_imu_topic
                        .clone()
                        .unwrap_or_else(|| {
                            let configured = config.prelio_node_params.imu_filtered_topic.clone();
                            if configured.is_empty() {
                                "imu/filtered".to_string()
                            } else {
                                configured
                            }
                        });
                    OfflineMcapConfig {
                        path,
                        rosbag_metadata_path: offline_args.output_rosbag_metadata.clone().or_else(
                            || {
                                offline_args
                                    .output_rosbag
                                    .as_ref()
                                    .map(|path| path.join("metadata.yaml"))
                            },
                        ),
                        topics: McapTopics {
                            cloud: normalize_mcap_topic(offline_args.mcap_cloud_topic.clone()),
                            odom: normalize_mcap_topic(offline_args.mcap_odom_topic.clone()),
                            path: normalize_mcap_topic(offline_args.mcap_path_topic.clone()),
                            tf: normalize_mcap_topic(offline_args.mcap_tf_topic.clone()),
                            filtered_imu: normalize_mcap_topic(filtered_imu_topic),
                        },
                    }
                });
            let mut rosbag_overwrite_confirmed = false;
            if let Some(rosbag_path) = offline_args.output_rosbag.as_ref() {
                if !confirm_output_overwrite(
                    "ROS bag output directory",
                    rosbag_path,
                    offline_args.yes,
                )? {
                    info!(
                        "ROS bag output directory '{}' was not overwritten; exiting.",
                        rosbag_path.display()
                    );
                    return Ok(());
                }
                rosbag_overwrite_confirmed = rosbag_path.exists();
                std::fs::create_dir_all(rosbag_path).map_err(|e| {
                    anyhow!(
                        "Failed to create ROS bag output directory '{}': {}",
                        rosbag_path.display(),
                        e
                    )
                })?;
                info!("ROS bag output directory: {}", rosbag_path.display());
            }
            if let Some(output_mcap) = output_mcap_config.as_ref() {
                if !confirm_output_overwrite(
                    "MCAP output",
                    &output_mcap.path,
                    offline_args.yes || rosbag_overwrite_confirmed,
                )? {
                    info!(
                        "MCAP output '{}' was not overwritten; exiting.",
                        output_mcap.path.display()
                    );
                    return Ok(());
                }
                info!("MCAP output target: {}", output_mcap.path.display());
                if let Some(metadata_path) = output_mcap.rosbag_metadata_path.as_ref() {
                    if !confirm_output_overwrite(
                        "ROS bag metadata",
                        metadata_path,
                        offline_args.yes || rosbag_overwrite_confirmed,
                    )? {
                        info!(
                            "ROS bag metadata '{}' was not overwritten; exiting.",
                            metadata_path.display()
                        );
                        return Ok(());
                    }
                    info!("ROS bag metadata target: {}", metadata_path.display());
                }
            }

            set_backend_from_config(&shared_stats, &config);

            if config.num_threads > 0 {
                #[cfg(feature = "rayon")]
                if let Err(e) = rayon::ThreadPoolBuilder::new()
                    .num_threads(config.num_threads)
                    .build_global()
                {
                    warn!(
                        "Failed to configure rayon thread pool to {} threads: {e}",
                        config.num_threads
                    );
                }
                #[cfg(not(feature = "rayon"))]
                warn!(
                    "Ignoring num_threads={}; thread-pool configuration requires the rayon feature.",
                    config.num_threads
                );
            }

            let show_tui = config.show_tui && std::io::stdout().is_terminal();

            if !show_tui {
                // TUI won't be used so restore stderr logging suppressed by init_logger
                set_tui_active(false);
            }

            // Update log level after config has set RUST_LOG
            update_log_level();

            let lidar_topic = &config.prelio_node_params.cloud_src_topic;
            let imu_topic = &config.prelio_node_params.imu_src_topic;
            let bagfile_input = offline_args.bagfile_path.to_string_lossy().to_string();
            let input_looks_like_tag = bagfile_input.contains(':');
            let bagfile_path_canon = resolve_offline_bagfile_path(&offline_args.bagfile_path)?;
            let bagfile_path = bagfile_path_canon.clone().to_string_lossy().to_string();
            let bagfile_label = if !offline_args.bagfile_path.exists() && input_looks_like_tag {
                bagfile_input
            } else {
                bagfile_path_canon
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            };
            shared_stats.set_bagfile(bagfile_label);

            let mt_content = format!(
                r#"
_bag.{{
  cloud.{{
    _topic = "{lidar_topic}"
    _short = "l"
    _type = Cloud
  }}
  imu.{{
    _topic = "{imu_topic}"
    _short = "i"
    _type = Imu
  }}
}}
_start_locked = false

# Setting triggers for topics.
# During IMU bootstrap, advance IMU only. The first active LIO tick
# advances to one cloud with its surrounding IMUs. After that, each
# tick_end publishes one cloud and then advances IMUs up to the next
# cloud without consuming that next cloud.
pf! i 1 imu_init_tick
pf! i, l 1 lio_start_tick
pf! l 1 lio_start_tick
pf! i, l 1 lio_start_tick
pf! l 1 tick_end
pf! i, l 1 tick_end

reset! {bagfile_path}

# Starting the feedback loop by playing the first IMU message (first step is IMU init)
pf! i 1
"#
            );

            // Start TUI immediately so we can see what's happening
            let (tui_stop_tx, tui_stop_rx) = std::sync::mpsc::channel();
            let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel();

            let tui_thread = if show_tui {
                Some(std::thread::spawn(move || {
                    let mut tui = StatsTui::new(stats_for_tui)
                        .with_stop_receiver(tui_stop_rx)
                        .with_shutdown_sender(shutdown_tx);
                    let _ = tui.run();
                }))
            } else {
                drop(shutdown_tx);
                None
            };

            // Define the main logic as a closure/block to easily handle errors and cleanup
            let export_trajectory_path =
                resolve_export_trajectory_path(offline_args.export_trajectory.clone());
            let export_trajectory_path_for_run = export_trajectory_path.clone();
            let run_offline_logic = || -> anyhow::Result<()> {
                let mt_file_path = std::env::temp_dir().join("pelorus_offline.mt");
                fs::write(&mt_file_path, mt_content)
                    .map_err(|e| anyhow!("Failed to write .mt file: {}", e))?;

                info!("Generated .mt file at: {:?}", mt_file_path);

                // Spawn minot in headless mode with --sync
                let mut minot_child = ProcessCommand::new(&offline_args.minot_path)
                    .arg("headless")
                    .arg(&mt_file_path)
                    .arg("--sync")
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .map_err(|e| anyhow!("Failed to spawn minot: {}", e))?;

                let minot_pid = minot_child.id();
                info!("Spawned minot process with PID: {}", minot_pid);

                let mut minot_stdin = minot_child
                    .stdin
                    .take()
                    .ok_or_else(|| anyhow!("Failed to open minot stdin"))?;
                let minot_stdout = minot_child
                    .stdout
                    .take()
                    .ok_or_else(|| anyhow!("Failed to open minot stdout"))?;
                let minot_stderr = minot_child
                    .stderr
                    .take()
                    .ok_or_else(|| anyhow!("Failed to open minot stderr"))?;

                info!("Waiting for minot to be ready...");
                use std::io::{BufRead, BufReader};

                let (runtime_tx, runtime_rx) = std::sync::mpsc::channel::<OfflineRuntimeEvent>();
                let ready_notified = Arc::new(AtomicBool::new(false));

                // Thread for stdout
                let runtime_tx_stdout = runtime_tx.clone();
                let ready_notified_stdout = ready_notified.clone();
                std::thread::spawn(move || {
                    let reader = BufReader::new(minot_stdout);
                    for line in reader.lines() {
                        match line {
                            Ok(line) => {
                                let trimmed = line.trim();
                                if !trimmed.is_empty() {
                                    log_minot_line(trimmed);
                                }

                                if line.to_lowercase().contains("end of bagfile") {
                                    let _ =
                                        runtime_tx_stdout.send(OfflineRuntimeEvent::MinotEndOfBag);
                                }

                                if line.contains("--sync")
                                    || line.to_lowercase().contains("waiting for input")
                                    || line.to_lowercase().contains("press enter")
                                    || line.to_lowercase().contains("stdin")
                                {
                                    if !ready_notified_stdout.swap(true, Ordering::SeqCst) {
                                        let _ =
                                            runtime_tx_stdout.send(OfflineRuntimeEvent::MinotReady);
                                    }
                                }
                            }
                            Err(e) => {
                                error!("[Minot] Error reading stdout: {}", e);
                                break;
                            }
                        }
                    }
                });

                // Thread for stderr
                let runtime_tx_stderr = runtime_tx.clone();
                let ready_notified_stderr = ready_notified.clone();
                std::thread::spawn(move || {
                    let reader = BufReader::new(minot_stderr);
                    for line in reader.lines() {
                        match line {
                            Ok(line) => {
                                let trimmed = line.trim();
                                if !trimmed.is_empty() {
                                    log_minot_line(trimmed);
                                }

                                if line.to_lowercase().contains("end of bagfile") {
                                    let _ =
                                        runtime_tx_stderr.send(OfflineRuntimeEvent::MinotEndOfBag);
                                }

                                // detect readiness on stderr
                                if line.contains(
                                    "Waiting for input on stdin before execution (--sync mode)",
                                ) || line.to_lowercase().contains("waiting for input")
                                    || line.to_lowercase().contains("press enter")
                                    || line.to_lowercase().contains("stdin")
                                {
                                    if !ready_notified_stderr.swap(true, Ordering::SeqCst) {
                                        let _ =
                                            runtime_tx_stderr.send(OfflineRuntimeEvent::MinotReady);
                                    }
                                }
                            }
                            Err(e) => {
                                error!("[Minot] Error reading stderr: {}", e);
                                break;
                            }
                        }
                    }
                });

                let runtime_tx_wait = runtime_tx.clone();
                std::thread::spawn(move || {
                    let status = minot_child.wait();
                    let _ = runtime_tx_wait.send(OfflineRuntimeEvent::MinotExited(status));
                });

                let runtime_tx_tui = runtime_tx.clone();
                std::thread::spawn(move || {
                    if shutdown_rx.recv().is_ok() {
                        let _ = runtime_tx_tui.send(OfflineRuntimeEvent::TuiShutdownRequested);
                    }
                });

                loop {
                    match runtime_rx
                        .recv()
                        .map_err(|_| anyhow!("Failed while waiting for Minot readiness"))?
                    {
                        OfflineRuntimeEvent::MinotReady => break,
                        OfflineRuntimeEvent::TuiShutdownRequested => {
                            let _ = signal_process(minot_pid, libc::SIGKILL);
                            return Ok(());
                        }
                        OfflineRuntimeEvent::MinotExited(Ok(status)) => {
                            return Err(anyhow!(
                                "Minot exited before readiness handshake: {}",
                                status
                            ));
                        }
                        OfflineRuntimeEvent::MinotExited(Err(e)) => {
                            return Err(anyhow!("Failed to wait on Minot: {}", e));
                        }
                        OfflineRuntimeEvent::MinotEndOfBag | OfflineRuntimeEvent::LioExited(_) => {}
                    }
                }

                info!("Minot is ready, starting pelorus offline processing...");

                let (pelorus_ready_tx, pelorus_ready_rx) = std::sync::mpsc::channel();

                // start Odometry in thread
                // Provide an atomic shutdown flag so we can request LIO to shutdown from main/TUI
                let lio_shutdown_flag = Arc::new(AtomicBool::new(false));
                let lio_shutdown_flag_for_thread = lio_shutdown_flag.clone();
                let (lio_stop_tx, lio_stop_rx) =
                    std::sync::mpsc::channel::<OfflineShutdownRequest>();
                let shared_stats_for_lio = shared_stats.clone();
                let runtime_tx_lio = runtime_tx.clone();
                let offline_prefetch_scans = offline_args.prefetch_scans;
                let output_mcap_config_for_run = output_mcap_config.clone();

                let lio_handle = std::thread::spawn(move || {
                    let synchronous_mode = config.prelio_node_params.synchronous_mode;
                    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        lio_with_ready_signal(
                            config,
                            Some(pelorus_ready_tx),
                            export_trajectory_path_for_run.clone(),
                            synchronous_mode,
                            Some(shared_stats_for_lio),
                            Some(lio_shutdown_flag_for_thread),
                            Some(lio_stop_rx),
                            offline_prefetch_scans,
                            output_mcap_config_for_run,
                        )
                    }))
                    .map_err(|_| "LIO thread panicked".to_string())
                    .and_then(|res| res.map_err(|e| format!("Odometry failed: {}", e)));

                    let _ = runtime_tx_lio.send(OfflineRuntimeEvent::LioExited(result.clone()));
                    result
                });

                info!("Waiting for pelorus to initialize...");
                pelorus_ready_rx
                    .recv()
                    .map_err(|_| anyhow!("Failed to receive Pelorus readiness signal"))?;
                info!("Pelorus is ready!");
                info!("Triggering minot to start processing...");

                use std::io::Write;
                minot_stdin
                    .write_all(b"\n")
                    .map_err(|e| anyhow!("Failed to write to minot stdin: {}", e))?;
                minot_stdin
                    .flush()
                    .map_err(|e| anyhow!("Failed to flush minot stdin: {}", e))?;

                let mut lio_shutdown_requested = false;
                let mut minot_force_stopped = false;
                let mut user_shutdown_requested = false;
                let mut wait_for_minot_exit = true;
                let mut lio_exit_result: Option<Result<(), String>> = None;
                let mut minot_exit_result: Option<std::io::Result<std::process::ExitStatus>> = None;

                while lio_exit_result.is_none()
                    || (wait_for_minot_exit && minot_exit_result.is_none())
                {
                    match runtime_rx
                        .recv()
                        .map_err(|_| anyhow!("Offline runtime channel closed unexpectedly"))?
                    {
                        OfflineRuntimeEvent::TuiShutdownRequested => {
                            user_shutdown_requested = true;
                            if !lio_shutdown_requested {
                                // User interrupt path:
                                // 1) request internal shutdown,
                                // 2) finish current in-flight frame,
                                // 3) terminate Minot only after LIO exits.
                                lio_shutdown_flag.store(true, Ordering::SeqCst);
                                lio_shutdown_requested = true;
                            }
                        }
                        OfflineRuntimeEvent::MinotEndOfBag => {
                            info!("Minot reached end of bagfile");
                            if !lio_shutdown_requested {
                                let _ =
                                    lio_stop_tx.send(OfflineShutdownRequest::DrainAfterEndOfBag);
                                lio_shutdown_requested = true;
                            }
                        }
                        OfflineRuntimeEvent::MinotExited(status) => {
                            minot_exit_result = Some(status);
                            if !lio_shutdown_requested {
                                let _ = lio_stop_tx.send(OfflineShutdownRequest::Immediate);
                                lio_shutdown_requested = true;
                            }
                        }
                        OfflineRuntimeEvent::LioExited(result) => {
                            lio_exit_result = Some(result);
                            if user_shutdown_requested
                                && minot_exit_result.is_none()
                                && !minot_force_stopped
                            {
                                // Stop external data source only after current-frame processing
                                // has completed and LIO has exited.
                                signal_process(minot_pid, libc::SIGKILL)
                                    .map_err(|e| anyhow!("Failed to terminate Minot: {}", e))?;
                                minot_force_stopped = true;
                                // On user interrupt, once LIO is down we don't wait on child
                                // wait-notification delivery to avoid deadlocks.
                                wait_for_minot_exit = false;
                            } else if minot_exit_result.is_none() && !minot_force_stopped {
                                signal_process(minot_pid, libc::SIGKILL)
                                    .map_err(|e| anyhow!("Failed to terminate Minot: {}", e))?;
                                minot_force_stopped = true;
                            }
                        }
                        OfflineRuntimeEvent::MinotReady => {}
                    }
                }

                match lio_handle.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => return Err(anyhow!("{}", e)),
                    Err(_) => return Err(anyhow!("LIO thread panicked")),
                }

                if wait_for_minot_exit {
                    match minot_exit_result.expect("minot exit must be observed before completion")
                    {
                        Ok(status) if status.success() || minot_force_stopped => {}
                        Ok(status) => return Err(anyhow!("Minot exited with status {}", status)),
                        Err(e) => return Err(anyhow!("Failed to wait for Minot exit: {}", e)),
                    }
                }

                Ok(())
            };

            match run_offline_logic() {
                Ok(_) => {
                    let _ = tui_stop_tx.send(());
                    if let Some(handle) = tui_thread {
                        let _ = handle.join();
                    }
                    Ok(())
                }
                Err(e) => {
                    error!("Offline run failed: {}", e);
                    let _ = tui_stop_tx.send(());
                    if let Some(handle) = tui_thread {
                        let _ = handle.join();
                    }
                    Err(e.into())
                }
            }
        }
        Command::Config(cmd) => match cmd {
            ConfigCommand::Generate(args) => {
                match args.kind {
                    GenerateKind::Full => {
                        print!("{}", DEFAULT_CONFIG_STR);
                        Ok(())
                    }
                    GenerateKind::Recommended => {
                        let mut default_ast =
                            ratslang::compile_code(DEFAULT_CONFIG_STR).map_err(|e| {
                                anyhow!("Failed to compile embedded default config: {}", e)
                            })?;

                        // Suppress unused-variable warnings when generating recommended config
                        default_ast.vars = default_ast.vars.with_drop_warning(false);

                        match default_ast.to_string_filtered("recommended") {
                            Some(recommended_config) => {
                                print!("{}", recommended_config);
                                Ok(())
                            }
                            None => Err(anyhow!(
                                "No @recommended annotations found in the default configuration"
                            )
                            .into()),
                        }
                    }
                }
            }
            ConfigCommand::Verify(args) => {
                let default_ast = ratslang::compile_code(DEFAULT_CONFIG_STR)
                    .map_err(|e| anyhow!("Failed to compile embedded default config: {}", e))?;

                let user_ast = if args.config_path.to_str() == Some("-") {
                    use std::io::Read;
                    let mut buffer = String::new();
                    std::io::stdin()
                        .read_to_string(&mut buffer)
                        .map_err(|e| anyhow!("Failed to read from stdin: {}", e))?;
                    ratslang::compile_code(&buffer)
                        .map_err(|e| anyhow!("Failed to compile config from stdin: {}", e))?
                } else {
                    let path = std::path::Path::new(&args.config_path);
                    if !path.exists() {
                        return Err(anyhow!(
                            "Error: Configuration file not found at '{}'",
                            args.config_path.to_string_lossy()
                        )
                        .into());
                    }
                    ratslang::compile_file(&path.to_path_buf(), None, None).map_err(|e| {
                        anyhow!(
                            "Failed to compile user config '{}': {}",
                            args.config_path.to_string_lossy(),
                            e
                        )
                    })?
                };

                let (_, (assigned_vars, evaluated_vars)) =
                    ast_to_config_with_user_usage(default_ast, user_ast)?;

                // Check for unused variables in user config
                let mut has_warnings = false;
                for var in assigned_vars.iter() {
                    if !evaluated_vars.contains(var) {
                        warn!("Variable '{}' is not recognized by Pelorus", var);
                        has_warnings = true;
                    }
                }

                if !has_warnings {
                    println!("Configuration is valid.");
                }

                Ok(())
            }
            #[cfg(feature = "r2r")]
            ConfigCommand::TfAutoRos2(args) => {
                use futures_lite::StreamExt;
                use prelio_io::tf_r2r::{TfListener, isometry_from_transform};

                let default_ast = ratslang::compile_code(DEFAULT_CONFIG_STR)
                    .map_err(|e| anyhow!("Failed to compile embedded default config: {}", e))?;
                let user_ast = if let Some(ref config_path) = args.config_path {
                    let path = std::path::Path::new(config_path);
                    if !path.exists() {
                        return Err(anyhow!(
                            "Error: Configuration file not found at '{}'",
                            config_path.to_string_lossy()
                        )
                        .into());
                    }
                    ratslang::compile_file(&path.to_path_buf(), None, None).map_err(|e| {
                        anyhow!(
                            "Failed to compile user config '{}': {}",
                            config_path.to_string_lossy(),
                            e
                        )
                    })?
                } else {
                    ratslang::compile_code("")?
                };
                let lio_config = ast_to_config(default_ast, user_ast)?;
                let lidar_topic = &lio_config.prelio_node_params.cloud_src_topic;
                let imu_topic = &lio_config.prelio_node_params.imu_src_topic;

                eprintln!(
                    "Connecting to ROS2 and collecting TF data (timeout: {}s)...",
                    args.timeout
                );

                let ctx = r2r::Context::create()?;
                let mut node = r2r::Node::create(ctx, "pelorus_tf_auto", "pelorus")?;

                let tf_listener = TfListener::new(&mut node);

                // Subscribe to lidar and IMU topics to capture their frame IDs
                let mut lidar_sub = node
                    .subscribe::<r2r::sensor_msgs::msg::PointCloud2>(
                        lidar_topic,
                        QosProfile::default(),
                    )
                    .map_err(|e| anyhow!("Failed to subscribe to {}: {}", lidar_topic, e))?;

                let mut imu_sub = node
                    .subscribe::<r2r::sensor_msgs::msg::Imu>(imu_topic, QosProfile::default())
                    .map_err(|e| anyhow!("Failed to subscribe to {}: {}", imu_topic, e))?;

                let (lidar_tx, lidar_rx) = std::sync::mpsc::channel::<String>();
                let (imu_tx, imu_rx) = std::sync::mpsc::channel::<String>();

                std::thread::spawn(move || {
                    future::block_on(async move {
                        if let Some(msg) = lidar_sub.next().await {
                            let _ = lidar_tx.send(msg.header.frame_id);
                        }
                    });
                });

                std::thread::spawn(move || {
                    future::block_on(async move {
                        if let Some(msg) = imu_sub.next().await {
                            let _ = imu_tx.send(msg.header.frame_id);
                        }
                    });
                });

                let timeout = Duration::from_secs_f64(args.timeout);
                let start = std::time::Instant::now();
                let mut lidar_frame: Option<String> = None;
                let mut imu_frame: Option<String> = None;

                while start.elapsed() < timeout {
                    node.spin_once(Duration::from_millis(10));

                    if lidar_frame.is_none() {
                        if let Ok(f) = lidar_rx.try_recv() {
                            eprintln!("Lidar frame ID: {}", f);
                            lidar_frame = Some(f);
                        }
                    }
                    if imu_frame.is_none() {
                        if let Ok(f) = imu_rx.try_recv() {
                            eprintln!("IMU frame ID: {}", f);
                            imu_frame = Some(f);
                        }
                    }
                    if lidar_frame.is_some() && imu_frame.is_some() {
                        break;
                    }
                }

                if lidar_frame.is_none() {
                    eprintln!(
                        "Warning: no message received on {} within timeout",
                        lidar_topic
                    );
                }
                if imu_frame.is_none() {
                    eprintln!(
                        "Warning: no message received on {} within timeout",
                        imu_topic
                    );
                }

                let t0 = r2r::builtin_interfaces::msg::Time { sec: 0, nanosec: 0 };

                let mut config_content = std::fs::read_to_string(&args.config).unwrap_or_default();
                if config_content.is_empty() {
                    config_content =
                        String::from("# Auto-generated by: pelorus config tf-auto-ros2\n\n");
                }

                let find_block_range = |s: &str, start_pos: usize| -> Option<(usize, usize)> {
                    let s_slice = &s[start_pos..];
                    if let Some(brace_start_rel) = s_slice.find('{') {
                        let brace_start = start_pos + brace_start_rel;
                        let mut depth = 0;
                        for (i, c) in s[brace_start..].char_indices() {
                            if c == '{' {
                                depth += 1;
                            } else if c == '}' {
                                depth -= 1;
                                if depth == 0 {
                                    return Some((brace_start, brace_start + i + 1));
                                }
                            }
                        }
                    }
                    None
                };

                // --- imu.from_lidar_transform ---
                if let (Some(lf), Some(imuf)) = (&lidar_frame, &imu_frame) {
                    match tf_listener.lookup_transform(lf, imuf, t0.clone()) {
                        Ok(tf) => {
                            let iso = isometry_from_transform(&tf.transform);
                            let t = &iso.translation;
                            let r = iso.rotation.to_rotation_matrix();
                            let m = r.matrix();
                            let new_block = format!(
                                "from_lidar_transform {{\n\
                                 \x20     use_tf_tree = false\n\
                                 \x20     translation = [{:.6}m, {:.6}m, {:.6}m]\n\
                                 \x20     rotation = [ [{:.9}, {:.9}, {:.9}],\n\
                                 \x20                 [{:.9}, {:.9}, {:.9}],\n\
                                 \x20                 [{:.9}, {:.9}, {:.9}] ]\n\
                                 \x20   }}",
                                t.x,
                                t.y,
                                t.z,
                                m[(0, 0)],
                                m[(0, 1)],
                                m[(0, 2)],
                                m[(1, 0)],
                                m[(1, 1)],
                                m[(1, 2)],
                                m[(2, 0)],
                                m[(2, 1)],
                                m[(2, 2)],
                            );

                            // Find imu block
                            if let Some(imu_pos) = config_content.find("imu ") {
                                if let Some((imu_start, imu_end)) =
                                    find_block_range(&config_content, imu_pos)
                                {
                                    let imu_inner = &config_content[imu_start..imu_end];
                                    if let Some(flt_pos_rel) =
                                        imu_inner.find("from_lidar_transform")
                                    {
                                        let flt_pos = imu_start + flt_pos_rel;
                                        if let Some((_, flt_end)) =
                                            find_block_range(&config_content, flt_pos)
                                        {
                                            config_content
                                                .replace_range(flt_pos..flt_end, &new_block);
                                        }
                                    } else {
                                        // imu exists but no from_lidar_transform, insert before closing }
                                        config_content.insert_str(
                                            imu_end - 1,
                                            &format!("\n  {}\n", new_block),
                                        );
                                    }
                                }
                            } else {
                                // imu block doesn't exist, append it
                                config_content
                                    .push_str(&format!("\nimu {{\n  {}\n}}\n", new_block));
                            }
                        }
                        Err(e) => eprintln!("TF lookup {lf} → {imuf} failed: {e}"),
                    }
                }

                // --- output.base_link.from_lidar_transform ---
                if let (Some(lf), Some(blf)) = (&lidar_frame, &lio_config.base_link_frame) {
                    match tf_listener.lookup_transform(lf, blf, t0) {
                        Ok(tf) => {
                            let iso = isometry_from_transform(&tf.transform);
                            let t = &iso.translation;
                            let r = iso.rotation.to_rotation_matrix();
                            let m = r.matrix();
                            let new_block = format!(
                                "from_lidar_transform {{\n\
                                 \x20       use_tf_tree = false\n\
                                 \x20       translation = [{:.6}m, {:.6}m, {:.6}m]\n\
                                 \x20       rotation = [ [{:.9}, {:.9}, {:.9}],\n\
                                 \x20                   [{:.9}, {:.9}, {:.9}],\n\
                                 \x20                   [{:.9}, {:.9}, {:.9}] ]\n\
                                 \x20     }}",
                                t.x,
                                t.y,
                                t.z,
                                m[(0, 0)],
                                m[(0, 1)],
                                m[(0, 2)],
                                m[(1, 0)],
                                m[(1, 1)],
                                m[(1, 2)],
                                m[(2, 0)],
                                m[(2, 1)],
                                m[(2, 2)],
                            );

                            // replacement for output.base_link.from_lidar_transform
                            if let Some(out_pos) = config_content.find("output ") {
                                if let Some((out_start, out_end)) =
                                    find_block_range(&config_content, out_pos)
                                {
                                    let out_inner = &config_content[out_start..out_end];
                                    if let Some(bl_pos_rel) = out_inner.find("base_link") {
                                        let bl_pos = out_start + bl_pos_rel;
                                        if let Some((bl_start, bl_end)) =
                                            find_block_range(&config_content, bl_pos)
                                        {
                                            let bl_inner = &config_content[bl_start..bl_end];
                                            if let Some(flt_pos_rel) =
                                                bl_inner.find("from_lidar_transform")
                                            {
                                                let flt_pos = bl_start + flt_pos_rel;
                                                if let Some((_, flt_end)) =
                                                    find_block_range(&config_content, flt_pos)
                                                {
                                                    config_content.replace_range(
                                                        flt_pos..flt_end,
                                                        &new_block,
                                                    );
                                                }
                                            } else {
                                                config_content.insert_str(
                                                    bl_end - 1,
                                                    &format!("\n    {}\n", new_block),
                                                );
                                            }
                                        }
                                    } else {
                                        config_content.insert_str(out_end - 1, &format!("\n  base_link {{\n    frame_id = \"{}\"\n    {}\n  }}\n", blf, new_block));
                                    }
                                }
                            } else {
                                config_content.push_str(&format!(
                                    "\noutput {{\n  base_link {{\n    frame_id = \"{}\"\n    {}\n  }}\n}}\n",
                                    blf, new_block
                                ));
                            }
                        }
                        Err(e) => eprintln!("TF lookup {lf} → {blf} failed: {e}"),
                    }
                }

                std::fs::write(&args.config, &config_content).map_err(|e| {
                    anyhow!(
                        "Failed to write config to '{}': {}",
                        args.config.display(),
                        e
                    )
                })?;

                println!("Updated: {}", args.config.display());
                Ok(())
            }
        },
    }
}

#![warn(clippy::std_instead_of_alloc, clippy::std_instead_of_core)]

extern crate alloc;

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
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
#[cfg(all(feature = "rat", unix))]
use core::sync::atomic::AtomicI32;
#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::{AtomicBool, Ordering};
use core::time::Duration;

mod completions;
#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;
#[cfg(feature = "mt_pubsub")]
use pelorus_core::PointCloudMode;
use pelorus_core::config::{
    ast_to_config, ast_to_config_with_user_usage, layer_overrides, parse_overrides,
    reject_unknown_overrides, user_config_source_with_checked_overrides,
};
#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
use pelorus_core::timestamp_to_secs;
#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
use pelorus_core::timing::RuntimeEvaluator;
use pelorus_core::{LioConfig, OdometryRate};
use pelorus_reg::ActiveBackend;
#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
use pelorus_reg::pipeline::Pipeline;
#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
use pelorus_reg::realtime::{RealtimePrediction, RealtimePredictorHandle};
#[cfg(feature = "hiroz")]
use pelorus_syncio::impl_hiroz;
#[cfg(feature = "r2r")]
use pelorus_syncio::impl_r2r;
#[cfg(feature = "mt_pubsub")]
use pelorus_syncio::impl_ros2_interfaces_jazzy_rkyv;
#[cfg(feature = "ros2-client")]
use ros2_interfaces_jazzy_serde::{
    builtin_interfaces::msg::Time,
    geometry_msgs::msg::{Point, PoseStamped, Quaternion, Transform, TransformStamped, Vector3},
    nav_msgs::msg::{Odometry, Path},
    sensor_msgs::msg::PointCloud2,
    std_msgs::msg::Header,
};
#[cfg(feature = "r2r")]
use std::ffi::OsString;
#[cfg(feature = "rat")]
use std::fs;
#[cfg(all(feature = "rat", unix))]
use std::os::unix::process::CommandExt;
#[cfg(feature = "rat")]
use std::process::{Command as ProcessCommand, Stdio};
#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
const PATH_LIMIT: usize = 10_000;
#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
const MCAP_PATH_LIMIT: usize = 512;
#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
const MCAP_PATH_MIN_PERIOD_NS: u64 = 1_000_000_000;
#[cfg(any(
    feature = "rat",
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
static PROCESS_SIGNAL_SHUTDOWN: AtomicBool = AtomicBool::new(false);
#[cfg(all(feature = "rat", unix))]
static ACTIVE_MINOT_PGID: AtomicI32 = AtomicI32::new(0);
const DEFAULT_OFFLINE_PREFETCH_SCANS: usize = 4;

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
use ros_pointcloud2::prelude::*;

#[global_allocator]
#[cfg(feature = "mimalloc")]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{Parser, Subcommand};
#[allow(unused_imports)]
use futures_lite::future;
#[cfg(feature = "hiroz")]
use hiroz::{Builder, context::ZContextBuilder};
#[cfg(feature = "hiroz")]
use hiroz_msgs::{
    builtin_interfaces::Time,
    geometry_msgs::{Point, PoseStamped, Quaternion, Transform, TransformStamped, Vector3},
    nav_msgs::{Odometry, Path},
    sensor_msgs::PointCloud2,
    std_msgs::{Header, String as RosString},
};
use log::{debug, error, info, warn};
#[cfg(feature = "mt_pubsub")]
use mt_pubsub::Node;
#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
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

/// Publishes registered clouds using the negotiated encoding.
///
/// Both publishers must exist before registration releases managed replay.
#[cfg(feature = "mt_pubsub")]
struct MtCloudPublisher {
    plain: mt_pubsub::Publisher<PointCloud2>,
    #[cfg(feature = "compressed-pointcloud")]
    compressed: Option<mt_pubsub::Publisher<pelorus_protocol::CompressedPointCloud2>>,
    #[cfg(feature = "compressed-pointcloud")]
    use_compressed: Arc<AtomicBool>,
}

#[cfg(feature = "mt_pubsub")]
impl MtCloudPublisher {
    async fn publish(&self, cloud: &PointCloud2) -> anyhow::Result<()> {
        #[cfg(feature = "compressed-pointcloud")]
        if self.use_compressed.load(Ordering::Acquire) {
            let publisher = self.compressed.as_ref().ok_or_else(|| {
                anyhow!("compressed registered clouds were negotiated but no publisher exists")
            })?;
            let cloud = impl_ros2_interfaces_jazzy_rkyv::to_pointcloud2_msg(cloud.clone());
            let compressed = pelorus_syncio::compress_cloud(cloud)?;
            return publisher.publish(&compressed).await;
        }
        self.plain.publish(cloud).await
    }
}

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
use pelorus_syncio::pelorus_sync::{ImuSyncMode, LocatedBufferView};
#[cfg(feature = "ros2-client")]
use pelorus_syncio::tf::TfBroadcaster;
#[cfg(feature = "hiroz")]
use pelorus_syncio::tf_hiroz::TfBroadcaster;
#[cfg(feature = "mt_pubsub")]
use pelorus_syncio::tf_lh::TfBroadcaster;
#[cfg(feature = "r2r")]
use pelorus_syncio::tf_r2r::TfBroadcaster;

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
use pelorus_reg::Registration;
#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
use pelorus_syncio::{SyncNode, SyncNodeInputs};
use std::path::PathBuf;

use pelorus_cli::mcap_output::{McapOutputWriter, McapTopics};
use pelorus_cli::output_target::{normalize_mcap_topic, prepare_output_targets, rosbag_mcap_path};
use pelorus_cli::pubsub::PubSubCoordMode;
use pelorus_cli::raw::RawSensor;
#[cfg(feature = "rat")]
use pelorus_cli::stats_tui::relay_child_line;
use pelorus_cli::stats_tui::{
    SharedStats, StatsTui, init_logger, set_tui_active, update_log_level,
};
#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz",
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

fn is_config_url(location: &str) -> bool {
    location.starts_with("http://") || location.starts_with("https://")
}

fn read_config_source(location: &std::path::Path) -> anyhow::Result<String> {
    if let Some(url) = location.to_str().filter(|location| is_config_url(location)) {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(30)))
            .build()
            .into();
        return agent
            .get(url)
            .call()
            .map_err(|error| anyhow!("Failed to fetch configuration '{url}': {error}"))?
            .body_mut()
            .read_to_string()
            .map_err(|error| anyhow!("Failed to read configuration '{url}': {error}"));
    }
    std::fs::read_to_string(location).map_err(|error| {
        anyhow!(
            "Failed to read configuration '{}': {error}",
            location.display()
        )
    })
}

fn load_user_config_with_checked_overrides(
    defaults: &str,
    location: Option<&std::path::Path>,
    overrides: &[pelorus_core::config::ConfigOverride],
) -> anyhow::Result<ratslang::Evaluated> {
    let source = location
        .map(read_config_source)
        .transpose()?
        .unwrap_or_default();
    let parent = location
        .filter(|location| !location.to_str().is_some_and(is_config_url))
        .and_then(|location| location.parent())
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    user_config_source_with_checked_overrides(defaults, &source, parent, overrides).map_err(
        |error| match location {
            Some(location) => anyhow!(
                "Failed to compile configuration '{}': {error}",
                location.display()
            ),
            None => anyhow!("Failed to compile command-line overrides: {error}"),
        },
    )
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    pub command: Command,

    /// Override one setting as KEY=VALUE. Repeatable.
    ///
    /// Applied after any --config-path file, so it wins over both the file and
    /// the embedded defaults. Values use Ratslang syntax, units included:
    /// -o mapping.voxel_size=0.5m -o output.odometry_rate=ImuAndLidar
    ///
    /// Unknown keys are rejected.
    #[arg(
        short = 'o',
        long = "override",
        global = true,
        value_name = "KEY=VALUE"
    )]
    pub overrides: Vec<String>,
}

#[derive(Debug, Clone, Subcommand)]
#[command()]
pub enum Command {
    /// Spawn the server process and wait for input using the compiled data provider
    Online(OnlineArgs),

    /// Use a Bagfile directly and run synchronously
    #[cfg(feature = "rat")]
    Offline(OfflineArgs),

    /// Create or validate configuration files
    #[command(subcommand)]
    Config(ConfigCommand),

    /// Install shell completions, or print them for a specific shell
    ///
    /// Run without a shell to detect the current shell and install completions.
    /// Example: `pelorus completions bash > pelorus_completions.sh`
    Completions {
        /// The shell whose completion script should be printed
        #[arg(value_enum)]
        shell: Option<Shell>,
    },
}

/// Arguments for the odometry pipeline and its online subcommands.
#[derive(Debug, Clone, Parser)]
#[command(args_conflicts_with_subcommands = true)]
pub struct OnlineArgs {
    #[command(subcommand)]
    pub mode: Option<OnlineMode>,

    #[command(flatten)]
    pub pipeline: ConfigArgs,
}

#[derive(Debug, Clone, Subcommand)]
pub enum OnlineMode {
    /// Publish the native sensor drivers onto the configured input topics
    /// without running any processing.
    ///
    /// Samples are published unchanged on `lidar.topic` and `imu.topic`.
    #[cfg(feature = "mt_pubsub")]
    Raw(RawArgs),
}

#[derive(Debug, Clone, Parser)]
pub struct RawArgs {
    /// Which sensor to publish. Defaults to every configured native driver.
    #[arg(value_enum, default_value_t = RawSensor::All)]
    pub sensor: RawSensor,

    /// Optional local path or HTTP(S) URL to a custom '.rl' configuration.
    /// If not provided, only the default configuration will be used
    #[arg(short, long)]
    pub config_path: Option<PathBuf>,

    /// Restrict Pelorus pub/sub communication to this host.
    #[arg(long, default_value_t = false)]
    pub local_only: bool,

    /// Pub/sub coordination. Must match the mode of the nodes Pelorus talks to.
    #[arg(long, value_enum, default_value_t = PubSubCoordMode::Peer)]
    pub coord_mode: PubSubCoordMode,

    #[command(flatten)]
    pub recording: RawRecordingArgs,

    /// Answer yes to overwrite prompts.
    #[arg(short = 'y', long, global = true, default_value_t = false)]
    pub yes: bool,

    #[command(subcommand)]
    pub mode: Option<RawMode>,
}

#[derive(Debug, Clone, Subcommand)]
pub enum RawMode {
    /// Walk through the orientations an IMU intrinsic calibration needs.
    ///
    /// Records prompted static poses and offers to fit them with `dedrunk`.
    Intrinsics(IntrinsicsArgs),
}

#[derive(Debug, Clone, Parser)]
pub struct IntrinsicsArgs {
    /// How many static orientations to walk through.
    ///
    /// Nine is the minimum for the full scale-and-misalignment model. More,
    /// spread over all octants, is better.
    #[arg(long, default_value_t = 12)]
    pub poses: usize,
}

/// Recording targets for raw sensor data.
#[derive(Debug, Clone, Parser)]
pub struct RawRecordingArgs {
    // Unique ids prevent clap from sharing defaults with `RecordingArgs`.
    /// Write the raw sensor output to this MCAP bagfile.
    #[arg(
        id = "raw_output_mcap",
        long = "output-mcap",
        global = true,
        value_name = "PATH",
        conflicts_with = "raw_output_rosbag"
    )]
    pub output_mcap: Option<PathBuf>,

    /// Also write ROS 2 bag metadata.yaml for the MCAP output.
    #[arg(
        id = "raw_output_rosbag_metadata",
        long = "output-rosbag-metadata",
        global = true,
        value_name = "PATH",
        requires = "raw_output_mcap"
    )]
    pub output_rosbag_metadata: Option<PathBuf>,

    /// Write the raw sensor output as a ROS 2 MCAP bag directory.
    #[arg(
        id = "raw_output_rosbag",
        long = "output-rosbag",
        global = true,
        value_name = "DIR",
        conflicts_with_all = ["raw_output_mcap", "raw_output_rosbag_metadata"]
    )]
    pub output_rosbag: Option<PathBuf>,

    /// MCAP topic for the raw PointCloud2 output. Defaults to the configured
    /// `lidar.topic`.
    #[arg(id = "raw_mcap_cloud_topic", long = "mcap-cloud-topic", global = true)]
    pub mcap_cloud_topic: Option<String>,

    /// MCAP topic for the raw IMU output. Defaults to the configured
    /// `imu.topic`.
    #[arg(id = "raw_mcap_imu_topic", long = "mcap-imu-topic", global = true)]
    pub mcap_imu_topic: Option<String>,
}

#[derive(Debug, Clone, Parser)]
pub struct OfflineArgs {
    /// Optional local path or HTTP(S) URL to a custom '.rl' configuration.
    /// If not provided, only the default configuration will be used
    #[arg(short, long)]
    pub config_path: Option<PathBuf>,

    /// Local bagfile path (file or directory), or a Marina dataset reference for Minot to stream.
    pub bagfile_path: PathBuf,

    /// Marina registry to use when BAGFILE_PATH is a remote dataset reference.
    #[arg(long)]
    pub registry: Option<String>,

    /// Stream a remote dataset without caching blocks or promoting it locally.
    #[arg(long)]
    pub no_cache_stream: bool,

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

    /// Export end-to-end latency samples as timestamp and nanoseconds:
    /// wall-clock time from the arrival of the scan message until its
    /// registered cloud is ready, including queueing and the wait for IMU
    /// measurements that cover the scan.
    /// If not provided, latencies are not exported.
    /// If provided without value, automatically generates a filename.
    /// If provided with a path, exports to that exact location.
    #[cfg(feature = "processing-time-export")]
    #[arg(
        long,
        value_name = "PATH",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    pub export_latency: Option<String>,

    /// Export processing time samples as timestamp and nanoseconds: the
    /// summed top-level profiler time actually spent computing the tick.
    /// Unlike --export-latency, this excludes queueing and waiting.
    /// If not provided, processing times are not exported.
    /// If provided without value, automatically generates a filename.
    /// If provided with a path, exports to that exact location.
    #[cfg(feature = "processing-time-export")]
    #[arg(
        long,
        value_name = "PATH",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    pub export_processing_time: Option<String>,

    /// Allow Minot and Pelorus pub/sub to use non-local networking in offline mode.
    #[arg(long, default_value_t = false)]
    pub open_router: bool,

    /// Pub/sub coordination of Pelorus and of the Minot turbine it spawns.
    #[arg(long, value_enum, default_value_t = PubSubCoordMode::Peer)]
    pub coord_mode: PubSubCoordMode,

    /// Answer yes to overwrite prompts.
    #[arg(short = 'y', long, default_value_t = false)]
    pub yes: bool,

    /// Number of offline scans to request ahead from Minot.
    ///
    /// Registration and publishing remain gated one scan at a time. This only
    /// keeps the preprocessing side fed while the current scan is registering.
    #[arg(long, default_value_t = DEFAULT_OFFLINE_PREFETCH_SCANS)]
    pub prefetch_scans: usize,

    /// Let Polarstern acknowledge each completed registered frame before the
    /// next steady-state offline scan is released.
    #[arg(long, default_value_t = false)]
    pub polarstern: bool,

    /// Also replay an extra bag topic, for consumers outside Pelorus such as
    /// Polarstern's camera colorization. `TYPE:TOPIC` with TYPE one of `image`,
    /// `compressed` or `camera-info`. Repeatable.
    ///
    /// Forwarded topics follow the TF schedule: they are published in bag order
    /// together with the IMU and TF around each cloud, and never hold a scan.
    #[arg(long = "forward-topic", value_name = "TYPE:TOPIC", value_parser = parse_forwarded_topic)]
    pub forward_topics: Vec<ForwardedTopic>,
}

/// A bag topic replayed next to the sensors Pelorus itself consumes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardedTopic {
    /// Minot's `_type` for the topic.
    pub minot_type: &'static str,
    pub topic: String,
}

fn parse_forwarded_topic(value: &str) -> Result<ForwardedTopic, String> {
    let (kind, topic) = value
        .split_once(':')
        .ok_or_else(|| format!("'{value}' is not TYPE:TOPIC"))?;
    let minot_type = match kind {
        "image" => "Image",
        "compressed" => "ImageCompressed",
        "camera-info" => "CameraInfo",
        other => {
            return Err(format!(
                "unknown topic type '{other}'; expected image, compressed or camera-info"
            ));
        }
    };
    if topic.is_empty() || topic.contains('"') || topic.contains(char::is_whitespace) {
        return Err(format!("'{topic}' is not a usable topic name"));
    }
    Ok(ForwardedTopic {
        minot_type,
        topic: topic.to_owned(),
    })
}

#[derive(Debug, Clone, Parser)]
pub struct ConfigArgs {
    /// Optional local path or HTTP(S) URL to a custom '.rl' configuration.
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

    /// Export end-to-end latency samples as timestamp and nanoseconds:
    /// wall-clock time from the arrival of the scan message until its
    /// registered cloud is ready, including queueing and the wait for IMU
    /// measurements that cover the scan.
    /// If not provided, latencies are not exported.
    /// If provided without value, automatically generates a filename.
    /// If provided with a path, exports to that exact location.
    #[cfg(feature = "processing-time-export")]
    #[arg(
        long,
        value_name = "PATH",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    pub export_latency: Option<String>,

    /// Export processing time samples as timestamp and nanoseconds: the
    /// summed top-level profiler time actually spent computing the tick.
    /// Unlike --export-latency, this excludes queueing and waiting.
    /// If not provided, processing times are not exported.
    /// If provided without value, automatically generates a filename.
    /// If provided with a path, exports to that exact location.
    #[cfg(feature = "processing-time-export")]
    #[arg(
        long,
        value_name = "PATH",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    pub export_processing_time: Option<String>,

    /// Publish tick_end checkpoint using ratlib from Minot.
    #[arg(short, long)]
    pub rat: bool,

    /// Restrict Pelorus pub/sub communication to this host.
    #[arg(long, default_value_t = false)]
    pub local_only: bool,

    /// Pub/sub coordination. Must match the mode of the nodes Pelorus talks to.
    #[arg(long, value_enum, default_value_t = PubSubCoordMode::Peer)]
    pub coord_mode: PubSubCoordMode,

    /// ROS node name used by the r2r backend.
    #[cfg(feature = "r2r")]
    #[arg(long, default_value = "pelorus")]
    pub node_name: String,

    /// ROS namespace used by the r2r backend.
    #[cfg(feature = "r2r")]
    #[arg(long, default_value = "pelorus")]
    pub namespace: String,

    /// Hiroz/Zenoh session mode.
    #[cfg(feature = "hiroz")]
    #[arg(long, default_value = "peer")]
    pub hiroz_mode: String,

    /// Hiroz/Zenoh endpoint to connect to. Can be passed multiple times.
    #[cfg(feature = "hiroz")]
    #[arg(long, value_name = "ENDPOINT")]
    pub hiroz_endpoint: Vec<String>,

    /// Answer yes to overwrite prompts.
    #[arg(short = 'y', long, default_value_t = false)]
    pub yes: bool,
}

#[cfg(feature = "r2r")]
#[derive(Debug, Clone)]
struct RosNodeArgs {
    node_name: String,
    namespace: String,
}

#[cfg(feature = "hiroz")]
#[derive(Debug, Clone)]
struct HirozNodeArgs {
    mode: String,
    endpoints: Vec<String>,
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

    /// Optional local path or HTTP(S) URL to a custom '.rl' configuration used to read topics.
    /// If not provided, the default configuration values are used
    #[arg(short, long)]
    pub config_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Parser)]
pub struct TestArgs {
    /// Local path or HTTP(S) URL of the configuration to test. Use '-' for stdin.
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

#[cfg(feature = "r2r")]
fn args_without_ros_args<I>(args: I) -> Vec<OsString>
where
    I: IntoIterator<Item = OsString>,
{
    let mut filtered = Vec::new();
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        if arg != "--ros-args" {
            filtered.push(arg);
            continue;
        }

        while let Some(arg) = args.next() {
            match arg.to_str() {
                Some("--") => break,
                Some(
                    "-r" | "--remap" | "-p" | "--param" | "--params-file" | "--log-level"
                    | "--enclave",
                ) => {
                    let _ = args.next();
                }
                Some(
                    "--disable-rosout-logs"
                    | "--disable-stdout-logs"
                    | "--disable-external-lib-logs"
                    | "--enable-rosout-logs"
                    | "--enable-stdout-logs"
                    | "--enable-external-lib-logs",
                ) => {}
                Some(value) if value.starts_with("__") || value.contains(":=") => {}
                Some(_) | None => {}
            }
        }
    }

    filtered
}

#[derive(Debug, Clone)]
struct McapOutputConfig {
    path: PathBuf,
    rosbag_metadata_path: Option<PathBuf>,
    topics: McapTopics,
}

fn build_mcap_output_config(config: &LioConfig) -> Option<McapOutputConfig> {
    build_mcap_output_config_with_filtered_topic(
        &config.mcap,
        &config.sync_node_params.imu_filtered_topic,
    )
    .map(|mut output| {
        if config.imu_params.lidar_only {
            output.topics.filtered_imu = None;
        }
        output
    })
}

fn build_mcap_output_config_with_filtered_topic(
    mcap: &pelorus_core::McapConfig,
    configured_filtered_imu_topic: &str,
) -> Option<McapOutputConfig> {
    mcap.file
        .as_ref()
        .map(PathBuf::from)
        .or_else(|| {
            mcap.rosbag_dir
                .as_ref()
                .map(|dir| rosbag_mcap_path(std::path::Path::new(dir)))
        })
        .map(|path| {
            let filtered_imu_topic = if configured_filtered_imu_topic.is_empty() {
                "imu/filtered".to_string()
            } else {
                configured_filtered_imu_topic.to_string()
            };
            McapOutputConfig {
                path,
                rosbag_metadata_path: mcap.rosbag_metadata.as_ref().map(PathBuf::from).or_else(
                    || {
                        mcap.rosbag_dir
                            .as_ref()
                            .map(|dir| std::path::Path::new(dir).join("metadata.yaml"))
                    },
                ),
                topics: McapTopics {
                    cloud: normalize_mcap_topic(mcap.cloud_topic.clone()),
                    odom: normalize_mcap_topic(mcap.odom_topic.clone()),
                    path: normalize_mcap_topic(mcap.path_topic.clone()),
                    tf: normalize_mcap_topic(mcap.tf_topic.clone()),
                    filtered_imu: Some(normalize_mcap_topic(filtered_imu_topic)),
                },
            }
        })
}

fn prepare_mcap_output_targets(
    mcap: &pelorus_core::McapConfig,
    output_mcap: Option<&McapOutputConfig>,
    assume_yes: bool,
) -> anyhow::Result<bool> {
    prepare_output_targets(
        mcap.rosbag_dir.as_deref().map(std::path::Path::new),
        output_mcap.map(|output| output.path.as_path()),
        output_mcap.and_then(|output| output.rosbag_metadata_path.as_deref()),
        assume_yes,
    )
}

#[cfg(any(
    feature = "rat",
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
extern "C" fn handle_shutdown_signal(signal: libc::c_int) {
    if PROCESS_SIGNAL_SHUTDOWN.swap(true, Ordering::SeqCst) {
        #[cfg(all(feature = "rat", unix))]
        {
            let pgid = ACTIVE_MINOT_PGID.load(Ordering::SeqCst);
            if pgid > 0 {
                unsafe {
                    libc::kill(-pgid, libc::SIGKILL);
                }
            }
        }

        if signal == libc::SIGINT || signal == libc::SIGTERM {
            unsafe {
                libc::signal(signal, libc::SIG_DFL);
                libc::raise(signal);
            }
        }
    }
}

#[cfg(any(
    feature = "rat",
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
fn install_shutdown_signal_handlers() {
    PROCESS_SIGNAL_SHUTDOWN.store(false, Ordering::SeqCst);
    unsafe {
        libc::signal(
            libc::SIGINT,
            handle_shutdown_signal as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            handle_shutdown_signal as *const () as libc::sighandler_t,
        );
    }
}

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
fn install_online_shutdown_signal_handler(shutdown_flag: Arc<AtomicBool>) {
    install_shutdown_signal_handlers();
    std::thread::spawn(move || {
        while !shutdown_flag.load(Ordering::Relaxed) {
            if PROCESS_SIGNAL_SHUTDOWN.load(Ordering::SeqCst) {
                shutdown_flag.store(true, Ordering::SeqCst);
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    });
}

#[cfg(not(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
)))]
fn install_online_shutdown_signal_handler(_shutdown_flag: Arc<AtomicBool>) {}

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
fn copy_array<const N: usize>(values: &[f64]) -> [f64; N] {
    let mut out = [0.0; N];
    out.copy_from_slice(&values[..N]);
    out
}

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
fn serde_time(time: &Time) -> ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time {
    ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time {
        sec: time.sec,
        nanosec: time.nanosec,
    }
}

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
fn serde_header(header: &Header) -> ros2_interfaces_jazzy_serde::std_msgs::msg::Header {
    ros2_interfaces_jazzy_serde::std_msgs::msg::Header {
        stamp: serde_time(&header.stamp),
        frame_id: header.frame_id.clone(),
    }
}

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
fn serde_vector3(vector: &Vector3) -> ros2_interfaces_jazzy_serde::geometry_msgs::msg::Vector3 {
    ros2_interfaces_jazzy_serde::geometry_msgs::msg::Vector3 {
        x: vector.x,
        y: vector.y,
        z: vector.z,
    }
}

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
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

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
fn serde_point(point: &Point) -> ros2_interfaces_jazzy_serde::geometry_msgs::msg::Point {
    ros2_interfaces_jazzy_serde::geometry_msgs::msg::Point {
        x: point.x,
        y: point.y,
        z: point.z,
    }
}

#[cfg(all(
    any(feature = "mt_pubsub", feature = "ros2-client", feature = "r2r"),
    not(feature = "hiroz")
))]
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

#[cfg(feature = "hiroz")]
fn serde_cloud(cloud: &PointCloud2) -> ros2_interfaces_jazzy_serde::sensor_msgs::msg::PointCloud2 {
    let cloud = impl_hiroz::to_pointcloud2_msg(cloud.clone());
    ros2_interfaces_jazzy_serde::sensor_msgs::msg::PointCloud2 {
        header: ros2_interfaces_jazzy_serde::std_msgs::msg::Header {
            stamp: ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time {
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
            .map(
                |field| ros2_interfaces_jazzy_serde::sensor_msgs::msg::PointField {
                    name: field.name.to_string(),
                    offset: field.offset,
                    datatype: field.datatype,
                    count: field.count,
                },
            )
            .collect(),
        is_bigendian: matches!(cloud.endian, ros_pointcloud2::Endian::Big),
        point_step: cloud.point_step,
        row_step: cloud.row_step,
        data: cloud.data,
        is_dense: matches!(cloud.dense, ros_pointcloud2::Denseness::Dense),
    }
}

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
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

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
fn realtime_prediction_to_odom(
    prediction: &RealtimePrediction,
    map_frame: &str,
    child_frame: &str,
) -> Odometry {
    let mut odom = Odometry::default();
    odom.header.stamp = Time {
        sec: prediction.stamp.sec,
        nanosec: prediction.stamp.nanosec,
    };
    odom.header.frame_id = map_frame.to_owned();
    odom.child_frame_id = child_frame.to_owned();
    odom.pose.pose.position.x = prediction.pose.translation.x;
    odom.pose.pose.position.y = prediction.pose.translation.y;
    odom.pose.pose.position.z = prediction.pose.translation.z;
    odom.pose.pose.orientation.x = prediction.pose.rotation.i;
    odom.pose.pose.orientation.y = prediction.pose.rotation.j;
    odom.pose.pose.orientation.z = prediction.pose.rotation.k;
    odom.pose.pose.orientation.w = prediction.pose.rotation.w;
    odom.twist.twist.linear.x = prediction.linear_velocity_body.x;
    odom.twist.twist.linear.y = prediction.linear_velocity_body.y;
    odom.twist.twist.linear.z = prediction.linear_velocity_body.z;
    odom.twist.twist.angular.x = prediction.angular_velocity_body.x;
    odom.twist.twist.angular.y = prediction.angular_velocity_body.y;
    odom.twist.twist.angular.z = prediction.angular_velocity_body.z;
    #[cfg(feature = "r2r")]
    {
        odom.pose.covariance = prediction.pose_covariance.to_vec();
        odom.twist.covariance = prediction.twist_covariance.to_vec();
    }
    #[cfg(not(feature = "r2r"))]
    {
        odom.pose.covariance = prediction.pose_covariance;
        odom.twist.covariance = prediction.twist_covariance;
    }
    odom
}

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
fn realtime_prediction_to_tf(
    prediction: &RealtimePrediction,
    map_frame: &str,
    child_frame: &str,
) -> TransformStamped {
    TransformStamped {
        header: Header {
            frame_id: map_frame.to_owned(),
            stamp: Time {
                sec: prediction.stamp.sec,
                nanosec: prediction.stamp.nanosec,
            },
        },
        child_frame_id: child_frame.to_owned(),
        transform: Transform {
            translation: Vector3 {
                x: prediction.pose.translation.x,
                y: prediction.pose.translation.y,
                z: prediction.pose.translation.z,
            },
            rotation: Quaternion {
                x: prediction.pose.rotation.i,
                y: prediction.pose.rotation.j,
                z: prediction.pose.rotation.k,
                w: prediction.pose.rotation.w,
            },
        },
    }
}

#[cfg(all(
    feature = "rerun",
    any(
        feature = "mt_pubsub",
        feature = "ros2-client",
        feature = "r2r",
        feature = "hiroz"
    )
))]
fn log_realtime_prediction_to_rerun(rec: &rerun::RecordingStream, prediction: &RealtimePrediction) {
    rec.set_timestamp_secs_since_epoch("sensor", timestamp_to_secs(&prediction.stamp));
    let pose = &prediction.pose;
    let transform = rerun::Transform3D::from_translation(glam::Vec3::new(
        pose.translation.x as f32,
        pose.translation.y as f32,
        pose.translation.z as f32,
    ))
    .with_quaternion(glam::Quat::from_xyzw(
        pose.rotation.i as f32,
        pose.rotation.j as f32,
        pose.rotation.k as f32,
        pose.rotation.w as f32,
    ));
    if let Err(error) = rec.log("world/latest_odometry", &transform) {
        debug!("Failed to log real-time odometry to Rerun: {error}");
    }

    let covariance = &prediction.pose_covariance;
    let position_covariance_world = nalgebra::Matrix3::new(
        covariance[0],
        covariance[1],
        covariance[2],
        covariance[6],
        covariance[7],
        covariance[8],
        covariance[12],
        covariance[13],
        covariance[14],
    );
    let pose_rotation = pose.rotation.to_rotation_matrix();
    let position_covariance_local =
        pose_rotation.matrix().transpose() * position_covariance_world * pose_rotation.matrix();
    let eigen = position_covariance_local.symmetric_eigen();
    let mut eigenvectors = eigen.eigenvectors;
    if eigenvectors.determinant() < 0.0 {
        eigenvectors.set_column(0, &(-eigenvectors.column(0)));
    }
    let half_sizes = glam::Vec3::new(
        eigen.eigenvalues.x.max(0.0).sqrt() as f32,
        eigen.eigenvalues.y.max(0.0).sqrt() as f32,
        eigen.eigenvalues.z.max(0.0).sqrt() as f32,
    );
    let covariance_rotation = UnitQuaternion::from_matrix(&eigenvectors);
    let covariance_quaternion = covariance_rotation.quaternion();
    let ellipsoid = rerun::Ellipsoids3D::update_fields()
        .with_half_sizes([half_sizes])
        .with_quaternions([glam::Quat::from_xyzw(
            covariance_quaternion.i as f32,
            covariance_quaternion.j as f32,
            covariance_quaternion.k as f32,
            covariance_quaternion.w as f32,
        )])
        .with_colors([rerun::Color::from_unmultiplied_rgba(64, 180, 255, 96)])
        .with_fill_mode(rerun::components::FillMode::TransparentFillMajorWireframe)
        .with_line_radii([rerun::Radius::new_ui_points(1.0)]);
    if let Err(error) = rec.log("world/latest_odometry/covariance", &ellipsoid) {
        debug!("Failed to log real-time covariance to Rerun: {error}");
    }
}

/// Color for a relevance score in `[0, 1]`, shared by every backend's rerun
/// map snapshot so the gradient does not depend on which backend happens to
/// be active. Warm (evicted soon) to cool (most relevant).
#[cfg(feature = "rerun")]
fn relevance_color(score: f32) -> rerun::Color {
    let value = score.clamp(0.0, 1.0);
    if value < 0.5 {
        let t = value * 2.0;
        rerun::Color::from_rgb(
            (150.0 + 95.0 * t) as u8,
            (58.0 + 116.0 * t) as u8,
            (70.0 - 20.0 * t) as u8,
        )
    } else {
        let t = (value - 0.5) * 2.0;
        rerun::Color::from_rgb(
            (245.0 - 171.0 * t) as u8,
            (174.0 + 42.0 * t) as u8,
            (50.0 + 166.0 * t) as u8,
        )
    }
}

/// Publishes one multiresolution level of the CPU registration map to Rerun
/// as ellipsoids, colored by the same recency-based relevance score as the
/// flat map (`log_map_snapshot_to_rerun`). Each level gets its own entity
/// path so it can be toggled independently in Rerun.
#[cfg(all(
    feature = "rerun",
    not(any(feature = "cuda", feature = "metal", feature = "vulkan"))
))]
fn log_multires_snapshot_to_rerun(
    rec: &rerun::RecordingStream,
    distributions: Vec<(pelorus_localmap::localmap::VoxelDistributionLru, u8)>,
    map_config: &pelorus_core::RerunMapConfig,
) {
    const MIN_HALF_SIZE: f32 = 0.02;
    const LEVELS: usize = 4;
    let mut centers: [Vec<glam::Vec3>; LEVELS] = Default::default();
    let mut half_sizes: [Vec<glam::Vec3>; LEVELS] = Default::default();
    let mut quaternions: [Vec<glam::Quat>; LEVELS] = Default::default();
    let mut colors: [Vec<rerun::Color>; LEVELS] = Default::default();

    for (scored, level) in &distributions {
        let level = (*level as usize).min(LEVELS - 1);
        let distribution = &scored.distribution;
        let mut eigen = distribution.covariance.symmetric_eigen();
        if eigen.eigenvectors.determinant() < 0.0 {
            let flipped = -eigen.eigenvectors.column(0);
            eigen.eigenvectors.set_column(0, &flipped);
        }
        let half_size = glam::Vec3::new(
            (map_config.visual_scale * eigen.eigenvalues.x.max(0.0).sqrt()).max(MIN_HALF_SIZE),
            (map_config.visual_scale * eigen.eigenvalues.y.max(0.0).sqrt()).max(MIN_HALF_SIZE),
            (map_config.visual_scale * eigen.eigenvalues.z.max(0.0).sqrt()).max(MIN_HALF_SIZE),
        );
        let rotation = UnitQuaternion::from_matrix(&eigen.eigenvectors);
        let q = rotation.quaternion();

        centers[level].push(glam::Vec3::new(
            distribution.mean.x,
            distribution.mean.y,
            distribution.mean.z,
        ));
        half_sizes[level].push(half_size);
        quaternions[level].push(glam::Quat::from_xyzw(q.i, q.j, q.k, q.w));
        colors[level].push(relevance_color(scored.lru_score));
    }

    for level in 0..LEVELS {
        let ellipsoids = rerun::Ellipsoids3D::update_fields()
            .with_centers(core::mem::take(&mut centers[level]))
            .with_half_sizes(core::mem::take(&mut half_sizes[level]))
            .with_quaternions(core::mem::take(&mut quaternions[level]))
            .with_colors(core::mem::take(&mut colors[level]))
            .with_fill_mode(rerun::components::FillMode::Solid);
        if let Err(err) = rec.log(format!("world/map/level{level}"), &ellipsoids) {
            error!("Error sending map snapshot to rerun {err}");
        }
    }
}

/// Publishes the CPU registration map to Rerun as ellipsoids.
/// Color goes from warm (evicted soon) to cool (most relevant), by LRU score.
#[cfg(all(
    feature = "rerun",
    not(any(feature = "cuda", feature = "metal", feature = "vulkan"))
))]
fn log_map_snapshot_to_rerun(
    rec: &rerun::RecordingStream,
    distributions: Vec<pelorus_localmap::localmap::VoxelDistributionLru>,
    map_config: &pelorus_core::RerunMapConfig,
) {
    const MIN_HALF_SIZE: f32 = 0.02;
    let mut centers = Vec::with_capacity(distributions.len());
    let mut half_sizes = Vec::with_capacity(distributions.len());
    let mut quaternions = Vec::with_capacity(distributions.len());
    let mut colors = Vec::with_capacity(distributions.len());

    for scored in &distributions {
        let distribution = &scored.distribution;
        let mut eigen = distribution.covariance.symmetric_eigen();
        if eigen.eigenvectors.determinant() < 0.0 {
            let flipped = -eigen.eigenvectors.column(0);
            eigen.eigenvectors.set_column(0, &flipped);
        }
        let half_size = glam::Vec3::new(
            (map_config.visual_scale * eigen.eigenvalues.x.max(0.0).sqrt()).max(MIN_HALF_SIZE),
            (map_config.visual_scale * eigen.eigenvalues.y.max(0.0).sqrt()).max(MIN_HALF_SIZE),
            (map_config.visual_scale * eigen.eigenvalues.z.max(0.0).sqrt()).max(MIN_HALF_SIZE),
        );
        let rotation = UnitQuaternion::from_matrix(&eigen.eigenvectors);
        let q = rotation.quaternion();

        centers.push(glam::Vec3::new(
            distribution.mean.x,
            distribution.mean.y,
            distribution.mean.z,
        ));
        half_sizes.push(half_size);
        quaternions.push(glam::Quat::from_xyzw(q.i, q.j, q.k, q.w));
        colors.push(relevance_color(scored.lru_score));
    }

    let ellipsoids = rerun::Ellipsoids3D::update_fields()
        .with_centers(centers)
        .with_half_sizes(half_sizes)
        .with_quaternions(quaternions)
        .with_colors(colors)
        .with_fill_mode(rerun::components::FillMode::Solid);

    if let Err(err) = rec.log("world/map", &ellipsoids) {
        error!("Error sending map snapshot to rerun {err}");
    }
}

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
fn drain_realtime_odometry_exports(
    receiver: &Option<std::sync::mpsc::Receiver<RealtimePrediction>>,
    trajectory_samples: &mut Option<Vec<(f64, Isometry3<f64>)>>,
    mcap_writer: &mut Option<McapOutputWriter<std::io::BufWriter<std::fs::File>>>,
    #[cfg(feature = "rerun")] rerun_rec: Option<&Arc<rerun::RecordingStream>>,
    map_frame: &str,
    child_frame: &str,
) -> anyhow::Result<()> {
    let Some(receiver) = receiver.as_ref() else {
        return Ok(());
    };

    while let Ok(prediction) = receiver.try_recv() {
        #[cfg(feature = "rerun")]
        if let Some(rerun_rec) = rerun_rec {
            log_realtime_prediction_to_rerun(rerun_rec, &prediction);
        }
        if let Some(samples) = trajectory_samples.as_mut() {
            let stamp = timestamp_to_secs(&prediction.stamp);
            samples.push((stamp, prediction.pose));
        }

        if let Some(writer) = mcap_writer.as_mut() {
            let odom = realtime_prediction_to_odom(&prediction, map_frame, child_frame);
            let tf = realtime_prediction_to_tf(&prediction, map_frame, child_frame);
            writer.write_odom(&serde_odom(&odom))?;
            writer.write_tf(&serde_tf_message(&tf))?;
        }
    }

    Ok(())
}

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
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

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
fn serde_path(path: &Path) -> ros2_interfaces_jazzy_serde::nav_msgs::msg::Path {
    ros2_interfaces_jazzy_serde::nav_msgs::msg::Path {
        header: serde_header(&path.header),
        poses: path.poses.iter().map(serde_pose_stamped).collect(),
    }
}

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
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

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
fn drain_filtered_imu_mcap(
    writer: &mut Option<McapOutputWriter<std::io::BufWriter<std::fs::File>>>,
    rx: &mut Option<std::sync::mpsc::Receiver<pelorus_core::ImuMsg>>,
) -> anyhow::Result<()> {
    let (Some(writer), Some(rx)) = (writer.as_mut(), rx.as_mut()) else {
        return Ok(());
    };

    while let Ok(msg) = rx.try_recv() {
        writer.write_filtered_imu(&pelorus_cli::mcap_output::imu_msg_to_ros(&msg))?;
    }
    Ok(())
}

/// Everything [`lio_with_ready_signal`] needs besides the [`LioConfig`].
struct LioRunOptions {
    ready_signal: Option<std::sync::mpsc::Sender<()>>,
    export_trajectory: Option<String>,
    #[cfg(feature = "processing-time-export")]
    export_latency: Option<String>,
    #[cfg(feature = "processing-time-export")]
    export_processing_time: Option<String>,
    synchronous_mode: bool,
    #[cfg(feature = "r2r")]
    ros_node_args: RosNodeArgs,
    #[cfg(feature = "hiroz")]
    hiroz_node_args: HirozNodeArgs,
    shared_stats: Option<Arc<SharedStats>>,
    /// When set to true, the LIO loop exits.
    shutdown_flag: Option<Arc<AtomicBool>>,
    /// Blocking shutdown signal for pipeline teardown mode.
    shutdown_request_rx: Option<std::sync::mpsc::Receiver<OfflineShutdownRequest>>,
    offline_prefetch_scans: usize,
    replay_tick_owner: ReplayTickOwner,
    /// Fired once the downstream consumer has registered, so the offline runner
    /// can hold the bag until Polarstern's subscribers exist.
    downstream_registered: Option<std::sync::mpsc::Sender<()>>,
    mt_pubsub_local_only: bool,
    mt_pubsub_coord_mode: PubSubCoordMode,
    output_mcap: Option<McapOutputConfig>,
}

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz",
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

#[cfg(all(
    feature = "processing-time-export",
    any(
        feature = "mt_pubsub",
        feature = "ros2-client",
        feature = "r2r",
        feature = "hiroz"
    )
))]
fn resolve_export_timing_path(export_path: Option<String>, kind: &str) -> Option<String> {
    match export_path {
        Some(path) if path.is_empty() => {
            let name = petname::petname(2, "_").unwrap_or_else(|| "unnamed".to_string());
            Some(format!(
                "pelorus_{}_{}_{}.tum",
                kind,
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

/// Who releases the steady-state offline replay trigger.
///
/// Everything outside managed downstream replay uses [`ReplayTickOwner::Pelorus`],
/// which is the unchanged standalone `pelorus offline` clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayTickOwner {
    /// Pelorus advances Minot itself after every scan it consumes.
    Pelorus,
    /// A registered downstream consumer advances Minot after every frame
    /// Pelorus publishes. Pelorus only advances for scans it consumes without
    /// publishing an authoritative frame, plus the mid-frame input it needs to
    /// close the current scan. Backpressure is structural: no trigger, no input.
    Downstream,
}

impl ReplayTickOwner {
    fn is_downstream(self) -> bool {
        matches!(self, Self::Downstream)
    }
}

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
fn release_minot_tick(
    shutdown_flag: Option<&Arc<AtomicBool>>,
    #[allow(unused_variables)] synchronous_mode: bool,
) {
    release_minot_ticks(shutdown_flag, synchronous_mode, 1);
}

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
fn release_minot_ticks(
    shutdown_flag: Option<&Arc<AtomicBool>>,
    #[allow(unused_variables)] synchronous_mode: bool,
    #[allow(unused_variables)] count: usize,
) {
    shutdown_flag
        .map(|flag| flag.load(Ordering::Relaxed))
        .unwrap_or(false);

    #[cfg(feature = "rat")]
    if synchronous_mode {
        for _ in 0..count {
            // `tick_end` is the actual action in the generated offline Minot
            // program. `pelorus:sd_end` has no matching rule there and can
            // block forever after a long downstream pause, preventing this
            // real replay action from ever being reached.
            let mut tick = String::new();
            rat::bacon("tick_end", &mut tick, rat::VariableType::StaticOnly).unwrap();
        }
    }
}

#[cfg(feature = "mt_pubsub")]
async fn release_minot_ticks_async(
    shutdown_flag: Option<&Arc<AtomicBool>>,
    #[allow(unused_variables)] synchronous_mode: bool,
    #[allow(unused_variables)] count: usize,
) -> anyhow::Result<()> {
    if shutdown_flag
        .map(|flag| flag.load(Ordering::Relaxed))
        .unwrap_or(false)
    {
        return Ok(());
    }

    #[cfg(feature = "rat")]
    if synchronous_mode {
        for _ in 0..count {
            let mut tick = String::new();
            rat::bacon_async("tick_end", &mut tick, rat::VariableType::StaticOnly).await?;
        }
    }

    Ok(())
}

/// Settle a replay refill against triggers already pulled mid-frame.
///
/// Preprocessing releases its own Minot triggers when it needs more input to
/// close a frame (see `ImuSyncMode::DemandDrivenReplay` below), each one
/// recorded as debt. Those triggers already advanced the bag, so the next
/// refill must ask for that many fewer scans or the bag runs ahead. Returns
/// what still has to be requested after cancelling as much debt as `requested`
/// covers. Leftover debt carries to later refills.
#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
fn request_count_after_replay_debt(debt: &AtomicUsize, requested: usize) -> usize {
    let previous = debt
        .try_update(Ordering::AcqRel, Ordering::Acquire, |outstanding| {
            Some(outstanding.saturating_sub(requested))
        })
        .expect("replay tick debt update always succeeds");
    requested.saturating_sub(previous)
}

#[cfg(all(
    test,
    any(
        feature = "mt_pubsub",
        feature = "ros2-client",
        feature = "r2r",
        feature = "hiroz"
    )
))]
mod replay_tick_debt_tests {
    use super::*;

    #[test]
    fn demand_ticks_are_deducted_from_later_refills() {
        let debt = AtomicUsize::new(2);

        assert_eq!(request_count_after_replay_debt(&debt, 4), 2);
        assert_eq!(debt.load(Ordering::Acquire), 0);
    }

    #[test]
    fn demand_debt_carries_across_small_refills() {
        let debt = AtomicUsize::new(3);

        assert_eq!(request_count_after_replay_debt(&debt, 1), 0);
        assert_eq!(debt.load(Ordering::Acquire), 2);
    }
}

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
#[cfg(any(feature = "mt_pubsub", feature = "hiroz"))]
fn stats_message(stats: &SharedStats) -> RosString {
    let snapshot = stats.get();
    RosString {
        data: format!(
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
        ),
    }
}

fn lio_with_ready_signal(
    mut config: LioConfig,
    options: LioRunOptions,
) -> Result<(), Box<dyn CoreError>> {
    let LioRunOptions {
        ready_signal,
        export_trajectory,
        #[cfg(feature = "processing-time-export")]
        export_latency,
        #[cfg(feature = "processing-time-export")]
        export_processing_time,
        #[allow(unused_variables)]
        synchronous_mode,
        #[cfg(feature = "r2r")]
        ros_node_args,
        #[cfg(feature = "hiroz")]
        hiroz_node_args,
        shared_stats,
        shutdown_flag,
        shutdown_request_rx,
        offline_prefetch_scans,
        replay_tick_owner,
        #[allow(unused_variables)]
        downstream_registered,
        #[allow(unused_variables)]
        mt_pubsub_local_only,
        #[allow(unused_variables)]
        mt_pubsub_coord_mode,
        output_mcap,
    } = options;
    let replay_tick_debt = Arc::new(AtomicUsize::new(0));
    // Set from the register handshake before the bag is released, read by the
    // cloud publisher on every frame. Under managed replay no frame can be
    // published before registration, so the publisher never reads a stale
    // value. Without managed replay it is decided by configuration alone.
    #[cfg(feature = "mt_pubsub")]
    let publish_compressed_clouds = Arc::new(AtomicBool::new(false));
    let export_trajectory = resolve_export_trajectory_path(export_trajectory);
    if let Some(path) = export_trajectory.as_ref() {
        warn!("Trajectory export target: {}", path);
    }
    #[cfg(feature = "processing-time-export")]
    let export_latency = resolve_export_timing_path(export_latency, "latency");
    #[cfg(feature = "processing-time-export")]
    if let Some(path) = export_latency.as_ref() {
        warn!("Latency export target: {}", path);
    }
    #[cfg(feature = "processing-time-export")]
    let export_processing_time =
        resolve_export_timing_path(export_processing_time, "processing_time");
    #[cfg(feature = "processing-time-export")]
    if let Some(path) = export_processing_time.as_ref() {
        warn!("Processing time export target: {}", path);
    }
    #[cfg(any(feature = "mt_pubsub", feature = "hiroz", feature = "rat"))]
    let rt = if cfg!(any(feature = "mt_pubsub", feature = "hiroz"))
        || (synchronous_mode && cfg!(feature = "rat"))
    {
        Some(Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap(),
        ))
    } else {
        None
    };

    #[cfg(not(any(feature = "mt_pubsub", feature = "hiroz")))]
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
                let node_config = mt_pubsub::NodeConfig::new("pelorus_node")
                    .mode(mt_pubsub::Qos::Reliable)
                    .coord_mode(mt_pubsub_coord_mode.into())
                    .local_only(mt_pubsub_local_only);

                Node::create(node_config).await
            })
            .map_err(|err| anyhow!("Failed to create mt_pubsub node: {err}"))?,
    );

    // Gate replay on consumer attachment and per-frame completion.
    #[cfg(feature = "mt_pubsub")]
    if replay_tick_owner.is_downstream() {
        // What this build and configuration can offer. The consumer may not be
        // able to decode it, in which case the gate downgrades to the baseline.
        let preferred_cloud_encoding = match config.cloud_publisher_mode {
            #[cfg(feature = "compressed-pointcloud")]
            PointCloudMode::Compressed => pelorus_protocol::CloudEncoding::CompressedPointCloud2,
            #[cfg(not(feature = "compressed-pointcloud"))]
            PointCloudMode::Compressed => pelorus_protocol::CloudEncoding::BASELINE,
            PointCloudMode::PointCloud2 => pelorus_protocol::CloudEncoding::PointCloud2,
        };
        let state = Arc::new(std::sync::Mutex::new(
            pelorus_protocol::DownstreamGate::new(preferred_cloud_encoding),
        ));
        let service = rt
            .as_ref()
            .expect("rt always exists with mt_pubsub feat")
            .block_on(mt_service::ServiceServer::<
                pelorus_protocol::DownstreamRequest,
                pelorus_protocol::DownstreamResponse,
            >::new(
                Arc::clone(&node),
                pelorus_protocol::DOWNSTREAM_CONTROL_SERVICE.to_owned(),
            ))?;
        let ready_publisher = rt
            .as_ref()
            .expect("rt always exists with mt_pubsub feat")
            .block_on(node.create_publisher::<pelorus_protocol::DownstreamReady>(
                pelorus_protocol::DOWNSTREAM_READY_TOPIC.to_owned(),
                mt_pubsub::Qos::Reliable,
            ))?;
        let callback_state = Arc::clone(&state);
        let callback_shutdown = shutdown_flag.clone();
        let callback_compressed = Arc::clone(&publish_compressed_clouds);
        let callback_replay_tick_debt = Arc::clone(&replay_tick_debt);
        // Every acknowledgement pays for exactly one scan, which would leave
        // Minot's round trip on the critical path of every frame. The first one
        // therefore pays for `prefetch_scans`, so the preprocessing side finds the
        // next scans already waiting, as it does when Pelorus owns the replay.
        let callback_prefill_pending = Arc::new(AtomicBool::new(true));
        let callback_prefetch_scans = offline_prefetch_scans.max(1);
        // Registering unlocks the bag exactly once. The offline runner holds
        // Minot in `--sync` until then, so no frame can be produced before the
        // consumer's subscribers exist.
        let callback_registered = Arc::new(std::sync::Mutex::new(downstream_registered));
        rt.as_ref()
            .expect("rt always exists with mt_pubsub feat")
            .spawn(mt_service::ServiceServer::start(
                service,
                Arc::new(move |request| {
                    let state = Arc::clone(&callback_state);
                    let shutdown = callback_shutdown.clone();
                    let registered = Arc::clone(&callback_registered);
                    let compressed = Arc::clone(&callback_compressed);
                    let replay_tick_debt = Arc::clone(&callback_replay_tick_debt);
                    let prefill_pending = Arc::clone(&callback_prefill_pending);
                    let prefetch_scans = callback_prefetch_scans;
                    async move {
                        debug!("managed downstream request: {request:?}");
                        let effect = state
                            .lock()
                            .map_err(|_| "Pelorus downstream control state is poisoned".to_owned())?
                            .handle(request);
                        debug!("managed downstream response: {:?}", effect.response);
                        // Ordered before the replay unlock below so the very
                        // first frame already goes out in the agreed encoding.
                        if effect.response.accepted {
                            compressed.store(
                                effect.response.cloud_encoding
                                    == pelorus_protocol::CloudEncoding::CompressedPointCloud2,
                                Ordering::Release,
                            );
                        }
                        if effect.unlock_replay {
                            let sender = registered
                                .lock()
                                .map_err(|_| {
                                    "Pelorus downstream registration channel is poisoned".to_owned()
                                })?
                                .take();
                            if let Some(sender) = sender {
                                sender.send(()).map_err(|_| {
                                    "Pelorus offline runner stopped before replay was unlocked"
                                        .to_owned()
                                })?;
                            }
                        }
                        if effect.request_shutdown
                            && let Some(shutdown) = shutdown.as_ref()
                        {
                            shutdown.store(true, Ordering::SeqCst);
                        }
                        if effect.completed_frame.is_some() {
                            let desired = if prefill_pending.swap(false, Ordering::AcqRel) {
                                prefetch_scans
                            } else {
                                1
                            };
                            let request_count =
                                request_count_after_replay_debt(&replay_tick_debt, desired);
                            if request_count > 0 {
                                release_minot_ticks_async(
                                    shutdown.as_ref(),
                                    synchronous_mode,
                                    request_count,
                                )
                                .await
                                .map_err(|error| {
                                    format!("Failed to advance managed Minot replay: {error:#}")
                                })?;
                            }
                        }
                        Ok(effect.response)
                    }
                }),
            ));
        let ready_state = Arc::clone(&state);
        rt.as_ref()
            .expect("rt always exists with mt_pubsub feat")
            .spawn(async move {
                let ready = pelorus_protocol::DownstreamReady {
                    protocol_version: pelorus_protocol::PROTOCOL_VERSION,
                    process_id: std::process::id(),
                    control_service: pelorus_protocol::DOWNSTREAM_CONTROL_SERVICE.to_owned(),
                };
                let mut interval = tokio::time::interval(Duration::from_millis(100));
                loop {
                    interval.tick().await;
                    if ready_state
                        .lock()
                        .map(|state| state.has_consumer())
                        .unwrap_or(true)
                    {
                        break;
                    }
                    if let Err(error) = ready_publisher.publish(&ready).await {
                        warn!("Failed to publish managed downstream readiness: {error}");
                    }
                }
            });
    }

    #[cfg(feature = "hiroz")]
    let node = {
        info!(
            "Starting Hiroz backend: mode={}, endpoints={:?}, cloud_topic={}, imu_topic={}",
            hiroz_node_args.mode,
            hiroz_node_args.endpoints,
            config.sync_node_params.cloud_src_topic,
            config.sync_node_params.imu_src_topic
        );
        let mut builder = ZContextBuilder::default()
            .with_mode(hiroz_node_args.mode.clone())
            .keyexpr_format(hiroz_protocol::KeyExprFormat::RmwZenoh);
        if !hiroz_node_args.endpoints.is_empty() {
            builder = builder.with_connect_endpoints(hiroz_node_args.endpoints.clone());
        }
        let ctx = builder
            .build()
            .map_err(|e| anyhow!("Failed to create Hiroz context: {e}"))?;
        ctx.create_node("pelorus_node")
            .with_namespace("/pelorus")
            .build()
            .map_err(|e| anyhow!("Failed to create Hiroz node: {e}"))?
    };

    #[cfg(feature = "rat")]
    if synchronous_mode && let Some(rt) = rt.as_ref() {
        rat::init("pelorus", None, Some(Arc::clone(rt))).unwrap();
    }

    #[cfg(feature = "r2r")]
    let ctx = r2r::Context::create()?;
    #[cfg(feature = "r2r")]
    let mut node = r2r::Node::create(ctx, &ros_node_args.node_name, &ros_node_args.namespace)?;

    // r2r's publishers are `Send` but deliberately not `Sync`, so anything
    // holding one is not `Sync` either and an `Arc` of it is not `Send`. The
    // broadcaster is therefore kept owned and cloned per thread - a clone is a
    // `Weak` handle to the same publishers, not a second one. The
    // `ros2-client` broadcaster shares fine behind an `Arc`.
    #[cfg(feature = "r2r")]
    let tf_broadcaster = TfBroadcaster::new(&mut node);
    #[cfg(feature = "ros2-client")]
    let tf_broadcaster = Arc::new(TfBroadcaster::new(&mut node));
    #[cfg(feature = "mt_pubsub")]
    let tf_broadcaster = {
        let rt = rt
            .as_ref()
            .expect("rt always exists with mt_pubsub feat")
            .clone();
        // Publish TF in the same network as Pelorus' other topics.
        let coord_mode = match mt_pubsub_coord_mode {
            // Local-only runs share the coordinator Pelorus or Minot already started.
            PubSubCoordMode::Autostart if mt_pubsub_local_only => mt_pubsub::CoordMode::External,
            mode => mode.into(),
        };
        Arc::new(TfBroadcaster::with_coord_mode(
            rt,
            mt_pubsub_local_only,
            mt_pubsub::Qos::Reliable,
            coord_mode,
        ))
    };
    #[cfg(feature = "hiroz")]
    let tf_broadcaster = Arc::new(
        TfBroadcaster::new(&node)
            .map_err(|e| anyhow!("Failed to create Hiroz TF broadcaster: {e}"))?,
    );

    let mut pipeline = Pipeline::new(&config);
    let map = pipeline.map();
    let _odom_ext = Isometry3::from_parts(
        nalgebra::Translation3::new(0.0, 0.0, 0.0),
        UnitQuaternion::from_matrix(&config.sync_params.wheel_to_imu_rotation),
    );

    // Set lidar vendor in shared stats if present
    if let Some(ref stats) = shared_stats {
        stats.set_lidar_vendor(config.sync_params.lidar_vendor);
    }

    #[cfg(feature = "rerun")]
    let rec = {
        if config.rerun.enabled {
            let rerun_server_addr = &config.rerun.server_addr;

            // First check if the server is reachable via TCP
            let is_reachable = match rerun_server_addr.parse() {
                Ok(rerun_socket_addr) => std::net::TcpStream::connect_timeout(
                    &rerun_socket_addr,
                    Duration::from_millis(500),
                )
                .is_ok(),
                Err(err) => {
                    warn!(
                        "Invalid output.rerun.server_addr '{}': {err}",
                        rerun_server_addr
                    );
                    false
                }
            };

            if !is_reachable {
                info!(
                    "Rerun server not reachable at {}. Visualization disabled.",
                    rerun_server_addr
                );
                None
            } else {
                // Server is reachable, try to connect
                match rerun::RecordingStreamBuilder::new("pelorus")
                    .connect_grpc_opts(format!("rerun+http://{rerun_server_addr}/proxy"))
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
            None
        }
    };

    #[cfg(not(feature = "rerun"))]
    if config.rerun.enabled {
        return Err(
            anyhow!("output.rerun = true requires a build with the 'rerun' feature").into(),
        );
    }

    #[cfg(not(feature = "rerun"))]
    let _rerun_not_enabled = ();

    let (ready_for_next_tx, ready_for_next_rx) = std::sync::mpsc::channel();
    let minot_replay_mode = synchronous_mode && shutdown_request_rx.is_some();
    let imu_sync_mode = if config.imu_params.lidar_only {
        ImuSyncMode::Disabled {
            lossless: minot_replay_mode,
        }
    } else if minot_replay_mode {
        let demand_tick_debt = Arc::clone(&replay_tick_debt);
        let demand_shutdown_flag = shutdown_flag.clone();
        ImuSyncMode::DemandDrivenReplay {
            // Advance replay to fill an incomplete IMU interval and track the
            // extra scan as debt for the replay-clock owner.
            request_next_scan: Box::new(move || {
                if demand_shutdown_flag
                    .as_ref()
                    .map(|flag| flag.load(Ordering::Relaxed))
                    .unwrap_or(false)
                {
                    return;
                }
                demand_tick_debt.fetch_add(1, Ordering::AcqRel);
                release_minot_tick(demand_shutdown_flag.as_ref(), synchronous_mode);
            }),
        }
    } else if config.sync_params.lidar_as_sync {
        ImuSyncMode::LidarDriven
    } else {
        ImuSyncMode::Continuous
    };
    let (filtered_imu_tx, filtered_imu_rx) =
        if output_mcap.is_some() && !config.imu_params.lidar_only {
            let (tx, rx) = std::sync::mpsc::channel();
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

    #[cfg(feature = "rerun")]
    let rerun_prediction_export = minot_replay_mode && rec.is_some();
    #[cfg(not(feature = "rerun"))]
    let rerun_prediction_export = false;
    let realtime_exports_enabled = config.odometry_rate == OdometryRate::ImuAndLidar
        && (export_trajectory.is_some() || output_mcap.is_some() || rerun_prediction_export);
    let (realtime_export_tx, realtime_export_rx) = if realtime_exports_enabled {
        let (tx, rx) = std::sync::mpsc::channel();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    let (realtime_predictor, realtime_prediction_rx) =
        if config.odometry_rate == OdometryRate::ImuAndLidar {
            let (handle, receiver) =
                RealtimePredictorHandle::new_with_export(&config, realtime_export_tx);
            (Some(handle), Some(receiver))
        } else {
            (None, None)
        };
    let realtime_imu_callback = realtime_predictor
        .as_ref()
        .map(RealtimePredictorHandle::imu_callback);
    let realtime_lidar_barrier_callback = realtime_predictor
        .as_ref()
        .map(RealtimePredictorHandle::lidar_barrier_callback);

    let startup_base_frame = config
        .base_to_lidar_tf_override
        .then(|| config.base_link_frame.clone())
        .flatten();
    let startup_tf_callback = (config.sync_params.imu_lidar_tf_override || startup_base_frame.is_some()).then(|| {
        let map = map.clone();
        let realtime = realtime_predictor.as_ref().map(RealtimePredictorHandle::startup_tf_callback);
        let callback: pelorus_syncio::StartupTfCallback = Arc::new(std::sync::Mutex::new(move |imu: Option<Isometry3<f64>>, base: Option<Isometry3<f64>>| {
            let mut registration = map.lock().unwrap();
            if let Some(imu) = imu {
                registration.set_lidar_to_imu(imu);
                info!("Applied Lidar-to-IMU TF before initialization: translation={:?}, rotation={:?}", imu.translation.vector, imu.rotation);
            }
            if let Some(base) = base {
                registration.set_base_to_lidar(base);
                info!("Applied base-to-Lidar TF before scan processing: translation={:?}, rotation={:?}", base.translation.vector, base.rotation);
            }
            if let Some(realtime) = &realtime { realtime(imu, base); }
        }));
        callback
    });

    #[allow(unused_mut)]
    let (mut sync_node, synced_rx) = SyncNode::new(
        #[cfg(any(feature = "ros2-client", feature = "r2r"))]
        &mut node,
        #[cfg(any(feature = "mt_pubsub", feature = "hiroz"))]
        &node,
        #[cfg(any(feature = "mt_pubsub", feature = "hiroz"))]
        rt.as_ref()
            .expect("rt always exists with mt_pubsub/hiroz feat")
            .clone(),
        #[cfg(feature = "mt_pubsub")]
        mt_pubsub_local_only,
        &config.sync_params,
        SyncNodeInputs {
            startup_tf_callback,
            startup_base_frame,
            node_params: &config.sync_node_params,
            imu_pose_predictor_init: (!config.imu_params.lidar_only)
                .then(|| pipeline.build_init_callback()),
            imu_pose_prediction_undistort: Some(pipeline.build_undistort()),
            imu_pose_prediction_init_guess: Some(pipeline.build_init_guess()),
            ready_for_next: Some(ready_for_next_rx),
            filtered_imu_sender: filtered_imu_tx.clone(),
            imu_sync_mode,
            realtime_imu_callback,
            realtime_lidar_barrier_callback,
        },
        #[cfg(feature = "rerun")]
        rec.clone(),
    )?;
    #[cfg(feature = "native-drivers")]
    {
        let mut sensor_drivers = config.sensor_drivers.clone();
        if config.imu_params.lidar_only {
            sensor_drivers.imu.driver_type = pelorus_core::ImuDriverType::Disabled;
        }
        sync_node.start_native_drivers(&sensor_drivers, filtered_imu_tx.clone(), None)?;
    }
    let sync_node = Arc::new(sync_node);
    let drain_on_shutdown = Arc::new(AtomicBool::new(false));
    if let Some(stop_rx) = shutdown_request_rx {
        let node_for_stop = Arc::clone(&sync_node);
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

    #[cfg(feature = "ros2-client")]
    let cloud_publisher = node
        .create_publisher(
            &node
                .create_topic(
                    &ros2_client::Name::new(namespace, "registered").unwrap(),
                    ros2_client::MessageTypeName::new("sensor_msgs", "PointCloud2"),
                    &pelorus_syncio::qos_to_ros2(config.registered_cloud_qos),
                )
                .unwrap(),
            Some(pelorus_syncio::qos_to_ros2(config.registered_cloud_qos)),
        )
        .unwrap();
    #[cfg(feature = "ros2-client")]
    let odom_publisher = node
        .create_publisher(
            &node
                .create_topic(
                    &ros2_client::Name::new(namespace, "odom").unwrap(),
                    ros2_client::MessageTypeName::new("nav_msgs", "Odometry"),
                    &pelorus_syncio::qos_to_ros2(config.odom_qos),
                )
                .unwrap(),
            Some(pelorus_syncio::qos_to_ros2(config.odom_qos)),
        )
        .unwrap();
    #[cfg(all(not(feature = "no-viz"), feature = "ros2-client"))]
    let lidar_path_publisher = node
        .create_publisher(
            &node
                .create_topic(
                    &ros2_client::Name::new(namespace, "path").unwrap(),
                    ros2_client::MessageTypeName::new("nav_msgs", "Path"),
                    &pelorus_syncio::qos_to_ros2(config.path_qos),
                )
                .unwrap(),
            Some(pelorus_syncio::qos_to_ros2(config.path_qos)),
        )
        .unwrap();
    #[cfg(all(not(feature = "no-viz"), feature = "ros2-client"))]
    let imu_odom_publisher = node
        .create_publisher(
            &node
                .create_topic(
                    &ros2_client::Name::new(namespace, "imu_odom").unwrap(),
                    ros2_client::MessageTypeName::new("nav_msgs", "Odometry"),
                    &pelorus_syncio::qos_to_ros2(config.imu_odom_qos),
                )
                .unwrap(),
            Some(pelorus_syncio::qos_to_ros2(config.imu_odom_qos)),
        )
        .unwrap();

    #[cfg(feature = "r2r")]
    let cloud_publisher =
        node.create_publisher::<PointCloud2>("registered", config.registered_cloud_qos.into())?;
    #[cfg(feature = "mt_pubsub")]
    let cloud_publisher = rt
        .as_ref()
        .expect("rt always exists with mt_pubsub feat")
        .block_on(async {
            #[cfg(not(feature = "compressed-pointcloud"))]
            if matches!(config.cloud_publisher_mode, PointCloudMode::Compressed) {
                return Err(anyhow!(
                    "Compressed point-cloud output requires the 'compressed-pointcloud' feature"
                ));
            }

            // The baseline publisher always exists: it is what a consumer that
            // cannot decode Cloudini gets downgraded to.
            let plain = node
                .create_publisher::<PointCloud2>(
                    pelorus_protocol::REGISTERED_CLOUD_TOPIC.to_owned(),
                    mt_pubsub::Qos::Reliable,
                )
                .await?;

            #[cfg(feature = "compressed-pointcloud")]
            let compressed = if matches!(config.cloud_publisher_mode, PointCloudMode::Compressed) {
                // Its own topic, because a subscriber is typed and a
                // compressed message on the plain topic would not decode.
                Some(
                    node.create_publisher::<pelorus_protocol::CompressedPointCloud2>(
                        pelorus_protocol::compressed_cloud_topic(
                            pelorus_protocol::REGISTERED_CLOUD_TOPIC,
                        ),
                        mt_pubsub::Qos::Reliable,
                    )
                    .await?,
                )
            } else {
                None
            };

            // Without a handshake to defer to, configuration is the decision.
            // Managed replay overwrites this from the register response before
            // the bag is ever released.
            if !replay_tick_owner.is_downstream() {
                publish_compressed_clouds.store(
                    matches!(config.cloud_publisher_mode, PointCloudMode::Compressed),
                    Ordering::Release,
                );
            }

            Ok::<_, anyhow::Error>(MtCloudPublisher {
                plain,
                #[cfg(feature = "compressed-pointcloud")]
                compressed,
                #[cfg(feature = "compressed-pointcloud")]
                use_compressed: Arc::clone(&publish_compressed_clouds),
            })
        })?;
    #[cfg(feature = "hiroz")]
    let cloud_publisher = node
        .create_pub::<PointCloud2>("registered")
        .with_qos(pelorus_syncio::qos_to_hiroz(config.registered_cloud_qos))
        .build()
        .map_err(|e| anyhow!("Failed to create Hiroz registered cloud publisher: {e}"))?;
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
    #[cfg(feature = "mt_pubsub")]
    let registered_frame_publisher = rt
        .as_ref()
        .expect("rt always exists with mt_pubsub feat")
        .block_on(async {
            node.create_publisher::<pelorus_protocol::RegisteredFrameMarker>(
                pelorus_protocol::REGISTERED_FRAME_TOPIC.to_owned(),
                mt_pubsub::Qos::Reliable,
            )
            .await
        })?;
    #[cfg(feature = "hiroz")]
    let odom_publisher = node
        .create_pub::<Odometry>("odom")
        .with_qos(pelorus_syncio::qos_to_hiroz(config.odom_qos))
        .build()
        .map_err(|e| anyhow!("Failed to create Hiroz odom publisher: {e}"))?;
    // Same for the odometry publisher: r2r's `Publisher<T>` is `Send` and
    // `Clone` (it is a `Weak` handle, so cloning is cheap), and wrapping it in
    // an `Arc` would only make it unusable across threads.
    #[cfg(not(feature = "r2r"))]
    let odom_publisher = Arc::new(odom_publisher);

    let realtime_child_frame = Arc::new(std::sync::RwLock::new(None::<String>));
    let _realtime_odom_publisher = realtime_prediction_rx.map(|receiver| {
        // Clones the `Arc` for the shared backends and the r2r handles
        // themselves otherwise. Handing a publisher to another thread is r2r's
        // documented pattern: creation stays on the node's thread, publishing
        // may move.
        let publisher = odom_publisher.clone();
        let prediction_tf_broadcaster = tf_broadcaster.clone();
        let map_frame = config.sync_params.map_frame.clone();
        let child_frame = Arc::clone(&realtime_child_frame);
        #[cfg(feature = "rerun")]
        let prediction_rec = rec.clone();
        #[cfg(feature = "mt_pubsub")]
        let runtime = rt
            .as_ref()
            .expect("rt always exists with mt_pubsub feat")
            .clone();
        std::thread::spawn(move || {
            while let Ok(prediction) = receiver.recv() {
                let Some(child) = child_frame.read().ok().and_then(|value| value.clone()) else {
                    continue;
                };
                let odom = realtime_prediction_to_odom(&prediction, &map_frame, &child);
                let tf = realtime_prediction_to_tf(&prediction, &map_frame, &child);
                #[cfg(feature = "rerun")]
                if let Some(prediction_rec) = prediction_rec.as_ref() {
                    log_realtime_prediction_to_rerun(prediction_rec, &prediction);
                }
                #[cfg(feature = "mt_pubsub")]
                {
                    if let Err(error) = runtime.block_on(publisher.publish(&odom)) {
                        debug!("Failed to publish real-time odometry: {error}");
                    }
                    if let Err(error) =
                        runtime.block_on(prediction_tf_broadcaster.send_transform(tf))
                    {
                        debug!("Failed to publish real-time transform: {error}");
                    }
                }
                #[cfg(not(feature = "mt_pubsub"))]
                {
                    if let Err(error) = publisher.publish(&odom) {
                        debug!("Failed to publish real-time odometry: {error}");
                    }
                    if let Err(error) = prediction_tf_broadcaster.send_transform(tf) {
                        debug!("Failed to publish real-time transform: {error}");
                    }
                }
            }
        })
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
    #[cfg(all(not(feature = "no-viz"), feature = "hiroz"))]
    let lidar_path_publisher = node
        .create_pub::<Path>("path")
        .with_qos(pelorus_syncio::qos_to_hiroz(config.path_qos))
        .build()
        .map_err(|e| anyhow!("Failed to create Hiroz path publisher: {e}"))?;
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
    #[cfg(all(not(feature = "no-viz"), feature = "hiroz"))]
    let imu_odom_publisher = node
        .create_pub::<Odometry>("imu_odom")
        .with_qos(pelorus_syncio::qos_to_hiroz(config.imu_odom_qos))
        .build()
        .map_err(|e| anyhow!("Failed to create Hiroz IMU odom publisher: {e}"))?;
    #[cfg(feature = "mt_pubsub")]
    let stats_publisher = rt
        .as_ref()
        .expect("rt always exists with mt_pubsub feat")
        .block_on(async {
            node.create_publisher::<RosString>("pelorus_stats".to_owned(), mt_pubsub::Qos::Reliable)
                .await
                .unwrap()
        });
    #[cfg(feature = "hiroz")]
    let stats_publisher = node
        .create_pub::<RosString>("pelorus_stats")
        .with_qos(pelorus_syncio::qos_to_hiroz(
            pelorus_core::Qos::SystemDefault,
        ))
        .build()
        .map_err(|e| anyhow!("Failed to create Hiroz stats publisher: {e}"))?;
    #[cfg(feature = "mt_pubsub")]
    let _stats_publish_task = shared_stats.as_ref().map(|stats| {
        let stats = Arc::clone(stats);
        let publisher = stats_publisher.clone();
        let shutdown = shutdown_flag.clone();
        rt.as_ref()
            .expect("rt always exists with mt_pubsub feat")
            .handle()
            .spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_millis(250));
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    interval.tick().await;
                    if shutdown
                        .as_ref()
                        .is_some_and(|flag| flag.load(Ordering::Relaxed))
                    {
                        return;
                    }
                    stats.update_system_stats();
                    let msg = stats_message(&stats);
                    if let Err(e) = publisher.publish(&msg).await {
                        debug!("Failed to publish Pelorus stats: {e}");
                    }
                }
            })
    });
    let mut t_world = nalgebra::Isometry3::identity();
    let mut t_last_current = nalgebra::Isometry3::identity();
    #[cfg(feature = "hiroz")]
    let mut last_stats_pub = std::time::Instant::now();

    let mut lidar_path_poses: VecDeque<PoseStamped> = VecDeque::new();
    let lidar_path_frame_id = config.sync_params.map_frame.clone();
    let mut last_mcap_path_write_time: Option<u64> = None;

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

    // Clone shutdown_flag for the processing thread. The original is used for the spinner.
    let processing_shutdown_flag = shutdown_flag.clone();

    let processing_tick_debt = Arc::clone(&replay_tick_debt);
    #[cfg(feature = "mt_pubsub")]
    let marker_compressed_clouds = Arc::clone(&publish_compressed_clouds);
    let export_trajectory_for_worker = export_trajectory.clone();
    #[cfg(feature = "processing-time-export")]
    let export_latency_for_worker = export_latency.clone();
    #[cfg(feature = "processing-time-export")]
    let export_processing_time_for_worker = export_processing_time.clone();
    let processing_handle = std::thread::spawn(move || -> anyhow::Result<()> {
        let mut filtered_imu_rx = filtered_imu_rx;
        #[cfg(feature = "rerun")]
        let mut last_map_snapshot_time: Option<f64> = None;
        let mut mcap_writer = match output_mcap {
            Some(output) => Some(McapOutputWriter::create(
                &output.path,
                &output.topics,
                output.rosbag_metadata_path,
            )?),
            None => None,
        };
        let downstream_owns_replay = replay_tick_owner.is_downstream();
        let gate_minot_replay_on_mcap =
            minot_replay_mode && !downstream_owns_replay && mcap_writer.is_some();
        let relay_minot_replay_on_registration = minot_replay_mode && mcap_writer.is_none();
        let mut minot_prefill_requested = false;
        // Initialize Synchronizer prediction
        let mut ready_for_next_tx = Some(ready_for_next_tx);
        let mut registration_ready_tx = None;
        let mut ready_relay_handle = None;
        if gate_minot_replay_on_mcap || relay_minot_replay_on_registration {
            let (relay_tx, relay_rx) = std::sync::mpsc::channel();
            registration_ready_tx = Some(relay_tx);
            let sync_ready_tx = ready_for_next_tx
                .as_ref()
                .expect("sync ready sender must exist")
                .clone();
            let relay_shutdown_flag = processing_shutdown_flag.clone();
            let relay_drain_on_shutdown = Arc::clone(&drain_on_shutdown);
            let relay_tick_debt = Arc::clone(&replay_tick_debt);
            let prefetch_scans = offline_prefetch_scans.max(1);
            ready_relay_handle = Some(std::thread::spawn(move || {
                let mut requested_prefill = false;
                while let Ok(scan_id) = relay_rx.recv() {
                    if sync_ready_tx.send(scan_id).is_err() {
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
                    // Under downstream ownership the consumer's own trigger
                    // pulls the next scan. This relay still releases Synchronizer's
                    // registration barrier, but must not advance bag input.
                    if !downstream_owns_replay {
                        let desired_request_count = if requested_prefill {
                            1
                        } else {
                            requested_prefill = true;
                            prefetch_scans
                        };
                        let request_count = request_count_after_replay_debt(
                            &relay_tick_debt,
                            desired_request_count,
                        );
                        if request_count > 0 {
                            release_minot_ticks(
                                relay_shutdown_flag.as_ref(),
                                synchronous_mode,
                                request_count,
                            );
                        }
                    }
                }
            }));
        }
        // A scan consumed without publishing an authoritative frame still owes
        // Minot exactly one trigger: the downstream consumer only pays for
        // frames it actually receives. Synchronizer's mid-frame demand may already
        // have advanced the bag, so repay that debt before asking for more.
        let release_unpublished_scan_tick = || {
            let request_count = request_count_after_replay_debt(&processing_tick_debt, 1);
            if request_count > 0 {
                release_minot_ticks(
                    processing_shutdown_flag.as_ref(),
                    synchronous_mode,
                    request_count,
                );
            }
        };
        let mut sync_node = Some(sync_node);

        let active_backend = ActiveBackend::detect(&config);
        log_active_backend(&config, &active_backend);
        if let Some(ref stats) = shared_stats {
            set_backend_from_runtime(stats, &config, &active_backend);
        }

        // TODO avoid casting f32, maybe res needs f64?
        match &active_backend {
            ActiveBackend::Cpu => {}
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(stream) => pelorus_cuda::init_map_with_mode(
                stream,
                config.map_capacity,
                config.map_resolution as f32,
                config.dist_update_sensitivity,
                config.neighbor_max_range as f32,
                pelorus_cuda::NeighborType::Neighbor26,
                config.map_multiresolution,
            ),
            #[cfg(feature = "metal")]
            ActiveBackend::Metal(stream) => pelorus_metal::init_map_with_mode(
                stream,
                config.map_capacity,
                config.map_resolution as f32,
                config.dist_update_sensitivity,
                config.neighbor_max_range as f32,
                pelorus_metal::NeighborType::Neighbor26,
                config.map_multiresolution,
            ),
            #[cfg(feature = "vulkan")]
            ActiveBackend::Vulkan(stream) => pelorus_vulkan::init_map_with_mode(
                stream,
                config.map_capacity,
                config.map_resolution as f32,
                config.dist_update_sensitivity,
                config.neighbor_max_range as f32,
                pelorus_vulkan::NeighborType::Neighbor26,
                config.map_multiresolution,
            ),
        }

        let mut trajectory_file = match export_trajectory_for_worker {
            Some(path_str) => {
                log::info!("Saving trajectory to {}", path_str);
                match std::fs::File::create(&path_str) {
                    Ok(f) => Some(f),
                    Err(e) => {
                        log::error!("Failed to create trajectory file {}: {}", path_str, e);
                        None
                    }
                }
            }
            None => {
                // No export
                None
            }
        };
        let mut trajectory_samples = if config.odometry_rate == OdometryRate::ImuAndLidar {
            trajectory_file.as_ref().map(|_| Vec::new())
        } else {
            None
        };
        #[cfg(feature = "processing-time-export")]
        let mut latency_file = match export_latency_for_worker {
            Some(path_str) => {
                log::info!("Saving processing times to {}", path_str);
                match std::fs::File::create(&path_str) {
                    Ok(f) => Some(f),
                    Err(e) => {
                        log::error!("Failed to create processing time file {}: {}", path_str, e);
                        None
                    }
                }
            }
            None => None,
        };
        #[cfg(feature = "processing-time-export")]
        let mut processing_time_file = match export_processing_time_for_worker {
            Some(path_str) => {
                log::info!("Saving raw processing times to {}", path_str);
                match std::fs::File::create(&path_str) {
                    Ok(f) => Some(f),
                    Err(e) => {
                        log::error!(
                            "Failed to create raw processing time file {}: {}",
                            path_str,
                            e
                        );
                        None
                    }
                }
            }
            None => None,
        };

        if let Some(signal) = ready_signal {
            let _ = signal.send(());
        }

        let mut static_tf_published = false;
        let mut realtime_export_child_frame: Option<String> = None;

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
            let sync_tick = match processing_shutdown_flag.as_ref() {
                Some(flag) => {
                    let shutdown_requested = flag.load(core::sync::atomic::Ordering::Relaxed);
                    let drain_requested =
                        drain_on_shutdown.load(core::sync::atomic::Ordering::SeqCst);
                    if shutdown_requested || drain_requested {
                        if drain_requested {
                            if !draining_after_end_of_bag {
                                draining_after_end_of_bag = true;
                                info!(
                                    "Shutdown requested after end-of-bag hint. Draining queued scans"
                                );
                            }
                            match synced_rx.recv_timeout(Duration::from_millis(50)) {
                                Ok(t) => t,
                                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                                | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                                    if let Some(ref sync_node) = sync_node {
                                        sync_node.request_shutdown();
                                    }
                                    // Drop sync gate and node ownership so all producer chains can close.
                                    drop(registration_ready_tx.take());
                                    drop(ready_for_next_tx.take());
                                    drop(sync_node.take());
                                    info!(
                                        "Shutdown requested after end-of-bag hint, queue drained, exiting processing loop"
                                    );
                                    break;
                                }
                            }
                        } else {
                            if let Some(ref sync_node) = sync_node {
                                sync_node.request_shutdown();
                            }
                            // Drop sync gate and node ownership so all producer chains can close.
                            drop(registration_ready_tx.take());
                            drop(ready_for_next_tx.take());
                            drop(sync_node.take());
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
            #[cfg(feature = "processing-time-export")]
            let latency_started = latency_file.as_ref().map(|_| sync_tick.received_at);
            // Record scan arrival for measuring actual sensor rate
            if config.base_to_lidar_tf_override {
                config.base_to_lidar = map.lock().unwrap().base_to_lidar;
            }
            if let Some(ref stats) = shared_stats {
                let stamp_sec = {
                    let s = timestamp_to_secs(&sync_tick.cloud_header.stamp);
                    if s > 0.0 {
                        s
                    } else {
                        timestamp_to_secs(&sync_tick.odom_time)
                    }
                };
                stats.record_scan_timestamp(stamp_sec);
            }

            eval.merge(&sync_tick.runtime_eval);
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
                    cuda_cloud.realloc_if_needed_cuda(stream, sync_tick.deskewed_cloud.len());
                    cuda_cloud.upload_from_slice_cuda(stream, sync_tick.deskewed_cloud.as_slice());
                }
                #[cfg(feature = "metal")]
                ActiveBackend::Metal(stream) => {
                    // Metal uses shared memory. Sync before overwriting to avoid races.
                    stream.sync();
                    persistent_cloud_buf
                        .realloc_if_needed_metal(stream, sync_tick.deskewed_cloud.len());
                    persistent_cloud_buf
                        .upload_from_slice_metal(stream, sync_tick.deskewed_cloud.as_slice());
                }
                #[cfg(feature = "vulkan")]
                ActiveBackend::Vulkan(stream) => {
                    stream.sync();
                    persistent_cloud_buf
                        .realloc_if_needed_vulkan(stream, sync_tick.deskewed_cloud.len());
                    persistent_cloud_buf
                        .upload_from_slice_vulkan(stream, sync_tick.deskewed_cloud.as_slice());
                }
            }

            let ram_cloud;
            #[allow(unreachable_patterns)]
            let cloud_for_reg: &LocatedBufferView<pelorus_core::PointXYZIT> = match &active_backend
            {
                ActiveBackend::Cpu => {
                    ram_cloud = LocatedBufferView::Ram(sync_tick.deskewed_cloud);
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

            let cloud_out = sync_tick.deskewed_cloud_passthrough;

            let orig_frame = sync_tick.cloud_header.frame_id.clone();
            let orig_time = sync_tick.cloud_header.stamp;
            let current_cloud_time = timestamp_to_secs(&sync_tick.odom_time);
            if let Some(existing_last_scan_time) = last_scan_time
                && current_cloud_time < existing_last_scan_time
            {
                let mut map = map.lock().unwrap();
                map.clear_localmap(&active_backend);

                info!("Resetting map");
                match &active_backend {
                    ActiveBackend::Cpu => {}
                    #[cfg(feature = "cuda")]
                    ActiveBackend::Cuda(stream) => pelorus_cuda::init_map_with_mode(
                        stream,
                        config.map_capacity,
                        config.map_resolution as f32,
                        config.dist_update_sensitivity,
                        config.neighbor_max_range as f32,
                        pelorus_cuda::NeighborType::Neighbor26,
                        config.map_multiresolution,
                    ),
                    #[cfg(feature = "metal")]
                    ActiveBackend::Metal(stream) => pelorus_metal::init_map_with_mode(
                        stream,
                        config.map_capacity,
                        config.map_resolution as f32,
                        config.dist_update_sensitivity,
                        config.neighbor_max_range as f32,
                        pelorus_metal::NeighborType::Neighbor26,
                        config.map_multiresolution,
                    ),
                    #[cfg(feature = "vulkan")]
                    ActiveBackend::Vulkan(stream) => pelorus_vulkan::init_map_with_mode(
                        stream,
                        config.map_capacity,
                        config.map_resolution as f32,
                        config.dist_update_sensitivity,
                        config.neighbor_max_range as f32,
                        pelorus_vulkan::NeighborType::Neighbor26,
                        config.map_multiresolution,
                    ),
                };
                t_world = nalgebra::Isometry3::identity();
                t_last_current = nalgebra::Isometry3::identity();
            }

            last_scan_time = Some(timestamp_to_secs(&orig_time));

            // use the last difference if there is no odom from sync
            let init_guess_lidar = if sync_tick.imu_odom.pose.pose.position.x == 0.0
                && sync_tick.imu_odom.pose.pose.position.y == 0.0
                && sync_tick.imu_odom.pose.pose.position.z == 0.0
            {
                // sync did not return a orientation (for example when not using IMU at all), use the last difference as guess
                if sync_tick.imu_odom.pose.pose.orientation.w == 1.0
                    && sync_tick.imu_odom.pose.pose.orientation.x == 0.0
                    && sync_tick.imu_odom.pose.pose.orientation.y == 0.0
                    && sync_tick.imu_odom.pose.pose.orientation.z == 0.0
                {
                    t_world * t_last_current
                } else {
                    // there is an orientation calculated by sync for undeskewing the cloud, so we can use that
                    let initial_guess_rotation =
                        nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                            sync_tick.imu_odom.pose.pose.orientation.w,
                            sync_tick.imu_odom.pose.pose.orientation.x,
                            sync_tick.imu_odom.pose.pose.orientation.y,
                            sync_tick.imu_odom.pose.pose.orientation.z,
                        ));
                    let initial_guess_lidar_local = nalgebra::Isometry3::from_parts(
                        t_last_current.translation,
                        initial_guess_rotation,
                    );
                    t_world * initial_guess_lidar_local
                }
            } else {
                let initial_guess_translation = Translation3::new(
                    sync_tick.imu_odom.pose.pose.position.x,
                    sync_tick.imu_odom.pose.pose.position.y,
                    sync_tick.imu_odom.pose.pose.position.z,
                );

                let initial_guess_rotation =
                    nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                        sync_tick.imu_odom.pose.pose.orientation.w,
                        sync_tick.imu_odom.pose.pose.orientation.x,
                        sync_tick.imu_odom.pose.pose.orientation.y,
                        sync_tick.imu_odom.pose.pose.orientation.z,
                    ));

                // init_guess from sync is IMU pose T_W_I. Convert to Lidar pose T_W_L.
                nalgebra::Isometry3::from_parts(initial_guess_translation, initial_guess_rotation)
                    * pipeline.ext()
            };

            let registration_result = {
                let mut map = map.lock().unwrap();
                // This correction barrier orders Synchronizer prediction independently
                // of downstream backpressure.
                let ready_for_registration = if relay_minot_replay_on_registration {
                    registration_ready_tx.as_ref()
                } else if gate_minot_replay_on_mcap {
                    None
                } else {
                    ready_for_next_tx.as_ref()
                };
                match map.register_and_map(
                    pelorus_reg::RegistrationScan {
                        cloud: cloud_for_reg,
                        imu_transform: init_guess_lidar,
                        lidar_timestamp: current_cloud_time,
                        scan_id: sync_tick.scan_id as usize,
                        map_only: sync_tick.map_only,
                    },
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
                    if let Some(ready) = registration_ready_tx
                        .as_ref()
                        .or(ready_for_next_tx.as_ref())
                    {
                        let _ = ready.send(sync_tick.scan_id);
                    }
                    if gate_minot_replay_on_mcap || relay_minot_replay_on_registration {
                        // The relay only advances Minot when Pelorus owns the
                        // replay clock, so pay for this initialization scan here.
                        if downstream_owns_replay {
                            release_unpublished_scan_tick();
                        }
                    } else if minot_replay_mode {
                        if downstream_owns_replay {
                            release_unpublished_scan_tick();
                        } else {
                            let request_count = if minot_prefill_requested {
                                1
                            } else {
                                minot_prefill_requested = true;
                                offline_prefetch_scans.max(1)
                            };
                            release_minot_ticks(
                                processing_shutdown_flag.as_ref(),
                                synchronous_mode,
                                request_count,
                            );
                        }
                    } else {
                        release_minot_tick(processing_shutdown_flag.as_ref(), synchronous_mode);
                    }
                    continue;
                } // skip outputting trivial solutions on initialization
                Err(err) => {
                    error!("{err}");
                    drain_filtered_imu_mcap(&mut mcap_writer, &mut filtered_imu_rx)?;
                    if let Some(ready) = registration_ready_tx
                        .as_ref()
                        .or(ready_for_next_tx.as_ref())
                    {
                        let _ = ready.send(sync_tick.scan_id);
                    }
                    if gate_minot_replay_on_mcap {
                    } else {
                        if downstream_owns_replay {
                            // The failed registration still consumed one scan.
                            release_unpublished_scan_tick();
                        } else if !minot_replay_mode {
                            release_minot_tick(processing_shutdown_flag.as_ref(), synchronous_mode);
                        }
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
            realtime_export_child_frame = Some(child_frame.clone());
            if let Ok(mut realtime_frame) = realtime_child_frame.write() {
                *realtime_frame = Some(child_frame.clone());
            }

            if !static_tf_published && config.base_link_frame.is_some() {
                // Republish the base_link <-> lidar extrinsic on /tf_static. Internally
                // `config.base_to_lidar` already carries this transform (manual config or
                // just overridden from TF above), but until now it was never re-emitted:
                // pelorus published `map_frame -> child_frame` (== base_link_frame here)
                // dynamically, while the cloud stayed stamped with `orig_frame` (the lidar
                // frame). A consumer with no other path from base_link to the lidar frame
                // (e.g. scope, which insists on a fully resolvable TF chain per message)
                // could never complete `map_frame -> base_link_frame -> orig_frame` and
                // silently dropped every cloud as a stale pending TF lookup.
                let base_to_lidar_tf = TransformStamped {
                    header: Header {
                        frame_id: child_frame.clone(),
                        stamp: Time {
                            sec: orig_time.sec,
                            nanosec: orig_time.nanosec,
                        },
                    },
                    child_frame_id: orig_frame.clone(),
                    transform: Transform {
                        translation: Vector3 {
                            x: config.base_to_lidar.translation.vector.x,
                            y: config.base_to_lidar.translation.vector.y,
                            z: config.base_to_lidar.translation.vector.z,
                        },
                        rotation: Quaternion {
                            x: config.base_to_lidar.rotation.quaternion().i,
                            y: config.base_to_lidar.rotation.quaternion().j,
                            z: config.base_to_lidar.rotation.quaternion().k,
                            w: config.base_to_lidar.rotation.quaternion().w,
                        },
                    },
                };
                #[cfg(feature = "mt_pubsub")]
                {
                    let inner_tf_broad = Arc::clone(&tf_broadcaster);
                    std::thread::spawn(move || {
                        future::block_on(async move {
                            if let Err(err) =
                                inner_tf_broad.send_static_transform(base_to_lidar_tf).await
                            {
                                error!("Failed to send static base_link-to-lidar transform: {err}");
                            }
                        });
                    });
                }
                #[cfg(not(feature = "mt_pubsub"))]
                if let Err(err) = tf_broadcaster.send_static_transform(base_to_lidar_tf) {
                    error!("Failed to send static base_link-to-lidar transform: {err}");
                }

                static_tf_published = true;
            }

            t_world = if config.base_link_frame.is_some() {
                register_pose.base_pose
            } else {
                register_pose.pose
            };

            // transformation from the last pose to the current pose in the local frame
            t_last_current = old_t_world.inverse() * t_world;

            let mut pelorus_odom = Odometry::default();
            pelorus_odom.header.stamp = Time {
                sec: sync_tick.odom_time.sec,
                nanosec: sync_tick.odom_time.nanosec,
            };
            pelorus_odom.header.frame_id = config.sync_params.map_frame.clone();
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

            let r_b_l = if config.base_link_frame.is_some() {
                config.base_to_lidar.rotation.inverse()
            } else {
                UnitQuaternion::identity()
            };
            let r_l_i = ekf_state.lidar_imu_extrinsic.rotation.inverse();
            let ang_vel_body = (r_b_l * r_l_i) * ekf_state.angular_velocity;
            pelorus_odom.twist.twist.angular.x = ang_vel_body.x;
            pelorus_odom.twist.twist.angular.y = ang_vel_body.y;
            pelorus_odom.twist.twist.angular.z = ang_vel_body.z;

            {
                let (pose_cov_eqf, twist_cov_eqf) = {
                    let map = pipeline.map();
                    let guard = map.lock().unwrap();
                    let lidar_to_child = if config.base_link_frame.is_some() {
                        config.base_to_lidar.inverse()
                    } else {
                        Isometry3::identity()
                    };
                    let pose_cov = guard.get_output_pose_covariance(&lidar_to_child);
                    let tw_cov = guard.get_output_twist_covariance(&t_world);
                    (pose_cov, tw_cov)
                };

                #[cfg(not(feature = "r2r"))]
                {
                    pelorus_odom.pose.covariance = pose_cov_eqf;
                    pelorus_odom.twist.covariance = twist_cov_eqf;
                }

                #[cfg(feature = "r2r")]
                {
                    pelorus_odom.pose.covariance = pose_cov_eqf.to_vec();
                    pelorus_odom.twist.covariance = twist_cov_eqf.to_vec();
                }
            }

            // Generate prediction-only export samples before serializing this
            // correction, keeping MCAP/TUM/Rerun timelines chronological.
            if realtime_export_rx.is_some()
                && let Some(predictor) = realtime_predictor.as_ref()
            {
                let registration = pipeline.map();
                let registration = registration.lock().unwrap();
                predictor.prepare_export(&registration, &sync_tick.imu_buffer);
            }

            #[cfg(feature = "rerun")]
            if let Some(ref rec) = rec {
                let odom_pose = Isometry3::from_parts(
                    nalgebra::Translation3::new(
                        pelorus_odom.pose.pose.position.x,
                        pelorus_odom.pose.pose.position.y,
                        pelorus_odom.pose.pose.position.z,
                    ),
                    UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                        pelorus_odom.pose.pose.orientation.w,
                        pelorus_odom.pose.pose.orientation.x,
                        pelorus_odom.pose.pose.orientation.y,
                        pelorus_odom.pose.pose.orientation.z,
                    )),
                );
                let odom_pose_rerun = ros_to_rerun.cast() * odom_pose;
                let odom_transform = rerun::Transform3D::from_translation(glam::Vec3::new(
                    odom_pose_rerun.translation.x as f32,
                    odom_pose_rerun.translation.y as f32,
                    odom_pose_rerun.translation.z as f32,
                ))
                .with_quaternion(glam::Quat::from_xyzw(
                    odom_pose_rerun.rotation.quaternion().i as f32,
                    odom_pose_rerun.rotation.quaternion().j as f32,
                    odom_pose_rerun.rotation.quaternion().k as f32,
                    odom_pose_rerun.rotation.quaternion().w as f32,
                ));

                if let Err(err) = rec.log("world/latest_odometry", &odom_transform) {
                    error!("Error sending latest odometry pose to rerun {err}");
                }

                if let Err(err) = rec.log(
                    "world/latest_odometry",
                    &rerun::TransformAxes3D::new(0.75).with_show_frame(false),
                ) {
                    error!("Error sending latest odometry axes to rerun {err}");
                }

                let pose_cov = &pelorus_odom.pose.covariance;
                let position_cov_world = nalgebra::Matrix3::new(
                    pose_cov[0],
                    pose_cov[1],
                    pose_cov[2],
                    pose_cov[6],
                    pose_cov[7],
                    pose_cov[8],
                    pose_cov[12],
                    pose_cov[13],
                    pose_cov[14],
                );
                let odom_rotation = odom_pose.rotation.to_rotation_matrix();
                let position_cov_local = odom_rotation.matrix().transpose()
                    * position_cov_world
                    * odom_rotation.matrix();
                let eig = position_cov_local.symmetric_eigen();
                let mut eig_vectors = eig.eigenvectors;
                if eig_vectors.determinant() < 0.0 {
                    eig_vectors.set_column(0, &(-eig_vectors.column(0)));
                }

                let covariance_half_sizes = glam::Vec3::new(
                    eig.eigenvalues.x.max(0.0).sqrt() as f32,
                    eig.eigenvalues.y.max(0.0).sqrt() as f32,
                    eig.eigenvalues.z.max(0.0).sqrt() as f32,
                );
                let covariance_rotation = UnitQuaternion::from_matrix(&eig_vectors);
                let covariance_quat = covariance_rotation.quaternion();
                let covariance_ellipsoid = rerun::Ellipsoids3D::update_fields()
                    .with_half_sizes([covariance_half_sizes])
                    .with_quaternions([glam::Quat::from_xyzw(
                        covariance_quat.i as f32,
                        covariance_quat.j as f32,
                        covariance_quat.k as f32,
                        covariance_quat.w as f32,
                    )])
                    .with_colors([rerun::Color::from_unmultiplied_rgba(64, 180, 255, 96)])
                    .with_fill_mode(rerun::components::FillMode::TransparentFillMajorWireframe)
                    .with_line_radii([rerun::Radius::new_ui_points(1.0)]);

                if let Err(err) = rec.log("world/latest_odometry/covariance", &covariance_ellipsoid)
                {
                    error!("Error sending latest odometry covariance to rerun {err}");
                }
            }

            // publish paths but only add new poses if the robot moved enough
            #[cfg(not(feature = "no-viz"))]
            let changed_lidar_path = {
                match lidar_path_poses.back() {
                    Some(last_pose) => {
                        let last_pose = last_pose.pose.clone();
                        let last_pose_position = last_pose.position;
                        let current_pose_position = &pelorus_odom.pose.pose.position;

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
                            pelorus_syncio::impl_ros2_interfaces_jazzy_serde::from_pointcloud2_msg(
                                msg,
                            );
                        #[cfg(feature = "mt_pubsub")]
                        let out = impl_ros2_interfaces_jazzy_rkyv::from_pointcloud2_msg(msg);
                        #[cfg(feature = "r2r")]
                        let out = impl_r2r::from_pointcloud2_msg(msg);
                        #[cfg(feature = "hiroz")]
                        let out = impl_hiroz::from_pointcloud2_msg(msg);

                        out
                    }
                    // ignore downsampled one if we have the raw
                    (Some(items), _) => {
                        let msg = PointCloud2Msg::try_from_vec(items.to_vec()).expect("msg");
                        #[cfg(feature = "ros2-client")]
                        let out =
                            pelorus_syncio::impl_ros2_interfaces_jazzy_serde::from_pointcloud2_msg(
                                msg,
                            );
                        #[cfg(feature = "mt_pubsub")]
                        let out = impl_ros2_interfaces_jazzy_rkyv::from_pointcloud2_msg(msg);
                        #[cfg(feature = "r2r")]
                        let out = impl_r2r::from_pointcloud2_msg(msg);
                        #[cfg(feature = "hiroz")]
                        let out = impl_hiroz::from_pointcloud2_msg(msg);

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
                            pelorus_syncio::impl_ros2_interfaces_jazzy_serde::from_pointcloud2_msg(
                                msg,
                            );
                        #[cfg(feature = "mt_pubsub")]
                        let out = impl_ros2_interfaces_jazzy_rkyv::from_pointcloud2_msg(msg);
                        #[cfg(feature = "r2r")]
                        let out = impl_r2r::from_pointcloud2_msg(msg);
                        #[cfg(feature = "hiroz")]
                        let out = impl_hiroz::from_pointcloud2_msg(msg);

                        out
                    }
                }
            };

            cloud.header.frame_id = orig_frame.clone();
            cloud.header.stamp = Time {
                sec: sync_tick.odom_time.sec,
                nanosec: sync_tick.odom_time.nanosec,
            };
            #[cfg(feature = "processing-time-export")]
            if let (Some(latency_file), Some(latency_started)) =
                (latency_file.as_mut(), latency_started)
            {
                let latency_ns = latency_started.elapsed().as_nanos();
                let content_to_write = format!("{current_cloud_time} {latency_ns}\n");

                std::io::Write::write_all(latency_file, content_to_write.as_bytes())?;
            }
            // Predictions preceding this correction must be serialized first,
            // preserving timestamp order in both trajectory export formats.
            drain_realtime_odometry_exports(
                &realtime_export_rx,
                &mut trajectory_samples,
                &mut mcap_writer,
                #[cfg(feature = "rerun")]
                if minot_replay_mode {
                    rec.as_ref()
                } else {
                    None
                },
                &config.sync_params.map_frame,
                &child_frame,
            )?;
            if let Some(samples) = trajectory_samples.as_mut() {
                samples.push((current_cloud_time, t_world));
            } else if let Some(file) = trajectory_file.as_mut() {
                let t = t_world.translation.vector;
                let q = t_world.rotation.quaternion();
                let line = format!(
                    "{} {} {} {} {} {} {} {}\n",
                    current_cloud_time, t.x, t.y, t.z, q.i, q.j, q.k, q.w
                );
                std::io::Write::write_all(file, line.as_bytes())?;
            }

            // -- TF --
            let tf = TransformStamped {
                header: Header {
                    frame_id: config.sync_params.map_frame.clone(),
                    stamp: Time {
                        sec: sync_tick.odom_time.sec,
                        nanosec: sync_tick.odom_time.nanosec,
                    },
                },
                child_frame_id: child_frame,
                transform: Transform {
                    translation: Vector3 {
                        x: t_world.translation.vector.x,
                        y: t_world.translation.vector.y,
                        z: t_world.translation.vector.z,
                    },
                    rotation: Quaternion {
                        x: t_world.rotation.quaternion().i,
                        y: t_world.rotation.quaternion().j,
                        z: t_world.rotation.quaternion().k,
                        w: t_world.rotation.quaternion().w,
                    },
                },
            };
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
                // Record the exact corrected odometry message published on
                // `odom`. High-rate predictions use the same frame and schema.
                let mcap_odom = pelorus_odom.clone();

                let mcap_tf = TransformStamped {
                    header: Header {
                        frame_id: config.sync_params.map_frame.clone(),
                        stamp: Time {
                            sec: sync_tick.odom_time.sec,
                            nanosec: sync_tick.odom_time.nanosec,
                        },
                    },
                    child_frame_id: pelorus_odom.child_frame_id.clone(),
                    transform: Transform {
                        translation: Vector3 {
                            x: t_world.translation.vector.x,
                            y: t_world.translation.vector.y,
                            z: t_world.translation.vector.z,
                        },
                        rotation: Quaternion {
                            x: t_world.rotation.quaternion().i,
                            y: t_world.rotation.quaternion().j,
                            z: t_world.rotation.quaternion().k,
                            w: t_world.rotation.quaternion().w,
                        },
                    },
                };

                writer.write_cloud(&serde_cloud(&cloud))?;
                writer.write_odom(&serde_odom(&mcap_odom))?;
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
                                stamp: Time {
                                    sec: pelorus_odom.header.stamp.sec,
                                    nanosec: pelorus_odom.header.stamp.nanosec,
                                },
                            },
                            poses: lidar_path_poses.iter().skip(skip).cloned().collect(),
                        };
                        writer.write_path(&serde_path(&lidar_path))?;
                    }
                }
                writer.write_tf(&serde_tf_message(&mcap_tf))?;
            }
            drain_filtered_imu_mcap(&mut mcap_writer, &mut filtered_imu_rx)?;
            if gate_minot_replay_on_mcap {
                if let Some(ready) = registration_ready_tx.as_ref() {
                    let _ = ready.send(sync_tick.scan_id);
                }
            } else if !relay_minot_replay_on_registration
                && minot_replay_mode
                && !downstream_owns_replay
            {
                let request_count = if minot_prefill_requested {
                    1
                } else {
                    minot_prefill_requested = true;
                    offline_prefetch_scans.max(1)
                };
                release_minot_ticks(
                    processing_shutdown_flag.as_ref(),
                    synchronous_mode,
                    request_count,
                );
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

            #[cfg(feature = "rerun")]
            if let Some(ref rec) = rec {
                #[cfg(feature = "ros2-client")]
                let ps = impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(cloud.clone());
                #[cfg(feature = "mt_pubsub")]
                let ps = impl_ros2_interfaces_jazzy_rkyv::to_pointcloud2_msg(cloud.clone());
                #[cfg(feature = "r2r")]
                let ps = impl_r2r::to_pointcloud2_msg(cloud.clone());
                #[cfg(feature = "hiroz")]
                let ps = impl_hiroz::to_pointcloud2_msg(cloud.clone());

                let gradient = colorous::TURBO;
                let t_world_lidar_rerun = (ros_to_rerun * register_pose.pose).cast();
                let (positions, colors): (Vec<_>, Vec<_>) = ps
                    .try_into_iter()
                    .unwrap()
                    .map(|v: PointXYZI| {
                        let color = gradient.eval_continuous(v.intensity as f64 / 255.0);
                        let v = t_world_lidar_rerun * nalgebra::Point3::new(v.x, v.y, v.z);
                        let pos = glam::Vec3::new(v.x, v.y, v.z);
                        let color_rgb = [color.r, color.g, color.b];

                        (pos, color_rgb)
                    })
                    .unzip();

                if let Err(err) = rec.log(
                    "world/latest_scan",
                    &rerun::Points3D::update_fields()
                        .with_positions(positions)
                        .with_colors(colors)
                        .with_radii([rerun::Radius::new_ui_points(1.25)]), // for video scene_points: 0.025
                ) {
                    error!("Error sending to rerun {err}");
                }
            }

            #[cfg(feature = "rerun")]
            if let Some(ref rec) = rec {
                let map_config = &config.rerun.map;
                let due = map_config.enabled
                    && last_map_snapshot_time
                        .map(|last| {
                            current_cloud_time < last
                                || current_cloud_time - last >= map_config.publish_interval_sec
                        })
                        .unwrap_or(true);
                if due {
                    last_map_snapshot_time = Some(current_cloud_time);

                    #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
                    {
                        if config.map_multiresolution {
                            let distributions = map.lock().unwrap().get_distributions_with_levels();
                            log_multires_snapshot_to_rerun(rec, distributions, map_config);
                        } else {
                            let distributions =
                                map.lock().unwrap().get_distributions_with_lru_scores();
                            log_map_snapshot_to_rerun(rec, distributions, map_config);
                        }
                    }

                    #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
                    {
                        #[cfg(feature = "cuda")]
                        let total_map = pelorus_cuda::get_map_with_recency();
                        #[cfg(feature = "metal")]
                        let total_map = pelorus_metal::get_map_with_recency();
                        #[cfg(feature = "vulkan")]
                        let total_map = pelorus_vulkan::get_map_with_recency();

                        // GPU backends have no true LRU list, but every backend's map
                        // cell carries the scan it was last touched on. A raw age-based
                        // decay would make almost the entire map read as "irrelevant"
                        // within a scan or two (a LiDAR does not revisit every voxel
                        // every tick), so relevance is ranked by recency *within each
                        // level* instead, same as the CPU multires path
                        // (see `rank_scores_by_recency`). Each level gets its own
                        // entity path so it can be toggled independently in Rerun.
                        const LEVELS: usize = 4;
                        let current_scan = sync_tick.scan_id;
                        let mut positions: [Vec<glam::Vec3>; LEVELS] = Default::default();
                        let mut ages: [Vec<u64>; LEVELS] = Default::default();
                        for (v, level, last_scan) in total_map {
                            let level = (level as usize).min(LEVELS - 1);
                            let v = ros_to_rerun.cast().transform_point(&v);
                            let pos = glam::Vec3::new(v.x, v.y, v.z);
                            positions[level].push(pos);
                            ages[level].push(current_scan.saturating_sub(last_scan as u64));
                        }
                        for level in 0..LEVELS {
                            let colors: Vec<rerun::Color> =
                                pelorus_localmap::rank_scores_by_recency(&ages[level])
                                    .into_iter()
                                    .map(relevance_color)
                                    .collect();
                            if let Err(err) = rec.log(
                                format!("world/map/level{level}"),
                                &rerun::Points3D::update_fields()
                                    .with_positions(core::mem::take(&mut positions[level]))
                                    .with_colors(colors),
                            ) {
                                error!("Error sending map snapshot to rerun {err}");
                            }
                        }
                    }
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

                    let marker = pelorus_protocol::RegisteredFrameMarker {
                        protocol_version: pelorus_protocol::PROTOCOL_VERSION,
                        scan_id: sync_tick.scan_id,
                        timestamp_sec: pelorus_odom.header.stamp.sec,
                        timestamp_nanosec: pelorus_odom.header.stamp.nanosec,
                        // The negotiated encoding, not the configured one: a
                        // consumer that could not decode Cloudini was
                        // downgraded at registration, and the marker has to
                        // describe the cloud that was actually published.
                        cloud_encoding: if marker_compressed_clouds.load(Ordering::Acquire) {
                            pelorus_protocol::CloudEncoding::CompressedPointCloud2
                        } else {
                            pelorus_protocol::CloudEncoding::PointCloud2
                        },
                        cloud_frame: cloud.header.frame_id.clone(),
                        odometry_frame: pelorus_odom.header.frame_id.clone(),
                    };
                    registered_frame_publisher
                        .publish(&marker)
                        .await
                        .expect("Failed to publish registered frame marker");
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

            #[cfg(feature = "mt_pubsub")]
            if downstream_owns_replay {
                debug!("published downstream frame {}", sync_tick.scan_id);
            }

            // Release the real-time barrier only after the authoritative
            // corrected message has been published. The predictor then
            // replays buffered IMUs and emits one caught-up state.
            if let Some(predictor) = realtime_predictor.as_ref() {
                let registration = pipeline.map();
                let registration = registration.lock().unwrap();
                predictor.corrected(&registration);
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
                        rerun::LineStrips3D::new(core::iter::once(lidar_path_strip))
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
                            stamp: pelorus_odom.header.stamp,
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

            #[cfg(all(not(feature = "no-viz"), feature = "mt_pubsub"))]
            let pub_lidar_path = Path {
                header: Header {
                    frame_id: lidar_path_frame_id.clone(),
                    stamp: pelorus_odom.header.stamp.clone(),
                },
                poses: lidar_path_poses.iter().cloned().collect(),
            };
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
                });

            let iteration_time_ms = eval.get_sum_of_top_level_times_ms();

            #[cfg(feature = "processing-time-export")]
            if let Some(processing_time_file) = processing_time_file.as_mut() {
                let processing_time_ns = (iteration_time_ms * 1_000_000.0).round() as u128;
                let content_to_write = format!("{current_cloud_time} {processing_time_ns}\n");

                std::io::Write::write_all(processing_time_file, content_to_write.as_bytes())?;
            }

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

            #[cfg(feature = "hiroz")]
            if let Some(ref stats) = shared_stats
                && last_stats_pub.elapsed() >= Duration::from_millis(250)
            {
                last_stats_pub = std::time::Instant::now();
                stats.update_system_stats();
                let msg = stats_message(stats);
                let _ = rt
                    .as_ref()
                    .expect("rt always exists with hiroz feat")
                    .block_on(async { stats_publisher.async_publish(&msg).await });
            }

            // Debug output (non-TUI timing metrics)
            if let Some(debug_interval) = debug_render_interval
                && last_render.elapsed() > debug_interval
            {
                last_render = std::time::Instant::now();
                // Only print debug output if TUI is not running
                // TUI runs when show_tui is true AND we're in a terminal
                let tui_is_running = config.show_tui && std::io::stdout().is_terminal();
                if !tui_is_running {
                    println!(
                        "{}[2J{}[1;L{}\n{}",
                        27 as char,
                        27 as char,
                        NAME,
                        eval.render()
                    );
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
        // Release GPU buffers before the backend tears the device down. Every
        // child object has to be gone before the device is destroyed.
        pipeline.release_gpu_buffers();
        #[cfg(feature = "vulkan")]
        if matches!(active_backend, ActiveBackend::Vulkan(_)) {
            drop(core::mem::replace(
                &mut persistent_cloud_buf,
                LocatedBufferView::Ram(Vec::new()),
            ));
        }
        match &active_backend {
            ActiveBackend::Cpu => {}
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(_) => pelorus_cuda::cleanup_map(),
            #[cfg(feature = "metal")]
            ActiveBackend::Metal(_) => pelorus_metal::cleanup_map(),
            #[cfg(feature = "vulkan")]
            // Releases the map *and* the downsample state, so the device and
            // instance are destroyed when `active_backend` drops rather than
            // being left alive for the NVIDIA driver's atexit handler to race
            // against its own worker threads.
            ActiveBackend::Vulkan(_) => pelorus_vulkan::shutdown(),
        }
        drain_filtered_imu_mcap(&mut mcap_writer, &mut filtered_imu_rx)?;
        if let Some(child_frame) = realtime_export_child_frame.as_deref() {
            drain_realtime_odometry_exports(
                &realtime_export_rx,
                &mut trajectory_samples,
                &mut mcap_writer,
                #[cfg(feature = "rerun")]
                if minot_replay_mode {
                    rec.as_ref()
                } else {
                    None
                },
                &config.sync_params.map_frame,
                child_frame,
            )?;
        }
        if let (Some(file), Some(samples)) = (trajectory_file.as_mut(), trajectory_samples.as_mut())
        {
            samples.sort_by(|left, right| left.0.total_cmp(&right.0));
            for (stamp, pose) in samples.iter() {
                let t = pose.translation.vector;
                let q = pose.rotation.quaternion();
                let line = format!(
                    "{} {} {} {} {} {} {} {}\n",
                    stamp, t.x, t.y, t.z, q.i, q.j, q.k, q.w
                );
                std::io::Write::write_all(file, line.as_bytes())?;
            }
        }
        if let Some(writer) = mcap_writer.take() {
            writer.finish()?;
        }
        info!("--- END sync tick ---");
        Ok(())
    });

    #[cfg(any(feature = "mt_pubsub", feature = "r2r", feature = "hiroz"))]
    let spinner_shutdown_flag = shutdown_flag.clone();

    #[cfg(feature = "mt_pubsub")]
    {
        // Poll so shutdown remains responsive while processing continues.
        loop {
            if processing_handle.is_finished() {
                info!("Processing thread finished in mt_pubsub spinner");
                break;
            }
            if let Some(ref flag) = spinner_shutdown_flag
                && flag.load(Ordering::Relaxed)
            {
                info!("Shutdown flag detected in mt_pubsub spinner");
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

    #[cfg(feature = "ros2-client")]
    {
        // ros2-client doesn't have a simple way to break the spinner, so we run it in a thread
        // and check the shutdown flag periodically
        let spinner_shutdown = shutdown_flag.clone();
        let _spinner_handle = std::thread::spawn(move || {
            future::block_on(async {
                if let Err(e) = node.spinner().unwrap().spin().await {
                    error!("Node did not exit successfully {e}");
                }
            });
        });

        // Monitor shutdown flag and wait for processing to complete
        loop {
            if let Some(ref flag) = spinner_shutdown
                && flag.load(Ordering::Relaxed)
            {
                info!("Shutdown flag detected, waiting for processing thread");
                break;
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
            node.spin_once(spin_timeout);
            if let Some(ref flag) = spinner_shutdown_flag
                && flag.load(Ordering::Relaxed) {
                    info!("Shutdown flag detected in r2r spinner");
                    break;
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

    #[cfg(feature = "hiroz")]
    {
        loop {
            if processing_handle.is_finished() {
                info!("Processing thread finished in Hiroz backend");
                break;
            }
            if let Some(ref flag) = spinner_shutdown_flag
                && flag.load(Ordering::Relaxed)
            {
                info!("Shutdown flag detected in Hiroz backend");
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

    #[cfg(not(any(
        feature = "mt_pubsub",
        feature = "ros2-client",
        feature = "r2r",
        feature = "hiroz"
    )))]
    {
        match processing_handle.join() {
            Ok(Ok(())) => Ok(()),
            Ok(Err(err)) => Err(err.into()),
            Err(_) => Err(anyhow!("LIO processing thread panicked").into()),
        }
    }
}

#[cfg(not(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
)))]
fn lio_with_ready_signal(
    _config: LioConfig,
    _options: LioRunOptions,
) -> Result<(), Box<dyn CoreError>> {
    Err(anyhow!(
        "No IO backend is enabled. Rebuild with one of: offline, mt_pubsub, ros2-client, r2r, or hiroz."
    )
    .into())
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

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
enum OfflineShutdownRequest {
    Immediate,
    DrainAfterEndOfBag,
}

#[cfg(feature = "rat")]
#[derive(Debug)]
struct MinotProcessGuard {
    pid: u32,
    owns_process_group: bool,
}

#[cfg(feature = "rat")]
impl MinotProcessGuard {
    fn new(pid: u32, owns_process_group: bool) -> Self {
        #[cfg(unix)]
        ACTIVE_MINOT_PGID.store(
            if owns_process_group {
                pid as i32
            } else {
                -(pid as i32)
            },
            Ordering::SeqCst,
        );
        Self {
            pid,
            owns_process_group,
        }
    }

    fn kill(&self) -> std::io::Result<()> {
        if self.owns_process_group {
            signal_process_group(self.pid, libc::SIGKILL)
        } else {
            signal_process(self.pid, libc::SIGKILL)
        }
    }
}

#[cfg(feature = "rat")]
impl Drop for MinotProcessGuard {
    fn drop(&mut self) {
        if let Err(e) = self.kill() {
            warn!(
                "Failed to kill Minot process group {} during cleanup: {}",
                self.pid, e
            );
        }
        #[cfg(unix)]
        {
            let _ = ACTIVE_MINOT_PGID.compare_exchange(
                if self.owns_process_group {
                    self.pid as i32
                } else {
                    -(self.pid as i32)
                },
                0,
                Ordering::SeqCst,
                Ordering::SeqCst,
            );
        }
    }
}

#[cfg(feature = "rat")]
fn recv_offline_runtime_event(
    receiver: &std::sync::mpsc::Receiver<OfflineRuntimeEvent>,
    signal_handled: &mut bool,
) -> anyhow::Result<OfflineRuntimeEvent> {
    loop {
        if !*signal_handled && PROCESS_SIGNAL_SHUTDOWN.load(Ordering::SeqCst) {
            *signal_handled = true;
            return Ok(OfflineRuntimeEvent::TuiShutdownRequested);
        }

        match receiver.recv_timeout(Duration::from_millis(50)) {
            Ok(event) => return Ok(event),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err(anyhow!("Offline runtime channel closed unexpectedly"));
            }
        }
    }
}

#[cfg(feature = "rat")]
fn join_minot_wait_thread(handle: std::thread::JoinHandle<()>) -> anyhow::Result<()> {
    handle
        .join()
        .map_err(|_| anyhow!("Minot wait thread panicked"))?;
    Ok(())
}

#[cfg(feature = "rat")]
fn signal_process_group(pgid: u32, signal: libc::c_int) -> std::io::Result<()> {
    let rc = unsafe { libc::kill(-(pgid as i32), signal) };
    if rc == 0 {
        return Ok(());
    }

    let err = std::io::Error::last_os_error();
    if err.raw_os_error() == Some(libc::ESRCH) {
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    if err.raw_os_error() == Some(libc::EPERM) {
        // macOS can return EPERM for a process group whose leader is exiting or
        // has already been reaped. Fall back to the Minot leader PID so an
        // actually running child is still terminated. ESRCH means it has exited.
        let leader_rc = unsafe { libc::kill(pgid as i32, signal) };
        if leader_rc == 0 {
            return Ok(());
        }

        let leader_err = std::io::Error::last_os_error();
        if leader_err.raw_os_error() == Some(libc::ESRCH) {
            return Ok(());
        }
        return Err(leader_err);
    }

    Err(err)
}

#[cfg(feature = "rat")]
fn signal_process(pid: u32, signal: libc::c_int) -> std::io::Result<()> {
    let rc = unsafe { libc::kill(pid as i32, signal) };
    if rc == 0 {
        return Ok(());
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(error)
    }
}

#[cfg(all(test, feature = "rat", target_os = "linux"))]
mod minot_process_guard_tests {
    use std::io::{BufRead, BufReader};

    use super::*;

    fn process_is_running(pid: u32) -> bool {
        let Ok(stat) = fs::read_to_string(format!("/proc/{pid}/stat")) else {
            return false;
        };
        stat.split_whitespace().nth(2) != Some("Z")
    }

    #[test]
    fn kills_descendants_after_group_leader_exits() {
        let mut command = ProcessCommand::new("sh");
        command
            .args([
                "-c",
                "sleep 30 </dev/null >/dev/null 2>&1 & child=$!; echo \"$child\"",
            ])
            .stdout(Stdio::piped());
        unsafe {
            command.pre_exec(|| {
                if libc::setpgid(0, 0) == 0 {
                    Ok(())
                } else {
                    Err(std::io::Error::last_os_error())
                }
            });
        }

        let mut leader = command
            .spawn()
            .expect("failed to spawn process-group leader");
        let guard = MinotProcessGuard::new(leader.id(), true);
        let stdout = leader.stdout.take().expect("missing leader stdout");
        let mut child_pid = String::new();
        BufReader::new(stdout)
            .read_line(&mut child_pid)
            .expect("failed to read descendant PID");
        let child_pid = child_pid
            .trim()
            .parse::<u32>()
            .expect("invalid descendant PID");

        assert!(leader.wait().expect("failed to wait for leader").success());
        assert!(process_is_running(child_pid));

        drop(guard);

        for _ in 0..100 {
            if !process_is_running(child_pid) {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        panic!("descendant {child_pid} survived Minot process-group cleanup");
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

fn log_active_backend(config: &LioConfig, active_backend: &ActiveBackend) {
    match active_backend {
        ActiveBackend::Cpu if config.cpu_only => info!("Backend: CPU (forced)"),
        ActiveBackend::Cpu => info!("Backend: CPU"),
        #[cfg(feature = "cuda")]
        ActiveBackend::Cuda(_) => info!("Backend: CUDA"),
        #[cfg(feature = "metal")]
        ActiveBackend::Metal(_) => info!("Backend: Metal"),
        #[cfg(feature = "vulkan")]
        ActiveBackend::Vulkan(_) => info!("Backend: Vulkan"),
    }
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

#[cfg(all(test, feature = "r2r"))]
mod ros_args_tests {
    use super::args_without_ros_args;
    use std::ffi::OsString;

    fn strings(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    #[test]
    fn removes_ros_args_section_after_pelorus_args() {
        let args = args_without_ros_args(strings(&[
            "pelorus",
            "online",
            "-c",
            "pelorus.rl",
            "--ros-args",
            "-r",
            "__node:=lidar_odom",
            "-r",
            "__ns:=/robot",
            "--params-file",
            "/tmp/launch_params.yaml",
        ]));

        assert_eq!(args, strings(&["pelorus", "online", "-c", "pelorus.rl"]));
    }

    #[test]
    fn keeps_pelorus_args_after_ros_args_delimiter() {
        let args = args_without_ros_args(strings(&[
            "pelorus",
            "--ros-args",
            "--log-level",
            "debug",
            "--",
            "online",
            "--local-only",
        ]));

        assert_eq!(args, strings(&["pelorus", "online", "--local-only"]));
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

#[cfg(any(
    feature = "mt_pubsub",
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz"
))]
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
fn build_offline_minot_config(
    bagfile_path: &str,
    lidar_topic: &str,
    imu_topic: &str,
    lidar_only: bool,
) -> String {
    build_offline_minot_config_with_topics(bagfile_path, lidar_topic, imu_topic, lidar_only, &[])
}

/// [`build_offline_minot_config`] plus extra bag topics.
///
/// Every `pf!` line that carries TF carries the extra topics too, so they are
/// published in bag order between two clouds and never decide when a scan is
/// released.
#[cfg(feature = "rat")]
fn build_offline_minot_config_with_topics(
    bagfile_path: &str,
    lidar_topic: &str,
    imu_topic: &str,
    lidar_only: bool,
    forwarded: &[ForwardedTopic],
) -> String {
    let mut extra_sensors = String::new();
    let mut extra_ids = String::new();
    for (index, forward) in forwarded.iter().enumerate() {
        extra_sensors.push_str(&format!(
            "\n  fwd{index}.{{\n    _topic = \"{}\"\n    _short = \"x{index}\"\n    _type = {}\n  }}",
            forward.topic, forward.minot_type
        ));
        extra_ids.push_str(&format!(", x{index}"));
    }
    // TF from the bag is always forwarded so configs that use the TF tree resolve their
    // frames offline. A bag without these topics simply never matches them.
    let tf_sensors = r#"
  tf.{
    _topic = "/tf"
    _short = "t"
    _type = Tf
  }
  tf_static.{
    _topic = "/tf_static"
    _short = "s"
    _type = Tf
  }"#;

    let script = if lidar_only {
        format!(
            r#"
_bag.{{
  cloud.{{
    _topic = "{lidar_topic}"
    _short = "l"
    _type = Cloud
  }}{tf_sensors}{extra_sensors}
}}
_start_locked = false

# Lidar-only replay has no IMU subscriber or initialization handshake.
# Each tick publishes one cloud together with the TF around it, stopping before the next cloud.
pf! [l, t, s], l 2 tick_end

reset! "{bagfile_path}"

# Start the feedback loop with the first cloud.
pf! [l, t, s], l 2
"#
        )
    } else {
        format!(
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
  }}{tf_sensors}{extra_sensors}
}}
_start_locked = false

# Setting triggers for topics.
# During IMU bootstrap, advance one IMU (with the TF around it) and stop before the next IMU.
# The first active LIO tick advances to one cloud with its surrounding IMUs. After that, each
# tick_end publishes one cloud and then advances IMUs up to the next cloud without consuming
# that next cloud. TF is sent along with the IMUs.
pf! [i, t, s], i 2 imu_init_tick
pf! [i, t, s], l 1 lio_start_tick
pf! l 1 lio_start_tick
pf! [i, t, s], l 1 lio_start_tick
pf! l 1 tick_end
pf! [i, t, s], l 1 tick_end

reset! "{bagfile_path}"

# Starting the feedback loop by playing the first IMU message (first step is IMU init)
pf! [i, t, s], i 2
"#
        )
    };
    script
        .replace("[l, t, s]", &format!("[l, t, s{extra_ids}]"))
        .replace("[i, t, s]", &format!("[i, t, s{extra_ids}]"))
}

#[cfg(all(test, feature = "rat"))]
mod offline_minot_config_tests {
    use super::{ForwardedTopic, build_offline_minot_config, build_offline_minot_config_with_topics};

    #[test]
    fn lidar_only_replay_requests_clouds_without_declaring_imu() {
        let config =
            build_offline_minot_config("/tmp/input.mcap", "/lidar/points", "/imu/data", true);

        assert!(config.contains("_topic = \"/lidar/points\""));
        assert!(config.contains("pf! [l, t, s], l 2 tick_end"));
        assert!(config.contains("pf! [l, t, s], l 2\n"));
        assert!(!config.contains("/imu/data"));
        assert!(!config.contains("_type = Imu"));
        assert!(!config.contains("imu_init_tick"));
        assert!(!config.contains("lio_start_tick"));
    }

    #[test]
    fn lidar_imu_replay_keeps_the_imu_bootstrap_schedule() {
        let config =
            build_offline_minot_config("/tmp/input.mcap", "/lidar/points", "/imu/data", false);

        assert!(config.contains("_topic = \"/imu/data\""));
        assert!(config.contains("pf! [i, t, s], i 2 imu_init_tick"));
        assert!(config.contains("pf! [i, t, s], l 1 lio_start_tick"));
        assert!(config.contains("pf! [i, t, s], l 1 tick_end"));
        assert!(config.contains("pf! [i, t, s], i 2\n"));
    }

    #[test]
    fn forwarded_topics_ride_with_tf_and_never_hold_a_scan() {
        let forwarded = [
            ForwardedTopic {
                minot_type: "ImageCompressed",
                topic: "/cam/image_raw/compressed".into(),
            },
            ForwardedTopic {
                minot_type: "CameraInfo",
                topic: "/cam/camera_info".into(),
            },
        ];
        for lidar_only in [true, false] {
            let config = build_offline_minot_config_with_topics(
                "/tmp/input.mcap",
                "/lidar/points",
                "/imu/data",
                lidar_only,
                &forwarded,
            );
            assert!(config.contains("_topic = \"/cam/image_raw/compressed\""));
            assert!(config.contains("_type = ImageCompressed"));
            assert!(config.contains("_short = \"x1\""));
            assert!(config.contains("_type = CameraInfo"));
            assert!(!config.contains("[l, t, s]") && !config.contains("[i, t, s]"));
            let tick = if lidar_only {
                "pf! [l, t, s, x0, x1], l 2 tick_end"
            } else {
                "pf! [i, t, s, x0, x1], l 1 tick_end"
            };
            assert!(config.contains(tick), "{config}");
            // Nothing forwarded: byte for byte the established schedule.
            assert_eq!(
                build_offline_minot_config("/tmp/input.mcap", "/lidar/points", "/imu/data", lidar_only),
                build_offline_minot_config_with_topics(
                    "/tmp/input.mcap", "/lidar/points", "/imu/data", lidar_only, &[]
                )
            );
        }
    }

    #[test]
    fn forward_topic_arguments_are_parsed_and_validated() {
        use super::parse_forwarded_topic;
        assert_eq!(
            parse_forwarded_topic("compressed:/a/b").unwrap(),
            ForwardedTopic { minot_type: "ImageCompressed", topic: "/a/b".into() }
        );
        assert_eq!(parse_forwarded_topic("image:/x").unwrap().minot_type, "Image");
        assert_eq!(parse_forwarded_topic("camera-info:/x").unwrap().minot_type, "CameraInfo");
        for bad in ["/no/type", "depth:/x", "image:", "image:/a b", "image:/a\"b"] {
            assert!(parse_forwarded_topic(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn replay_always_forwards_tf_topics() {
        for lidar_only in [true, false] {
            let config = build_offline_minot_config(
                "/tmp/input.mcap",
                "/lidar/points",
                "/imu/data",
                lidar_only,
            );

            assert!(config.contains("_topic = \"/tf\""));
            assert!(config.contains("_topic = \"/tf_static\""));
            assert_eq!(config.matches("_type = Tf").count(), 2);
        }
    }
}

/// Loads configuration the same way `pelorus online` does, then hands over to
/// the raw driver runner.
#[cfg(feature = "mt_pubsub")]
fn run_raw(
    default_config: &str,
    args: RawArgs,
    cli_overrides: &[pelorus_core::config::ConfigOverride],
) -> Result<(), Box<dyn CoreError>> {
    let default_ast = ratslang::compile_code(default_config)
        .map_err(|e| anyhow!("Failed to compile embedded default config: {}", e))?;

    let user_ast = load_user_config_with_checked_overrides(
        default_config,
        args.config_path.as_deref(),
        cli_overrides,
    )?;
    let config = ast_to_config(default_ast, user_ast)?;

    update_log_level();
    // Raw mode leaves the terminal available for driver logs.
    set_tui_active(false);

    let intrinsics = match &args.mode {
        Some(RawMode::Intrinsics(intrinsics)) => {
            // Guided calibration accepts IMU-only recordings.
            if args.sensor != RawSensor::Imu {
                return Err(anyhow::anyhow!(
                    "`raw ... intrinsics` guides an IMU calibration, so it only applies to \
                     `pelorus online raw imu`."
                )
                .into());
            }
            // Guided sessions require a saved recording for calibration.
            if args.recording.output_rosbag.is_none() && args.recording.output_mcap.is_none() {
                return Err(anyhow::anyhow!(
                    "`raw imu intrinsics` needs somewhere to put the recording. \
                     Pass --output-rosbag <DIR>."
                )
                .into());
            }
            Some(pelorus_cli::raw::IntrinsicsOptions {
                poses: intrinsics.poses,
            })
        }
        None => None,
    };

    let options = pelorus_cli::raw::RawOptions {
        sensor: args.sensor,
        intrinsics,
        output_mcap: args.recording.output_mcap,
        output_rosbag: args.recording.output_rosbag,
        output_rosbag_metadata: args.recording.output_rosbag_metadata,
        mcap_cloud_topic: args.recording.mcap_cloud_topic,
        mcap_imu_topic: args.recording.mcap_imu_topic,
        local_only: args.local_only,
        coord_mode: args.coord_mode,
        yes: args.yes,
    };

    let shutdown = Arc::new(AtomicBool::new(false));
    install_online_shutdown_signal_handler(Arc::clone(&shutdown));

    #[cfg(feature = "native-drivers")]
    return pelorus_cli::raw::run(config, options, shutdown).map_err(Into::into);

    #[cfg(not(feature = "native-drivers"))]
    {
        let _ = (config, options, shutdown);
        Err(anyhow!(
            "`pelorus online raw` needs a native sensor driver. Rebuild with the `sensor-drivers` feature (or one of `velodyne-driver`, `xsens-driver`, `ouster-driver`, `robosense-driver`)."
        )
        .into())
    }
}

fn main() -> Result<(), Box<dyn CoreError>> {
    completions::activate_dynamic_completions(Args::command, "pelorus");

    // Initialize our custom logger that works with both TUI and console
    init_logger();

    #[cfg(feature = "r2r")]
    let args = Args::parse_from(args_without_ros_args(std::env::args_os()));
    #[cfg(not(feature = "r2r"))]
    let args = Args::parse();

    const DEFAULT_CONFIG_STR: &str = pelorus_core::DEFAULT_CONFIG_STR;

    // Parsed before anything is spawned: a malformed --override should fail on
    // the command line, not partway into a run.
    let cli_overrides = parse_overrides(&args.overrides)?;

    match args.command {
        Command::Completions { shell } => {
            if let Some(shell) = shell {
                let mut cmd = Args::command();
                generate(
                    shell,
                    &mut cmd,
                    "pelorus".to_owned(),
                    &mut std::io::stdout(),
                );
                Ok(())
            } else {
                completions::install("pelorus").map_err(Into::into)
            }
        }
        /* `run` wrapper removed; `online`/`offline` are top-level commands */
        Command::Online(online_args) => {
            #[cfg(feature = "mt_pubsub")]
            if let Some(OnlineMode::Raw(raw_args)) = online_args.mode {
                return run_raw(DEFAULT_CONFIG_STR, raw_args, &cli_overrides);
            }
            let config_args = online_args.pipeline;
            let default_ast = ratslang::compile_code(DEFAULT_CONFIG_STR)
                .map_err(|e| anyhow!("Failed to compile embedded default config: {}", e))?;

            let user_ast = load_user_config_with_checked_overrides(
                DEFAULT_CONFIG_STR,
                config_args.config_path.as_deref(),
                &cli_overrides,
            )?;

            let config = ast_to_config(default_ast, user_ast)?;

            let output_mcap_config = build_mcap_output_config(&config);
            if !prepare_mcap_output_targets(
                &config.mcap,
                output_mcap_config.as_ref(),
                config_args.yes,
            )? {
                return Ok(());
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
                    "Ignoring num_threads={}. Thread-pool configuration requires the rayon feature.",
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
                #[cfg(feature = "processing-time-export")]
                let lio_export_latency = config_args.export_latency.clone();
                #[cfg(feature = "processing-time-export")]
                let lio_export_processing_time = config_args.export_processing_time.clone();
                let lio_sync = config_args.rat;
                let lio_stats = Some(shared_stats);
                let lio_output_mcap = output_mcap_config.clone();
                let lio_local_only = config_args.local_only;
                let lio_coord_mode = config_args.coord_mode;
                #[cfg(feature = "r2r")]
                let lio_ros_node_args = RosNodeArgs {
                    node_name: config_args.node_name.clone(),
                    namespace: config_args.namespace.clone(),
                };
                #[cfg(feature = "hiroz")]
                let lio_hiroz_node_args = HirozNodeArgs {
                    mode: config_args.hiroz_mode.clone(),
                    endpoints: config_args.hiroz_endpoint.clone(),
                };

                let lio_shutdown_flag = Arc::new(AtomicBool::new(false));
                let lio_shutdown_flag_for_thread = lio_shutdown_flag.clone();

                let lio_handle = std::thread::spawn(move || -> Result<(), String> {
                    lio_with_ready_signal(
                        lio_config,
                        LioRunOptions {
                            ready_signal: None,
                            export_trajectory: lio_export,
                            #[cfg(feature = "processing-time-export")]
                            export_latency: lio_export_latency,
                            #[cfg(feature = "processing-time-export")]
                            export_processing_time: lio_export_processing_time,
                            synchronous_mode: lio_sync,
                            #[cfg(feature = "r2r")]
                            ros_node_args: lio_ros_node_args,
                            #[cfg(feature = "hiroz")]
                            hiroz_node_args: lio_hiroz_node_args,
                            shared_stats: lio_stats,
                            shutdown_flag: Some(lio_shutdown_flag_for_thread),
                            shutdown_request_rx: None,
                            offline_prefetch_scans: 1,
                            replay_tick_owner: ReplayTickOwner::Pelorus,
                            downstream_registered: None,
                            mt_pubsub_local_only: lio_local_only,
                            mt_pubsub_coord_mode: lio_coord_mode,
                            output_mcap: lio_output_mcap,
                        },
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
                let lio_shutdown_flag = Arc::new(AtomicBool::new(false));
                install_online_shutdown_signal_handler(Arc::clone(&lio_shutdown_flag));
                lio_with_ready_signal(
                    config,
                    LioRunOptions {
                        ready_signal: None,
                        export_trajectory: config_args.export_trajectory,
                        #[cfg(feature = "processing-time-export")]
                        export_latency: config_args.export_latency,
                        #[cfg(feature = "processing-time-export")]
                        export_processing_time: config_args.export_processing_time,
                        synchronous_mode: config_args.rat,
                        #[cfg(feature = "r2r")]
                        ros_node_args: RosNodeArgs {
                            node_name: config_args.node_name,
                            namespace: config_args.namespace,
                        },
                        #[cfg(feature = "hiroz")]
                        hiroz_node_args: HirozNodeArgs {
                            mode: config_args.hiroz_mode,
                            endpoints: config_args.hiroz_endpoint,
                        },
                        shared_stats: None,
                        shutdown_flag: Some(lio_shutdown_flag),
                        shutdown_request_rx: None,
                        offline_prefetch_scans: 1,
                        replay_tick_owner: ReplayTickOwner::Pelorus,
                        downstream_registered: None,
                        mt_pubsub_local_only: config_args.local_only,
                        mt_pubsub_coord_mode: config_args.coord_mode,
                        output_mcap: output_mcap_config,
                    },
                )
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

            let user_ast = load_user_config_with_checked_overrides(
                DEFAULT_CONFIG_STR,
                offline_args.config_path.as_deref(),
                &cli_overrides,
            )?;

            let config = {
                let mut config = ast_to_config(default_ast, user_ast)?;
                config.sync_node_params.synchronous_mode = true;
                config
            };

            let output_mcap_config = build_mcap_output_config(&config);
            if !prepare_mcap_output_targets(
                &config.mcap,
                output_mcap_config.as_ref(),
                offline_args.yes,
            )? {
                return Ok(());
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
                    "Ignoring num_threads={}. Thread-pool configuration requires the rayon feature.",
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

            let lidar_topic = &config.sync_node_params.cloud_src_topic;
            let imu_topic = &config.sync_node_params.imu_src_topic;
            // Resolve local paths before writing the temporary .mt file.
            let bagfile_path = if offline_args.bagfile_path.exists() {
                canonicalize_bagfile_path(&offline_args.bagfile_path)?
                    .to_string_lossy()
                    .into_owned()
            } else {
                offline_args.bagfile_path.to_string_lossy().into_owned()
            };
            if bagfile_path.contains('"') {
                return Err(
                    anyhow!("Bagfile paths and Marina tags cannot contain a double quote").into(),
                );
            }
            let bagfile_label = bagfile_path.clone();
            shared_stats.set_bagfile(bagfile_label);

            let mt_content = build_offline_minot_config_with_topics(
                &bagfile_path,
                lidar_topic,
                imu_topic,
                config.imu_params.lidar_only,
                &offline_args.forward_topics,
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
            #[cfg(feature = "processing-time-export")]
            let export_latency_path =
                resolve_export_timing_path(offline_args.export_latency.clone(), "latency");
            #[cfg(feature = "processing-time-export")]
            let export_processing_time_path = resolve_export_timing_path(
                offline_args.export_processing_time.clone(),
                "processing_time",
            );
            let export_trajectory_path_for_run = export_trajectory_path.clone();
            #[cfg(feature = "processing-time-export")]
            let export_latency_path_for_run = export_latency_path.clone();
            #[cfg(feature = "processing-time-export")]
            let export_processing_time_path_for_run = export_processing_time_path.clone();
            let offline_local_only = !offline_args.open_router;
            let run_offline_logic = || -> anyhow::Result<()> {
                let startup_started = std::time::Instant::now();
                install_shutdown_signal_handlers();

                // Minot is spawned before the LIO node is initialized, so bootstrap
                // the shared coordinator here and keep it alive for the entire run.
                // Start mode avoids a discovery timeout when this process is the owner.
                #[cfg(feature = "mt_pubsub")]
                let (_coordinator_runtime, _coordinator_node) = {
                    let runtime = Arc::new(
                        tokio::runtime::Builder::new_multi_thread()
                            .enable_all()
                            .build()
                            .map_err(|e| anyhow!("Failed to create coordinator runtime: {e}"))?,
                    );
                    let node = runtime
                        .block_on(Node::create(
                            mt_pubsub::NodeConfig::new("pelorus_offline_coordinator")
                                .mode(mt_pubsub::Qos::Reliable)
                                .coord_mode(mt_pubsub::CoordMode::Start)
                                .local_only(offline_local_only),
                        ))
                        .map_err(|e| anyhow!("Failed to bootstrap offline coordinator: {e}"))?;
                    info!(
                        "Offline coordinator ready after {:.3}s",
                        startup_started.elapsed().as_secs_f64()
                    );
                    (runtime, node)
                };

                let mt_file_path = std::env::temp_dir().join("pelorus_offline.mt");
                fs::write(&mt_file_path, mt_content)
                    .map_err(|e| anyhow!("Failed to write .mt file: {}", e))?;

                info!("Generated .mt file at: {:?}", mt_file_path);

                // Spawn Minot in its own process group so cleanup can kill Minot and
                // any background coordinator it starts.
                let mut minot_command = ProcessCommand::new(&offline_args.minot_path);
                minot_command
                    .arg("headless")
                    .arg(&mt_file_path)
                    .arg("--sync")
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped());
                if let Some(registry) = &offline_args.registry {
                    minot_command.arg("--registry").arg(registry);
                }
                if offline_args.no_cache_stream {
                    minot_command.arg("--no-cache-stream");
                }
                // Request structured child-process log records from Minot.
                for (key, value) in mt_log::child_env("Minot") {
                    minot_command.env(key, value);
                }
                if offline_local_only {
                    minot_command.arg("--local-only");
                }
                // The turbine must publish in the mode Pelorus subscribes with.
                minot_command.env(
                    "wind_ratpub_coord_mode",
                    offline_args.coord_mode.minot_turbine_env(),
                );

                #[cfg(unix)]
                if !offline_args.polarstern {
                    unsafe {
                        minot_command.pre_exec(|| {
                            if libc::setpgid(0, 0) == 0 {
                                Ok(())
                            } else {
                                Err(std::io::Error::last_os_error())
                            }
                        });
                    }
                }

                let mut minot_child = minot_command
                    .spawn()
                    .map_err(|e| anyhow!("Failed to spawn minot: {}", e))?;

                let minot_pid = minot_child.id();
                let minot_guard = MinotProcessGuard::new(minot_pid, !offline_args.polarstern);
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
                                    relay_child_line(trimmed, "Minot");
                                }

                                if line.to_lowercase().contains("end of bagfile") {
                                    let _ =
                                        runtime_tx_stdout.send(OfflineRuntimeEvent::MinotEndOfBag);
                                }

                                if line.contains(
                                    "Waiting for input on stdin before execution (--sync mode)",
                                ) && !ready_notified_stdout.swap(true, Ordering::SeqCst)
                                {
                                    let _ = runtime_tx_stdout.send(OfflineRuntimeEvent::MinotReady);
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
                                    relay_child_line(trimmed, "Minot");
                                }

                                if line.to_lowercase().contains("end of bagfile") {
                                    let _ =
                                        runtime_tx_stderr.send(OfflineRuntimeEvent::MinotEndOfBag);
                                }

                                // Minot emits this only after its coordinator connection and
                                // bagfile setup are complete. Keep this an exact protocol marker.
                                // Unrelated log messages mentioning stdin are not readiness.
                                if line.contains(
                                    "Waiting for input on stdin before execution (--sync mode)",
                                ) && !ready_notified_stderr.swap(true, Ordering::SeqCst)
                                {
                                    let _ = runtime_tx_stderr.send(OfflineRuntimeEvent::MinotReady);
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
                let mut minot_wait_handle = Some(std::thread::spawn(move || {
                    let status = minot_child.wait();
                    let _ = runtime_tx_wait.send(OfflineRuntimeEvent::MinotExited(status));
                }));

                let runtime_tx_tui = runtime_tx.clone();
                std::thread::spawn(move || {
                    if shutdown_rx.recv().is_ok() {
                        let _ = runtime_tx_tui.send(OfflineRuntimeEvent::TuiShutdownRequested);
                    }
                });

                let mut signal_handled = false;
                loop {
                    match recv_offline_runtime_event(&runtime_rx, &mut signal_handled)? {
                        OfflineRuntimeEvent::MinotReady => break,
                        OfflineRuntimeEvent::TuiShutdownRequested => {
                            let _ = minot_guard.kill();
                            if let Some(handle) = minot_wait_handle.take() {
                                join_minot_wait_thread(handle)?;
                            }
                            return Ok(());
                        }
                        OfflineRuntimeEvent::MinotExited(Ok(status)) => {
                            minot_guard.kill().map_err(|e| {
                                anyhow!("Failed to terminate Minot process group: {}", e)
                            })?;
                            if let Some(handle) = minot_wait_handle.take() {
                                join_minot_wait_thread(handle)?;
                            }
                            return Err(anyhow!(
                                "Minot exited before readiness handshake: {}",
                                status
                            ));
                        }
                        OfflineRuntimeEvent::MinotExited(Err(e)) => {
                            minot_guard.kill().map_err(|kill_error| {
                                anyhow!(
                                    "Failed to terminate Minot process group after wait error: {}",
                                    kill_error
                                )
                            })?;
                            if let Some(handle) = minot_wait_handle.take() {
                                join_minot_wait_thread(handle)?;
                            }
                            return Err(anyhow!("Failed to wait on Minot: {}", e));
                        }
                        OfflineRuntimeEvent::MinotEndOfBag | OfflineRuntimeEvent::LioExited(_) => {}
                    }
                }

                info!(
                    "Minot is ready after {:.3}s. Starting pelorus offline processing...",
                    startup_started.elapsed().as_secs_f64()
                );

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
                let replay_tick_owner = if offline_args.polarstern {
                    ReplayTickOwner::Downstream
                } else {
                    ReplayTickOwner::Pelorus
                };
                let (downstream_registered_tx, downstream_registered_rx) =
                    if replay_tick_owner.is_downstream() {
                        let (tx, rx) = std::sync::mpsc::channel();
                        (Some(tx), Some(rx))
                    } else {
                        (None, None)
                    };
                let output_mcap_config_for_run = output_mcap_config.clone();
                let lio_local_only = offline_local_only;
                let lio_coord_mode = offline_args.coord_mode;

                let lio_handle = std::thread::spawn(move || {
                    let synchronous_mode = config.sync_node_params.synchronous_mode;
                    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        lio_with_ready_signal(
                            config,
                            LioRunOptions {
                                ready_signal: Some(pelorus_ready_tx),
                                export_trajectory: export_trajectory_path_for_run.clone(),
                                #[cfg(feature = "processing-time-export")]
                                export_latency: export_latency_path_for_run.clone(),
                                #[cfg(feature = "processing-time-export")]
                                export_processing_time: export_processing_time_path_for_run.clone(),
                                synchronous_mode,
                                #[cfg(feature = "r2r")]
                                ros_node_args: RosNodeArgs {
                                    node_name: "pelorus".to_string(),
                                    namespace: "pelorus".to_string(),
                                },
                                #[cfg(feature = "hiroz")]
                                hiroz_node_args: HirozNodeArgs {
                                    mode: "peer".to_string(),
                                    endpoints: Vec::new(),
                                },
                                shared_stats: Some(shared_stats_for_lio),
                                shutdown_flag: Some(lio_shutdown_flag_for_thread),
                                shutdown_request_rx: Some(lio_stop_rx),
                                offline_prefetch_scans,
                                replay_tick_owner,
                                downstream_registered: downstream_registered_tx,
                                mt_pubsub_local_only: lio_local_only,
                                mt_pubsub_coord_mode: lio_coord_mode,
                                output_mcap: output_mcap_config_for_run,
                            },
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
                info!(
                    "Pelorus is ready after {:.3}s",
                    startup_started.elapsed().as_secs_f64()
                );

                // Hold synchronous replay until the downstream subscribers register.
                if let Some(registered) = downstream_registered_rx {
                    info!("Waiting for the downstream consumer to register...");
                    registered.recv().map_err(|_| {
                        anyhow!(
                            "Pelorus stopped before a downstream consumer registered on '{}'",
                            pelorus_protocol::DOWNSTREAM_CONTROL_SERVICE
                        )
                    })?;
                    info!(
                        "Downstream consumer registered after {:.3}s",
                        startup_started.elapsed().as_secs_f64()
                    );
                }

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
                    match recv_offline_runtime_event(&runtime_rx, &mut signal_handled)? {
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
                            minot_guard
                                .kill()
                                .map_err(|e| anyhow!("Failed to terminate Minot: {}", e))?;
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
                                minot_guard
                                    .kill()
                                    .map_err(|e| anyhow!("Failed to terminate Minot: {}", e))?;
                                minot_force_stopped = true;
                                // On user interrupt, once LIO is down we don't wait on child
                                // wait-notification delivery to avoid deadlocks.
                                wait_for_minot_exit = false;
                            } else if minot_exit_result.is_none() && !minot_force_stopped {
                                minot_guard
                                    .kill()
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

                if let Some(handle) = minot_wait_handle.take() {
                    join_minot_wait_thread(handle)?;
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

                // Verify the effective configuration, including global overrides.
                let user_ast = if args.config_path.as_os_str() == "-" {
                    use std::io::Read;
                    let mut buffer = String::new();
                    std::io::stdin()
                        .read_to_string(&mut buffer)
                        .map_err(|e| anyhow!("Failed to read from stdin: {}", e))?;
                    layer_overrides(&buffer, std::path::Path::new("."), &cli_overrides)
                        .map_err(|e| anyhow!("Failed to compile config from stdin: {}", e))?
                } else {
                    load_user_config_with_checked_overrides(
                        DEFAULT_CONFIG_STR,
                        Some(&args.config_path),
                        &cli_overrides,
                    )?
                };

                let (_, (assigned_vars, evaluated_vars)) =
                    ast_to_config_with_user_usage(default_ast, user_ast)?;
                reject_unknown_overrides(&cli_overrides, &evaluated_vars)?;

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
                use pelorus_syncio::tf_r2r::{TfListener, isometry_from_transform};

                let default_ast = ratslang::compile_code(DEFAULT_CONFIG_STR)
                    .map_err(|e| anyhow!("Failed to compile embedded default config: {}", e))?;
                let user_ast = load_user_config_with_checked_overrides(
                    DEFAULT_CONFIG_STR,
                    args.config_path.as_deref(),
                    &cli_overrides,
                )?;
                let lio_config = ast_to_config(default_ast, user_ast)?;
                let lidar_topic = &lio_config.sync_node_params.cloud_src_topic;
                let imu_topic = &lio_config.sync_node_params.imu_src_topic;

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

                    if lidar_frame.is_none()
                        && let Ok(f) = lidar_rx.try_recv() {
                            eprintln!("Lidar frame ID: {}", f);
                            lidar_frame = Some(f);
                        }
                    if imu_frame.is_none()
                        && let Ok(f) = imu_rx.try_recv() {
                            eprintln!("IMU frame ID: {}", f);
                            imu_frame = Some(f);
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
                    match tf_listener.lookup_transform(imuf, lf, t0.clone()) {
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
                        Err(e) => eprintln!("TF lookup {imuf} ← {lf} failed: {e}"),
                    }
                }

                // --- output.base_link.from_lidar_transform ---
                if let (Some(lf), Some(blf)) = (&lidar_frame, &lio_config.base_link_frame) {
                    match tf_listener.lookup_transform(blf, lf, t0) {
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
                        Err(e) => eprintln!("TF lookup {blf} ← {lf} failed: {e}"),
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

#[cfg(test)]
mod mcap_output_config_tests {
    use super::*;

    fn config_with_mcap_overrides(overrides: &str) -> LioConfig {
        let defaults =
            ratslang::compile_code(include_str!("../../pelorus_core/default.rl")).unwrap();
        let user = ratslang::compile_code(overrides).unwrap();
        ast_to_config(defaults, user).unwrap()
    }

    #[test]
    fn realtime_tf_matches_realtime_odometry() {
        let prediction = RealtimePrediction {
            stamp: pelorus_core::time_msg_from_secs(12.25),
            pose: Isometry3::from_parts(
                nalgebra::Translation3::new(1.0, 2.0, 3.0),
                nalgebra::UnitQuaternion::from_euler_angles(0.1, -0.2, 0.3),
            ),
            linear_velocity_body: nalgebra::Vector3::new(4.0, 5.0, 6.0),
            angular_velocity_body: nalgebra::Vector3::new(0.4, 0.5, 0.6),
            pose_covariance: [0.0; 36],
            twist_covariance: [0.0; 36],
        };

        let odom = realtime_prediction_to_odom(&prediction, "map", "base_link");
        let tf = realtime_prediction_to_tf(&prediction, "map", "base_link");

        assert_eq!(tf.header.stamp.sec, odom.header.stamp.sec);
        assert_eq!(tf.header.stamp.nanosec, odom.header.stamp.nanosec);
        assert_eq!(tf.header.frame_id, odom.header.frame_id);
        assert_eq!(tf.child_frame_id, odom.child_frame_id);
        assert_eq!(tf.transform.translation.x, odom.pose.pose.position.x);
        assert_eq!(tf.transform.translation.y, odom.pose.pose.position.y);
        assert_eq!(tf.transform.translation.z, odom.pose.pose.position.z);
        assert_eq!(tf.transform.rotation.x, odom.pose.pose.orientation.x);
        assert_eq!(tf.transform.rotation.y, odom.pose.pose.orientation.y);
        assert_eq!(tf.transform.rotation.z, odom.pose.pose.orientation.z);
        assert_eq!(tf.transform.rotation.w, odom.pose.pose.orientation.w);
    }

    #[test]
    fn plain_online_still_parses_its_pipeline_flags() {
        let args = Args::parse_from(["pelorus", "online", "--export-trajectory", "/tmp/traj.tum"]);
        let Command::Online(online) = args.command else {
            panic!("expected online");
        };
        assert!(online.mode.is_none());
        assert_eq!(
            online.pipeline.export_trajectory,
            Some("/tmp/traj.tum".to_string())
        );
    }

    #[cfg(feature = "mt_pubsub")]
    #[test]
    fn raw_takes_the_sensor_and_its_own_recording_flags() {
        let args = Args::parse_from([
            "pelorus",
            "online",
            "raw",
            "imu",
            "--output-rosbag",
            "/tmp/imu_bag",
        ]);
        let Command::Online(online) = args.command else {
            panic!("expected online");
        };
        let Some(OnlineMode::Raw(raw)) = online.mode else {
            panic!("expected raw");
        };
        assert_eq!(raw.sensor, RawSensor::Imu);
        assert_eq!(
            raw.recording.output_rosbag,
            Some(PathBuf::from("/tmp/imu_bag"))
        );
        assert_eq!(raw.recording.mcap_cloud_topic, None);
        assert_eq!(raw.recording.mcap_imu_topic, None);
    }

    #[cfg(feature = "mt_pubsub")]
    #[test]
    fn raw_imu_takes_the_intrinsics_sub_mode() {
        let args = Args::parse_from([
            "pelorus",
            "online",
            "raw",
            "imu",
            "intrinsics",
            "--output-rosbag",
            "/tmp/imu_poses",
        ]);
        let Command::Online(online) = args.command else {
            panic!("expected online");
        };
        let Some(OnlineMode::Raw(raw)) = online.mode else {
            panic!("expected raw");
        };
        assert_eq!(raw.sensor, RawSensor::Imu);
        let Some(RawMode::Intrinsics(intrinsics)) = raw.mode else {
            panic!("expected the intrinsics sub-mode");
        };
        assert_eq!(intrinsics.poses, 12);
        assert_eq!(
            raw.recording.output_rosbag,
            Some(PathBuf::from("/tmp/imu_poses"))
        );
    }

    #[cfg(feature = "mt_pubsub")]
    #[test]
    fn pipeline_flags_cannot_be_combined_with_raw() {
        assert!(
            Args::try_parse_from(["pelorus", "online", "--export-trajectory", "raw", "imu"])
                .is_err()
        );
    }

    #[test]
    fn builds_direct_mcap_output_without_metadata_by_default() {
        let config = config_with_mcap_overrides("output.mcap.file = \"out.mcap\"");

        let output =
            build_mcap_output_config_with_filtered_topic(&config.mcap, "imu/ahrs").unwrap();

        assert_eq!(output.path, PathBuf::from("out.mcap"));
        assert_eq!(output.rosbag_metadata_path, None);
        assert_eq!(output.topics.filtered_imu.as_deref(), Some("/imu/ahrs"));
    }

    #[test]
    fn builds_rosbag_directory_paths() {
        let config = config_with_mcap_overrides("output.mcap.rosbag_dir = \"bags/run1\"");

        let output =
            build_mcap_output_config_with_filtered_topic(&config.mcap, "imu/ahrs").unwrap();

        assert_eq!(output.path, PathBuf::from("bags/run1/run1.mcap"));
        assert_eq!(
            output.rosbag_metadata_path,
            Some(PathBuf::from("bags/run1/metadata.yaml"))
        );
    }

    #[test]
    fn file_and_rosbag_dir_are_mutually_exclusive() {
        let defaults =
            ratslang::compile_code(include_str!("../../pelorus_core/default.rl")).unwrap();
        let user = ratslang::compile_code(
            "output.mcap.file = \"out.mcap\"\noutput.mcap.rosbag_dir = \"bags/run1\"",
        )
        .unwrap();
        assert!(ast_to_config(defaults, user).is_err());
    }

    #[test]
    fn rosbag_metadata_requires_file_not_rosbag_dir() {
        let defaults =
            ratslang::compile_code(include_str!("../../pelorus_core/default.rl")).unwrap();
        let user = ratslang::compile_code(
            "output.mcap.rosbag_dir = \"bags/run1\"\noutput.mcap.rosbag_metadata = \"meta.yaml\"",
        )
        .unwrap();
        assert!(ast_to_config(defaults, user).is_err());
    }

    #[test]
    fn normalizes_topic_overrides() {
        let config = config_with_mcap_overrides(
            "output.mcap.file = \"out.mcap\"\n\
             output.mcap.cloud_topic = \"cloud\"\n\
             output.mcap.odom_topic = \"odom\"\n\
             output.mcap.path_topic = \"path\"\n\
             output.mcap.tf_topic = \"tf\"",
        );

        let output =
            build_mcap_output_config_with_filtered_topic(&config.mcap, "imu/custom").unwrap();

        assert_eq!(output.topics.cloud, "/cloud");
        assert_eq!(output.topics.odom, "/odom");
        assert_eq!(output.topics.path, "/path");
        assert_eq!(output.topics.tf, "/tf");
        assert_eq!(output.topics.filtered_imu.as_deref(), Some("/imu/custom"));
    }

    #[test]
    fn uses_default_filtered_imu_topic_when_configured_topic_is_empty() {
        let config = config_with_mcap_overrides("output.mcap.file = \"out.mcap\"");

        let output = build_mcap_output_config_with_filtered_topic(&config.mcap, "").unwrap();

        assert_eq!(output.topics.filtered_imu.as_deref(), Some("/imu/filtered"));
    }

    #[test]
    fn lidar_only_mcap_omits_imu_even_with_a_topic_override() {
        let config = config_with_mcap_overrides(
            "output.mcap.file = \"out.mcap\"\n\
             imu.lidar_only = true\n\
             output.mcap.cloud_topic = \"custom\"",
        );

        let output = build_mcap_output_config(&config).unwrap();
        assert!(output.topics.filtered_imu.is_none());
        assert_eq!(
            output.topics.cloud,
            normalize_mcap_topic(config.mcap.cloud_topic)
        );
    }
}

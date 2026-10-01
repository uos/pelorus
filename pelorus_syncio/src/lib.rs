#[cfg(feature = "ros2-client")]
pub mod tf;
#[cfg(feature = "hiroz")]
pub mod tf_hiroz;
#[cfg(feature = "mt_pubsub")]
pub mod tf_lh;
#[cfg(feature = "r2r")]
pub mod tf_r2r;

#[allow(unused_imports)]
use futures_lite::{StreamExt, future};
#[allow(unused_imports)]
use log::{debug, error, info, warn};
#[cfg(any(feature = "r2r", feature = "hiroz"))]
use pelorus_core::OdomMsg;
#[cfg(feature = "mt_pubsub")]
use pelorus_core::PointCloudMode;
#[cfg(any(
    feature = "ros2-client",
    feature = "mt_pubsub",
    feature = "r2r",
    feature = "hiroz"
))]
use pelorus_core::SyncProcessedCloud;
#[cfg(feature = "native-drivers")]
use pelorus_core::{
    ImuDriverType, LidarDriverType, OusterPoint, RobosensePoint, SensorDriverConfig, VelodynePoint,
    time_msg_from_secs,
};
use pelorus_core::{
    ImuMsg, ImuPredictor, ImuPredictorUndistort, LidarVendor, PointXYZIT, SyncNodeParameters,
    SyncParameters, timing::RuntimeEvaluator,
};
#[cfg(feature = "native-drivers")]
use pelorus_drivers::{DriverError, DriverManager, RawLidarScan, SensorIngress, SensorOutputSink};
pub use pelorus_sync;

/// Re-exported so consumers can implement `SensorIngress` without taking their
/// own dependency on the optional driver crate.
#[cfg(feature = "native-drivers")]
pub use pelorus_drivers;

#[cfg(any(
    feature = "ros2-client",
    feature = "mt_pubsub",
    feature = "r2r",
    feature = "hiroz"
))]
use anyhow::anyhow;
#[cfg(any(
    feature = "ros2-client",
    feature = "mt_pubsub",
    feature = "r2r",
    feature = "hiroz",
    feature = "native-drivers",
))]
use pelorus_sync::timestamp_to_secs;

#[cfg(feature = "hiroz")]
use hiroz::Builder;
#[cfg(feature = "hiroz")]
use hiroz::node::ZNode;
#[cfg(feature = "hiroz")]
use hiroz_msgs::{
    builtin_interfaces::Time,
    geometry_msgs::{
        Point, Pose, PoseWithCovariance, Quaternion, Twist, TwistWithCovariance, Vector3,
    },
    nav_msgs::Odometry,
    sensor_msgs::{Imu, PointCloud2},
    std_msgs::Header,
};
#[cfg(feature = "mt_pubsub")]
use mt_pubsub::Node;
#[cfg(any(
    feature = "ros2-client",
    feature = "mt_pubsub",
    feature = "r2r",
    feature = "hiroz"
))]
use pelorus_sync::PreprocessingPointCloud;
#[cfg(feature = "r2r")]
use r2r::{
    builtin_interfaces::msg::Time,
    geometry_msgs::msg::{
        Point, Pose, PoseWithCovariance, Quaternion, Twist, TwistWithCovariance, Vector3,
    },
    nav_msgs::msg::Odometry,
    sensor_msgs::msg::{Imu, PointCloud2},
    std_msgs::msg::Header,
};
use ros_pointcloud2::prelude::*;

#[cfg(feature = "mt_pubsub")]
struct ArchivedPointCloud {
    message: mt_pubsub::ArchivedMessage<PointCloud2>,
}

/// Wire-compatible rkyv representation of Cloudini's ROS helper types.
///
/// These live in `pelorus_protocol` so producer and consumer share one
/// definition.
#[cfg(feature = "compressed-pointcloud")]
pub use pelorus_protocol::{
    CompressedHeaderMsg, CompressedPointCloud2, CompressedPointFieldMsg, CompressedTimeMsg,
};

/// Cloudini codec for the shared wire type.
#[cfg(feature = "compressed-pointcloud")]
pub fn compress_cloud(
    msg: ros_pointcloud2::PointCloud2Msg,
) -> cloudini::Result<CompressedPointCloud2> {
    use cloudini::ros::{CompressExt, CompressionConfig};

    // Cloudini's default 1 mm quantisation + ZSTD. TODO make configurable.
    let msg = msg.compress(CompressionConfig::default())?;
    Ok(CompressedPointCloud2 {
        header: CompressedHeaderMsg {
            seq: msg.header.seq,
            stamp: CompressedTimeMsg {
                sec: msg.header.stamp.sec,
                nanosec: msg.header.stamp.nanosec,
            },
            frame_id: msg.header.frame_id,
        },
        height: msg.height,
        width: msg.width,
        fields: msg
            .fields
            .into_iter()
            .map(|field| CompressedPointFieldMsg {
                name: field.name.as_str().to_owned(),
                offset: field.offset,
                datatype: field.datatype,
                count: field.count,
            })
            .collect(),
        is_bigendian: msg.is_bigendian,
        point_step: msg.point_step,
        row_step: msg.row_step,
        compressed_data: msg.compressed_data,
        is_dense: msg.is_dense,
        format: msg.format,
    })
}

#[cfg(feature = "compressed-pointcloud")]
pub fn decompress_cloud(
    cloud: CompressedPointCloud2,
) -> cloudini::Result<ros_pointcloud2::PointCloud2Msg> {
    cloudini::ros::CompressedPointCloud2 {
        header: ros_pointcloud2::ros::HeaderMsg {
            seq: cloud.header.seq,
            stamp: ros_pointcloud2::ros::TimeMsg {
                sec: cloud.header.stamp.sec,
                nanosec: cloud.header.stamp.nanosec,
            },
            frame_id: cloud.header.frame_id,
        },
        height: cloud.height,
        width: cloud.width,
        fields: cloud
            .fields
            .into_iter()
            .map(|field| ros_pointcloud2::ros::PointFieldMsg {
                name: field.name.into(),
                offset: field.offset,
                datatype: field.datatype,
                count: field.count,
            })
            .collect(),
        is_bigendian: cloud.is_bigendian,
        point_step: cloud.point_step,
        row_step: cloud.row_step,
        compressed_data: cloud.compressed_data,
        is_dense: cloud.is_dense,
        format: cloud.format,
    }
    .decompress()
}

#[cfg(feature = "mt_pubsub")]
impl ros_pointcloud2::PointCloud2Source for ArchivedPointCloud {
    fn point_count(&self) -> usize {
        let cloud = self.message.archived();
        usize::try_from(u32::from(cloud.width)).expect("u32 fits usize")
            * usize::try_from(u32::from(cloud.height)).expect("u32 fits usize")
    }

    fn field(&self, name: &str) -> Option<ros_pointcloud2::PointFieldDescription> {
        self.message
            .archived()
            .fields
            .iter()
            .find(|field| field.name.as_str() == name)
            .map(|field| ros_pointcloud2::PointFieldDescription {
                offset: field.offset.into(),
                datatype: field.datatype,
            })
    }

    fn endian(&self) -> ros_pointcloud2::Endian {
        if self.message.archived().is_bigendian {
            ros_pointcloud2::Endian::Big
        } else {
            ros_pointcloud2::Endian::Little
        }
    }

    fn point_step(&self) -> u32 {
        self.message.archived().point_step.into()
    }

    fn data(&self) -> &[u8] {
        self.message.archived().data.as_slice()
    }
}

#[cfg(feature = "mt_pubsub")]
impl PreprocessingPointCloud for ArchivedPointCloud {
    fn header(&self) -> ros_pointcloud2::ros::HeaderMsg {
        let header = &self.message.archived().header;
        ros_pointcloud2::ros::HeaderMsg {
            seq: 0,
            stamp: ros_pointcloud2::ros::TimeMsg {
                sec: header.stamp.sec.into(),
                nanosec: header.stamp.nanosec.into(),
            },
            frame_id: header.frame_id.as_str().to_owned(),
        }
    }

    fn source(&self) -> &dyn ros_pointcloud2::PointCloud2Source {
        self
    }
}

#[cfg(feature = "mt_pubsub")]
enum IncomingMtPointCloud {
    Owned(PointCloud2),
    Archived(mt_pubsub::ArchivedMessage<PointCloud2>),
    #[cfg(feature = "compressed-pointcloud")]
    Compressed(CompressedPointCloud2),
}

#[cfg(feature = "mt_pubsub")]
enum MtCloudSubscriber {
    PointCloud2(mt_pubsub::Subscriber<PointCloud2>),
    #[cfg(feature = "compressed-pointcloud")]
    Compressed(mt_pubsub::Subscriber<CompressedPointCloud2>),
}
#[cfg(feature = "ros2-client")]
use ros2_client::ros2;
#[cfg(all(
    feature = "mt_pubsub",
    not(feature = "r2r"),
    not(feature = "ros2-client")
))]
use ros2_interfaces_jazzy_rkyv::{
    builtin_interfaces::msg::Time,
    geometry_msgs::msg::{
        Point, Pose, PoseWithCovariance, Quaternion, Twist, TwistWithCovariance, Vector3,
    },
    nav_msgs::msg::Odometry,
    sensor_msgs::msg::{Imu, PointCloud2},
    std_msgs::msg::Header,
};
#[cfg(all(feature = "ros2-client", not(feature = "r2r")))]
use ros2_interfaces_jazzy_serde::{
    builtin_interfaces::msg::Time,
    geometry_msgs::msg::{
        Point, Pose, PoseWithCovariance, Quaternion, Twist, TwistWithCovariance, Vector3,
    },
    nav_msgs::msg::Odometry,
    sensor_msgs::msg::{Imu, PointCloud2},
    std_msgs::msg::Header,
};
#[cfg(feature = "native-drivers")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub type ImuInitCallback = Arc<Mutex<dyn FnMut(&[ImuMsg]) + Send + Sync>>;

#[cfg(test)]
mod startup_tf_tests {
    use super::*;

    #[test]
    fn tf_is_applied_before_deferred_initialization_and_only_once() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let applied = events.clone();
        let initialized = events.clone();
        let mut gate = TfStartup {
            apply: Arc::new(Mutex::new(move |_, _| applied.lock().unwrap().push("tf"))),
            initialize: Some(Arc::new(Mutex::new(move |_: &[ImuMsg]| {
                initialized.lock().unwrap().push("init")
            }))),
            pending: None,
            resolved: false,
            initialized: false,
        };
        gate.initialize(&[ImuMsg::default()]);
        assert!(events.lock().unwrap().is_empty());
        assert!(!gate.initialized);
        assert!(gate.resolve(Some(nalgebra::Isometry3::identity()), None));
        assert!(gate.resolve(Some(nalgebra::Isometry3::identity()), None));
        assert_eq!(*events.lock().unwrap(), ["tf", "init"]);
    }

    #[test]
    fn tf_before_imu_still_waits_for_initialization() {
        let mut gate = TfStartup {
            apply: Arc::new(Mutex::new(|_, _| {})),
            initialize: Some(Arc::new(Mutex::new(|_: &[ImuMsg]| {}))),
            pending: None,
            resolved: false,
            initialized: false,
        };
        assert!(!gate.resolve(None, None));
        gate.initialize(&[ImuMsg::default()]);
        assert!(gate.resolve(None, None));
    }
}
/// How long a replayed scan waits for its sensor TF before it is dropped.
#[cfg(any(
    feature = "ros2-client",
    feature = "mt_pubsub",
    feature = "r2r",
    feature = "hiroz"
))]
const REPLAY_SENSOR_TF_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

pub type StartupTfCallback = Arc<
    Mutex<
        dyn FnMut(Option<nalgebra::Isometry3<f64>>, Option<nalgebra::Isometry3<f64>>) + Send + Sync,
    >,
>;

/// Defer initialization without blocking the IMU ingress/replay thread. Replay
/// must be allowed to deliver the first cloud so its sensor frame can be resolved.
struct TfStartup {
    #[cfg_attr(
        not(any(
            feature = "ros2-client",
            feature = "mt_pubsub",
            feature = "r2r",
            feature = "hiroz"
        )),
        allow(dead_code)
    )]
    apply: StartupTfCallback,
    initialize: Option<ImuInitCallback>,
    pending: Option<Vec<ImuMsg>>,
    resolved: bool,
    initialized: bool,
}

impl TfStartup {
    fn initialize(&mut self, msgs: &[ImuMsg]) {
        if self.resolved {
            if let Some(init) = &self.initialize {
                (init.lock().unwrap())(msgs);
                info!("EKF: Imu initialized after requested sensor TF was applied");
            }
            self.initialized = true;
        } else {
            self.pending = Some(msgs.to_vec());
            info!("IMU initialization deferred: waiting for requested sensor TF");
        }
    }

    #[cfg_attr(
        not(any(
            feature = "ros2-client",
            feature = "mt_pubsub",
            feature = "r2r",
            feature = "hiroz"
        )),
        allow(dead_code)
    )]
    fn resolve(
        &mut self,
        imu: Option<nalgebra::Isometry3<f64>>,
        base: Option<nalgebra::Isometry3<f64>>,
    ) -> bool {
        if !self.resolved {
            (self.apply.lock().unwrap())(imu, base);
            self.resolved = true;
            if let Some(msgs) = self.pending.take() {
                self.initialize(&msgs);
            }
        }
        self.initialized
    }
}

pub struct SyncNodeInputs<'a> {
    pub startup_tf_callback: Option<StartupTfCallback>,
    pub startup_base_frame: Option<String>,
    pub node_params: &'a SyncNodeParameters,
    pub imu_pose_predictor_init: Option<ImuInitCallback>,
    pub imu_pose_prediction_undistort: Option<Box<ImuPredictorUndistort>>,
    pub imu_pose_prediction_init_guess: Option<Box<ImuPredictor>>,
    pub ready_for_next: Option<std::sync::mpsc::Receiver<u64>>,
    pub filtered_imu_sender: Option<std::sync::mpsc::Sender<ImuMsg>>,
    pub imu_sync_mode: pelorus_sync::ImuSyncMode,
    pub realtime_imu_callback: Option<pelorus_sync::RealtimeImuCallback>,
    pub realtime_lidar_barrier_callback: Option<pelorus_sync::RealtimeLidarBarrierCallback>,
}
#[cfg(feature = "hiroz")]
use zenoh_buffers::buffer::Buffer;

#[cfg(feature = "ros2-client")]
impl_pointcloud2_for_ros2_interfaces_jazzy_serde!();
#[cfg(feature = "mt_pubsub")]
impl_pointcloud2_for_ros2_interfaces_jazzy_rkyv!();
#[cfg(feature = "r2r")]
impl_pointcloud2_for_r2r!();
#[cfg(feature = "hiroz")]
impl_pointcloud2_for_hiroz!();

#[cfg(any(
    feature = "ros2-client",
    feature = "mt_pubsub",
    feature = "r2r",
    feature = "hiroz"
))]
const N_NTH_SCAN: usize = 1; // TODO to config
#[cfg(feature = "rat")]
const MINOT_IMU_INIT_TICK: &str = "imu_init_tick";
#[cfg(feature = "rat")]
const MINOT_LIO_START_TICK: &str = "lio_start_tick";

#[cfg(all(
    feature = "rat",
    any(feature = "ros2-client", feature = "mt_pubsub", feature = "r2r")
))]
fn publish_minot_trigger(synchronous_mode: bool, trigger: &str) {
    if synchronous_mode {
        let mut payload = String::new();
        rat::bacon(trigger, &mut payload, rat::VariableType::StaticOnly).unwrap();
    }
}

#[cfg(all(feature = "rat", feature = "hiroz"))]
async fn publish_minot_trigger_async(synchronous_mode: bool, trigger: &str) {
    if synchronous_mode {
        let mut payload = String::new();
        rat::bacon_async(trigger, &mut payload, rat::VariableType::StaticOnly)
            .await
            .unwrap();
    }
}

#[cfg(feature = "rat")]
fn minot_sd_start_timestamp() -> rat::NetArray<f64> {
    static PRELIO_MONO_START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let now_ns = PRELIO_MONO_START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_nanos() as f64;
    rat::NetArray::from(nalgebra::DMatrix::<f64>::from_element(1, 1, now_ns))
}

pub struct State {
    startup_tf: Option<Arc<Mutex<TfStartup>>>,
    startup_base_frame: Option<String>,
    /// Managed replay sends each scan exactly once and waits for it to be processed. A scan
    /// that arrives before its sensor TF must then wait for the TF instead of being dropped.
    #[cfg_attr(
        not(any(
            feature = "ros2-client",
            feature = "mt_pubsub",
            feature = "r2r",
            feature = "hiroz"
        )),
        allow(dead_code)
    )]
    wait_for_sensor_tf: bool,
    pub global_polygon: Option<Vec<nalgebra::Point2<f32>>>,
    pub sensor_tf: Option<nalgebra::geometry::Isometry3<f32>>,
    pub imu_tf: Option<nalgebra::geometry::Isometry3<f64>>,

    pub sync: Arc<Mutex<pelorus_sync::Synchronizer>>,
    pub sync_params: SyncParameters,
    #[cfg(any(
        feature = "ros2-client",
        feature = "mt_pubsub",
        feature = "r2r",
        feature = "hiroz"
    ))]
    take_nth_scan: usize,
    pub lidar_vendor: LidarVendor,
    #[cfg(feature = "rerun")]
    pub rec: Option<Arc<rerun::RecordingStream>>,

    pub last_lidar_time: Option<f64>,
}

impl State {
    pub fn new(
        config: &SyncParameters,
        #[cfg(feature = "rerun")] rec: Option<Arc<rerun::RecordingStream>>,
        imu_sync_mode: pelorus_sync::ImuSyncMode,
        hooks: pelorus_sync::SynchronizerHooks,
    ) -> anyhow::Result<Self> {
        let wait_for_sensor_tf = matches!(
            imu_sync_mode,
            pelorus_sync::ImuSyncMode::DemandDrivenReplay { .. }
                | pelorus_sync::ImuSyncMode::Disabled { lossless: true }
        );
        Ok(State {
            startup_tf: None,
            startup_base_frame: None,
            wait_for_sensor_tf,
            #[cfg(feature = "rerun")]
            rec,
            global_polygon: None,
            sensor_tf: None,
            imu_tf: None,
            sync: Arc::new(Mutex::new(pelorus_sync::Synchronizer::create(
                config.imu.gravity,
                config.imu.madgwick_setting,
                &config.imu.calibration,
                config,
                config.lidar_vendor,
                imu_sync_mode,
                hooks,
            )?)),
            sync_params: config.clone(),
            #[cfg(any(
                feature = "ros2-client",
                feature = "mt_pubsub",
                feature = "r2r",
                feature = "hiroz"
            ))]
            take_nth_scan: N_NTH_SCAN,
            lidar_vendor: config.lidar_vendor,
            last_lidar_time: None,
        })
    }
}

#[cfg(any(
    feature = "ros2-client",
    feature = "mt_pubsub",
    feature = "r2r",
    feature = "hiroz"
))]
fn cloud_callback(
    state: Arc<Mutex<State>>,
    #[cfg(feature = "ros2-client")] tf_listener: Arc<tf::TfListener>,
    #[cfg(feature = "r2r")] tf_listener: Arc<tf_r2r::TfListener>,
    #[cfg(feature = "mt_pubsub")] tf_listener: Arc<tf_lh::TfListener>,
    #[cfg(feature = "hiroz")] tf_listener: Arc<tf_hiroz::TfListener>,
    #[cfg(feature = "ros2-client")] msg: PointCloud2,
    #[cfg(feature = "mt_pubsub")] msg: IncomingMtPointCloud,
    #[cfg(feature = "r2r")] msg: r2r::sensor_msgs::msg::PointCloud2,
    #[cfg(feature = "hiroz")] msg: PointCloud2,
    first_imu_tf: Option<String>,
) -> anyhow::Result<()> {
    fn duplicate_timestamp<T: Clone>(timestamp: &T) -> T {
        timestamp.clone()
    }

    let callback_started = std::time::Instant::now();
    #[cfg(not(feature = "mt_pubsub"))]
    let orig_header = msg.header.clone();
    #[cfg(feature = "ros2-client")]
    let in_msg = impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(msg);
    #[cfg(feature = "mt_pubsub")]
    let (orig_header, in_msg): (_, Box<dyn PreprocessingPointCloud>) = match msg {
        IncomingMtPointCloud::Owned(msg) => {
            let msg = impl_ros2_interfaces_jazzy_rkyv::to_pointcloud2_msg(msg);
            let header = msg.header.clone();
            (header, Box::new(msg))
        }
        IncomingMtPointCloud::Archived(message) => {
            let header = {
                let archived = message.archived();
                ros_pointcloud2::ros::HeaderMsg {
                    seq: 0,
                    stamp: ros_pointcloud2::ros::TimeMsg {
                        sec: archived.header.stamp.sec.into(),
                        nanosec: archived.header.stamp.nanosec.into(),
                    },
                    frame_id: archived.header.frame_id.as_str().to_owned(),
                }
            };
            (header, Box::new(ArchivedPointCloud { message }))
        }
        #[cfg(feature = "compressed-pointcloud")]
        IncomingMtPointCloud::Compressed(message) => {
            let msg = decompress_cloud(message)
                .map_err(|error| anyhow!("Failed to decompress Cloudini point cloud: {error}"))?;
            let header = msg.header.clone();
            (header, Box::new(msg))
        }
    };
    #[cfg(feature = "r2r")]
    let in_msg = impl_r2r::to_pointcloud2_msg(msg);
    #[cfg(feature = "hiroz")]
    let in_msg = impl_hiroz::to_pointcloud2_msg(msg);

    #[cfg(not(feature = "mt_pubsub"))]
    let in_msg: Box<dyn PreprocessingPointCloud> = Box::new(in_msg);

    #[cfg(feature = "mt_pubsub")]
    let in_stamp = Time {
        sec: orig_header.stamp.sec,
        nanosec: orig_header.stamp.nanosec,
    };
    #[cfg(not(feature = "mt_pubsub"))]
    let in_stamp = orig_header.stamp;
    let points_n = in_msg.source().point_count();
    if points_n == 0 {
        return Ok(());
    }
    {
        let wait_for_sensor_tf = state.lock().unwrap().wait_for_sensor_tf;
        // Live input drops a scan whose sensor TF is still missing and tries again with the next
        // one. Managed replay only moves on after this scan, so it keeps looking up for a while.
        let lookup_sensor_tf = |target: &str, source: &str| {
            let deadline = std::time::Instant::now() + REPLAY_SENSOR_TF_TIMEOUT;
            loop {
                match tf_listener.lookup_transform(target, source, duplicate_timestamp(&in_stamp)) {
                    Ok(tf) => break Ok(tf),
                    Err(_) if wait_for_sensor_tf && std::time::Instant::now() < deadline => {
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    Err(error) => break Err(error),
                }
            }
        };
        let (imu_tf, sensor_tf) = {
            let (imu_tf_override, global_zero_frame, _, sensor_tf) = {
                let mut state = state.lock().unwrap();

                // throttle the number of scans
                if state.take_nth_scan > 1 {
                    state.take_nth_scan -= 1;
                    return Ok(());
                } else {
                    state.take_nth_scan = N_NTH_SCAN;
                }

                (
                    state.sync_params.imu_lidar_tf_override,
                    state.sync_params.global_zero_frame.clone(),
                    state.sync_params.map_frame.clone(),
                    state.sensor_tf,
                )
            };

            {
                let sensor_tf = match sensor_tf {
                    Some(tf) => tf,
                    None => {
                        if global_zero_frame.is_empty() {
                            nalgebra::geometry::Isometry3::identity()
                        } else {
                            let tf = tf_listener.lookup_transform(
                                orig_header.frame_id.clone().as_str(),
                                global_zero_frame.clone().as_str(),
                                duplicate_timestamp(&in_stamp),
                            )?;
                            nalgebra::geometry::Isometry3::from_parts(
                                nalgebra::geometry::Translation3::new(
                                    tf.transform.translation.x as f32,
                                    tf.transform.translation.y as f32,
                                    tf.transform.translation.z as f32,
                                ),
                                nalgebra::geometry::UnitQuaternion::from_quaternion(
                                    nalgebra::geometry::Quaternion::new(
                                        tf.transform.rotation.w as f32,
                                        tf.transform.rotation.x as f32,
                                        tf.transform.rotation.y as f32,
                                        tf.transform.rotation.z as f32,
                                    ),
                                ),
                            )
                            .inverse()
                        }
                    }
                };

                let imu_tf = if imu_tf_override {
                    match first_imu_tf {
                        None => {
                            warn!("Waiting for first IMU Message to get frame.");
                            return Ok(());
                        }
                        Some(first_imu_tf_frame) => {
                            let tf = match lookup_sensor_tf(
                                &first_imu_tf_frame,
                                &orig_header.frame_id,
                            ) {
                                Ok(tf) => tf,
                                Err(error) => {
                                    if wait_for_sensor_tf {
                                        warn!("Dropping scan, IMU-to-Lidar TF is missing: {error}");
                                    } else {
                                        debug!("Waiting for IMU-to-Lidar TF: {error}");
                                    }
                                    return Ok(());
                                }
                            };
                            nalgebra::geometry::Isometry3::from_parts(
                                nalgebra::geometry::Translation3::new(
                                    tf.transform.translation.x,
                                    tf.transform.translation.y,
                                    tf.transform.translation.z,
                                ),
                                nalgebra::geometry::UnitQuaternion::from_quaternion(
                                    nalgebra::geometry::Quaternion::new(
                                        tf.transform.rotation.w,
                                        tf.transform.rotation.x,
                                        tf.transform.rotation.y,
                                        tf.transform.rotation.z,
                                    ),
                                ),
                            )
                        }
                    }
                } else {
                    state
                        .lock()
                        .unwrap()
                        .imu_tf
                        .unwrap_or(nalgebra::geometry::Isometry3::identity())
                };

                // dbg!(imu_tf);
                (imu_tf, sensor_tf)
            }
        };

        let (startup, base_frame) = {
            let state = state.lock().unwrap();
            (state.startup_tf.clone(), state.startup_base_frame.clone())
        };
        if let Some(startup) = startup {
            let base = if let Some(frame) = base_frame {
                let tf = match lookup_sensor_tf(&frame, &orig_header.frame_id) {
                    Ok(tf) => tf,
                    Err(error) => {
                        if wait_for_sensor_tf {
                            warn!("Dropping scan, base-to-Lidar TF is missing: {error}");
                        } else {
                            debug!("Waiting for base-to-Lidar TF: {error}");
                        }
                        return Ok(());
                    }
                };
                Some(nalgebra::Isometry3::from_parts(
                    nalgebra::Translation3::new(
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        tf.transform.translation.z,
                    ),
                    nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                        tf.transform.rotation.w,
                        tf.transform.rotation.x,
                        tf.transform.rotation.y,
                        tf.transform.rotation.z,
                    )),
                ))
            } else {
                None
            };
            let imu = state
                .lock()
                .unwrap()
                .sync_params
                .imu_lidar_tf_override
                .then_some(imu_tf);
            if !startup.lock().unwrap().resolve(imu, base) {
                return Ok(());
            }
        }

        {
            let mut state = state.lock().unwrap();
            let header = in_msg.header();
            let time = timestamp_to_secs(&header.stamp);
            if let Some(last_lidar_time) = state.last_lidar_time
                && time < last_lidar_time
            {
                // Skip out of order cloud
                return Ok(());
            }
            state.last_lidar_time.replace(time);
            state.sensor_tf = Some(sensor_tf);
            state.imu_tf = Some(imu_tf);
        }

        let params = {
            let (global_polygon, invert_polygon, global_zero_frame, map_frame, mut params) = {
                let state = state.lock().unwrap();

                (
                    state.global_polygon.clone(),
                    state.sync_params.passthrough_filter.invert_polygon,
                    state.sync_params.global_zero_frame.clone(),
                    state.sync_params.map_frame.clone(),
                    state.sync_params.clone(), // TODO overhead clone on every cloud but now params can not change anymore while runtime
                )
            };

            if let Some(poly) = global_polygon.as_ref() {
                let tf = if global_zero_frame.is_empty() {
                    nalgebra::geometry::Isometry3::identity()
                } else {
                    let tf = tf_listener.lookup_transform(
                        map_frame.as_str(),
                        global_zero_frame.as_str(),
                        in_stamp,
                    )?;
                    if map_frame.is_empty() {
                        return Err(anyhow!(
                            "global_zero_frame is given but map_frame is not. Can not update polygon."
                        ));
                    }
                    nalgebra::geometry::Isometry3::from_parts(
                        nalgebra::geometry::Translation3::new(
                            tf.transform.translation.x as f32,
                            tf.transform.translation.y as f32,
                            tf.transform.translation.z as f32,
                        ),
                        nalgebra::geometry::UnitQuaternion::from_quaternion(
                            nalgebra::geometry::Quaternion::new(
                                tf.transform.rotation.w as f32,
                                tf.transform.rotation.x as f32,
                                tf.transform.rotation.y as f32,
                                tf.transform.rotation.z as f32,
                            ),
                        ),
                    )
                    .inverse()
                };

                let mut cupcl_poly = Vec::new();
                for p in poly {
                    let tp = tf * nalgebra::Point3::new(p.x, p.y, 0.0);
                    cupcl_poly.push(nalgebra::Point2::new(tp.x, tp.y));
                }
                params.passthrough_filter.polygon = Some(cupcl_poly);
                params.passthrough_filter.invert_polygon = invert_polygon;
            };

            params
        };

        {
            let mut state = state.lock().unwrap();
            state.sync_params = params;
        }
    };

    #[cfg(feature = "rat")]
    {
        #[cfg(not(any(feature = "mt_pubsub", feature = "hiroz")))]
        let mut ts = minot_sd_start_timestamp();
        #[cfg(not(any(feature = "mt_pubsub", feature = "hiroz")))]
        rat::bacon("pelorus:sd_start", &mut ts, rat::VariableType::F64).ok();
    }

    {
        let sync = state.lock().unwrap().sync.clone();
        let mut sync = sync.lock().unwrap();
        sync.add_pointcloud_source(in_msg, callback_started);
    }

    Ok(())
}

pub struct SyncNode {
    pub state: Arc<std::sync::Mutex<State>>,
    #[cfg(feature = "ros2-client")]
    pub tf_listener: Arc<tf::TfListener>,
    #[cfg(feature = "mt_pubsub")]
    pub tf_listener: Arc<tf_lh::TfListener>,
    #[cfg(feature = "r2r")]
    pub tf_listener: Arc<tf_r2r::TfListener>,
    #[cfg(feature = "hiroz")]
    pub tf_listener: Arc<tf_hiroz::TfListener>,
    pub logger: String,
    #[cfg(feature = "native-drivers")]
    native_drivers: Option<DriverManager>,
    #[cfg(feature = "native-drivers")]
    native_imu_init: Option<ImuInitCallback>,
    #[cfg(feature = "native-drivers")]
    native_imu_init_time: f64,
}

#[cfg(feature = "native-drivers")]
#[derive(Default)]
struct NativeImuInitState {
    start_time: Option<f64>,
    buffer: Option<Vec<ImuMsg>>,
}

#[cfg(feature = "native-drivers")]
struct NativeIngress {
    state: Arc<Mutex<State>>,
    lidar_driver_type: LidarDriverType,
    filtered_imu_sender: Option<std::sync::mpsc::Sender<ImuMsg>>,
    imu_init: Option<ImuInitCallback>,
    imu_init_time: f64,
    imu_init_state: Mutex<NativeImuInitState>,
    imu_initialized: AtomicBool,
}

#[cfg(feature = "native-drivers")]
pub fn validate_native_lidar_vendor(
    driver_type: &LidarDriverType,
    lidar_vendor: LidarVendor,
) -> Result<(), DriverError> {
    if matches!(
        (driver_type, lidar_vendor),
        (&LidarDriverType::Disabled, _)
            | (&LidarDriverType::Velodyne, LidarVendor::Velodyne)
            | (&LidarDriverType::Ouster, LidarVendor::Ouster)
            | (&LidarDriverType::Robosense, LidarVendor::Robosense)
    ) {
        Ok(())
    } else {
        Err(DriverError::Driver(format!(
            "native lidar driver {driver_type:?} is incompatible with lidar.vendor {lidar_vendor:?}"
        )))
    }
}

#[cfg(feature = "native-drivers")]
pub fn ouster_offset_nanoseconds(seconds: f64) -> Result<u32, DriverError> {
    let nanoseconds = seconds * 1e9;
    if !nanoseconds.is_finite() || nanoseconds < 0.0 || nanoseconds > u32::MAX as f64 {
        return Err(DriverError::Driver(format!(
            "Ouster point time offset {seconds} seconds is outside the uint32 nanosecond range"
        )));
    }
    Ok(nanoseconds.round() as u32)
}

#[cfg(feature = "native-drivers")]
pub fn native_scan_to_cloud(
    scan: RawLidarScan,
    driver_type: &LidarDriverType,
) -> Result<ros_pointcloud2::PointCloud2Msg, DriverError> {
    let frame_id = scan.frame_id;
    let stamp = scan.stamp;
    let mut cloud = match driver_type {
        LidarDriverType::Velodyne => {
            let points = scan
                .points
                .into_iter()
                .map(|point| VelodynePoint {
                    x: point.xyz[0],
                    y: point.xyz[1],
                    z: point.xyz[2],
                    intensity: point.intensity * 255.0,
                    time: point.time_offset_secs as f32,
                })
                .collect();
            ros_pointcloud2::PointCloud2Msg::try_from_vec(points)
        }
        LidarDriverType::Ouster => {
            let points = scan
                .points
                .into_iter()
                .map(|point| {
                    Ok(OusterPoint {
                        x: point.xyz[0],
                        y: point.xyz[1],
                        z: point.xyz[2],
                        intensity: point.intensity * 255.0,
                        t: ouster_offset_nanoseconds(point.time_offset_secs)?,
                    })
                })
                .collect::<Result<Vec<_>, DriverError>>()?;
            ros_pointcloud2::PointCloud2Msg::try_from_vec(points)
        }
        LidarDriverType::Robosense => {
            // RobosensePoint stores an absolute epoch per point; pelorus_sync
            // subtracts the header stamp again when it deskews the scan.
            let points = scan
                .points
                .into_iter()
                .map(|point| RobosensePoint {
                    x: point.xyz[0],
                    y: point.xyz[1],
                    z: point.xyz[2],
                    intensity: point.intensity * 255.0,
                    timestamp: stamp + point.time_offset_secs,
                })
                .collect();
            ros_pointcloud2::PointCloud2Msg::try_from_vec(points)
        }
        LidarDriverType::Disabled => {
            return Err(DriverError::Driver(
                "received a native lidar scan while the native lidar driver is disabled".into(),
            ));
        }
    }
    .map_err(|error| DriverError::Driver(format!("native cloud conversion: {error}")))?;
    cloud.header.frame_id = frame_id;
    cloud.header.stamp = time_msg_from_secs(stamp);
    Ok(cloud)
}

#[cfg(feature = "native-drivers")]
impl SensorIngress for NativeIngress {
    fn push_lidar(&self, scan: RawLidarScan) -> Result<(), DriverError> {
        if !self.imu_initialized.load(Ordering::Acquire) {
            return Ok(());
        }
        let cloud = native_scan_to_cloud(scan, &self.lidar_driver_type)?;
        self.state
            .lock()
            .unwrap()
            .sync
            .lock()
            .unwrap()
            .add_pointcloud(cloud);
        Ok(())
    }
    fn push_imu(&self, imu: ImuMsg) -> Result<(), DriverError> {
        if let Some(init) = self.imu_init.as_ref() {
            let imu_time = timestamp_to_secs(&imu.stamp);
            let mut state = self.imu_init_state.lock().unwrap();
            let start = *state.start_time.get_or_insert(imu_time);
            if imu_time - start < self.imu_init_time {
                if let Some(buffer) = state.buffer.as_mut() {
                    buffer.push(imu.clone());
                }
            } else if let Some(buffer) = state.buffer.take() {
                if !buffer.is_empty() {
                    init.lock().unwrap()(&buffer);
                }
                self.imu_initialized.store(true, Ordering::Release);
            }
        }
        let filtered = self.state.lock().unwrap().sync.lock().unwrap().add_imu(imu);
        if let (Some(v), Some(tx)) = (filtered, self.filtered_imu_sender.as_ref()) {
            let _ = tx.send(v);
        }
        Ok(())
    }

    fn driver_failed(&self, error: &DriverError) {
        log::error!("Native sensor driver failed. Stopping Pelorus ingress: {error}");
        if let Ok(state) = self.state.lock()
            && let Ok(mut sync) = state.sync.lock()
        {
            sync.shutdown();
        }
    }
}

#[cfg(feature = "mt_pubsub")]
pub fn qos_to_mt(qos: pelorus_core::Qos) -> mt_pubsub::Qos {
    match qos {
        pelorus_core::Qos::Sensor(pelorus_core::Reliability::Reliable) => mt_pubsub::Qos::Reliable,
        pelorus_core::Qos::Sensor(pelorus_core::Reliability::BestEffort)
        | pelorus_core::Qos::SystemDefault => mt_pubsub::Qos::BestEffort,
    }
}

#[cfg(feature = "ros2-client")]
pub fn qos_to_ros2(value: pelorus_core::Qos) -> ros2::QosPolicies {
    match value {
        pelorus_core::Qos::Sensor(rel) => ros2::QosPolicyBuilder::new()
            .durability(ros2::policy::Durability::Volatile)
            .deadline(ros2::policy::Deadline(ros2::Duration::INFINITE))
            .ownership(ros2::policy::Ownership::Shared)
            .reliability(match rel {
                pelorus_core::Reliability::Reliable => ros2::policy::Reliability::Reliable {
                    max_blocking_time: ros2::Duration::INFINITE,
                },
                pelorus_core::Reliability::BestEffort => ros2::policy::Reliability::BestEffort,
            })
            .history(ros2::policy::History::KeepLast { depth: 5 })
            .lifespan(ros2::policy::Lifespan {
                duration: ros2::Duration::INFINITE,
            })
            .build(),
        _ => ros2::QosPolicyBuilder::new()
            .durability(ros2::policy::Durability::Volatile)
            .deadline(ros2::policy::Deadline(ros2::Duration::INFINITE))
            .ownership(ros2::policy::Ownership::Shared)
            .reliability(ros2::policy::Reliability::Reliable {
                max_blocking_time: ros2::Duration::from_millis(100),
            })
            .history(ros2::policy::History::KeepLast { depth: 1 })
            .lifespan(ros2::policy::Lifespan {
                duration: ros2::Duration::INFINITE,
            })
            .build(),
    }
}

#[cfg(feature = "hiroz")]
pub fn qos_to_hiroz(value: pelorus_core::Qos) -> hiroz::qos::QosProfile {
    use std::num::NonZeroUsize;

    use hiroz::qos::{QosDurability, QosHistory, QosProfile, QosReliability};

    let (reliability, depth) = match value {
        pelorus_core::Qos::Sensor(pelorus_core::Reliability::Reliable) => {
            (QosReliability::Reliable, 5)
        }
        pelorus_core::Qos::Sensor(pelorus_core::Reliability::BestEffort) => {
            (QosReliability::BestEffort, 5)
        }
        pelorus_core::Qos::SystemDefault => (QosReliability::Reliable, 1),
    };

    QosProfile {
        reliability,
        durability: QosDurability::Volatile,
        history: QosHistory::KeepLast(NonZeroUsize::new(depth).expect("depth is non-zero")),
        ..Default::default()
    }
}

#[cfg(feature = "hiroz")]
pub fn qos_to_hiroz_static() -> hiroz::qos::QosProfile {
    use std::num::NonZeroUsize;

    use hiroz::qos::{QosDurability, QosHistory, QosProfile, QosReliability};

    QosProfile {
        reliability: QosReliability::Reliable,
        durability: QosDurability::TransientLocal,
        history: QosHistory::KeepLast(NonZeroUsize::new(1).expect("depth is non-zero")),
        ..Default::default()
    }
}

#[derive(Debug)]
pub struct Tick {
    pub scan_id: u64,
    pub received_at: std::time::Instant,
    pub deskewed_cloud: Vec<PointXYZIT>,
    pub deskewed_cloud_passthrough: Option<Vec<PointXYZIT>>,
    pub imu_buffer: Vec<ImuMsg>,
    pub map_only: bool,
    pub cloud_header: HeaderMsg,
    pub odom_time: TimeMsg,
    #[cfg(any(
        feature = "ros2-client",
        feature = "mt_pubsub",
        feature = "r2r",
        feature = "hiroz"
    ))]
    pub imu_odom: Odometry,
    pub runtime_eval: RuntimeEvaluator,
}

#[cfg(feature = "ros2-client")]
fn split_namespace_topic(input: &str) -> (String, String) {
    let parts: Vec<&str> = input.split('/').collect();

    if parts.is_empty() {
        // Handle empty string case
        (String::from(""), String::from(""))
    } else {
        let last_part = parts.last().unwrap(); // Safe because we checked is_empty()
        let mut namespace = String::new();

        for (i, &part) in parts.iter().enumerate() {
            if i < parts.len() - 1 {
                if !namespace.is_empty() {
                    namespace.push('/');
                }
                namespace.push_str(part);
            }
        }

        (namespace, "/".to_string() + last_part) // Add the leading slash to the topic
    }
}

impl SyncNode {
    #[cfg(feature = "native-drivers")]
    pub fn start_native_drivers(
        &mut self,
        config: &SensorDriverConfig,
        filtered_imu_sender: Option<std::sync::mpsc::Sender<ImuMsg>>,
        sink: Option<Arc<dyn SensorOutputSink>>,
    ) -> anyhow::Result<()> {
        let lidar_vendor = self.state.lock().unwrap().lidar_vendor;
        validate_native_lidar_vendor(&config.lidar.driver_type, lidar_vendor)?;
        let ingress = Arc::new(NativeIngress {
            state: Arc::clone(&self.state),
            lidar_driver_type: config.lidar.driver_type.clone(),
            filtered_imu_sender,
            imu_init: self.native_imu_init.clone(),
            imu_init_time: self.native_imu_init_time,
            imu_init_state: Mutex::new(NativeImuInitState {
                start_time: None,
                buffer: Some(Vec::new()),
            }),
            imu_initialized: AtomicBool::new(
                self.native_imu_init.is_none() || config.imu.driver_type == ImuDriverType::Disabled,
            ),
        });
        self.native_drivers = Some(DriverManager::start(config, ingress, sink)?);
        Ok(())
    }
    pub fn new(
        #[cfg(feature = "ros2-client")] node: &mut ros2_client::Node,
        #[cfg(feature = "r2r")] node: &mut r2r::Node,
        #[cfg(feature = "mt_pubsub")] node: &Node,
        #[cfg(feature = "hiroz")] node: &ZNode,
        #[cfg(any(feature = "mt_pubsub", feature = "hiroz"))] rt: Arc<tokio::runtime::Runtime>,
        #[cfg(feature = "mt_pubsub")] mt_pubsub_local_only: bool,
        params: &SyncParameters,
        mut inputs: SyncNodeInputs<'_>,
        #[cfg(feature = "rerun")] rec: Option<Arc<rerun::RecordingStream>>,
    ) -> anyhow::Result<(Self, std::sync::mpsc::Receiver<Tick>)> {
        let startup_tf = inputs.startup_tf_callback.take().map(|apply| {
            let initialize = inputs.imu_pose_predictor_init.take();
            let initialized = initialize.is_none();
            let gate = Arc::new(Mutex::new(TfStartup {
                apply, initialize, pending: None, resolved: false, initialized,
            }));
            if !initialized {
                let gate_for_init = gate.clone();
                inputs.imu_pose_predictor_init = Some(Arc::new(Mutex::new(move |msgs: &[ImuMsg]| {
                    gate_for_init.lock().unwrap().initialize(msgs);
                })));
            }
            info!("Waiting for requested sensor TF before starting SLAM. Configured extrinsics will not be used as fallback");
            gate
        });
        #[cfg(any(
            feature = "ros2-client",
            feature = "mt_pubsub",
            feature = "r2r",
            feature = "hiroz"
        ))]
        let enable_imu_ingress = !matches!(
            &inputs.imu_sync_mode,
            pelorus_sync::ImuSyncMode::Disabled { .. }
        );
        #[cfg(any(
            feature = "rat",
            feature = "ros2-client",
            feature = "mt_pubsub",
            feature = "r2r",
            feature = "hiroz",
            feature = "native-drivers",
        ))]
        let node_params = inputs.node_params;
        #[cfg(any(
            feature = "ros2-client",
            feature = "mt_pubsub",
            feature = "r2r",
            feature = "hiroz"
        ))]
        let imu_pose_predictor_init = inputs.imu_pose_predictor_init.clone();
        #[cfg(any(
            feature = "ros2-client",
            feature = "mt_pubsub",
            feature = "r2r",
            feature = "hiroz"
        ))]
        let filtered_imu_sender = inputs.filtered_imu_sender.clone();
        #[cfg(feature = "native-drivers")]
        let native_imu_init = inputs.imu_pose_predictor_init.clone();
        #[cfg(feature = "ros2-client")]
        let tf_listener = Arc::new(tf::TfListener::new(node));

        #[cfg(feature = "r2r")]
        let tf_listener = Arc::new(tf_r2r::TfListener::new(node));

        #[cfg(feature = "mt_pubsub")]
        // BestEffort would skip the backlog of a burst and lose static transforms for good.
        let tf_listener = Arc::new(tf_lh::TfListener::new(
            node,
            rt.clone(),
            mt_pubsub::Qos::TryReliable,
        ));

        #[cfg(feature = "hiroz")]
        let tf_listener = Arc::new(
            tf_hiroz::TfListener::new(node)
                .map_err(|e| anyhow!("Failed to create Hiroz TF listener: {e}"))?,
        );

        #[cfg(feature = "rat")]
        let synchronous_mode = node_params.synchronous_mode;

        let mut state = State::new(
            params,
            #[cfg(feature = "rerun")]
            rec.clone(),
            inputs.imu_sync_mode,
            pelorus_sync::SynchronizerHooks {
                imu_pose_prediction_undistort: inputs.imu_pose_prediction_undistort,
                imu_pose_prediction_init_guess: inputs.imu_pose_prediction_init_guess,
                ready_for_next: inputs.ready_for_next,
                realtime_imu_callback: inputs.realtime_imu_callback,
                realtime_lidar_barrier_callback: inputs.realtime_lidar_barrier_callback,
            },
        )?;
        state.startup_tf = startup_tf;
        state.startup_base_frame = inputs.startup_base_frame;

        #[cfg(any(
            feature = "ros2-client",
            feature = "mt_pubsub",
            feature = "r2r",
            feature = "hiroz"
        ))]
        let chan = {
            let mut sync = state.sync.lock().unwrap();
            sync.get_chan().unwrap()
        };

        let (tx, rx) = std::sync::mpsc::channel();
        #[cfg(not(any(
            feature = "ros2-client",
            feature = "mt_pubsub",
            feature = "r2r",
            feature = "hiroz"
        )))]
        let _ = tx;

        #[cfg(any(
            feature = "ros2-client",
            feature = "mt_pubsub",
            feature = "r2r",
            feature = "hiroz"
        ))]
        std::thread::spawn(move || {
            Self::transform_types(tx, chan);
        });

        // first option None = override not set.
        #[cfg(any(
            feature = "ros2-client",
            feature = "mt_pubsub",
            feature = "r2r",
            feature = "hiroz"
        ))]
        let imu_tf_frame: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        #[cfg(feature = "ros2-client")]
        let cloud_subscriber = node
            .create_subscription::<PointCloud2>(
                &node
                    .create_topic(
                        &ros2_client::Name::new(
                            &split_namespace_topic(&node_params.cloud_src_topic).0,
                            split_namespace_topic(&node_params.cloud_src_topic)
                                .1
                                .trim_start_matches("/"),
                        )
                        .unwrap(),
                        ros2_client::MessageTypeName::new("sensor_msgs", "PointCloud2"),
                        &qos_to_ros2(node_params.cloud_qos),
                    )
                    .unwrap(),
                Some(qos_to_ros2(node_params.cloud_qos)),
            )
            .unwrap();

        #[cfg(feature = "r2r")]
        let cloud_subscriber = node
            .subscribe::<PointCloud2>(&node_params.cloud_src_topic, node_params.cloud_qos.into())?;

        #[cfg(feature = "mt_pubsub")]
        let mut cloud_subscriber = rt.block_on(async {
            match node_params.cloud_subscriber_mode {
                PointCloudMode::PointCloud2 => {
                    Ok::<_, anyhow::Error>(MtCloudSubscriber::PointCloud2(
                        node.create_subscriber::<PointCloud2>(
                            node_params.cloud_src_topic.clone(),
                            10,
                            qos_to_mt(node_params.cloud_qos),
                        )
                        .await?,
                    ))
                }
                #[cfg(feature = "compressed-pointcloud")]
                PointCloudMode::Compressed => Ok(MtCloudSubscriber::Compressed(
                    node.create_subscriber::<CompressedPointCloud2>(
                        node_params.cloud_src_topic.clone(),
                        10,
                        qos_to_mt(node_params.cloud_qos),
                    )
                    .await?,
                )),
                #[cfg(not(feature = "compressed-pointcloud"))]
                PointCloudMode::Compressed => Err(anyhow!(
                    "Compressed point-cloud input requires the 'compressed-pointcloud' feature"
                )),
            }
        })?;

        #[cfg(feature = "hiroz")]
        let cloud_subscriber = {
            debug!(
                "Creating Hiroz cloud subscriber: topic={}, qos={:?}",
                node_params.cloud_src_topic, node_params.cloud_qos
            );
            node.create_sub::<PointCloud2>(&node_params.cloud_src_topic)
                .with_qos(qos_to_hiroz(node_params.cloud_qos))
                .build()
                .map_err(|e| anyhow!("Failed to create Hiroz cloud subscriber: {e}"))?
        };

        #[cfg(feature = "r2r")]
        let wheel_subscriber =
            node.subscribe::<Odometry>(&node_params.wheel_src_topic, node_params.wheel_qos.into())?;

        #[cfg(feature = "mt_pubsub")]
        let mut wheel_subscriber = rt.block_on(async {
            node.create_subscriber::<Odometry>(
                node_params.wheel_src_topic.clone(),
                10,
                qos_to_mt(node_params.wheel_qos),
            )
            .await
            .unwrap()
        });

        #[cfg(feature = "hiroz")]
        let wheel_subscriber = {
            debug!(
                "Creating Hiroz wheel odom subscriber: topic={}, qos={:?}",
                node_params.wheel_src_topic, node_params.wheel_qos
            );
            node.create_sub::<Odometry>(&node_params.wheel_src_topic)
                .with_qos(qos_to_hiroz(node_params.wheel_qos))
                .build()
                .map_err(|e| anyhow!("Failed to create Hiroz wheel subscriber: {e}"))?
        };

        // Share with cloud callback, TODO maybe the only one who needs it, so remove mutex and arc?
        let state = Arc::new(std::sync::Mutex::new(state));

        #[cfg(all(
            not(feature = "no-imu-sync"),
            any(
                feature = "ros2-client",
                feature = "mt_pubsub",
                feature = "r2r",
                feature = "hiroz"
            )
        ))]
        struct ImuInitState {
            start_time: Option<f64>,
            buffer: Option<Vec<ImuMsg>>,
        }
        #[cfg(all(
            not(feature = "no-imu-sync"),
            any(
                feature = "ros2-client",
                feature = "mt_pubsub",
                feature = "r2r",
                feature = "hiroz"
            )
        ))]
        let imu_init_state = Arc::new(Mutex::new(ImuInitState {
            start_time: None,
            buffer: Some(Vec::new()),
        }));

        #[cfg(any(
            feature = "ros2-client",
            feature = "mt_pubsub",
            feature = "r2r",
            feature = "hiroz"
        ))]
        let tf_listener_clone = tf_listener.clone();
        #[cfg(any(
            feature = "ros2-client",
            feature = "mt_pubsub",
            feature = "r2r",
            feature = "hiroz"
        ))]
        let state_clone = state.clone();
        #[cfg(any(
            feature = "ros2-client",
            feature = "mt_pubsub",
            feature = "r2r",
            feature = "hiroz"
        ))]
        let imu_tf_frame_lidar_inner = Arc::clone(&imu_tf_frame);
        // #[cfg(all(not(feature = "no-imu-sync"), feature = "mt_pubsub"))]
        // let lidar_imu_init_buffer = imu_init_buffer.clone();
        #[cfg(feature = "ros2-client")]
        let filtered_imu_sender_ros2 = filtered_imu_sender.clone();
        #[cfg(feature = "ros2-client")]
        std::thread::spawn(move || {
            loop {
                match cloud_subscriber.take() {
                    Err(e) => {
                        error!("Could not read cloud: {e}");
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    Ok(Some((msg, _msg_info))) => {
                        // error!("received cloud message");
                        // TODO if config says we are system time, we can also the source timestamp of dds
                        /*if let Some(source_timestamp) = msg_info.source_timestamp() {
                            msg.header.stamp = source_timestamp
                        }*/
                        let imu_tf_frame_lidar_inner =
                            imu_tf_frame_lidar_inner.clone().lock().unwrap().clone();

                        match cloud_callback(
                            state_clone.clone(),
                            tf_listener_clone.clone(),
                            msg,
                            imu_tf_frame_lidar_inner,
                        ) {
                            Ok(_) => {}
                            Err(e) => {
                                error!("Error in cloud callback: {e}");
                            }
                        }
                    }
                    Ok(None) => {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }
            }
        });
        #[cfg(feature = "r2r")]
        std::thread::spawn(move || {
            future::block_on(async {
                cloud_subscriber
                    .for_each(|msg| {
                        let imu_tf_frame_lidar_inner =
                            imu_tf_frame_lidar_inner.clone().lock().unwrap().clone();

                        match cloud_callback(
                            state_clone.clone(),
                            tf_listener_clone.clone(),
                            msg,
                            imu_tf_frame_lidar_inner,
                        ) {
                            Ok(_) => {}
                            Err(e) => {
                                error!("{:?}", e);
                            }
                        }
                    })
                    .await;
            })
        });

        #[cfg(feature = "mt_pubsub")]
        {
            let wheel_sub_rt = Arc::clone(&rt);
            let state_wheel = state.clone();
            let ext = nalgebra::Isometry3::from_parts(
                nalgebra::Translation3::new(0.0, 0.0, 0.0),
                nalgebra::UnitQuaternion::from_matrix(&params.wheel_to_imu_rotation),
            );
            std::thread::spawn(move || {
                wheel_sub_rt.block_on(async {
                    while let Some(msg) = wheel_subscriber.next().await {
                        // Transform odom to IMU frame and send to sync for sync'd processing

                        use pelorus_core::OdomMsg;
                        let v = nalgebra::Vector3::new(
                            msg.twist.twist.linear.x,
                            msg.twist.twist.linear.y,
                            msg.twist.twist.linear.z,
                        );
                        let t = ext.transform_vector(&v);

                        let odom_msg = OdomMsg {
                            twist_linear: t,
                            twist_angular: nalgebra::Vector3::new(
                                msg.twist.twist.angular.x,
                                msg.twist.twist.angular.y,
                                msg.twist.twist.angular.z,
                            ),
                            twist_cov: msg.twist.covariance,
                            stamp: pelorus_sync::TimeMsg {
                                sec: msg.header.stamp.sec,
                                nanosec: msg.header.stamp.nanosec,
                            },
                        };

                        // Send to sync's sync worker for synchronized lidar-imu-odom processing
                        let state_lock = state_wheel.lock().unwrap();
                        state_lock.sync.lock().unwrap().add_odom(odom_msg);
                    }
                })
            });
        }
        #[cfg(feature = "r2r")]
        {
            let state_wheel = state.clone();
            let ext = nalgebra::Isometry3::from_parts(
                nalgebra::Translation3::new(0.0, 0.0, 0.0),
                nalgebra::UnitQuaternion::from_matrix(&params.wheel_to_imu_rotation),
            );
            std::thread::spawn(move || {
                future::block_on(async {
                    wheel_subscriber
                        .for_each(|msg| {
                            let mut odom_msg = odom_to_sync(msg);
                            odom_msg.twist_linear = ext.transform_vector(&odom_msg.twist_linear);
                            let state_lock = state_wheel.lock().unwrap();
                            state_lock.sync.lock().unwrap().add_odom(odom_msg);
                        })
                        .await
                })
            });
        }
        #[cfg(feature = "hiroz")]
        {
            let wheel_sub_rt = Arc::clone(&rt);
            let state_wheel = state.clone();
            let ext = nalgebra::Isometry3::from_parts(
                nalgebra::Translation3::new(0.0, 0.0, 0.0),
                nalgebra::UnitQuaternion::from_matrix(&params.wheel_to_imu_rotation),
            );
            wheel_sub_rt.spawn(async move {
                let mut received_first = false;
                while let Ok(msg) = wheel_subscriber.async_recv().await {
                    if !received_first {
                        received_first = true;
                        info!(
                            "Received first Hiroz wheel odom message: frame={}, stamp={}.{}",
                            msg.header.frame_id, msg.header.stamp.sec, msg.header.stamp.nanosec
                        );
                    }
                    let mut odom_msg = odom_to_sync(msg);
                    odom_msg.twist_linear = ext.transform_vector(&odom_msg.twist_linear);
                    let state_lock = state_wheel.lock().unwrap();
                    state_lock.sync.lock().unwrap().add_odom(odom_msg);
                }
            });
        }
        // let state_clone = state.clone();

        #[cfg(feature = "mt_pubsub")]
        {
            let cloud_sub_rt = Arc::clone(&rt);
            std::thread::spawn(move || {
                cloud_sub_rt.block_on(async {
                    loop {
                        let msg = match &mut cloud_subscriber {
                            MtCloudSubscriber::PointCloud2(subscriber) => {
                                if mt_pubsub_local_only {
                                    let Some(message) = subscriber.next_archived().await else {
                                        break;
                                    };
                                    IncomingMtPointCloud::Archived(message)
                                } else {
                                    let Some(msg) = subscriber.next().await else {
                                        break;
                                    };
                                    IncomingMtPointCloud::Owned(msg)
                                }
                            }
                            #[cfg(feature = "compressed-pointcloud")]
                            MtCloudSubscriber::Compressed(subscriber) => {
                                let Some(msg) = subscriber.next().await else {
                                    break;
                                };
                                IncomingMtPointCloud::Compressed(msg)
                            }
                        };
                        // #[cfg(not(feature = "no-imu-sync"))]
                        // {
                        //     let b = lidar_imu_init_buffer.lock().unwrap();
                        //     if b.is_some() {
                        //         continue;
                        //     }
                        // }
                        // debug!("received cloud");
                        let imu_tf_frame_lidar_inner =
                            imu_tf_frame_lidar_inner.clone().lock().unwrap().clone();

                        #[cfg(feature = "rat")]
                        {
                            let mut ts = minot_sd_start_timestamp();
                            rat::bacon_async("pelorus:sd_start", &mut ts, rat::VariableType::F64)
                                .await
                                .ok();
                        }

                        match cloud_callback(
                            state_clone.clone(),
                            tf_listener_clone.clone(),
                            msg,
                            imu_tf_frame_lidar_inner,
                        ) {
                            Ok(_) => {}
                            Err(e) => {
                                error!("{:?}", e);
                            }
                        }
                    }
                })
            });
        }
        #[cfg(feature = "hiroz")]
        {
            let cloud_sub_rt = Arc::clone(&rt);
            cloud_sub_rt.spawn(async move {
                let mut received_first = false;
                while let Ok(msg) = cloud_subscriber.async_recv().await {
                    if !received_first {
                        received_first = true;
                        info!(
                            "Received first Hiroz cloud message: frame={}, stamp={}.{}, size={}x{}, data_bytes={}",
                            msg.header.frame_id,
                            msg.header.stamp.sec,
                            msg.header.stamp.nanosec,
                            msg.width,
                            msg.height,
                            msg.data.len()
                        );
                    }
                    let imu_tf_frame_lidar_inner =
                        imu_tf_frame_lidar_inner.clone().lock().unwrap().clone();

                    #[cfg(feature = "rat")]
                    {
                        let mut ts = minot_sd_start_timestamp();
                        rat::bacon_async("pelorus:sd_start", &mut ts, rat::VariableType::F64)
                            .await
                            .ok();
                    }

                    match cloud_callback(
                        state_clone.clone(),
                        tf_listener_clone.clone(),
                        msg,
                        imu_tf_frame_lidar_inner,
                    ) {
                        Ok(_) => {}
                        Err(e) => {
                            error!("{:?}", e);
                        }
                    }
                }
            });
        }
        #[cfg(any(
            feature = "ros2-client",
            feature = "mt_pubsub",
            feature = "r2r",
            feature = "hiroz"
        ))]
        let state_clone = state.clone();
        #[cfg(feature = "ros2-client")]
        let imu_subscription = node
            .create_subscription::<Imu>(
                &node
                    .create_topic(
                        &ros2_client::Name::new(
                            &split_namespace_topic(&node_params.imu_src_topic).0,
                            split_namespace_topic(&node_params.imu_src_topic)
                                .1
                                .trim_start_matches("/"),
                        )
                        .unwrap(),
                        ros2_client::MessageTypeName::new("sensor_msgs", "Imu"),
                        &qos_to_ros2(node_params.imu_qos),
                    )
                    .unwrap(),
                None,
            )
            .unwrap();

        #[cfg(all(feature = "r2r", not(feature = "no-imu-sync")))]
        let mut imu_subscription =
            node.subscribe::<Imu>(&node_params.imu_src_topic, node_params.imu_qos.into())?;

        #[cfg(feature = "hiroz")]
        let imu_subscription = {
            debug!(
                "Creating Hiroz IMU subscriber: topic={}, qos={:?}",
                node_params.imu_src_topic, node_params.imu_qos
            );
            node.create_sub::<Imu>(&node_params.imu_src_topic)
                .with_qos(qos_to_hiroz(node_params.imu_qos))
                .build()
                .map_err(|e| anyhow!("Failed to create Hiroz IMU subscriber: {e}"))?
        };

        // do not publish filtered imu if no topic is provided
        #[cfg(feature = "ros2-client")]
        let imu_filtered_publisher = if node_params.imu_filtered_topic.is_empty() {
            None
        } else {
            Some(Arc::new(Mutex::new(
                node.create_publisher(
                    &node
                        .create_topic(
                            &ros2_client::Name::new(
                                &split_namespace_topic(&node_params.imu_filtered_topic).0,
                                split_namespace_topic(&node_params.imu_filtered_topic)
                                    .1
                                    .trim_start_matches("/"),
                            )
                            .unwrap(),
                            ros2_client::MessageTypeName::new("sensor_msgs", "Imu"),
                            &qos_to_ros2(node_params.imu_filtered_qos),
                        )
                        .unwrap(),
                    None,
                )
                .unwrap(),
            )))
        };
        #[cfg(feature = "r2r")]
        let imu_filtered_publisher = if node_params.imu_filtered_topic.is_empty() {
            None
        } else {
            Some(Arc::new(Mutex::new(node.create_publisher::<Imu>(
                &node_params.imu_filtered_topic,
                node_params.imu_filtered_qos.into(),
            )?)))
        };
        #[cfg(feature = "hiroz")]
        let imu_filtered_publisher = if node_params.imu_filtered_topic.is_empty() {
            None
        } else {
            Some(Arc::new(
                node.create_pub::<Imu>(&node_params.imu_filtered_topic)
                    .with_qos(qos_to_hiroz(node_params.imu_filtered_qos))
                    .build()
                    .map_err(|e| anyhow!("Failed to create Hiroz filtered IMU publisher: {e}"))?,
            ))
        };

        #[cfg(any(
            feature = "ros2-client",
            feature = "mt_pubsub",
            feature = "r2r",
            feature = "hiroz"
        ))]
        let imu_tf_frame_lidar_inner = Arc::clone(&imu_tf_frame);
        #[cfg(any(
            feature = "ros2-client",
            feature = "mt_pubsub",
            feature = "r2r",
            feature = "hiroz"
        ))]
        let init_imu_time = node_params.imu_predictor_init_time_sec;
        #[cfg(feature = "ros2-client")]
        std::thread::spawn(move || {
            if !enable_imu_ingress {
                return;
            }
            // TODO use while let
            let imu_pose_predictor_init = imu_pose_predictor_init.clone();
            loop {
                match imu_subscription.take() {
                    Err(e) => {
                        error!("Could not read imu: {e}");
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    Ok(Some((msg, _msg_info))) => {
                        // debug!("received imu message");
                        imu_tf_frame_lidar_inner
                            .clone()
                            .lock()
                            .unwrap()
                            .get_or_insert(msg.header.frame_id.clone());
                        let imu_init_state = imu_init_state.clone();

                        let imu_sync = imu_to_sync(msg.clone());
                        let curr_stamp = timestamp_to_secs(&imu_sync.stamp);
                        #[cfg(feature = "rat")]
                        let mut minot_bootstrap_trigger: Option<
                            &'static str,
                        > = None;

                        if let Some(imu_pose_predictor_init) = imu_pose_predictor_init.as_ref() {
                            let mut imu_pose_predictor_init =
                                imu_pose_predictor_init.lock().unwrap();
                            let mut init_state = imu_init_state.lock().unwrap();

                            if init_state.start_time.is_none() {
                                init_state.start_time = Some(curr_stamp);
                            }

                            if let Some(init_time) = init_state.start_time {
                                let delta = curr_stamp - init_time;
                                if delta < init_imu_time {
                                    if let Some(ref mut buffer) = init_state.buffer {
                                        // Buffer IMU for EKF init callback, but still forward it
                                        // to sync so scan/IMU sync is not backend-timing dependent.
                                        buffer.push(imu_sync.clone());

                                        #[cfg(feature = "rat")]
                                        {
                                            minot_bootstrap_trigger = Some(MINOT_IMU_INIT_TICK);
                                        }
                                    }
                                } else {
                                    // apply the init buffer
                                    if let Some(buffer) = init_state.buffer.take() {
                                        imu_pose_predictor_init(&buffer);
                                        info!("EKF: IMU initialization samples collected");
                                        #[cfg(feature = "rat")]
                                        {
                                            minot_bootstrap_trigger = Some(MINOT_LIO_START_TICK);
                                        }
                                    }
                                }
                            }
                        }

                        let sync = state_clone.lock().unwrap().sync.clone();
                        let filtered = sync.lock().unwrap().add_imu(imu_sync);
                        #[cfg(feature = "rat")]
                        if let Some(trigger) = minot_bootstrap_trigger {
                            publish_minot_trigger(synchronous_mode, trigger);
                        }
                        if let Some(filtered) = filtered {
                            if let Some(sender) = filtered_imu_sender_ros2.as_ref() {
                                let _ = sender.send(filtered.clone());
                            }
                            if let Some(imu_filtered_publisher) = imu_filtered_publisher.as_ref() {
                                imu_filtered_publisher
                                    .clone()
                                    .lock()
                                    .unwrap()
                                    .publish(&sync_to_imu(filtered))
                                    .expect("could not publish filtered imu");
                            }
                        }
                    }
                    Ok(None) => {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }
            }
        });

        #[cfg(feature = "mt_pubsub")]
        if enable_imu_ingress {
            let filtered_imu_sender_mt = filtered_imu_sender.clone();
            let imu_sub_rt = Arc::clone(&rt);
            let mut imu_sub = imu_sub_rt.block_on(async {
                node.create_subscriber::<Imu>(
                    node_params.imu_src_topic.clone(),
                    100,
                    qos_to_mt(node_params.imu_qos),
                )
                .await
                .unwrap()
            });

            let imu_imu_init_state = imu_init_state.clone();
            std::thread::spawn(move || {
                let imu_pose_predictor_init = imu_pose_predictor_init.clone();
                while let Some(msg) = imu_sub_rt.block_on(imu_sub.next()) {
                    // debug!("received imu");
                    let sync = {
                        let state = state_clone.lock().unwrap();
                        imu_tf_frame_lidar_inner
                            .clone()
                            .lock()
                            .unwrap()
                            .get_or_insert(msg.header.frame_id.clone());
                        state.sync.clone()
                    };
                    let imu_init_state = imu_imu_init_state.clone();

                    let imu_sync = imu_to_sync(msg.clone());
                    let curr_stamp = timestamp_to_secs(&imu_sync.stamp);
                    #[cfg(feature = "rat")]
                    let mut minot_bootstrap_trigger: Option<&'static str> = None;

                    if let Some(imu_pose_predictor_init) = imu_pose_predictor_init.as_ref() {
                        let mut imu_pose_predictor_init = imu_pose_predictor_init.lock().unwrap();
                        let mut init_state = imu_init_state.lock().unwrap();

                        if init_state.start_time.is_none() {
                            init_state.start_time = Some(curr_stamp);
                        }

                        if let Some(init_time) = init_state.start_time {
                            let delta = curr_stamp - init_time;
                            if delta < init_imu_time {
                                if let Some(ref mut buffer) = init_state.buffer {
                                    // Buffer IMU for EKF init callback, but still forward it
                                    // to sync so scan/IMU sync is not backend-timing dependent.
                                    buffer.push(imu_sync.clone());

                                    #[cfg(feature = "rat")]
                                    {
                                        minot_bootstrap_trigger = Some(MINOT_IMU_INIT_TICK);
                                    }
                                }
                            } else {
                                // apply the init buffer
                                if let Some(buffer) = init_state.buffer.take() {
                                    imu_pose_predictor_init(&buffer);
                                    info!("EKF: IMU initialization samples collected");
                                    #[cfg(feature = "rat")]
                                    {
                                        minot_bootstrap_trigger = Some(MINOT_LIO_START_TICK);
                                    }
                                }
                            }
                        }
                    }

                    let filtered = sync.lock().unwrap().add_imu(imu_sync);
                    #[cfg(feature = "rat")]
                    if let Some(trigger) = minot_bootstrap_trigger {
                        publish_minot_trigger(synchronous_mode, trigger);
                    }

                    if let (Some(filtered), Some(sender)) =
                        (filtered, filtered_imu_sender_mt.as_ref())
                    {
                        let _ = sender.send(filtered);
                    }
                }
            });
        }

        // TODO inline with upper because logic is same but conditionals around block on and err handling
        #[cfg(all(feature = "r2r", not(feature = "no-imu-sync")))]
        let filtered_imu_sender_r2r = filtered_imu_sender.clone();
        #[cfg(all(feature = "r2r", not(feature = "no-imu-sync")))]
        std::thread::spawn(move || {
            if !enable_imu_ingress {
                return;
            }
            let imu_pose_predictor_init = imu_pose_predictor_init.clone();
            while let Some(msg) = future::block_on(imu_subscription.next()) {
                let sync = {
                    let state = state_clone.lock().unwrap();
                    imu_tf_frame_lidar_inner
                        .clone()
                        .lock()
                        .unwrap()
                        .get_or_insert(msg.header.frame_id.clone());
                    state.sync.clone()
                };
                let imu_init_state = imu_init_state.clone();

                let imu_sync = imu_to_sync(msg.clone());
                let curr_stamp = timestamp_to_secs(&imu_sync.stamp);
                #[cfg(feature = "rat")]
                let mut minot_bootstrap_trigger: Option<&'static str> = None;

                if let Some(imu_pose_predictor_init) = imu_pose_predictor_init.as_ref() {
                    let mut imu_pose_predictor_init = imu_pose_predictor_init.lock().unwrap();
                    let mut init_state = imu_init_state.lock().unwrap();

                    if init_state.start_time.is_none() {
                        init_state.start_time = Some(curr_stamp);
                    }

                    if let Some(init_time) = init_state.start_time {
                        let delta = curr_stamp - init_time;
                        if delta < init_imu_time {
                            if let Some(ref mut buffer) = init_state.buffer {
                                // Buffer IMU for EKF init callback, but still forward it
                                // to sync so scan/IMU sync is not backend-timing dependent.
                                buffer.push(imu_sync.clone());

                                #[cfg(feature = "rat")]
                                {
                                    minot_bootstrap_trigger = Some(MINOT_IMU_INIT_TICK);
                                }
                            }
                        } else {
                            // apply the init buffer
                            if let Some(buffer) = init_state.buffer.take() {
                                imu_pose_predictor_init(&buffer);
                                info!("EKF: IMU initialization samples collected");
                                #[cfg(feature = "rat")]
                                {
                                    minot_bootstrap_trigger = Some(MINOT_LIO_START_TICK);
                                }
                            }
                        }
                    }
                }

                let filtered = sync.lock().unwrap().add_imu(imu_sync);
                #[cfg(feature = "rat")]
                if let Some(trigger) = minot_bootstrap_trigger {
                    publish_minot_trigger(synchronous_mode, trigger);
                }
                if let Some(filtered) = filtered {
                    if let Some(sender) = filtered_imu_sender_r2r.as_ref() {
                        let _ = sender.send(filtered.clone());
                    }
                    if let Some(imu_filtered_publisher) = imu_filtered_publisher.as_ref() {
                        imu_filtered_publisher
                            .clone()
                            .lock()
                            .unwrap()
                            .publish(&sync_to_imu(filtered))
                            .expect("could not publish filtered imu");
                    }
                }
            }
        });

        #[cfg(feature = "hiroz")]
        if enable_imu_ingress {
            let filtered_imu_sender_hiroz = filtered_imu_sender.clone();
            let imu_sub_rt = Arc::clone(&rt);
            let imu_imu_init_state = imu_init_state.clone();
            imu_sub_rt.spawn(async move {
                let imu_pose_predictor_init = imu_pose_predictor_init.clone();
                let mut received_first = false;
                while let Ok(msg) = imu_subscription.async_recv().await {
                    if !received_first {
                        received_first = true;
                        info!(
                            "Received first Hiroz IMU message: frame={}, stamp={}.{}",
                            msg.header.frame_id, msg.header.stamp.sec, msg.header.stamp.nanosec
                        );
                    }
                    let sync = {
                        let state = state_clone.lock().unwrap();
                        imu_tf_frame_lidar_inner
                            .clone()
                            .lock()
                            .unwrap()
                            .get_or_insert(msg.header.frame_id.clone());
                        state.sync.clone()
                    };
                    let imu_init_state = imu_imu_init_state.clone();

                    let imu_sync = imu_to_sync(msg.clone());
                    let curr_stamp = timestamp_to_secs(&imu_sync.stamp);
                    #[cfg(feature = "rat")]
                    let mut minot_bootstrap_trigger: Option<&'static str> = None;

                    if let Some(imu_pose_predictor_init) = imu_pose_predictor_init.as_ref() {
                        let mut imu_pose_predictor_init = imu_pose_predictor_init.lock().unwrap();
                        let mut init_state = imu_init_state.lock().unwrap();

                        if init_state.start_time.is_none() {
                            init_state.start_time = Some(curr_stamp);
                        }

                        if let Some(init_time) = init_state.start_time {
                            let delta = curr_stamp - init_time;
                            if delta < init_imu_time {
                                if let Some(ref mut buffer) = init_state.buffer {
                                    buffer.push(imu_sync.clone());

                                    #[cfg(feature = "rat")]
                                    {
                                        minot_bootstrap_trigger = Some(MINOT_IMU_INIT_TICK);
                                    }
                                }
                            } else if let Some(buffer) = init_state.buffer.take() {
                                imu_pose_predictor_init(&buffer);
                                info!("EKF: IMU initialization samples collected");
                                #[cfg(feature = "rat")]
                                {
                                    minot_bootstrap_trigger = Some(MINOT_LIO_START_TICK);
                                }
                            }
                        }
                    }

                    let filtered = sync.lock().unwrap().add_imu(imu_sync);
                    #[cfg(feature = "rat")]
                    if let Some(trigger) = minot_bootstrap_trigger {
                        publish_minot_trigger_async(synchronous_mode, trigger).await;
                    }
                    if let Some(filtered) = filtered {
                        if let Some(sender) = filtered_imu_sender_hiroz.as_ref() {
                            let _ = sender.send(filtered.clone());
                        }
                        if let Some(imu_filtered_publisher) = imu_filtered_publisher.as_ref() {
                            imu_filtered_publisher
                                .async_publish(&sync_to_imu(filtered))
                                .await
                                .expect("could not publish filtered imu");
                        }
                    }
                }
            });
        }

        Ok((
            Self {
                state,
                #[cfg(any(
                    feature = "ros2-client",
                    feature = "mt_pubsub",
                    feature = "r2r",
                    feature = "hiroz"
                ))]
                tf_listener,
                logger: "".to_string(), // TODO rm logger prefix
                #[cfg(feature = "native-drivers")]
                native_drivers: None,
                #[cfg(feature = "native-drivers")]
                native_imu_init,
                #[cfg(feature = "native-drivers")]
                native_imu_init_time: node_params.imu_predictor_init_time_sec,
            },
            rx,
        ))
    }

    pub fn request_shutdown(&self) {
        if let Ok(state) = self.state.lock()
            && let Ok(mut sync) = state.sync.lock()
        {
            sync.shutdown();
        }
    }

    /// Translates ROS types to Synchronizer Types
    #[cfg(any(
        feature = "ros2-client",
        feature = "mt_pubsub",
        feature = "r2r",
        feature = "hiroz"
    ))]
    pub fn transform_types(
        sender: std::sync::mpsc::Sender<Tick>,
        receiver: std::sync::mpsc::Receiver<SyncProcessedCloud>,
    ) {
        while let Ok(synced_data) = receiver.recv() {
            let pose = PoseWithCovariance {
                pose: Pose {
                    position: Point {
                        x: synced_data.end_of_frame_guess.translation.vector.x,
                        y: synced_data.end_of_frame_guess.translation.vector.y,
                        z: synced_data.end_of_frame_guess.translation.vector.z,
                    },
                    orientation: Quaternion {
                        x: synced_data.end_of_frame_guess.rotation.quaternion().i,
                        y: synced_data.end_of_frame_guess.rotation.quaternion().j,
                        z: synced_data.end_of_frame_guess.rotation.quaternion().k,
                        w: synced_data.end_of_frame_guess.rotation.quaternion().w,
                    },
                },
                #[cfg(not(feature = "r2r"))]
                covariance: synced_data.pose_covariance,
                #[cfg(feature = "r2r")]
                covariance: synced_data.pose_covariance.to_vec(),
            };

            let twist = TwistWithCovariance {
                twist: Twist {
                    linear: Vector3 {
                        x: synced_data.end_of_frame_twist.linear.x,
                        y: synced_data.end_of_frame_twist.linear.y,
                        z: synced_data.end_of_frame_twist.linear.z,
                    },
                    angular: Vector3 {
                        x: synced_data.end_of_frame_twist.angular.x,
                        y: synced_data.end_of_frame_twist.angular.y,
                        z: synced_data.end_of_frame_twist.angular.z,
                    },
                },
                #[cfg(not(feature = "r2r"))]
                covariance: synced_data.twist_covariance,
                #[cfg(feature = "r2r")]
                covariance: synced_data.twist_covariance.to_vec(),
            };

            let odom = Odometry {
                header: Header {
                    stamp: Time {
                        sec: synced_data.end_of_frame_time.sec,
                        nanosec: synced_data.end_of_frame_time.nanosec,
                    },
                    frame_id: synced_data.header.frame_id.clone(),
                },
                child_frame_id: synced_data.imu_frame,
                pose,
                twist,
            };

            let res = Tick {
                scan_id: synced_data.scan_id,
                received_at: synced_data.received_at,
                map_only: synced_data.should_map_only,
                deskewed_cloud: synced_data.cloud,
                deskewed_cloud_passthrough: synced_data.passthrough,
                imu_buffer: synced_data.imu_buffer,
                cloud_header: synced_data.header,
                imu_odom: odom,
                runtime_eval: synced_data.runtime_eval,
                odom_time: synced_data.lidar_end_time_abs,
            };

            if sender.send(res).is_err() {
                warn!("Closing transform thread.");
                return;
            }
        }
    }
}

impl Drop for SyncNode {
    fn drop(&mut self) {
        self.request_shutdown();
    }
}

// only used with r2r
#[cfg(any(feature = "r2r", feature = "hiroz"))]
fn odom_to_sync(msg: Odometry) -> OdomMsg {
    OdomMsg {
        twist_linear: nalgebra::Vector3::new(
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ),
        twist_angular: nalgebra::Vector3::new(
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ),
        twist_cov: [
            msg.twist.covariance[0],
            msg.twist.covariance[1],
            msg.twist.covariance[2],
            msg.twist.covariance[3],
            msg.twist.covariance[4],
            msg.twist.covariance[5],
            msg.twist.covariance[6],
            msg.twist.covariance[7],
            msg.twist.covariance[8],
            msg.twist.covariance[9],
            msg.twist.covariance[10],
            msg.twist.covariance[11],
            msg.twist.covariance[12],
            msg.twist.covariance[13],
            msg.twist.covariance[14],
            msg.twist.covariance[15],
            msg.twist.covariance[16],
            msg.twist.covariance[17],
            msg.twist.covariance[18],
            msg.twist.covariance[19],
            msg.twist.covariance[20],
            msg.twist.covariance[21],
            msg.twist.covariance[22],
            msg.twist.covariance[23],
            msg.twist.covariance[24],
            msg.twist.covariance[25],
            msg.twist.covariance[26],
            msg.twist.covariance[27],
            msg.twist.covariance[28],
            msg.twist.covariance[29],
            msg.twist.covariance[30],
            msg.twist.covariance[31],
            msg.twist.covariance[32],
            msg.twist.covariance[33],
            msg.twist.covariance[34],
            msg.twist.covariance[35],
        ],
        stamp: ros_pointcloud2::ros::TimeMsg {
            sec: msg.header.stamp.sec,
            nanosec: msg.header.stamp.nanosec,
        },
    }
}

#[cfg(any(
    feature = "ros2-client",
    feature = "mt_pubsub",
    feature = "r2r",
    feature = "hiroz"
))]
fn imu_to_sync(msg: Imu) -> ImuMsg {
    ImuMsg {
        stamp: ros_pointcloud2::ros::TimeMsg {
            sec: msg.header.stamp.sec,
            nanosec: msg.header.stamp.nanosec,
        },
        frame_id: msg.header.frame_id,
        orientation: [
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
        ],
        orientation_covariance: [
            msg.orientation_covariance[0],
            msg.orientation_covariance[1],
            msg.orientation_covariance[2],
            msg.orientation_covariance[3],
            msg.orientation_covariance[4],
            msg.orientation_covariance[5],
            msg.orientation_covariance[6],
            msg.orientation_covariance[7],
            msg.orientation_covariance[8],
        ],
        angular_velocity: [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ],
        angular_velocity_covariance: [
            msg.angular_velocity_covariance[0],
            msg.angular_velocity_covariance[1],
            msg.angular_velocity_covariance[2],
            msg.angular_velocity_covariance[3],
            msg.angular_velocity_covariance[4],
            msg.angular_velocity_covariance[5],
            msg.angular_velocity_covariance[6],
            msg.angular_velocity_covariance[7],
            msg.angular_velocity_covariance[8],
        ],
        linear_acceleration: [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ],
        linear_acceleration_covariance: [
            msg.linear_acceleration_covariance[0],
            msg.linear_acceleration_covariance[1],
            msg.linear_acceleration_covariance[2],
            msg.linear_acceleration_covariance[3],
            msg.linear_acceleration_covariance[4],
            msg.linear_acceleration_covariance[5],
            msg.linear_acceleration_covariance[6],
            msg.linear_acceleration_covariance[7],
            msg.linear_acceleration_covariance[8],
        ],
    }
}

#[cfg(all(test, feature = "native-drivers"))]
mod native_ingress_tests {
    use super::*;
    use pelorus_drivers::{RawLidarPoint, ReceiveMetadata};

    fn scan(offset_seconds: f64) -> RawLidarScan {
        RawLidarScan {
            frame_id: "os_sensor".into(),
            stamp: 1_700_000_000.25,
            points: vec![RawLidarPoint {
                xyz: [1.0, 2.0, 3.0],
                intensity: 0.5,
                time_offset_secs: offset_seconds,
            }],
            receive: ReceiveMetadata {
                received_at: std::time::UNIX_EPOCH,
                sender: None,
                packets_received: 1,
                packets_dropped: 0,
            },
        }
    }

    #[test]
    fn ouster_scan_uses_uint32_nanosecond_timestamp_field() {
        let cloud = native_scan_to_cloud(scan(0.012_345_678), &LidarDriverType::Ouster).unwrap();

        assert_eq!(cloud.header.frame_id, "os_sensor");
        assert_eq!(cloud.header.stamp.sec, 1_700_000_000);
        assert_eq!(cloud.header.stamp.nanosec, 250_000_000);
        assert!(cloud.fields.iter().any(|field| field.name == "t"));
        assert!(!cloud.fields.iter().any(|field| field.name == "time"));

        let points: Vec<OusterPoint> = cloud.try_into_iter().unwrap().collect();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].t, 12_345_678);
        assert_eq!(points[0].intensity, 127.5);
    }

    #[test]
    fn velodyne_scan_keeps_float_seconds_timestamp_field() {
        let cloud = native_scan_to_cloud(scan(0.0125), &LidarDriverType::Velodyne).unwrap();

        assert!(cloud.fields.iter().any(|field| field.name == "time"));
        assert!(!cloud.fields.iter().any(|field| field.name == "t"));

        let points: Vec<VelodynePoint> = cloud.try_into_iter().unwrap().collect();
        assert!((points[0].time - 0.0125).abs() < f32::EPSILON);
    }

    #[test]
    fn robosense_scan_carries_an_absolute_epoch_per_point() {
        let cloud = native_scan_to_cloud(scan(0.012_345), &LidarDriverType::Robosense).unwrap();

        assert_eq!(cloud.header.stamp.sec, 1_700_000_000);
        assert_eq!(cloud.header.stamp.nanosec, 250_000_000);
        assert!(cloud.fields.iter().any(|field| field.name == "timestamp"));
        assert!(!cloud.fields.iter().any(|field| field.name == "t"));

        // pelorus_sync recovers the per-point offset by subtracting the header
        // stamp again, so that subtraction is what has to come back intact.
        // An absolute epoch near 1.7e9 has an f64 ULP of ~238 ns, so the offset
        // comes back quantized to ~76 ns here. That is far below the 1 us the
        // wire format can express, but it does put a floor under this check.
        let points: Vec<RobosensePoint> = cloud.try_into_iter().unwrap().collect();
        assert_eq!(points.len(), 1);
        assert!((points[0].timestamp - 1_700_000_000.25 - 0.012_345).abs() < 1e-6);
        assert_eq!(points[0].intensity, 127.5);
    }

    #[test]
    fn robosense_microsecond_offsets_survive_the_absolute_epoch() {
        // Block time offsets are whole microseconds. Carrying them as an
        // absolute epoch near 1.7e9 leaves only ~0.4 us of f64 resolution, so
        // check the smallest offset the wire format can express still returns.
        let cloud = native_scan_to_cloud(scan(0.000_001), &LidarDriverType::Robosense).unwrap();
        let points: Vec<RobosensePoint> = cloud.try_into_iter().unwrap().collect();
        let recovered = points[0].timestamp - 1_700_000_000.25;
        assert!(
            (recovered - 0.000_001).abs() < 1e-7,
            "1 us offset came back as {recovered} s"
        );
    }

    #[test]
    fn a_reordered_robosense_packet_may_predate_the_scan_stamp() {
        // A packet that arrives before the one that opened the frame produces a
        // negative offset. Unlike Ouster, which packs offsets into unsigned
        // nanoseconds and must reject these, Robosense carries a signed
        // absolute epoch, so it round-trips and pelorus_sync clamps it at zero.
        let cloud = native_scan_to_cloud(scan(-0.000_2), &LidarDriverType::Robosense).unwrap();
        let points: Vec<RobosensePoint> = cloud.try_into_iter().unwrap().collect();
        assert!(points[0].timestamp < 1_700_000_000.25);
    }

    #[test]
    fn incompatible_native_driver_and_vendor_are_rejected() {
        assert!(
            validate_native_lidar_vendor(&LidarDriverType::Ouster, LidarVendor::Velodyne).is_err()
        );
        assert!(
            validate_native_lidar_vendor(&LidarDriverType::Velodyne, LidarVendor::Ouster).is_err()
        );
        assert!(
            validate_native_lidar_vendor(&LidarDriverType::Ouster, LidarVendor::Ouster).is_ok()
        );
        // The driver writes the RobosensePoint layout, so pelorus_sync has to
        // be reading the scan back with the matching vendor.
        assert!(
            validate_native_lidar_vendor(&LidarDriverType::Robosense, LidarVendor::Robosense)
                .is_ok()
        );
        assert!(
            validate_native_lidar_vendor(&LidarDriverType::Robosense, LidarVendor::Ouster).is_err()
        );
    }

    #[test]
    fn invalid_ouster_point_offset_is_rejected() {
        assert!(native_scan_to_cloud(scan(-0.001), &LidarDriverType::Ouster).is_err());
        assert!(native_scan_to_cloud(scan(5.0), &LidarDriverType::Ouster).is_err());
    }
}

#[cfg(any(
    feature = "ros2-client",
    feature = "r2r",
    feature = "hiroz",
    feature = "mt_pubsub"
))]
pub fn sync_to_imu(msg: ImuMsg) -> Imu {
    Imu {
        header: Header {
            stamp: Time {
                sec: msg.stamp.sec,
                nanosec: msg.stamp.nanosec,
            },
            frame_id: msg.frame_id,
        },
        orientation: Quaternion {
            x: msg.orientation[1],
            y: msg.orientation[2],
            z: msg.orientation[3],
            w: msg.orientation[0],
        },
        #[cfg(any(feature = "ros2-client", feature = "hiroz"))]
        orientation_covariance: [
            msg.orientation_covariance[0],
            msg.orientation_covariance[1],
            msg.orientation_covariance[2],
            msg.orientation_covariance[3],
            msg.orientation_covariance[4],
            msg.orientation_covariance[5],
            msg.orientation_covariance[6],
            msg.orientation_covariance[7],
            msg.orientation_covariance[8],
        ],
        #[cfg(feature = "mt_pubsub")]
        orientation_covariance: [
            msg.orientation_covariance[0],
            msg.orientation_covariance[1],
            msg.orientation_covariance[2],
            msg.orientation_covariance[3],
            msg.orientation_covariance[4],
            msg.orientation_covariance[5],
            msg.orientation_covariance[6],
            msg.orientation_covariance[7],
            msg.orientation_covariance[8],
        ],
        #[cfg(feature = "r2r")]
        orientation_covariance: [
            msg.orientation_covariance[0],
            msg.orientation_covariance[1],
            msg.orientation_covariance[2],
            msg.orientation_covariance[3],
            msg.orientation_covariance[4],
            msg.orientation_covariance[5],
            msg.orientation_covariance[6],
            msg.orientation_covariance[7],
            msg.orientation_covariance[8],
        ]
        .to_vec(),
        angular_velocity: Vector3 {
            x: msg.angular_velocity[0],
            y: msg.angular_velocity[1],
            z: msg.angular_velocity[2],
        },
        #[cfg(any(feature = "ros2-client", feature = "hiroz"))]
        angular_velocity_covariance: [
            msg.angular_velocity_covariance[0],
            msg.angular_velocity_covariance[1],
            msg.angular_velocity_covariance[2],
            msg.angular_velocity_covariance[3],
            msg.angular_velocity_covariance[4],
            msg.angular_velocity_covariance[5],
            msg.angular_velocity_covariance[6],
            msg.angular_velocity_covariance[7],
            msg.angular_velocity_covariance[8],
        ],
        #[cfg(feature = "mt_pubsub")]
        angular_velocity_covariance: [
            msg.angular_velocity_covariance[0],
            msg.angular_velocity_covariance[1],
            msg.angular_velocity_covariance[2],
            msg.angular_velocity_covariance[3],
            msg.angular_velocity_covariance[4],
            msg.angular_velocity_covariance[5],
            msg.angular_velocity_covariance[6],
            msg.angular_velocity_covariance[7],
            msg.angular_velocity_covariance[8],
        ],
        #[cfg(feature = "r2r")]
        angular_velocity_covariance: [
            msg.angular_velocity_covariance[0],
            msg.angular_velocity_covariance[1],
            msg.angular_velocity_covariance[2],
            msg.angular_velocity_covariance[3],
            msg.angular_velocity_covariance[4],
            msg.angular_velocity_covariance[5],
            msg.angular_velocity_covariance[6],
            msg.angular_velocity_covariance[7],
            msg.angular_velocity_covariance[8],
        ]
        .to_vec(),
        linear_acceleration: Vector3 {
            x: msg.linear_acceleration[0],
            y: msg.linear_acceleration[1],
            z: msg.linear_acceleration[2],
        },
        #[cfg(any(feature = "ros2-client", feature = "hiroz"))]
        linear_acceleration_covariance: [
            msg.linear_acceleration_covariance[0],
            msg.linear_acceleration_covariance[1],
            msg.linear_acceleration_covariance[2],
            msg.linear_acceleration_covariance[3],
            msg.linear_acceleration_covariance[4],
            msg.linear_acceleration_covariance[5],
            msg.linear_acceleration_covariance[6],
            msg.linear_acceleration_covariance[7],
            msg.linear_acceleration_covariance[8],
        ],
        #[cfg(feature = "mt_pubsub")]
        linear_acceleration_covariance: [
            msg.linear_acceleration_covariance[0],
            msg.linear_acceleration_covariance[1],
            msg.linear_acceleration_covariance[2],
            msg.linear_acceleration_covariance[3],
            msg.linear_acceleration_covariance[4],
            msg.linear_acceleration_covariance[5],
            msg.linear_acceleration_covariance[6],
            msg.linear_acceleration_covariance[7],
            msg.linear_acceleration_covariance[8],
        ],
        #[cfg(feature = "r2r")]
        linear_acceleration_covariance: [
            msg.linear_acceleration_covariance[0],
            msg.linear_acceleration_covariance[1],
            msg.linear_acceleration_covariance[2],
            msg.linear_acceleration_covariance[3],
            msg.linear_acceleration_covariance[4],
            msg.linear_acceleration_covariance[5],
            msg.linear_acceleration_covariance[6],
            msg.linear_acceleration_covariance[7],
            msg.linear_acceleration_covariance[8],
        ]
        .to_vec(),
    }
}

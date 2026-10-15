//! The running half of `pelorus online raw`: drivers in, publishers and a bag
//! out, and nothing in between.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use log::{error, info, warn};

use pelorus_core::{ImuMsg, LidarDriverType, LioConfig, PointCloudMode};
use pelorus_syncio::pelorus_drivers::{
    DriverError, DriverManager, RawLidarScan, ReceiveMetadata, SensorIngress,
};
use ros2_interfaces_jazzy_rkyv::sensor_msgs::msg::{Imu, PointCloud2};

use super::guide::Guide;
use super::{RawOptions, RawPlan, RawSensor, plan_raw_run};
use crate::mcap_output::{
    IMU_MSG, McapWriter, POINTCLOUD2_MSG, TopicId, cloud_msg_to_ros, imu_msg_to_ros, stamp_to_nanos,
};
use crate::output_target::{normalize_mcap_topic, prepare_output_targets, rosbag_mcap_path};

/// Queue depth between the driver threads and the sink thread.
///
/// Driver threads use bounded non-blocking queues to protect socket and serial
/// input. Full queues drop and count the newest sample.
const LIDAR_QUEUE_DEPTH: usize = 8;
const IMU_QUEUE_DEPTH: usize = 512;

/// How often the raw process reports what it has been doing.
const REPORT_INTERVAL: Duration = Duration::from_secs(5);

/// One measurement on its way from a driver thread to the sink thread.
///
/// ROS conversion runs on the sink thread to keep driver reads responsive.
enum RawSample {
    Lidar(Box<RawLidarScan>),
    Imu(Box<ImuMsg>),
}

#[derive(Default)]
struct RawCounters {
    lidar_published: AtomicU64,
    imu_published: AtomicU64,
    lidar_dropped: AtomicU64,
    imu_dropped: AtomicU64,
    packets_received: AtomicU64,
    packets_dropped: AtomicU64,
}

/// The driver-facing end of the raw pipeline.
///
/// Passes raw samples to the sink without odometry preprocessing.
struct RawIngress {
    tx: SyncSender<RawSample>,
    counters: Arc<RawCounters>,
    stop: Arc<AtomicBool>,
}

impl RawIngress {
    fn offer(&self, sample: RawSample, dropped: &AtomicU64) -> Result<(), DriverError> {
        match self.tx.try_send(sample) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => {
                dropped.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(TrySendError::Disconnected(_)) => {
                self.stop.store(true, Ordering::Release);
                Err(DriverError::Driver("raw output pipeline closed".into()))
            }
        }
    }
}

impl SensorIngress for RawIngress {
    fn push_lidar(&self, scan: RawLidarScan) -> Result<(), DriverError> {
        self.record_receive(&scan.receive);
        self.offer(
            RawSample::Lidar(Box::new(scan)),
            &self.counters.lidar_dropped,
        )
    }

    fn push_imu(&self, imu: ImuMsg) -> Result<(), DriverError> {
        self.offer(RawSample::Imu(Box::new(imu)), &self.counters.imu_dropped)
    }

    fn driver_failed(&self, error: &DriverError) {
        error!("Native sensor driver failed. Stopping raw output: {error}");
        self.stop.store(true, Ordering::Release);
    }
}

impl RawIngress {
    /// Stores the driver's cumulative packet counters.
    fn record_receive(&self, receive: &ReceiveMetadata) {
        self.counters
            .packets_received
            .store(receive.packets_received, Ordering::Relaxed);
        self.counters
            .packets_dropped
            .store(receive.packets_dropped, Ordering::Relaxed);
    }
}

/// MCAP destination for a raw run, if one was requested.
struct RawRecording {
    writer: McapWriter<std::io::BufWriter<std::fs::File>>,
    cloud: Option<TopicId>,
    imu: Option<TopicId>,
}

impl RawRecording {
    fn open(
        mcap_path: &Path,
        metadata_path: Option<PathBuf>,
        plan: &RawPlan,
        cloud_topic: &str,
        imu_topic: &str,
    ) -> Result<Self> {
        let mut writer = McapWriter::create(mcap_path, metadata_path)?;
        // Only the sensors this run actually drives get a channel: an IMU-only
        // bag that advertises an empty point-cloud topic would misrepresent
        // what was recorded.
        let cloud = plan
            .lidar
            .then(|| writer.add_topic(cloud_topic, POINTCLOUD2_MSG))
            .transpose()?;
        let imu = plan
            .imu
            .then(|| writer.add_topic(imu_topic, IMU_MSG))
            .transpose()?;
        Ok(Self { writer, cloud, imu })
    }
}

struct Sink<'a> {
    rt: &'a tokio::runtime::Runtime,
    cloud_publisher: Option<mt_pubsub::Publisher<PointCloud2>>,
    #[cfg(feature = "compressed-pointcloud")]
    compressed_publisher: Option<mt_pubsub::Publisher<pelorus_protocol::CompressedPointCloud2>>,
    imu_publisher: Option<mt_pubsub::Publisher<Imu>>,
    lidar_driver_type: LidarDriverType,
    counters: Arc<RawCounters>,
}

/// Runs `pelorus online raw` until the shutdown flag is set or a driver fails.
pub fn run(config: LioConfig, options: RawOptions, shutdown: Arc<AtomicBool>) -> Result<()> {
    let plan = plan_raw_run(&config, options.sensor)?;

    let mcap_path = options.output_mcap.clone().or_else(|| {
        options
            .output_rosbag
            .as_ref()
            .map(|path| rosbag_mcap_path(path))
    });
    // Metadata describes an MCAP file, so it only exists alongside one.
    let metadata_path = mcap_path.as_ref().and_then(|_| {
        options.output_rosbag_metadata.clone().or_else(|| {
            options
                .output_rosbag
                .as_ref()
                .map(|path| path.join("metadata.yaml"))
        })
    });
    let bag_cloud_topic = normalize_mcap_topic(
        options
            .mcap_cloud_topic
            .clone()
            .unwrap_or_else(|| plan.cloud_topic.clone()),
    );
    let bag_imu_topic = normalize_mcap_topic(
        options
            .mcap_imu_topic
            .clone()
            .unwrap_or_else(|| plan.imu_topic.clone()),
    );

    if !prepare_output_targets(
        options.output_rosbag.as_deref(),
        mcap_path.as_deref(),
        metadata_path.as_deref(),
        options.yes,
    )? {
        return Ok(());
    }

    // Raw output uses the encoding expected by the configured input subscriber.
    let compressed = matches!(plan.cloud_mode, PointCloudMode::Compressed);
    #[cfg(not(feature = "compressed-pointcloud"))]
    if compressed {
        anyhow::bail!(
            "`lidar.subscriber_mode = Compressed` requires the 'compressed-pointcloud' feature."
        );
    }

    let recording = match mcap_path.as_deref() {
        Some(path) => Some(RawRecording::open(
            path,
            metadata_path.clone(),
            &plan,
            &bag_cloud_topic,
            &bag_imu_topic,
        )?),
        None => None,
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("Failed to build the async runtime for raw output")?;

    let node = Arc::new(
        rt.block_on(async {
            let node_config = mt_pubsub::NodeConfig::new("pelorus_raw_node")
                .mode(mt_pubsub::Qos::Reliable)
                .coord_mode(options.coord_mode.into())
                .local_only(options.local_only);
            mt_pubsub::Node::create(node_config).await
        })
        .map_err(|err| anyhow!("Failed to create mt_pubsub node: {err}"))?,
    );

    let cloud_qos = pelorus_syncio::qos_to_mt(config.sync_node_params.cloud_qos);
    let imu_qos = pelorus_syncio::qos_to_mt(config.sync_node_params.imu_qos);

    let cloud_publisher = if plan.lidar && !compressed {
        Some(
            rt.block_on(node.create_publisher::<PointCloud2>(plan.cloud_topic.clone(), cloud_qos))?,
        )
    } else {
        None
    };
    #[cfg(feature = "compressed-pointcloud")]
    let compressed_publisher = if plan.lidar && compressed {
        // A subscriber is typed, so the compressed stream lives on its own
        // topic — the same one `lidar.subscriber_mode = Compressed` listens on.
        Some(rt.block_on(
            node.create_publisher::<pelorus_protocol::CompressedPointCloud2>(
                pelorus_protocol::compressed_cloud_topic(&plan.cloud_topic),
                cloud_qos,
            ),
        )?)
    } else {
        None
    };
    let imu_publisher = if plan.imu {
        Some(rt.block_on(node.create_publisher::<Imu>(plan.imu_topic.clone(), imu_qos))?)
    } else {
        None
    };

    if plan.lidar {
        info!(
            "Raw lidar driver {:?} publishing on '{}'{}",
            plan.drivers.lidar.driver_type,
            plan.cloud_topic,
            if compressed { " (compressed)" } else { "" }
        );
    }
    if plan.imu {
        info!(
            "Raw IMU driver {:?} publishing unfiltered samples on '{}'",
            plan.drivers.imu.driver_type, plan.imu_topic
        );
    }

    let (tx, rx) = sync_channel(if plan.lidar {
        LIDAR_QUEUE_DEPTH + IMU_QUEUE_DEPTH
    } else {
        IMU_QUEUE_DEPTH
    });
    let counters = Arc::new(RawCounters::default());
    let ingress = Arc::new(RawIngress {
        tx,
        counters: Arc::clone(&counters),
        stop: Arc::clone(&shutdown),
    });

    let drivers = DriverManager::start(&plan.drivers, ingress, None)
        .map_err(|error| anyhow!("Failed to start native sensor drivers: {error}"))?;

    let sink = Sink {
        rt: &rt,
        cloud_publisher,
        #[cfg(feature = "compressed-pointcloud")]
        compressed_publisher,
        imu_publisher,
        lidar_driver_type: plan.drivers.lidar.driver_type.clone(),
        counters: Arc::clone(&counters),
    };

    let guide = options
        .intrinsics
        .map(|intrinsics| Guide::new(intrinsics.poses));

    let result = drain(rx, &sink, recording, &shutdown, options.sensor, guide);

    // Stop the drivers before reporting the result, so a failure surfacing
    // during teardown is not attributed to the sink.
    let driver_result = drivers.shutdown();

    // Stop transport tasks before shutting down their runtime.
    drop(sink);
    drop(node);
    rt.shutdown_background();

    // Report the sink error first when both the sink and a driver failed.
    result?;
    driver_result.map_err(|error| anyhow!("Native sensor driver failed: {error}"))?;

    // The bag is closed by now, so dedrunk can read it.
    if options.intrinsics.is_some() {
        super::dedrunk::offer_fit(
            options.output_rosbag.as_deref().or(mcap_path.as_deref()),
            options.yes,
        )?;
    }
    Ok(())
}

/// Publishes and records until shutdown, then flushes whatever is still queued.
fn drain(
    rx: Receiver<RawSample>,
    sink: &Sink<'_>,
    mut recording: Option<RawRecording>,
    shutdown: &AtomicBool,
    sensor: RawSensor,
    mut guide: Option<Guide>,
) -> Result<()> {
    let mut last_report = Instant::now();
    while !shutdown.load(Ordering::Acquire) {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(sample) => {
                if handle_sample(sample, sink, recording.as_mut(), guide.as_mut())? {
                    break;
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
        if last_report.elapsed() >= REPORT_INTERVAL {
            report(&sink.counters, sensor);
            last_report = Instant::now();
        }
    }

    // Whatever the drivers already handed over belongs in the bag. Truncating
    // at the moment of Ctrl-C would lose samples that were merely in flight.
    // The drivers are still running here, so this drains what is queued rather
    // than waiting for the stream to end.
    while let Ok(sample) = rx.try_recv() {
        handle_sample(sample, sink, recording.as_mut(), guide.as_mut())?;
    }

    report(&sink.counters, sensor);
    if let Some(recording) = recording {
        recording
            .writer
            .finish()
            .context("Failed to finalize the raw MCAP output")?;
        info!("Raw recording finalized.");
    }
    Ok(())
}

/// Returns true when the guided recording has captured every orientation.
fn handle_sample(
    sample: RawSample,
    sink: &Sink<'_>,
    recording: Option<&mut RawRecording>,
    guide: Option<&mut Guide>,
) -> Result<bool> {
    match sample {
        RawSample::Lidar(scan) => {
            let cloud = pelorus_syncio::native_scan_to_cloud(*scan, &sink.lidar_driver_type)
                .map_err(|error| anyhow!("Failed to convert a raw scan: {error}"))?;

            if let Some(recording) = recording
                && let Some(topic) = recording.cloud
            {
                let msg = cloud_msg_to_ros(cloud.clone());
                let time = stamp_to_nanos(&msg.header.stamp);
                recording.writer.write(topic, &msg, time)?;
            }

            #[cfg(feature = "compressed-pointcloud")]
            if let Some(publisher) = sink.compressed_publisher.as_ref() {
                let compressed = pelorus_syncio::compress_cloud(cloud)
                    .map_err(|error| anyhow!("Failed to compress a raw scan: {error}"))?;
                sink.rt.block_on(publisher.publish(&compressed))?;
                sink.counters
                    .lidar_published
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(false);
            }

            if let Some(publisher) = sink.cloud_publisher.as_ref() {
                let msg = pelorus_syncio::impl_ros2_interfaces_jazzy_rkyv::from_pointcloud2_msg(cloud);
                sink.rt.block_on(publisher.publish(&msg))?;
                sink.counters
                    .lidar_published
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        RawSample::Imu(imu) => {
            if let Some(recording) = recording
                && let Some(topic) = recording.imu
            {
                let msg = imu_msg_to_ros(&imu);
                let time = stamp_to_nanos(&msg.header.stamp);
                recording.writer.write(topic, &msg, time)?;
            }

            // Prompt from the same samples that go into the bag, so what the
            // operator is told matches what will be fitted.
            let mut finished = false;
            if let Some(guide) = guide {
                let t = imu.stamp.sec as f64 + imu.stamp.nanosec as f64 * 1e-9;
                finished = guide.push(t, imu.linear_acceleration, imu.angular_velocity);
            }

            if let Some(publisher) = sink.imu_publisher.as_ref() {
                let msg = pelorus_syncio::sync_to_imu(*imu);
                sink.rt.block_on(publisher.publish(&msg))?;
                sink.counters.imu_published.fetch_add(1, Ordering::Relaxed);
            }

            return Ok(finished);
        }
    }
    Ok(false)
}

fn report(counters: &RawCounters, sensor: RawSensor) {
    let lidar = counters.lidar_published.load(Ordering::Relaxed);
    let imu = counters.imu_published.load(Ordering::Relaxed);
    let lidar_dropped = counters.lidar_dropped.load(Ordering::Relaxed);
    let imu_dropped = counters.imu_dropped.load(Ordering::Relaxed);
    let packets_dropped = counters.packets_dropped.load(Ordering::Relaxed);

    info!(
        "Raw {}: published {lidar} scans and {imu} IMU samples.",
        sensor.label()
    );
    if lidar_dropped > 0 || imu_dropped > 0 {
        warn!(
            "Raw output could not keep up: dropped {lidar_dropped} scans and {imu_dropped} IMU samples."
        );
    }
    if packets_dropped > 0 {
        warn!(
            "Sensor reported {packets_dropped} dropped packets out of {} received.",
            counters.packets_received.load(Ordering::Relaxed)
        );
    }
}

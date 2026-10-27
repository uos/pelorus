//! High-level Rust Interface for Pelorus Lidar-Inertial Odometry.
//!
//! [`Handle`] runs synchronization and registration on background threads and
//! needs the `threads` feature. [`Odometer`] runs every step on the calling
//! thread and is always available.
//!
//! # Example
//!
//! ```rust,no_run
//! use pelorus::Odometer;
//!
//! let mut odometer = Odometer::new(None).unwrap();
//!
//! // Feed sensor data...
//! // odometer.add_imu(...);
//! // odometer.add_points_xyzit(sec, nanosec, &points);
//!
//! while let Some(scan) = odometer.next_scan() {
//!     println!("position: {:?}, {} points", scan.position(), scan.points().len());
//! }
//! ```

#![warn(clippy::std_instead_of_alloc, clippy::std_instead_of_core)]

extern crate alloc;

#[cfg(feature = "threads")]
use alloc::sync::Arc;
#[cfg(feature = "threads")]
use core::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "rayon")]
use std::sync::OnceLock;
#[cfg(feature = "threads")]
use std::sync::{Mutex, mpsc};
#[cfg(feature = "threads")]
use std::thread;

use anyhow::anyhow;
use log::error;
pub use pelorus_core::PointXYZIT;
#[cfg(feature = "threads")]
use pelorus_core::SyncProcessedCloud;
use pelorus_core::config::{ast_to_config, layer_overrides};
use pelorus_core::{ImuMsg, LidarVendor, LioConfig, OdomMsg};
use pelorus_reg::lio::{LioCore, LioScan, LioStep};
use pelorus_reg::pipeline::Pipeline;
use pelorus_sync::SteppedIngress;
#[cfg(feature = "threads")]
use pelorus_sync::{ScanAdmission, Synchronizer};
pub use ros_pointcloud2::PointCloud2Msg;
use ros_pointcloud2::prelude::*;
use ros2_interfaces_jazzy_serde::sensor_msgs::msg::{PointCloud2, PointField};

const DEFAULT_CONFIG_STR: &str = pelorus_core::DEFAULT_CONFIG_STR;

impl_pointcloud2_for_ros2_interfaces_jazzy_serde!();

/// Errors returned by [`Handle`] operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no data available yet")]
    NoData,
    #[error("pipeline channel disconnected")]
    Disconnected,
    #[error("cannot poll while a callback is registered. Call clear_callback() first")]
    CallbackActive,
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// A field descriptor for raw [`Handle::add_pointcloud_raw`] input.
pub struct Pcloud2Field {
    pub name: String,
    pub offset: u32,
    pub datatype: u8,
}

/// A PointCloud2-style cloud for [`Handle::add_pointcloud_raw`].
pub struct RawPointCloud<'a> {
    pub header_sec: i32,
    pub header_nanosec: u32,
    pub width: u32,
    pub fields: &'a [Pcloud2Field],
    pub point_step: u32,
    pub row_step: u32,
    pub data: &'a [u8],
    pub is_dense: bool,
}

/// An IMU measurement for [`Handle::add_imu`].
///
/// All covariance arrays are row-major.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuMeasurement {
    pub timestamp_sec: i32,
    pub timestamp_nanosec: u32,
    /// Orientation quaternion `[w, x, y, z]`.
    pub orientation: [f64; 4],
    pub orientation_covariance: [f64; 9],
    pub angular_velocity: [f64; 3],
    pub angular_velocity_covariance: [f64; 9],
    pub linear_acceleration: [f64; 3],
    pub linear_acceleration_covariance: [f64; 9],
}

/// A single output point with XYZ coordinates and intensity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointXyzi {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
}

/// Odometry result produced for each registered scan.
///
/// # Frame conventions
/// - **World frame (W)**: fixed global reference, Z-up, aligned with IMU at init.
/// - **IMU frame (I)**: body-fixed frame attached to the IMU sensor.
/// - **Lidar frame (L)**: sensor-fixed frame attached to the Lidar.
#[derive(Debug, Clone)]
pub struct Odometry {
    pub timestamp_sec: i32,
    pub timestamp_nanosec: u32,
    /// Position `[x, y, z]` in world frame (metres).
    pub position: [f64; 3],
    /// Orientation quaternion `[w, x, y, z]`.
    pub orientation: [f64; 4],
    /// 6×6 pose covariance (row-major).
    pub pose_covariance: [f64; 36],
    /// Linear velocity `[vx, vy, vz]` in body frame (m/s).
    pub linear_velocity: [f64; 3],
    /// Angular velocity `[wx, wy, wz]` in body frame (rad/s).
    pub angular_velocity: [f64; 3],
    /// Accelerometer bias `[bx, by, bz]` in IMU frame (m/s²).
    pub accel_bias: [f64; 3],
    /// Gyroscope bias `[bx, by, bz]` in IMU frame (rad/s).
    pub gyro_bias: [f64; 3],
    /// 6×6 twist covariance (row-major).
    pub twist_covariance: [f64; 36],
    /// Scan ID matching the input point cloud.
    pub scan_id: u64,
    /// Registered point cloud in the lidar-world frame.
    pub points: Vec<PointXyzi>,
    /// Frame whose pose is expressed above (base_link or lidar frame name).
    pub frame_id: String,
    /// Processing latency from processed cloud input to odometry output, in nanoseconds.
    #[cfg(feature = "processing-time-export")]
    pub processing_time_ns: u128,
}

/// A result at the origin with identity orientation, zero velocity, biases and
/// covariances, and no points.
impl Default for Odometry {
    fn default() -> Self {
        Odometry {
            timestamp_sec: 0,
            timestamp_nanosec: 0,
            position: [0.0; 3],
            orientation: [1.0, 0.0, 0.0, 0.0],
            pose_covariance: [0.0; 36],
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            accel_bias: [0.0; 3],
            gyro_bias: [0.0; 3],
            twist_covariance: [0.0; 36],
            scan_id: 0,
            points: Vec::new(),
            frame_id: String::new(),
            #[cfg(feature = "processing-time-export")]
            processing_time_ns: 0,
        }
    }
}

#[cfg(feature = "threads")]
type OdometryCallback = Arc<dyn Fn(Odometry) + Send + Sync>;
#[cfg(feature = "threads")]
type SharedOdometryCallback = Arc<Mutex<Option<OdometryCallback>>>;

impl From<&Scan<'_>> for Odometry {
    fn from(scan: &Scan<'_>) -> Self {
        let (timestamp_sec, timestamp_nanosec) = scan.timestamp();
        let (linear_velocity, angular_velocity) = scan.twist();
        Odometry {
            timestamp_sec,
            timestamp_nanosec,
            position: scan.position(),
            orientation: scan.orientation(),
            pose_covariance: *scan.pose_covariance(),
            linear_velocity,
            angular_velocity,
            accel_bias: scan.accel_bias(),
            gyro_bias: scan.gyro_bias(),
            twist_covariance: *scan.twist_covariance(),
            scan_id: scan.scan_id(),
            points: scan.points_xyzi().collect(),
            frame_id: scan.frame_id().to_string(),
            #[cfg(feature = "processing-time-export")]
            processing_time_ns: scan.processing_time_ns(),
        }
    }
}

fn build_cloud_msg(
    sec: i32,
    nanosec: u32,
    num_points: usize,
    mut point_at: impl FnMut(usize) -> (f32, f32, f32, f32, u32),
) -> PointCloud2Msg {
    let point_step: usize = 20;
    let mut data = Vec::with_capacity(num_points * point_step);
    for i in 0..num_points {
        let (x, y, z, intensity, t) = point_at(i);
        data.extend_from_slice(&x.to_le_bytes());
        data.extend_from_slice(&y.to_le_bytes());
        data.extend_from_slice(&z.to_le_bytes());
        data.extend_from_slice(&intensity.to_le_bytes());
        data.extend_from_slice(&t.to_le_bytes());
    }

    let cloud = PointCloud2 {
        header: ros2_interfaces_jazzy_serde::std_msgs::msg::Header {
            stamp: ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time { sec, nanosec },
            frame_id: String::new(),
        },
        height: 1,
        width: num_points as u32,
        fields: vec![
            PointField {
                name: "x".to_string(),
                offset: 0,
                datatype: 7,
                count: 1,
            },
            PointField {
                name: "y".to_string(),
                offset: 4,
                datatype: 7,
                count: 1,
            },
            PointField {
                name: "z".to_string(),
                offset: 8,
                datatype: 7,
                count: 1,
            },
            PointField {
                name: "intensity".to_string(),
                offset: 12,
                datatype: 7,
                count: 1,
            },
            PointField {
                name: "t".to_string(),
                offset: 16,
                datatype: 6,
                count: 1,
            },
        ],
        is_bigendian: false,
        point_step: point_step as u32,
        row_step: (num_points * point_step) as u32,
        data,
        is_dense: true,
    };

    impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(cloud)
}

fn imu_msg(imu: ImuMeasurement) -> ImuMsg {
    let ImuMeasurement {
        timestamp_sec,
        timestamp_nanosec,
        orientation,
        orientation_covariance,
        angular_velocity,
        angular_velocity_covariance,
        linear_acceleration,
        linear_acceleration_covariance,
    } = imu;
    ImuMsg {
        stamp: ros_pointcloud2::ros::TimeMsg {
            sec: timestamp_sec,
            nanosec: timestamp_nanosec,
        },
        frame_id: String::new(),
        orientation,
        orientation_covariance,
        angular_velocity,
        angular_velocity_covariance,
        linear_acceleration,
        linear_acceleration_covariance,
    }
}

fn odom_msg(
    timestamp_sec: i32,
    timestamp_nanosec: u32,
    twist_linear: [f64; 3],
    twist_angular: [f64; 3],
    twist_covariance: [f64; 36],
) -> OdomMsg {
    OdomMsg {
        twist_linear: nalgebra::Vector3::new(twist_linear[0], twist_linear[1], twist_linear[2]),
        twist_angular: nalgebra::Vector3::new(twist_angular[0], twist_angular[1], twist_angular[2]),
        twist_cov: twist_covariance,
        // Programmatic inputs are interpreted in the IMU frame. ROS users can
        // configure the wheel-to-IMU rotation on the synchronized input path.
        twist_frame_to_imu: nalgebra::Matrix3::identity(),
        stamp: ros_pointcloud2::ros::TimeMsg {
            sec: timestamp_sec,
            nanosec: timestamp_nanosec,
        },
    }
}

fn xyzit_cloud_msg(sec: i32, nanosec: u32, points: &[[f32; 5]]) -> PointCloud2Msg {
    build_cloud_msg(sec, nanosec, points.len(), |i| {
        let p = points[i];
        (p[0], p[1], p[2], p[3], p[4] as u32)
    })
}

fn xyzt_cloud_msg(sec: i32, nanosec: u32, points: &[[f32; 4]]) -> PointCloud2Msg {
    build_cloud_msg(sec, nanosec, points.len(), |i| {
        let p = points[i];
        (p[0], p[1], p[2], 0.0f32, p[3] as u32)
    })
}

fn raw_cloud_msg(cloud: RawPointCloud<'_>) -> PointCloud2Msg {
    let RawPointCloud {
        header_sec,
        header_nanosec,
        width,
        fields,
        point_step,
        row_step,
        data,
        is_dense,
    } = cloud;
    let ros_fields: Vec<PointField> = fields
        .iter()
        .map(|f| PointField {
            name: f.name.clone(),
            offset: f.offset,
            datatype: f.datatype,
            count: 1,
        })
        .collect();

    let cloud = PointCloud2 {
        header: ros2_interfaces_jazzy_serde::std_msgs::msg::Header {
            stamp: ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time {
                sec: header_sec,
                nanosec: header_nanosec,
            },
            frame_id: String::new(),
        },
        height: 1,
        width,
        fields: ros_fields,
        is_bigendian: false,
        point_step,
        row_step,
        data: data.to_vec(),
        is_dense,
    };

    impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(cloud)
}

fn prepare_config(
    mut config: LioConfig,
    device_specific_pointcloud2: bool,
) -> (LioConfig, Pipeline) {
    configure_thread_pool(config.num_threads);
    if !device_specific_pointcloud2 {
        config.sync_params.lidar_vendor = LidarVendor::Ouster;
    }
    let pipeline = Pipeline::new(&config);
    (config, pipeline)
}

fn is_config_url(location: &str) -> bool {
    location.starts_with("http://") || location.starts_with("https://")
}

fn read_config_source(location: &str) -> anyhow::Result<String> {
    #[cfg(not(feature = "http-config"))]
    if is_config_url(location) {
        anyhow::bail!(
            "Loading configuration '{location}' from a URL requires the http-config feature"
        );
    }
    #[cfg(feature = "http-config")]
    if is_config_url(location) {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(core::time::Duration::from_secs(30)))
            .build()
            .into();
        return agent
            .get(location)
            .call()
            .map_err(|error| anyhow!("Failed to fetch configuration '{location}': {error}"))?
            .body_mut()
            .read_to_string()
            .map_err(|error| anyhow!("Failed to read configuration '{location}': {error}"));
    }
    std::fs::read_to_string(location)
        .map_err(|error| anyhow!("Failed to read configuration '{location}': {error}"))
}

fn load_config(config_path: Option<&str>) -> anyhow::Result<LioConfig> {
    let source = config_path
        .map(read_config_source)
        .transpose()?
        .unwrap_or_default();
    let parent = config_path
        .filter(|location| !is_config_url(location))
        .and_then(|location| std::path::Path::new(location).parent())
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    compile_config(&source, parent).map_err(|error| match config_path {
        Some(location) => anyhow!("Failed to compile user config '{location}': {error:#}"),
        None => anyhow!("Failed to compile empty user config: {error:#}"),
    })
}

/// Layer `.rl` source over the embedded defaults. Relative includes resolve
/// from `parent`.
fn compile_config(source: &str, parent: &std::path::Path) -> anyhow::Result<LioConfig> {
    let default_ast = ratslang::compile_code(DEFAULT_CONFIG_STR)
        .map_err(|e| anyhow!("Failed to compile embedded default config: {e}"))?;
    let user_ast = layer_overrides(source, parent, &[])?;
    ast_to_config(default_ast, user_ast).map_err(|e| anyhow!("Failed to build configuration: {e}"))
}

/// The calling side of a [`Handle`]: which clouds the odometry takes, and the
/// synchronizer whose threads preprocess and synchronize them.
#[cfg(feature = "threads")]
struct Intake {
    admission: ScanAdmission,
    sync: Synchronizer,
}

/// Register every synchronized scan with the [`LioCore`], as the pipelined
/// driver of `pelorus offline` does. Each scan releases the synchronizer to the
/// next one once the map is updated.
#[cfg(feature = "threads")]
fn processing_main(
    mut core: LioCore,
    synced: mpsc::Receiver<SyncProcessedCloud>,
    ready_for_next: mpsc::Sender<u64>,
    results: mpsc::Sender<Odometry>,
    callback: SharedOdometryCallback,
    shutdown: Arc<AtomicBool>,
) {
    while let Ok(processed) = synced.recv() {
        // Keep draining after shutdown so the synchronizer threads can finish.
        if shutdown.load(Ordering::Relaxed) {
            continue;
        }
        #[cfg(feature = "processing-time-export")]
        let started = pelorus_core::Instant::now();
        let scan = match core.step(processed, Some(&ready_for_next)) {
            LioStep::Registered(scan) => scan,
            LioStep::MapOnly { scan_id } => {
                let _ = ready_for_next.send(scan_id);
                continue;
            }
            LioStep::Failed { scan_id, error } => {
                error!("registration of scan #{scan_id} failed: {error:#}");
                let _ = ready_for_next.send(scan_id);
                continue;
            }
        };
        let odometry = Odometry::from(&Scan {
            scan: &scan,
            core: &core,
            #[cfg(feature = "processing-time-export")]
            processing_time_ns: started.elapsed().as_nanos(),
        });
        let registered = callback
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(Arc::clone));
        match registered {
            Some(callback) => callback(odometry),
            None => {
                let scan_id = odometry.scan_id;
                if let Err(e) = results.send(odometry) {
                    error!("failed to send odometry of scan #{scan_id}: {e}");
                }
            }
        }
    }
    core.shutdown();
}

#[cfg(feature = "rayon")]
static RAYON_THREADS: OnceLock<usize> = OnceLock::new();

/// Configure the rayon global thread pool.
///
/// Called once at Handle creation. If a previous Handle already configured
/// the pool with a different count, a warning is logged and the request is
/// ignored (first Handle wins — rayon only allows one global init).
fn configure_thread_pool(num_threads: usize) {
    if num_threads == 0 {
        return; // 0 means "use all cores" — leave rayon's default alone
    }
    #[cfg(feature = "rayon")]
    {
        let prev = RAYON_THREADS.get_or_init(|| {
            if let Err(e) = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
            {
                error!("Failed to configure rayon thread pool to {num_threads} threads: {e}");
            }
            num_threads
        });
        if *prev != num_threads {
            log::warn!(
                "Rayon thread pool already configured with {} threads. \
             Ignoring request for {} threads (first Handle wins).",
                prev,
                num_threads
            );
        }
    }
    #[cfg(not(feature = "rayon"))]
    log::warn!(
        "Ignoring num_threads={num_threads}. Thread-pool configuration requires the rayon feature."
    );
}

// ---------- public API ----------

/// Handle for the Pelorus Lidar-Inertial Odometry pipeline.
///
/// Manages the full processing pipeline: IMU pre-integration, point-cloud
/// undistortion, scan registration, and state estimation.  Dropping the handle
/// shuts the pipeline down gracefully.
///
/// The stages run on their own threads, as in the pipelined driver of
/// `pelorus offline`: preprocessing, synchronization with the IMU, and
/// registration. The next scan is preprocessed while the current one registers.
/// No input is dropped. When the queues are full, the `add_*` calls block until
/// the threads catch up. The same input sequence therefore produces the same
/// results as the [`Odometer`] and as `pelorus offline`.
///
/// # Callback vs polling
///
/// Results can be consumed in two ways (mutually exclusive):
/// - Polling: [`recv`](Handle::recv) (blocking) or [`try_recv`](Handle::try_recv)
///   (non-blocking).
/// - Callback: register a closure with [`set_callback`](Handle::set_callback).
///   It is called from the processing thread for every result.  While a callback
///   is registered, `recv`/`try_recv` return [`Error::CallbackActive`].
#[cfg(feature = "threads")]
pub struct Handle {
    intake: Mutex<Option<Intake>>,
    /// Behind a lock so the handle can be shared between threads.
    result_receiver: Mutex<mpsc::Receiver<Odometry>>,
    callback: SharedOdometryCallback,
    processing: Option<thread::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
}

#[cfg(feature = "threads")]
impl Handle {
    /// Create a new handle using the embedded default configuration.
    ///
    /// Pass a local path or HTTP(S) URL to overlay a `.rl` config on the defaults.
    pub fn new(config_path: Option<&str>) -> Result<Self, Error> {
        Self::new_with_options(config_path, true)
    }

    /// Create a new handle with additional creation-time options.
    ///
    /// - `config_path`: optional local path or HTTP(S) URL to a `.rl` config.
    /// - `device_specific_pointcloud2`: when `true` (default) the embedded
    ///   vendor-specific PointCloud2 parsing is used.  When `false` a generic
    ///   layout is expected where the per-point `t` field is a header-relative
    ///   nanosecond offset.
    pub fn new_with_options(
        config_path: Option<&str>,
        device_specific_pointcloud2: bool,
    ) -> Result<Self, Error> {
        let (config, pipeline) =
            prepare_config(load_config(config_path)?, device_specific_pointcloud2);
        let admission = ScanAdmission::new(
            (!config.imu_params.lidar_only).then(|| pipeline.build_init_callback()),
            config.sync_node_params.imu_predictor_init_time_sec,
        );
        let (ready_tx, ready_rx) = mpsc::channel();
        let (sync, synced_rx) = pipeline.create_sync(Some(ready_rx))?;
        let (result_tx, result_rx) = mpsc::channel();
        let callback: SharedOdometryCallback = Arc::new(Mutex::new(None));
        let shutdown = Arc::new(AtomicBool::new(false));

        // Paired with the release in `Drop`, so process-wide GPU resources are torn
        // down when the last handle goes away rather than at `exit()`.
        pelorus_reg::backend_session_retain();

        let processing_callback = Arc::clone(&callback);
        let processing_shutdown = Arc::clone(&shutdown);
        let processing = thread::spawn(move || {
            processing_main(
                LioCore::new(config, pipeline),
                synced_rx,
                ready_tx,
                result_tx,
                processing_callback,
                processing_shutdown,
            );
        });

        Ok(Handle {
            intake: Mutex::new(Some(Intake { admission, sync })),
            result_receiver: Mutex::new(result_rx),
            callback,
            processing: Some(processing),
            shutdown,
        })
    }

    /// Run `f` on the open intake.
    fn with_intake<T>(&self, f: impl FnOnce(&mut Intake) -> T) -> Result<T, Error> {
        let mut intake = self.intake.lock().unwrap();
        let intake = intake.as_mut().ok_or_else(|| anyhow!("Handle is closed"))?;
        Ok(f(intake))
    }

    /// Pass a cloud on to synchronization. Returns its scan id, or `0` for a
    /// cloud the odometry drops: one before the IMU initialization, an empty
    /// one or one older than the previous cloud.
    fn send_cloud(&self, cloud: PointCloud2Msg) -> Result<u64, Error> {
        self.with_intake(|intake| {
            let Some(scan_id) = intake.admission.admit(&cloud) else {
                return 0;
            };
            let numbered = intake.sync.add_pointcloud(cloud);
            debug_assert_eq!(
                numbered, scan_id,
                "the synchronizer numbers exactly the clouds the admission takes"
            );
            scan_id
        })
    }

    /// Add an IMU measurement.
    pub fn add_imu(&self, imu: ImuMeasurement) -> Result<(), Error> {
        let imu = imu_msg(imu);
        self.with_intake(|intake| {
            intake.admission.observe_imu(&imu);
            intake.sync.add_imu(imu);
        })
    }

    /// Add a wheel-odometry measurement.
    pub fn add_odom(
        &self,
        timestamp_sec: i32,
        timestamp_nanosec: u32,
        twist_linear: [f64; 3],
        twist_angular: [f64; 3],
        twist_covariance: [f64; 36],
    ) -> Result<(), Error> {
        let odom = odom_msg(
            timestamp_sec,
            timestamp_nanosec,
            twist_linear,
            twist_angular,
            twist_covariance,
        );
        self.with_intake(|intake| intake.sync.add_odom(odom))
    }

    /// Add a point cloud as `[x, y, z, intensity, t]` f32 tuples.
    ///
    /// `t` is a per-point nanosecond offset relative to the scan header
    /// timestamp `(timestamp_sec, timestamp_nanosec)`.
    ///
    /// Returns the scan ID assigned to this cloud, or `0` for a dropped cloud.
    pub fn add_points_xyzit(
        &self,
        timestamp_sec: i32,
        timestamp_nanosec: u32,
        points: &[[f32; 5]],
    ) -> Result<u64, Error> {
        self.send_cloud(xyzit_cloud_msg(timestamp_sec, timestamp_nanosec, points))
    }

    /// Add a point cloud as `[x, y, z, t]` f32 tuples (intensity defaults to 0).
    ///
    /// `t` is a per-point nanosecond offset relative to the scan header
    /// timestamp.
    pub fn add_points_xyzt(
        &self,
        timestamp_sec: i32,
        timestamp_nanosec: u32,
        points: &[[f32; 4]],
    ) -> Result<u64, Error> {
        self.send_cloud(xyzt_cloud_msg(timestamp_sec, timestamp_nanosec, points))
    }

    /// Add a raw PointCloud2-formatted point cloud.
    ///
    /// Accepts the same data layout as a ROS2 `sensor_msgs/PointCloud2` message.
    /// For simpler input formats prefer [`add_points_xyzit`](Handle::add_points_xyzit)
    /// or [`add_points_xyzt`](Handle::add_points_xyzt).
    ///
    /// Returns the scan ID assigned to this cloud, or `0` for a dropped cloud.
    pub fn add_pointcloud_raw(&self, cloud: RawPointCloud<'_>) -> Result<u64, Error> {
        self.send_cloud(raw_cloud_msg(cloud))
    }

    /// Add a point cloud that is already a [`PointCloud2Msg`], as decoded from a
    /// bag or a subscription.
    ///
    /// Returns the scan ID assigned to this cloud, or `0` for a dropped cloud.
    pub fn add_pointcloud2(&self, cloud: PointCloud2Msg) -> Result<u64, Error> {
        self.send_cloud(cloud)
    }

    /// Non-blocking receive.
    ///
    /// Returns `Ok(None)` when no result is ready yet.
    /// Returns `Err(Error::CallbackActive)` when a callback is registered.
    pub fn try_recv(&self) -> Result<Option<Odometry>, Error> {
        if self.callback.lock().unwrap().is_some() {
            return Err(Error::CallbackActive);
        }

        match self.result_receiver.lock().unwrap().try_recv() {
            Ok(odometry) => Ok(Some(odometry)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => Err(Error::Disconnected),
        }
    }

    /// Blocking receive — waits until the next odometry result is ready.
    ///
    /// Returns `Err(Error::CallbackActive)` when a callback is registered.
    pub fn recv(&self) -> Result<Odometry, Error> {
        if self.callback.lock().unwrap().is_some() {
            return Err(Error::CallbackActive);
        }

        self.result_receiver
            .lock()
            .unwrap()
            .recv()
            .map_err(|_| Error::Disconnected)
    }

    /// Register an asynchronous callback invoked from the processing thread.
    ///
    /// While a callback is set, results are delivered directly to it and are
    /// **not** queued for [`recv`](Handle::recv) / [`try_recv`](Handle::try_recv).
    /// Any already-queued results are drained when the callback is installed.
    ///
    /// The callback must not call back into this [`Handle`] as the processing
    /// thread holds no locks during the call, but the callback itself runs on
    /// the processing thread so avoid blocking inside it.
    pub fn set_callback<F>(&mut self, f: F)
    where
        F: Fn(Odometry) + Send + Sync + 'static,
    {
        let mut guard = self.callback.lock().unwrap();
        // Drain any queued results while holding the lock to close the race
        // window where an output is enqueued after we think we've drained.
        let receiver = self.result_receiver.lock().unwrap();
        while receiver.try_recv().is_ok() {}
        *guard = Some(Arc::new(f));
    }

    /// Remove the current callback. Subsequent results are queued for polling.
    pub fn clear_callback(&mut self) {
        *self.callback.lock().unwrap() = None;
    }

    /// Return the library version string.
    pub fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}

#[cfg(feature = "threads")]
impl Drop for Handle {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);

        // Dropping the synchronizer closes its channels, which ends its threads
        // and then the processing thread.
        if let Ok(mut intake) = self.intake.lock() {
            intake.take();
        }
        if let Some(processing) = self.processing.take() {
            let _ = processing.join();
        }

        // After the processing thread is joined, so its GPU buffers are gone first.
        pelorus_reg::backend_session_release();
    }
}

/// A scan an [`Odometer`] registered, borrowed from it until the next
/// [`next_scan`](Odometer::next_scan).
///
/// Every accessor reads the registration result in place. Nothing is copied
/// until the caller copies it.
///
/// # Frame conventions
/// - **World frame (W)**: fixed global reference, Z-up, aligned with IMU at init.
/// - **Body frame**: the configured base link, or the Lidar frame without one.
pub struct Scan<'a> {
    scan: &'a LioScan,
    core: &'a LioCore,
    #[cfg(feature = "processing-time-export")]
    processing_time_ns: u128,
}

impl Scan<'_> {
    /// Scan ID matching the input point cloud.
    pub fn scan_id(&self) -> u64 {
        self.scan.scan_id
    }

    /// End of the scan as `(sec, nanosec)`.
    pub fn timestamp(&self) -> (i32, u32) {
        (self.scan.odom_time.sec, self.scan.odom_time.nanosec)
    }

    /// Position `[x, y, z]` of the body frame in world frame (metres).
    pub fn position(&self) -> [f64; 3] {
        self.scan.t_world.translation.vector.into()
    }

    /// Orientation quaternion `[w, x, y, z]` of the body frame.
    pub fn orientation(&self) -> [f64; 4] {
        let q = self.scan.t_world.rotation.quaternion();
        [q.w, q.i, q.j, q.k]
    }

    /// 6×6 pose covariance (row-major).
    pub fn pose_covariance(&self) -> &[f64; 36] {
        &self.scan.pose_covariance
    }

    /// Linear velocity `[vx, vy, vz]` and angular velocity `[wx, wy, wz]` in
    /// body frame (m/s, rad/s).
    pub fn twist(&self) -> ([f64; 3], [f64; 3]) {
        let (linear, angular) = self.scan.ekf_state.body_twist(&self.scan.t_world);
        (linear.into(), angular.into())
    }

    /// 6×6 twist covariance (row-major).
    pub fn twist_covariance(&self) -> &[f64; 36] {
        &self.scan.twist_covariance
    }

    /// Accelerometer bias `[bx, by, bz]` in IMU frame (m/s²).
    pub fn accel_bias(&self) -> [f64; 3] {
        self.scan.ekf_state.accel_bias.into()
    }

    /// Gyroscope bias `[bx, by, bz]` in IMU frame (rad/s).
    pub fn gyro_bias(&self) -> [f64; 3] {
        self.scan.ekf_state.gyro_bias.into()
    }

    /// Frame whose pose is reported: the base link, or the Lidar frame without one.
    pub fn frame_id(&self) -> &str {
        self.core
            .config()
            .base_link_frame
            .as_deref()
            .unwrap_or(&self.scan.cloud_header.frame_id)
    }

    /// The deskewed scan in the Lidar frame.
    pub fn points(&self) -> &[PointXYZIT] {
        self.scan
            .passthrough
            .as_deref()
            .unwrap_or(self.core.deskewed_cloud())
    }

    /// [`points`](Self::points) as [`PointXyzi`], converted while iterating.
    pub fn points_xyzi(&self) -> impl ExactSizeIterator<Item = PointXyzi> + '_ {
        self.points().iter().map(|p| PointXyzi {
            x: p.x,
            y: p.y,
            z: p.z,
            intensity: p.intensity,
        })
    }

    /// [`points`](Self::points) in the world frame, transformed with the
    /// registered Lidar pose while iterating.
    pub fn world_points(&self) -> impl ExactSizeIterator<Item = PointXyzi> + '_ {
        let lidar_pose = self.scan.register_pose.pose;
        self.points().iter().map(move |p| {
            let world =
                lidar_pose * nalgebra::Point3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z));
            PointXyzi {
                x: world.x as f32,
                y: world.y as f32,
                z: world.z as f32,
                intensity: p.intensity,
            }
        })
    }

    /// Time the registration of this scan took, in nanoseconds.
    #[cfg(feature = "processing-time-export")]
    pub fn processing_time_ns(&self) -> u128 {
        self.processing_time_ns
    }
}

/// Lidar-inertial odometry that runs every step on the calling thread.
///
/// The `add_*` calls preprocess and buffer their input. Each
/// [`next_scan`](Odometer::next_scan) synchronizes, deskews and registers the
/// next scan whose IMU interval is complete. No thread is started, so the same
/// input sequence produces the same output sequence on every run when the CPU
/// backend is used and the `os-rng-seed` feature is off.
///
/// Rayon still splits heavy loops across its global pool unless
/// `parallelism.execution_mode = SingleCore` or the `rayon` feature is off.
/// Every call returns only after that work finished.
pub struct Odometer {
    /// Taken on drop to release the backend.
    core: Option<LioCore>,
    ingress: SteppedIngress,
    /// The scan [`last_scan`](Odometer::last_scan) borrows.
    last_scan: Option<Box<LioScan>>,
    #[cfg(feature = "processing-time-export")]
    last_processing_time_ns: u128,
}

impl Odometer {
    /// Create an odometer using the embedded default configuration.
    ///
    /// Pass a local path or HTTP(S) URL to overlay a `.rl` config on the defaults.
    pub fn new(config_path: Option<&str>) -> Result<Self, Error> {
        Self::new_with_options(config_path, true)
    }

    /// Create an odometer with additional creation-time options. See
    /// [`Handle::new_with_options`].
    pub fn new_with_options(
        config_path: Option<&str>,
        device_specific_pointcloud2: bool,
    ) -> Result<Self, Error> {
        Self::with_config(load_config(config_path)?, device_specific_pointcloud2)
    }

    /// Create an odometer from `.rl` source text layered over the embedded
    /// defaults, for platforms without a file system such as the browser.
    /// Relative includes resolve from `.`. See [`Handle::new_with_options`] for
    /// `device_specific_pointcloud2`.
    pub fn from_config_source(
        source: &str,
        device_specific_pointcloud2: bool,
    ) -> Result<Self, Error> {
        let config = compile_config(source, std::path::Path::new("."))
            .map_err(|error| anyhow!("Failed to compile configuration source: {error:#}"))?;
        Self::with_config(config, device_specific_pointcloud2)
    }

    fn with_config(config: LioConfig, device_specific_pointcloud2: bool) -> Result<Self, Error> {
        let (config, pipeline) = prepare_config(config, device_specific_pointcloud2);
        let ingress = SteppedIngress::new(
            pipeline.create_stepped_sync()?,
            (!config.imu_params.lidar_only).then(|| pipeline.build_init_callback()),
            config.sync_node_params.imu_predictor_init_time_sec,
        );

        // Paired with the release in `Drop`.
        pelorus_reg::backend_session_retain();

        Ok(Odometer {
            core: Some(LioCore::new(config, pipeline)),
            ingress,
            last_scan: None,
            #[cfg(feature = "processing-time-export")]
            last_processing_time_ns: 0,
        })
    }

    /// Add an IMU measurement.
    pub fn add_imu(&mut self, imu: ImuMeasurement) {
        self.ingress.push_imu(imu_msg(imu));
    }

    /// Add a wheel-odometry measurement. See [`Handle::add_odom`].
    pub fn add_odom(
        &mut self,
        timestamp_sec: i32,
        timestamp_nanosec: u32,
        twist_linear: [f64; 3],
        twist_angular: [f64; 3],
        twist_covariance: [f64; 36],
    ) {
        self.ingress.push_odom(odom_msg(
            timestamp_sec,
            timestamp_nanosec,
            twist_linear,
            twist_angular,
            twist_covariance,
        ));
    }

    /// Add a point cloud as `[x, y, z, intensity, t]` f32 tuples. See
    /// [`Handle::add_points_xyzit`].
    ///
    /// Returns the scan ID assigned to this cloud, or `0` for a dropped cloud.
    pub fn add_points_xyzit(
        &mut self,
        timestamp_sec: i32,
        timestamp_nanosec: u32,
        points: &[[f32; 5]],
    ) -> u64 {
        self.add_cloud(xyzit_cloud_msg(timestamp_sec, timestamp_nanosec, points))
    }

    /// Add a point cloud as `[x, y, z, t]` f32 tuples. See
    /// [`Handle::add_points_xyzt`].
    pub fn add_points_xyzt(
        &mut self,
        timestamp_sec: i32,
        timestamp_nanosec: u32,
        points: &[[f32; 4]],
    ) -> u64 {
        self.add_cloud(xyzt_cloud_msg(timestamp_sec, timestamp_nanosec, points))
    }

    /// Add a raw PointCloud2-formatted point cloud. See
    /// [`Handle::add_pointcloud_raw`].
    pub fn add_pointcloud_raw(&mut self, cloud: RawPointCloud<'_>) -> u64 {
        self.add_cloud(raw_cloud_msg(cloud))
    }

    /// Add a point cloud that is already a [`PointCloud2Msg`], as decoded from a
    /// bag or a subscription. The point buffer moves into the odometer without
    /// a copy.
    pub fn add_pointcloud2(&mut self, cloud: PointCloud2Msg) -> u64 {
        self.add_cloud(cloud)
    }

    /// Empty clouds and clouds older than the previous one are dropped.
    fn add_cloud(&mut self, cloud_msg: PointCloud2Msg) -> u64 {
        self.ingress
            .push_cloud(Box::new(cloud_msg), pelorus_core::Instant::now())
            .unwrap_or(0)
    }

    /// The resolved configuration.
    pub fn config(&self) -> &LioConfig {
        self.core
            .as_ref()
            .expect("the core lives until drop")
            .config()
    }

    /// Number of scans waiting for IMU data.
    pub fn pending_scans(&self) -> usize {
        self.ingress.pending_scans()
    }

    /// Whether the IMU initialization ran, or none is needed.
    pub fn imu_initialized(&self) -> bool {
        self.ingress.imu_initialized()
    }

    /// Register the next scan whose IMU interval is complete. Returns `None`
    /// once no queued scan can be completed with the input added so far.
    ///
    /// Initialization scans only build the map and are passed over, as is a
    /// scan whose registration fails, which is logged.
    ///
    /// ```rust,no_run
    /// # let mut odometer = pelorus::Odometer::new(None).unwrap();
    /// while let Some(scan) = odometer.next_scan() {
    ///     println!("{:?}: {} points", scan.position(), scan.points().len());
    /// }
    /// ```
    pub fn next_scan(&mut self) -> Option<Scan<'_>> {
        pelorus_core::complete_synchronously(self.next_scan_inner(false))
    }

    /// Process a scan while allowing asynchronous WebGPU readbacks.
    /// Use this instead of [`next_scan`](Self::next_scan) after
    /// [`enable_webgpu`](Self::enable_webgpu). Without WebGPU both behave the same.
    #[cfg(feature = "webgpu")]
    pub async fn next_scan_async(&mut self) -> Option<Scan<'_>> {
        self.next_scan_inner(true).await
    }

    /// Enable experimental single- or multiresolution GPU correspondence search.
    /// Failure leaves the existing CPU path available.
    #[cfg(feature = "webgpu")]
    pub async fn enable_webgpu(&mut self) -> Result<(), Error> {
        self.core
            .as_mut()
            .expect("the core lives until drop")
            .enable_webgpu()
            .await?;
        Ok(())
    }

    #[cfg(feature = "webgpu")]
    pub fn webgpu_status(&self) -> Result<&str, String> {
        self.core
            .as_ref()
            .expect("the core lives until drop")
            .webgpu_status()
    }

    async fn next_scan_inner(&mut self, asynchronous: bool) -> Option<Scan<'_>> {
        // The next step overwrites the deskewed cloud the last scan points into.
        self.last_scan = None;
        let core = self.core.as_mut().expect("the core lives until drop");
        loop {
            let processed = self.ingress.next_synced()?;
            #[cfg(feature = "processing-time-export")]
            let started = pelorus_core::Instant::now();
            let step = if asynchronous {
                core.step_async(processed, None).await
            } else {
                core.step(processed, None)
            };
            match step {
                LioStep::Registered(scan) => {
                    #[cfg(feature = "processing-time-export")]
                    {
                        self.last_processing_time_ns = started.elapsed().as_nanos();
                    }
                    self.last_scan = Some(scan);
                    return self.last_scan();
                }
                LioStep::MapOnly { .. } => {}
                LioStep::Failed { scan_id, error } => {
                    error!("registration of scan #{scan_id} failed: {error:#}");
                }
            }
        }
    }

    /// The scan the last [`next_scan`](Self::next_scan) returned, until the
    /// next call to it. `None` once that call returned `None`.
    pub fn last_scan(&self) -> Option<Scan<'_>> {
        Some(Scan {
            scan: self.last_scan.as_deref()?,
            core: self.core.as_ref()?,
            #[cfg(feature = "processing-time-export")]
            processing_time_ns: self.last_processing_time_ns,
        })
    }
}

impl Drop for Odometer {
    fn drop(&mut self) {
        if let Some(core) = self.core.take() {
            core.shutdown();
        }
        // After the core is gone, so its GPU buffers are released first.
        pelorus_reg::backend_session_release();
    }
}

#[cfg(all(test, feature = "http-config"))]
mod tests {
    use super::*;

    #[test]
    fn loads_an_http_config() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0; 1024];
            let _ = stream.read(&mut request).unwrap();
            let body = "output.odometry_rate = ImuAndLidar\n";
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
        });

        let config = load_config(Some(&format!("http://{address}/pelorus.rl"))).unwrap();
        assert_eq!(
            config.odometry_rate,
            pelorus_core::OdometryRate::ImuAndLidar
        );
    }
}

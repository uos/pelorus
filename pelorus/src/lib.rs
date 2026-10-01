//! High-level Rust Interface for Pelorus Lidar-Inertial Odometry.
//!
//! # Example
//!
//! ```rust,no_run
//! use pelorus::Handle;
//!
//! let mut handle = Handle::new(None).unwrap();
//!
//! // Feed sensor data...
//! // handle.add_imu(...).unwrap();
//! // handle.add_points_xyzit(sec, nanosec, &points).unwrap();
//!
//! // Blocking receive
//! let odom = handle.recv().unwrap();
//! println!("position: {:?}", odom.position);
//! ```

#![warn(clippy::std_instead_of_alloc, clippy::std_instead_of_core)]

extern crate alloc;

use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "rayon")]
use std::sync::OnceLock;
use std::sync::{Mutex, mpsc};
use std::thread;

use anyhow::anyhow;
use log::error;
use nalgebra::{Isometry3, Matrix3, SMatrix};
use pelorus_core::config::{ast_to_config, layer_overrides};
use pelorus_core::{ImuMsg, LidarVendor, LioConfig, OdomMsg, SyncProcessedCloud};
use pelorus_reg::pipeline::{PelorusOutput, Pipeline};
use pelorus_sync::{ImuInitCallback, Synchronizer};
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

struct ImuInitState {
    start_time: Option<f64>,
    buffer: Option<Vec<ImuMsg>>,
}

type OdometryCallback = Arc<dyn Fn(Odometry) + Send + Sync>;
type SharedOdometryCallback = Arc<Mutex<Option<OdometryCallback>>>;

struct ProcessingThread {
    pipeline: Pipeline,
    sync_receiver: mpsc::Receiver<SyncProcessedCloud>,
    result_sender: mpsc::Sender<PelorusOutput>,
    ready_for_next: mpsc::Sender<u64>,
    callback: SharedOdometryCallback,
    shutdown: Arc<AtomicBool>,
    config: LioConfig,
    ext_lidar_to_imu: Isometry3<f64>,
}

fn to_odometry(raw: &PelorusOutput, config: &LioConfig, ext: &Isometry3<f64>) -> Odometry {
    let pose = raw.pose * config.base_to_lidar.inverse();
    let linvel = pose.rotation.inverse() * raw.linear_velocity;
    let r_b_i = config.base_to_lidar.rotation.inverse() * ext.rotation.inverse();
    let angvel = r_b_i * raw.angular_velocity;
    let q = pose.rotation.quaternion();
    let frame_id = config
        .base_link_frame
        .as_deref()
        .unwrap_or(&raw.frame_id)
        .to_string();
    let (pose_covariance, twist_covariance) = if config.base_link_frame.is_some() {
        let lidar_to_base = config.base_to_lidar.inverse();
        (
            transform_pose_covariance(raw.pose_covariance, &raw.pose, &lidar_to_base),
            transform_twist_covariance(raw.twist_covariance, &lidar_to_base),
        )
    } else {
        (raw.pose_covariance, raw.twist_covariance)
    };
    let points = raw
        .cloud
        .iter()
        .map(|p| PointXyzi {
            x: p.x,
            y: p.y,
            z: p.z,
            intensity: p.intensity,
        })
        .collect();

    Odometry {
        timestamp_sec: raw.timestamp.sec,
        timestamp_nanosec: raw.timestamp.nanosec,
        position: [pose.translation.x, pose.translation.y, pose.translation.z],
        orientation: [q.w, q.i, q.j, q.k],
        pose_covariance,
        linear_velocity: [linvel.x, linvel.y, linvel.z],
        angular_velocity: [angvel.x, angvel.y, angvel.z],
        accel_bias: [raw.accel_bias.x, raw.accel_bias.y, raw.accel_bias.z],
        gyro_bias: [raw.gyro_bias.x, raw.gyro_bias.y, raw.gyro_bias.z],
        twist_covariance,
        scan_id: raw.scan_id,
        points,
        frame_id,
        #[cfg(feature = "processing-time-export")]
        processing_time_ns: raw.processing_time_ns,
    }
}

fn transform_pose_covariance(
    covariance: [f64; 36],
    lidar_pose: &Isometry3<f64>,
    lidar_to_output: &Isometry3<f64>,
) -> [f64; 36] {
    let r_wl = lidar_pose.rotation.to_rotation_matrix();
    let r_lo = lidar_to_output.rotation.to_rotation_matrix();
    let t_lo = lidar_to_output.translation.vector;

    let mut jacobian = SMatrix::<f64, 6, 6>::identity();
    jacobian
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-r_wl.matrix() * skew_symmetric(&t_lo)));
    jacobian
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&r_lo.matrix().transpose());

    let input_cov = SMatrix::<f64, 6, 6>::from_fn(|row, col| covariance[row * 6 + col]);
    matrix6_to_array(jacobian * input_cov * jacobian.transpose())
}

fn transform_twist_covariance(
    covariance: [f64; 36],
    lidar_to_output: &Isometry3<f64>,
) -> [f64; 36] {
    let r_ol = lidar_to_output.rotation.inverse().to_rotation_matrix();
    let mut transform = SMatrix::<f64, 6, 6>::zeros();
    transform
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(r_ol.matrix());
    transform
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(r_ol.matrix());

    let input_cov = SMatrix::<f64, 6, 6>::from_fn(|row, col| covariance[row * 6 + col]);
    matrix6_to_array(transform * input_cov * transform.transpose())
}

fn skew_symmetric(v: &nalgebra::Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

fn matrix6_to_array(matrix: SMatrix<f64, 6, 6>) -> [f64; 36] {
    let mut out = [0.0; 36];
    for row in 0..6 {
        for col in 0..6 {
            out[row * 6 + col] = matrix[(row, col)];
        }
    }
    out
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

fn is_config_url(location: &str) -> bool {
    location.starts_with("http://") || location.starts_with("https://")
}

fn read_config_source(location: &str) -> anyhow::Result<String> {
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
    let default_ast = ratslang::compile_code(DEFAULT_CONFIG_STR)
        .map_err(|e| anyhow!("Failed to compile embedded default config: {e}"))?;
    let source = config_path
        .map(read_config_source)
        .transpose()?
        .unwrap_or_default();
    let parent = config_path
        .filter(|location| !is_config_url(location))
        .and_then(|location| std::path::Path::new(location).parent())
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let user_ast = layer_overrides(&source, parent, &[]).map_err(|error| match config_path {
        Some(location) => anyhow!("Failed to compile user config '{location}': {error}"),
        None => anyhow!("Failed to compile empty user config: {error}"),
    })?;
    ast_to_config(default_ast, user_ast).map_err(|e| anyhow!("Failed to build configuration: {e}"))
}

fn processing_thread_main(thread: ProcessingThread) {
    let mut sent_count: usize = 0;

    while let Ok(processed) = thread.sync_receiver.recv() {
        if thread.shutdown.load(Ordering::Relaxed) {
            continue;
        }

        let scan_id = processed.scan_id;
        let registration = thread
            .pipeline
            .register_processed_cloud(processed, Some(&thread.ready_for_next));

        match registration {
            Ok(Some(output)) => {
                let cb_opt = thread
                    .callback
                    .lock()
                    .ok()
                    .and_then(|g| g.as_ref().map(Arc::clone));

                if let Some(cb) = cb_opt {
                    cb(to_odometry(
                        &output,
                        &thread.config,
                        &thread.ext_lidar_to_imu,
                    ));
                } else {
                    sent_count += 1;
                    if let Err(e) = thread.result_sender.send(output) {
                        error!("failed to send odometry #{sent_count}: {e}");
                    }
                }
            }
            Ok(None) => {
                // Map-only scans return before the registration barrier, so the
                // correction ack for this scan id has to come from here or
                // Synchronizer waits on it forever.
                if let Err(e) = thread.ready_for_next.send(scan_id) {
                    error!("failed to acknowledge map-only scan #{scan_id}: {e}");
                }
            }
            Err(e) => {
                error!("fatal registration error. Stopping processing thread: {e:#}");
                break;
            }
        }
    }
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
/// # Callback vs polling
///
/// Results can be consumed in two ways (mutually exclusive):
/// - Polling: [`recv`](Handle::recv) (blocking) or [`try_recv`](Handle::try_recv)
///   (non-blocking).
/// - Callback: register a closure with [`set_callback`](Handle::set_callback).
///   It is called from the processing thread for every result.  While a callback
///   is registered, `recv`/`try_recv` return [`Error::CallbackActive`].
pub struct Handle {
    sync: Arc<Mutex<Option<Synchronizer>>>,
    result_receiver: mpsc::Receiver<PelorusOutput>,
    callback: SharedOdometryCallback,
    imu_init_cb: Option<ImuInitCallback>,
    imu_init_state: Arc<Mutex<ImuInitState>>,
    imu_predictor_init_time_sec: f64,
    _processing_thread: Option<thread::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    config: LioConfig,
    ext_lidar_to_imu: Isometry3<f64>,
}

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
        let mut config = load_config(config_path)?;

        configure_thread_pool(config.num_threads);

        if !device_specific_pointcloud2 {
            config.sync_params.lidar_vendor = LidarVendor::Ouster;
        }

        let pipeline = Pipeline::new(&config);
        let imu_init_cb = (!config.imu_params.lidar_only).then(|| pipeline.build_init_callback());
        let imu_predictor_init_time_sec = config.sync_node_params.imu_predictor_init_time_sec;
        let imu_init_state = Arc::new(Mutex::new(ImuInitState {
            start_time: None,
            buffer: Some(Vec::new()),
        }));

        let (ready_tx, ready_rx) = mpsc::channel::<u64>();
        let (sync, sync_receiver) = pipeline.create_sync(Some(ready_rx))?;
        let (result_tx, result_rx) = mpsc::channel();

        let callback: SharedOdometryCallback = Arc::new(Mutex::new(None));
        let shutdown = Arc::new(AtomicBool::new(false));
        let ext_lidar_to_imu = pipeline.ext();

        let processing_thread = ProcessingThread {
            pipeline,
            sync_receiver,
            result_sender: result_tx,
            ready_for_next: ready_tx,
            callback: Arc::clone(&callback),
            shutdown: Arc::clone(&shutdown),
            config: config.clone(),
            ext_lidar_to_imu,
        };

        let thread_handle = thread::spawn(move || processing_thread_main(processing_thread));

        // Paired with the release in `Drop`, so process-wide GPU resources are torn
        // down when the last handle goes away rather than at `exit()`.
        pelorus_reg::backend_session_retain();

        Ok(Handle {
            sync: Arc::new(Mutex::new(Some(sync))),
            result_receiver: result_rx,
            callback,
            imu_init_cb,
            imu_init_state,
            imu_predictor_init_time_sec,
            _processing_thread: Some(thread_handle),
            shutdown,
            config,
            ext_lidar_to_imu,
        })
    }

    /// Add an IMU measurement.
    pub fn add_imu(&self, imu: ImuMeasurement) -> Result<(), Error> {
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
        let timestamp = timestamp_sec as f64 + (timestamp_nanosec as f64 / 1e9);
        let imu_msg = ImuMsg {
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
        };

        let mut init_buffer_to_apply: Option<Vec<ImuMsg>> = None;
        if let Some(_cb) = self.imu_init_cb.as_ref() {
            let mut init_state = self.imu_init_state.lock().unwrap();

            if init_state.start_time.is_none() {
                init_state.start_time = Some(timestamp);
            }

            if let Some(start_time) = init_state.start_time {
                if timestamp < start_time {
                    init_state.start_time = Some(timestamp);
                    if let Some(buf) = init_state.buffer.as_mut() {
                        buf.clear();
                    }
                }

                let init_time = self.imu_predictor_init_time_sec;
                let delta = timestamp - init_state.start_time.unwrap_or(timestamp);

                if init_time <= 0.0 {
                    if let Some(buf) = init_state.buffer.as_mut() {
                        buf.push(imu_msg.clone());
                    }
                    init_buffer_to_apply = init_state.buffer.take();
                } else if delta < init_time {
                    if let Some(buf) = init_state.buffer.as_mut() {
                        buf.push(imu_msg);
                        return Ok(());
                    }
                } else {
                    init_buffer_to_apply = init_state.buffer.take();
                }
            }
        }

        if let (Some(cb), Some(buffer)) = (self.imu_init_cb.as_ref(), init_buffer_to_apply)
            && !buffer.is_empty()
        {
            cb.lock().unwrap()(&buffer);
        }

        let mut sync_guard = self.sync.lock().unwrap();
        let sync = sync_guard
            .as_mut()
            .ok_or_else(|| anyhow!("Handle is closed"))?;
        sync.add_imu(imu_msg);
        Ok(())
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
        let odom_msg = OdomMsg {
            twist_linear: nalgebra::Vector3::new(twist_linear[0], twist_linear[1], twist_linear[2]),
            twist_angular: nalgebra::Vector3::new(
                twist_angular[0],
                twist_angular[1],
                twist_angular[2],
            ),
            twist_cov: twist_covariance,
            stamp: ros_pointcloud2::ros::TimeMsg {
                sec: timestamp_sec,
                nanosec: timestamp_nanosec,
            },
        };

        let mut sync_guard = self.sync.lock().unwrap();
        let sync = sync_guard
            .as_mut()
            .ok_or_else(|| anyhow!("Handle is closed"))?;
        sync.add_odom(odom_msg);
        Ok(())
    }

    /// Add a point cloud as `[x, y, z, intensity, t]` f32 tuples.
    ///
    /// `t` is a per-point nanosecond offset relative to the scan header
    /// timestamp `(timestamp_sec, timestamp_nanosec)`.
    ///
    /// Returns the scan ID assigned to this cloud, or `0` on failure.
    pub fn add_points_xyzit(
        &self,
        timestamp_sec: i32,
        timestamp_nanosec: u32,
        points: &[[f32; 5]],
    ) -> Result<u64, Error> {
        if points.is_empty() {
            return Ok(0);
        }

        let cloud_msg = build_cloud_msg(timestamp_sec, timestamp_nanosec, points.len(), |i| {
            let p = points[i];
            (p[0], p[1], p[2], p[3], p[4] as u32)
        });

        let mut sync_guard = self.sync.lock().unwrap();
        let sync = sync_guard
            .as_mut()
            .ok_or_else(|| anyhow!("Handle is closed"))?;
        Ok(sync.add_pointcloud(cloud_msg))
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
        if points.is_empty() {
            return Ok(0);
        }

        let cloud_msg = build_cloud_msg(timestamp_sec, timestamp_nanosec, points.len(), |i| {
            let p = points[i];
            (p[0], p[1], p[2], 0.0f32, p[3] as u32)
        });

        let mut sync_guard = self.sync.lock().unwrap();
        let sync = sync_guard
            .as_mut()
            .ok_or_else(|| anyhow!("Handle is closed"))?;
        Ok(sync.add_pointcloud(cloud_msg))
    }

    /// Add a raw PointCloud2-formatted point cloud.
    ///
    /// Accepts the same data layout as a ROS2 `sensor_msgs/PointCloud2` message.
    /// For simpler input formats prefer [`add_points_xyzit`](Handle::add_points_xyzit)
    /// or [`add_points_xyzt`](Handle::add_points_xyzt).
    ///
    /// Returns the scan ID assigned to this cloud or fails if Handle is closed.
    pub fn add_pointcloud_raw(&self, cloud: RawPointCloud<'_>) -> Result<u64, Error> {
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

        let cloud_msg = impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(cloud);

        let mut sync_guard = self.sync.lock().unwrap();
        let sync = sync_guard
            .as_mut()
            .ok_or_else(|| anyhow!("Handle is closed"))?;
        Ok(sync.add_pointcloud(cloud_msg))
    }

    /// Non-blocking receive.
    ///
    /// Returns `Ok(None)` when no result is ready yet.
    /// Returns `Err(Error::CallbackActive)` when a callback is registered.
    pub fn try_recv(&self) -> Result<Option<Odometry>, Error> {
        if self.callback.lock().unwrap().is_some() {
            return Err(Error::CallbackActive);
        }

        match self.result_receiver.try_recv() {
            Ok(raw) => Ok(Some(to_odometry(
                &raw,
                &self.config,
                &self.ext_lidar_to_imu,
            ))),
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

        match self.result_receiver.recv() {
            Ok(raw) => Ok(to_odometry(&raw, &self.config, &self.ext_lidar_to_imu)),
            Err(_) => Err(Error::Disconnected),
        }
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
        while self.result_receiver.try_recv().is_ok() {}
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

impl Drop for Handle {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);

        if let Ok(mut guard) = self.sync.lock() {
            let _ = guard.take();
        }

        if let Some(handle) = self._processing_thread.take() {
            let _ = handle.join();
        }

        // After the worker is joined, so its GPU buffers are gone first.
        pelorus_reg::backend_session_release();
    }
}

#[cfg(test)]
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

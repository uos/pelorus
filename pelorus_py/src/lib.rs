//! Python bindings for Pelorus.

use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySequence};
#[cfg(feature = "rayon")]
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use log::{error, info};
use nalgebra::{Isometry3, Matrix3, SMatrix};

use pelorus_core::config::ast_to_config;
use pelorus_core::{ImuMsg, LidarVendor, LioConfig, OdomMsg, SyncProcessedCloud};
use pelorus_reg::pipeline::{PelorusOutput, Pipeline};
use pelorus_sync::Synchronizer;
use ros_pointcloud2::prelude::*;
use ros2_interfaces_jazzy_serde::sensor_msgs::msg::{PointCloud2, PointField};

const DEFAULT_CONFIG_STR: &str = pelorus_core::DEFAULT_CONFIG_STR;

#[cfg(any(
    all(feature = "cuda", feature = "metal"),
    all(feature = "cuda", feature = "vulkan"),
    all(feature = "metal", feature = "vulkan"),
))]
compile_error!("Select at most one Pelorus acceleration backend: Metal, CUDA, or Vulkan");

impl_pointcloud2_for_ros2_interfaces_jazzy_serde!();

/// Helper to convert sec/nanosec to f64 seconds
fn sec_nanosec_to_timestamp(sec: i32, nanosec: u32) -> f64 {
    sec as f64 + (nanosec as f64 / 1e9)
}

trait NumpyPointScalar: Copy {
    fn to_f64(self) -> f64;
}

impl NumpyPointScalar for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl NumpyPointScalar for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}

fn point_component_to_f32(value: f64, point_index: usize, field: &str) -> PyResult<f32> {
    if !value.is_finite() || value < f32::MIN as f64 || value > f32::MAX as f64 {
        return Err(PyValueError::new_err(format!(
            "point {point_index} field '{field}' must be a finite float32 value"
        )));
    }
    Ok(value as f32)
}

fn point_timestamp_to_u32(value: f64, point_index: usize) -> PyResult<u32> {
    if !value.is_finite() || value < 0.0 || value > u32::MAX as f64 {
        return Err(PyValueError::new_err(format!(
            "point {point_index} field 't' must be a finite nanosecond offset in 0..={}",
            u32::MAX
        )));
    }
    let rounded = value.round();
    if rounded > u32::MAX as f64 {
        return Err(PyValueError::new_err(format!(
            "point {point_index} field 't' must be a finite nanosecond offset in 0..={}",
            u32::MAX
        )));
    }
    Ok(rounded as u32)
}

fn validate_pointcloud_buffer(
    height: u32,
    width: u32,
    point_step: u32,
    row_step: u32,
    data_len: usize,
) -> PyResult<()> {
    if height == 0 || width == 0 {
        return Err(PyValueError::new_err(
            "PointCloud2 height and width must be greater than zero",
        ));
    }
    if point_step == 0 {
        return Err(PyValueError::new_err(
            "PointCloud2 point_step must be greater than zero",
        ));
    }

    let minimum_row_step = width
        .checked_mul(point_step)
        .ok_or_else(|| PyValueError::new_err("PointCloud2 row size overflows u32"))?;
    if row_step < minimum_row_step {
        return Err(PyValueError::new_err(format!(
            "PointCloud2 row_step ({row_step}) is smaller than width * point_step ({minimum_row_step})"
        )));
    }

    let required_data_len = (height as usize)
        .checked_mul(row_step as usize)
        .ok_or_else(|| PyValueError::new_err("PointCloud2 data size overflows usize"))?;
    if data_len < required_data_len {
        return Err(PyValueError::new_err(format!(
            "PointCloud2 data contains {data_len} bytes but at least {required_data_len} are required"
        )));
    }

    Ok(())
}

#[cfg(feature = "rayon")]
static RAYON_THREADS: OnceLock<usize> = OnceLock::new();

fn configure_thread_pool(num_threads: usize) {
    if num_threads == 0 {
        return;
    }
    #[cfg(feature = "rayon")]
    {
        let configured_threads = RAYON_THREADS.get_or_init(|| {
            if let Err(e) = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
            {
                error!("Failed to configure rayon thread pool to {num_threads} threads: {e}");
            }
            num_threads
        });
        if *configured_threads != num_threads {
            log::warn!(
                "Rayon thread pool already configured with {} threads. Ignoring request for {} threads (first Handle wins).",
                configured_threads,
                num_threads
            );
        }
    }
    #[cfg(not(feature = "rayon"))]
    log::warn!(
        "Ignoring num_threads={num_threads}. Thread-pool configuration requires the rayon feature."
    );
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

fn load_pelorus_config(config_path: Option<&str>) -> anyhow::Result<LioConfig> {
    let default_ast = ratslang::compile_code(DEFAULT_CONFIG_STR)
        .map_err(|e| anyhow::anyhow!("Failed to compile embedded default config: {e}"))?;
    let user_ast = if let Some(path) = config_path {
        let path = std::path::Path::new(path);
        if !path.exists() {
            return Err(anyhow::anyhow!(
                "Configuration file not found: {}",
                path.display()
            ));
        }
        ratslang::compile_file(&path.to_path_buf(), None, None)
            .map_err(|e| anyhow::anyhow!("Failed to compile user config {}: {e}", path.display()))?
    } else {
        ratslang::compile_code("")?
    };

    ast_to_config(default_ast, user_ast)
        .map_err(|e| anyhow::anyhow!("Failed to build configuration: {e}"))
}

struct ImuInitState {
    start_time: Option<f64>,
    buffer: Option<Vec<ImuMsg>>,
}

type ImuInitCallback = Arc<Mutex<dyn FnMut(&[ImuMsg]) + Send + Sync>>;

struct ProcessingThread {
    pipeline: Pipeline,
    sync_receiver: mpsc::Receiver<SyncProcessedCloud>,
    result_sender: mpsc::Sender<PelorusOutput>,
    ready_for_next: mpsc::Sender<u64>,
    odom_callback: Arc<Mutex<Option<Py<PyAny>>>>,
    shutdown: Arc<AtomicBool>,
}

fn dispatch_output(
    output: PelorusOutput,
    odom_callback: &Arc<Mutex<Option<Py<PyAny>>>>,
    result_sender: &mpsc::Sender<PelorusOutput>,
    sent_count: &mut usize,
    ext_lidar_to_imu: &nalgebra::Isometry3<f64>,
    config: &LioConfig,
) {
    // Check if callback is set
    let cb_opt = odom_callback.lock().ok().and_then(|guard| {
        guard
            .as_ref()
            .map(|cb| Python::attach(|py| cb.clone_ref(py)))
    });

    if let Some(cb) = cb_opt {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            if let Err(e) = PelorusHandle::fill_odom_dict(&dict, &output, config, ext_lidar_to_imu)
            {
                error!("failed to build odometry dict for callback #{sent_count}: {e}");
                return;
            }

            if let Err(e) = cb.bind(py).call1((dict,)) {
                error!("odometry callback raised for #{sent_count}: {e}");
                e.print(py);
            }
        });
    } else {
        *sent_count += 1;
        if let Err(e) = result_sender.send(output) {
            error!("failed to send odometry #{sent_count}: {e}");
        }
    }
}

fn processing_thread_main(thread: ProcessingThread) {
    let mut sent_count: usize = 0;

    if let Err(e) = thread.ready_for_next.send(0) {
        error!("failed to send initial ready signal: {e}");
    }

    while let Ok(processed) = thread.sync_receiver.recv() {
        // Shutdown mode: keep draining the sync output channel so Synchronizer's producer thread
        // doesn't panic on send after Synchronizer drops its receiver.
        if thread.shutdown.load(Ordering::Relaxed) {
            continue;
        }

        let scan_id = processed.scan_id;
        let lidar_to_imu = nalgebra::Isometry3::from_parts(
            thread
                .pipeline
                .config
                .imu_params
                .initial_translation_lidar_to_imu
                .into(),
            nalgebra::UnitQuaternion::from_matrix(
                &thread
                    .pipeline
                    .config
                    .imu_params
                    .initial_rotation_lidar_to_imu,
            ),
        );

        let registration = thread
            .pipeline
            .register_processed_cloud(processed, Some(&thread.ready_for_next));

        match registration {
            Ok(Some(output)) => {
                dispatch_output(
                    output,
                    &thread.odom_callback,
                    &thread.result_sender,
                    &mut sent_count,
                    &lidar_to_imu,
                    &thread.pipeline.config,
                );
            }
            Ok(None) => {
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

    info!("processing thread exiting (sync channel disconnected)");
}

/// Python wrapper for Pelorus handle
#[pyclass(name = "Handle", module = "pelorus")]
struct PelorusHandle {
    sync: Arc<Mutex<Option<Synchronizer>>>,
    result_receiver: Arc<Mutex<mpsc::Receiver<PelorusOutput>>>,
    odom_callback: Arc<Mutex<Option<Py<PyAny>>>>,
    imu_init_cb: Option<ImuInitCallback>,
    imu_init_state: Arc<Mutex<ImuInitState>>,
    imu_predictor_init_time_sec: f64,
    _processing_thread: Option<thread::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    config: LioConfig,
}

#[pymethods]
impl PelorusHandle {
    #[new]
    #[pyo3(signature = (config_path=None, device_specific_pointcloud2=true))]
    /// Create a new Pelorus instance.
    ///
    /// - `config_path`: optional path to a `.rl` config file. If omitted, embedded defaults are used.
    /// - `device_specific_pointcloud2`: when true (default) Pelorus will use device-specific parsing
    ///   for PointCloud2 inputs (field meanings and timestamps are vendor/device-dependent). When
    ///   false, Pelorus uses the generic parsing path where per-point `t` is interpreted as a
    ///   header-relative nanoseconds offset (see README for the recommended layout).
    fn new(config_path: Option<&str>, device_specific_pointcloud2: bool) -> PyResult<Self> {
        let mut config = load_pelorus_config(config_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load configuration: {}", e)))?;

        configure_thread_pool(config.num_threads);

        if !device_specific_pointcloud2 {
            // Select the generic parsing path (internally uses a vendor mode, but we keep the
            // public API vendor-agnostic).
            config.sync_params.lidar_vendor = LidarVendor::Ouster;
        }

        // Ensure Rust logging works when used from Python (RUST_LOG env var)
        let _ = env_logger::try_init();

        let pipeline = Pipeline::new(&config);

        let imu_init_cb = Some(pipeline.build_init_callback());
        let imu_predictor_init_time_sec = config.sync_node_params.imu_predictor_init_time_sec;
        let imu_init_state = Arc::new(Mutex::new(ImuInitState {
            start_time: None,
            buffer: Some(Vec::new()),
        }));

        // Create the ready_for_next channel for synchronization with Synchronizer
        let (ready_tx, ready_rx) = mpsc::channel::<u64>();

        // Pass the receiver to Synchronizer for proper synchronization
        let (sync, sync_receiver) = pipeline.create_sync(Some(ready_rx)).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to initialize Synchronizer: {}", e))
        })?;

        let (result_tx, result_rx) = mpsc::channel();

        let odom_callback: Arc<Mutex<Option<Py<PyAny>>>> = Arc::new(Mutex::new(None));
        let shutdown: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
        let keep_config = pipeline.config.clone();

        let processing_thread = ProcessingThread {
            pipeline,
            sync_receiver,
            result_sender: result_tx,
            ready_for_next: ready_tx,
            odom_callback: Arc::clone(&odom_callback),
            shutdown: Arc::clone(&shutdown),
        };

        let thread_handle = thread::spawn(move || {
            processing_thread_main(processing_thread);
        });

        // Paired with the release in `Drop`, so process-wide GPU resources are torn
        // down when the last handle goes away rather than at `exit()`.
        pelorus_reg::backend_session_retain();

        Ok(PelorusHandle {
            sync: Arc::new(Mutex::new(Some(sync))),
            result_receiver: Arc::new(Mutex::new(result_rx)),
            odom_callback,
            imu_init_cb,
            imu_init_state,
            imu_predictor_init_time_sec,
            _processing_thread: Some(thread_handle),
            shutdown,
            config: keep_config,
        })
    }

    /// Set an odometry callback (async) like `pelorus_set_callback` in the C bindings.
    ///
    /// Semantics (mirrors C):
    /// - If a callback is set (non-None), odometry results are delivered to the callback from a
    ///   background thread and are NOT queued for `try_recv()` / `recv()`.
    /// - If callback is None, results are queued and can be consumed via `try_recv()` / `recv()`.
    fn set_callback(&mut self, py: Python<'_>, callback: Option<Py<PyAny>>) -> PyResult<()> {
        if let Some(cb) = &callback {
            let bound = cb.bind(py);
            if !bound.is_callable() {
                return Err(PyValueError::new_err("callback must be callable or None"));
            }
        }

        // Acquire callback lock first to make enabling/disabling atomic w.r.t. the dispatcher,
        // then drain any queued results while still holding that lock so there's no race window.
        let mut cb_guard = self.odom_callback.lock().unwrap();
        if callback.is_some() {
            let receiver = self.result_receiver.lock().unwrap();
            while receiver.try_recv().is_ok() {}
            *cb_guard = callback;
        } else {
            *cb_guard = None;
        }
        Ok(())
    }

    /// Disable odometry callback (equivalent to `set_callback(None)`).
    fn clear_callback(&mut self) -> PyResult<()> {
        *self.odom_callback.lock().unwrap() = None;
        Ok(())
    }

    /// Add an IMU message
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (timestamp_sec, timestamp_nanosec, orientation, orientation_covariance, angular_velocity, angular_velocity_covariance, linear_acceleration, linear_acceleration_covariance))]
    fn add_imu(
        &self,
        timestamp_sec: i32,
        timestamp_nanosec: u32,
        orientation: [f64; 4],
        orientation_covariance: [f64; 9],
        angular_velocity: [f64; 3],
        angular_velocity_covariance: [f64; 9],
        linear_acceleration: [f64; 3],
        linear_acceleration_covariance: [f64; 9],
    ) -> PyResult<()> {
        let timestamp = sec_nanosec_to_timestamp(timestamp_sec, timestamp_nanosec);
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
                // Handle bag restarts / time going backwards during init.
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
            .ok_or_else(|| PyRuntimeError::new_err("Handle is closed"))?;
        sync.add_imu(imu_msg);
        Ok(())
    }

    /// Add a ROS2 `sensor_msgs/msg/Imu` Python object directly.
    ///
    /// Works with messages coming from `mcap_ros2` or `rclpy`.
    fn add_imu_ros(&self, imu: &Bound<'_, PyAny>) -> PyResult<()> {
        let (sec, nanosec, _frame_id) = Self::extract_header_stamp_frame_id(imu)?;

        let orientation_any = imu.getattr("orientation")?;
        let orientation = [
            orientation_any.getattr("w")?.extract::<f64>()?,
            orientation_any.getattr("x")?.extract::<f64>()?,
            orientation_any.getattr("y")?.extract::<f64>()?,
            orientation_any.getattr("z")?.extract::<f64>()?,
        ];

        let angular_any = imu.getattr("angular_velocity")?;
        let angular_velocity = [
            angular_any.getattr("x")?.extract::<f64>()?,
            angular_any.getattr("y")?.extract::<f64>()?,
            angular_any.getattr("z")?.extract::<f64>()?,
        ];

        let linear_any = imu.getattr("linear_acceleration")?;
        let linear_acceleration = [
            linear_any.getattr("x")?.extract::<f64>()?,
            linear_any.getattr("y")?.extract::<f64>()?,
            linear_any.getattr("z")?.extract::<f64>()?,
        ];

        let orientation_covariance_vec: Vec<f64> =
            imu.getattr("orientation_covariance")?.extract()?;
        let angular_velocity_covariance_vec: Vec<f64> =
            imu.getattr("angular_velocity_covariance")?.extract()?;
        let linear_acceleration_covariance_vec: Vec<f64> =
            imu.getattr("linear_acceleration_covariance")?.extract()?;

        let orientation_covariance: [f64; 9] = orientation_covariance_vec
            .try_into()
            .map_err(|_| PyValueError::new_err("orientation_covariance must have 9 elements"))?;
        let angular_velocity_covariance: [f64; 9] =
            angular_velocity_covariance_vec.try_into().map_err(|_| {
                PyValueError::new_err("angular_velocity_covariance must have 9 elements")
            })?;
        let linear_acceleration_covariance: [f64; 9] =
            linear_acceleration_covariance_vec.try_into().map_err(|_| {
                PyValueError::new_err("linear_acceleration_covariance must have 9 elements")
            })?;

        self.add_imu(
            sec,
            nanosec,
            orientation,
            orientation_covariance,
            angular_velocity,
            angular_velocity_covariance,
            linear_acceleration,
            linear_acceleration_covariance,
        )
    }

    /// Add an odometry message
    #[pyo3(signature = (timestamp_sec, timestamp_nanosec, twist_linear, twist_angular, twist_covariance))]
    fn add_odom(
        &self,
        timestamp_sec: i32,
        timestamp_nanosec: u32,
        twist_linear: [f64; 3],
        twist_angular: [f64; 3],
        twist_covariance: [f64; 36],
    ) -> PyResult<()> {
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
            .ok_or_else(|| PyRuntimeError::new_err("Handle is closed"))?;
        sync.add_odom(odom_msg);
        Ok(())
    }

    /// Add a ROS2 `nav_msgs/msg/Odometry` Python object directly.
    ///
    /// Only uses `header.stamp` and `twist` fields.
    fn add_odom_ros(&self, odom: &Bound<'_, PyAny>) -> PyResult<()> {
        let (sec, nanosec, _frame_id) = Self::extract_header_stamp_frame_id(odom)?;
        let twist = odom.getattr("twist")?;
        let twist_twist = twist.getattr("twist")?;
        let lin = twist_twist.getattr("linear")?;
        let ang = twist_twist.getattr("angular")?;

        let twist_linear = [
            lin.getattr("x")?.extract::<f64>()?,
            lin.getattr("y")?.extract::<f64>()?,
            lin.getattr("z")?.extract::<f64>()?,
        ];
        let twist_angular = [
            ang.getattr("x")?.extract::<f64>()?,
            ang.getattr("y")?.extract::<f64>()?,
            ang.getattr("z")?.extract::<f64>()?,
        ];

        let cov_vec: Vec<f64> = twist.getattr("covariance")?.extract()?;
        let twist_covariance: [f64; 36] = cov_vec
            .try_into()
            .map_err(|_| PyValueError::new_err("twist.covariance must have 36 elements"))?;

        self.add_odom(sec, nanosec, twist_linear, twist_angular, twist_covariance)
    }

    /// Add a point cloud
    ///
    /// Accepts multiple input formats:
    /// 1. PointCloud2 dict (ROS2 format)
    /// 2. NumPy float32/float64 array (Nx3/Nx4/Nx5)
    ///    - Nx3: [x, y, z]
    ///    - Nx4: [x, y, z, t] where `t` is `uint32` nanoseconds offset (relative to header timestamp)
    ///    - Nx5: [x, y, z, intensity, t]
    /// 3. Python sequence of point sequences with the same shapes as above
    ///
    /// Timestamp semantics:
    /// - For dict/ROS2 PointCloud2: uses `header.stamp` as the scan reference time.
    /// - For numpy/list inputs: you must provide `timestamp_sec`/`timestamp_nanosec` which become the
    ///   scan header timestamp (absolute), and optional per-point `t` values are interpreted as
    ///   nanoseconds offset relative to that header time.
    #[pyo3(signature = (cloud, timestamp_sec=None, timestamp_nanosec=None))]
    fn add_pointcloud(
        &mut self,
        cloud: &Bound<'_, PyAny>,
        timestamp_sec: Option<i32>,
        timestamp_nanosec: Option<u32>,
    ) -> PyResult<u64> {
        if let Ok(dict) = cloud.cast::<PyDict>() {
            let scan_id = self.add_pointcloud_from_dict(dict)?;
            Ok(scan_id)
        } else if let Ok(array) = cloud.extract::<PyReadonlyArray2<f32>>() {
            let sec = timestamp_sec.ok_or_else(|| {
                PyValueError::new_err("timestamp_sec required for numpy array input")
            })?;
            let nanosec = timestamp_nanosec.ok_or_else(|| {
                PyValueError::new_err("timestamp_nanosec required for numpy array input")
            })?;
            let scan_id = self.add_pointcloud_from_numpy(array.as_array(), sec, nanosec)?;
            Ok(scan_id)
        } else if let Ok(array) = cloud.extract::<PyReadonlyArray2<f64>>() {
            let sec = timestamp_sec.ok_or_else(|| {
                PyValueError::new_err("timestamp_sec required for numpy array input")
            })?;
            let nanosec = timestamp_nanosec.ok_or_else(|| {
                PyValueError::new_err("timestamp_nanosec required for numpy array input")
            })?;
            let scan_id = self.add_pointcloud_from_numpy(array.as_array(), sec, nanosec)?;
            Ok(scan_id)
        } else if let Ok(sequence) = cloud.cast::<PySequence>() {
            let sec = timestamp_sec.ok_or_else(|| {
                PyValueError::new_err("timestamp_sec required for sequence input")
            })?;
            let nanosec = timestamp_nanosec.ok_or_else(|| {
                PyValueError::new_err("timestamp_nanosec required for sequence input")
            })?;
            let scan_id = self.add_pointcloud_from_sequence(sequence, sec, nanosec)?;
            Ok(scan_id)
        } else {
            Err(PyValueError::new_err(
                "cloud must be a PointCloud2 dict, float32/float64 NumPy array (Nx3/Nx4/Nx5), or sequence of points",
            ))
        }
    }

    /// Add a ROS2 `sensor_msgs/msg/PointCloud2` Python object directly.
    fn add_pointcloud2_ros(&mut self, cloud: &Bound<'_, PyAny>) -> PyResult<u64> {
        self.add_pointcloud_from_ros_msg(cloud)
    }

    /// Try to receive odometry (non-blocking)
    fn try_recv(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        // Fail fast when callback mode is enabled.
        if self.odom_callback.lock().unwrap().is_some() {
            return Err(PyRuntimeError::new_err(
                "Callback is set. try_recv() unavailable. Use callback API or clear_callback() first.",
            ));
        }

        let receiver = self.result_receiver.lock().unwrap();
        match receiver.try_recv() {
            Ok(odom) => {
                let dict = PyDict::new(py);
                let lidar_to_imu = nalgebra::Isometry3::from_parts(
                    self.config
                        .imu_params
                        .initial_translation_lidar_to_imu
                        .into(),
                    nalgebra::UnitQuaternion::from_matrix(
                        &self.config.imu_params.initial_rotation_lidar_to_imu,
                    ),
                );
                Self::fill_odom_dict(&dict, &odom, &self.config, &lidar_to_imu)?;
                Ok(Some(dict.into()))
            }
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => {
                Err(PyRuntimeError::new_err("Channel disconnected"))
            }
        }
    }

    /// Blocking receive odometry
    fn recv(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Fail fast when callback mode is enabled.
        if self.odom_callback.lock().unwrap().is_some() {
            return Err(PyRuntimeError::new_err(
                "Callback is set. recv() unavailable. Use callback API or clear_callback() first.",
            ));
        }

        // If IMU initialization hasn't completed yet (buffer still present), return last known odometry
        if let Ok(init_state) = self.imu_init_state.lock()
            && init_state.buffer.is_some()
        {
            // No odometry yet during init. Block until we get one.
        }

        let receiver_arc = Arc::clone(&self.result_receiver);

        // Release GIL while waiting
        let odom = py
            .detach(move || {
                let receiver = receiver_arc.lock().unwrap();
                receiver.recv()
            })
            .map_err(|_| PyRuntimeError::new_err("Channel disconnected"))?;

        let dict = PyDict::new(py);
        let lidar_to_imu = nalgebra::Isometry3::from_parts(
            self.config
                .imu_params
                .initial_translation_lidar_to_imu
                .into(),
            nalgebra::UnitQuaternion::from_matrix(
                &self.config.imu_params.initial_rotation_lidar_to_imu,
            ),
        );

        Self::fill_odom_dict(&dict, &odom, &self.config, &lidar_to_imu)?;
        Ok(dict.into())
    }

    /// Get version
    #[staticmethod]
    fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}

impl Drop for PelorusHandle {
    fn drop(&mut self) {
        // Two-phase shutdown:
        // 1) Tell the processing thread to stop releasing Synchronizer ticks once Synchronizer is dropped.
        // 2) Drop Synchronizer to stop its producer thread.
        // 3) Join the processing thread (it will drain until sync_receiver disconnects).
        self.shutdown.store(true, Ordering::Relaxed);

        // Drop Synchronizer while the processing thread is still alive and draining.
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

impl PelorusHandle {
    fn extract_stamp_sec_nanosec(stamp: &Bound<'_, PyAny>) -> PyResult<(i32, u32)> {
        let sec: i32 = stamp.getattr("sec")?.extract()?;
        let nanosec: u32 = stamp.getattr("nanosec")?.extract()?;
        Ok((sec, nanosec))
    }

    fn extract_header_stamp_frame_id(msg: &Bound<'_, PyAny>) -> PyResult<(i32, u32, String)> {
        let header = msg.getattr("header")?;
        let stamp = header.getattr("stamp")?;
        let (sec, nanosec) = Self::extract_stamp_sec_nanosec(&stamp)?;
        let frame_id: String = header.getattr("frame_id")?.extract()?;
        Ok((sec, nanosec, frame_id))
    }

    /// Helper to fill odometry dict with timestamps
    fn fill_odom_dict(
        dict: &Bound<'_, PyDict>,
        odom: &PelorusOutput,
        config: &LioConfig,
        ext_lidar_to_imu: &nalgebra::Isometry3<f64>,
    ) -> PyResult<()> {
        dict.set_item("timestamp_sec", odom.timestamp.sec)?;
        dict.set_item("timestamp_nanosec", odom.timestamp.nanosec)?;
        // Keep a float timestamp for convenience/backwards compatibility.
        let timestamp = sec_nanosec_to_timestamp(odom.timestamp.sec, odom.timestamp.nanosec);
        dict.set_item("timestamp", timestamp)?;

        // Always apply base_to_lidar (identity by default = no-op when unconfigured).
        let pose = odom.pose * config.base_to_lidar.inverse();
        let linvel = pose.rotation.inverse() * odom.linear_velocity;
        let r_b_i = config.base_to_lidar.rotation.inverse() * ext_lidar_to_imu.rotation.inverse();
        let angvel = r_b_i * odom.angular_velocity;
        let frame_id = config.base_link_frame.as_deref().unwrap_or(&odom.frame_id);
        dict.set_item("frame_id", frame_id)?;
        let (pose_covariance, twist_covariance) = if config.base_link_frame.is_some() {
            let lidar_to_base = config.base_to_lidar.inverse();
            (
                transform_pose_covariance(odom.pose_covariance, &odom.pose, &lidar_to_base),
                transform_twist_covariance(odom.twist_covariance, &lidar_to_base),
            )
        } else {
            (odom.pose_covariance, odom.twist_covariance)
        };

        let position = vec![pose.translation.x, pose.translation.y, pose.translation.z];
        dict.set_item("position", position)?;

        let q = pose.rotation.quaternion();
        let orientation = vec![q.w, q.i, q.j, q.k];
        dict.set_item("orientation", orientation)?;

        dict.set_item("pose_covariance", pose_covariance.to_vec())?;

        let linear_velocity = vec![linvel.x, linvel.y, linvel.z];
        dict.set_item("linear_velocity", linear_velocity)?;

        let angular_velocity = vec![angvel.x, angvel.y, angvel.z];
        dict.set_item("angular_velocity", angular_velocity)?;

        dict.set_item(
            "accel_bias",
            vec![odom.accel_bias.x, odom.accel_bias.y, odom.accel_bias.z],
        )?;
        dict.set_item(
            "gyro_bias",
            vec![odom.gyro_bias.x, odom.gyro_bias.y, odom.gyro_bias.z],
        )?;

        dict.set_item("twist_covariance", twist_covariance.to_vec())?;

        let num_points = odom.cloud.len();
        let mut flat_data = Vec::with_capacity(num_points * 4);
        for p in &odom.cloud {
            flat_data.push(p.x as f64);
            flat_data.push(p.y as f64);
            flat_data.push(p.z as f64);
            flat_data.push(p.intensity as f64);
        }
        let array = ndarray::Array2::from_shape_vec((num_points, 4), flat_data)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create array: {}", e)))?;
        let points_array = array.into_pyarray(dict.py());
        dict.set_item("points", points_array)?;
        dict.set_item("scan_id", odom.scan_id)?;
        #[cfg(feature = "processing-time-export")]
        dict.set_item("processing_time_ns", odom.processing_time_ns)?;

        Ok(())
    }

    /// Add point cloud from PointCloud2 dict
    fn add_pointcloud_from_dict(&self, cloud_dict: &Bound<'_, PyDict>) -> PyResult<u64> {
        // Extract header - PyDict.get_item returns PyResult<Option<Bound>>
        let header = cloud_dict
            .get_item("header")?
            .ok_or_else(|| PyValueError::new_err("Missing 'header' field"))?;
        // header is now Bound<PyAny>, and PyAny.get_item() raises KeyError if not found
        let stamp = header.get_item("stamp")?;
        let sec: i32 = stamp.get_item("sec")?.extract()?;
        let nanosec: u32 = stamp.get_item("nanosec")?.extract()?;
        let frame_id: String = header.get_item("frame_id")?.extract()?;

        // Extract cloud data - PyDict.get_item returns PyResult<Option<Bound>>
        let height: u32 = cloud_dict
            .get_item("height")?
            .ok_or_else(|| PyValueError::new_err("Missing 'height' field"))?
            .extract()?;
        let width: u32 = cloud_dict
            .get_item("width")?
            .ok_or_else(|| PyValueError::new_err("Missing 'width' field"))?
            .extract()?;
        let is_bigendian: bool = cloud_dict
            .get_item("is_bigendian")?
            .ok_or_else(|| PyValueError::new_err("Missing 'is_bigendian' field"))?
            .extract()?;
        let point_step: u32 = cloud_dict
            .get_item("point_step")?
            .ok_or_else(|| PyValueError::new_err("Missing 'point_step' field"))?
            .extract()?;
        let row_step: u32 = cloud_dict
            .get_item("row_step")?
            .ok_or_else(|| PyValueError::new_err("Missing 'row_step' field"))?
            .extract()?;
        let is_dense: bool = cloud_dict
            .get_item("is_dense")?
            .ok_or_else(|| PyValueError::new_err("Missing 'is_dense' field"))?
            .extract()?;

        // Extract fields
        let fields_bound = cloud_dict
            .get_item("fields")?
            .ok_or_else(|| PyValueError::new_err("Missing 'fields' field"))?;
        let fields_list = fields_bound.cast::<PyList>()?;
        let mut fields = Vec::new();
        for field in fields_list.iter() {
            // field is PyAny, so get_item() raises KeyError
            let name: String = field.get_item("name")?.extract()?;
            let offset: u32 = field.get_item("offset")?.extract()?;
            let datatype: u8 = field.get_item("datatype")?.extract()?;
            let count: u32 = field.get_item("count")?.extract()?;
            fields.push(PointField {
                name,
                offset,
                datatype,
                count,
            });
        }

        // Extract data
        let data: Vec<u8> = cloud_dict
            .get_item("data")?
            .ok_or_else(|| PyValueError::new_err("Missing 'data' field"))?
            .extract()?;
        if fields.is_empty() {
            return Err(PyValueError::new_err("PointCloud2 fields cannot be empty"));
        }
        validate_pointcloud_buffer(height, width, point_step, row_step, data.len())?;

        // Create PointCloud2 message
        let cloud_msg = PointCloud2 {
            header: ros2_interfaces_jazzy_serde::std_msgs::msg::Header {
                stamp: ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time { sec, nanosec },
                frame_id,
            },
            height,
            width,
            fields,
            is_bigendian,
            point_step,
            row_step,
            data,
            is_dense,
        };

        let cloud_msg_internal: PointCloud2Msg =
            impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(cloud_msg);
        let mut sync_guard = self.sync.lock().unwrap();
        let sync = sync_guard
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Handle is closed"))?;
        let scan_id = sync.add_pointcloud(cloud_msg_internal);
        Ok(scan_id)
    }

    fn add_pointcloud_from_ros_msg(&self, cloud_msg: &Bound<'_, PyAny>) -> PyResult<u64> {
        let header = cloud_msg.getattr("header")?;
        let stamp = header.getattr("stamp")?;
        let (sec, nanosec) = Self::extract_stamp_sec_nanosec(&stamp)?;
        let frame_id: String = header.getattr("frame_id")?.extract()?;

        let height: u32 = cloud_msg.getattr("height")?.extract()?;
        let width: u32 = cloud_msg.getattr("width")?.extract()?;
        let is_bigendian: bool = cloud_msg.getattr("is_bigendian")?.extract()?;
        let point_step: u32 = cloud_msg.getattr("point_step")?.extract()?;
        let row_step: u32 = cloud_msg.getattr("row_step")?.extract()?;
        let is_dense: bool = cloud_msg.getattr("is_dense")?.extract()?;

        let fields_any = cloud_msg.getattr("fields")?;
        let fields_list = fields_any.cast::<PyList>()?;
        let mut fields: Vec<PointField> = Vec::with_capacity(fields_list.len());
        for field_any in fields_list.iter() {
            let name: String = field_any.getattr("name")?.extract()?;
            let offset: u32 = field_any.getattr("offset")?.extract()?;
            let datatype: u8 = field_any.getattr("datatype")?.extract()?;
            let count: u32 = field_any.getattr("count")?.extract()?;
            fields.push(PointField {
                name,
                offset,
                datatype,
                count,
            });
        }

        let data_any = cloud_msg.getattr("data")?;
        let data: Vec<u8> = data_any.extract()?;
        if fields.is_empty() {
            return Err(PyValueError::new_err("PointCloud2 fields cannot be empty"));
        }
        validate_pointcloud_buffer(height, width, point_step, row_step, data.len())?;

        let cloud_msg = PointCloud2 {
            header: ros2_interfaces_jazzy_serde::std_msgs::msg::Header {
                stamp: ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time { sec, nanosec },
                frame_id,
            },
            height,
            width,
            fields,
            is_bigendian,
            point_step,
            row_step,
            data,
            is_dense,
        };

        let cloud_msg_internal: PointCloud2Msg =
            impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(cloud_msg);
        let mut sync_guard = self.sync.lock().unwrap();
        let sync = sync_guard
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Handle is closed"))?;
        let scan_id = sync.add_pointcloud(cloud_msg_internal);
        Ok(scan_id)
    }

    /// Add point cloud from a NumPy array.
    fn add_pointcloud_from_numpy<T: NumpyPointScalar>(
        &self,
        array: ndarray::ArrayView2<'_, T>,
        timestamp_sec: i32,
        timestamp_nanosec: u32,
    ) -> PyResult<u64> {
        let shape = array.shape();

        if shape.len() != 2 {
            return Err(PyValueError::new_err("Array must be 2D (Nx3/Nx4/Nx5)"));
        }

        let num_points = shape[0];
        let num_fields = shape[1];

        if num_points == 0 {
            return Err(PyValueError::new_err("Point cloud cannot be empty"));
        }
        if num_fields != 3 && num_fields != 4 && num_fields != 5 {
            return Err(PyValueError::new_err(
                "Array must have 3, 4, or 5 columns: [x, y, z], [x, y, z, t], or [x, y, z, intensity, t]",
            ));
        }

        // Build point cloud data.
        // Always emit Ouster-like fields: x,y,z (f32), intensity (f32), t (u32).
        // For Nx4, the 4th column is treated as t. Intensity is 0.
        let has_t = num_fields >= 4;
        let has_intensity = num_fields == 5;
        let point_step = 20;
        let mut data = Vec::with_capacity(num_points * point_step);

        for i in 0..num_points {
            // x, y, z
            for j in 0..3 {
                let field = ["x", "y", "z"][j];
                let value = point_component_to_f32(array[[i, j]].to_f64(), i, field)?;
                data.extend_from_slice(&value.to_le_bytes());
            }

            // intensity (f32)
            let intensity = if has_intensity {
                point_component_to_f32(array[[i, 3]].to_f64(), i, "intensity")?
            } else {
                0.0
            };
            data.extend_from_slice(&intensity.to_le_bytes());

            // t (u32)
            let t_val: u32 = if has_t {
                let idx = if has_intensity { 4 } else { 3 };
                point_timestamp_to_u32(array[[i, idx]].to_f64(), i)?
            } else {
                0
            };
            data.extend_from_slice(&t_val.to_le_bytes());
        }

        // Build fields
        let fields = vec![
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
        ];

        let cloud_msg = PointCloud2 {
            header: ros2_interfaces_jazzy_serde::std_msgs::msg::Header {
                stamp: ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time {
                    sec: timestamp_sec,
                    nanosec: timestamp_nanosec,
                },
                frame_id: "".to_string(),
            },
            height: 1,
            width: num_points as u32,
            fields,
            is_bigendian: false,
            point_step: point_step as u32,
            row_step: (num_points * point_step) as u32,
            data,
            is_dense: true,
        };

        let cloud_msg_internal: PointCloud2Msg =
            impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(cloud_msg);
        let mut sync_guard = self.sync.lock().unwrap();
        let sync = sync_guard
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Handle is closed"))?;
        let scan_id = sync.add_pointcloud(cloud_msg_internal);
        Ok(scan_id)
    }

    /// Add point cloud from a sequence of point sequences.
    fn add_pointcloud_from_sequence(
        &self,
        points: &Bound<'_, PySequence>,
        timestamp_sec: i32,
        timestamp_nanosec: u32,
    ) -> PyResult<u64> {
        let num_points = points.len()?;

        if num_points == 0 {
            return Err(PyValueError::new_err("Point cloud cannot be empty"));
        }

        // Check first point to determine format
        let first_point = points.get_item(0)?;
        let first_point_sequence = first_point.cast::<PySequence>().map_err(|_| {
            PyValueError::new_err("Each point must be a list, tuple, or other sequence")
        })?;
        let num_fields = first_point_sequence.len()?;

        if num_fields != 3 && num_fields != 4 && num_fields != 5 {
            return Err(PyValueError::new_err(
                "Each point must have 3, 4, or 5 values: [x, y, z], [x, y, z, t], or [x, y, z, intensity, t]",
            ));
        }

        let has_t = num_fields >= 4;
        let has_intensity = num_fields == 5;
        let point_step = 20;
        let mut data = Vec::with_capacity(num_points * point_step);

        for point_index in 0..num_points {
            let point = points.get_item(point_index)?;
            let point_sequence = point.cast::<PySequence>().map_err(|_| {
                PyValueError::new_err("Each point must be a list, tuple, or other sequence")
            })?;

            if point_sequence.len()? != num_fields {
                return Err(PyValueError::new_err(
                    "All points must have the same number of fields",
                ));
            }

            // x, y, z
            for i in 0..3 {
                let value: f64 = point_sequence.get_item(i)?.extract()?;
                let field = ["x", "y", "z"][i];
                let val = point_component_to_f32(value, point_index, field)?;
                data.extend_from_slice(&val.to_le_bytes());
            }

            // intensity
            let intensity: f32 = if has_intensity {
                let value: f64 = point_sequence.get_item(3)?.extract()?;
                point_component_to_f32(value, point_index, "intensity")?
            } else {
                0.0
            };
            data.extend_from_slice(&intensity.to_le_bytes());

            // t
            let t_val: u32 = if has_t {
                let idx = if has_intensity { 4 } else { 3 };
                let value: f64 = point_sequence.get_item(idx)?.extract()?;
                point_timestamp_to_u32(value, point_index)?
            } else {
                0
            };
            data.extend_from_slice(&t_val.to_le_bytes());
        }

        // Build fields
        let fields = vec![
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
        ];

        let cloud_msg = PointCloud2 {
            header: ros2_interfaces_jazzy_serde::std_msgs::msg::Header {
                stamp: ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time {
                    sec: timestamp_sec,
                    nanosec: timestamp_nanosec,
                },
                frame_id: "".to_string(),
            },
            height: 1,
            width: num_points as u32,
            fields,
            is_bigendian: false,
            point_step: point_step as u32,
            row_step: (num_points * point_step) as u32,
            data,
            is_dense: true,
        };

        let cloud_msg_internal: PointCloud2Msg =
            impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(cloud_msg);
        let mut sync_guard = self.sync.lock().unwrap();
        let sync = sync_guard
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Handle is closed"))?;
        let scan_id = sync.add_pointcloud(cloud_msg_internal);
        Ok(scan_id)
    }
}

#[pymodule]
fn pelorus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PelorusHandle>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

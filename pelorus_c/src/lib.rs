//! C FFI bindings for Pelorus

use std::ffi::{CStr, c_char, c_void};

use std::path::Path;
use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use anyhow::anyhow;
use log::{error, warn};
use pelorus_core::config::ast_to_config;
use pelorus_core::{ImuMsg, LidarVendor, LioConfig, OdomMsg, PrelioProcessedCloud};
use pelorus_reg::pipeline::{PelorusOutput, Pipeline};
use prelio::Prelio;
use ratslang;
use ros_pointcloud2::prelude::*;
use ros2_interfaces_jazzy_serde::sensor_msgs::msg::PointCloud2;
use ros2_interfaces_jazzy_serde::sensor_msgs::msg::PointField;
use std::thread;

const DEFAULT_CONFIG_STR: &str = include_str!("../../default.rl");

impl_pointcloud2_for_ros2_interfaces_jazzy_serde!();

struct ImuInitState {
    start_time: Option<f64>,
    buffer: Option<Vec<ImuMsg>>,
}

fn load_pelorus_config(config_path: Option<&str>) -> anyhow::Result<LioConfig> {
    let default_ast = ratslang::compile_code(DEFAULT_CONFIG_STR)
        .map_err(|e| anyhow!("Failed to compile embedded default config: {e}"))?;
    let user_ast = if let Some(path) = config_path {
        let path = Path::new(path);
        if !path.exists() {
            return Err(anyhow!("Configuration file not found: {}", path.display()));
        }
        ratslang::compile_file(&path.to_path_buf(), None, None)
            .map_err(|e| anyhow!("Failed to compile user config {}: {e}", path.display()))?
    } else {
        ratslang::compile_code("")?
    };

    ast_to_config(default_ast, user_ast).map_err(|e| anyhow!("Failed to build configuration: {e}"))
}

struct ProcessingThread {
    pipeline: Pipeline,
    prelio_receiver: mpsc::Receiver<PrelioProcessedCloud>,
    result_sender: mpsc::Sender<PelorusOutput>,
    callback_info: Arc<Mutex<Option<CallbackInfo>>>,
    ready_for_next: mpsc::Sender<()>,
    shutdown: Arc<AtomicBool>,
    // Latest odometry produced (for synchronous retrieval during init)
    last_output: Arc<Mutex<Option<PelorusOutput>>>,
    // Configuration
    config: LioConfig,
    // Extrinsic transform from Lidar to IMU for frame conversion
    ext_lidar_to_imu: nalgebra::Isometry3<f64>,
}

struct CallbackInfo {
    callback: unsafe extern "C" fn(*const pelorus_odometry, *mut c_void),
    user_data: *mut c_void,
}

unsafe impl Send for CallbackInfo {}
unsafe impl Sync for CallbackInfo {}

pub struct PelorusHandle {
    prelio: Arc<Mutex<Option<Prelio>>>,
    result_receiver: mpsc::Receiver<PelorusOutput>,
    callback_info: Arc<Mutex<Option<CallbackInfo>>>,
    // Latest odometry produced (used when IMU initialization not yet complete)
    last_output: Arc<Mutex<Option<PelorusOutput>>>,
    imu_init_cb: Option<Arc<Mutex<dyn FnMut(&[ImuMsg]) + Send + Sync>>>,
    imu_init_state: Arc<Mutex<ImuInitState>>,
    imu_predictor_init_time_sec: f64,
    _processing_thread: Option<thread::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    // Configuration
    config: LioConfig,
    // Extrinsic transform from Lidar to IMU for frame conversion
    ext_lidar_to_imu: nalgebra::Isometry3<f64>,
}

impl Drop for PelorusHandle {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);

        // Drop Prelio first (while the processing thread is still alive) so it stops producing.
        if let Ok(mut prelio_guard) = self.prelio.lock() {
            prelio_guard.take();
        }

        if let Some(handle) = self._processing_thread.take() {
            if let Err(e) = handle.join() {
                warn!("processing thread join failed: {e:?}");
            }
        }
    }
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct pelorus_create_options {
    pub config_path: *const c_char,
    pub device_specific_pointcloud2: i8,
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct pelorus_imu_msg {
    pub timestamp_sec: i32,
    pub timestamp_nanosec: u32,
    pub frame_id: *const c_char,
    pub orientation: [f64; 4], // [w, x, y, z]
    pub orientation_covariance: [f64; 9],
    pub angular_velocity: [f64; 3], // [x, y, z] rad/s
    pub angular_velocity_covariance: [f64; 9],
    pub linear_acceleration: [f64; 3], // [x, y, z] m/s²
    pub linear_acceleration_covariance: [f64; 9],
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct pelorus_point_field {
    pub name: *const c_char,
    pub offset: u32,
    pub datatype: u8,
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct pelorus_pointcloud2 {
    pub header_timestamp_sec: i32,
    pub header_timestamp_nanosec: u32,
    pub width: u32,
    pub fields: *const pelorus_point_field,
    pub point_step: u32,
    pub row_step: u32,
    pub data: *const u8,
    pub data_len: usize,
    pub is_dense: bool,
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct pelorus_point_xyzi {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32, // expected 0..255 in float
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct pelorus_odometry {
    pub timestamp_sec: i32,
    pub timestamp_nanosec: u32,

    // Pose (position + orientation)
    // FRAME: Pose is in Lidar World frame.
    pub position: [f64; 3],         // [x, y, z] in meters
    pub orientation: [f64; 4],      // quaternion [w, x, y, z]
    pub pose_covariance: [f64; 36], // 6x6 covariance matrix, in IMU World frame

    // Velocity (linear + angular)
    // FRAME: Velocities and covariances are in Lidar World frame.
    pub linear_velocity: [f64; 3],   // [vx, vy, vz] in m/s
    pub angular_velocity: [f64; 3],  // [wx, wy, wz] in rad/s
    pub twist_covariance: [f64; 36], // 6x6 covariance matrix, in IMU World frame

    // Scan id corresponding to the input pointcloud
    pub scan_id: u64,

    // Registered point cloud (optional)
    // FRAME: In Lidar World frame
    pub points: *mut pelorus_point_xyzi,
    pub num_points: usize,
}

// Status codes used by the C API. Mirrors `pelorus_status` in the public header.
#[allow(non_camel_case_types)]
#[repr(i32)]
pub enum PelorusStatus {
    OK = 0,
    NO_DATA = 1,
    ERROR = -1,
    CALLBACK_SET = -2,
    DISCONNECTED = -3,
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct pelorus_odom_msg {
    pub timestamp_sec: i32,
    pub timestamp_nanosec: u32,
    pub twist_linear: [f64; 3],      // [vx, vy, vz] m/s
    pub twist_angular: [f64; 3],     // [wx, wy, wz] rad/s
    pub twist_covariance: [f64; 36], // 6x6 covariance matrix
}

fn dispatch_output(
    output: PelorusOutput,
    callback_info: &Arc<Mutex<Option<CallbackInfo>>>,
    result_sender: &mpsc::Sender<PelorusOutput>,
    last_output: &Arc<Mutex<Option<PelorusOutput>>>,
    sent_count: &mut usize,
    config: &LioConfig,
    ext_lidar_to_imu: &nalgebra::Isometry3<f64>,
) {
    // Store latest output for synchronous queries during IMU init
    if let Ok(mut last) = last_output.lock() {
        *last = Some(output.clone());
    }

    // Check if callback is set
    let callback_info_opt = callback_info
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|info| (info.callback, info.user_data)));

    if let Some((callback, user_data)) = callback_info_opt {
        let c_odom = convert_to_c_odometry(&output, config, ext_lidar_to_imu);
        unsafe {
            callback(&c_odom, user_data);
        }
        unsafe {
            if !c_odom.points.is_null() {
                let points = Vec::from_raw_parts(
                    c_odom.points as *mut pelorus_point_xyzi,
                    c_odom.num_points,
                    c_odom.num_points,
                );
                drop(points);
            }
        }
    } else {
        *sent_count += 1;
        if let Err(e) = result_sender.send(output) {
            error!("failed to send odometry #{sent_count}: {e}");
        }
    }
}

fn processing_thread_main(thread: ProcessingThread) {
    let mut sent_count: usize = 0;

    if !thread.shutdown.load(Ordering::Relaxed) {
        if let Err(e) = thread.ready_for_next.send(()) {
            error!("failed to send initial ready signal: {e}");
        }
    }

    while let Ok(processed) = thread.prelio_receiver.recv() {
        if thread.shutdown.load(Ordering::Relaxed) {
            continue;
        }

        match thread
            .pipeline
            .register_processed_cloud(processed, Some(&thread.ready_for_next))
        {
            Some(output) => {
                dispatch_output(
                    output,
                    &thread.callback_info,
                    &thread.result_sender,
                    &thread.last_output,
                    &mut sent_count,
                    &thread.config,
                    &thread.ext_lidar_to_imu,
                );
            }
            None => {}
        }
    }
}

fn convert_to_c_odometry(
    odom: &PelorusOutput,
    _config: &LioConfig,
    ext_lidar_to_imu: &nalgebra::Isometry3<f64>,
) -> pelorus_odometry {
    let (pose, linvel, angvel, cloud) = (
        odom.pose,
        odom.pose.rotation.inverse() * odom.linear_velocity,
        ext_lidar_to_imu.rotation.inverse() * odom.angular_velocity,
        odom.cloud.clone(),
    );

    let (pvec, pcount) = if cloud.is_empty() {
        (ptr::null_mut(), 0)
    } else {
        // Convert to output point format (drop per-point timestamp)
        let mut points_vec: Vec<pelorus_point_xyzi> = Vec::with_capacity(cloud.len());
        for p in cloud.iter() {
            points_vec.push(pelorus_point_xyzi {
                x: p.x,
                y: p.y,
                z: p.z,
                intensity: p.intensity,
            });
        }
        let out = (points_vec.as_mut_ptr(), points_vec.len());
        std::mem::forget(points_vec);
        out
    };

    let q = pose.rotation.quaternion();

    pelorus_odometry {
        timestamp_sec: odom.timestamp.sec,
        timestamp_nanosec: odom.timestamp.nanosec,
        position: [pose.translation.x, pose.translation.y, pose.translation.z],
        orientation: [q.w, q.i, q.j, q.k],
        pose_covariance: odom.pose_covariance,
        linear_velocity: [linvel.x, linvel.y, linvel.z],
        angular_velocity: [angvel.x, angvel.y, angvel.z],
        twist_covariance: odom.twist_covariance,
        scan_id: odom.scan_id,
        points: pvec,
        num_points: pcount,
    }
}

/// Create a new Pelorus instance with full LIO pipeline
///
/// # Safety
/// This function is safe to call from C, but the returned handle must be freed with `pelorus_destroy`.
///
/// # Parameters
/// - `config_path`: Optional path to a `.rl` configuration file. Pass NULL to use default configuration.
///
/// # Returns
/// Opaque handle to Pelorus instance, or null on error
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pelorus_create(config_path: *const c_char) -> *mut PelorusHandle {
    // Default behavior: device-specific PointCloud2 interpretation.
    unsafe { pelorus_create_internal(config_path, true) }
}

/// Create a new Pelorus instance with optional creation-time flags.
///
/// # Safety
/// Same as `pelorus_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pelorus_create_with_options(
    options: *const pelorus_create_options,
) -> *mut PelorusHandle {
    if options.is_null() {
        return unsafe { pelorus_create_internal(ptr::null(), true) };
    }

    let opts = unsafe { &*options };
    let device_specific_pointcloud2 = match opts.device_specific_pointcloud2 {
        0 => false,
        1 => true,
        _ => true, // default
    };

    unsafe { pelorus_create_internal(opts.config_path, device_specific_pointcloud2) }
}

unsafe fn pelorus_create_internal(
    config_path: *const c_char,
    device_specific_pointcloud2: bool,
) -> *mut PelorusHandle {
    // Parse config path if provided
    let config_path_opt = if config_path.is_null() {
        None
    } else {
        match unsafe { CStr::from_ptr(config_path) }.to_str() {
            Ok(s) => Some(s),
            Err(_) => {
                error!("Invalid UTF-8 in config path");
                return ptr::null_mut();
            }
        }
    };

    // Load configuration
    let mut config = match load_pelorus_config(config_path_opt) {
        Ok(cfg) => cfg,
        Err(e) => {
            error!("Failed to load configuration: {}", e);
            return ptr::null_mut();
        }
    };

    if !device_specific_pointcloud2 {
        // For non-device-specific PointCloud2 inputs, select the generic parsing path.
        // (Internally this uses a specific vendor mode, but we keep the public API vendor-agnostic.)
        config.prelio_params.lidar_vendor = LidarVendor::Ouster;
    }

    let pipeline = Pipeline::new(&config);

    let imu_init_cb = Some(pipeline.build_init_callback());
    let imu_predictor_init_time_sec = config.prelio_node_params.imu_predictor_init_time_sec;
    let imu_init_state = Arc::new(Mutex::new(ImuInitState {
        start_time: None,
        buffer: Some(Vec::new()),
    }));

    // Create the ready_for_next channel for synchronization
    let (ready_tx, ready_rx) = mpsc::channel::<()>();

    // Pass the receiver to Prelio for proper synchronization
    let (prelio, prelio_receiver) = match pipeline.create_prelio(Some(ready_rx)) {
        Ok(parts) => parts,
        Err(e) => {
            error!("Failed to initialize Prelio: {}", e);
            return ptr::null_mut();
        }
    };

    let (result_tx, result_rx) = mpsc::channel();

    let callback_info = Arc::new(Mutex::new(None));

    // Shared last-output storage for quick synchronous lookup
    let last_output = Arc::new(Mutex::new(None));

    let shutdown = Arc::new(AtomicBool::new(false));

    // Get extrinsic transform from pipeline
    let ext_lidar_to_imu = pipeline.ext();

    let processing_thread = ProcessingThread {
        pipeline: pipeline,
        prelio_receiver,
        result_sender: result_tx,
        callback_info: callback_info.clone(),
        ready_for_next: ready_tx,
        shutdown: shutdown.clone(),
        last_output: last_output.clone(),
        config: config.clone(),
        ext_lidar_to_imu,
    };

    let thread_handle = thread::spawn(move || {
        processing_thread_main(processing_thread);
    });

    let handle = Box::new(PelorusHandle {
        prelio: Arc::new(Mutex::new(Some(prelio))),
        result_receiver: result_rx,
        callback_info,
        last_output: last_output.clone(),
        imu_init_cb,
        imu_init_state,
        imu_predictor_init_time_sec,
        _processing_thread: Some(thread_handle),
        shutdown,
        config: config.clone(),
        ext_lidar_to_imu,
    });

    Box::into_raw(handle)
}

/// Destroy a Pelorus instance
///
/// # Safety
/// The handle must be a valid pointer returned from `pelorus_create` and must not be used after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pelorus_destroy(handle: *mut PelorusHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

/// Add an IMU message to Pelorus
///
/// # Safety
/// - `handle` must be a valid pointer returned from `pelorus_create`
/// - `imu` must be a valid pointer to a pelorus_imu_msg
///
/// # Returns
/// 0 on success, -1 on error
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pelorus_add_imu(
    handle: *mut PelorusHandle,
    imu: *const pelorus_imu_msg,
) -> i32 {
    if handle.is_null() || imu.is_null() {
        return -1;
    }

    unsafe {
        let handle = &mut *handle;
        let imu_data = &*imu;

        // Convert C IMU to Rust IMU
        let frame_id = if imu_data.frame_id.is_null() {
            String::new()
        } else {
            match CStr::from_ptr(imu_data.frame_id).to_str() {
                Ok(s) => s.to_string(),
                Err(_) => return -1,
            }
        };

        let imu_msg = ImuMsg {
            stamp: ros_pointcloud2::ros::TimeMsg {
                sec: imu_data.timestamp_sec,
                nanosec: imu_data.timestamp_nanosec,
            },
            frame_id,
            orientation: imu_data.orientation,
            orientation_covariance: imu_data.orientation_covariance,
            angular_velocity: imu_data.angular_velocity,
            angular_velocity_covariance: imu_data.angular_velocity_covariance,
            linear_acceleration: imu_data.linear_acceleration,
            linear_acceleration_covariance: imu_data.linear_acceleration_covariance,
        };

        // Mirror `prelio_io`: buffer initial IMU window, initialize EKF once, then start feeding IMU into Prelio.
        let timestamp = prelio::timestamp_to_secs(&imu_msg.stamp);
        let mut init_buffer_to_apply: Option<Vec<ImuMsg>> = None;
        if let Some(_cb) = handle.imu_init_cb.as_ref() {
            if let Ok(mut init_state) = handle.imu_init_state.lock() {
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

                    let init_time = handle.imu_predictor_init_time_sec;
                    let delta = timestamp - init_state.start_time.unwrap_or(timestamp);

                    if init_time <= 0.0 {
                        if let Some(buf) = init_state.buffer.as_mut() {
                            buf.push(imu_msg.clone());
                        }
                        init_buffer_to_apply = init_state.buffer.take();
                    } else if delta < init_time {
                        if let Some(buf) = init_state.buffer.as_mut() {
                            buf.push(imu_msg);
                            return 0;
                        }
                    } else {
                        init_buffer_to_apply = init_state.buffer.take();
                    }
                }
            }
        }

        if let (Some(cb), Some(buffer)) = (handle.imu_init_cb.as_ref(), init_buffer_to_apply) {
            if !buffer.is_empty() {
                cb.lock().unwrap()(&buffer);
            }
        }

        // Process IMU (ignore filtered output for now)
        match handle.prelio.lock() {
            Ok(mut prelio) => {
                if let Some(prelio) = prelio.as_mut() {
                    prelio.add_imu(imu_msg);
                    0
                } else {
                    -1
                }
            }
            Err(_) => -1,
        }
    }
}

/// Add an odometry message to Pelorus
///
/// # Safety
/// - `handle` must be a valid pointer returned from `pelorus_create`
/// - `odom` must be a valid pointer to a pelorus_odom_msg
///
/// # Returns
/// 0 on success, -1 on error
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pelorus_add_odom(
    handle: *mut PelorusHandle,
    odom: *const pelorus_odom_msg,
) -> i32 {
    if handle.is_null() || odom.is_null() {
        return -1;
    }

    unsafe {
        let handle = &mut *handle;
        let odom_data = &*odom;

        let odom_msg = OdomMsg {
            twist_linear: nalgebra::Vector3::new(
                odom_data.twist_linear[0],
                odom_data.twist_linear[1],
                odom_data.twist_linear[2],
            ),
            twist_angular: nalgebra::Vector3::new(
                odom_data.twist_angular[0],
                odom_data.twist_angular[1],
                odom_data.twist_angular[2],
            ),
            twist_cov: odom_data.twist_covariance,
            stamp: ros_pointcloud2::ros::TimeMsg {
                sec: odom_data.timestamp_sec,
                nanosec: odom_data.timestamp_nanosec,
            },
        };

        match handle.prelio.lock() {
            Ok(mut prelio) => {
                if let Some(prelio) = prelio.as_mut() {
                    prelio.add_odom(odom_msg);
                    0
                } else {
                    -1
                }
            }
            Err(_) => -1,
        }
    }
}

/// Add a point cloud to Pelorus
///
/// # Safety
/// - `handle` must be a valid pointer returned from `pelorus_create`
/// - `cloud` must be a valid pointer to a pelorus_pointcloud2
///
/// # Returns
/// 0 on error, >0 scan ID on success
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pelorus_add_pointcloud(
    handle: *mut PelorusHandle,
    cloud: *const pelorus_pointcloud2,
) -> u64 {
    if handle.is_null() || cloud.is_null() {
        return 0;
    }

    unsafe {
        let handle = &mut *handle;
        let cloud_data = &*cloud;

        // Header frame id removed from public C API: always use empty string.
        let frame_id = String::new();

        // Expect a single non-null field descriptor (name must not be NULL)
        if cloud_data.fields.is_null() {
            return 0;
        }
        let f = &*cloud_data.fields;
        if f.name.is_null() {
            return 0;
        }
        let name = match CStr::from_ptr(f.name).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return 0,
        };
        let fields = vec![ros2_interfaces_jazzy_serde::sensor_msgs::msg::PointField {
            name,
            offset: f.offset,
            datatype: f.datatype,
            count: 1,
        }];

        // Copy data buffer
        let data = slice::from_raw_parts(cloud_data.data, cloud_data.data_len).to_vec();

        // Create PointCloud2 message (simplified: unstructured single-row cloud, little-endian)
        let cloud_msg = PointCloud2 {
            header: ros2_interfaces_jazzy_serde::std_msgs::msg::Header {
                stamp: ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time {
                    sec: cloud_data.header_timestamp_sec,
                    nanosec: cloud_data.header_timestamp_nanosec,
                },
                frame_id,
            },
            height: 1,
            width: cloud_data.width,
            fields,
            is_bigendian: false, // little-endian assumed
            point_step: cloud_data.point_step,
            row_step: cloud_data.row_step,
            data,
            is_dense: cloud_data.is_dense,
        };

        // Convert to PointCloud2Msg
        let cloud_msg_internal: PointCloud2Msg =
            impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(cloud_msg);

        match handle.prelio.lock() {
            Ok(mut prelio) => {
                if let Some(prelio) = prelio.as_mut() {
                    let scan_id = prelio.add_pointcloud(cloud_msg_internal);
                    scan_id
                } else {
                    0
                }
            }
            Err(_) => 0,
        }
    }
}

fn make_frame_id(frame_id: *const c_char) -> Option<String> {
    if frame_id.is_null() {
        return Some(String::new());
    }
    match unsafe { CStr::from_ptr(frame_id) }.to_str() {
        Ok(s) => Some(s.to_string()),
        Err(_) => None,
    }
}

fn build_cloud_xyzit_t(
    header_timestamp_sec: i32,
    header_timestamp_nanosec: u32,
    frame_id: String,
    num_points: usize,
    mut point_at: impl FnMut(usize) -> (f32, f32, f32, f32, u32),
) -> PointCloud2Msg {
    // Always emit x,y,z,intensity,t (t is nanoseconds offset).
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
                sec: header_timestamp_sec,
                nanosec: header_timestamp_nanosec,
            },
            frame_id,
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

    impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(cloud_msg)
}

/// Add raw points from a packed `float` array (Nx4): [x,y,z,t].
///
/// `t` is interpreted as nanoseconds offset relative to the scan header timestamp.
///
/// # Safety
/// - `handle` must be a valid pointer returned from `pelorus_create*`
/// - `xyzt` must point to at least `num_points*4` floats
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pelorus_add_points_xyzt_f32(
    handle: *mut PelorusHandle,
    header_timestamp_sec: i32,
    header_timestamp_nanosec: u32,
    frame_id: *const c_char,
    xyzt: *const f32,
    num_points: usize,
) -> u64 {
    if handle.is_null() || xyzt.is_null() {
        return 0;
    }
    if num_points == 0 {
        return 0;
    }

    // Convert Nx4 [x,y,z,t] to Nx5 [x,y,z,intensity,t] with intensity=0.0
    let xyzt_slice = unsafe { slice::from_raw_parts(xyzt, num_points * 4) };
    let mut xyzit: Vec<f32> = Vec::with_capacity(num_points * 5);
    for i in 0..num_points {
        xyzit.push(xyzt_slice[i * 4 + 0]);
        xyzit.push(xyzt_slice[i * 4 + 1]);
        xyzit.push(xyzt_slice[i * 4 + 2]);
        xyzit.push(0.0f32); // intensity
        xyzit.push(xyzt_slice[i * 4 + 3]);
    }

    // Delegate to the primary Nx5 implementation
    unsafe {
        pelorus_add_points_xyzit_f32(
            handle,
            header_timestamp_sec,
            header_timestamp_nanosec,
            frame_id,
            xyzit.as_ptr(),
            num_points,
        )
    }
}

/// Add raw points from a packed `float` array (Nx5): [x,y,z,intensity,t].
///
/// `t` is interpreted as nanoseconds offset relative to the scan header timestamp.
///
/// # Safety
/// - `handle` must be a valid pointer returned from `pelorus_create*`
/// - `xyzit` must point to at least `num_points*5` floats
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pelorus_add_points_xyzit_f32(
    handle: *mut PelorusHandle,
    header_timestamp_sec: i32,
    header_timestamp_nanosec: u32,
    frame_id: *const c_char,
    xyzit: *const f32,
    num_points: usize,
) -> u64 {
    if handle.is_null() || xyzit.is_null() {
        return 0;
    }
    if num_points == 0 {
        return 0;
    }

    let frame_id = match make_frame_id(frame_id) {
        Some(s) => s,
        None => return 0,
    };
    let xyzit = unsafe { slice::from_raw_parts(xyzit, num_points * 5) };

    let cloud_msg_internal = build_cloud_xyzit_t(
        header_timestamp_sec,
        header_timestamp_nanosec,
        frame_id,
        num_points,
        |i| {
            let x = xyzit[i * 5 + 0];
            let y = xyzit[i * 5 + 1];
            let z = xyzit[i * 5 + 2];
            let intensity = xyzit[i * 5 + 3];
            let t = xyzit[i * 5 + 4] as u32;
            (x, y, z, intensity, t)
        },
    );

    unsafe {
        let handle = &mut *handle;
        match handle.prelio.lock() {
            Ok(mut prelio) => {
                if let Some(prelio) = prelio.as_mut() {
                    let scan_id = prelio.add_pointcloud(cloud_msg_internal);
                    scan_id
                } else {
                    0
                }
            }
            Err(_) => 0,
        }
    }
}

/// Try to receive odometry from Pelorus (non-blocking)
///
/// This function retrieves the latest odometry estimate including:
/// - Position and orientation (pose)
/// - Linear and angular velocity (twist)
/// - Covariance matrices
/// - Registered point cloud
///
/// # Safety
/// - `handle` must be a valid pointer returned from `pelorus_create`
/// - `result` must be a valid pointer to write the result
///
/// # Returns
/// 1 if odometry was received, 0 if no data is available, -1 on error
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pelorus_try_recv(
    handle: *mut PelorusHandle,
    result: *mut pelorus_odometry,
) -> i32 {
    if handle.is_null() || result.is_null() {
        return PelorusStatus::ERROR as i32;
    }

    unsafe {
        let handle = &mut *handle;

        // Fail fast if a callback is installed.
        match handle.callback_info.lock() {
            Ok(cb_guard) => {
                if cb_guard.is_some() {
                    return PelorusStatus::CALLBACK_SET as i32;
                }
            }
            Err(_) => {
                return PelorusStatus::ERROR as i32;
            }
        }

        match handle.result_receiver.try_recv() {
            Ok(odom) => {
                let result_out = &mut *result;
                *result_out =
                    convert_to_c_odometry(&odom, &handle.config, &handle.ext_lidar_to_imu);
                PelorusStatus::OK as i32
            }
            Err(mpsc::TryRecvError::Empty) => PelorusStatus::NO_DATA as i32,
            Err(mpsc::TryRecvError::Disconnected) => PelorusStatus::DISCONNECTED as i32,
        }
    }
}

/// Blocking receive odometry from Pelorus (synchronous)
///
/// This function blocks until odometry data is available or an error occurs.
/// Use this for synchronous processing where you want to wait for each result.
///
/// # Safety
/// - `handle` must be a valid pointer returned from `pelorus_create`
/// - `result` must be a valid pointer to write the result
///
/// # Returns
/// 1 if odometry was received, -1 on error (e.g., channel disconnected)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pelorus_recv(
    handle: *mut PelorusHandle,
    result: *mut pelorus_odometry,
) -> i32 {
    if handle.is_null() || result.is_null() {
        return PelorusStatus::ERROR as i32;
    }

    unsafe {
        let handle = &mut *handle;

        // Fail fast if a callback is installed.
        match handle.callback_info.lock() {
            Ok(cb_guard) => {
                if cb_guard.is_some() {
                    return PelorusStatus::CALLBACK_SET as i32;
                }
            }
            Err(_) => {
                return PelorusStatus::ERROR as i32;
            }
        }

        // If IMU initialization hasn't completed yet (buffer still present), return last known odometry
        if let Ok(init_state) = handle.imu_init_state.lock() {
            if init_state.buffer.is_some() {
                if let Ok(last) = handle.last_output.lock() {
                    if let Some(odom) = last.as_ref() {
                        let result_out = &mut *result;
                        *result_out =
                            convert_to_c_odometry(odom, &handle.config, &handle.ext_lidar_to_imu);
                        // During IMU initialization there's no deskewed/registered cloud available.
                        // Clear any registered cloud so the caller can attach the raw input scan instead.
                        result_out.points = ptr::null_mut();
                        result_out.num_points = 0;
                        return PelorusStatus::OK as i32;
                    } else {
                        // No odometry available yet
                        return PelorusStatus::NO_DATA as i32;
                    }
                } else {
                    return PelorusStatus::ERROR as i32;
                }
            }
        }

        match handle.result_receiver.recv() {
            Ok(odom) => {
                let result_out = &mut *result;
                *result_out =
                    convert_to_c_odometry(&odom, &handle.config, &handle.ext_lidar_to_imu);
                PelorusStatus::OK as i32
            }
            Err(_) => PelorusStatus::DISCONNECTED as i32,
        }
    }
}

/// Blocking receive odometry for a specific scan_id
///
/// This function blocks until an odometry result with the requested `scan_id` is available.
/// Non-matching outputs are cached for subsequent calls.
///
/// # Safety
/// - `handle` must be a valid pointer returned from `pelorus_create`
/// - `result` must be a valid pointer to write the result
///
/// # Returns
/// 1 if odometry for `scan_id` was received, -1 on error

/// Set a callback for asynchronous odometry reception
///
/// When a callback is set, odometry results will be delivered via the callback
/// instead of being queued for pelorus_try_recv() or pelorus_recv().
/// The callback is invoked from a background thread.
///
/// # Safety
/// - `handle` must be a valid pointer returned from `pelorus_create`
/// - `callback` must be a valid function pointer or NULL
/// - `user_data` can be any pointer (including NULL) and will be passed to the callback
///
/// # Returns
/// 0 on success, -1 on error
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pelorus_set_callback(
    handle: *mut PelorusHandle,
    callback: Option<unsafe extern "C" fn(*const pelorus_odometry, *mut c_void)>,
    user_data: *mut c_void,
) -> i32 {
    if handle.is_null() {
        return PelorusStatus::ERROR as i32;
    }

    unsafe {
        let handle = &mut *handle;

        match handle.callback_info.lock() {
            Ok(mut guard) => {
                if let Some(cb) = callback {
                    // Drain any queued results while holding the callback lock so there's no race
                    // window where an output could be queued after we thought we drained.
                    while let Ok(_) = handle.result_receiver.try_recv() {}
                    *guard = Some(CallbackInfo {
                        callback: cb,
                        user_data,
                    });
                } else {
                    *guard = None;
                }
                PelorusStatus::OK as i32
            }
            Err(_) => PelorusStatus::ERROR as i32,
        }
    }
}

/// Free memory allocated for odometry result
///
/// # Safety
/// - `odom` must be a valid pointer to a pelorus_odometry that was filled by `pelorus_try_recv` or `pelorus_recv`
/// - This function should only be called once per odometry result
/// - Do NOT call this for odometry results received via callbacks (memory is freed automatically)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pelorus_free_odometry(odom: *mut pelorus_odometry) {
    if odom.is_null() {
        return;
    }

    unsafe {
        let odom_data = &mut *odom;
        if !odom_data.points.is_null() {
            // Reconstruct the Vec with same length and capacity (allocated with exact size)
            let points =
                Vec::from_raw_parts(odom_data.points, odom_data.num_points, odom_data.num_points);
            drop(points);
            odom_data.points = ptr::null_mut();
            odom_data.num_points = 0;
        }
    }
}

/// Get the version string of the library
///
/// # Returns
/// A pointer to a null-terminated string. The string is valid for the lifetime of the program.
#[unsafe(no_mangle)]
pub extern "C" fn pelorus_version() -> *const c_char {
    static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;
    use std::ptr;

    unsafe extern "C" fn dummy_cb(_odom: *const pelorus_odometry, _user: *mut c_void) {}

    #[test]
    fn test_recv_fails_when_callback_set() {
        unsafe {
            let h = pelorus_create(ptr::null());
            assert!(!h.is_null());

            let res = pelorus_set_callback(h, Some(dummy_cb), ptr::null_mut());
            assert_eq!(res, PelorusStatus::OK as i32);

            let mut odom = mem::MaybeUninit::<pelorus_odometry>::uninit();
            let r = pelorus_try_recv(h, odom.as_mut_ptr());
            assert_eq!(
                r,
                PelorusStatus::CALLBACK_SET as i32,
                "try_recv should fail with CALLBACK_SET when callback is set"
            );

            let r2 = pelorus_recv(h, odom.as_mut_ptr());
            assert_eq!(
                r2,
                PelorusStatus::CALLBACK_SET as i32,
                "recv should fail with CALLBACK_SET when callback is set"
            );

            // Clear callback and ensure we can destroy handle cleanly
            let res = pelorus_set_callback(h, None, ptr::null_mut());
            assert_eq!(res, PelorusStatus::OK as i32);
            pelorus_destroy(h);
        }
    }
}

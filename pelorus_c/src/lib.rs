//! C FFI bindings for Pelorus

use std::ffi::{CStr, c_char};
use std::path::Path;
use std::ptr;
use std::slice;
use std::sync::{Arc, Mutex, mpsc};

use anyhow::anyhow;
use pelorus_core::config::ast_to_config;
use pelorus_core::{ImuMsg, LioConfig, OdomMsg, PrelioProcessedCloud};
use pelorus_reg::pipeline::{PelorusOutput, Pipeline};
use prelio::Prelio;
use ratslang;
use ros_pointcloud2::prelude::*;
use ros2_interfaces_jazzy_serde::sensor_msgs::msg::PointCloud2;
use std::thread;

const DEFAULT_CONFIG_STR: &str = include_str!("../../default.rl");

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
}

pub struct PelorusHandle {
    prelio: Arc<Mutex<Prelio>>,
    result_receiver: mpsc::Receiver<PelorusOutput>,
    _processing_thread: Option<thread::JoinHandle<()>>,
}

#[repr(C)]
pub struct CImuMsg {
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

#[repr(C)]
pub struct CPointCloudHeader {
    pub timestamp_sec: i32,
    pub timestamp_nanosec: u32,
    pub frame_id: *const c_char,
}

#[repr(C)]
pub struct CPointField {
    pub name: *const c_char,
    pub offset: u32,
    pub datatype: u8,
    pub count: u32,
}

#[repr(C)]
pub struct CPointCloud2 {
    pub header_timestamp_sec: i32,
    pub header_timestamp_nanosec: u32,
    pub header_frame_id: *const c_char,
    pub height: u32,
    pub width: u32,
    pub fields: *const CPointField,
    pub fields_len: usize,
    pub is_bigendian: bool,
    pub point_step: u32,
    pub row_step: u32,
    pub data: *const u8,
    pub data_len: usize,
    pub is_dense: bool,
}

#[repr(C)]
pub struct CPointXYZI {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
}

#[repr(C)]
pub struct CPelorusOdometry {
    pub timestamp_sec: i32,
    pub timestamp_nanosec: u32,

    // Pose (position + orientation)
    pub position: [f64; 3],         // [x, y, z] in meters
    pub orientation: [f64; 4],      // quaternion [w, x, y, z]
    pub pose_covariance: [f64; 36], // 6x6 covariance matrix

    // Velocity (linear + angular)
    pub linear_velocity: [f64; 3],   // [vx, vy, vz] in m/s
    pub angular_velocity: [f64; 3],  // [wx, wy, wz] in rad/s
    pub twist_covariance: [f64; 36], // 6x6 covariance matrix

    // Registered point cloud (optional)
    pub points: *mut CPointXYZI,
    pub num_points: usize,
}

#[repr(C)]
pub struct COdomMsg {
    pub timestamp_sec: i32,
    pub timestamp_nanosec: u32,
    pub twist_linear: [f64; 3],      // [vx, vy, vz] m/s
    pub twist_angular: [f64; 3],     // [wx, wy, wz] rad/s
    pub twist_covariance: [f64; 36], // 6x6 covariance matrix
}

fn processing_thread_main(thread: ProcessingThread) {
    while let Ok(processed) = thread.prelio_receiver.recv() {
        if let Some(output) = thread.pipeline.register_processed_cloud(processed) {
            let _ = thread.result_sender.send(output);
        }
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
    // Parse config path if provided
    let config_path_opt = if config_path.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(config_path).to_str() {
                Ok(s) => Some(s),
                Err(_) => {
                    log::error!("Invalid UTF-8 in config path");
                    return ptr::null_mut();
                }
            }
        }
    };

    // Load configuration
    let config = match load_pelorus_config(config_path_opt) {
        Ok(cfg) => cfg,
        Err(e) => {
            log::error!("Failed to load configuration: {}", e);
            return ptr::null_mut();
        }
    };

    let pipeline = Pipeline::new(&config);

    let (prelio, prelio_receiver) = match pipeline.create_prelio(None) {
        Ok(parts) => parts,
        Err(e) => {
            log::error!("Failed to initialize Prelio: {}", e);
            return ptr::null_mut();
        }
    };

    let (result_tx, result_rx) = mpsc::channel();

    let processing_thread = ProcessingThread {
        pipeline: pipeline.clone(),
        prelio_receiver,
        result_sender: result_tx,
    };

    let thread_handle = thread::spawn(move || {
        processing_thread_main(processing_thread);
    });

    let handle = Box::new(PelorusHandle {
        prelio: Arc::new(Mutex::new(prelio)),
        result_receiver: result_rx,
        _processing_thread: Some(thread_handle),
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
/// - `imu` must be a valid pointer to a CImuMsg
///
/// # Returns
/// 0 on success, -1 on error
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pelorus_add_imu(handle: *mut PelorusHandle, imu: *const CImuMsg) -> i32 {
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

        // Process IMU (ignore filtered output for now)
        match handle.prelio.lock() {
            Ok(mut prelio) => {
                prelio.add_imu(imu_msg);
                0
            }
            Err(_) => -1,
        }
    }
}

/// Add an odometry message to Pelorus
///
/// # Safety
/// - `handle` must be a valid pointer returned from `pelorus_create`
/// - `odom` must be a valid pointer to a COdomMsg
///
/// # Returns
/// 0 on success, -1 on error
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pelorus_add_odom(
    handle: *mut PelorusHandle,
    odom: *const COdomMsg,
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
                prelio.add_odom(odom_msg);
                0
            }
            Err(_) => -1,
        }
    }
}

/// Add a point cloud to Pelorus
///
/// # Safety
/// - `handle` must be a valid pointer returned from `pelorus_create`
/// - `header` must be a valid pointer to a CPointCloudHeader
/// - `points` must be a valid pointer to an array of `num_points` floats in XYZI format
///
/// # Returns
/// 0 on success, -1 on error
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pelorus_add_pointcloud(
    handle: *mut PelorusHandle,
    cloud: *const CPointCloud2,
) -> i32 {
    if handle.is_null() || cloud.is_null() {
        return -1;
    }

    unsafe {
        let handle = &mut *handle;
        let cloud_data = &*cloud;

        let frame_id = if cloud_data.header_frame_id.is_null() {
            String::new()
        } else {
            match CStr::from_ptr(cloud_data.header_frame_id).to_str() {
                Ok(s) => s.to_string(),
                Err(_) => return -1,
            }
        };

        let fields_slice = slice::from_raw_parts(cloud_data.fields, cloud_data.fields_len);
        let fields: Vec<_> = fields_slice
            .iter()
            .map(|f| {
                let name = if f.name.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(f.name).to_str().unwrap_or("").to_string()
                };
                ros2_interfaces_jazzy_serde::sensor_msgs::msg::PointField {
                    name,
                    offset: f.offset,
                    datatype: f.datatype,
                    count: f.count,
                }
            })
            .collect();

        // Copy data buffer
        let data = slice::from_raw_parts(cloud_data.data, cloud_data.data_len).to_vec();

        // Create PointCloud2 message
        let cloud_msg = PointCloud2 {
            header: ros2_interfaces_jazzy_serde::std_msgs::msg::Header {
                stamp: ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time {
                    sec: cloud_data.header_timestamp_sec,
                    nanosec: cloud_data.header_timestamp_nanosec,
                },
                frame_id,
            },
            height: cloud_data.height,
            width: cloud_data.width,
            fields,
            is_bigendian: cloud_data.is_bigendian,
            point_step: cloud_data.point_step,
            row_step: cloud_data.row_step,
            data,
            is_dense: cloud_data.is_dense,
        };

        // Convert to PointCloud2Msg
        let cloud_msg_internal: PointCloud2Msg = cloud_msg.into();

        match handle.prelio.lock() {
            Ok(mut prelio) => {
                prelio.add_pointcloud(cloud_msg_internal);
                0
            }
            Err(_) => -1,
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
    result: *mut CPelorusOdometry,
) -> i32 {
    if handle.is_null() || result.is_null() {
        return -1;
    }

    unsafe {
        let handle = &mut *handle;

        match handle.result_receiver.try_recv() {
            Ok(odom) => {
                let result_out = &mut *result;

                result_out.timestamp_sec = odom.timestamp.sec;
                result_out.timestamp_nanosec = odom.timestamp.nanosec;

                result_out.position = [
                    odom.pose.translation.x,
                    odom.pose.translation.y,
                    odom.pose.translation.z,
                ];
                let q = odom.pose.rotation.quaternion();
                result_out.orientation = [q.w, q.i, q.j, q.k];
                result_out.pose_covariance = odom.pose_covariance;

                result_out.linear_velocity = [
                    odom.linear_velocity.x,
                    odom.linear_velocity.y,
                    odom.linear_velocity.z,
                ];
                result_out.angular_velocity = [
                    odom.angular_velocity.x,
                    odom.angular_velocity.y,
                    odom.angular_velocity.z,
                ];
                result_out.twist_covariance = odom.twist_covariance;

                let mut points_vec: Vec<CPointXYZI> = odom
                    .cloud
                    .iter()
                    .map(|p| CPointXYZI {
                        x: p.x,
                        y: p.y,
                        z: p.z,
                        intensity: p.intensity,
                    })
                    .collect();

                result_out.num_points = points_vec.len();
                result_out.points = points_vec.as_mut_ptr();
                std::mem::forget(points_vec);

                1
            }
            Err(mpsc::TryRecvError::Empty) => 0,
            Err(mpsc::TryRecvError::Disconnected) => -1,
        }
    }
}

/// Free memory allocated for odometry result
///
/// # Safety
/// - `odom` must be a valid pointer to a CPelorusOdometry that was filled by `pelorus_try_recv`
/// - This function should only be called once per odometry result
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pelorus_free_odometry(odom: *mut CPelorusOdometry) {
    if odom.is_null() {
        return;
    }

    unsafe {
        let odom_data = &mut *odom;
        if !odom_data.points.is_null() {
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

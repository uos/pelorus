use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;
use std::slice;

use log::error;
use pelorus::{Handle, Odometry as RsOdometry, Pcloud2Field};

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
    pub orientation: [f64; 4],
    pub orientation_covariance: [f64; 9],
    pub angular_velocity: [f64; 3],
    pub angular_velocity_covariance: [f64; 9],
    pub linear_acceleration: [f64; 3],
    pub linear_acceleration_covariance: [f64; 9],
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct pelorus_odom_msg {
    pub timestamp_sec: i32,
    pub timestamp_nanosec: u32,
    pub twist_linear: [f64; 3],
    pub twist_angular: [f64; 3],
    pub twist_covariance: [f64; 36],
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
    pub intensity: f32,
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct pelorus_odometry {
    pub timestamp_sec: i32,
    pub timestamp_nanosec: u32,
    pub position: [f64; 3],
    pub orientation: [f64; 4],
    pub pose_covariance: [f64; 36],
    pub linear_velocity: [f64; 3],
    pub angular_velocity: [f64; 3],
    pub twist_covariance: [f64; 36],
    pub scan_id: u64,
    pub points: *mut pelorus_point_xyzi,
    pub num_points: usize,
    pub child_frame_id: *mut c_char,
}

// Status codes used by the C API.
#[allow(non_camel_case_types)]
#[repr(i32)]
pub enum PelorusStatus {
    OK = 0,
    NO_DATA = 1,
    ERROR = -1,
    CALLBACK_SET = -2,
    DISCONNECTED = -3,
}

pub struct PelorusHandle {
    inner: Handle,
}

fn rs_to_c_odometry(odom: RsOdometry) -> pelorus_odometry {
    let child_frame_id = CString::new(odom.frame_id.as_str())
        .map(|s| s.into_raw())
        .unwrap_or(ptr::null_mut());

    let (pvec, pcount) = if odom.points.is_empty() {
        (ptr::null_mut(), 0)
    } else {
        let mut v: Vec<pelorus_point_xyzi> = odom
            .points
            .into_iter()
            .map(|p| pelorus_point_xyzi {
                x: p.x,
                y: p.y,
                z: p.z,
                intensity: p.intensity,
            })
            .collect();
        let ptr = v.as_mut_ptr();
        let len = v.len();
        std::mem::forget(v);
        (ptr, len)
    };

    pelorus_odometry {
        timestamp_sec: odom.timestamp_sec,
        timestamp_nanosec: odom.timestamp_nanosec,
        position: odom.position,
        orientation: odom.orientation,
        pose_covariance: odom.pose_covariance,
        linear_velocity: odom.linear_velocity,
        angular_velocity: odom.angular_velocity,
        twist_covariance: odom.twist_covariance,
        scan_id: odom.scan_id,
        points: pvec,
        num_points: pcount,
        child_frame_id,
    }
}

unsafe fn free_odometry_fields(odom: &mut pelorus_odometry) {
    if !odom.points.is_null() {
        unsafe {
            drop(Vec::from_raw_parts(
                odom.points,
                odom.num_points,
                odom.num_points,
            ));
        }
        odom.points = ptr::null_mut();
        odom.num_points = 0;
    }
    if !odom.child_frame_id.is_null() {
        unsafe {
            drop(CString::from_raw(odom.child_frame_id));
        }
        odom.child_frame_id = ptr::null_mut();
    }
}

/// Wraps `*mut c_void` so it can cross thread boundaries.
/// The C caller is responsible for ensuring the pointer remains valid.
struct SendSyncPtr(*mut c_void);
unsafe impl Send for SendSyncPtr {}
unsafe impl Sync for SendSyncPtr {}

impl SendSyncPtr {
    fn as_ptr(&self) -> *mut c_void {
        self.0
    }
}

fn map_error(e: &pelorus::Error) -> i32 {
    use pelorus::Error;
    match e {
        Error::CallbackActive => PelorusStatus::CALLBACK_SET as i32,
        Error::Disconnected => PelorusStatus::DISCONNECTED as i32,
        Error::NoData => PelorusStatus::NO_DATA as i32,
        Error::Other(_) => PelorusStatus::ERROR as i32,
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
    match Handle::new(config_path_opt) {
        Ok(inner) => Box::into_raw(Box::new(PelorusHandle { inner })),
        Err(e) => {
            error!("Failed to create Pelorus: {e}");
            ptr::null_mut()
        }
    }
}

/// Create a new Pelorus instance with optional creation-time flags.
///
/// # Safety
/// Same as `pelorus_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pelorus_create_with_options(
    options: *const pelorus_create_options,
) -> *mut PelorusHandle {
    let (config_path_opt, device_specific) = if options.is_null() {
        (None, true)
    } else {
        let opts = unsafe { &*options };
        let cp = if opts.config_path.is_null() {
            None
        } else {
            match unsafe { CStr::from_ptr(opts.config_path) }.to_str() {
                Ok(s) => Some(s),
                Err(_) => {
                    error!("Invalid UTF-8 in config path");
                    return ptr::null_mut();
                }
            }
        };
        (cp, opts.device_specific_pointcloud2 != 0)
    };
    match Handle::new_with_options(config_path_opt, device_specific) {
        Ok(inner) => Box::into_raw(Box::new(PelorusHandle { inner })),
        Err(e) => {
            error!("Failed to create Pelorus: {e}");
            ptr::null_mut()
        }
    }
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
        return PelorusStatus::ERROR as i32;
    }
    let handle = unsafe { &*handle };
    let imu = unsafe { &*imu };
    match handle.inner.add_imu(
        imu.timestamp_sec,
        imu.timestamp_nanosec,
        imu.orientation,
        imu.orientation_covariance,
        imu.angular_velocity,
        imu.angular_velocity_covariance,
        imu.linear_acceleration,
        imu.linear_acceleration_covariance,
    ) {
        Ok(()) => 0,
        Err(e) => {
            error!("pelorus_add_imu: {e}");
            map_error(&e)
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
        return PelorusStatus::ERROR as i32;
    }
    let handle = unsafe { &*handle };
    let odom = unsafe { &*odom };
    match handle.inner.add_odom(
        odom.timestamp_sec,
        odom.timestamp_nanosec,
        odom.twist_linear,
        odom.twist_angular,
        odom.twist_covariance,
    ) {
        Ok(()) => 0,
        Err(e) => {
            error!("pelorus_add_odom: {e}");
            map_error(&e)
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
    let handle = unsafe { &*handle };
    let cloud = unsafe { &*cloud };
    if cloud.fields.is_null() || cloud.data.is_null() {
        return 0;
    }
    let f = unsafe { &*cloud.fields };
    if f.name.is_null() {
        return 0;
    }
    let name = match unsafe { CStr::from_ptr(f.name) }.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return 0,
    };
    let data = unsafe { slice::from_raw_parts(cloud.data, cloud.data_len) };
    match handle.inner.add_pointcloud_raw(
        cloud.header_timestamp_sec,
        cloud.header_timestamp_nanosec,
        cloud.width,
        &[Pcloud2Field {
            name,
            offset: f.offset,
            datatype: f.datatype,
        }],
        cloud.point_step,
        cloud.row_step,
        data,
        cloud.is_dense,
    ) {
        Ok(id) => id,
        Err(e) => {
            error!("pelorus_add_pointcloud: {e}");
            0
        }
    }
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
    _frame_id: *const c_char,
    xyzt: *const f32,
    num_points: usize,
) -> u64 {
    if handle.is_null() || xyzt.is_null() || num_points == 0 {
        return 0;
    }
    let handle = unsafe { &*handle };
    let s = unsafe { slice::from_raw_parts(xyzt, num_points * 4) };
    let points: Vec<[f32; 4]> = (0..num_points)
        .map(|i| [s[i * 4], s[i * 4 + 1], s[i * 4 + 2], s[i * 4 + 3]])
        .collect();
    match handle
        .inner
        .add_points_xyzt(header_timestamp_sec, header_timestamp_nanosec, &points)
    {
        Ok(id) => id,
        Err(e) => {
            error!("pelorus_add_points_xyzt_f32: {e}");
            0
        }
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
    _frame_id: *const c_char,
    xyzit: *const f32,
    num_points: usize,
) -> u64 {
    if handle.is_null() || xyzit.is_null() || num_points == 0 {
        return 0;
    }
    let handle = unsafe { &*handle };
    let s = unsafe { slice::from_raw_parts(xyzit, num_points * 5) };
    let points: Vec<[f32; 5]> = (0..num_points)
        .map(|i| {
            [
                s[i * 5],
                s[i * 5 + 1],
                s[i * 5 + 2],
                s[i * 5 + 3],
                s[i * 5 + 4],
            ]
        })
        .collect();
    match handle
        .inner
        .add_points_xyzit(header_timestamp_sec, header_timestamp_nanosec, &points)
    {
        Ok(id) => id,
        Err(e) => {
            error!("pelorus_add_points_xyzit_f32: {e}");
            0
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
    let handle = unsafe { &*handle };
    match handle.inner.try_recv() {
        Ok(Some(odom)) => {
            unsafe {
                *result = rs_to_c_odometry(odom);
            }
            PelorusStatus::OK as i32
        }
        Ok(None) => PelorusStatus::NO_DATA as i32,
        Err(e) => map_error(&e),
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
    let handle = unsafe { &*handle };
    match handle.inner.recv() {
        Ok(odom) => {
            unsafe {
                *result = rs_to_c_odometry(odom);
            }
            PelorusStatus::OK as i32
        }
        Err(e) => map_error(&e),
    }
}

/// Callback function type for asynchronous odometry reception.
///
/// `odom` is valid only for the duration of the callback.
/// `user_data` is the pointer passed to `pelorus_set_callback`.
pub type PelorusOdometryCallback =
    Option<unsafe extern "C" fn(*const pelorus_odometry, *mut c_void)>;

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
    callback: PelorusOdometryCallback,
    user_data: *mut c_void,
) -> i32 {
    if handle.is_null() {
        return PelorusStatus::ERROR as i32;
    }
    let handle = unsafe { &mut *handle };
    if let Some(cb) = callback {
        let ud = SendSyncPtr(user_data);
        handle.inner.set_callback(move |odom: RsOdometry| {
            let mut c_odom = rs_to_c_odometry(odom);
            unsafe {
                cb(&c_odom, ud.as_ptr());
            }
            unsafe {
                free_odometry_fields(&mut c_odom);
            }
        });
    } else {
        handle.inner.clear_callback();
    }
    PelorusStatus::OK as i32
}

/// Free memory allocated for odometry result
///
/// # Safety
/// - `odom` must be a valid pointer to a pelorus_odometry that was filled by `pelorus_try_recv` or `pelorus_recv`
/// - This function should only be called once per odometry result
/// - Do NOT call this for odometry results received via callbacks (memory is freed automatically)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pelorus_free_odometry(odom: *mut pelorus_odometry) {
    if !odom.is_null() {
        unsafe {
            free_odometry_fields(&mut *odom);
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

            let res = pelorus_set_callback(h, None, ptr::null_mut());
            assert_eq!(res, PelorusStatus::OK as i32);
            pelorus_destroy(h);
        }
    }
}

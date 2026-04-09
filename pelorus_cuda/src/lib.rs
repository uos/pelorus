#![cfg(feature = "cuda_available")]

use std::{fmt::Debug, sync::Arc};

use ffi::Point;
use pelorus_core::PointXYZIT;
use ros_pointcloud2::points::PointXYZINormal;

mod ffi;

pub use ffi::{Matrix3, NeighborType, Pose, Quaternion, Vector3};

#[derive(Debug)]
struct Inner {
    raw: *mut std::os::raw::c_void,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // This runs exactly once when the last Arc is dropped.
        unsafe {
            ffi::cupcl_destroy_stream(self.raw);
        }
    }
}

pub struct Stream {
    stream: Arc<Inner>,
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    pub fn new() -> Self {
        let raw = unsafe { ffi::cupcl_create_stream() };
        Self {
            stream: Arc::new(Inner { raw }),
        }
    }

    pub fn sync(&self) {
        unsafe {
            ffi::cupcl_sync_stream(self.stream.raw);
        }
    }
}

pub fn is_available() -> bool {
    unsafe { ffi::cupcl_is_available() }
}

pub fn get_cuda_versions() -> ffi::CudaVersion {
    unsafe { ffi::get_cuda_versions() }
}

pub fn get_cuda_device_name() -> Option<String> {
    let mut out = [0 as std::os::raw::c_char; 256];
    let ok = unsafe { ffi::get_cuda_device_name(out.as_mut_ptr(), out.len()) };
    if !ok {
        return None;
    }

    let name = unsafe { std::ffi::CStr::from_ptr(out.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    if name.is_empty() { None } else { Some(name) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaSemanticVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl std::fmt::Display for CudaSemanticVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl CudaSemanticVersion {
    fn from_cuda_encoded(version: i32) -> Option<Self> {
        if version <= 0 {
            return None;
        }
        let version = version as u32;
        Some(Self {
            major: version / 1000,
            minor: (version % 1000) / 10,
            patch: version % 10,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaMetadata {
    pub device_name: String,
    pub runtime_version: Option<CudaSemanticVersion>,
    pub driver_version: Option<CudaSemanticVersion>,
}

pub fn get_cuda_metadata() -> Option<CudaMetadata> {
    let versions = get_cuda_versions();
    let device_name = get_cuda_device_name()?;
    Some(CudaMetadata {
        device_name,
        runtime_version: CudaSemanticVersion::from_cuda_encoded(versions.runtime),
        driver_version: CudaSemanticVersion::from_cuda_encoded(versions.driver),
    })
}

// GPU Timing state
static TIMING_STATE: std::sync::OnceLock<std::sync::Mutex<CudaTimingState>> =
    std::sync::OnceLock::new();

struct CudaTimingState {
    planes_start: *mut std::os::raw::c_void,
    planes_stop: *mut std::os::raw::c_void,
    add_to_map_start: *mut std::os::raw::c_void,
    add_to_map_stop: *mut std::os::raw::c_void,
    pending_planes: std::sync::atomic::AtomicBool,
    pending_add_to_map: std::sync::atomic::AtomicBool,
}

unsafe impl Send for CudaTimingState {}
unsafe impl Sync for CudaTimingState {}

impl Drop for CudaTimingState {
    fn drop(&mut self) {
        unsafe {
            if !self.planes_start.is_null() {
                ffi::cupcl_event_destroy(self.planes_start);
            }
            if !self.planes_stop.is_null() {
                ffi::cupcl_event_destroy(self.planes_stop);
            }
            if !self.add_to_map_start.is_null() {
                ffi::cupcl_event_destroy(self.add_to_map_start);
            }
            if !self.add_to_map_stop.is_null() {
                ffi::cupcl_event_destroy(self.add_to_map_stop);
            }
        }
    }
}

fn get_timing_state() -> &'static std::sync::Mutex<CudaTimingState> {
    TIMING_STATE.get_or_init(|| {
        std::sync::Mutex::new(CudaTimingState {
            planes_start: unsafe { ffi::cupcl_event_create() },
            planes_stop: unsafe { ffi::cupcl_event_create() },
            add_to_map_start: unsafe { ffi::cupcl_event_create() },
            add_to_map_stop: unsafe { ffi::cupcl_event_create() },
            pending_planes: std::sync::atomic::AtomicBool::new(false),
            pending_add_to_map: std::sync::atomic::AtomicBool::new(false),
        })
    })
}

/// Try to retrieve GPU timing measurements from previous operations.
/// Returns (planes_ms, add_to_map_ms) if timing results are available.
/// This is non-blocking - returns None if events are not ready.
pub fn try_get_gpu_timings() -> Option<(f64, f64)> {
    let state = get_timing_state().lock().unwrap();

    let mut planes_ms = None;
    let mut add_to_map_ms = None;

    // Check planes timing
    if state
        .pending_planes
        .load(std::sync::atomic::Ordering::Acquire)
    {
        let ready = unsafe { ffi::cupcl_event_query(state.planes_stop) };
        if ready != 0 {
            let ms = unsafe { ffi::cupcl_event_elapsed(state.planes_start, state.planes_stop) };
            planes_ms = Some(ms as f64);
            state
                .pending_planes
                .store(false, std::sync::atomic::Ordering::Release);
        }
    }

    // Check add_to_map timing
    if state
        .pending_add_to_map
        .load(std::sync::atomic::Ordering::Acquire)
    {
        let ready = unsafe { ffi::cupcl_event_query(state.add_to_map_stop) };
        if ready != 0 {
            let ms =
                unsafe { ffi::cupcl_event_elapsed(state.add_to_map_start, state.add_to_map_stop) };
            add_to_map_ms = Some(ms as f64);
            state
                .pending_add_to_map
                .store(false, std::sync::atomic::Ordering::Release);
        }
    }

    match (planes_ms, add_to_map_ms) {
        (Some(p), Some(a)) => Some((p, a)),
        (Some(p), None) => Some((p, 0.0)),
        (None, Some(a)) => Some((0.0, a)),
        (None, None) => None,
    }
}

#[derive(Debug, Clone)]
pub struct CudaBuffer {
    pub gpu_ptr: *mut ::std::os::raw::c_void,
    pub element_size: usize,
    pub n: usize,
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            ffi::cupcl_free_buffer(self.gpu_ptr);
        }
    }
}

unsafe impl Send for CudaBuffer {}

impl CudaBuffer {
    pub fn new(stream: &Stream, element_size: usize, n: usize) -> Self {
        let my_stream = stream.stream.raw;
        let gpu_ptr =
            unsafe { ffi::cupcl_create_empty_buffer(my_stream, element_size as u32, n as u32) };
        Self {
            gpu_ptr,
            element_size,
            n,
        }
    }

    pub fn clone(&self, stream: &Stream) -> Self {
        let new = Self::new(stream, self.element_size, self.n);
        let my_stream = stream.stream.raw;
        unsafe {
            ffi::cupcl_copy_buffer(
                my_stream,
                self.gpu_ptr as *mut ::std::os::raw::c_void,
                new.gpu_ptr as *mut ::std::os::raw::c_void,
                self.element_size as u32,
                self.n as u32,
            )
        };
        new
    }

    pub fn upload<T>(&mut self, stream: &Stream, buffer: &[T]) {
        assert!(size_of::<T>() == self.element_size, "Wrong alignment of T");
        assert!(
            buffer.len() <= self.n,
            "Source buffer is larger than destination GPU buffer"
        );
        let my_stream = stream.stream.raw;
        unsafe {
            ffi::cupcl_upload_buffer(
                my_stream,
                self.gpu_ptr,
                buffer.as_ptr() as *mut ::std::os::raw::c_void,
                self.element_size as u32,
                buffer.len() as u32,
            )
        };
        self.n = buffer.len();
    }

    pub fn download<T: Default + Clone>(&self, stream: &Stream) -> Vec<T> {
        assert!(size_of::<T>() == self.element_size, "Wrong alignment of T");
        let mut v = vec![T::default(); self.n];
        let my_stream = stream.stream.raw;
        unsafe {
            ffi::cupcl_download_buffer(
                my_stream,
                v.as_mut_ptr() as *mut ::std::os::raw::c_void,
                self.gpu_ptr,
                self.element_size as u32,
                self.n as u32,
            )
        }

        v
    }

    pub fn download_into<T: Default + Clone>(&self, stream: &Stream, dst: &mut [T]) {
        assert!(size_of::<T>() == self.element_size, "Wrong alignment of T");
        assert!(
            dst.len() <= self.n,
            "Destination buffer on CPU is too small"
        );
        let my_stream = stream.stream.raw;
        unsafe {
            ffi::cupcl_download_buffer(
                my_stream,
                dst.as_mut_ptr() as *mut ::std::os::raw::c_void,
                self.gpu_ptr,
                self.element_size as u32,
                dst.len() as u32,
            )
        }
    }

    pub fn from_vec<T>(stream: &Stream, vec: Vec<T>, element_size: usize) -> Self {
        let my_stream = stream.stream.raw;
        let gpu_ptr = unsafe {
            ffi::cupcl_create_buffer(
                my_stream,
                vec.as_ptr() as *mut ::std::os::raw::c_void,
                element_size as u32,
                vec.len() as u32,
            )
        };
        Self {
            gpu_ptr,
            element_size,
            n: vec.len(),
        }
    }

    pub fn from_cloud(stream: &Stream, vec: &[PointXYZIT]) -> Self {
        let element_size = size_of::<PointXYZIT>();
        let my_stream = stream.stream.raw;
        let gpu_ptr = unsafe {
            ffi::cupcl_create_buffer(
                my_stream,
                vec.as_ptr() as *mut ::std::os::raw::c_void,
                element_size as u32,
                vec.len() as u32,
            )
        };
        Self {
            gpu_ptr,
            element_size,
            n: vec.len(),
        }
    }

    pub fn into_cloud(&self, stream: &Stream) -> Vec<PointXYZIT> {
        let cpu_buff = self.download(stream);
        stream.sync();
        cpu_buff
    }
}

pub fn get_map() -> Vec<nalgebra::Point3<f32>> {
    let mut out = vec![PointXYZIT::default(); unsafe { ffi::get_map_capacity() as usize }];
    let mut actual_size: u32 = 0;

    unsafe {
        ffi::get_map(out.as_mut_ptr() as *mut Point, &mut actual_size as *mut u32);
        out.set_len(actual_size as usize);
    };

    out.iter()
        .map(|p| nalgebra::Point3::new(p.x, p.y, p.z))
        .collect()
}

pub fn init_map(
    stream: &Stream,
    capacity: usize,
    voxel_size: f32,
    sensitivity: f32,
    min_dist: f32,
    nearby_seach_strategy: NeighborType,
) {
    assert!(capacity > 0, "No capacity");
    let capacity = capacity.next_power_of_two();
    assert!(
        (capacity & (capacity - 1)) == 0,
        "Capacity is not a power of 2"
    );

    // Register cleanup hook on first init
    ensure_cleanup_registered();

    let my_stream = stream.stream.raw;
    unsafe {
        ffi::init_map(
            my_stream,
            capacity as u32,
            sensitivity,
            min_dist,
            voxel_size,
            nearby_seach_strategy.into(),
        )
    }
}

pub fn add_to_map(
    stream: &Stream,
    cloud: &CudaBuffer,
    ekf_pose_t: &Vector3,
    ekf_pose_r: &Quaternion,
) {
    let pose = Pose {
        position: *ekf_pose_t,
        orientation: *ekf_pose_r,
    };
    let my_stream = stream.stream.raw;
    let timing_state = get_timing_state().lock().unwrap();
    unsafe {
        ffi::cupcl_event_record(timing_state.add_to_map_start, my_stream);
        ffi::add_to_map(
            my_stream,
            cloud.gpu_ptr as *mut ffi::Point,
            cloud.n as u32,
            pose,
        );
        ffi::cupcl_event_record(timing_state.add_to_map_stop, my_stream);
    }
    timing_state
        .pending_add_to_map
        .store(true, std::sync::atomic::Ordering::Release);
}
pub fn get_map_size() -> u32 {
    unsafe { ffi::get_map_size() }
}

pub fn cleanup_map() {
    unsafe { ffi::cleanup_map() }
    // Clean up timing events
    if let Some(state) = TIMING_STATE.get() {
        let mut timing = state.lock().unwrap();
        unsafe {
            if !timing.planes_start.is_null() {
                ffi::cupcl_event_destroy(timing.planes_start);
                timing.planes_start = std::ptr::null_mut();
            }
            if !timing.planes_stop.is_null() {
                ffi::cupcl_event_destroy(timing.planes_stop);
                timing.planes_stop = std::ptr::null_mut();
            }
            if !timing.add_to_map_start.is_null() {
                ffi::cupcl_event_destroy(timing.add_to_map_start);
                timing.add_to_map_start = std::ptr::null_mut();
            }
            if !timing.add_to_map_stop.is_null() {
                ffi::cupcl_event_destroy(timing.add_to_map_stop);
                timing.add_to_map_stop = std::ptr::null_mut();
            }
        }
    }
}

// Global guard to ensure cleanup happens before CUDA teardown
static CLEANUP_GUARD: std::sync::Once = std::sync::Once::new();

fn ensure_cleanup_registered() {
    CLEANUP_GUARD.call_once(|| {
        // Register cleanup to run at program exit, before global destructors
        extern "C" fn cleanup_at_exit() {
            unsafe { ffi::cleanup_map() }
        }
        unsafe {
            libc::atexit(cleanup_at_exit);
        }
    });
}

pub fn planes_cuda(
    stream: &Stream,
    cloud: &CudaBuffer,
    ekf_pose_t: &Vector3,
    ekf_pose_r: &Quaternion,
    ekf_imu_offset_t: &Vector3,
    ekf_imu_offset_r: &Quaternion,
    out_buff: &mut Vec<PointXYZINormal>,
    out_buff_gpu: &mut CudaBuffer,
) {
    assert!(out_buff.len() >= cloud.n);

    let lidar_to_imu = Pose {
        position: *ekf_imu_offset_t,
        orientation: *ekf_imu_offset_r,
    };
    let pose = Pose {
        position: *ekf_pose_t,
        orientation: *ekf_pose_r,
    };

    let _ = const {
        assert!(size_of::<ffi::Point>() == size_of::<PointXYZIT>());
        assert!(size_of::<ffi::PointNormal>() == size_of::<PointXYZINormal>());
    };

    let size = {
        let my_stream = stream.stream.raw;
        let timing_state = get_timing_state().lock().unwrap();
        unsafe {
            ffi::cupcl_event_record(timing_state.planes_start, my_stream);
            let size = ffi::planes(
                my_stream,
                cloud.gpu_ptr as *mut ffi::Point,
                cloud.n as u32,
                lidar_to_imu,
                pose,
                out_buff_gpu.gpu_ptr as *mut ffi::PointNormal,
            );
            ffi::cupcl_event_record(timing_state.planes_stop, my_stream);
            timing_state
                .pending_planes
                .store(true, std::sync::atomic::Ordering::Release);
            size
        }
    };
    let size = (size as usize).min(out_buff.len()).min(out_buff_gpu.n);

    // copy to cpu side
    out_buff_gpu.download_into(stream, out_buff);
    unsafe { out_buff.set_len(size) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloud_gpu_roundtrip() {
        let stream = Stream::new();
        let cloud = vec![
            PointXYZIT {
                x: 1.0,
                y: 0.0,
                z: 1.5,
                intensity: 0.0,
                timestamp: 2.0,
            },
            PointXYZIT {
                x: 2.0,
                y: 0.0,
                z: 1.5,
                intensity: 0.0,
                timestamp: 3.0,
            },
            PointXYZIT {
                x: 3.0,
                y: 0.0,
                z: 1.5,
                intensity: 0.0,
                timestamp: 4.0,
            },
        ];
        let gpu = CudaBuffer::from_cloud(&stream, &cloud);
        let cpu = gpu.into_cloud(&stream);
        for i in 0..cloud.len() {
            assert_eq!(cpu[i], cloud[i]);
        }
    }
}

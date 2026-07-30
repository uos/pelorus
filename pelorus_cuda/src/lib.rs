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

/// Try to retrieve GPU timing measurements from previous operations.
/// Returns (planes_ms, add_to_map_ms) if timing results are available.
/// This is non-blocking - returns None if events are not ready.
pub fn try_get_gpu_timings() -> Option<(f64, f64)> {
    // Disabled: CUDA event timing can hit invalid-resource-handle errors
    // under some runtime/context combinations. Keep algorithm behavior
    // unchanged by reporting no external timing data.
    None
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
    scan_id: usize,
) {
    let pose = Pose {
        position: *ekf_pose_t,
        orientation: *ekf_pose_r,
    };
    let my_stream = stream.stream.raw;
    unsafe {
        ffi::add_to_map(
            my_stream,
            cloud.gpu_ptr as *mut ffi::Point,
            cloud.n as u32,
            pose,
            scan_id as u32,
        );
    }
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

#[derive(Debug, Clone, Copy, Default)]
pub struct CudaDownsampleTimings {
    pub total_ms: f64,
    pub upload_ms: f64,
    pub key_build_ms: f64,
    pub keyed_sort_ms: f64,
    pub run_length_encode_ms: f64,
    pub run_starts_scan_ms: f64,
    pub unique_count_sync_ms: f64,
    pub medoid_reduce_ms: f64,
    pub timestamp_key_build_ms: f64,
    pub timestamp_sort_ms: f64,
    pub download_ms: f64,
}

fn downsample_timings_from_ffi(
    ffi_timings: ffi::CudaDownsampleTimings,
    total_ms: f64,
) -> CudaDownsampleTimings {
    CudaDownsampleTimings {
        total_ms,
        upload_ms: ffi_timings.upload_ms,
        key_build_ms: ffi_timings.key_build_ms,
        keyed_sort_ms: ffi_timings.keyed_sort_ms,
        run_length_encode_ms: ffi_timings.run_length_encode_ms,
        run_starts_scan_ms: ffi_timings.run_starts_scan_ms,
        unique_count_sync_ms: ffi_timings.unique_count_sync_ms,
        medoid_reduce_ms: ffi_timings.medoid_reduce_ms,
        timestamp_key_build_ms: ffi_timings.timestamp_key_build_ms,
        timestamp_sort_ms: ffi_timings.timestamp_sort_ms,
        download_ms: ffi_timings.download_ms,
    }
}

pub fn downsample_points_cuda(
    stream: &Stream,
    filtered_cloud: &[PointXYZIT],
    point_skip: usize,
    voxel_size: f32,
) -> (Vec<PointXYZIT>, CudaDownsampleTimings) {
    const {
        assert!(size_of::<ffi::Point>() == size_of::<PointXYZIT>());
    };

    if filtered_cloud.is_empty() {
        return (Vec::new(), CudaDownsampleTimings::default());
    }

    let point_skip = point_skip.max(1);
    let out_capacity = filtered_cloud.len().div_ceil(point_skip);
    let mut out = vec![PointXYZIT::default(); out_capacity];
    let mut ffi_timings = std::mem::MaybeUninit::<ffi::CudaDownsampleTimings>::zeroed();

    let start = std::time::Instant::now();
    let size = unsafe {
        ffi::downsample_points(
            stream.stream.raw,
            filtered_cloud.as_ptr() as *const ffi::Point,
            filtered_cloud.len() as u32,
            point_skip as u32,
            voxel_size,
            out.as_mut_ptr() as *mut ffi::Point,
            out.len() as u32,
            ffi_timings.as_mut_ptr(),
        )
    } as usize;
    let ffi_timings = unsafe { ffi_timings.assume_init() };
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    out.truncate(size.min(out_capacity));

    (out, downsample_timings_from_ffi(ffi_timings, total_ms))
}

pub fn downsample_points_cuda_device(
    stream: &Stream,
    filtered_cloud: &[PointXYZIT],
    point_skip: usize,
    voxel_size: f32,
    out_gpu: &mut CudaBuffer,
) -> (usize, CudaDownsampleTimings) {
    const {
        assert!(size_of::<ffi::Point>() == size_of::<PointXYZIT>());
    };

    assert!(
        out_gpu.element_size == size_of::<PointXYZIT>(),
        "Output GPU buffer element size mismatch"
    );
    assert!(
        !out_gpu.gpu_ptr.is_null(),
        "Output GPU buffer pointer is null"
    );

    if filtered_cloud.is_empty() {
        out_gpu.n = 0;
        return (0, CudaDownsampleTimings::default());
    }

    let point_skip = point_skip.max(1);
    let required_capacity = filtered_cloud.len().div_ceil(point_skip);
    let out_capacity = out_gpu.n;
    assert!(
        out_capacity >= required_capacity,
        "Output GPU buffer capacity is too small: have {}, need at least {}",
        out_capacity,
        required_capacity
    );

    let mut ffi_timings = std::mem::MaybeUninit::<ffi::CudaDownsampleTimings>::zeroed();
    let start = std::time::Instant::now();
    let produced = unsafe {
        ffi::downsample_points_device(
            stream.stream.raw,
            filtered_cloud.as_ptr() as *const ffi::Point,
            filtered_cloud.len() as u32,
            point_skip as u32,
            voxel_size,
            out_gpu.gpu_ptr as *mut ffi::Point,
            out_capacity as u32,
            ffi_timings.as_mut_ptr(),
        )
    } as usize;

    let ffi_timings = unsafe { ffi_timings.assume_init() };
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let produced = produced.min(out_capacity);
    out_gpu.n = produced;

    (produced, downsample_timings_from_ffi(ffi_timings, total_ms))
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CudaPlaneTimings {
    pub kernel_compact_ms: f64,
    pub download_ms: f64,
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
) -> CudaPlaneTimings {
    assert!(out_buff.len() >= cloud.n);

    let lidar_to_imu = Pose {
        position: *ekf_imu_offset_t,
        orientation: *ekf_imu_offset_r,
    };
    let pose = Pose {
        position: *ekf_pose_t,
        orientation: *ekf_pose_r,
    };

    const {
        assert!(size_of::<ffi::Point>() == size_of::<PointXYZIT>());
        assert!(size_of::<ffi::PointNormal>() == size_of::<PointXYZINormal>());
    };

    let kernel_compact_start = std::time::Instant::now();
    let size = {
        let my_stream = stream.stream.raw;
        unsafe {
            ffi::planes(
                my_stream,
                cloud.gpu_ptr as *mut ffi::Point,
                cloud.n as u32,
                lidar_to_imu,
                pose,
                out_buff_gpu.gpu_ptr as *mut ffi::PointNormal,
            )
        }
    };
    let kernel_compact_ms = kernel_compact_start.elapsed().as_secs_f64() * 1000.0;
    let downloaded = (size as usize).min(out_buff.len()).min(out_buff_gpu.n);

    // CUDA compacts valid planes on-device in cloud-input order, so only the
    // compacted prefix has to cross back to the CPU for Jacobian assembly.
    let download_start = std::time::Instant::now();
    out_buff_gpu.download_into(stream, &mut out_buff[..downloaded]);
    stream.sync();
    let download_ms = download_start.elapsed().as_secs_f64() * 1000.0;
    unsafe { out_buff.set_len(downloaded) };

    CudaPlaneTimings {
        kernel_compact_ms,
        download_ms,
    }
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

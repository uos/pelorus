//! `pelorus_metal` — Apple Metal port of `pelorus_cuda`.
//!
//! Gated on the `metal_available` cfg flag, which is emitted by `build.rs`
//! when `--features metal_enable` is passed on macOS.

#![cfg(metal_available)]

mod buffer;
mod map;
mod pipeline;
mod radix_sort;
mod stream;
mod types;

mod shaders;

pub use buffer::MetalBuffer;
pub use stream::Stream;
pub use types::{Matrix3, NeighborType, Point, PointNormal, Pose, Quaternion, Vector3};

use pelorus_core::PointXYZIT;
use ros_pointcloud2::points::PointXYZINormal;

// ── Global Allocation Tracking ───────────────────────────────────────────────

static ALLOCATED_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn increment_allocation(bytes: u64) {
    ALLOCATED_BYTES.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn decrement_allocation(bytes: u64) {
    ALLOCATED_BYTES.fetch_sub(bytes, std::sync::atomic::Ordering::Relaxed);
}

/// Return the total number of bytes currently allocated for Metal buffers by this process.
///
/// Note: This tracks process-local buffer allocations and does not include internal Metal
/// overhead, pipeline state memory, or other system-level GPU resources.
pub fn get_used_memory() -> u64 {
    ALLOCATED_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

// ── GPU Timing State ──────────────────────────────────────────────────────────

static LAST_PLANES_GPU_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LAST_ADD_TO_MAP_GPU_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn record_planes_gpu_time(ms: f64) {
    LAST_PLANES_GPU_MS.store(ms.to_bits(), std::sync::atomic::Ordering::Release);
}

pub(crate) fn record_add_to_map_gpu_time(ms: f64) {
    LAST_ADD_TO_MAP_GPU_MS.store(ms.to_bits(), std::sync::atomic::Ordering::Release);
}

/// Try to retrieve GPU timing measurements from previous operations.
/// Returns (planes_ms, add_to_map_ms) if timing results are available.
/// This is non-blocking - returns the most recently measured values.
pub fn try_get_gpu_timings() -> Option<(f64, f64)> {
    let planes_bits = LAST_PLANES_GPU_MS.load(std::sync::atomic::Ordering::Acquire);
    let add_to_map_bits = LAST_ADD_TO_MAP_GPU_MS.load(std::sync::atomic::Ordering::Acquire);

    let planes_ms = f64::from_bits(planes_bits);
    let add_to_map_ms = f64::from_bits(add_to_map_bits);

    if planes_ms > 0.0 || add_to_map_ms > 0.0 {
        Some((planes_ms, add_to_map_ms))
    } else {
        None
    }
}

// ── Size checks ───────────────────────────────────────────────────────────────

const _: () = {
    assert!(size_of::<types::Point>() == size_of::<PointXYZIT>());
};

// ── Public API ────────────────────────────────────────────────────────────────

/// Initialise the global voxel map.
///
/// Must be called exactly once before `add_to_map` or `planes_metal`.
pub fn init_map(
    stream: &Stream,
    capacity: usize,
    voxel_size: f32,
    sensitivity: f32,
    min_dist: f32,
    _nbtype: NeighborType,
) {
    assert!(capacity > 0, "capacity must be > 0");
    map::init_map_impl(stream, capacity, voxel_size, sensitivity, min_dist, _nbtype);

    // Register cleanup hook (runs before Metal teardown on process exit)
    ensure_cleanup_registered();
}

/// Add a point cloud to the global map.
///
/// `cloud` must have been created with `MetalBuffer::from_cloud` or
/// `MetalBuffer::new(stream, size_of::<PointXYZIT>(), n)`.
///
/// The orientation parameter is **not** a rotation matrix any more; it is a
/// unit quaternion packed in [`types::Quaternion`].
/// This matches the GPU-side representation and avoids an expensive
/// `Matrix3 -> UnitQuaternion` conversion at every call.
/// The CUDA backend still accepts a matrix and performs its own conversion.
pub fn add_to_map(
    stream: &Stream,
    cloud: &MetalBuffer,
    ekf_pose_t: &Vector3,
    ekf_pose_r: &Quaternion,
    scan_id: usize,
) {
    let gpu_time_ms = map::add_to_map_impl(stream, cloud, ekf_pose_t, ekf_pose_r, scan_id);
    record_add_to_map_gpu_time(gpu_time_ms);
}

/// Return the voxel map means as a `Vec<Point3<f32>>`.
pub fn get_map() -> Vec<nalgebra::Point3<f32>> {
    map::get_map_impl()
}

/// Return the number of occupied voxels.
pub fn get_map_size() -> u32 {
    map::get_map_size_impl()
}

/// Free all GPU resources held by the map.
pub fn cleanup_map() {
    map::cleanup_map_impl();
}

/// Compute plane-fitted normals for every point in `cloud`.
///
/// Results are written into `out_buff` (CPU) and `out_buff_gpu` (GPU).
/// Both must have capacity ≥ `cloud.n`.
/// Plane-fitting kernel that operates on a point cloud in the map frame.
///
/// Both the lidar pose (`ekf_pose_r`) and the imu offset (`ekf_imu_offset_r`)
/// are now supplied as unit quaternions rather than rotation matrices.  This
/// mirrors the change made to [`add_to_map`] and avoids an extra matrix
/// conversion when dispatching the compute kernel.
pub fn planes_metal(
    stream: &Stream,
    cloud: &MetalBuffer,
    ekf_pose_t: &Vector3,
    ekf_pose_r: &Quaternion,
    ekf_imu_offset_t: &Vector3,
    ekf_imu_offset_r: &Quaternion,
    out_buff: &mut Vec<PointXYZINormal>,
    out_buff_gpu: &mut MetalBuffer,
) {
    let gpu_time_ms = map::planes_metal_impl(
        stream,
        cloud,
        ekf_pose_t,
        ekf_pose_r,
        ekf_imu_offset_t,
        ekf_imu_offset_r,
        out_buff,
        out_buff_gpu,
    );
    record_planes_gpu_time(gpu_time_ms);
}

// ── Cleanup registration ──────────────────────────────────────────────────────

static CLEANUP_GUARD: std::sync::Once = std::sync::Once::new();

fn ensure_cleanup_registered() {
    CLEANUP_GUARD.call_once(|| {
        extern "C" fn cleanup_at_exit() {
            map::cleanup_map_impl();
        }
        unsafe {
            libc::atexit(cleanup_at_exit);
        }
    });
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_roundtrip() {
        // same mutual-exclusion helper used by map.rs tests
        let _lock = crate::map::acquire_test_lock();
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
                intensity: 0.5,
                timestamp: 3.0,
            },
            PointXYZIT {
                x: 3.0,
                y: 1.0,
                z: 2.0,
                intensity: 1.0,
                timestamp: 4.0,
            },
        ];

        let gpu = MetalBuffer::from_cloud(&stream, &cloud);
        // No GPU work — just verify that shared-memory roundtrip works.
        stream.sync();
        let cpu = gpu.into_cloud(&stream);

        assert_eq!(cpu.len(), cloud.len());
        for (a, b) in cpu.iter().zip(cloud.iter()) {
            assert_eq!(a.x, b.x);
            assert_eq!(a.y, b.y);
            assert_eq!(a.z, b.z);
            assert_eq!(a.intensity, b.intensity);
            assert_eq!(a.timestamp, b.timestamp);
        }
    }

    #[test]
    fn map_init_and_add() {
        let _lock = crate::map::acquire_test_lock();
        let stream = Stream::new();
        let cloud = vec![
            PointXYZIT {
                x: 1.0,
                y: 1.0,
                z: 0.0,
                intensity: 0.0,
                timestamp: 0.0,
            },
            PointXYZIT {
                x: 1.1,
                y: 1.1,
                z: 0.0,
                intensity: 0.0,
                timestamp: 0.0,
            },
        ];
        let gpu_cloud = MetalBuffer::from_cloud(&stream, &cloud);

        init_map(&stream, 1024, 0.5, 0.1, 0.01, NeighborType::Neighbor26);

        let t = Vector3::new(0.0, 0.0, 0.0);
        let quat = Quaternion::default(); // identity rotation

        add_to_map(&stream, &gpu_cloud, &t, &quat);
        stream.sync();

        let size = get_map_size();
        assert!(
            size > 0,
            "Map should have at least one voxel after adding points"
        );

        let pts = get_map();
        assert!(!pts.is_empty());

        cleanup_map();
    }
}

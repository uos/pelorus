//! `pelorus_metal` — Apple Metal port of `pelorus_cuda`.
//!
//! Gated on the `metal_available` cfg flag, which is emitted by `build.rs`
//! when `--features metal_enable` is passed on macOS.

#![cfg(metal_available)]

mod buffer;
mod map;
mod pipeline;
mod stream;
mod types;

mod shaders;

pub use buffer::MetalBuffer;
pub use stream::Stream;
pub use types::{Matrix3, NeighborType, Pose, Quaternion, Vector3};

use pelorus_core::PointXYZIT;
use ros_pointcloud2::points::PointXYZINormal;

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
    nbtype: NeighborType,
) {
    assert!(capacity > 0, "capacity must be > 0");
    map::init_map_impl(stream, capacity, voxel_size, sensitivity, min_dist, nbtype);

    // Register cleanup hook (runs before Metal teardown on process exit)
    ensure_cleanup_registered();
}

/// Add a point cloud to the global map.
///
/// `cloud` must have been created with `MetalBuffer::from_cloud` or
/// `MetalBuffer::new(stream, size_of::<PointXYZIT>(), n)`.
pub fn add_to_map(
    stream: &Stream,
    cloud: &MetalBuffer,
    ekf_pose_t: &Vector3,
    ekf_pose_r: &Matrix3,
) {
    map::add_to_map_impl(stream, cloud, ekf_pose_t, ekf_pose_r);
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
pub fn planes_metal(
    stream: &Stream,
    cloud: &MetalBuffer,
    ekf_pose_t: &Vector3,
    ekf_pose_r: &Matrix3,
    ekf_imu_offset_t: &Vector3,
    ekf_imu_offset_r: &Matrix3,
    out_buff: &mut Vec<PointXYZINormal>,
    out_buff_gpu: &mut MetalBuffer,
) {
    map::planes_metal_impl(
        stream,
        cloud,
        ekf_pose_t,
        ekf_pose_r,
        ekf_imu_offset_t,
        ekf_imu_offset_r,
        out_buff,
        out_buff_gpu,
    );
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
        let stream = Stream::new();

        let cloud = vec![
            PointXYZIT { x: 1.0, y: 0.0, z: 1.5, intensity: 0.0, timestamp: 2.0 },
            PointXYZIT { x: 2.0, y: 0.0, z: 1.5, intensity: 0.5, timestamp: 3.0 },
            PointXYZIT { x: 3.0, y: 1.0, z: 2.0, intensity: 1.0, timestamp: 4.0 },
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
        let stream = Stream::new();
        let cloud = vec![
            PointXYZIT { x: 1.0, y: 1.0, z: 0.0, intensity: 0.0, timestamp: 0.0 },
            PointXYZIT { x: 1.1, y: 1.1, z: 0.0, intensity: 0.0, timestamp: 0.0 },
        ];
        let gpu_cloud = MetalBuffer::from_cloud(&stream, &cloud);

        init_map(&stream, 1024, 0.5, 0.1, 0.01, NeighborType::Neighbor26);

        let t = Vector3::new(0.0, 0.0, 0.0);
        let r = Matrix3::from(nalgebra::Matrix3::<f64>::identity().data.0);

        add_to_map(&stream, &gpu_cloud, &t, &r);

        let size = get_map_size();
        assert!(size > 0, "Map should have at least one voxel after adding points");

        let pts = get_map();
        assert!(!pts.is_empty());

        cleanup_map();
    }
}

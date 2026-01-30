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
        let my_stream = stream.stream.raw;
        unsafe {
            ffi::cupcl_upload_buffer(
                my_stream,
                self.gpu_ptr,
                buffer.as_ptr() as *mut ::std::os::raw::c_void,
                self.element_size as u32,
                self.n as u32,
            )
        }
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
                self.n as u32,
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

pub fn add_to_map(stream: &Stream, cloud: &CudaBuffer, ekf_pose_t: &Vector3, ekf_pose_r: &Matrix3) {
    let pose_r = nalgebra::UnitQuaternion::from_matrix(&(*ekf_pose_r).into());
    let pose = Pose {
        position: *ekf_pose_t,
        orientation: Quaternion::new(pose_r.i, pose_r.j, pose_r.k, pose_r.w),
    };
    let my_stream = stream.stream.raw;
    unsafe {
        ffi::add_to_map(
            my_stream,
            cloud.gpu_ptr as *mut ffi::Point,
            cloud.n as u32,
            pose,
        );
    }
}
pub fn get_map_size() -> u32 {
    unsafe { ffi::get_map_size() }
}

pub fn cleanup_map() {
    unsafe { ffi::cleanup_map() }
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
    ekf_pose_r: &Matrix3,
    ekf_imu_offset_t: &Vector3,
    ekf_imu_offset_r: &Matrix3,
    out_buff: &mut Vec<PointXYZINormal>,
    out_buff_gpu: &mut CudaBuffer,
) {
    assert!(out_buff.len() >= cloud.n);

    let ext_r_quat = nalgebra::UnitQuaternion::from_matrix(&(*ekf_imu_offset_r).into());
    let lidar_to_imu = Pose {
        position: *ekf_imu_offset_t,
        orientation: Quaternion::new(ext_r_quat.i, ext_r_quat.j, ext_r_quat.k, ext_r_quat.w),
    };
    let pose_r_quat = nalgebra::UnitQuaternion::from_matrix(&(*ekf_pose_r).into());
    let pose = Pose {
        position: *ekf_pose_t,
        orientation: Quaternion::new(pose_r_quat.i, pose_r_quat.j, pose_r_quat.k, pose_r_quat.w),
    };

    let _ = const {
        assert!(size_of::<ffi::Point>() == size_of::<PointXYZIT>());
        assert!(size_of::<ffi::PointNormal>() == size_of::<PointXYZINormal>());
    };

    let size = {
        let my_stream = stream.stream.raw;
        let size = unsafe {
            ffi::planes(
                my_stream,
                cloud.gpu_ptr as *mut ffi::Point,
                cloud.n as u32,
                lidar_to_imu,
                pose,
                out_buff_gpu.gpu_ptr as *mut ffi::PointNormal,
            )
        };
        size
    };

    // copy to cpu side
    out_buff_gpu.download_into(stream, out_buff);
    unsafe { out_buff.set_len(size as usize) };
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

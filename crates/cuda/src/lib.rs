use std::fmt::Debug;

use ffi::Point;
use log::debug;
use ros_pointcloud2::points::{PointXYZ, PointXYZINormal};

mod ffi;

pub use ffi::{Matrix3, NeighborType, Pose, Quaternion, Vector3};

pub struct Stream {
    pub stream: *mut ::std::os::raw::c_void,
}

impl Stream {
    pub fn new() -> Self {
        let stream = unsafe { ffi::cupcl_create_stream() };
        Self { stream }
    }

    pub fn sync(&self) {
        unsafe { ffi::cupcl_sync_stream(self.stream) }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            ffi::cupcl_destroy_stream(self.stream);
        };
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

impl CudaBuffer {
    pub fn new(stream: &Stream, element_size: usize, n: usize) -> Self {
        let gpu_ptr =
            unsafe { ffi::cupcl_create_empty_buffer(stream.stream, element_size as u32, n as u32) };
        Self {
            gpu_ptr,
            element_size,
            n,
        }
    }

    pub fn clone(&self, stream: &Stream) -> Self {
        let new = Self::new(stream, self.element_size, self.n);
        unsafe {
            ffi::cupcl_copy_buffer(
                stream.stream,
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
        unsafe {
            ffi::cupcl_upload_buffer(
                stream.stream,
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
        unsafe {
            ffi::cupcl_download_buffer(
                stream.stream,
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
            dst.len() >= self.n,
            "Destination buffer on CPU is too small"
        );
        unsafe {
            ffi::cupcl_download_buffer(
                stream.stream,
                dst.as_mut_ptr() as *mut ::std::os::raw::c_void,
                self.gpu_ptr,
                self.element_size as u32,
                self.n as u32,
            )
        }
    }

    pub fn from_vec<T>(stream: &Stream, vec: Vec<T>, element_size: usize) -> Self {
        let gpu_ptr = unsafe {
            ffi::cupcl_create_buffer(
                stream.stream,
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

    pub fn from_cloud(stream: &Stream, vec: &[PointXYZ]) -> Self {
        let element_size = size_of::<f32>() * 4;
        let gpu_ptr = unsafe {
            ffi::cupcl_create_buffer(
                stream.stream,
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

    pub fn into_cloud(&self, stream: &Stream) -> Vec<PointXYZ> {
        let cpu_buff = self.download(stream);
        stream.sync();
        cpu_buff
    }

    // pub fn as_slice<T>(&self, stream: &Stream) -> &[T] {
    //     stream.sync();
    //     unsafe { slice::from_raw_parts(self.as_ptr::<T>(), self.n) }
    // }

    // pub fn as_mut_slice<T>(&self, stream: &Stream) -> &mut [T] {
    //     stream.sync();
    //     unsafe { slice::from_raw_parts_mut(self.as_ptr::<T>(), self.n) }
    // }

    // pub fn as_cloud_mut(&self, stream: &Stream) -> &[PointXYZ] {
    //     stream.sync();
    //     unsafe { slice::from_raw_parts_mut(self.as_ptr::<PointXYZ>(), self.n) }
    // }

    // pub fn read_value<T: Copy>(&self) -> &T {
    //     debug_assert_eq!(self.element_size * self.n, std::mem::size_of::<T>());
    //     unsafe { &*(self.gpu_ptr as *mut T) }
    // }

    // pub fn as_ptr<T>(&self) -> *mut T {
    //     self.gpu_ptr as *mut T
    // }
}

pub fn get_map() -> Vec<nalgebra::Point3<f32>> {
    let mut out = vec![PointXYZ::default(); unsafe { ffi::get_map_capacity() as usize }];
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
    assert!(
        (capacity & (capacity - 1)) == 0,
        "Capacity is not a power of 2"
    );
    debug!("MAP INIT");
    unsafe {
        ffi::init_map(
            stream.stream,
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
    unsafe {
        ffi::add_to_map(
            stream.stream,
            cloud.gpu_ptr as *mut ffi::Point,
            cloud.n as u32,
            pose,
        );
    }
}
pub fn get_map_size() -> u32 {
    unsafe { ffi::get_map_size() }
}

pub fn planes_cuda(
    stream: &Stream,
    neighbors: &mut CudaBuffer,
    neighbor_sizes: &mut CudaBuffer,
    cloud: &CudaBuffer,
    ekf_pose_t: &Vector3,
    ekf_pose_r: &Matrix3,
    ekf_imu_offset_t: &Vector3,
    ekf_imu_offset_r: &Matrix3,
    k: usize,
    neighbor_max_range: f32,
    plane_point_error_threshold: f64,
    first_it_local: bool,
    out_buff: &mut Vec<PointXYZINormal>,
    out_buff_gpu: &mut CudaBuffer,
    converged: bool,
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

    assert_eq!(size_of::<ffi::Point>(), size_of::<PointXYZ>());
    assert_eq!(size_of::<ffi::PointNormal>(), size_of::<PointXYZINormal>());

    let size = unsafe {
        ffi::planes(
            stream.stream,
            cloud.gpu_ptr as *mut ffi::Point,
            cloud.n as u32,
            neighbors.gpu_ptr as *mut ffi::Point,
            neighbor_sizes.gpu_ptr as *mut u32,
            lidar_to_imu,
            pose,
            k as u32,
            neighbor_max_range,
            plane_point_error_threshold,
            first_it_local,
            out_buff_gpu.gpu_ptr as *mut ffi::PointNormal,
            converged,
        )
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
            PointXYZ::new(1.0, 0.0, 1.5),
            PointXYZ::new(2.0, 0.0, 1.5),
            PointXYZ::new(3.0, 0.0, 1.5),
        ];
        let gpu = CudaBuffer::from_cloud(&stream, &cloud);
        let cpu = gpu.into_cloud(&stream);
        for i in 0..cloud.len() {
            assert_eq!(cpu[i], cloud[i]);
        }
    }
}

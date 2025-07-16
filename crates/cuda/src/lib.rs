use core::slice;

use ros_pointcloud2::points::{PointXYZ, PointXYZINormal};

pub mod ffi;

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
pub struct UnifiedBuffer {
    pub gpu_ptr: *mut ::std::os::raw::c_void,
    pub element_size: usize,
    pub n: usize,
}

impl Drop for UnifiedBuffer {
    fn drop(&mut self) {
        unsafe {
            ffi::cupcl_free_buffer(self.gpu_ptr);
        }
    }
}

impl UnifiedBuffer {
    pub fn new(stream: &Stream, element_size: usize, n: usize) -> Self {
        let gpu_ptr =
            unsafe { ffi::cupcl_create_empty_buffer(stream.stream, element_size as u32, n as u32) };
        Self {
            gpu_ptr,
            element_size,
            n,
        }
    }

    pub fn upload(&mut self, stream: &Stream, buffer: &[u8]) {
        //assert_eq!(buffer.len(), self.element_size * self.n);
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

    pub fn as_cloud(&self, stream: &Stream) -> &[PointXYZ] {
        stream.sync();
        unsafe { slice::from_raw_parts(self.as_ptr::<PointXYZ>(), self.n) }
    }

    pub fn as_slice<T>(&self, stream: &Stream) -> &[T] {
        stream.sync();
        unsafe { slice::from_raw_parts(self.as_ptr::<T>(), self.n) }
    }

    pub fn as_mut_slice<T>(&self, stream: &Stream) -> &mut [T] {
        stream.sync();
        unsafe { slice::from_raw_parts_mut(self.as_ptr::<T>(), self.n) }
    }

    pub fn as_cloud_mut(&self, stream: &Stream) -> &[PointXYZ] {
        stream.sync();
        unsafe { slice::from_raw_parts_mut(self.as_ptr::<PointXYZ>(), self.n) }
    }

    pub fn read_value<T: Copy>(&self) -> &T {
        debug_assert_eq!(self.element_size * self.n, std::mem::size_of::<T>());
        unsafe { &*(self.gpu_ptr as *mut T) }
    }

    pub fn as_ptr<T>(&self) -> *mut T {
        self.gpu_ptr as *mut T
    }
}

pub fn planes_cuda(
    stream: &Stream,
    neighbors: &mut UnifiedBuffer,
    neighbor_sizes: &mut UnifiedBuffer,
    cloud: &UnifiedBuffer,
) -> Vec<PointXYZINormal> {
    // TODO
    todo!()
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
        let gpu = UnifiedBuffer::from_cloud(&stream, &cloud);
        let cpu = gpu.as_cloud(&stream);
        for i in 0..cloud.len() {
            assert_eq!(cpu[i], cloud[i]);
        }
    }
}

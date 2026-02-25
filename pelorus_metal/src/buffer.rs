use bytemuck::Pod;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use pelorus_core::PointXYZIT;

use crate::stream::Stream;

/// A Metal buffer allocated with `StorageModeShared`.
pub struct MetalBuffer {
    pub(crate) buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub element_size: usize,
    pub n: usize,
}

unsafe impl Send for MetalBuffer {}

impl Drop for MetalBuffer {
    fn drop(&mut self) {
        crate::decrement_allocation(self.buf.length() as u64);
    }
}

impl MetalBuffer {
    pub fn new(stream: &Stream, element_size: usize, n: usize) -> Self {
        let bytes = (element_size * n) as u64;
        let buf = stream
            .device
            .newBufferWithLength_options(bytes as _, MTLResourceOptions::StorageModeShared)
            .expect("Failed to create buffer");

        crate::increment_allocation(buf.length() as u64);
        Self {
            buf,
            element_size,
            n,
        }
    }

    pub fn capacity(&self) -> usize {
        self.buf.length() as usize / self.element_size
    }

    pub fn as_slice<T: Pod>(&self) -> &[T] {
        assert_eq!(size_of::<T>(), self.element_size, "element size mismatch");
        unsafe { std::slice::from_raw_parts(self.buf.contents().as_ptr().cast(), self.n) }
    }

    pub fn as_slice_mut<T: Pod>(&mut self) -> &mut [T] {
        assert_eq!(size_of::<T>(), self.element_size, "element size mismatch");
        unsafe { std::slice::from_raw_parts_mut(self.buf.contents().as_ptr().cast(), self.n) }
    }

    pub fn upload<T: Pod>(&mut self, _stream: &Stream, src: &[T]) {
        assert_eq!(size_of::<T>(), self.element_size, "element size mismatch");
        assert!(
            src.len() <= self.capacity(),
            "source ({}) larger than buffer capacity ({})",
            src.len(),
            self.capacity()
        );
        unsafe {
            let dst =
                std::slice::from_raw_parts_mut(self.buf.contents().as_ptr().cast::<T>(), src.len());
            dst.copy_from_slice(src);
        }
        self.n = src.len();
    }

    pub fn download<T: Pod + Default + Clone>(&self, _stream: &Stream) -> Vec<T> {
        self.as_slice::<T>().to_vec()
    }

    pub fn from_cloud(stream: &Stream, cloud: &[PointXYZIT]) -> Self {
        let element_size = size_of::<PointXYZIT>();
        let mut mb = Self::new(stream, element_size, cloud.len());
        mb.as_slice_mut::<PointXYZIT>().copy_from_slice(cloud);
        mb
    }

    pub fn into_cloud(&self, _stream: &Stream) -> Vec<PointXYZIT> {
        self.as_slice::<PointXYZIT>().to_vec()
    }

    pub fn byte_len(&self) -> usize {
        self.element_size * self.n
    }

    pub(crate) fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &*self.buf
    }
}

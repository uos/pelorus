use bytemuck::Pod;
use metal::{Buffer as MtlBuffer, MTLResourceOptions};
use pelorus_core::PointXYZIT;

use crate::stream::Stream;
use crate::types::Point;

/// A Metal buffer allocated with `StorageModeShared`.
///
/// On Apple Silicon (unified memory) the CPU and GPU share the same physical
/// pages, so no explicit upload/download copies are necessary.  `as_slice` and
/// `as_slice_mut` give direct CPU access; call `stream.sync()` after GPU work
/// before reading from the CPU.
pub struct MetalBuffer {
    pub(crate) buf: MtlBuffer,
    pub element_size: usize,
    pub n: usize,
}

unsafe impl Send for MetalBuffer {}

impl MetalBuffer {
    /// Allocate a zeroed shared buffer for `n` elements of `element_size` bytes.
    pub fn new(stream: &Stream, element_size: usize, n: usize) -> Self {
        let bytes = (element_size * n) as u64;
        let buf = stream.device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        // Zero the contents
        unsafe {
            std::ptr::write_bytes(buf.contents() as *mut u8, 0, (element_size * n) as usize);
        }
        Self { buf, element_size, n }
    }

    /// Read the buffer contents as a typed slice.
    ///
    /// # Safety
    /// The GPU must have finished writing before this is called (use `stream.sync()`).
    pub fn as_slice<T: Pod>(&self) -> &[T] {
        assert_eq!(size_of::<T>(), self.element_size, "element size mismatch");
        unsafe { std::slice::from_raw_parts(self.buf.contents() as *const T, self.n) }
    }

    /// Mutably access the buffer contents from the CPU.
    ///
    /// Writes are immediately visible to the next GPU dispatch.
    pub fn as_slice_mut<T: Pod>(&mut self) -> &mut [T] {
        assert_eq!(size_of::<T>(), self.element_size, "element size mismatch");
        unsafe { std::slice::from_raw_parts_mut(self.buf.contents() as *mut T, self.n) }
    }

    /// Upload `src` into the buffer (memcpy on shared memory — no GPU staging).
    pub fn upload<T: Pod>(&mut self, _stream: &Stream, src: &[T]) {
        assert!(src.len() <= self.n, "source larger than buffer");
        assert_eq!(size_of::<T>(), self.element_size, "element size mismatch");
        let dst = self.as_slice_mut::<T>();
        dst[..src.len()].copy_from_slice(src);
    }

    /// Download all `n` elements into a `Vec<T>` (memcpy on shared memory).
    ///
    /// Call `stream.sync()` first to ensure GPU writes are complete.
    pub fn download<T: Pod + Default + Clone>(&self, _stream: &Stream) -> Vec<T> {
        self.as_slice::<T>().to_vec()
    }

    // ── Convenience constructors ─────────────────────────────────────────────

    /// Allocate a shared buffer and copy `cloud` into it.
    ///
    /// `PointXYZIT` and `Point` share the same `#[repr(C)]` layout (5 × f32 = 20 bytes),
    /// so a byte-level copy is safe.
    pub fn from_cloud(stream: &Stream, cloud: &[PointXYZIT]) -> Self {
        let element_size = size_of::<PointXYZIT>();
        let mb = Self::new(stream, element_size, cloud.len());
        // SAFETY: same size, same repr(C) layout.
        unsafe {
            std::ptr::copy_nonoverlapping(
                cloud.as_ptr() as *const u8,
                mb.buf.contents() as *mut u8,
                element_size * cloud.len(),
            );
        }
        mb
    }

    /// Read the buffer as a `Vec<PointXYZIT>`.
    ///
    /// Call `stream.sync()` before calling this method.
    pub fn into_cloud(&self, _stream: &Stream) -> Vec<PointXYZIT> {
        // SAFETY: same size, same repr(C) layout as Point.
        let points: &[Point] = self.as_slice::<Point>();
        points.iter().copied().map(PointXYZIT::from).collect()
    }

    /// Number of bytes in the buffer.
    pub fn byte_len(&self) -> usize {
        self.element_size * self.n
    }

    /// Pointer to the start of the Metal buffer (for MSL buffer bindings).
    pub(crate) fn metal_buffer(&self) -> &MtlBuffer {
        &self.buf
    }
}

// Compile-time check: PointXYZIT and our GPU Point type must be the same size.
const _: [u8; size_of::<PointXYZIT>()] = [0u8; size_of::<Point>()];

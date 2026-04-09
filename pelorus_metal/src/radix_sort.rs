use crate::buffer::MetalBuffer;
use crate::pipeline::{THREADS_PER_GROUP, get_pipelines, grid1d, group1d};
use objc2_metal::MTLComputeCommandEncoder;
use objc2_metal::MTLSize;

pub const RADIX_SIZE: usize = 256;

pub struct RadixSortBuffers {
    pub alt_keys: MetalBuffer,
    pub alt_values: MetalBuffer,
    pub block_histograms: MetalBuffer,
    pub block_offsets: MetalBuffer,
    pub bucket_counts: MetalBuffer,
    pub bucket_offsets: MetalBuffer,
    pub max_elements: usize,
}

impl RadixSortBuffers {
    pub fn new(stream: &crate::stream::Stream, max_elements: usize) -> Self {
        let max_blocks = max_elements.div_ceil(THREADS_PER_GROUP as usize).max(1);
        Self {
            alt_keys: MetalBuffer::new(stream, size_of::<u32>(), max_elements.max(1)),
            alt_values: MetalBuffer::new(stream, size_of::<u32>(), max_elements.max(1)),
            block_histograms: MetalBuffer::new(stream, size_of::<u32>(), max_blocks * RADIX_SIZE),
            block_offsets: MetalBuffer::new(stream, size_of::<u32>(), max_blocks * RADIX_SIZE),
            bucket_counts: MetalBuffer::new(stream, size_of::<u32>(), RADIX_SIZE),
            bucket_offsets: MetalBuffer::new(stream, size_of::<u32>(), RADIX_SIZE),
            max_elements: max_elements.max(1),
        }
    }
}

pub fn dispatch_radix_sort_u32_pairs(
    stream: &crate::stream::Stream,
    encoder: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>>,
    keys: &MetalBuffer,
    values: &MetalBuffer,
    scratch: &RadixSortBuffers,
    n: usize,
    key_xor: u32,
) {
    if n == 0 {
        return;
    }

    let pipelines = get_pipelines(&*stream.device);
    let num_blocks = n.div_ceil(THREADS_PER_GROUP as usize);
    let threadgroups = MTLSize {
        width: num_blocks,
        height: 1,
        depth: 1,
    };
    let total_hist_entries = (num_blocks * RADIX_SIZE) as u32;
    let n_u32 = n as u32;
    let num_blocks_u32 = num_blocks as u32;

    let mut in_keys = keys.metal_buffer();
    let mut in_values = values.metal_buffer();
    let mut out_keys = scratch.alt_keys.metal_buffer();
    let mut out_values = scratch.alt_values.metal_buffer();

    for shift in (0u32..32u32).step_by(8) {
        encoder.setComputePipelineState(&pipelines.radix_zero_hist);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(scratch.block_histograms.metal_buffer()), 0, 0);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new(&total_hist_entries as *const u32 as *mut _).unwrap(),
                size_of::<u32>(),
                1,
            );
            encoder.dispatchThreads_threadsPerThreadgroup(
                grid1d(total_hist_entries as usize),
                group1d(),
            );
        }
        encoder.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);

        encoder.setComputePipelineState(&pipelines.radix_block_histogram);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(in_keys), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(scratch.block_histograms.metal_buffer()), 0, 1);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(),
                size_of::<u32>(),
                2,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new(&shift as *const u32 as *mut _).unwrap(),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new(&key_xor as *const u32 as *mut _).unwrap(),
                size_of::<u32>(),
                4,
            );
            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, group1d());
        }
        encoder.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);

        encoder.setComputePipelineState(&pipelines.radix_prefix_blocks);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(scratch.block_histograms.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(scratch.block_offsets.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(scratch.bucket_counts.metal_buffer()), 0, 2);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new(&num_blocks_u32 as *const u32 as *mut _).unwrap(),
                size_of::<u32>(),
                3,
            );
            encoder.dispatchThreads_threadsPerThreadgroup(grid1d(RADIX_SIZE), group1d());
        }
        encoder.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);

        encoder.setComputePipelineState(&pipelines.radix_bucket_offsets);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(scratch.bucket_counts.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(scratch.bucket_offsets.metal_buffer()), 0, 1);
            encoder.dispatchThreads_threadsPerThreadgroup(grid1d(1), group1d());
        }
        encoder.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);

        encoder.setComputePipelineState(&pipelines.radix_scatter_pairs);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(in_keys), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(in_values), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out_keys), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(out_values), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(scratch.block_offsets.metal_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(scratch.bucket_offsets.metal_buffer()), 0, 5);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new(&shift as *const u32 as *mut _).unwrap(),
                size_of::<u32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new(&key_xor as *const u32 as *mut _).unwrap(),
                size_of::<u32>(),
                8,
            );
            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, group1d());
        }
        encoder.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);

        std::mem::swap(&mut in_keys, &mut out_keys);
        std::mem::swap(&mut in_values, &mut out_values);
    }
}

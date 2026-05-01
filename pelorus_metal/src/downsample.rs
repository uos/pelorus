use std::sync::{Mutex, OnceLock};

use objc2_metal::{
    MTLBarrierScope, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
};
use pelorus_core::PointXYZIT;

use crate::buffer::MetalBuffer;
use crate::pipeline::{get_pipelines, grid1d, group1d};
use crate::radix_sort::RadixSortBuffers;
use crate::{MetalDownsampleTimings, Stream};

#[repr(C)]
#[derive(Clone, Copy)]
struct DownsampleParams {
    voxel_size: f32,
    selected_count: u32,
    point_skip: u32,
    unique_count: u32,
}

struct DownsampleScratch {
    input: MetalBuffer,
    selected: MetalBuffer,
    keys: MetalBuffer,
    indices: MetalBuffer,
    run_starts: MetalBuffer,
    run_ends: MetalBuffer,
    counter: MetalBuffer,
    medoids: MetalBuffer,
    radix: RadixSortBuffers,
    capacity: usize,
}

impl DownsampleScratch {
    fn new(stream: &Stream, capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            input: MetalBuffer::new(stream, size_of::<PointXYZIT>(), capacity),
            selected: MetalBuffer::new(stream, size_of::<PointXYZIT>(), capacity),
            keys: MetalBuffer::new(stream, size_of::<u32>(), capacity),
            indices: MetalBuffer::new(stream, size_of::<u32>(), capacity),
            run_starts: MetalBuffer::new(stream, size_of::<u32>(), capacity),
            run_ends: MetalBuffer::new(stream, size_of::<u32>(), capacity),
            counter: MetalBuffer::new(stream, size_of::<u32>(), 1),
            medoids: MetalBuffer::new(stream, size_of::<PointXYZIT>(), capacity),
            radix: RadixSortBuffers::new(stream, capacity),
            capacity,
        }
    }
}

static DOWNSAMPLE_STATE: OnceLock<Mutex<Option<DownsampleScratch>>> = OnceLock::new();

fn state() -> &'static Mutex<Option<DownsampleScratch>> {
    DOWNSAMPLE_STATE.get_or_init(|| Mutex::new(None))
}

fn ensure_state(
    stream: &Stream,
    capacity: usize,
) -> std::sync::MutexGuard<'static, Option<DownsampleScratch>> {
    let mut guard = state().lock().expect("metal downsample mutex poisoned");
    let needs_init = guard
        .as_ref()
        .is_none_or(|scratch| scratch.capacity < capacity || scratch.radix.max_elements < capacity);
    if needs_init {
        *guard = Some(DownsampleScratch::new(stream, capacity.next_power_of_two()));
    }
    guard
}

pub fn downsample_points_metal_impl(
    stream: &Stream,
    filtered_cloud: &[PointXYZIT],
    point_skip: usize,
    voxel_size: f32,
) -> (Vec<PointXYZIT>, MetalDownsampleTimings) {
    if filtered_cloud.is_empty() {
        return (Vec::new(), MetalDownsampleTimings::default());
    }

    let start_total = std::time::Instant::now();
    let point_skip = point_skip.max(1);
    let selected_count = filtered_cloud.len().div_ceil(point_skip);
    let mut guard = ensure_state(stream, filtered_cloud.len().max(selected_count));
    let scratch = guard
        .as_mut()
        .expect("metal downsample state was not initialized");

    let start = std::time::Instant::now();
    scratch.input.upload(stream, filtered_cloud);
    scratch.counter.upload(stream, &[0u32]);
    let upload_ms = start.elapsed().as_secs_f64() * 1000.0;

    let pipes = get_pipelines(&*stream.device);
    let params = DownsampleParams {
        voxel_size,
        selected_count: selected_count as u32,
        point_skip: point_skip as u32,
        unique_count: 0,
    };

    let cb = stream.queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");

    enc.setComputePipelineState(&pipes.downsample_build_keys);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(scratch.input.metal_buffer()), 0, 0);
        enc.setBuffer_offset_atIndex(Some(scratch.selected.metal_buffer()), 0, 1);
        enc.setBuffer_offset_atIndex(Some(scratch.keys.metal_buffer()), 0, 2);
        enc.setBuffer_offset_atIndex(Some(scratch.indices.metal_buffer()), 0, 3);
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
            size_of::<DownsampleParams>(),
            4,
        );
        enc.dispatchThreads_threadsPerThreadgroup(grid1d(selected_count), group1d());
    }
    enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

    crate::radix_sort::dispatch_radix_sort_u32_pairs(
        stream,
        &enc,
        &scratch.keys,
        &scratch.indices,
        &scratch.radix,
        selected_count,
        0x8000_0000,
    );
    enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

    enc.setComputePipelineState(&pipes.downsample_rle);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(scratch.keys.metal_buffer()), 0, 0);
        enc.setBuffer_offset_atIndex(Some(scratch.run_starts.metal_buffer()), 0, 1);
        enc.setBuffer_offset_atIndex(Some(scratch.run_ends.metal_buffer()), 0, 2);
        enc.setBuffer_offset_atIndex(Some(scratch.counter.metal_buffer()), 0, 3);
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
            size_of::<DownsampleParams>(),
            4,
        );
        enc.dispatchThreads_threadsPerThreadgroup(grid1d(selected_count), group1d());
    }
    enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
    enc.setComputePipelineState(&pipes.downsample_medoids);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(scratch.selected.metal_buffer()), 0, 0);
        enc.setBuffer_offset_atIndex(Some(scratch.indices.metal_buffer()), 0, 1);
        enc.setBuffer_offset_atIndex(Some(scratch.run_starts.metal_buffer()), 0, 2);
        enc.setBuffer_offset_atIndex(Some(scratch.run_ends.metal_buffer()), 0, 3);
        enc.setBuffer_offset_atIndex(Some(scratch.medoids.metal_buffer()), 0, 4);
        enc.setBuffer_offset_atIndex(Some(scratch.counter.metal_buffer()), 0, 5);
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
            size_of::<DownsampleParams>(),
            6,
        );
        enc.dispatchThreads_threadsPerThreadgroup(grid1d(selected_count), group1d());
    }
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    let gpu_ms = (cb.GPUEndTime() - cb.GPUStartTime()) * 1000.0;

    let unique_start = std::time::Instant::now();
    let unique_count = scratch.counter.as_slice::<u32>()[0] as usize;
    let unique_count_sync_ms = unique_start.elapsed().as_secs_f64() * 1000.0;

    if unique_count == 0 {
        return (
            Vec::new(),
            MetalDownsampleTimings {
                total_ms: start_total.elapsed().as_secs_f64() * 1000.0,
                upload_ms,
                key_build_ms: gpu_ms,
                unique_count_sync_ms,
                ..Default::default()
            },
        );
    }

    scratch.medoids.n = unique_count;
    let download_start = std::time::Instant::now();
    let mut out = scratch.medoids.download::<PointXYZIT>(stream);
    let download_ms = download_start.elapsed().as_secs_f64() * 1000.0;

    let timestamp_sort_start = std::time::Instant::now();
    out.sort_unstable_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());
    let timestamp_sort_ms = timestamp_sort_start.elapsed().as_secs_f64() * 1000.0;

    (
        out,
        MetalDownsampleTimings {
            total_ms: start_total.elapsed().as_secs_f64() * 1000.0,
            upload_ms,
            key_build_ms: gpu_ms,
            run_length_encode_ms: 0.0,
            unique_count_sync_ms,
            timestamp_sort_ms,
            download_ms,
            ..Default::default()
        },
    )
}

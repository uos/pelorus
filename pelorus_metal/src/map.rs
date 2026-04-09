use bytemuck::{Pod, Zeroable};
#[cfg(feature = "metal_enable")]
use rayon::prelude::*;
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::pipeline::{get_pipelines, grid1d, group1d};
use crate::stream::Stream;
use crate::types::NeighborType;

use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder,
};

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub struct KeyValuePair {
    pub key: i32,
    pub value: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuVoxelDist {
    pub lock_flag: i32,
    pub transform: [f32; 9],
    pub point_count: u32,
    pub mean: [f32; 3],
    pub covariance: [f32; 9],
    pub last_scan: u32,
}

const _: () = assert!(size_of::<GpuVoxelDist>() == 96);
#[cfg(test)]
const EMPTY_KEY: i32 = i32::MIN;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BuildVoxelsParams {
    pub pose_quat: [f32; 4],
    pub pose_trans: [f32; 3],
    pub _pad0: f32,
    pub resolution: f32,
    pub scan_number: u32,
    pub sensitivity: f32,
    pub min_dist: f32,
    pub unique_key_count: u32,
    pub table_cap: u32,
    pub dist_cap: u32,
    pub n_points: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PlanesParams {
    pub lidar_to_imu_quat: [f32; 4], // 16 bytes, offset 0
    pub lidar_to_imu_pos: [f32; 3],  // 12 bytes, offset 16
    pub _pad0: f32,                  // 4 bytes, offset 28
    pub pose_quat: [f32; 4],         // 16 bytes, offset 32
    pub pose_pos: [f32; 3],          // 12 bytes, offset 48
    pub _pad1: f32,                  // 4 bytes, offset 60
    pub cloud_size: u32,             // 4 bytes, offset 64
    pub table_cap: u32,              // 4 bytes, offset 68
    pub resolution: f32,             // 4 bytes, offset 72
    pub sphericity_threshold: f32,   // 4 bytes, offset 76
    pub nbtype: u32,                 // 4 bytes, offset 80 -> Total 84
}

/// Tests use this mutex to avoid map state races
#[cfg(test)]
pub(crate) static TEST_MAP_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub struct ScratchBuffers {
    pub keys: crate::buffer::MetalBuffer,
    pub indices: crate::buffer::MetalBuffer,
    pub radix: crate::radix_sort::RadixSortBuffers,
    pub run_starts: crate::buffer::MetalBuffer,
    pub run_ends: crate::buffer::MetalBuffer,
    pub unique_keys: crate::buffer::MetalBuffer,
    pub target_indices: crate::buffer::MetalBuffer,
    pub update_flags: crate::buffer::MetalBuffer,
    pub sorted_points: crate::buffer::MetalBuffer,
    pub counter: crate::buffer::MetalBuffer,
    pub lru_keys: crate::buffer::MetalBuffer,
    pub lru_mapping: crate::buffer::MetalBuffer,
    pub capacity: usize,
}

pub struct MetalMapState {
    pub keys_buf: crate::buffer::MetalBuffer,
    pub values_buf: crate::buffer::MetalBuffer,
    pub dist_buf: crate::buffer::MetalBuffer,
    pub dist_keys_buf: crate::buffer::MetalBuffer,
    pub scratch: ScratchBuffers,
    pub dist_cap: u32,
    pub table_cap: u32,
    pub resolution: f32,
    pub sensitivity: f32,
    pub min_dist: f32,
    pub scan_number: u32,
    pub nbtype: u32,
    add_cursor: u32,
    lru_cursor: u32,
    parity_trace_enabled: bool,
}

static MAP_STATE: OnceLock<Mutex<Option<MetalMapState>>> = OnceLock::new();

fn global_state() -> &'static Mutex<Option<MetalMapState>> {
    MAP_STATE.get_or_init(|| Mutex::new(None))
}

pub fn get_state() -> MutexGuard<'static, Option<MetalMapState>> {
    global_state().lock().unwrap()
}

fn parity_trace_enabled() -> bool {
    std::env::var("PELORUS_METAL_PARITY_TRACE")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "yes" || v == "on"
        })
        .unwrap_or(false)
}

fn cpu_assign_trace_enabled() -> bool {
    std::env::var("PELORUS_METAL_CPU_ASSIGN")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "yes" || v == "on"
        })
        .unwrap_or(false)
}

fn host_hash_fn(key: i32) -> u32 {
    let mut h = key as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}

fn host_hashmap_lookup(table_keys: &[i32], table_vals: &[u32], table_cap: u32, key: i32) -> u32 {
    if key == i32::MIN {
        return u32::MAX;
    }
    let mut slot = (host_hash_fn(key) & (table_cap - 1)) as usize;
    for _ in 0..table_cap {
        let cur = table_keys[slot];
        if cur == key {
            return table_vals[slot];
        }
        if cur == i32::MIN {
            return u32::MAX;
        }
        slot = (slot + 1) & ((table_cap - 1) as usize);
    }
    u32::MAX
}

fn sample_hash_i32(data: &[i32]) -> u64 {
    // Fast FNV-1a over a bounded prefix for low debug overhead.
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h = FNV_OFFSET;
    for &x in data.iter().take(4096) {
        h ^= x as u32 as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn validate_sorted_runs(
    sorted_keys: &[i32],
    unique_keys: &[i32],
    run_starts: &[u32],
    run_ends: &[u32],
) -> Option<String> {
    for i in 1..sorted_keys.len() {
        if sorted_keys[i - 1] > sorted_keys[i] {
            return Some(format!(
                "sorted_keys not monotonic at i={} prev={} cur={}",
                i,
                sorted_keys[i - 1],
                sorted_keys[i]
            ));
        }
    }

    let n = sorted_keys.len();
    for i in 0..unique_keys.len() {
        let start = run_starts[i] as usize;
        let end = run_ends[i] as usize;
        if start >= n || end > n || start >= end {
            return Some(format!(
                "invalid run bounds at i={} start={} end={} n={}",
                i, start, end, n
            ));
        }
        let key = unique_keys[i];
        if sorted_keys[start] != key {
            return Some(format!(
                "run start key mismatch at i={} start={} key={} sorted={}",
                i, start, key, sorted_keys[start]
            ));
        }
        if start > 0 && sorted_keys[start - 1] == key {
            return Some(format!(
                "run start is not first occurrence at i={} start={} key={}",
                i, start, key
            ));
        }
        if sorted_keys[end - 1] != key {
            return Some(format!(
                "run end key mismatch at i={} end={} key={} sorted={}",
                i,
                end,
                key,
                sorted_keys[end - 1]
            ));
        }
        if end < n && sorted_keys[end] == key {
            return Some(format!(
                "run end is not last occurrence at i={} end={} key={}",
                i, end, key
            ));
        }
    }
    None
}

pub fn init_map_impl(
    stream: &Stream,
    capacity: usize,
    voxel_size: f32,
    sensitivity: f32,
    min_dist: f32,
    _nbtype: NeighborType,
) {
    let capacity = capacity.next_power_of_two();
    let dist_cap = capacity as u32;
    let table_cap = dist_cap * 2;

    let keys_buf = crate::buffer::MetalBuffer::new(stream, size_of::<i32>(), table_cap as usize);
    let values_buf = crate::buffer::MetalBuffer::new(stream, size_of::<u32>(), table_cap as usize);
    let dist_buf =
        crate::buffer::MetalBuffer::new(stream, size_of::<GpuVoxelDist>(), dist_cap as usize);
    let dist_keys_buf =
        crate::buffer::MetalBuffer::new(stream, size_of::<i32>(), dist_cap as usize);

    unsafe {
        std::ptr::write_bytes(
            dist_buf.buf.contents().as_ptr().cast::<u8>(),
            0,
            dist_buf.buf.length() as usize,
        );
        let slice = std::slice::from_raw_parts_mut(
            dist_keys_buf.buf.contents().as_ptr().cast::<i32>(),
            dist_cap as usize,
        );
        slice.fill(i32::MIN);
    }

    // Initial scratch buffers with capacity 1; these will be resized on demand in add_to_map_impl.
    // We just need valid buffers to pass to the GPU during init.
    let scratch_cap = 1;
    let scratch = ScratchBuffers {
        keys: crate::buffer::MetalBuffer::new(stream, size_of::<i32>(), scratch_cap),
        indices: crate::buffer::MetalBuffer::new(stream, size_of::<u32>(), scratch_cap),
        radix: crate::radix_sort::RadixSortBuffers::new(stream, scratch_cap),
        run_starts: crate::buffer::MetalBuffer::new(stream, size_of::<u32>(), scratch_cap),
        run_ends: crate::buffer::MetalBuffer::new(stream, size_of::<u32>(), scratch_cap),
        unique_keys: crate::buffer::MetalBuffer::new(stream, size_of::<i32>(), scratch_cap),
        target_indices: crate::buffer::MetalBuffer::new(stream, size_of::<u32>(), scratch_cap),
        update_flags: crate::buffer::MetalBuffer::new(stream, size_of::<u32>(), scratch_cap),
        sorted_points: crate::buffer::MetalBuffer::new(
            stream,
            size_of::<crate::types::Point>(),
            scratch_cap,
        ),
        counter: crate::buffer::MetalBuffer::new(stream, size_of::<u32>(), 5),
        lru_keys: crate::buffer::MetalBuffer::new(stream, size_of::<u32>(), dist_cap as usize),
        lru_mapping: crate::buffer::MetalBuffer::new(stream, size_of::<u32>(), dist_cap as usize),
        // keep the logical capacity zero so resize logic still fires immediately
        capacity: 0,
    };

    let initial_lru_mapping: Vec<u32> = (0..dist_cap).collect();
    let counter_init = [0u32, 0u32, 0u32, 0u32, 0u32];
    let mut scratch = scratch;
    scratch.lru_mapping.upload(stream, &initial_lru_mapping);
    scratch.counter.upload(stream, &counter_init);

    {
        let pipes = get_pipelines(&*stream.device);
        let cb = stream.queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        enc.setComputePipelineState(&pipes.hashmap_init);
        let cap_u32 = table_cap;
        unsafe {
            enc.setBuffer_offset_atIndex(Some(keys_buf.metal_buffer()), 0, 0);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&cap_u32 as *const u32 as *mut _).unwrap(),
                size_of::<u32>(),
                1,
            );
            enc.dispatchThreads_threadsPerThreadgroup(grid1d(table_cap as usize), group1d());
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
    }

    let state = MetalMapState {
        keys_buf,
        values_buf,
        dist_buf,
        dist_keys_buf,
        scratch,
        dist_cap,
        table_cap,
        resolution: voxel_size,
        sensitivity,
        min_dist,
        scan_number: 0,
        nbtype: _nbtype.as_u32(),
        add_cursor: 0,
        lru_cursor: 0,
        parity_trace_enabled: parity_trace_enabled(),
    };

    *get_state() = Some(state);
}

pub fn add_to_map_impl(
    stream: &Stream,
    cloud_buf: &crate::buffer::MetalBuffer,
    pose_t: &crate::types::Vector3,
    pose_r: &crate::types::Quaternion,
) -> f64 {
    use nalgebra::UnitQuaternion;
    use std::time::Instant;

    let start_all = Instant::now();
    let mut total_gpu_time_ms = 0.0;

    let mut guard = get_state();
    let state = guard.as_mut().expect("Map not initialised");

    state.scan_number += 1;
    let scan_number = state.scan_number;
    let resolution = state.resolution;
    let dist_cap = state.dist_cap;
    let table_cap = state.table_cap;

    // rotation already provided as quaternion; just convert into nalgebra type
    let q: UnitQuaternion<f64> = (*pose_r).into();
    let pose_quat = [q.i as f32, q.j as f32, q.k as f32, q.w as f32];
    let pose_trans = [pose_t.x as f32, pose_t.y as f32, pose_t.z as f32];

    let n_points = cloud_buf.n;
    if n_points == 0 {
        return 0.0;
    }

    if state.scratch.capacity < n_points {
        let new_cap = (n_points as f32 * 1.5) as usize;
        state.scratch.keys = crate::buffer::MetalBuffer::new(stream, size_of::<i32>(), new_cap);
        state.scratch.indices = crate::buffer::MetalBuffer::new(stream, size_of::<u32>(), new_cap);
        state.scratch.radix = crate::radix_sort::RadixSortBuffers::new(stream, new_cap.max(state.dist_cap as usize));
        state.scratch.run_starts =
            crate::buffer::MetalBuffer::new(stream, size_of::<u32>(), new_cap);
        state.scratch.run_ends = crate::buffer::MetalBuffer::new(stream, size_of::<u32>(), new_cap);
        state.scratch.unique_keys =
            crate::buffer::MetalBuffer::new(stream, size_of::<i32>(), new_cap);
        state.scratch.target_indices =
            crate::buffer::MetalBuffer::new(stream, size_of::<u32>(), new_cap);
        state.scratch.update_flags =
            crate::buffer::MetalBuffer::new(stream, size_of::<u32>(), new_cap);
        state.scratch.sorted_points =
            crate::buffer::MetalBuffer::new(stream, size_of::<crate::types::Point>(), new_cap);
        state.scratch.capacity = new_cap;
    }
    if state.scratch.radix.max_elements < n_points.max(state.dist_cap as usize) {
        state.scratch.radix = crate::radix_sort::RadixSortBuffers::new(
            stream,
            n_points.max(state.dist_cap as usize),
        );
    }

    let params = BuildVoxelsParams {
        pose_quat,
        pose_trans,
        _pad0: 0.0,
        resolution,
        scan_number,
        sensitivity: state.sensitivity,
        min_dist: state.min_dist,
        unique_key_count: 0,
        table_cap,
        dist_cap,
        n_points: n_points as u32,
    };
    let counters_host = [0u32, state.add_cursor, state.lru_cursor, 0u32, 0u32];
    state.scratch.counter.upload(stream, &counters_host);
    let map_was_full = state.add_cursor >= state.dist_cap;

    let pipes = get_pipelines(&*stream.device);
    let cb = stream.queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");

    enc.setComputePipelineState(&pipes.calculate_keys);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(cloud_buf.metal_buffer()), 0, 0);
        enc.setBuffer_offset_atIndex(Some(state.scratch.keys.metal_buffer()), 0, 1);
        enc.setBuffer_offset_atIndex(Some(state.scratch.indices.metal_buffer()), 0, 2);
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
            size_of::<BuildVoxelsParams>(),
            3,
        );
        enc.dispatchThreads_threadsPerThreadgroup(grid1d(n_points), group1d());
    }
    enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

    crate::radix_sort::dispatch_radix_sort_u32_pairs(
        stream,
        &enc,
        &state.scratch.keys,
        &state.scratch.indices,
        &state.scratch.radix,
        n_points,
        0x8000_0000,
    );
    enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

    enc.setComputePipelineState(&pipes.rle);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(state.scratch.keys.metal_buffer()), 0, 0);
        enc.setBuffer_offset_atIndex(Some(state.scratch.unique_keys.metal_buffer()), 0, 1);
        enc.setBuffer_offset_atIndex(Some(state.scratch.run_starts.metal_buffer()), 0, 2);
        enc.setBuffer_offset_atIndex(Some(state.scratch.run_ends.metal_buffer()), 0, 3);
        enc.setBuffer_offset_atIndex(Some(state.scratch.counter.metal_buffer()), 0, 4);
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
            size_of::<BuildVoxelsParams>(),
            5,
        );
        enc.dispatchThreads_threadsPerThreadgroup(grid1d(n_points), group1d());
    }
    enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

    if cpu_assign_trace_enabled() {
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        total_gpu_time_ms += (cb.GPUEndTime() - cb.GPUStartTime()) * 1000.0;

        let counters_result = state.scratch.counter.as_slice::<u32>();
        let unique_count = counters_result[0] as usize;
        if unique_count == 0 {
            return total_gpu_time_ms;
        }

        let unique_keys = state.scratch.unique_keys.as_slice::<i32>()[..unique_count].to_vec();
        let table_keys = state.keys_buf.as_slice::<i32>().to_vec();
        let table_vals = state.values_buf.as_slice::<u32>().to_vec();
        let lru_mapping = state.scratch.lru_mapping.as_slice::<u32>().to_vec();
        let mut target_indices = Vec::with_capacity(unique_count);
        let mut update_flags = Vec::with_capacity(unique_count);
        let mut updates = 0u32;
        let mut inserts = 0u32;

        {
            let dist_cap = state.dist_cap;
            let dist_keys = state.dist_keys_buf.as_slice_mut::<i32>();
            for &key in &unique_keys {
                let found = host_hashmap_lookup(&table_keys, &table_vals, state.table_cap, key);
                if found != u32::MAX {
                    updates = updates.saturating_add(1);
                    target_indices.push(found);
                    update_flags.push(1);
                    continue;
                }

                inserts = inserts.saturating_add(1);
                let old_add = state.add_cursor;
                state.add_cursor = state.add_cursor.saturating_add(1);
                let assign_idx = if old_add < dist_cap {
                    old_add
                } else {
                    let lru_idx = state.lru_cursor;
                    state.lru_cursor = state.lru_cursor.saturating_add(1);
                    lru_mapping[(lru_idx % dist_cap) as usize]
                };
                target_indices.push(assign_idx);
                update_flags.push(0);
                dist_keys[assign_idx as usize] = key;
            }
        }

        state.scratch.target_indices.upload(stream, &target_indices);
        state.scratch.update_flags.upload(stream, &update_flags);
        state.scratch.counter.upload(
            stream,
            &[
                unique_count as u32,
                state.add_cursor,
                state.lru_cursor,
                updates,
                inserts,
            ],
        );

        let cb = stream.queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");

        enc.setComputePipelineState(&pipes.reorder_points);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(cloud_buf.metal_buffer()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(state.scratch.sorted_points.metal_buffer()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(state.scratch.indices.metal_buffer()), 0, 2);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                size_of::<BuildVoxelsParams>(),
                3,
            );
            enc.dispatchThreads_threadsPerThreadgroup(grid1d(n_points), group1d());
        }
        enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

        enc.setComputePipelineState(&pipes.build_voxels);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(state.scratch.sorted_points.metal_buffer()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(state.scratch.indices.metal_buffer()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(state.scratch.run_starts.metal_buffer()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(state.scratch.run_ends.metal_buffer()), 0, 3);
            enc.setBuffer_offset_atIndex(Some(state.scratch.unique_keys.metal_buffer()), 0, 4);
            enc.setBuffer_offset_atIndex(Some(state.scratch.target_indices.metal_buffer()), 0, 5);
            enc.setBuffer_offset_atIndex(Some(state.scratch.update_flags.metal_buffer()), 0, 6);
            enc.setBuffer_offset_atIndex(Some(state.dist_buf.metal_buffer()), 0, 7);
            enc.setBuffer_offset_atIndex(Some(state.scratch.counter.metal_buffer()), 0, 8);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                size_of::<BuildVoxelsParams>(),
                9,
            );
            enc.dispatchThreads_threadsPerThreadgroup(grid1d(n_points), group1d());
        }
        enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

        enc.setComputePipelineState(&pipes.hashmap_init);
        let table_cap_u32 = state.table_cap;
        unsafe {
            enc.setBuffer_offset_atIndex(Some(state.keys_buf.metal_buffer()), 0, 0);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&table_cap_u32 as *const u32 as *mut _).unwrap(),
                size_of::<u32>(),
                1,
            );
            enc.dispatchThreads_threadsPerThreadgroup(grid1d(state.table_cap as usize), group1d());
        }
        enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

        enc.setComputePipelineState(&pipes.rebuild_hashmap_keyed);
        let valid_size_u32 = state.dist_cap;
        unsafe {
            enc.setBuffer_offset_atIndex(Some(state.keys_buf.metal_buffer()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(state.values_buf.metal_buffer()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(state.dist_keys_buf.metal_buffer()), 0, 2);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&valid_size_u32 as *const u32 as *mut _).unwrap(),
                size_of::<u32>(),
                3,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&table_cap_u32 as *const u32 as *mut _).unwrap(),
                size_of::<u32>(),
                4,
            );
            enc.dispatchThreads_threadsPerThreadgroup(grid1d(valid_size_u32 as usize), group1d());
        }

        if map_was_full {
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            enc.setComputePipelineState(&pipes.lru_score);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(state.dist_buf.metal_buffer()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(state.scratch.lru_keys.metal_buffer()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(state.scratch.lru_mapping.metal_buffer()), 0, 2);
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                    size_of::<BuildVoxelsParams>(),
                    3,
                );
                enc.dispatchThreads_threadsPerThreadgroup(grid1d(state.dist_cap as usize), group1d());
            }

            crate::radix_sort::dispatch_radix_sort_u32_pairs(
                stream,
                &enc,
                &state.scratch.lru_keys,
                &state.scratch.lru_mapping,
                &state.scratch.radix,
                state.dist_cap as usize,
                0,
            );
        }

        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        total_gpu_time_ms += (cb.GPUEndTime() - cb.GPUStartTime()) * 1000.0;

        let counters_result = state.scratch.counter.as_slice::<u32>();
        let unique_count = counters_result[0] as usize;
        state.add_cursor = counters_result[1].min(state.dist_cap);
        state.lru_cursor = counters_result[2].min(state.dist_cap);
        if state.parity_trace_enabled
            && (scan_number <= 600 && scan_number % 50 == 0 || scan_number % 200 == 0)
        {
            let active = state.add_cursor.min(state.dist_cap) as usize;
            let dist_keys = state.dist_keys_buf.as_slice::<i32>();
            let dist_key_hash = sample_hash_i32(&dist_keys[..active.min(dist_keys.len())]);
            let updates = counters_result[3];
            let inserts = counters_result[4];
            let unique_count = unique_count.min(n_points);
            let sorted_keys = &state.scratch.keys.as_slice::<i32>()[..n_points];
            let unique_keys = &state.scratch.unique_keys.as_slice::<i32>()[..unique_count];
            let run_starts = &state.scratch.run_starts.as_slice::<u32>()[..unique_count];
            let run_ends = &state.scratch.run_ends.as_slice::<u32>()[..unique_count];
            if let Some(problem) = validate_sorted_runs(sorted_keys, unique_keys, run_starts, run_ends)
            {
                log::error!("[Metal-Parity] scan={} run-check FAILED: {}", scan_number, problem);
            }
            log::warn!(
                "[Metal-Parity] scan={} unique={} updates={} inserts={} add_cursor={} lru_cursor={} map_full={} dist_hash={:016x} cpu_assign=1",
                scan_number,
                unique_count,
                updates,
                inserts,
                state.add_cursor,
                state.lru_cursor,
                state.add_cursor >= state.dist_cap,
                dist_key_hash
            );
        }
        log::info!(
            "[Metal] add_to_map: total={:?}, points={}, unique={}, add_cursor={}, lru_cursor={}, gpu_time={:.2}ms",
            start_all.elapsed(),
            n_points,
            unique_count,
            state.add_cursor,
            state.lru_cursor,
            total_gpu_time_ms
        );

        return total_gpu_time_ms;
    }

    enc.setComputePipelineState(&pipes.assign_indices);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(state.scratch.unique_keys.metal_buffer()), 0, 0);
        enc.setBuffer_offset_atIndex(Some(state.keys_buf.metal_buffer()), 0, 1);
        enc.setBuffer_offset_atIndex(Some(state.values_buf.metal_buffer()), 0, 2);
        enc.setBuffer_offset_atIndex(Some(state.dist_keys_buf.metal_buffer()), 0, 3);
        enc.setBuffer_offset_atIndex(Some(state.scratch.lru_mapping.metal_buffer()), 0, 4);
        enc.setBuffer_offset_atIndex(Some(state.scratch.target_indices.metal_buffer()), 0, 5);
        enc.setBuffer_offset_atIndex(Some(state.scratch.update_flags.metal_buffer()), 0, 6);
        enc.setBuffer_offset_atIndex(Some(state.scratch.counter.metal_buffer()), 0, 7);
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
            size_of::<BuildVoxelsParams>(),
            8,
        );
        enc.dispatchThreads_threadsPerThreadgroup(grid1d(n_points), group1d());
    }
    enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

    enc.setComputePipelineState(&pipes.reorder_points);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(cloud_buf.metal_buffer()), 0, 0);
        enc.setBuffer_offset_atIndex(Some(state.scratch.sorted_points.metal_buffer()), 0, 1);
        enc.setBuffer_offset_atIndex(Some(state.scratch.indices.metal_buffer()), 0, 2);
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
            size_of::<BuildVoxelsParams>(),
            3,
        );
        enc.dispatchThreads_threadsPerThreadgroup(grid1d(n_points), group1d());
    }
    enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

    enc.setComputePipelineState(&pipes.build_voxels);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(state.scratch.sorted_points.metal_buffer()), 0, 0);
        enc.setBuffer_offset_atIndex(Some(state.scratch.indices.metal_buffer()), 0, 1);
        enc.setBuffer_offset_atIndex(Some(state.scratch.run_starts.metal_buffer()), 0, 2);
        enc.setBuffer_offset_atIndex(Some(state.scratch.run_ends.metal_buffer()), 0, 3);
        enc.setBuffer_offset_atIndex(Some(state.scratch.unique_keys.metal_buffer()), 0, 4);
        enc.setBuffer_offset_atIndex(Some(state.scratch.target_indices.metal_buffer()), 0, 5);
        enc.setBuffer_offset_atIndex(Some(state.scratch.update_flags.metal_buffer()), 0, 6);
        enc.setBuffer_offset_atIndex(Some(state.dist_buf.metal_buffer()), 0, 7);
        enc.setBuffer_offset_atIndex(Some(state.scratch.counter.metal_buffer()), 0, 8);
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
            size_of::<BuildVoxelsParams>(),
            9,
        );
        enc.dispatchThreads_threadsPerThreadgroup(grid1d(n_points), group1d());
    }
    enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

    enc.setComputePipelineState(&pipes.hashmap_init);
    let table_cap_u32 = state.table_cap;
    unsafe {
        enc.setBuffer_offset_atIndex(Some(state.keys_buf.metal_buffer()), 0, 0);
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(&table_cap_u32 as *const u32 as *mut _).unwrap(),
            size_of::<u32>(),
            1,
        );
        enc.dispatchThreads_threadsPerThreadgroup(grid1d(state.table_cap as usize), group1d());
    }
    enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

    enc.setComputePipelineState(&pipes.rebuild_hashmap_keyed);
    let valid_size_u32 = state.dist_cap;
    unsafe {
        enc.setBuffer_offset_atIndex(Some(state.keys_buf.metal_buffer()), 0, 0);
        enc.setBuffer_offset_atIndex(Some(state.values_buf.metal_buffer()), 0, 1);
        enc.setBuffer_offset_atIndex(Some(state.dist_keys_buf.metal_buffer()), 0, 2);
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(&valid_size_u32 as *const u32 as *mut _).unwrap(),
            size_of::<u32>(),
            3,
        );
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(&table_cap_u32 as *const u32 as *mut _).unwrap(),
            size_of::<u32>(),
            4,
        );
        enc.dispatchThreads_threadsPerThreadgroup(grid1d(valid_size_u32 as usize), group1d());
    }

    if map_was_full {
        enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        enc.setComputePipelineState(&pipes.lru_score);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(state.dist_buf.metal_buffer()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(state.scratch.lru_keys.metal_buffer()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(state.scratch.lru_mapping.metal_buffer()), 0, 2);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                size_of::<BuildVoxelsParams>(),
                3,
            );
            enc.dispatchThreads_threadsPerThreadgroup(grid1d(state.dist_cap as usize), group1d());
        }

        crate::radix_sort::dispatch_radix_sort_u32_pairs(
            stream,
            &enc,
            &state.scratch.lru_keys,
            &state.scratch.lru_mapping,
            &state.scratch.radix,
            state.dist_cap as usize,
            0,
        );
    }

    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    total_gpu_time_ms += (cb.GPUEndTime() - cb.GPUStartTime()) * 1000.0;

    let counters_result = state.scratch.counter.as_slice::<u32>();
    let unique_count = counters_result[0] as usize;
    state.add_cursor = counters_result[1].min(state.dist_cap);
    state.lru_cursor = counters_result[2].min(state.dist_cap);
    if state.parity_trace_enabled
        && (scan_number <= 600 && scan_number % 50 == 0 || scan_number % 200 == 0)
    {
        let active = state.add_cursor.min(state.dist_cap) as usize;
        let dist_keys = state.dist_keys_buf.as_slice::<i32>();
        let dist_key_hash = sample_hash_i32(&dist_keys[..active.min(dist_keys.len())]);
        let updates = counters_result[3];
        let inserts = counters_result[4];
        let unique_count = unique_count.min(n_points);
        let sorted_keys = &state.scratch.keys.as_slice::<i32>()[..n_points];
        let unique_keys = &state.scratch.unique_keys.as_slice::<i32>()[..unique_count];
        let run_starts = &state.scratch.run_starts.as_slice::<u32>()[..unique_count];
        let run_ends = &state.scratch.run_ends.as_slice::<u32>()[..unique_count];
        if let Some(problem) = validate_sorted_runs(sorted_keys, unique_keys, run_starts, run_ends) {
            log::error!("[Metal-Parity] scan={} run-check FAILED: {}", scan_number, problem);
        }
        log::warn!(
            "[Metal-Parity] scan={} unique={} updates={} inserts={} add_cursor={} lru_cursor={} map_full={} dist_hash={:016x}",
            scan_number,
            unique_count,
            updates,
            inserts,
            state.add_cursor,
            state.lru_cursor,
            state.add_cursor >= state.dist_cap,
            dist_key_hash
        );
    }
    log::info!(
        "[Metal] add_to_map: total={:?}, points={}, unique={}, add_cursor={}, lru_cursor={}, gpu_time={:.2}ms",
        start_all.elapsed(),
        n_points,
        unique_count,
        state.add_cursor,
        state.lru_cursor,
        total_gpu_time_ms
    );

    total_gpu_time_ms
}

#[cfg(test)]
pub(crate) fn acquire_test_lock() -> std::sync::MutexGuard<'static, ()> {
    match TEST_MAP_MUTEX.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn cpu_rle(sorted_keys: &[i32]) -> (Vec<i32>, Vec<u32>, Vec<u32>) {
        let mut unique_keys = Vec::new();
        let mut run_starts = Vec::new();
        let mut run_ends = Vec::new();

        for (idx, &key) in sorted_keys.iter().enumerate() {
            if key == EMPTY_KEY {
                continue;
            }
            if idx > 0 && sorted_keys[idx - 1] == key {
                continue;
            }
            let end = sorted_keys.partition_point(|candidate| *candidate <= key);
            unique_keys.push(key);
            run_starts.push(idx as u32);
            run_ends.push(end as u32);
        }

        (unique_keys, run_starts, run_ends)
    }

    fn cpu_assign_indices_vulkan_style(
        unique_keys: &[i32],
        existing: &HashMap<i32, u32>,
        dist_cap: u32,
        add_cursor: u32,
        lru_cursor: u32,
        lru_mapping: &[u32],
    ) -> (Vec<u32>, Vec<u32>, Vec<i32>, u32, u32) {
        let mut target_indices = Vec::with_capacity(unique_keys.len());
        let mut update_flags = Vec::with_capacity(unique_keys.len());
        let mut dist_keys = vec![EMPTY_KEY; dist_cap as usize];
        let mut add_cursor = add_cursor;
        let mut lru_cursor = lru_cursor;

        for &key in unique_keys {
            if let Some(&idx) = existing.get(&key) {
                target_indices.push(idx);
                update_flags.push(1);
                continue;
            }

            let old_add = add_cursor;
            add_cursor = add_cursor.saturating_add(1);
            let idx = if old_add < dist_cap {
                old_add
            } else {
                let map_idx = lru_mapping[(lru_cursor % dist_cap) as usize];
                lru_cursor = lru_cursor.saturating_add(1);
                map_idx
            };

            dist_keys[idx as usize] = key;
            target_indices.push(idx);
            update_flags.push(0);
        }

        (target_indices, update_flags, dist_keys, add_cursor, lru_cursor)
    }

    #[test]
    fn cpu_sort_matches_gpu_semantics() {
        let _lock = acquire_test_lock();
        let keys: Vec<i32> = vec![3, -1, 2, -1, 0, i32::MIN, i32::MAX];
        let idxs: Vec<u32> = (0..keys.len() as u32).collect();

        // apply the same code used in add_to_map_impl
        let n_points = keys.len();
        let mut keys_slice = keys.clone();
        let mut inds_slice = idxs.clone();

        // sort
        let mut packed: Vec<u64> = Vec::with_capacity(n_points);
        for i in 0..n_points {
            let k_u = (keys_slice[i] as u32) ^ 0x8000_0000;
            let idx = inds_slice[i];
            packed.push(((k_u as u64) << 32) | (idx as u64));
        }
        packed.sort_unstable();

        for (i, p) in packed.into_iter().enumerate() {
            let k_u = ((p >> 32) as u32) ^ 0x8000_0000;
            let idx = (p & 0xFFFFFFFF) as u32;
            keys_slice[i] = k_u as i32;
            inds_slice[i] = idx;
        }

        // now compute expected using straightforward packing+sort directly
        let mut expected: Vec<(i32, u32)> =
            keys.iter().copied().zip(idxs.iter().copied()).collect();
        expected.sort_unstable_by_key(|&(k, idx)| {
            let k_u = (k as u32) ^ 0x8000_0000;
            ((k_u as u64) << 32) | (idx as u64)
        });

        for i in 0..n_points {
            assert_eq!(keys_slice[i], expected[i].0);
            assert_eq!(inds_slice[i], expected[i].1);
        }
    }

    #[test]
    fn cpu_rle_matches_expected_runs() {
        let _lock = acquire_test_lock();
        let sorted_keys = vec![-9, -9, 1, 1, 4, 7, 7, 9];
        let (unique, starts, ends) = cpu_rle(&sorted_keys);
        assert_eq!(unique, vec![-9, 1, 4, 7, 9]);
        assert_eq!(starts, vec![0, 2, 4, 5, 7]);
        assert_eq!(ends, vec![2, 4, 5, 7, 8]);
    }

    #[test]
    fn cpu_assign_indices_uses_updates_then_lru_victims() {
        let _lock = acquire_test_lock();
        let existing = HashMap::from([(11, 3u32)]);
        let lru_mapping = vec![2u32, 1u32, 0u32, 3u32];
        let (targets, flags, dist_keys, add_cursor, lru_cursor) =
            cpu_assign_indices_vulkan_style(&[11, 22, 33, 44], &existing, 4, 4, 0, &lru_mapping);

        assert_eq!(targets, vec![3, 2, 1, 0]);
        assert_eq!(flags, vec![1, 0, 0, 0]);
        assert_eq!(dist_keys[1], 33);
        assert_eq!(add_cursor, 7);
        assert_eq!(lru_cursor, 3);
    }

    #[test]
    fn gpu_radix_sort_matches_cpu_reference() {
        let _lock = acquire_test_lock();
        let stream = crate::stream::Stream::new();
        let keys: Vec<i32> = vec![3, -1, 2, -1, 0, i32::MIN, i32::MAX, 7, -9, 3, 2];
        let idxs: Vec<u32> = (0..keys.len() as u32).collect();

        let mut key_buf = crate::buffer::MetalBuffer::new(&stream, size_of::<i32>(), keys.len());
        key_buf.as_slice_mut::<i32>().copy_from_slice(&keys);
        let mut idx_buf = crate::buffer::MetalBuffer::new(&stream, size_of::<u32>(), idxs.len());
        idx_buf.as_slice_mut::<u32>().copy_from_slice(&idxs);
        let radix = crate::radix_sort::RadixSortBuffers::new(&stream, keys.len());

        let cb = stream.queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        crate::radix_sort::dispatch_radix_sort_u32_pairs(
            &stream,
            &enc,
            &key_buf,
            &idx_buf,
            &radix,
            keys.len(),
            0x8000_0000,
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();

        let got_keys = key_buf.as_slice::<i32>().to_vec();
        let got_idxs = idx_buf.as_slice::<u32>().to_vec();

        let mut expected: Vec<(i32, u32)> =
            keys.iter().copied().zip(idxs.iter().copied()).collect();
        expected.sort_by_key(|&(k, idx)| (((k as u32) ^ 0x8000_0000) as u64) << 32 | idx as u64);

        for (i, (exp_k, exp_idx)) in expected.into_iter().enumerate() {
            assert_eq!(got_keys[i], exp_k, "key mismatch at {i}");
            assert_eq!(got_idxs[i], exp_idx, "idx mismatch at {i}");
        }
    }
}

pub fn get_map_impl() -> Vec<nalgebra::Point3<f32>> {
    let guard = get_state();
    let state = match guard.as_ref() {
        Some(s) => s,
        None => return vec![],
    };

    let valid = (state.add_cursor as usize).min(state.dist_cap as usize);

    let dists = state.dist_buf.as_slice::<GpuVoxelDist>();
    let dists = &dists[..valid];

    dists
        .iter()
        .filter(|d| d.point_count > 0)
        .map(|d| nalgebra::Point3::new(d.mean[0], d.mean[1], d.mean[2]))
        .collect()
}

pub fn get_map_size_impl() -> u32 {
    let guard = get_state();
    match guard.as_ref() {
        Some(s) => s.add_cursor.min(s.dist_cap),
        None => 0,
    }
}

pub fn cleanup_map_impl() {
    *get_state() = None;
}

pub fn planes_metal_impl(
    stream: &Stream,
    cloud: &crate::buffer::MetalBuffer,
    ekf_pose_t: &crate::types::Vector3,
    ekf_pose_r: &crate::types::Quaternion,
    ekf_imu_offset_t: &crate::types::Vector3,
    ekf_imu_offset_r: &crate::types::Quaternion,
    out_buff: &mut Vec<ros_pointcloud2::points::PointXYZINormal>,
    out_buff_gpu: &mut crate::buffer::MetalBuffer,
) -> f64 {
    use nalgebra::UnitQuaternion;
    use std::time::Instant;

    let t_all = Instant::now();

    let mut guard = get_state();
    let state = guard.as_mut().expect("Map not initialised");

    // quaternion inputs already
    let ext_q: UnitQuaternion<f64> = (*ekf_imu_offset_r).into();
    let pose_q: UnitQuaternion<f64> = (*ekf_pose_r).into();

    let lidar_to_imu_quat = [
        ext_q.i as f32,
        ext_q.j as f32,
        ext_q.k as f32,
        ext_q.w as f32,
    ];
    let lidar_to_imu_pos = [
        ekf_imu_offset_t.x as f32,
        ekf_imu_offset_t.y as f32,
        ekf_imu_offset_t.z as f32,
    ];
    let pose_quat = [
        pose_q.i as f32,
        pose_q.j as f32,
        pose_q.k as f32,
        pose_q.w as f32,
    ];
    let pose_pos = [
        ekf_pose_t.x as f32,
        ekf_pose_t.y as f32,
        ekf_pose_t.z as f32,
    ];

    unsafe {
        *(out_buff_gpu.buf.contents().as_ptr().cast::<u32>()) = 0;
        *(state.scratch.counter.buf.contents().as_ptr().cast::<u32>()) = 0;
    }

    let params = PlanesParams {
        lidar_to_imu_quat,
        lidar_to_imu_pos,
        _pad0: 0.0,
        pose_quat,
        pose_pos,
        _pad1: 0.0,
        cloud_size: cloud.n as u32,
        table_cap: state.table_cap,
        resolution: state.resolution,
        sphericity_threshold: 0.5,
        nbtype: state.nbtype,
    };

    let t_gpu = Instant::now();
    let gpu_time_ms = {
        let pipes = get_pipelines(&*stream.device);
        let cb = stream.queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        enc.setComputePipelineState(&pipes.plane_krnl);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(cloud.metal_buffer()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(state.keys_buf.metal_buffer()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(state.values_buf.metal_buffer()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(state.dist_buf.metal_buffer()), 0, 3);
            enc.setBuffer_offset_atIndex(Some(state.scratch.counter.metal_buffer()), 0, 4);
            enc.setBuffer_offset_atIndex(Some(out_buff_gpu.metal_buffer()), 0, 5);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                size_of::<PlanesParams>(),
                6,
            );
            enc.dispatchThreads_threadsPerThreadgroup(grid1d(cloud.n), group1d());
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
        let start = cb.GPUStartTime();
        let end = cb.GPUEndTime();
        (end - start) * 1000.0
    };
    let dur_gpu = t_gpu.elapsed();

    let t_copy = Instant::now();
    let out_size =
        unsafe { *(state.scratch.counter.buf.contents().as_ptr().cast::<u32>()) } as usize;
    let gpu_normals = out_buff_gpu.as_slice::<crate::types::PointNormal>();
    let gpu_normals = &gpu_normals[..out_size.min(out_buff.len())];

    unsafe { out_buff.set_len(out_size.min(out_buff.capacity())) };

    #[cfg(feature = "metal_enable")]
    {
        out_buff
            .par_iter_mut()
            .zip(gpu_normals.par_iter())
            .for_each(|(dst, src)| {
                *dst = (*src).into();
            });
    }
    #[cfg(not(feature = "metal_enable"))]
    {
        for (dst, src) in out_buff.iter_mut().zip(gpu_normals.iter()) {
            *dst = (*src).into();
        }
    }
    let dur_copy = t_copy.elapsed();

    log::info!(
        "[Metal] planes: total={:?}, gpu={:?}, gpu_time={:.2}ms, copy={:?}, size={}",
        t_all.elapsed(),
        dur_gpu,
        gpu_time_ms,
        dur_copy,
        out_size
    );

    gpu_time_ms
}

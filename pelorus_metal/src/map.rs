use bytemuck::{Pod, Zeroable};
#[cfg(feature = "metal_enable")]
use rayon::prelude::*;
use std::collections::HashMap;
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
const EMPTY_KEY: i32 = i32::MIN;

#[derive(Clone, Copy, Default)]
struct SortKey {
    index: u32,
    temporal_dist: u64,
    point_count: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BuildVoxelsParams {
    pub pose_quat: [f32; 4],   // 16 bytes, offset 0
    pub pose_trans: [f32; 3],  // 12 bytes, offset 16
    pub _pad0: f32,            // 4 bytes, offset 28
    pub resolution: f32,       // 4 bytes, offset 32
    pub scan_number: u32,      // 4 bytes, offset 36
    pub sensitivity: f32,      // 4 bytes, offset 40
    pub min_dist: f32,         // 4 bytes, offset 44
    pub unique_key_count: u32, // 4 bytes, offset 48
    pub table_cap: u32,        // 4 bytes, offset 52
    pub dist_cap: u32,         // 4 bytes, offset 56
    pub _pad1: u32,            // 4 bytes, offset 60 -> Total 64
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
    pub packed: crate::buffer::MetalBuffer,
    pub run_starts: crate::buffer::MetalBuffer,
    pub run_ends: crate::buffer::MetalBuffer,
    pub unique_keys: crate::buffer::MetalBuffer,
    pub target_indices: crate::buffer::MetalBuffer,
    pub update_flags: crate::buffer::MetalBuffer,
    pub sorted_points: crate::buffer::MetalBuffer,
    pub counter: crate::buffer::MetalBuffer,
    pub capacity: usize,
}

pub struct MetalMapState {
    pub keys_buf: crate::buffer::MetalBuffer,
    pub values_buf: crate::buffer::MetalBuffer,
    pub dist_buf: crate::buffer::MetalBuffer,
    pub dist_keys_buf: crate::buffer::MetalBuffer,
    pub add_cursor_buf: crate::buffer::MetalBuffer,
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
    lru_mapping: Vec<u32>,
    lru_mapping_keys: Vec<SortKey>,
    key_to_index: HashMap<i32, u32>,
    pending_lru_rebuild: bool,
    pending_lru_scan_number: u32,
}

static MAP_STATE: OnceLock<Mutex<Option<MetalMapState>>> = OnceLock::new();

fn global_state() -> &'static Mutex<Option<MetalMapState>> {
    MAP_STATE.get_or_init(|| Mutex::new(None))
}

pub fn get_state() -> MutexGuard<'static, Option<MetalMapState>> {
    global_state().lock().unwrap()
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
    let add_cursor_buf = crate::buffer::MetalBuffer::new(stream, size_of::<u32>(), 1);

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
        std::ptr::write_bytes(
            add_cursor_buf.buf.contents().as_ptr().cast::<u8>(),
            0,
            add_cursor_buf.buf.length() as usize,
        );
    }

    // Initial scratch buffers with capacity 1; these will be resized on demand in add_to_map_impl.
    // We just need valid buffers to pass to the GPU during init.
    let scratch_cap = 1;
    let scratch = ScratchBuffers {
        keys: crate::buffer::MetalBuffer::new(stream, size_of::<i32>(), scratch_cap),
        indices: crate::buffer::MetalBuffer::new(stream, size_of::<u32>(), scratch_cap),
        packed: crate::buffer::MetalBuffer::new(stream, size_of::<u64>(), scratch_cap),
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
        counter: crate::buffer::MetalBuffer::new(stream, size_of::<u32>(), 1),
        // keep the logical capacity zero so resize logic still fires immediately
        capacity: 0,
    };

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
        add_cursor_buf,
        scratch,
        dist_cap,
        table_cap,
        resolution: voxel_size,
        sensitivity,
        min_dist,
        scan_number: 0,
        nbtype: _nbtype as u32,
        add_cursor: 0,
        lru_cursor: 0,
        lru_mapping: (0..dist_cap).collect(),
        lru_mapping_keys: vec![SortKey::default(); dist_cap as usize],
        key_to_index: HashMap::with_capacity(dist_cap as usize),
        pending_lru_rebuild: false,
        pending_lru_scan_number: 0,
    };

    *get_state() = Some(state);
}

fn assign_target_indices_with_cpu_lru(
    state: &mut MetalMapState,
    unique_keys: &[i32],
    target_indices: &mut Vec<u32>,
    update_flags: &mut Vec<u32>,
) {
    target_indices.clear();
    update_flags.clear();
    target_indices.reserve(unique_keys.len());
    update_flags.reserve(unique_keys.len());

    let dist_cap = state.dist_cap;
    let dist_keys_mut = state.dist_keys_buf.as_slice_mut::<i32>();
    for &key in unique_keys {
        if let Some(&idx) = state.key_to_index.get(&key) {
            target_indices.push(idx);
            update_flags.push(1);
            continue;
        }

        if state.add_cursor < dist_cap {
            let idx = state.add_cursor;
            state.add_cursor += 1;
            dist_keys_mut[idx as usize] = key;
            state.key_to_index.insert(key, idx);
            target_indices.push(idx);
            update_flags.push(0);
            continue;
        }

        if state.lru_cursor >= dist_cap {
            log::warn!("Replacing too many points inside LRU Cache. Increase the map capacity!");
            break;
        }

        let victim = state.lru_mapping[state.lru_cursor as usize];
        state.lru_cursor += 1;

        let old_key = dist_keys_mut[victim as usize];
        if old_key != EMPTY_KEY {
            state.key_to_index.remove(&old_key);
        }
        dist_keys_mut[victim as usize] = key;
        state.key_to_index.insert(key, victim);

        target_indices.push(victim);
        update_flags.push(0);
    }

    state.add_cursor = state.add_cursor.min(dist_cap);
    let add_cursor_ptr = state.add_cursor_buf.buf.contents().as_ptr().cast::<u32>();
    unsafe {
        *add_cursor_ptr = state.add_cursor;
    }
}

fn rebuild_cpu_lru_order(state: &mut MetalMapState, scan_number: u32) {
    if state.add_cursor < state.dist_cap {
        return;
    }

    let dists = state.dist_buf.as_slice::<GpuVoxelDist>();
    for (i, &idx) in state.lru_mapping.iter().enumerate() {
        let item = &dists[idx as usize];
        state.lru_mapping_keys[i] = SortKey {
            index: idx,
            temporal_dist: scan_number.saturating_sub(item.last_scan) as u64,
            point_count: item.point_count as u64,
        };
    }

    state.lru_mapping_keys.sort_unstable_by(|a, b| {
        if a.point_count == 0 && b.point_count == 0 {
            return std::cmp::Ordering::Equal;
        }
        if a.point_count == 0 {
            return std::cmp::Ordering::Less;
        }
        if b.point_count == 0 {
            return std::cmp::Ordering::Greater;
        }

        let left = (a.temporal_dist as u128) * (b.point_count as u128);
        let right = (b.temporal_dist as u128) * (a.point_count as u128);
        right.cmp(&left)
    });

    for (i, key) in state.lru_mapping_keys.iter().enumerate() {
        state.lru_mapping[i] = key.index;
    }
    state.lru_cursor = 0;
}

pub fn add_to_map_impl(
    stream: &Stream,
    cloud_buf: &crate::buffer::MetalBuffer,
    pose_t: &crate::types::Vector3,
    pose_r: &crate::types::Quaternion,
) {
    use nalgebra::UnitQuaternion;
    use std::time::Instant;

    let start_all = Instant::now();

    // Rebuild CPU-side LRU ordering from the previous scan once GPU writes are complete.
    // This removes a forced wait at the end of each frame while keeping identical eviction logic.
    {
        let mut guard = get_state();
        let state = guard.as_mut().expect("Map not initialised");
        if state.pending_lru_rebuild {
            stream.sync();
            rebuild_cpu_lru_order(state, state.pending_lru_scan_number);
            state.pending_lru_rebuild = false;
        }
    }

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
        return;
    }

    if state.scratch.capacity < n_points {
        let new_cap = (n_points as f32 * 1.5) as usize; // Slight over-allocation
        state.scratch.keys = crate::buffer::MetalBuffer::new(stream, size_of::<i32>(), new_cap);
        state.scratch.indices = crate::buffer::MetalBuffer::new(stream, size_of::<u32>(), new_cap);
        state.scratch.packed = crate::buffer::MetalBuffer::new(stream, size_of::<u64>(), new_cap);
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

    let params = BuildVoxelsParams {
        pose_quat,
        pose_trans,
        _pad0: 0.0,
        resolution,
        scan_number,
        sensitivity: state.sensitivity,
        min_dist: state.min_dist,
        unique_key_count: n_points as u32,
        table_cap,
        dist_cap,
        _pad1: 0,
    };

    let t_keys = Instant::now();

    let pipes = get_pipelines(&*stream.device);
    let cb = stream.queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(&pipes.calculate_keys);
    let mut pk = params;
    pk.unique_key_count = n_points as u32; // cloud size
    unsafe {
        enc.setBuffer_offset_atIndex(Some(cloud_buf.metal_buffer()), 0, 0);
        enc.setBuffer_offset_atIndex(Some(state.scratch.keys.metal_buffer()), 0, 1);
        enc.setBuffer_offset_atIndex(Some(state.scratch.indices.metal_buffer()), 0, 2);
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(&pk as *const _ as *mut _).unwrap(),
            size_of::<BuildVoxelsParams>(),
            3,
        );
        enc.dispatchThreads_threadsPerThreadgroup(grid1d(n_points), group1d());
        enc.endEncoding();
    }
    cb.commit();
    let dur_keys = t_keys.elapsed();

    let t_sort = Instant::now();

    let padded_n = n_points.next_power_of_two();
    let pipes = get_pipelines(&*stream.device);
    let cb = stream.queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");

    enc.setComputePipelineState(&pipes.pack_keys);
    let n_u32 = n_points as u32;
    unsafe {
        enc.setBuffer_offset_atIndex(Some(state.scratch.keys.metal_buffer()), 0, 0);
        enc.setBuffer_offset_atIndex(Some(state.scratch.indices.metal_buffer()), 0, 1);
        enc.setBuffer_offset_atIndex(Some(state.scratch.packed.metal_buffer()), 0, 2);
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(),
            size_of::<u32>(),
            3,
        );
        enc.dispatchThreads_threadsPerThreadgroup(grid1d(n_points), group1d());
    }

    if n_points < padded_n {
        enc.setComputePipelineState(&pipes.init_pad);
        let diff = (padded_n - n_points) as u32;
        let cap_u32 = state.scratch.capacity as u32;
        unsafe {
            enc.setBuffer_offset_atIndex(Some(state.scratch.packed.metal_buffer()), 0, 0);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(),
                size_of::<u32>(),
                1,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&cap_u32 as *const u32 as *mut _).unwrap(),
                size_of::<u32>(),
                2,
            );
            enc.dispatchThreads_threadsPerThreadgroup(grid1d(diff as usize), group1d());
        }
    }

    // Barrier: pack/pad wrote packed buffer, bitonic reads+writes it
    enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
    enc.setComputePipelineState(&pipes.bitonic_step);
    let mut k = 2;
    while k <= padded_n {
        let mut j = k >> 1;
        while j > 0 {
            let k_u32 = k as u32;
            let j_u32 = j as u32;
            unsafe {
                enc.setBuffer_offset_atIndex(Some(state.scratch.packed.metal_buffer()), 0, 0);
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&j_u32 as *const u32 as *mut _).unwrap(),
                    size_of::<u32>(),
                    1,
                );
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&k_u32 as *const u32 as *mut _).unwrap(),
                    size_of::<u32>(),
                    2,
                );
                enc.dispatchThreads_threadsPerThreadgroup(grid1d(padded_n), group1d());
            }
            // Barrier between each bitonic step (read-write dependency on packed)
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            j >>= 1;
        }
        k <<= 1;
    }

    enc.setComputePipelineState(&pipes.unpack_keys);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(state.scratch.packed.metal_buffer()), 0, 0);
        enc.setBuffer_offset_atIndex(Some(state.scratch.keys.metal_buffer()), 0, 1);
        enc.setBuffer_offset_atIndex(Some(state.scratch.indices.metal_buffer()), 0, 2);
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(),
            size_of::<u32>(),
            3,
        );
        enc.dispatchThreads_threadsPerThreadgroup(grid1d(n_points), group1d());
    }

    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();

    let sorted_keys_gpu = state.scratch.keys.as_slice::<i32>();
    let sorted_keys = &sorted_keys_gpu[..n_points];

    let mut unique_keys: Vec<i32> = Vec::new();
    let mut run_starts: Vec<u32> = Vec::new();
    let mut run_ends: Vec<u32> = Vec::new();

    if !sorted_keys.is_empty() {
        let mut current_key = sorted_keys[0];
        let mut start_idx = 0;

        for (i, &key) in sorted_keys.iter().enumerate().skip(1) {
            if key != current_key {
                unique_keys.push(current_key);
                run_starts.push(start_idx as u32);
                run_ends.push(i as u32);
                current_key = key;
                start_idx = i;
            }
        }
        unique_keys.push(current_key);
        run_starts.push(start_idx as u32);
        run_ends.push(n_points as u32);
    }

    let dur_sort = t_sort.elapsed();

    let mut target_indices: Vec<u32> = Vec::with_capacity(unique_keys.len());
    let mut update_flags: Vec<u32> = Vec::with_capacity(unique_keys.len());
    assign_target_indices_with_cpu_lru(state, &unique_keys, &mut target_indices, &mut update_flags);
    let n_assigned = target_indices.len();
    if n_assigned == 0 {
        return;
    }

    unique_keys.truncate(n_assigned);
    run_starts.truncate(n_assigned);
    run_ends.truncate(n_assigned);

    let copy_to = |dst: &crate::buffer::MetalBuffer, src: &[u8]| unsafe {
        std::ptr::copy_nonoverlapping(
            src.as_ptr(),
            dst.buf.contents().as_ptr().cast::<u8>(),
            src.len(),
        );
    };
    copy_to(
        &state.scratch.run_starts,
        bytemuck::cast_slice::<u32, u8>(&run_starts),
    );
    copy_to(
        &state.scratch.run_ends,
        bytemuck::cast_slice::<u32, u8>(&run_ends),
    );
    copy_to(
        &state.scratch.unique_keys,
        bytemuck::cast_slice::<i32, u8>(&unique_keys),
    );
    copy_to(
        &state.scratch.target_indices,
        bytemuck::cast_slice::<u32, u8>(&target_indices),
    );
    copy_to(
        &state.scratch.update_flags,
        bytemuck::cast_slice::<u32, u8>(&update_flags),
    );

    let t_build = Instant::now();

    {
        let pipes = get_pipelines(&*stream.device);
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

        // Barrier: reorder_points writes sorted_points, build_voxels reads it
        enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

        let mut pb = params;
        pb.unique_key_count = n_assigned as u32;

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
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&pb as *const _ as *mut _).unwrap(),
                size_of::<BuildVoxelsParams>(),
                8,
            );
            enc.dispatchThreads_threadsPerThreadgroup(grid1d(n_assigned), group1d());
        }

        // Barrier: build_voxels writes dists/dist_keys, hashmap rebuild reads them
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

        // Barrier: hashmap_init writes table_keys, rebuild reads them
        enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

        enc.setComputePipelineState(&pipes.rebuild_hashmap_keyed);
        let valid_size_u32 = state.add_cursor.min(state.dist_cap);
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

        enc.endEncoding();
        cb.commit();
    }
    let dur_build = t_build.elapsed();
    state.pending_lru_rebuild = true;
    state.pending_lru_scan_number = scan_number;

    log::info!(
        "[Metal] add_to_map: total={:?}, keys={:?}, sort={:?}, build+rebuild_submit={:?}",
        start_all.elapsed(),
        dur_keys,
        dur_sort,
        dur_build
    );
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
            let k_u = keys_slice[i] as u32;
            let idx = inds_slice[i];
            packed.push(((k_u as u64) << 32) | (idx as u64));
        }
        packed.sort_unstable();

        for (i, p) in packed.into_iter().enumerate() {
            let k_u = (p >> 32) as u32;
            let idx = (p & 0xFFFFFFFF) as u32;
            keys_slice[i] = k_u as i32;
            inds_slice[i] = idx;
        }

        // now compute expected using straightforward packing+sort directly
        let mut expected: Vec<(i32, u32)> =
            keys.iter().copied().zip(idxs.iter().copied()).collect();
        expected.sort_unstable_by_key(|&(k, idx)| {
            let k_u = k as u32;
            ((k_u as u64) << 32) | (idx as u64)
        });

        for i in 0..n_points {
            assert_eq!(keys_slice[i], expected[i].0);
            assert_eq!(inds_slice[i], expected[i].1);
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
) {
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
    {
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
    }
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
        "[Metal] planes: total={:?}, gpu={:?}, copy={:?}, size={}",
        t_all.elapsed(),
        dur_gpu,
        dur_copy,
        out_size
    );
}

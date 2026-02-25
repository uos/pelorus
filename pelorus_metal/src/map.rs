use bytemuck::{Pod, Zeroable};
use metal::MTLResourceOptions;
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::pipeline::{get_pipelines, group1d, grid1d};
use crate::stream::Stream;
use crate::types::NeighborType;

// ── GPU-side structs (must exactly match MSL layout) ─────────────────────────

/// Hash table entry: key is the int32 voxel hash, value is the index into
/// the distribution buffer.  EMPTY_KEY = i32::MIN indicates an empty slot.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub struct KeyValuePair {
    pub key: i32,
    pub value: u32,
}

/// Voxel distribution stored in the GPU buffer.
///
/// Layout (92 bytes, all 4-byte aligned — matches the MSL `VoxelDistribution`):
/// ```text
/// mean[3]:     12 bytes  offset  0
/// point_count:  4 bytes  offset 12
/// cov[9]:      36 bytes  offset 16
/// last_scan:    4 bytes  offset 52
/// L[9]:        36 bytes  offset 56
/// total:       92 bytes
/// ```
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuVoxelDist {
    pub mean: [f32; 3],
    pub point_count: u32,
    pub cov: [f32; 9],
    pub last_scan: u32,
    pub l: [f32; 9],  // Cholesky factor
}

// Compile-time size assertion
const _: () = assert!(size_of::<GpuVoxelDist>() == 92);

// ── Build-voxels params struct (matches MSL BuildVoxelsParams) ────────────────
/// Parameters passed to the `build_voxels` kernel.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BuildVoxelsParams {
    pub pose_quat: [f32; 4],       // quaternion (x,y,z,w)
    pub pose_trans: [f32; 3],      // translation
    pub resolution: f32,
    pub scan_number: u32,
    pub sensitivity: f32,
    pub min_dist: f32,
    pub unique_key_count: u32,
    pub table_cap: u32,
    pub dist_cap: u32,
    pub _pad: u32,
}

// ── Planes params struct (matches MSL PlanesParams) ───────────────────────────
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PlanesParams {
    pub lidar_to_imu_quat: [f32; 4],
    pub lidar_to_imu_pos: [f32; 3],
    pub _pad0: f32,
    pub pose_quat: [f32; 4],
    pub pose_pos: [f32; 3],
    pub _pad1: f32,
    pub cloud_size: u32,
    pub table_cap: u32,
    pub resolution: f32,
    pub sphericity_threshold: f32,
}

// ── Global state ─────────────────────────────────────────────────────────────

pub struct MetalMapState {
    /// Hash table: `table_cap` KeyValuePair entries.
    /// We store keys and values as separate flat arrays for atomic access.
    /// keys_buf: [i32; table_cap]
    pub keys_buf: metal::Buffer,
    /// values_buf: [u32; table_cap]
    pub values_buf: metal::Buffer,
    /// Distribution buffer: [GpuVoxelDist; dist_cap]
    pub dist_buf: metal::Buffer,
    /// Companion key buffer: [i32; dist_cap] (which voxel key each dist slot holds)
    pub dist_keys_buf: metal::Buffer,
    /// Atomic cursor for claiming dist buffer slots.
    pub add_cursor_buf: metal::Buffer,
    pub dist_cap: u32,
    pub table_cap: u32,
    pub resolution: f32,
    pub sensitivity: f32,
    pub min_dist: f32,
    pub nbtype: NeighborType,
    pub scan_number: u32,
}

static MAP_STATE: OnceLock<Mutex<Option<MetalMapState>>> = OnceLock::new();

fn global_state() -> &'static Mutex<Option<MetalMapState>> {
    MAP_STATE.get_or_init(|| Mutex::new(None))
}

pub fn get_state() -> MutexGuard<'static, Option<MetalMapState>> {
    global_state().lock().unwrap()
}

// ── Initialise the map ───────────────────────────────────────────────────────

pub fn init_map_impl(
    stream: &Stream,
    capacity: usize,
    voxel_size: f32,
    sensitivity: f32,
    min_dist: f32,
    nbtype: NeighborType,
) {
    let capacity = capacity.next_power_of_two();
    let dist_cap = capacity as u32;
    let table_cap = dist_cap * 2;

    let alloc = |bytes: usize| -> metal::Buffer {
        let buf = stream.device.new_buffer(
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes) };
        buf
    };

    let keys_buf = alloc(table_cap as usize * size_of::<i32>());
    let values_buf = alloc(table_cap as usize * size_of::<u32>());
    let dist_buf = alloc(dist_cap as usize * size_of::<GpuVoxelDist>());
    let dist_keys_buf = alloc(dist_cap as usize * size_of::<i32>());
    let add_cursor_buf = alloc(size_of::<u32>());

    // Fill hash table keys with EMPTY_KEY via GPU kernel
    {
        let pipes = get_pipelines(&stream.device);
        let cb = stream.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipes.hashmap_init);
        enc.set_buffer(0, Some(&keys_buf), 0);
        let cap_u32 = table_cap;
        enc.set_bytes(1, size_of::<u32>() as u64, &cap_u32 as *const u32 as *const _);
        enc.dispatch_threads(grid1d(table_cap as usize), group1d());
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Also init dist_keys with EMPTY_KEY
    {
        let slice = unsafe {
            std::slice::from_raw_parts_mut(
                dist_keys_buf.contents() as *mut i32,
                dist_cap as usize,
            )
        };
        slice.fill(i32::MIN);
    }

    let state = MetalMapState {
        keys_buf,
        values_buf,
        dist_buf,
        dist_keys_buf,
        add_cursor_buf,
        dist_cap,
        table_cap,
        resolution: voxel_size,
        sensitivity,
        min_dist,
        nbtype,
        scan_number: 0,
    };

    *get_state() = Some(state);
}

// ── Add points to the map ─────────────────────────────────────────────────────

pub fn add_to_map_impl(
    stream: &Stream,
    cloud_buf: &crate::buffer::MetalBuffer,
    pose_t: &crate::types::Vector3,
    pose_r: &crate::types::Matrix3,
) {
    use nalgebra::UnitQuaternion;

    let mut guard = get_state();
    let state = guard.as_mut().expect("Map not initialised — call init_map first");

    state.scan_number += 1;
    let scan_number = state.scan_number;
    let resolution = state.resolution;
    let dist_cap = state.dist_cap;
    let table_cap = state.table_cap;

    // ── Convert pose to quaternion (f32) ──────────────────────────────────────
    let rot_f64: nalgebra::Matrix3<f64> = (*pose_r).into();
    let q = UnitQuaternion::from_matrix(&rot_f64);
    let pose_quat = [q.i as f32, q.j as f32, q.k as f32, q.w as f32];
    let pose_trans = [pose_t.x as f32, pose_t.y as f32, pose_t.z as f32];

    let n_points = cloud_buf.n;
    if n_points == 0 {
        return;
    }

    // ── CPU: compute per-point voxel keys and sort ────────────────────────────
    // Points are in shared memory — read directly.
    let points: &[crate::types::Point] = unsafe {
        std::slice::from_raw_parts(
            cloud_buf.buf.contents() as *const crate::types::Point,
            n_points,
        )
    };

    // Transform each point to global frame on CPU; compute voxel key
    let qv = nalgebra::Vector3::new(pose_quat[0] as f64, pose_quat[1] as f64, pose_quat[2] as f64);
    let qw = pose_quat[3] as f64;
    let trans_v = nalgebra::Vector3::new(
        pose_trans[0] as f64, pose_trans[1] as f64, pose_trans[2] as f64,
    );

    let mut keyed: Vec<(i32, u32)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let local = nalgebra::Vector3::new(p.x as f64, p.y as f64, p.z as f64);
            // Quaternion rotate: v + 2w(u×v) + 2(u×(u×v))
            let uv = qv.cross(&local);
            let uuv = qv.cross(&uv);
            let global = local + (uv * qw + uuv) * 2.0 + trans_v;

            let inv = 1.0 / resolution as f64;
            let gx = global.x.floor() as i32;
            let gy = global.y.floor() as i32;
            let gz = global.z.floor() as i32;
            // Same grid hash as CUDA
            let key = ((gx.wrapping_mul(73856093i32) as u32)
                ^ (gy.wrapping_mul(471943i32) as u32)
                ^ (gz.wrapping_mul(83492791i32) as u32)) as i32;
            let _ = inv; // used for floor; compiler will optimize
            (key, i as u32)
        })
        .collect();

    keyed.sort_unstable_by_key(|(k, _)| *k);

    // ── Compute run starts/ends ───────────────────────────────────────────────
    let mut unique_keys: Vec<i32> = Vec::new();
    let mut run_starts: Vec<u32> = Vec::new();
    let mut run_ends: Vec<u32> = Vec::new();
    let mut sorted_indices: Vec<u32> = Vec::with_capacity(n_points);

    let mut i = 0usize;
    while i < keyed.len() {
        let key = keyed[i].0;
        let start = sorted_indices.len() as u32;
        while i < keyed.len() && keyed[i].0 == key {
            sorted_indices.push(keyed[i].1);
            i += 1;
        }
        unique_keys.push(key);
        run_starts.push(start);
        run_ends.push(sorted_indices.len() as u32);
    }

    let n_unique = unique_keys.len();
    if n_unique == 0 {
        return;
    }

    // ── Upload CPU arrays to shared GPU buffers ───────────────────────────────
    let make_buf = |data: &[u8]| -> metal::Buffer {
        let buf = stream.device.new_buffer(
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut u8, data.len());
        }
        buf
    };

    let sorted_idx_buf = make_buf(bytemuck::cast_slice::<u32, u8>(&sorted_indices));
    let run_starts_buf = make_buf(bytemuck::cast_slice::<u32, u8>(&run_starts));
    let run_ends_buf   = make_buf(bytemuck::cast_slice::<u32, u8>(&run_ends));
    let unique_keys_buf = make_buf(bytemuck::cast_slice::<i32, u8>(&unique_keys));

    // ── Build params ──────────────────────────────────────────────────────────
    let params = BuildVoxelsParams {
        pose_quat,
        pose_trans,
        resolution,
        scan_number,
        sensitivity: state.sensitivity,
        min_dist: state.min_dist,
        unique_key_count: n_unique as u32,
        table_cap,
        dist_cap,
        _pad: 0,
    };

    // ── Dispatch build_voxels ─────────────────────────────────────────────────
    {
        let pipes = get_pipelines(&stream.device);
        let cb = stream.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipes.build_voxels);

        enc.set_buffer(0, Some(cloud_buf.metal_buffer()), 0);
        enc.set_buffer(1, Some(&sorted_idx_buf), 0);
        enc.set_buffer(2, Some(&run_starts_buf), 0);
        enc.set_buffer(3, Some(&run_ends_buf), 0);
        enc.set_buffer(4, Some(&unique_keys_buf), 0);
        enc.set_buffer(5, Some(&state.keys_buf), 0);
        enc.set_buffer(6, Some(&state.values_buf), 0);
        enc.set_buffer(7, Some(&state.dist_buf), 0);
        enc.set_buffer(8, Some(&state.dist_keys_buf), 0);
        enc.set_buffer(9, Some(&state.add_cursor_buf), 0);
        enc.set_bytes(
            10,
            size_of::<BuildVoxelsParams>() as u64,
            &params as *const BuildVoxelsParams as *const _,
        );

        enc.dispatch_threads(grid1d(n_unique), group1d());
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // ── Rebuild hashmap (clear + re-insert from dist buffer) ──────────────────
    let add_cursor = unsafe {
        *(state.add_cursor_buf.contents() as *const u32)
    };
    let valid_size = (add_cursor as usize).min(dist_cap as usize);

    {
        let pipes = get_pipelines(&stream.device);
        let cb = stream.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();

        // First: clear the hash table keys back to EMPTY_KEY
        enc.set_compute_pipeline_state(&pipes.hashmap_init);
        enc.set_buffer(0, Some(&state.keys_buf), 0);
        enc.set_bytes(1, size_of::<u32>() as u64, &table_cap as *const u32 as *const _);
        enc.dispatch_threads(grid1d(table_cap as usize), group1d());

        // NOTE: Metal does not allow pipeline state changes mid-encoder for blit.
        // End this encoder and start a new one for rebuild.
        enc.end_encoding();

        let enc2 = cb.new_compute_command_encoder();
        enc2.set_compute_pipeline_state(&pipes.rebuild_hashmap_keyed);
        enc2.set_buffer(0, Some(&state.keys_buf), 0);
        enc2.set_buffer(1, Some(&state.values_buf), 0);
        enc2.set_buffer(2, Some(&state.dist_keys_buf), 0);
        enc2.set_bytes(3, size_of::<u32>() as u64, &(valid_size as u32) as *const u32 as *const _);
        enc2.set_bytes(4, size_of::<u32>() as u64, &table_cap as *const u32 as *const _);
        enc2.dispatch_threads(grid1d(valid_size.max(1)), group1d());
        enc2.end_encoding();

        cb.commit();
        cb.wait_until_completed();
    }
}

// ── Get map points ────────────────────────────────────────────────────────────

pub fn get_map_impl() -> Vec<nalgebra::Point3<f32>> {
    let guard = get_state();
    let state = match guard.as_ref() {
        Some(s) => s,
        None => return vec![],
    };

    let add_cursor = unsafe {
        *(state.add_cursor_buf.contents() as *const u32)
    };
    let valid = (add_cursor as usize).min(state.dist_cap as usize);

    let dists: &[GpuVoxelDist] = unsafe {
        std::slice::from_raw_parts(
            state.dist_buf.contents() as *const GpuVoxelDist,
            valid,
        )
    };

    dists
        .iter()
        .filter(|d| d.point_count > 0)
        .map(|d| nalgebra::Point3::new(d.mean[0], d.mean[1], d.mean[2]))
        .collect()
}

pub fn get_map_size_impl() -> u32 {
    let guard = get_state();
    match guard.as_ref() {
        Some(s) => {
            let cursor = unsafe { *(s.add_cursor_buf.contents() as *const u32) };
            cursor.min(s.dist_cap)
        }
        None => 0,
    }
}

pub fn cleanup_map_impl() {
    *get_state() = None;
}

// ── Planes computation ────────────────────────────────────────────────────────

pub fn planes_metal_impl(
    stream: &Stream,
    cloud: &crate::buffer::MetalBuffer,
    ekf_pose_t: &crate::types::Vector3,
    ekf_pose_r: &crate::types::Matrix3,
    ekf_imu_offset_t: &crate::types::Vector3,
    ekf_imu_offset_r: &crate::types::Matrix3,
    out_buff: &mut Vec<ros_pointcloud2::points::PointXYZINormal>,
    out_buff_gpu: &mut crate::buffer::MetalBuffer,
) {
    use nalgebra::UnitQuaternion;

    assert!(out_buff.len() >= cloud.n, "out_buff too small");

    let guard = get_state();
    let state = guard.as_ref().expect("Map not initialised");

    // Build quaternions
    let ext_rot: nalgebra::Matrix3<f64> = (*ekf_imu_offset_r).into();
    let ext_q = UnitQuaternion::from_matrix(&ext_rot);
    let pose_rot: nalgebra::Matrix3<f64> = (*ekf_pose_r).into();
    let pose_q = UnitQuaternion::from_matrix(&pose_rot);

    let lidar_to_imu_quat = [ext_q.i as f32, ext_q.j as f32, ext_q.k as f32, ext_q.w as f32];
    let lidar_to_imu_pos = [
        ekf_imu_offset_t.x as f32,
        ekf_imu_offset_t.y as f32,
        ekf_imu_offset_t.z as f32,
    ];
    let pose_quat = [pose_q.i as f32, pose_q.j as f32, pose_q.k as f32, pose_q.w as f32];
    let pose_pos = [ekf_pose_t.x as f32, ekf_pose_t.y as f32, ekf_pose_t.z as f32];

    // Zero the out_size counter
    unsafe {
        *(out_buff_gpu.buf.contents() as *mut u32) = 0;
    }

    // We use the first 4 bytes of out_buff_gpu as an atomic counter.
    // Construct a dedicated counter buffer.
    let counter_buf = stream.device.new_buffer(
        size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe { *(counter_buf.contents() as *mut u32) = 0; }

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
    };

    {
        let pipes = get_pipelines(&stream.device);
        let cb = stream.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipes.plane_krnl);
        enc.set_buffer(0, Some(cloud.metal_buffer()), 0);
        enc.set_buffer(1, Some(&state.keys_buf), 0);
        enc.set_buffer(2, Some(&state.values_buf), 0);
        enc.set_buffer(3, Some(&state.dist_buf), 0);
        enc.set_buffer(4, Some(&counter_buf), 0);
        enc.set_buffer(5, Some(out_buff_gpu.metal_buffer()), 0);
        enc.set_bytes(
            6,
            size_of::<PlanesParams>() as u64,
            &params as *const PlanesParams as *const _,
        );
        enc.dispatch_threads(grid1d(cloud.n), group1d());
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Read back results
    let out_size = unsafe { *(counter_buf.contents() as *const u32) } as usize;
    let gpu_normals: &[crate::types::PointNormal] = unsafe {
        std::slice::from_raw_parts(
            out_buff_gpu.buf.contents() as *const crate::types::PointNormal,
            out_size.min(out_buff.len()),
        )
    };

    unsafe { out_buff.set_len(out_size.min(out_buff.capacity())) };
    for (dst, src) in out_buff.iter_mut().zip(gpu_normals.iter()) {
        *dst = (*src).into();
    }
}

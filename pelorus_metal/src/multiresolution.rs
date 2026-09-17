//! CPU-orchestrated multiresolution distribution pyramid for the Metal backend.

use crate::buffer::MetalBuffer;
use crate::pipeline::{get_pipelines, grid1d, group1d};
use crate::stream::Stream;
use crate::types::{Point, PointNormal as GpuPointNormal, Quaternion, Vector3};
use bytemuck::{Pod, Zeroable};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use rayon::prelude::*;
use objc2_metal::{
    MTLBarrierScope, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder,
};
use ros_pointcloud2::points::PointXYZINormal;
use std::cmp::Ordering;
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Mutex, MutexGuard, OnceLock};

#[path = "multiresolution_transform.rs"]
mod transform;
use transform::transform_sample;

const LEVELS: usize = 4;

struct AddFence(Retained<ProtocolObject<dyn MTLCommandBuffer>>);
unsafe impl Send for AddFence {}
unsafe impl Sync for AddFence {}

// Must match `MRCell` in shaders/multiresolution.metal.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct Cell {
    key: [i32; 3],
    scan: u32,
    count: u64,
    mean: [f32; 3],
    cov: [f32; 9],
}
const _: () = assert!(size_of::<Cell>() == 72);

// Must match `MRFit` in shaders/multiresolution.metal.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Fit {
    status: u32,
    mean: [f32; 3],
    normal: [f32; 3],
    tangent: [f32; 6],
    extent: [f32; 2],
    quality: f32,
    d: f32,
}
const _: () = assert!(size_of::<Fit>() == 68);
impl Default for Fit {
    fn default() -> Self {
        Zeroable::zeroed()
    }
}

// Must match `MRQueryParams` in shaders/multiresolution.metal.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QueryParams {
    lidar_to_imu_quat: [f32; 4],
    lidar_to_imu_pos: [f32; 3],
    _pad0: f32,
    pose_quat: [f32; 4],
    pose_trans: [f32; 3],
    _pad1: f32,
    resolution: f32,
    table_cap: u32,
    n: u32,
    level_sizes: [u32; 4],
    fitcache_cap: u32,
    /// Monotonic per-query-dispatch counter.
    fitcache_gen: u32,
    /// First generation belonging to the current scan.
    fitcache_epoch: u32,
    _pad2: u32,
}
const _: () = assert!(size_of::<QueryParams>() == 108);

// Must match `MRCacheParams` in shaders/multiresolution.metal.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CacheParams {
    n: u32,
    resolution: f32,
    cap: u32,
    _pad: u32,
}
const _: () = assert!(size_of::<CacheParams>() == 16);

fn key_cmp(a: &[i32; 3], b: &[i32; 3]) -> Ordering {
    a[0].cmp(&b[0]).then(a[1].cmp(&b[1])).then(a[2].cmp(&b[2]))
}

fn coordinate(p: f32, r: f32) -> i32 {
    let v = (p / r).floor();
    if v >= 2147483648.0 {
        i32::MAX
    } else if v <= -2147483648.0 {
        i32::MIN
    } else {
        v as i32
    }
}

fn grid_key(p: [f32; 3], r: f32) -> [i32; 3] {
    [coordinate(p[0], r), coordinate(p[1], r), coordinate(p[2], r)]
}

fn parent_coord(x: i32) -> i32 {
    x / 2 - i32::from(x < 0 && x % 2 != 0)
}

fn parent_key(k: [i32; 3]) -> [i32; 3] {
    [parent_coord(k[0]), parent_coord(k[1]), parent_coord(k[2])]
}

#[cfg(test)]
fn neighbor_coord(x: i32, d: i32) -> i32 {
    if d > 0 && x == i32::MAX {
        x
    } else if d < 0 && x == i32::MIN {
        x
    } else {
        x + d
    }
}

fn pool(a: &mut Cell, b: &Cell) {
    if b.count == 0 {
        return;
    }
    if a.count == 0 {
        *a = *b;
        return;
    }
    let n1 = a.count as f32;
    let n2 = b.count as f32;
    let n = n1 + n2;
    let w = n1 * n2 / n;
    let delta: [f32; 3] = std::array::from_fn(|i| a.mean[i] - b.mean[i]);
    for i in 0..3 {
        a.mean[i] = (n1 * a.mean[i] + n2 * b.mean[i]) / n;
    }
    for r in 0..3 {
        for c in 0..3 {
            a.cov[3 * r + c] =
                (n1 * a.cov[3 * r + c] + n2 * b.cov[3 * r + c] + delta[r] * delta[c] * w) / n;
        }
    }
    a.count = n as u64;
}

fn temporal_update(a: &mut Cell, b: &Cell, sensitivity: f32) {
    a.scan = b.scan;
    let delta: [f32; 3] = std::array::from_fn(|i| a.mean[i] - b.mean[i]);
    let l00 = (a.cov[0] + 1e-6).sqrt();
    let l10 = a.cov[3] / l00;
    let l20 = a.cov[6] / l00;
    let l11 = (a.cov[4] + 1e-6 - l10 * l10).sqrt();
    let l21 = (a.cov[7] - l20 * l10) / l11;
    let l22 = (a.cov[8] + 1e-6 - (l20 * l20 + l21 * l21)).sqrt();
    let mut m = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
    if l00.is_finite() && l11.is_finite() && l22.is_finite() && l00 > 0.0 && l11 > 0.0 && l22 > 0.0
    {
        let y0 = delta[0] / l00;
        let y1 = (delta[1] - l10 * y0) / l11;
        let y2 = (delta[2] - (l20 * y0 + l21 * y1)) / l22;
        let x2 = y2 / l22;
        let x1 = (y1 - l21 * x2) / l11;
        let x0 = (y0 - (l10 * x1 + l20 * x2)) / l00;
        m = delta[0] * x0 + delta[1] * x1 + delta[2] * x2;
    }
    let alpha = (-m / (2.0 * (sensitivity / (a.count as f32).ln().max(1.0)))).exp();
    if alpha < 1e-5 {
        return;
    }
    let oma = 1.0 - alpha;
    for i in 0..3 {
        a.mean[i] = oma * a.mean[i] + alpha * b.mean[i];
    }
    for r in 0..3 {
        for c in 0..3 {
            a.cov[3 * r + c] = oma * a.cov[3 * r + c]
                + alpha * b.cov[3 * r + c]
                + alpha * oma * (delta[r] * delta[c]);
        }
    }
    a.count += b.count;
}

#[derive(Clone, Copy)]
enum Mode {
    Raw,
    Pool,
}

fn combine_runs(mut cells: Vec<Cell>, mode: Mode) -> Vec<Cell> {
    cells.par_sort_by(|a, b| key_cmp(&a.key, &b.key));
    let mut out = Vec::with_capacity(cells.len());
    let mut i = 0;
    while i < cells.len() {
        let mut end = i + 1;
        while end < cells.len() && cells[end].key == cells[i].key {
            end += 1;
        }
        let run = &cells[i..end];
        let c = match mode {
            Mode::Raw => {
                let n = run.len() as f32;
                let mut mean = [0f32; 3];
                for cell in run {
                    for k in 0..3 {
                        mean[k] += cell.mean[k];
                    }
                }
                for m in &mut mean {
                    *m /= n;
                }
                let mut cov = [0f32; 9];
                for cell in run {
                    let delta: [f32; 3] = std::array::from_fn(|k| cell.mean[k] - mean[k]);
                    for r in 0..3 {
                        for k in 0..3 {
                            cov[3 * r + k] += delta[r] * delta[k];
                        }
                    }
                }
                for v in &mut cov {
                    *v /= n;
                }
                Cell {
                    key: run[0].key,
                    scan: run[0].scan,
                    count: run.len() as u64,
                    mean,
                    cov,
                }
            }
            Mode::Pool => {
                let mut c = run[0];
                for cell in &run[1..] {
                    pool(&mut c, cell);
                }
                c
            }
        };
        out.push(c);
        i = end;
    }
    out
}

fn evict_cmp(a: &Cell, b: &Cell, scan: u32) -> Ordering {
    let age_a = scan.saturating_sub(a.scan) as u128;
    let age_b = scan.saturating_sub(b.scan) as u128;
    let ha = age_a * b.count as u128;
    let hb = age_b * a.count as u128;
    hb.cmp(&ha).then_with(|| key_cmp(&a.key, &b.key))
}

#[cfg(test)]
fn find_cell(cells: &[Cell], key: [i32; 3]) -> Option<usize> {
    cells.binary_search_by(|c| key_cmp(&c.key, &key)).ok()
}

#[cfg(test)]
fn fit(cell: &Cell, resolution: f32) -> Fit {
    let mut f = Fit::default();
    if cell.count < 6 {
        return f;
    }
    f.status = 1;
    if cell.cov.iter().any(|v| !v.is_finite()) {
        return f;
    }
    let cov = nalgebra::Matrix3::new(
        cell.cov[0],
        cell.cov[1],
        cell.cov[2],
        cell.cov[3],
        cell.cov[4],
        cell.cov[5],
        cell.cov[6],
        cell.cov[7],
        cell.cov[8],
    );
    let eigen = cov.symmetric_eigen();
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| {
        eigen.eigenvalues[i]
            .partial_cmp(&eigen.eigenvalues[j])
            .unwrap_or(Ordering::Equal)
    });
    let small = eigen.eigenvalues[idx[0]].max(0.0);
    let middle = eigen.eigenvalues[idx[1]];
    let large = eigen.eigenvalues[idx[2]];
    if !middle.is_finite() || !large.is_finite() {
        return f;
    }
    if middle < 1e-4 {
        f.status = 0;
        return f;
    }
    let thickness = resolution * 0.15;
    if small > thickness * thickness || small > 0.1 * middle {
        return f;
    }
    f.status = 2;
    for k in 0..3 {
        f.mean[k] = cell.mean[k];
        f.normal[k] = eigen.eigenvectors[(k, idx[0])];
        f.tangent[k] = eigen.eigenvectors[(k, idx[1])];
        f.tangent[3 + k] = eigen.eigenvectors[(k, idx[2])];
    }
    f.extent[0] = 3.0 * middle.max(0.0).sqrt() + resolution * 0.5;
    f.extent[1] = 3.0 * large.max(0.0).sqrt() + resolution * 0.5;
    let noise = resolution * 0.04;
    f.quality =
        (small + noise * noise) * (1.0 / (middle - small).max(1e-6) + 1.0 / (large - small).max(1e-6));
    f.d = -(f.normal[0] * f.mean[0] + f.normal[1] * f.mean[1] + f.normal[2] * f.mean[2]);
    f
}

#[cfg(test)]
fn neighborhood_fit(cells: &[Cell], key: [i32; 3], resolution: f32) -> Fit {
    let mut pooled = Cell::default();
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                let k = [
                    neighbor_coord(key[0], dx),
                    neighbor_coord(key[1], dy),
                    neighbor_coord(key[2], dz),
                ];
                if let Some(i) = find_cell(cells, k) {
                    pool(&mut pooled, &cells[i]);
                }
            }
        }
    }
    fit(&pooled, resolution)
}

struct Level {
    /// Number of live cells, held in `gpu_cells[level][cur]`.
    len: usize,
    /// Which of the two cell buffers is current.
    cur: usize,
    budget: usize,
}

pub(crate) struct MultiResolutionMap {
    levels: [Level; LEVELS],
    /// Cells live here, not in a `Vec`: the buffers are `StorageModeShared`,
    /// so the CPU reads and writes them directly with no copy. Two per level,
    /// used ping-pong, because folding a scan in produces a new ordered array
    /// and doing that straight into device memory removes both a 5 MB
    /// allocation and a 5 MB upload per level per scan.
    gpu_cells: [[MetalBuffer; 2]; LEVELS],
    gpu_fits: [MetalBuffer; LEVELS],
    /// Per-level open-addressing index: slot -> cell index, `u32::MAX` empty.
    /// Sized once from the budget so it never reallocates mid-run.
    gpu_tables: [MetalBuffer; LEVELS],
    table_cap: usize,
    /// Per-scan cache of live neighborhood fits, shared by the ICP iterations.
    fitcache_flags: MetalBuffer,
    fitcache_keys: MetalBuffer,
    fitcache_fits: MetalBuffer,
    fitcache_cap: usize,
    fitcache_gen: u32,
    fitcache_epoch: u32,
    /// Reused across scans: eviction would otherwise allocate and initialise
    /// two ~70k-element vectors per level per scan.
    evict_idx: Vec<u32>,
    evict_doomed: Vec<bool>,
    /// The command buffer from the previous `add`.
    last_add_cb: Option<AddFence>,
    base: f32,
    sensitivity: f32,
}

impl MultiResolutionMap {
    fn new(stream: &Stream, capacity: usize, base: f32, sensitivity: f32) -> Self {
        assert!(capacity >= LEVELS && base.is_finite() && base > 0.0);
        let levels = std::array::from_fn(|i| {
            let budget = capacity / LEVELS + usize::from(i < capacity % LEVELS);
            Level { len: 0, cur: 0, budget }
        });
        // +1 so a scan can always be folded in before eviction trims back.
        let gpu_cells: [[MetalBuffer; 2]; LEVELS] = std::array::from_fn(|i: usize| {
            std::array::from_fn(|_| {
                MetalBuffer::new(stream, size_of::<Cell>(), levels[i].budget.max(1))
            })
        });
        let gpu_fits =
            std::array::from_fn(|i: usize| MetalBuffer::new(stream, size_of::<Fit>(), levels[i].budget.max(1)));
        let table_cap = (levels.iter().map(|l| l.budget).max().unwrap_or(1).max(1) * 2)
            .next_power_of_two();
        let gpu_tables = std::array::from_fn(|_| {
            let mut table = MetalBuffer::new(stream, size_of::<u32>(), table_cap);
            table.as_slice_mut::<u32>().fill(u32::MAX);
            table
        });
        let fitcache_cap = 131072usize;
        let mut fitcache_flags = MetalBuffer::new(stream, size_of::<u32>(), fitcache_cap);
        fitcache_flags.as_slice_mut::<u32>().fill(0);
        Self {
            levels,
            gpu_cells,
            gpu_fits,
            gpu_tables,
            table_cap,
            fitcache_flags,
            fitcache_keys: MetalBuffer::new(stream, size_of::<i32>() * 4, fitcache_cap),
            fitcache_fits: MetalBuffer::new(stream, size_of::<Fit>(), fitcache_cap),
            fitcache_cap,
            fitcache_gen: 0,
            fitcache_epoch: 0,
            evict_idx: Vec::new(),
            evict_doomed: Vec::new(),
            last_add_cb: None,
            base,
            sensitivity,
        }
    }

    fn add(&mut self, stream: &Stream, cloud: &MetalBuffer, t: &Vector3, q: &Quaternion, scan: usize) {
        if let Some(fence) = self.last_add_cb.take() {
            fence.0.waitUntilCompleted();
        }
        let scan = scan as u32;
        let points = cloud.as_slice::<Point>();
        let quat: nalgebra::UnitQuaternion<f64> = (*q).into();
        let rot = [quat.i, quat.j, quat.k, quat.w];
        let trans = [t.x, t.y, t.z];

        let mut batch: Vec<Cell> = points
            .iter()
            .filter_map(|p| {
                let local = [p.x, p.y, p.z];
                let world = transform_sample(rot, trans, local);
                if world.iter().any(|v| !v.is_finite()) {
                    return None;
                }
                Some(Cell {
                    key: grid_key(world, self.base * 0.5),
                    scan,
                    count: 1,
                    mean: world,
                    cov: [0.0; 9],
                })
            })
            .collect();
        if batch.is_empty() {
            return;
        }
        let t_raw = std::time::Instant::now();
        batch = combine_runs(batch, Mode::Raw);
        let d_raw = t_raw.elapsed();
        if batch.is_empty() {
            return;
        }

        let mut d_pool = std::time::Duration::ZERO;
        let mut d_update = std::time::Duration::ZERO;
        let mut d_evict = std::time::Duration::ZERO;
        let mut n_fits = 0usize;
        for level in 0..LEVELS {
            if level > 0 {
                let t = std::time::Instant::now();
                for c in &mut batch {
                    c.key = parent_key(c.key);
                }
                batch = combine_runs(batch, Mode::Pool);
                d_pool += t.elapsed();
            }
            if batch.is_empty() {
                continue;
            }
            let t = std::time::Instant::now();
            let (len, cur, budget) = {
                let lvl = &self.levels[level];
                (lvl.len, lvl.cur, lvl.budget)
            };
            // Grow both buffers together if a scan cannot fit before eviction.
            let want = len + batch.len();
            if self.gpu_cells[level][0].capacity() < want {
                self.gpu_cells[level] = std::array::from_fn(|k| {
                    let mut fresh = MetalBuffer::new(stream, size_of::<Cell>(), want);
                    if k == cur {
                        fresh.as_slice_mut::<Cell>()[..len]
                            .copy_from_slice(&self.gpu_cells[level][cur].as_slice::<Cell>()[..len]);
                    }
                    fresh
                });
            }
            // Disjoint &mut into the two halves of the pair, so the merge can
            // read one buffer and write the other.
            let (src, dst) = {
                let [a, b] = &mut self.gpu_cells[level];
                if cur == 0 { (&*a, b) } else { (&*b, a) }
            };
            dst.n = dst.capacity();
            let cells = &src.as_slice::<Cell>()[..len];
            let out = dst.as_slice_mut::<Cell>();
            let mut w = 0usize;
            let (mut i, mut j) = (0usize, 0usize);
            while i < len && j < batch.len() {
                match key_cmp(&cells[i].key, &batch[j].key) {
                    Ordering::Less => {
                        out[w] = cells[i];
                        i += 1;
                    }
                    Ordering::Greater => {
                        out[w] = batch[j];
                        j += 1;
                    }
                    Ordering::Equal => {
                        // The same fold the old `Mode::Update` performed: the
                        // existing map cell is the accumulator, the new
                        // observation is blended into it.
                        let mut cell = cells[i];
                        temporal_update(&mut cell, &batch[j], self.sensitivity);
                        out[w] = cell;
                        i += 1;
                        j += 1;
                    }
                }
                w += 1;
            }
            out[w..w + (len - i)].copy_from_slice(&cells[i..len]);
            w += len - i;
            out[w..w + (batch.len() - j)].copy_from_slice(&batch[j..]);
            w += batch.len() - j;
            d_update += t.elapsed();

            let t = std::time::Instant::now();
            if w > budget {
                let excess = w - budget;
                let live = dst.as_slice_mut::<Cell>();
                let idx = &mut self.evict_idx;
                idx.clear();
                idx.extend(0..w as u32);
                idx.select_nth_unstable_by(excess, |&a, &b| {
                    evict_cmp(&live[a as usize], &live[b as usize], scan)
                });
                let doomed = &mut self.evict_doomed;
                doomed.clear();
                doomed.resize(w, false);
                for &k in &idx[..excess] {
                    doomed[k as usize] = true;
                }
                let mut keep = 0usize;
                for k in 0..w {
                    if !doomed[k] {
                        live[keep] = live[k];
                        keep += 1;
                    }
                }
                w = keep;
            }
            d_evict += t.elapsed();
            n_fits += w;
            let lvl = &mut self.levels[level];
            lvl.len = w;
            lvl.cur = 1 - cur;
        }

        for level in 0..LEVELS {
            let n = self.levels[level].len;
            if self.gpu_fits[level].capacity() < n.max(1) {
                self.gpu_fits[level] = MetalBuffer::new(stream, size_of::<Fit>(), n.max(1));
            }
            self.gpu_fits[level].n = n;
            self.gpu_cells[level][self.levels[level].cur].n = n;
        }
        let t_gpu = std::time::Instant::now();
        {
            let pipes = get_pipelines(&stream.device);
            let cb = stream.queue.commandBuffer().expect("cb");
            let enc = cb.computeCommandEncoder().expect("enc");

            let params: [CacheParams; LEVELS] = std::array::from_fn(|level| CacheParams {
                n: self.levels[level].len as u32,
                resolution: self.base,
                cap: self.table_cap as u32,
                _pad: 0,
            });
            let bind = |enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
                            level: usize,
                            index: usize| unsafe {
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&params[level] as *const _ as *mut _).unwrap(),
                    size_of::<CacheParams>(),
                    index,
                );
            };

            enc.setComputePipelineState(&pipes.mr_table_init);
            for level in 0..LEVELS {
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(self.gpu_tables[level].metal_buffer()), 0, 0);
                }
                bind(&enc, level, 1);
                enc.dispatchThreads_threadsPerThreadgroup(grid1d(self.table_cap), group1d());
            }
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

            enc.setComputePipelineState(&pipes.mr_table_build);
            for level in 0..LEVELS {
                let n = self.levels[level].len;
                if n == 0 {
                    continue;
                }
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(self.gpu_tables[level].metal_buffer()), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(self.gpu_cells[level][self.levels[level].cur].metal_buffer()), 0, 1);
                }
                bind(&enc, level, 2);
                enc.dispatchThreads_threadsPerThreadgroup(grid1d(n), group1d());
            }
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

            enc.setComputePipelineState(&pipes.mr_cache_fits);
            for level in 0..LEVELS {
                let n = self.levels[level].len;
                if n == 0 {
                    continue;
                }
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(self.gpu_cells[level][self.levels[level].cur].metal_buffer()), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(self.gpu_fits[level].metal_buffer()), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(self.gpu_tables[level].metal_buffer()), 0, 2);
                }
                bind(&enc, level, 3);
                enc.dispatchThreads_threadsPerThreadgroup(grid1d(n), group1d());
            }
            enc.endEncoding();
            cb.commit();
            self.last_add_cb = Some(AddFence(cb));
        }

        // The map just changed, so everything cached before now is stale.
        self.fitcache_epoch = self.fitcache_gen + 1;

        log::debug!(
            "[Metal-Multires] add: raw={:?}, pool={:?}, update={:?}, evict={:?}, gpu_enqueue={:?}, cells={}",
            d_raw,
            d_pool,
            d_update,
            d_evict,
            t_gpu.elapsed(),
            n_fits
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn planes(
        &mut self,
        stream: &Stream,
        cloud: &MetalBuffer,
        t: &Vector3,
        q: &Quaternion,
        ext_t: &Vector3,
        ext_q: &Quaternion,
        out: &mut Vec<PointXYZINormal>,
        out_gpu: &mut MetalBuffer,
    ) -> f64 {
        use std::time::Instant;

        let n = cloud.n;
        out.clear();
        if n == 0 {
            return 0.0;
        }
        if out_gpu.capacity() < n {
            *out_gpu = MetalBuffer::new(stream, size_of::<GpuPointNormal>(), n);
        }

        // One generation per query dispatch; entries published by this one
        // become readable to the next.
        self.fitcache_gen += 1;

        let q: nalgebra::UnitQuaternion<f64> = (*q).into();
        let ext_q: nalgebra::UnitQuaternion<f64> = (*ext_q).into();
        let params = QueryParams {
            lidar_to_imu_quat: [ext_q.i as f32, ext_q.j as f32, ext_q.k as f32, ext_q.w as f32],
            lidar_to_imu_pos: [ext_t.x as f32, ext_t.y as f32, ext_t.z as f32],
            _pad0: 0.0,
            pose_quat: [q.i as f32, q.j as f32, q.k as f32, q.w as f32],
            pose_trans: [t.x as f32, t.y as f32, t.z as f32],
            _pad1: 0.0,
            resolution: self.base,
            table_cap: self.table_cap as u32,
            n: n as u32,
            level_sizes: std::array::from_fn(|i| self.levels[i].len as u32),
            fitcache_cap: self.fitcache_cap as u32,
            fitcache_gen: self.fitcache_gen,
            fitcache_epoch: self.fitcache_epoch,
            _pad2: 0,
        };

        let t_gpu = Instant::now();
        let gpu_time_ms = {
            let pipes = get_pipelines(&stream.device);
            let cb = stream.queue.commandBuffer().expect("cb");
            let enc = cb.computeCommandEncoder().expect("enc");
            enc.setComputePipelineState(&pipes.mr_query);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(cloud.metal_buffer()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(self.gpu_cells[0][self.levels[0].cur].metal_buffer()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(self.gpu_cells[1][self.levels[1].cur].metal_buffer()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(self.gpu_cells[2][self.levels[2].cur].metal_buffer()), 0, 3);
                enc.setBuffer_offset_atIndex(Some(self.gpu_cells[3][self.levels[3].cur].metal_buffer()), 0, 4);
                enc.setBuffer_offset_atIndex(Some(self.gpu_fits[0].metal_buffer()), 0, 5);
                enc.setBuffer_offset_atIndex(Some(self.gpu_fits[1].metal_buffer()), 0, 6);
                enc.setBuffer_offset_atIndex(Some(self.gpu_fits[2].metal_buffer()), 0, 7);
                enc.setBuffer_offset_atIndex(Some(self.gpu_fits[3].metal_buffer()), 0, 8);
                enc.setBuffer_offset_atIndex(Some(self.gpu_tables[0].metal_buffer()), 0, 9);
                enc.setBuffer_offset_atIndex(Some(self.gpu_tables[1].metal_buffer()), 0, 10);
                enc.setBuffer_offset_atIndex(Some(self.gpu_tables[2].metal_buffer()), 0, 11);
                enc.setBuffer_offset_atIndex(Some(self.gpu_tables[3].metal_buffer()), 0, 12);
                enc.setBuffer_offset_atIndex(Some(self.fitcache_flags.metal_buffer()), 0, 13);
                enc.setBuffer_offset_atIndex(Some(self.fitcache_keys.metal_buffer()), 0, 14);
                enc.setBuffer_offset_atIndex(Some(self.fitcache_fits.metal_buffer()), 0, 15);
                enc.setBuffer_offset_atIndex(Some(out_gpu.metal_buffer()), 0, 16);
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                    size_of::<QueryParams>(),
                    17,
                );
                enc.dispatchThreads_threadsPerThreadgroup(grid1d(n), group1d());
            }
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            (cb.GPUEndTime() - cb.GPUStartTime()) * 1000.0
        };
        let dur_gpu = t_gpu.elapsed();

        let gpu_normals = out_gpu.as_slice::<GpuPointNormal>();
        for src in gpu_normals.iter().take(n) {
            if src.nx != 0.0 || src.ny != 0.0 || src.nz != 0.0 {
                out.push((*src).into());
            }
        }
        log::debug!(
            "[Metal-Multires] planes: gpu={:?}, gpu_time={:.2}ms, valid={}",
            dur_gpu,
            gpu_time_ms,
            out.len()
        );
        gpu_time_ms
    }

    fn capacity(&self) -> u32 {
        self.levels.iter().map(|l| l.budget as u32).sum()
    }

    fn size(&self) -> u32 {
        self.levels.iter().map(|l| l.len as u32).sum()
    }

    fn get_map_with_recency(&self) -> Vec<(nalgebra::Point3<f32>, u8, u32)> {
        let mut out = Vec::with_capacity(self.levels.iter().map(|l| l.len).sum());
        for (level, l) in self.levels.iter().enumerate() {
            let cells = &self.gpu_cells[level][l.cur].as_slice::<Cell>()[..l.len];
            out.extend(cells.iter().map(|c| {
                (
                    nalgebra::Point3::new(c.mean[0], c.mean[1], c.mean[2]),
                    level as u8,
                    c.scan,
                )
            }));
        }
        out
    }
}

static MULTIRES_STATE: OnceLock<Mutex<Option<MultiResolutionMap>>> = OnceLock::new();
static USING_MULTIRES: AtomicBool = AtomicBool::new(false);

fn state() -> MutexGuard<'static, Option<MultiResolutionMap>> {
    MULTIRES_STATE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap()
}

pub(crate) fn is_active() -> bool {
    USING_MULTIRES.load(AtomicOrdering::Acquire)
}

pub(crate) fn init(stream: &Stream, capacity: usize, resolution: f32, sensitivity: f32) {
    *state() = Some(MultiResolutionMap::new(stream, capacity, resolution, sensitivity));
    USING_MULTIRES.store(true, AtomicOrdering::Release);
}

pub(crate) fn add_to_map(
    stream: &Stream,
    cloud: &MetalBuffer,
    t: &Vector3,
    q: &Quaternion,
    scan_id: usize,
) -> f64 {
    let mut guard = state();
    let m = guard.as_mut().expect("multiresolution map not initialised");
    m.add(stream, cloud, t, q, scan_id);
    // Pyramid maintenance is CPU-side. No GPU time to report here.
    0.0
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn planes(
    stream: &Stream,
    cloud: &MetalBuffer,
    t: &Vector3,
    q: &Quaternion,
    ext_t: &Vector3,
    ext_q: &Quaternion,
    out: &mut Vec<PointXYZINormal>,
    out_gpu: &mut MetalBuffer,
) -> f64 {
    let mut guard = state();
    let m = guard.as_mut().expect("multiresolution map not initialised");
    m.planes(stream, cloud, t, q, ext_t, ext_q, out, out_gpu)
}

pub(crate) fn get_map() -> Vec<nalgebra::Point3<f32>> {
    get_map_with_recency()
        .into_iter()
        .map(|(p, _, _)| p)
        .collect()
}

pub(crate) fn get_map_with_recency() -> Vec<(nalgebra::Point3<f32>, u8, u32)> {
    match state().as_ref() {
        Some(m) => m.get_map_with_recency(),
        None => Vec::new(),
    }
}

pub(crate) fn get_map_size() -> u32 {
    state().as_ref().map(|m| m.size()).unwrap_or(0)
}

pub(crate) fn get_map_capacity() -> u32 {
    state().as_ref().map(|m| m.capacity()).unwrap_or(0)
}

pub(crate) fn cleanup() {
    *state() = None;
    USING_MULTIRES.store(false, AtomicOrdering::Release);
}

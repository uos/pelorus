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
    d: f32,
}
const _: () = assert!(size_of::<Fit>() == 64);
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
    /// Slots of the miss cache (power of two).
    miss_cap: u32,
    /// Map version the miss cache is valid for, advanced after every `add`.
    miss_version: u32,
    /// Capacity of the miss work list.
    work_cap: u32,
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
    /// Cells of the previous level version, only read by `mr_refresh_fits`.
    prev_n: u32,
}
const _: () = assert!(size_of::<CacheParams>() == 16);

// Must match `MRMissWork` in shaders/multiresolution.metal. Only written by the GPU.
#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MissWork {
    level: u32,
    slot: u32,
    key: [i32; 3],
}
const _: () = assert!(size_of::<MissWork>() == 20);

fn set_params(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, params: &CacheParams, index: usize) {
    unsafe {
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(params as *const _ as *mut _).unwrap(),
            size_of::<CacheParams>(),
            index,
        );
    }
}

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
    /// Which of the two cell/fit/table buffers is current.
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
    /// Fits and index are ping-pong like the cells (same `cur`), so the fits
    /// of the previous level version can be carried over to the next one.
    gpu_fits: [[MetalBuffer; 2]; LEVELS],
    /// Per-level open-addressing index: slot -> cell index, `u32::MAX` empty.
    /// Sized once from the budget so it never reallocates mid-run.
    gpu_tables: [[MetalBuffer; 2]; LEVELS],
    table_cap: usize,
    /// One u32 per cell of the level version being built: set if one of its
    /// 27 neighbor cells was updated, inserted or evicted in this scan.
    gpu_dirty: [MetalBuffer; LEVELS],
    /// Keys (x, y, z, unused) updated, inserted or evicted in this scan.
    gpu_changed: [MetalBuffer; LEVELS],
    /// Miss cache: fits of cells absent from a level, valid for one map
    /// version (`miss_version`). State per slot: version << 1 | ready.
    miss_state: MetalBuffer,
    miss_keys: MetalBuffer,
    miss_fits: MetalBuffer,
    miss_cap: usize,
    miss_version: u32,
    /// Absent cells listed by `mr_collect_misses` for `mr_compute_misses`.
    miss_work: MetalBuffer,
    /// Number of absent cells `mr_collect_misses` listed.
    work_count: MetalBuffer,
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
        let gpu_fits = std::array::from_fn(|i: usize| {
            std::array::from_fn(|_| MetalBuffer::new(stream, size_of::<Fit>(), levels[i].budget.max(1)))
        });
        let table_cap = (levels.iter().map(|l| l.budget).max().unwrap_or(1).max(1) * 2)
            .next_power_of_two();
        let gpu_tables = std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                let mut table = MetalBuffer::new(stream, size_of::<u32>(), table_cap);
                table.as_slice_mut::<u32>().fill(u32::MAX);
                table
            })
        });
        let gpu_dirty =
            std::array::from_fn(|i: usize| MetalBuffer::new(stream, size_of::<u32>(), levels[i].budget.max(1)));
        let gpu_changed = std::array::from_fn(|_| MetalBuffer::new(stream, size_of::<[i32; 4]>(), 1));
        let miss_cap = 131072usize;
        // State 0 belongs to version 0, which is never current: all slots empty.
        let mut miss_state = MetalBuffer::new(stream, size_of::<u32>(), miss_cap);
        miss_state.as_slice_mut::<u32>().fill(0);
        Self {
            levels,
            gpu_cells,
            gpu_fits,
            gpu_tables,
            table_cap,
            gpu_dirty,
            gpu_changed,
            miss_state,
            miss_keys: MetalBuffer::new(stream, size_of::<i32>() * 4, miss_cap),
            miss_fits: MetalBuffer::new(stream, size_of::<Fit>(), miss_cap),
            miss_cap,
            miss_version: 1,
            miss_work: MetalBuffer::new(stream, size_of::<MissWork>(), 1),
            work_count: MetalBuffer::new(stream, size_of::<u32>(), 1),
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
        // Per level: cells of the previous version, whether this scan built a
        // new version, and how many keys it changed (see `gpu_changed`).
        let mut prev_len = [0usize; LEVELS];
        let mut updated = [false; LEVELS];
        let mut n_changed = [0usize; LEVELS];
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

            // A fit only changes if one of its 27 neighbor cells was updated,
            // inserted or evicted. List those keys for `mr_mark_dirty`: this
            // scan's keys at this level, then the evicted cells below. The
            // previous `add`'s command buffer has completed (waited for at the
            // top), so the buffer is free to overwrite.
            let excess = w.saturating_sub(budget);
            let changed_n = batch.len() + excess;
            if self.gpu_changed[level].capacity() < changed_n {
                self.gpu_changed[level] = MetalBuffer::new(stream, size_of::<[i32; 4]>(), changed_n);
            }
            self.gpu_changed[level].n = changed_n;
            let changed = self.gpu_changed[level].as_slice_mut::<[i32; 4]>();
            for (c, b) in changed.iter_mut().zip(&batch) {
                *c = [b.key[0], b.key[1], b.key[2], 0];
            }

            let t = std::time::Instant::now();
            if w > budget {
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
                let mut evicted = batch.len();
                for k in 0..w {
                    if !doomed[k] {
                        live[keep] = live[k];
                        keep += 1;
                    } else {
                        let key = live[k].key;
                        changed[evicted] = [key[0], key[1], key[2], 0];
                        evicted += 1;
                    }
                }
                debug_assert_eq!(evicted, changed_n);
                w = keep;
            }
            d_evict += t.elapsed();
            n_fits += w;
            n_changed[level] = changed_n;
            prev_len[level] = len;
            updated[level] = true;
            let lvl = &mut self.levels[level];
            lvl.len = w;
            lvl.cur = 1 - cur;
        }

        // Levels this scan did not touch keep their cells, fits and index.
        for level in (0..LEVELS).filter(|&level| updated[level]) {
            let n = self.levels[level].len;
            let next = self.levels[level].cur;
            // Only the new version may be reallocated: the fits of the previous
            // one (`1 - next`) are carried over.
            if self.gpu_fits[level][next].capacity() < n.max(1) {
                self.gpu_fits[level][next] = MetalBuffer::new(stream, size_of::<Fit>(), n.max(1));
            }
            self.gpu_fits[level][next].n = n;
            self.gpu_cells[level][next].n = n;
            if self.gpu_dirty[level].capacity() < n.max(1) {
                self.gpu_dirty[level] = MetalBuffer::new(stream, size_of::<u32>(), n.max(1));
            }
            self.gpu_dirty[level].n = n;
            self.gpu_dirty[level].as_slice_mut::<u32>().fill(0);
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
                prev_n: prev_len[level] as u32,
            });
            let next = |level: usize| self.levels[level].cur;
            let prev = |level: usize| 1 - self.levels[level].cur;

            enc.setComputePipelineState(&pipes.mr_table_init);
            for level in 0..LEVELS {
                if !updated[level] {
                    continue;
                }
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(self.gpu_tables[level][next(level)].metal_buffer()), 0, 0);
                }
                set_params(&enc, &params[level], 1);
                enc.dispatchThreads_threadsPerThreadgroup(grid1d(self.table_cap), group1d());
            }
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

            enc.setComputePipelineState(&pipes.mr_table_build);
            for level in 0..LEVELS {
                let n = self.levels[level].len;
                if !updated[level] || n == 0 {
                    continue;
                }
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(self.gpu_tables[level][next(level)].metal_buffer()), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(self.gpu_cells[level][next(level)].metal_buffer()), 0, 1);
                }
                set_params(&enc, &params[level], 2);
                enc.dispatchThreads_threadsPerThreadgroup(grid1d(n), group1d());
            }
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

            // Only fits next to this scan's updates, insertions and evictions change.
            enc.setComputePipelineState(&pipes.mr_mark_dirty);
            for level in 0..LEVELS {
                let n = self.levels[level].len;
                if !updated[level] || n == 0 || n_changed[level] == 0 {
                    continue;
                }
                let mark = CacheParams { n: n_changed[level] as u32, ..params[level] };
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(self.gpu_changed[level].metal_buffer()), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(self.gpu_cells[level][next(level)].metal_buffer()), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(self.gpu_tables[level][next(level)].metal_buffer()), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(self.gpu_dirty[level].metal_buffer()), 0, 3);
                }
                set_params(&enc, &mark, 4);
                enc.dispatchThreads_threadsPerThreadgroup(grid1d(n_changed[level]), group1d());
            }
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

            enc.setComputePipelineState(&pipes.mr_refresh_fits);
            for level in 0..LEVELS {
                let n = self.levels[level].len;
                if !updated[level] || n == 0 {
                    continue;
                }
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(self.gpu_cells[level][prev(level)].metal_buffer()), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(self.gpu_fits[level][prev(level)].metal_buffer()), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(self.gpu_tables[level][prev(level)].metal_buffer()), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(self.gpu_cells[level][next(level)].metal_buffer()), 0, 3);
                    enc.setBuffer_offset_atIndex(Some(self.gpu_fits[level][next(level)].metal_buffer()), 0, 4);
                    enc.setBuffer_offset_atIndex(Some(self.gpu_tables[level][next(level)].metal_buffer()), 0, 5);
                    enc.setBuffer_offset_atIndex(Some(self.gpu_dirty[level].metal_buffer()), 0, 6);
                }
                set_params(&enc, &params[level], 7);
                enc.dispatchThreads_threadsPerThreadgroup(grid1d(n), group1d());
            }
            enc.endEncoding();
            cb.commit();
            self.last_add_cb = Some(AddFence(cb));
        }

        // The map changed, so cached fits of absent cells are stale.
        self.miss_version += 1;

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

        // Every point lists at most one absent cell per level. The previous
        // `planes` has completed (it waits below) and the in-flight `add` does
        // not touch these buffers, so the CPU may write them here.
        let work_cap = LEVELS * n;
        if self.miss_work.capacity() < work_cap {
            self.miss_work = MetalBuffer::new(stream, size_of::<MissWork>(), work_cap);
        }
        self.work_count.as_slice_mut::<u32>()[0] = 0;

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
            miss_cap: self.miss_cap as u32,
            miss_version: self.miss_version,
            work_cap: work_cap as u32,
            _pad2: 0,
        };

        let t_gpu = Instant::now();
        let gpu_time_ms = {
            let pipes = get_pipelines(&stream.device);
            let cb = stream.queue.commandBuffer().expect("cb");
            let enc = cb.computeCommandEncoder().expect("enc");
            let set_query_params = |index: usize| unsafe {
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                    size_of::<QueryParams>(),
                    index,
                );
            };
            let cur = |level: usize| self.levels[level].cur;

            // Pass 1: claim a miss-cache slot for every absent cell and list it.
            enc.setComputePipelineState(&pipes.mr_collect_misses);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(cloud.metal_buffer()), 0, 0);
                for level in 0..LEVELS {
                    enc.setBuffer_offset_atIndex(Some(self.gpu_cells[level][cur(level)].metal_buffer()), 0, 1 + level);
                    enc.setBuffer_offset_atIndex(Some(self.gpu_tables[level][cur(level)].metal_buffer()), 0, 5 + level);
                }
                enc.setBuffer_offset_atIndex(Some(self.miss_state.metal_buffer()), 0, 9);
                enc.setBuffer_offset_atIndex(Some(self.miss_keys.metal_buffer()), 0, 10);
                enc.setBuffer_offset_atIndex(Some(self.miss_work.metal_buffer()), 0, 11);
                enc.setBuffer_offset_atIndex(Some(self.work_count.metal_buffer()), 0, 12);
            }
            set_query_params(13);
            enc.dispatchThreads_threadsPerThreadgroup(grid1d(n), group1d());
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

            // Pass 2: one fit per listed absent cell (threads past the listed
            // count return at once).
            enc.setComputePipelineState(&pipes.mr_compute_misses);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(self.miss_work.metal_buffer()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(self.work_count.metal_buffer()), 0, 1);
                for level in 0..LEVELS {
                    enc.setBuffer_offset_atIndex(Some(self.gpu_cells[level][cur(level)].metal_buffer()), 0, 2 + level);
                    enc.setBuffer_offset_atIndex(Some(self.gpu_tables[level][cur(level)].metal_buffer()), 0, 6 + level);
                }
                enc.setBuffer_offset_atIndex(Some(self.miss_state.metal_buffer()), 0, 10);
                enc.setBuffer_offset_atIndex(Some(self.miss_fits.metal_buffer()), 0, 11);
            }
            set_query_params(12);
            enc.dispatchThreads_threadsPerThreadgroup(grid1d(work_cap), group1d());
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);

            // Pass 3: plane selection, reading absent-cell fits from the cache.
            enc.setComputePipelineState(&pipes.mr_query);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(cloud.metal_buffer()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(self.gpu_cells[0][self.levels[0].cur].metal_buffer()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(self.gpu_cells[1][self.levels[1].cur].metal_buffer()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(self.gpu_cells[2][self.levels[2].cur].metal_buffer()), 0, 3);
                enc.setBuffer_offset_atIndex(Some(self.gpu_cells[3][self.levels[3].cur].metal_buffer()), 0, 4);
                enc.setBuffer_offset_atIndex(Some(self.gpu_fits[0][self.levels[0].cur].metal_buffer()), 0, 5);
                enc.setBuffer_offset_atIndex(Some(self.gpu_fits[1][self.levels[1].cur].metal_buffer()), 0, 6);
                enc.setBuffer_offset_atIndex(Some(self.gpu_fits[2][self.levels[2].cur].metal_buffer()), 0, 7);
                enc.setBuffer_offset_atIndex(Some(self.gpu_fits[3][self.levels[3].cur].metal_buffer()), 0, 8);
                enc.setBuffer_offset_atIndex(Some(self.gpu_tables[0][self.levels[0].cur].metal_buffer()), 0, 9);
                enc.setBuffer_offset_atIndex(Some(self.gpu_tables[1][self.levels[1].cur].metal_buffer()), 0, 10);
                enc.setBuffer_offset_atIndex(Some(self.gpu_tables[2][self.levels[2].cur].metal_buffer()), 0, 11);
                enc.setBuffer_offset_atIndex(Some(self.gpu_tables[3][self.levels[3].cur].metal_buffer()), 0, 12);
                enc.setBuffer_offset_atIndex(Some(self.miss_state.metal_buffer()), 0, 13);
                enc.setBuffer_offset_atIndex(Some(self.miss_keys.metal_buffer()), 0, 14);
                enc.setBuffer_offset_atIndex(Some(self.miss_fits.metal_buffer()), 0, 15);
                enc.setBuffer_offset_atIndex(Some(out_gpu.metal_buffer()), 0, 16);
            }
            set_query_params(17);
            enc.dispatchThreads_threadsPerThreadgroup(grid1d(n), group1d());
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

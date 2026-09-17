//! CPU-orchestrated multiresolution distribution pyramid for the Metal backend.
//!
//! Metal has no double-precision floating point, so pose composition and the
//! per-cell eigendecomposition run in float — the same precision tier the
//! existing single-resolution Metal backend already uses (see
//! `shaders/localmap.metal`). Building the pyramid (folding in new samples,
//! merging with existing cells, evicting the least-supported ones, and
//! caching neighborhood fits) touches at most `capacity / 4` cells per level
//! and runs once per scan, so it is done on the CPU in plain Rust — mirroring
//! `pelorus_vulkan::multiresolution`'s GLSL kernels function for function
//! (using nalgebra's `symmetric_eigen` in place of its hand-rolled Jacobi
//! sweep) — rather than as an unverifiable custom bitonic-sort/prefix-scan
//! Metal kernel network.
//!
//! The hot path — correspondence search for every point in a scan against
//! the pyramid, called every ICP iteration — runs as a real Metal compute
//! kernel (`mr_query`, see `shaders/multiresolution.metal`), mirroring the
//! existing single-resolution `plane_krnl`.

use crate::buffer::MetalBuffer;
use crate::pipeline::{get_pipelines, grid1d, group1d};
use crate::stream::Stream;
use crate::types::{Point, PointNormal as GpuPointNormal, Quaternion, Vector3};
use bytemuck::{Pod, Zeroable};
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};
use ros_pointcloud2::points::PointXYZINormal;
use std::cmp::Ordering;
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Mutex, MutexGuard, OnceLock};

const LEVELS: usize = 4;

// Must match `MRCell` in shaders/multiresolution.metal exactly (72 bytes,
// no padding).
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

// Must match `MRFit` in shaders/multiresolution.metal exactly (68 bytes).
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

// Must match `MRQueryParams` in shaders/multiresolution.metal exactly (100 bytes).
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
    level_sizes: [u32; 4],
    n: u32,
    _pad2: [u32; 3],
}
const _: () = assert!(size_of::<QueryParams>() == 100);

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

fn neighbor_coord(x: i32, d: i32) -> i32 {
    if d > 0 && x == i32::MAX {
        x
    } else if d < 0 && x == i32::MIN {
        x
    } else {
        x + d
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// Matches `quat_rotate`/`iso_transform` in shaders/common.metal (float,
// quaternion as (x, y, z, w)).
fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let u = [q[0], q[1], q[2]];
    let uv = cross(u, v);
    let t = [2.0 * uv[0], 2.0 * uv[1], 2.0 * uv[2]];
    let ut = cross(u, t);
    [
        v[0] + q[3] * t[0] + ut[0],
        v[1] + q[3] * t[1] + ut[1],
        v[2] + q[3] * t[2] + ut[2],
    ]
}

fn iso_transform(q: [f32; 4], t: [f32; 3], p: [f32; 3]) -> [f32; 3] {
    let r = quat_rotate(q, p);
    [r[0] + t[0], r[1] + t[1], r[2] + t[2]]
}

/// Incorporate `b`'s statistics into `a` via the parallel-mean/covariance
/// formula. Mirrors `mr_pool` in the Vulkan/CUDA kernels.
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

/// Blend a new observation `b` into the existing cell `a` by an amount
/// governed by their Mahalanobis distance and `sensitivity`, instead of a
/// plain running-average pool. Mirrors `mr_update` in the Vulkan/CUDA
/// kernels (Cholesky forward/back substitution, same formulas).
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
    /// Exact two-pass mean/covariance over the whole run (fresh raw samples
    /// at level 0, before any incremental pooling). Mirrors GLSL `nbtype==2`.
    Raw,
    /// Parent-level pooling of an already-deduped batch. Mirrors `mr_pool`.
    Pool,
    /// Fold a new observation into an existing map cell. Mirrors `mr_update`.
    Update,
}

/// Sorts `cells` by key and folds each run of equal-key cells into one,
/// using `mode`. Mirrors the Vulkan/CUDA `MR_COMBINE` kernel plus its
/// caller-side sort.
fn combine_runs(mut cells: Vec<Cell>, mode: Mode, sensitivity: f32) -> Vec<Cell> {
    cells.sort_by(|a, b| key_cmp(&a.key, &b.key));
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
            Mode::Update => {
                let mut c = run[0];
                for cell in &run[1..] {
                    temporal_update(&mut c, cell, sensitivity);
                }
                c
            }
        };
        out.push(c);
        i = end;
    }
    out
}

/// Pairwise eviction comparator: `Less` means `a` is less well-supported
/// relative to its age than `b`, and should be evicted first. Cross-
/// multiplies age×support (as `u128`, avoiding the 96-bit extended-multiply
/// trick the GLSL/CUDA kernels need without a wide integer type) instead of
/// dividing. Mirrors `evict_less` in the Vulkan/CUDA kernels.
fn evict_cmp(a: &Cell, b: &Cell, scan: u32) -> Ordering {
    let age_a = scan.saturating_sub(a.scan) as u128;
    let age_b = scan.saturating_sub(b.scan) as u128;
    let ha = age_a * b.count as u128;
    let hb = age_b * a.count as u128;
    hb.cmp(&ha).then_with(|| key_cmp(&a.key, &b.key))
}

fn find_cell(cells: &[Cell], key: [i32; 3]) -> Option<usize> {
    cells.binary_search_by(|c| key_cmp(&c.key, &key)).ok()
}

/// Fit a plane to a pooled cell. Mirrors `mr_fit` in the Vulkan/CUDA
/// kernels, using nalgebra's `symmetric_eigen` in place of the hand-rolled
/// Jacobi sweep. Both are just eigendecomposition of a symmetric 3x3. This
/// substitution changes only the implementation, not the algorithm, and
/// nalgebra's is safe, well-tested Rust already used elsewhere in this
/// workspace for the same operation.
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

/// 3x3x3 neighborhood pool + fit, used to precompute each cell's cached fit.
/// Mirrors `mr_neighborhood` (cache path) in the Vulkan/CUDA kernels.
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
    cells: Vec<Cell>,
    fits: Vec<Fit>,
    budget: usize,
}

pub(crate) struct MultiResolutionMap {
    levels: [Level; LEVELS],
    gpu_cells: [MetalBuffer; LEVELS],
    gpu_fits: [MetalBuffer; LEVELS],
    base: f32,
    sensitivity: f32,
}

impl MultiResolutionMap {
    fn new(stream: &Stream, capacity: usize, base: f32, sensitivity: f32) -> Self {
        assert!(capacity >= LEVELS && base.is_finite() && base > 0.0);
        let levels = std::array::from_fn(|i| {
            let budget = capacity / LEVELS + usize::from(i < capacity % LEVELS);
            Level {
                cells: Vec::new(),
                fits: Vec::new(),
                budget,
            }
        });
        let gpu_cells =
            std::array::from_fn(|i: usize| MetalBuffer::new(stream, size_of::<Cell>(), levels[i].budget.max(1)));
        let gpu_fits =
            std::array::from_fn(|i: usize| MetalBuffer::new(stream, size_of::<Fit>(), levels[i].budget.max(1)));
        Self {
            levels,
            gpu_cells,
            gpu_fits,
            base,
            sensitivity,
        }
    }

    fn add(&mut self, stream: &Stream, cloud: &MetalBuffer, t: &Vector3, q: &Quaternion, scan: usize) {
        let scan = scan as u32;
        let points = cloud.as_slice::<Point>();
        let quat: nalgebra::UnitQuaternion<f64> = (*q).into();
        let rot = [quat.i as f32, quat.j as f32, quat.k as f32, quat.w as f32];
        let trans = [t.x as f32, t.y as f32, t.z as f32];

        let mut batch: Vec<Cell> = points
            .iter()
            .filter_map(|p| {
                let local = [p.x, p.y, p.z];
                let world = iso_transform(rot, trans, local);
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
        batch = combine_runs(batch, Mode::Raw, self.sensitivity);
        if batch.is_empty() {
            return;
        }

        for level in 0..LEVELS {
            if level > 0 {
                for c in &mut batch {
                    c.key = parent_key(c.key);
                }
                batch = combine_runs(batch, Mode::Pool, self.sensitivity);
            }
            if batch.is_empty() {
                continue;
            }
            let lvl = &mut self.levels[level];
            let merged: Vec<Cell> = lvl.cells.iter().copied().chain(batch.iter().copied()).collect();
            let mut merged = combine_runs(merged, Mode::Update, self.sensitivity);
            if merged.len() > lvl.budget {
                merged.sort_by(|a, b| evict_cmp(a, b, scan));
                let excess = merged.len() - lvl.budget;
                merged.drain(0..excess);
                merged.sort_by(|a, b| key_cmp(&a.key, &b.key));
            }
            lvl.fits = merged
                .iter()
                .map(|c| neighborhood_fit(&merged, c.key, self.base))
                .collect();
            lvl.cells = merged;
        }

        for level in 0..LEVELS {
            let lvl = &self.levels[level];
            let want = lvl.cells.len().max(1);
            if self.gpu_cells[level].capacity() < want {
                self.gpu_cells[level] = MetalBuffer::new(stream, size_of::<Cell>(), want);
                self.gpu_fits[level] = MetalBuffer::new(stream, size_of::<Fit>(), want);
            }
            self.gpu_cells[level].upload(stream, &lvl.cells);
            self.gpu_fits[level].upload(stream, &lvl.fits);
        }
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
            level_sizes: std::array::from_fn(|i| self.levels[i].cells.len() as u32),
            n: n as u32,
            _pad2: [0; 3],
        };

        let t_gpu = Instant::now();
        let gpu_time_ms = {
            let pipes = get_pipelines(&stream.device);
            let cb = stream.queue.commandBuffer().expect("cb");
            let enc = cb.computeCommandEncoder().expect("enc");
            enc.setComputePipelineState(&pipes.mr_query);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(cloud.metal_buffer()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(self.gpu_cells[0].metal_buffer()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(self.gpu_cells[1].metal_buffer()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(self.gpu_cells[2].metal_buffer()), 0, 3);
                enc.setBuffer_offset_atIndex(Some(self.gpu_cells[3].metal_buffer()), 0, 4);
                enc.setBuffer_offset_atIndex(Some(self.gpu_fits[0].metal_buffer()), 0, 5);
                enc.setBuffer_offset_atIndex(Some(self.gpu_fits[1].metal_buffer()), 0, 6);
                enc.setBuffer_offset_atIndex(Some(self.gpu_fits[2].metal_buffer()), 0, 7);
                enc.setBuffer_offset_atIndex(Some(self.gpu_fits[3].metal_buffer()), 0, 8);
                enc.setBuffer_offset_atIndex(Some(out_gpu.metal_buffer()), 0, 9);
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                    size_of::<QueryParams>(),
                    10,
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
        self.levels.iter().map(|l| l.cells.len() as u32).sum()
    }

    fn get_map_with_recency(&self) -> Vec<(nalgebra::Point3<f32>, u8, u32)> {
        let mut out = Vec::with_capacity(self.levels.iter().map(|l| l.cells.len()).sum());
        for (level, l) in self.levels.iter().enumerate() {
            out.extend(l.cells.iter().map(|c| {
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

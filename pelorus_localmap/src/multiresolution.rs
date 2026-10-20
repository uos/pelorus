//! Bounded CPU distribution pyramid. Each level receives the
//! same observations exactly once. Only one level contributes a query residual.
//! Inspired by VoxelMap's geometry-dependent support and MARS's surfel pyramid.
use nalgebra::{Isometry3, Matrix3, Point3, Vector3};
use pelorus_core::PointXYZIT;
use rustc_hash::FxHashMap;
use std::hash::{BuildHasher, BuildHasherDefault};
use std::sync::{OnceLock, RwLock};

use crate::VoxelDistribution;

type Grid = (i32, i32, i32);

const MISS_SHARDS: usize = 16;

/// Fits of cells that are absent from a level, valid until the next map update.
/// The iterated correction queries the same absent cells up to max_iter + 1 times
/// per scan, and the fit only depends on the key and the unchanged map.
struct MissCache {
    shards: [RwLock<FxHashMap<Grid, Option<PlaneMatch>>>; MISS_SHARDS],
}

impl MissCache {
    fn new() -> Self {
        Self {
            shards: std::array::from_fn(|_| RwLock::new(FxHashMap::default())),
        }
    }

    fn shard(&self, key: &Grid) -> &RwLock<FxHashMap<Grid, Option<PlaneMatch>>> {
        let hash = BuildHasherDefault::<rustc_hash::FxHasher>::default().hash_one(key);
        &self.shards[(hash >> 32) as usize % MISS_SHARDS]
    }

    fn get_or_insert(
        &self,
        key: Grid,
        fit: impl FnOnce() -> Option<PlaneMatch>,
    ) -> Option<PlaneMatch> {
        let shard = self.shard(&key);
        if let Some(cached) = shard.read().unwrap().get(&key) {
            return *cached;
        }
        let computed = fit();
        *shard.write().unwrap().entry(key).or_insert(computed)
    }

    fn clear(&mut self) {
        for shard in &mut self.shards {
            shard.get_mut().unwrap().clear();
        }
    }
}

fn plane_eigen(covariance: Matrix3<f32>) -> (Vector3<f32>, Matrix3<f32>) {
    let mut a = covariance.cast::<f64>();
    let mut v = Matrix3::<f64>::identity();
    for _ in 0..24 {
        let (mut p, mut q) = (0, 1);
        if a[(0, 2)].abs() > a[(p, q)].abs() {
            (p, q) = (0, 2);
        }
        if a[(1, 2)].abs() > a[(p, q)].abs() {
            (p, q) = (1, 2);
        }
        if a[(p, q)].abs() < 1e-14 * (a[(0, 0)].abs() + a[(1, 1)].abs() + a[(2, 2)].abs() + 1e-20) {
            break;
        }
        let tau = (a[(q, q)] - a[(p, p)]) / (2.0 * a[(p, q)]);
        let t = (if tau < 0.0 { -1.0 } else { 1.0 }) / (tau.abs() + (1.0 + tau * tau).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let (app, aqq, apq) = (a[(p, p)], a[(q, q)], a[(p, q)]);
        a[(p, p)] = app - t * apq;
        a[(q, q)] = aqq + t * apq;
        a[(p, q)] = 0.0;
        a[(q, p)] = 0.0;
        for k in 0..3 {
            if k != p && k != q {
                let (kp, kq) = (a[(k, p)], a[(k, q)]);
                a[(k, p)] = c * kp - s * kq;
                a[(p, k)] = a[(k, p)];
                a[(k, q)] = s * kp + c * kq;
                a[(q, k)] = a[(k, q)];
            }
            let (vp, vq) = (v[(k, p)], v[(k, q)]);
            v[(k, p)] = c * vp - s * vq;
            v[(k, q)] = s * vp + c * vq;
        }
    }
    let mut indices = [0, 1, 2];
    for i in 0..3 {
        for j in i + 1..3 {
            if a[(indices[j], indices[j])] < a[(indices[i], indices[i])] {
                indices.swap(i, j);
            }
        }
    }
    (
        Vector3::from_fn(|i, _| a[(indices[i], indices[i])] as f32),
        Matrix3::from_fn(|r, c| v[(r, indices[c])] as f32),
    )
}

fn saturating_u32(value: impl TryInto<u32>) -> u32 {
    value.try_into().unwrap_or(u32::MAX)
}

/// Moments of one map cell. Correspondences only read the moments, so the
/// Cholesky factor that [`VoxelDistribution`] keeps for sampling is not stored.
/// Counts and scan numbers fit in 32 bits, which keeps a cell small on
/// memory-constrained targets.
#[derive(Clone, Copy, Debug)]
struct CellDistribution {
    mean: Point3<f32>,
    covariance: Matrix3<f32>,
    point_count: u32,
    last_updated_at_scan: u32,
}

impl CellDistribution {
    fn from_voxel(distribution: &VoxelDistribution) -> Self {
        Self {
            mean: distribution.mean,
            covariance: distribution.covariance,
            point_count: saturating_u32(distribution.point_count),
            last_updated_at_scan: saturating_u32(distribution.last_updated_at_scan),
        }
    }

    /// Moments only. The identity factor is never read by pooling or fitting.
    fn to_voxel(self) -> VoxelDistribution {
        VoxelDistribution {
            transform: Matrix3::identity(),
            point_count: self.point_count.into(),
            mean: self.mean,
            covariance: self.covariance,
            last_updated_at_scan: self.last_updated_at_scan as usize,
        }
    }

    /// Full distribution for callers outside the map, with its Cholesky factor.
    fn export(self) -> VoxelDistribution {
        let mut distribution = self.to_voxel();
        if let Some(chol) = nalgebra::Cholesky::new(self.covariance) {
            distribution.transform = chol.l();
        }
        distribution
    }
}

fn update_distribution(
    a: &mut CellDistribution,
    b: &VoxelDistribution,
    sensitivity: f32,
    scan: u32,
) {
    if b.point_count == 0 {
        return;
    }
    a.last_updated_at_scan = scan;
    // Ablation: plain count-weighted merging, as for the registration neighborhood,
    // without the maturity-weighted kernel.
    #[cfg(feature = "plain-update")]
    {
        let _ = sensitivity;
        let mut pooled = a.to_voxel();
        pooled.pool(*b);
        *a = CellDistribution {
            last_updated_at_scan: scan,
            ..CellDistribution::from_voxel(&pooled)
        };
        return;
    }
    #[allow(unreachable_code)]
    if a.point_count == 0 {
        a.mean = b.mean;
        a.covariance = b.covariance;
        a.point_count = saturating_u32(b.point_count);
        return;
    }
    let delta = a.mean - b.mean;
    let l00 = (a.covariance[(0, 0)] + 1e-6).sqrt();
    let l10 = a.covariance[(1, 0)] / l00;
    let l20 = a.covariance[(2, 0)] / l00;
    let l11 = (a.covariance[(1, 1)] + 1e-6 - l10 * l10).sqrt();
    let l21 = (a.covariance[(2, 1)] - l20 * l10) / l11;
    let l22 = (a.covariance[(2, 2)] + 1e-6 - (l20 * l20 + l21 * l21)).sqrt();
    let dot = |x: Vector3<f32>, y: Vector3<f32>| (x.x * y.x + x.y * y.y) + x.z * y.z;
    let mut m = dot(delta, delta);
    if [l00, l11, l22].iter().all(|x| x.is_finite() && *x > 0.0) {
        let y0 = delta.x / l00;
        let y1 = (delta.y - l10 * y0) / l11;
        let y2 = (delta.z - (l20 * y0 + l21 * y1)) / l22;
        let x2 = y2 / l22;
        let x1 = (y1 - l21 * x2) / l11;
        let x0 = (y0 - (l10 * x1 + l20 * x2)) / l00;
        m = dot(delta, Vector3::new(x0, x1, x2));
    }
    // The kernel narrows as the voxel matures.
    let maturity_decay = (a.point_count as f32).ln().max(1.0);
    let alpha = (-m / (2.0 * (sensitivity / maturity_decay))).exp();
    if alpha < 1e-5 {
        return;
    }
    let oma = 1.0 - alpha;
    for i in 0..3 {
        a.mean[i] = oma * a.mean[i] + alpha * b.mean[i];
    }
    for r in 0..3 {
        for c in 0..3 {
            a.covariance[(r, c)] = oma * a.covariance[(r, c)]
                + alpha * b.covariance[(r, c)]
                + alpha * oma * (delta[r] * delta[c]);
        }
    }
    a.point_count = a.point_count.saturating_add(saturating_u32(b.point_count));
}

struct Cell {
    distribution: CellDistribution,
    neighborhood: OnceLock<Option<PlaneMatch>>,
}

/// Cells live in a vector reserved once at `capacity`; the hash map only maps a
/// grid key to a slot. A scan's new cells are ranked against the existing ones
/// before insertion, so a level never holds more than `capacity` cells and
/// neither allocation grows during a run. The index keeps at least half of its
/// slots free, so tombstones from evictions are rehashed in place instead of
/// doubling the table.
struct Level {
    resolution: f32,
    capacity: usize,
    index: FxHashMap<Grid, u32>,
    keys: Vec<Grid>,
    cells: Vec<Cell>,
    miss: MissCache,
}

impl Level {
    fn new(resolution: f32, capacity: usize) -> Self {
        Self {
            resolution,
            capacity,
            index: FxHashMap::with_capacity_and_hasher(2 * capacity, Default::default()),
            keys: Vec::with_capacity(capacity),
            cells: Vec::with_capacity(capacity),
            miss: MissCache::new(),
        }
    }

    fn get(&self, key: &Grid) -> Option<&Cell> {
        self.index.get(key).map(|&slot| &self.cells[slot as usize])
    }

    fn len(&self) -> usize {
        self.cells.len()
    }

    fn clear(&mut self) {
        self.index.clear();
        self.keys.clear();
        self.cells.clear();
        self.miss.clear();
    }

    fn remove(&mut self, key: &Grid) {
        let Some(slot) = self.index.remove(key) else {
            return;
        };
        let slot = slot as usize;
        self.keys.swap_remove(slot);
        self.cells.swap_remove(slot);
        if slot < self.keys.len() {
            self.index.insert(self.keys[slot], slot as u32);
        }
    }

    /// Fuses one scan into the level. Existing cells are updated first, then the
    /// scan's new cells compete with all cells for the capacity, exactly as if they
    /// were inserted and trimmed afterwards, but without exceeding the capacity.
    fn insert_scan(
        &mut self,
        batches: &[(Grid, VoxelDistribution)],
        scan: usize,
        sensitivity: f32,
    ) {
        let scan = saturating_u32(scan);
        let mut new_cells = Vec::new();
        for &(key, batch) in batches {
            match self.index.get(&key) {
                Some(&slot) => update_distribution(
                    &mut self.cells[slot as usize].distribution,
                    &batch,
                    sensitivity,
                    scan,
                ),
                None => new_cells.push((key, CellDistribution::from_voxel(&batch))),
            }
        }
        let excess = (self.len() + new_cells.len()).saturating_sub(self.capacity);
        let mut skipped = rustc_hash::FxHashSet::default();
        if excess > 0 {
            let rank = |key: Grid, d: &CellDistribution| {
                (
                    key,
                    scan.saturating_sub(d.last_updated_at_scan) as u128,
                    d.point_count.max(1) as u128,
                )
            };
            let mut ranks: Vec<_> = self
                .keys
                .iter()
                .zip(&self.cells)
                .map(|(&key, cell)| rank(key, &cell.distribution))
                .chain(new_cells.iter().map(|(key, d)| rank(*key, d)))
                .collect();
            // Same age / support principle as the fixed map. Coordinate tie breaking
            // makes eviction independent of hash iteration and thread scheduling.
            // Ablation: plain LRU, evict by age alone and ignore support.
            #[cfg(feature = "lru-eviction")]
            ranks.select_nth_unstable_by(excess - 1, |a, b| {
                b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0))
            });
            #[cfg(not(feature = "lru-eviction"))]
            ranks.select_nth_unstable_by(excess - 1, |a, b| {
                (b.1 * a.2).cmp(&(a.1 * b.2)).then_with(|| a.0.cmp(&b.0))
            });
            for (key, _, _) in &ranks[..excess] {
                if self.index.contains_key(key) {
                    self.remove(key);
                } else {
                    skipped.insert(*key);
                }
            }
        }
        for (key, distribution) in new_cells {
            if skipped.contains(&key) {
                continue;
            }
            self.index.insert(key, self.cells.len() as u32);
            self.keys.push(key);
            self.cells.push(Cell {
                distribution,
                neighborhood: OnceLock::new(),
            });
        }
    }

    fn grid(&self, point: &Point3<f32>) -> Grid {
        (
            (point.x / self.resolution).floor() as i32,
            (point.y / self.resolution).floor() as i32,
            (point.z / self.resolution).floor() as i32,
        )
    }

    fn neighborhood(&self, point: &Point3<f32>) -> VoxelDistribution {
        let grid = self.grid(point);
        let mut pooled = VoxelDistribution::new();
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let key = (
                        grid.0.saturating_add(x),
                        grid.1.saturating_add(y),
                        grid.2.saturating_add(z),
                    );
                    if let Some(cell) = self.get(&key) {
                        pooled.pool(cell.distribution.to_voxel());
                    }
                }
            }
        }
        pooled
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PlaneMatch {
    pub plane: [f32; 4],
    pub level: u8,
    mean: Point3<f32>,
    tangents: [Vector3<f32>; 2],
    extents: [f32; 2],
}

impl PlaneMatch {
    fn contains_projection(&self, query: &Point3<f32>) -> bool {
        let delta = query - self.mean;
        self.tangents
            .iter()
            .zip(self.extents)
            .all(|(tangent, extent)| tangent.dot(&delta).abs() <= extent)
    }
}

pub(crate) struct MultiResolutionMap {
    levels: [Level; 4],
    sensitivity: f32,
    // A metric thickness limit, independent of the selected support scale.
    max_thickness: f32,
}

impl MultiResolutionMap {
    pub fn new(resolution: f32, capacity: usize, sensitivity: f32) -> Self {
        assert!(resolution.is_finite() && resolution > 0.0);
        assert!(capacity >= 4);
        Self {
            levels: std::array::from_fn(|index| {
                Level::new(
                    resolution * [0.5, 1.0, 2.0, 4.0][index],
                    capacity / 4 + usize::from(index < capacity % 4),
                )
            }),
            sensitivity,
            max_thickness: resolution * 0.15,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.levels.iter().all(|level| level.len() == 0)
    }

    pub fn clear(&mut self) {
        for level in &mut self.levels {
            level.clear();
        }
    }

    #[cfg(not(feature = "nn_kdt"))]
    pub fn distributions(&self) -> Vec<VoxelDistribution> {
        // Visualize the base-resolution layer without duplicating pyramid points.
        self.levels[1]
            .cells
            .iter()
            .map(|c| c.distribution.export())
            .collect()
    }

    /// All four levels, each cell tagged with the level it lives at
    /// (0 = finest, 3 = coarsest). Unlike [`Self::distributions`] this does
    /// not collapse to a single layer, so callers can visualize the actual
    /// pyramid structure.
    pub fn distributions_with_levels(&self) -> Vec<(VoxelDistribution, u8)> {
        self.levels
            .iter()
            .enumerate()
            .flat_map(|(level, l)| l.cells.iter().map(move |c| (c.distribution.export(), level as u8)))
            .collect()
    }

    pub fn add(&mut self, points: &[PointXYZIT], transform: &Isometry3<f64>, scan: usize) {
        let mut keyed: Vec<_> = points
            .iter()
            .filter_map(|p| {
                let world = transform
                    .transform_point(&Point3::new(p.x, p.y, p.z).cast())
                    .cast::<f32>();
                world
                    .coords
                    .iter()
                    .all(|x| x.is_finite())
                    .then(|| (self.levels[0].grid(&world), world))
            })
            .collect();
        // Preserve input order inside each voxel, matching the GPU reductions.
        keyed.sort_by_key(|(key, _)| *key);
        let mut batches = Vec::new();
        for group in keyed.chunk_by(|a, b| a.0 == b.0) {
            let mean = Point3::from(
                group.iter().map(|(_, p)| p.coords).sum::<Vector3<f32>>() / group.len() as f32,
            );
            let covariance = group.iter().fold(Matrix3::zeros(), |sum, (_, p)| {
                let delta = p - mean;
                sum + delta * delta.transpose()
            }) / group.len() as f32;
            batches.push((
                group[0].0,
                VoxelDistribution {
                    mean,
                    covariance,
                    point_count: group.len() as u64,
                    last_updated_at_scan: scan,
                    transform: Matrix3::identity(),
                },
            ));
        }
        for (index, level) in self.levels.iter_mut().enumerate() {
            if index > 0 {
                // Pool this scan's observations, never the accumulated map:
                // otherwise old observations would be counted again each scan.
                for (key, _) in &mut batches {
                    *key = (
                        key.0.div_euclid(2),
                        key.1.div_euclid(2),
                        key.2.div_euclid(2),
                    );
                }
                // Parent cells must pool children in the same order on every backend.
                batches.sort_by_key(|(key, _)| *key);
                batches = batches
                    .chunk_by(|a, b| a.0 == b.0)
                    .map(|group| {
                        let mut distribution = VoxelDistribution::new();
                        for (_, d) in group {
                            distribution.pool(*d);
                        }
                        distribution.last_updated_at_scan = scan;
                        (group[0].0, distribution)
                    })
                    .collect();
            }
            level.insert_scan(&batches, scan, self.sensitivity);
            // Fits depend on neighboring cells too: invalidate after all updates
            // and evictions, before the next scan's parallel correspondence search.
            for cell in level.cells.iter_mut() {
                cell.neighborhood.take();
            }
            level.miss.clear();
        }
    }

    fn fit(&self, distribution: VoxelDistribution, level: u8) -> Option<PlaneMatch> {
        if distribution.point_count < 6 {
            return None;
        }
        if !distribution.covariance.iter().all(|v| v.is_finite()) {
            return None;
        }
        let (eigenvalues, eigenvectors) = plane_eigen(distribution.covariance);
        let [small, middle, large] = [0, 1, 2];
        let thickness = eigenvalues[small].max(0.0);
        let spread = eigenvalues[middle];
        if !spread.is_finite() || !eigenvalues[large].is_finite() {
            return None;
        }
        if spread < 1e-4 {
            return None;
        }
        if thickness > self.max_thickness.powi(2) || thickness > 0.1 * spread {
            return None;
        }
        let normal = eigenvectors.column(small).into_owned();
        // Restrict association to the observed finite patch in both tangent
        // directions. A small floor handles sparse narrow but valid patches.
        let tangents = [middle, large].map(|i| eigenvectors.column(i).into_owned());
        let extents = [middle, large]
            .map(|i| 3.0 * eigenvalues[i].max(0.0).sqrt() + self.levels[0].resolution);
        // Registration applies its range-dependent residual gate as before.
        let d = -normal.dot(&distribution.mean.coords);
        Some(PlaneMatch {
            plane: [normal.x, normal.y, normal.z, d],
            level,
            mean: distribution.mean,
            tangents,
            extents,
        })
    }

    pub fn query(&self, point: &Point3<f32>) -> Option<PlaneMatch> {
        if !point.coords.iter().all(|x| x.is_finite()) {
            return None;
        }
        // The finest level with a valid plane supplies the residual. Coarser
        // levels only step in where the finer ones are sparse or degenerate.
        for (index, level) in (0u8..).zip(&self.levels) {
            let key = level.grid(point);
            let candidate = if let Some(cell) = level.get(&key) {
                *cell
                    .neighborhood
                    .get_or_init(|| self.fit(level.neighborhood(point), index))
            } else {
                level
                    .miss
                    .get_or_insert(key, || self.fit(level.neighborhood(point), index))
            };
            // A thick or degenerate fine cell (an edge, a pole) still lets a
            // coarser level supply the plane it is part of.
            let Some(candidate) = candidate else {
                continue;
            };
            if candidate.contains_projection(point) {
                return Some(candidate);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Old but well supported (S = 10 / 100), recent but sparse (S = 2 / 1), fresh (S = 0).
    fn trimmed_level() -> Level {
        let mut level = Level::new(1.0, 2);
        for (x, scan, point_count) in [(0, 0, 100), (1, 8, 1), (2, 10, 1)] {
            let distribution = VoxelDistribution {
                point_count,
                last_updated_at_scan: scan,
                ..VoxelDistribution::new()
            };
            level.insert_scan(&[((x, 0, 0), distribution)], scan, 10.0);
        }
        assert_eq!(level.len(), 2);
        level
    }

    #[test]
    fn level_never_exceeds_capacity() {
        let mut level = Level::new(1.0, 3);
        let reserved = (level.index.capacity(), level.cells.capacity());
        for scan in 0..50 {
            let batches: Vec<_> = (0..5)
                .map(|i| {
                    let distribution = VoxelDistribution {
                        point_count: 1 + i as u64,
                        last_updated_at_scan: scan,
                        ..VoxelDistribution::new()
                    };
                    ((scan as i32 * 5 + i, 0, 0), distribution)
                })
                .collect();
            level.insert_scan(&batches, scan, 10.0);
            assert!(level.len() <= 3);
            for (slot, key) in level.keys.iter().enumerate() {
                assert_eq!(level.index[key] as usize, slot);
            }
        }
        assert_eq!((level.index.capacity(), level.cells.capacity()), reserved);
    }

    #[cfg(not(feature = "lru-eviction"))]
    #[test]
    fn trim_evicts_highest_age_per_support() {
        let level = trimmed_level();
        assert!(level.get(&(0, 0, 0)).is_some());
        assert!(level.get(&(1, 0, 0)).is_none());
        assert!(level.get(&(2, 0, 0)).is_some());
    }

    #[cfg(feature = "lru-eviction")]
    #[test]
    fn trim_evicts_least_recently_updated() {
        let level = trimmed_level();
        assert!(level.get(&(0, 0, 0)).is_none());
        assert!(level.get(&(1, 0, 0)).is_some());
        assert!(level.get(&(2, 0, 0)).is_some());
    }
}

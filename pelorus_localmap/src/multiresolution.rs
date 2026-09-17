//! Bounded CPU distribution pyramid. Each level receives the
//! same observations exactly once. Only one level contributes a query residual.
//! Inspired by VoxelMap's geometry-dependent support and MARS's surfel pyramid.
use nalgebra::{Isometry3, Matrix3, Point3, Vector3};
use pelorus_core::PointXYZIT;
use rustc_hash::FxHashMap;
use std::sync::OnceLock;

use crate::VoxelDistribution;

type Grid = (i32, i32, i32);

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

fn update_distribution(
    a: &mut VoxelDistribution,
    b: &VoxelDistribution,
    sensitivity: f32,
    scan: usize,
) {
    if b.point_count == 0 {
        return;
    }
    a.last_updated_at_scan = scan;
    if a.point_count == 0 {
        a.mean = b.mean;
        a.covariance = b.covariance;
        a.point_count = b.point_count;
        a.transform = b.transform;
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
    let alpha = (-m / (2.0 * (sensitivity / (a.point_count as f32).ln().max(1.0)))).exp();
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
    a.point_count += b.point_count;
    if let Some(chol) = nalgebra::Cholesky::new(a.covariance) {
        a.transform = chol.l();
    }
}

struct Cell {
    distribution: VoxelDistribution,
    neighborhood: OnceLock<Result<PlaneMatch, FitFailure>>,
}

#[derive(Clone, Copy, Debug)]
enum FitFailure {
    InsufficientSupport,
    ConflictingGeometry,
}

struct Level {
    resolution: f32,
    capacity: usize,
    cells: FxHashMap<Grid, Cell>,
}

impl Level {
    fn grid(&self, point: &Point3<f32>) -> Grid {
        (
            (point.x / self.resolution).floor() as i32,
            (point.y / self.resolution).floor() as i32,
            (point.z / self.resolution).floor() as i32,
        )
    }

    fn trim(&mut self, scan: usize) {
        let excess = self.cells.len().saturating_sub(self.capacity);
        if excess == 0 {
            return;
        }
        // Same age / support principle as the fixed map. Coordinate tie breaking
        // makes eviction independent of hash iteration and thread scheduling.
        let mut ranks: Vec<_> = self
            .cells
            .iter()
            .map(|(&key, cell)| {
                let d = &cell.distribution;
                (
                    key,
                    scan.saturating_sub(d.last_updated_at_scan) as u128,
                    d.point_count.max(1) as u128,
                )
            })
            .collect();
        ranks.select_nth_unstable_by(excess, |a, b| {
            (b.1 * a.2).cmp(&(a.1 * b.2)).then_with(|| a.0.cmp(&b.0))
        });
        for (key, _, _) in &ranks[..excess] {
            self.cells.remove(key);
        }
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
                    if let Some(cell) = self.cells.get(&key) {
                        pooled.pool(cell.distribution);
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
    pub level: usize,
    mean: Point3<f32>,
    normal: Vector3<f32>,
    tangents: [Vector3<f32>; 2],
    extents: [f32; 2],
    normal_quality: f32,
}

impl PlaneMatch {
    fn agrees_with_normal(&self, other: &Self) -> bool {
        let cosine = self.normal.dot(&other.normal).abs().min(1.0);
        // Squared sine is sign-invariant and approximates squared angular
        // disagreement for close normals. Confident fits should not rotate
        // merely because a larger support still passes the broad angle cap.
        cosine >= 0.98 && 1.0 - cosine * cosine <= self.normal_quality + other.normal_quality
    }

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
            levels: std::array::from_fn(|index| Level {
                resolution: resolution * [0.5, 1.0, 2.0, 4.0][index],
                capacity: capacity / 4 + usize::from(index < capacity % 4),
                cells: FxHashMap::default(),
            }),
            sensitivity,
            max_thickness: resolution * 0.15,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.levels.iter().all(|level| level.cells.is_empty())
    }

    pub fn clear(&mut self) {
        for level in &mut self.levels {
            level.cells.clear();
        }
    }

    pub fn distributions(&self) -> Vec<VoxelDistribution> {
        // Visualize the base-resolution layer without duplicating pyramid points.
        self.levels[1]
            .cells
            .values()
            .map(|c| c.distribution)
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
            .flat_map(|(level, l)| l.cells.values().map(move |c| (c.distribution, level as u8)))
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
            for &(key, batch) in &batches {
                match level.cells.entry(key) {
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(Cell {
                            distribution: batch,
                            neighborhood: OnceLock::new(),
                        });
                    }
                    std::collections::hash_map::Entry::Occupied(mut entry) => {
                        update_distribution(
                            &mut entry.get_mut().distribution,
                            &batch,
                            self.sensitivity,
                            scan,
                        );
                    }
                }
            }
            level.trim(scan);
            // Fits depend on neighboring cells too: invalidate after all updates
            // and evictions, before the next scan's parallel correspondence search.
            for cell in level.cells.values_mut() {
                cell.neighborhood.take();
            }
        }
    }

    fn fit(&self, distribution: VoxelDistribution, level: usize) -> Result<PlaneMatch, FitFailure> {
        if distribution.point_count < 6 {
            return Err(FitFailure::InsufficientSupport);
        }
        if !distribution.covariance.iter().all(|v| v.is_finite()) {
            return Err(FitFailure::ConflictingGeometry);
        }
        let (eigenvalues, eigenvectors) = plane_eigen(distribution.covariance);
        let [small, middle, large] = [0, 1, 2];
        let thickness = eigenvalues[small].max(0.0);
        let spread = eigenvalues[middle];
        if !spread.is_finite() || !eigenvalues[large].is_finite() {
            return Err(FitFailure::ConflictingGeometry);
        }
        if spread < 1e-4 {
            return Err(FitFailure::InsufficientSupport);
        }
        if thickness > self.max_thickness.powi(2) || thickness > 0.1 * spread {
            return Err(FitFailure::ConflictingGeometry);
        }
        let normal = eigenvectors.column(small).into_owned();
        // Restrict association to the observed finite patch in both tangent
        // directions. A small floor handles sparse narrow but valid patches.
        let tangents = [middle, large].map(|i| eigenvectors.column(i).into_owned());
        let extents = [middle, large]
            .map(|i| 3.0 * eigenvalues[i].max(0.0).sqrt() + self.levels[0].resolution);
        // Prefer a well-constrained normal, not just a large support region.
        // This is a geometric quality proxy, not calibrated plane uncertainty:
        // robust temporal blending makes point_count unsuitable as an IID count.
        // The noise floor prevents nearly rank-two tiny patches winning solely
        // because their measured thickness rounded to zero.
        let noise_floor = (self.levels[1].resolution * 0.04).powi(2);
        let normal_quality = (thickness + noise_floor)
            * (1.0 / (eigenvalues[middle] - thickness).max(1e-6)
                + 1.0 / (eigenvalues[large] - thickness).max(1e-6));
        // Registration applies its range-dependent residual gate as before.
        let d = -normal.dot(&distribution.mean.coords);
        Ok(PlaneMatch {
            plane: [normal.x, normal.y, normal.z, d],
            level,
            mean: distribution.mean,
            normal,
            tangents,
            extents,
            normal_quality,
        })
    }

    pub fn query(&self, point: &Point3<f32>) -> Option<PlaneMatch> {
        if !point.coords.iter().all(|x| x.is_finite()) {
            return None;
        }
        let mut selected: Option<PlaneMatch> = None;
        for (index, level) in self.levels.iter().enumerate() {
            let candidate = if let Some(cell) = level.cells.get(&level.grid(point)) {
                *cell
                    .neighborhood
                    .get_or_init(|| self.fit(level.neighborhood(point), index))
            } else {
                self.fit(level.neighborhood(point), index)
            };
            let candidate = match candidate {
                Ok(candidate) => candidate,
                Err(FitFailure::InsufficientSupport) => continue,
                Err(FitFailure::ConflictingGeometry) => break,
            };
            if !candidate.contains_projection(point) {
                continue;
            }
            if let Some(finer) = selected {
                // Do not let a larger neighborhood rotate or shift a supported
                // local surface. Normal signs are arbitrary.
                if !finer.agrees_with_normal(&candidate)
                    || finer.normal.dot(&(candidate.mean - finer.mean)).abs() > self.max_thickness
                    || candidate.normal.dot(&(finer.mean - candidate.mean)).abs()
                        > self.max_thickness
                {
                    break;
                }
                if candidate.normal_quality >= finer.normal_quality {
                    continue;
                }
            }
            selected = Some(candidate);
        }
        selected
    }
}


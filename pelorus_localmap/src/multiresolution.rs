//! Experimental, bounded CPU distribution pyramid. Each level receives the
//! same observations exactly once. Only one level contributes a query residual.
//! Inspired by VoxelMap's geometry-dependent support and MARS's surfel pyramid.
use nalgebra::{Isometry3, Matrix3, Point3, Vector3};
use pelorus_core::PointXYZIT;
use rustc_hash::FxHashMap;
use std::sync::OnceLock;

use crate::VoxelDistribution;

type Grid = (i32, i32, i32);

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
            .flat_map(|(level, l)| {
                l.cells
                    .values()
                    .map(move |c| (c.distribution, level as u8))
            })
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
        keyed.sort_unstable_by_key(|(key, _)| *key);
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
                batches.sort_unstable_by_key(|(key, _)| *key);
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
                        entry.get_mut().distribution.update(
                            &batch,
                            self.sensitivity,
                            level.resolution / 4.0,
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
        let eigen = distribution.covariance.symmetric_eigen();
        let mut indices = [0, 1, 2];
        indices.sort_unstable_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));
        let [small, middle, large] = indices;
        let thickness = eigen.eigenvalues[small].max(0.0);
        let spread = eigen.eigenvalues[middle];
        if !spread.is_finite() {
            return Err(FitFailure::ConflictingGeometry);
        }
        if spread < 1e-4 {
            return Err(FitFailure::InsufficientSupport);
        }
        if thickness > self.max_thickness.powi(2) || thickness > 0.1 * spread {
            return Err(FitFailure::ConflictingGeometry);
        }
        let normal = eigen.eigenvectors.column(small).into_owned();
        // Restrict association to the observed finite patch in both tangent
        // directions. A small floor handles sparse narrow but valid patches.
        let tangents = [middle, large].map(|i| eigen.eigenvectors.column(i).into_owned());
        let extents = [middle, large]
            .map(|i| 3.0 * eigen.eigenvalues[i].max(0.0).sqrt() + self.levels[0].resolution);
        // Prefer a well-constrained normal, not just a large support region.
        // This is a geometric quality proxy, not calibrated plane uncertainty:
        // robust temporal blending makes point_count unsuitable as an IID count.
        // The noise floor prevents nearly rank-two tiny patches winning solely
        // because their measured thickness rounded to zero.
        let noise_floor = (self.levels[1].resolution * 0.04).powi(2);
        let normal_quality = (thickness + noise_floor)
            * (1.0 / (eigen.eigenvalues[middle] - thickness).max(1e-6)
                + 1.0 / (eigen.eigenvalues[large] - thickness).max(1e-6));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_agreement_respects_fit_quality_and_sign() {
        let mut map = MultiResolutionMap::new(0.5, 12000, 10.0);
        map.add(&patch(0.1, 10, 0.0), &Isometry3::identity(), 1);
        let mut fine = map.query(&Point3::new(0.1, 0.1, 0.01)).unwrap();
        fine.normal = Vector3::z();
        fine.normal_quality = 0.0001;
        let mut coarse = fine;
        let angle = 5.0_f32.to_radians();
        coarse.normal = Vector3::new(angle.sin(), 0.0, angle.cos());
        assert!(fine.normal.dot(&coarse.normal) > 0.98);
        assert!(!fine.agrees_with_normal(&coarse));
        coarse.normal_quality = 0.01;
        assert!(fine.agrees_with_normal(&coarse));
        coarse.normal = -coarse.normal;
        assert!(fine.agrees_with_normal(&coarse));
        coarse.normal = Vector3::x();
        coarse.normal_quality = 10.0;
        assert!(!fine.agrees_with_normal(&coarse));
    }

    fn point(x: f32, y: f32, z: f32) -> PointXYZIT {
        PointXYZIT {
            x,
            y,
            z,
            ..Default::default()
        }
    }

    fn patch(step: f32, count: i32, z: f32) -> Vec<PointXYZIT> {
        (-count..=count)
            .flat_map(|x| {
                (-count..=count)
                    .map(move |y| point(x as f32 * step + 0.03, y as f32 * step + 0.03, z))
            })
            .collect()
    }

    #[test]
    fn sparse_plane_uses_coarse_support() {
        let mut map = MultiResolutionMap::new(0.5, 12000, 10.0);
        map.add(&patch(0.6, 3, 0.0), &Isometry3::identity(), 1);
        let query = Point3::new(0.03, 0.03, 0.01);
        assert!(map.fit(map.levels[0].neighborhood(&query), 0).is_err());
        let matched = map.query(&query).unwrap();
        assert!(matched.level >= 2);
        assert!(matched.normal.z.abs() > 0.999);
        assert!(matched.plane[3].abs() < 1e-5);
    }

    #[test]
    fn corner_keeps_fine_surface() {
        let mut points = patch(0.05, 30, 0.0);
        points.extend(
            (-30..=30)
                .flat_map(|y| (0..=30).map(move |z| point(0.0, y as f32 * 0.05, z as f32 * 0.05))),
        );
        let mut map = MultiResolutionMap::new(0.5, 12000, 10.0);
        map.add(&points, &Isometry3::identity(), 1);
        let matched = map.query(&Point3::new(0.6, 0.1, 0.01)).unwrap();
        assert_eq!(matched.level, 0);
        assert!(matched.normal.z.abs() > 0.999);
    }

    #[test]
    fn parallel_surfaces_are_not_replaced_by_midplane() {
        let mut points = patch(0.05, 30, 0.01);
        points.extend(patch(0.05, 30, 0.61));
        let mut map = MultiResolutionMap::new(0.5, 12000, 10.0);
        map.add(&points, &Isometry3::identity(), 1);
        let matched = map.query(&Point3::new(0.1, 0.1, 0.01)).unwrap();
        assert_eq!(matched.level, 0);
        assert!(
            (matched.normal.dot(&Point3::new(0.1, 0.1, 0.01).coords) + matched.plane[3]).abs()
                < 1e-4
        );
    }

    #[test]
    fn line_and_nonfinite_queries_have_no_plane() {
        let mut map = MultiResolutionMap::new(0.5, 100, 10.0);
        let points: Vec<_> = (0..100).map(|i| point(i as f32 * 0.01, 0.0, 0.0)).collect();
        map.add(&points, &Isometry3::identity(), 1);
        assert!(map.query(&Point3::new(0.2, 0.0, 0.0)).is_none());
        assert!(map.query(&Point3::new(f32::NAN, 0.0, 0.0)).is_none());
    }

    #[test]
    fn observations_are_counted_once_per_level_with_negative_coordinates() {
        let mut map = MultiResolutionMap::new(0.5, 12000, 10.0);
        let points = patch(0.07, 10, -0.25);
        for scan in 1..=2 {
            map.add(&points, &Isometry3::identity(), scan);
        }
        for level in &map.levels {
            assert_eq!(
                level
                    .cells
                    .values()
                    .map(|d| d.distribution.point_count)
                    .sum::<u64>(),
                2 * points.len() as u64
            );
            for (&key, d) in &level.cells {
                assert_eq!(key, level.grid(&d.distribution.mean));
            }
        }
        for key in map.levels[0].cells.keys() {
            let parent = (
                key.0.div_euclid(2),
                key.1.div_euclid(2),
                key.2.div_euclid(2),
            );
            assert!(map.levels[1].cells.contains_key(&parent));
        }
    }

    #[test]
    fn total_capacity_is_bounded_and_reset_allows_reuse() {
        let mut map = MultiResolutionMap::new(0.5, 101, 10.0);
        for scan in 1..=5 {
            map.add(&patch(0.2, 20, scan as f32), &Isometry3::identity(), scan);
            assert!(map.levels.iter().map(|l| l.cells.len()).sum::<usize>() <= 101);
        }
        map.clear();
        assert!(map.is_empty());
        map.add(&patch(0.05, 3, 0.0), &Isometry3::identity(), 6);
        assert!(map.query(&Point3::origin()).is_some());
    }

    #[test]
    fn finite_support_rejects_extrapolation() {
        let mut map = MultiResolutionMap::new(0.5, 100, 10.0);
        map.add(&patch(0.01, 3, 0.0), &Isometry3::identity(), 1);
        assert!(map.query(&Point3::new(1.0, 0.0, 0.0)).is_none());
        assert!(map.query(&Point3::new(0.03, 0.03, 0.01)).is_some());
    }

    #[test]
    fn cached_rejection_is_invalidated_when_neighbors_change() {
        let mut map = MultiResolutionMap::new(0.5, 12000, 10.0);
        let query = Point3::new(0.03, 0.03, 0.0);
        map.add(
            &[point(query.x, query.y, query.z)],
            &Isometry3::identity(),
            1,
        );
        assert!(map.query(&query).is_none());
        let grid = map.levels[0].grid(&query);
        assert!(
            map.levels[0].cells[&grid]
                .neighborhood
                .get()
                .unwrap()
                .is_err()
        );
        let neighbors: Vec<_> = patch(0.06, 10, 0.0)
            .into_iter()
            .filter(|p| map.levels[0].grid(&Point3::new(p.x, p.y, p.z)) != grid)
            .collect();
        map.add(&neighbors, &Isometry3::identity(), 2);
        assert!(map.levels[0].cells[&grid].neighborhood.get().is_none());
        assert!(map.query(&query).is_some());
    }

    #[test]
    fn parallel_queries_share_identical_fits() {
        let mut map = MultiResolutionMap::new(0.5, 12000, 10.0);
        map.add(&patch(0.05, 20, 0.0), &Isometry3::identity(), 1);
        let map = &map;
        let planes = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(move || map.query(&Point3::new(0.1, 0.1, 0.01)).unwrap().plane)
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(planes.iter().all(|p| p == &planes[0]));
    }

    #[test]
    fn coarser_valid_patch_must_improve_normal_quality() {
        let mut map = MultiResolutionMap::new(0.5, 100, 10.0);
        // Both patches satisfy the thickness and planarity gates and have the
        // same normal/center. The larger one has substantially more roughness.
        for (index, covariance) in [
            Matrix3::from_diagonal(&Vector3::new(0.01, 0.01, 0.0)),
            Matrix3::from_diagonal(&Vector3::new(0.05, 0.05, 0.004)),
        ]
        .into_iter()
        .enumerate()
        {
            map.levels[index].cells.insert(
                (0, 0, 0),
                Cell {
                    distribution: VoxelDistribution {
                        covariance,
                        mean: Point3::origin(),
                        point_count: 100,
                        ..VoxelDistribution::new()
                    },
                    neighborhood: OnceLock::new(),
                },
            );
        }
        assert!(
            map.fit(map.levels[1].neighborhood(&Point3::origin()), 1)
                .is_ok()
        );
        assert_eq!(map.query(&Point3::origin()).unwrap().level, 0);
    }

    #[test]
    fn coarse_ground_must_not_hide_a_conflicting_fine_surface() {
        let mut map = MultiResolutionMap::new(0.5, 100, 10.0);
        for (index, covariance) in [
            Matrix3::from_diagonal(&Vector3::new(0.01, 0.02, 0.008)),
            Matrix3::from_diagonal(&Vector3::new(0.5, 0.5, 0.001)),
        ]
        .into_iter()
        .enumerate()
        {
            map.levels[index].cells.insert(
                (0, 0, 0),
                Cell {
                    distribution: VoxelDistribution {
                        covariance,
                        mean: Point3::origin(),
                        point_count: 100,
                        ..VoxelDistribution::new()
                    },
                    neighborhood: OnceLock::new(),
                },
            );
        }
        // The coarser patch could pass its own fit gates after dilution by many
        // ground observations, but fine-scale evidence already contradicts it.
        assert!(
            map.fit(map.levels[1].neighborhood(&Point3::origin()), 1)
                .is_ok()
        );
        assert!(map.query(&Point3::origin()).is_none());
    }
}

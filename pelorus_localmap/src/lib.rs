#[cfg(all(feature = "os-rng-seed", target_arch = "wasm32", target_os = "unknown"))]
compile_error!("the os-rng-seed feature is not available in the browser");

/// Seed of the neighbor sampling, so the same input gives the same map.
#[cfg(not(feature = "os-rng-seed"))]
const SAMPLING_SEED: u64 = 0x5045_4c4f_5255_5321;

#[cfg(feature = "nn_kdt")]
use kiddo::SquaredEuclidean;
use localmap::Livodish;

pub mod localmap;
mod multiresolution;
pub use multiresolution::{DistributionLevel, MultiResolutionSnapshot};
use rand_distr::{Distribution, Normal};

use nalgebra::{Point3, Vector3};
use rand::{Rng, SeedableRng};

use nalgebra::{Cholesky, Matrix3};
use pelorus_core::adaptive_execution::ExecutionMode;
#[cfg(all(feature = "rayon", not(feature = "nn_kdt")))]
use rayon::slice::ParallelSliceMut;
use ros_pointcloud2::points::PointXYZ;

use serde::{Deserialize, Serialize};

#[cfg(all(feature = "rayon", not(feature = "nn_kdt")))]
fn sort_unstable_by_key<T, K, F>(slice: &mut [T], f: F, mode: ExecutionMode)
where
    T: Send,
    K: Ord + Send,
    F: Fn(&T) -> K + Sync,
{
    match mode {
        ExecutionMode::Sequential => slice.sort_unstable_by_key(f),
        ExecutionMode::Rayon => slice.par_sort_unstable_by_key(f),
    }
}

#[cfg(all(not(feature = "rayon"), not(feature = "nn_kdt")))]
fn sort_unstable_by_key<T, K, F>(slice: &mut [T], f: F, _mode: ExecutionMode)
where
    K: Ord,
    F: Fn(&T) -> K,
{
    slice.sort_unstable_by_key(f);
}

pub struct IVoxHashMap {
    nearby_type: IVoxNearbyType,
    inverse_resolution: f32,
    resolution: f32,
    scan_number: usize,
    #[cfg(feature = "nn_kdt")]
    kd_map: kiddo::float::kdtree::KdTree<f32, u64, 3, 64, u32>,
    #[cfg(feature = "nn_kdt")]
    kd_points: std::collections::VecDeque<std::sync::Arc<(u64, nalgebra::Point3<f32>)>>,
    #[cfg(feature = "nn_kdt")]
    kd_points_lookup: std::collections::HashMap<u64, std::sync::Arc<(u64, nalgebra::Point3<f32>)>>,
    #[cfg(feature = "nn_kdt")]
    pt_id: u64,
    #[cfg(feature = "nn_kdt")]
    max_map_size: usize,

    livodish: localmap::Livodish,
    multiresolution: Option<multiresolution::MultiResolutionMap>,
    rng: rand_pcg::Pcg64Mcg,
}

#[derive(Clone, Copy, Debug)]
pub struct VoxelDistribution {
    pub transform: Matrix3<f32>,
    pub point_count: u64,
    pub mean: Point3<f32>,
    pub covariance: Matrix3<f32>,
    pub last_updated_at_scan: usize,
}

impl Default for VoxelDistribution {
    fn default() -> Self {
        Self::new()
    }
}

impl VoxelDistribution {
    pub fn from_point(first_point: &Point3<f32>) -> Self {
        VoxelDistribution {
            point_count: 1,
            mean: *first_point,
            covariance: Matrix3::zeros(),
            transform: Matrix3::identity(),
            last_updated_at_scan: 0,
        }
    }

    pub fn new() -> Self {
        VoxelDistribution {
            point_count: 0,
            mean: Point3::origin(),
            covariance: Matrix3::zeros(),
            transform: Matrix3::identity(),
            last_updated_at_scan: 0,
        }
    }

    /// Creates a VoxelDistribution from a slice of 3D points.
    ///
    /// This function calculates the mean and covariance of the provided points
    /// to create a statistical representation of their distribution.
    ///
    /// # Arguments
    /// * `points` - A slice of `Point3<f32>` to be included in the distribution.
    ///
    /// # Returns
    /// A `VoxelDistribution` representing the provided points. Returns an empty
    /// distribution if the input slice is empty.
    pub fn from_points(points: &[(Point3<f32>, i32)], current_scan_number: usize) -> Self {
        let point_count = points.len();

        if point_count == 0 {
            return Self::new();
        }

        let sum_of_points: Vector3<f32> = points.iter().map(|p| p.0.coords).sum();
        let mean = Point3::from(sum_of_points / point_count as f32);

        let mut covariance = Matrix3::<f32>::zeros();
        for (point, _) in points {
            let diff = point - mean;
            covariance += diff * diff.transpose();
        }

        covariance /= point_count as f32;

        VoxelDistribution {
            point_count: point_count as u64,
            mean,
            covariance,
            transform: Matrix3::identity(),
            last_updated_at_scan: current_scan_number,
        }
    }

    pub fn pool(&mut self, other: Self) {
        let n1 = self.point_count as f32;
        let n2 = other.point_count as f32;

        if n2 == 0.0 {
            return;
        }
        if n1 == 0.0 {
            self.mean = other.mean;
            self.covariance = other.covariance;
            self.point_count = other.point_count;
            return;
        }

        let total_points = n1 + n2;

        let new_mean_vec = (n1 * self.mean.coords + n2 * other.mean.coords) / total_points;
        let new_mean = Point3::from(new_mean_vec);

        let mean_diff = self.mean - other.mean;
        let outer_product_mean_diff = mean_diff * mean_diff.transpose();

        let new_covariance = (n1 * self.covariance
            + n2 * other.covariance
            + outer_product_mean_diff * (n1 * n2 / total_points))
            / total_points;

        self.mean = new_mean;
        self.covariance = new_covariance;

        self.point_count = total_points as u64;
    }

    /// Fuses another distribution into this one using a weighted update.
    /// The weight is determined by the Mahalanobis distance between the two,
    /// ensuring that dissimilar distributions have less influence.
    pub fn update(
        &mut self,
        other: &Self,
        sensitivity: f32,
        _min_dist: f32,
        current_scan_number: usize,
    ) {
        if other.point_count == 0 {
            return;
        }

        self.last_updated_at_scan = current_scan_number;

        if self.point_count == 0 {
            self.point_count = other.point_count;
            self.mean = other.mean;
            self.covariance = other.covariance;
            self.transform = other.transform;
            return;
        }

        let mean_diff = self.mean - other.mean;

        let regularization_factor = 1e-6;
        let regularized_cov = self.covariance + Matrix3::identity() * regularization_factor;

        let mahalanobis_sq = if let Some(chol) = Cholesky::new(regularized_cov) {
            chol.solve(&mean_diff).dot(&mean_diff)
        } else {
            mean_diff.norm_squared()
        };

        let maturity_decay = (self.point_count as f32).ln().max(1.0);
        let effective_sensitivity = sensitivity / maturity_decay;

        // A large distance results in a weight close to 0.
        let alpha = (-mahalanobis_sq / (2.0 * effective_sensitivity)).exp();

        if alpha < 1e-5 {
            return;
        }

        let new_mean = Point3::from((1.0 - alpha) * self.mean.coords + alpha * other.mean.coords);

        let outer_product_mean_diff = mean_diff * mean_diff.transpose();
        let new_covariance = (1.0 - alpha) * self.covariance
            + alpha * other.covariance
            + alpha * (1.0 - alpha) * outer_product_mean_diff;

        self.mean = new_mean;
        self.covariance = new_covariance;
        self.point_count += other.point_count; // Aggregate point counts

        if let Some(chol) = Cholesky::new(self.covariance) {
            self.transform = chol.l();
        }
    }

    pub fn update_from_point(
        &mut self,
        new_point: &Point3<f32>,
        sensitivity: f32,
        min_dist: f32,
        current_scan_number: usize,
    ) {
        self.last_updated_at_scan = current_scan_number;

        if self.point_count < 2 {
            if self.point_count == 1 {
                let new_point_dist_to_mean_sq = (self.mean - new_point).norm_squared();
                if new_point_dist_to_mean_sq > min_dist * min_dist {
                    return;
                }
            }

            self.point_count += 1;
            let n = self.point_count as f32;
            let old_mean = self.mean;
            let diff_from_old_mean = *new_point - old_mean;
            self.mean = old_mean + diff_from_old_mean / n;

            if self.point_count > 1 {
                let diff_from_new_mean = *new_point - self.mean;
                let scatter_matrix_update = diff_from_old_mean * diff_from_new_mean.transpose();
                self.covariance = ((n - 2.0) / (n - 1.0)) * self.covariance
                    + (1.0 / (n - 1.0)) * scatter_matrix_update;
            }
            self.transform = match Cholesky::new(self.covariance) {
                Some(chol) => chol.l(),
                None => {
                    let eigen = nalgebra::SymmetricEigen::new(self.covariance);
                    Matrix3::from_diagonal(&eigen.eigenvalues.map(|e| e.max(0.0).sqrt()))
                }
            };
            return;
        }

        let maturity_decay = (self.point_count as f32 - 1.0).ln().max(1.0);
        let effective_sensitivity = sensitivity / maturity_decay;

        let diff_from_old_mean = new_point - self.mean;

        let regularization_factor = 1e-6;
        let regularized_cov = self.covariance + Matrix3::identity() * regularization_factor;

        let distance_sq = if let Some(chol) = Cholesky::new(regularized_cov) {
            self.transform = chol.l();
            chol.solve(&diff_from_old_mean).dot(&diff_from_old_mean)
        } else {
            let eigen = nalgebra::SymmetricEigen::new(self.covariance);
            self.transform = Matrix3::from_diagonal(&eigen.eigenvalues.map(|e| e.max(0.0).sqrt()));
            diff_from_old_mean.norm_squared()
        };

        let alpha = (-distance_sq / (2.0 * effective_sensitivity)).exp();

        let new_mean = ((1.0 - alpha) * self.mean.coords + alpha * new_point.coords).into();

        let new_cov_sample = (1.0 - alpha) * diff_from_old_mean;
        self.covariance =
            (1.0 - alpha) * self.covariance + alpha * (new_cov_sample * new_cov_sample.transpose());
        self.mean = new_mean;

        self.point_count += 1;
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub enum IVoxNearbyType {
    Center,
    Nearby6,

    #[default]
    Nearby18,

    Nearby26,
}

impl Default for IVoxHashMap {
    fn default() -> Self {
        Self::new(0.2, 100_000, IVoxNearbyType::default(), 3.0)
    }
}

#[derive(Debug)]
pub struct EncompassingVoxelIter {
    /// The coordinate of the minimum corner of the encompassing voxel block (the origin).
    origin: (i32, i32, i32),
    /// The scaling factor, determining the side length of the voxel cube.
    factor: i32,
    /// The current position in the iteration, from 0 to factor^3 - 1.
    current_index: i32,
    /// The total number of voxels to iterate over (factor^3).
    max_index: i32,
}

/// Implementation of the Iterator trait for `EncompassingVoxelIter`.
impl Iterator for EncompassingVoxelIter {
    type Item = (i32, i32, i32);

    /// Advances the iterator and returns the next voxel coordinate.
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.max_index {
            return None;
        }

        let i = self.current_index;
        let f = self.factor;

        let dz = i / (f * f);
        let remainder = i % (f * f);
        let dy = remainder / f;
        let dx = remainder % f;

        let (ox, oy, oz) = self.origin;
        let next_voxel = (ox + dx, oy + dy, oz + dz);
        self.current_index += 1;

        Some(next_voxel)
    }
}

/// Creates an iterator that yields all the smaller voxels that constitute
/// a larger voxel defined by a scaling `factor`.
///
/// # Arguments
/// * `grid_key` - The coordinate of any small voxel within the desired larger block.
/// * `factor` - The scaling factor (e.g., 2 for a 2x2x2 block).
///
/// # Returns
/// An `EncompassingVoxelIter` that iterates over the `factor`^3 voxels.
pub fn encompassing_voxel_iter(grid_key: (i32, i32, i32), factor: i32) -> EncompassingVoxelIter {
    assert!(factor > 0, "Factor must be positive.");

    let (gx, gy, gz) = grid_key;

    let ox = gx.div_euclid(factor) * factor;
    let oy = gy.div_euclid(factor) * factor;
    let oz = gz.div_euclid(factor) * factor;

    EncompassingVoxelIter {
        origin: (ox, oy, oz),
        factor,
        current_index: 0,
        max_index: factor.pow(3),
    }
}

#[cfg(not(feature = "nn_kdt"))]
fn hash(point: &Point3<f32>, voxel_size: f32) -> i32 {
    const PRIME_1: u32 = 73_856_093;
    const PRIME_2: u32 = 471_943;
    const PRIME_3: u32 = 83_492_791;

    let xyz = point * (1.0 / voxel_size);

    let x = xyz.x.floor() as i32;
    let y = xyz.y.floor() as i32;
    let z = xyz.z.floor() as i32;

    let h = (x as u32)
        .wrapping_mul(PRIME_1)
        .wrapping_add((y as u32).wrapping_mul(PRIME_2))
        .wrapping_add((z as u32).wrapping_mul(PRIME_3));

    h as i32
}

impl IVoxHashMap {
    /// Snapshot all pyramid levels without re-hashing distribution means or
    /// populating the CPU correspondence cache.
    pub fn multiresolution_snapshot(&self) -> Option<MultiResolutionSnapshot<'_>> {
        self.multiresolution.as_ref().map(|map| map.snapshot())
    }

    /// Snapshot source for the experimental single-resolution WebGPU search.
    /// `None` means the map uses an unsupported representation or neighborhood.
    pub fn single_resolution_snapshot(
        &self,
    ) -> Option<(f32, impl Iterator<Item = (i32, &VoxelDistribution)>)> {
        if cfg!(feature = "nn_kdt")
            || self.multiresolution.is_some()
            || !matches!(self.nearby_type, IVoxNearbyType::Nearby26)
        {
            return None;
        }
        Some((self.resolution, self.livodish.keyed_distributions()))
    }

    pub fn new(resolution: f32, capacity: usize, nbtype: IVoxNearbyType, sensitivity: f32) -> Self {
        Self {
            nearby_type: nbtype,
            inverse_resolution: 1.0 / resolution,
            resolution,
            scan_number: 0,
            #[cfg(feature = "nn_kdt")]
            kd_map: kiddo::float::kdtree::KdTree::<f32, u64, 3, 64, u32>::with_capacity(capacity),
            #[cfg(feature = "nn_kdt")]
            kd_points: std::collections::VecDeque::with_capacity(capacity),
            #[cfg(feature = "nn_kdt")]
            kd_points_lookup: std::collections::HashMap::new(),
            livodish: Livodish::new(capacity, sensitivity, resolution / 4.0),
            multiresolution: None,
            #[cfg(not(feature = "os-rng-seed"))]
            rng: rand_pcg::Pcg64Mcg::seed_from_u64(SAMPLING_SEED),
            #[cfg(feature = "os-rng-seed")]
            rng: rand_pcg::Pcg64Mcg::from_os_rng(),
            #[cfg(feature = "nn_kdt")]
            pt_id: 0,
            #[cfg(feature = "nn_kdt")]
            max_map_size: capacity,
        }
    }

    /// Enable the experimental CPU hierarchy. Call before inserting map points.
    pub fn enable_multiresolution(&mut self, capacity: usize, sensitivity: f32) {
        if cfg!(feature = "nn_kdt") {
            panic!("multiresolution is incompatible with nn_kdt");
        }
        assert!(
            self.is_empty(),
            "enable multiresolution before inserting points"
        );
        self.multiresolution = Some(multiresolution::MultiResolutionMap::new(
            self.resolution,
            capacity,
            sensitivity,
        ));
        // Release the unused fixed-map storage: capacity is a total cell budget.
        self.livodish = Livodish::new(0, sensitivity, self.resolution / 4.0);
    }

    pub fn is_empty(&self) -> bool {
        if let Some(map) = &self.multiresolution {
            return map.is_empty();
        }
        #[cfg(feature = "nn_kdt")]
        return self.kd_map.size() == 0;

        #[cfg(not(feature = "nn_kdt"))]
        return self.livodish.is_empty();
    }

    pub fn add(
        &mut self,
        points: &[pelorus_core::PointXYZIT],
        transform: &nalgebra::Isometry3<f64>,
        scan_id: usize,
    ) {
        #[cfg(feature = "rayon")]
        let execution_mode = ExecutionMode::Rayon;
        #[cfg(not(feature = "rayon"))]
        let execution_mode = ExecutionMode::Sequential;
        let _ = self.add_with_execution_mode(points, transform, scan_id, execution_mode);
    }

    pub fn add_with_execution_mode(
        &mut self,
        points: &[pelorus_core::PointXYZIT],
        transform: &nalgebra::Isometry3<f64>,
        scan_id: usize,
        execution_mode: ExecutionMode,
    ) -> Option<f64> {
        self.scan_number = scan_id;
        if let Some(map) = &mut self.multiresolution {
            map.add(points, transform, scan_id);
            return None;
        }

        #[cfg(feature = "nn_kdt")]
        {
            // The kd-tree map has no parallel insertion path.
            let _ = execution_mode;
            if self.scan_number == 1 {
                points.iter().for_each(|point| {
                    let point_global = transform
                        .transform_point(&nalgebra::Point3::new(point.x, point.y, point.z).cast());
                    if self
                        .kd_map
                        .nearest_n_within::<SquaredEuclidean>(
                            &[
                                point_global.x as f32,
                                point_global.y as f32,
                                point_global.z as f32,
                            ],
                            self.resolution * self.resolution,
                            std::num::NonZero::new(1).unwrap(),
                            false,
                        )
                        .is_empty()
                    {
                        self.pt_id += 1;
                        self.kd_map.add(
                            &[
                                point_global.x as f32,
                                point_global.y as f32,
                                point_global.z as f32,
                            ],
                            self.pt_id,
                        );
                        let p = std::sync::Arc::new((self.pt_id, point_global.cast()));
                        self.kd_points.push_front(p.clone());
                        self.kd_points_lookup.insert(self.pt_id, p.clone());
                    }
                });
            } else {
                // check if any point is closer than 25cm, if not, add to tree
                let f = points
                    .iter() // TODO as par iter
                    .filter(|point| {
                        let point_global = transform.transform_point(
                            &nalgebra::Point3::new(point.x, point.y, point.z).cast(),
                        );
                        self.kd_map
                            .nearest_n_within::<SquaredEuclidean>(
                                &[
                                    point_global.x as f32,
                                    point_global.y as f32,
                                    point_global.z as f32,
                                ],
                                self.resolution * self.resolution,
                                std::num::NonZero::new(1).unwrap(),
                                false,
                            )
                            .is_empty()
                    })
                    .collect::<Vec<_>>();

                let cur_len = self.kd_points.len();
                let new_len = cur_len + f.len();
                let target_len = self.max_map_size - f.len();
                if new_len > self.max_map_size {
                    while self.kd_points.len() >= target_len {
                        if let Some(rc) = self.kd_points.pop_back() {
                            self.kd_map.remove(&[rc.1.x, rc.1.y, rc.1.z], rc.0);
                            self.kd_points_lookup.remove(&rc.0);
                        }
                    }
                }

                f.into_iter().for_each(|point| {
                    let point_global = transform
                        .transform_point(&nalgebra::Point3::new(point.x, point.y, point.z).cast());
                    self.pt_id += 1;
                    self.kd_map.add(
                        &[
                            point_global.x as f32,
                            point_global.y as f32,
                            point_global.z as f32,
                        ],
                        self.pt_id,
                    );
                    let p = std::sync::Arc::new((self.pt_id, point_global.cast()));
                    self.kd_points.push_front(p.clone());
                    self.kd_points_lookup.insert(self.pt_id, p.clone());
                });
            }
        }

        #[cfg(not(feature = "nn_kdt"))]
        {
            let current_scan_number = self.scan_number;

            let mut keyed_points = points
                .iter()
                .map(|p| {
                    let transform_f32: nalgebra::Isometry3<f32> = transform.cast();
                    let global_p = transform_f32.transform_point(&Point3::new(p.x, p.y, p.z));
                    (global_p, hash(&global_p, self.resolution))
                })
                .collect::<Vec<_>>();

            let sort_started = pelorus_core::Instant::now();
            sort_unstable_by_key(&mut keyed_points, |(_, key)| *key, execution_mode);
            let mut adaptive_nanos = sort_started.elapsed().as_nanos() as f64;

            // TODO chunk by seems extremely fast on M3 Pro
            // #[cfg(feature = "parallel")]
            // {
            // if keyed_points.len() > PAR_SORT_BOUNDARY {

            // use rayon::{iter::ParallelIterator, slice::ParallelSlice};
            // keyed_points
            //     .par_chunk_by(|a, b| a.0 == b.0)
            //     .map(|group_slice| {
            //         let dist = VoxelDistribution::from_points(group_slice, current_scan_number);
            //         let grid = self.pos2grid(&PointXYZ {
            //             x: dist.mean.x,
            //             y: dist.mean.y,
            //             z: dist.mean.z,
            //         });
            //         let voxel_key = Self::hash_grid(&grid);
            //         (dist, voxel_key)
            //     })
            //     .collect::<Vec<_>>()
            //     .into_iter()
            //     .for_each(|(dist, voxel_key)| {
            //         self.livodish.add(dist, voxel_key, self.scan_number);
            //     });
            // } else {
            //     keyed_points
            //         .chunk_by(|a, b| a.0 == b.0)
            //         .map(|group_slice| find_medoid_in_slice(group_slice))
            //         .collect()
            // }
            // }
            // #[cfg(not(feature = "parallel"))]
            // {
            keyed_points
                .chunk_by(|a, b| a.1 == b.1)
                .for_each(|group_slice| {
                    let dist = VoxelDistribution::from_points(group_slice, current_scan_number);
                    let grid = self.pos2grid(&PointXYZ {
                        x: dist.mean.x,
                        y: dist.mean.y,
                        z: dist.mean.z,
                    });
                    let voxel_key = Self::hash_grid(&grid);
                    self.livodish
                        .add(dist, voxel_key, self.scan_number, execution_mode);
                });
            // }

            // Re-arranges the map to be ready for new inserts.
            let build_started = pelorus_core::Instant::now();
            self.livodish.build(self.scan_number, execution_mode);
            adaptive_nanos += build_started.elapsed().as_nanos() as f64;
            Some(adaptive_nanos / points.len().max(1) as f64)
        }

        #[cfg(feature = "nn_kdt")]
        None
    }

    #[inline(always)]
    pub fn sample_from_distribution<R: Rng + ?Sized>(
        distribution: VoxelDistribution,
        num_samples: usize,
        rng: &mut R,
        result: &mut [nalgebra::Point3<f32>],
    ) {
        let standard_normal = Normal::new(0.0, 1.0).unwrap();
        let mut samples_iter = standard_normal.sample_iter(rng);

        for res in result.iter_mut().take(num_samples) {
            let rand_vec = Vector3::new(
                samples_iter.next().unwrap(),
                samples_iter.next().unwrap(),
                samples_iter.next().unwrap(),
            );
            *res = distribution.mean + distribution.transform * rand_vec;
        }
    }

    #[inline(always)]
    #[must_use]
    pub fn k_nearest_neighbor_query(
        &self,
        query: &PointXYZ,
        k: usize,
        max_range: f32,
        result: &mut [PointXYZ],
        out_size: &mut usize,
    ) -> bool {
        let sq_max_range = max_range * max_range;

        if k == 0 || result.is_empty() {
            *out_size = 0;
            return false;
        }

        let mut rng = self.rng.clone();
        let query_p = nalgebra::Point3::new(query.x, query.y, query.z);

        const MAX_NEIGHBORS: usize = 27;
        let mut cached_neighbors: [Option<VoxelDistribution>; MAX_NEIGHBORS] =
            [None; MAX_NEIGHBORS];
        let mut num_cached = 0;
        let mut total_density = 0u64;

        let mut cache_and_sum_density = |grids: &[(i32, i32, i32)]| {
            for g in grids {
                let k = Self::hash_grid(g);

                if let Some(dist) = self.livodish.find(&k)
                    && dist.point_count > 0
                    && num_cached < MAX_NEIGHBORS
                {
                    cached_neighbors[num_cached] = Some(dist);
                    total_density += dist.point_count;
                    num_cached += 1;
                }
            }
        };

        let grid = self.pos2grid(query);
        match self.nearby_type {
            IVoxNearbyType::Center => cache_and_sum_density(&[grid]),
            IVoxNearbyType::Nearby6 => cache_and_sum_density(&Self::nearby_6(grid)),
            IVoxNearbyType::Nearby18 => cache_and_sum_density(&Self::nearby_18(grid)),
            IVoxNearbyType::Nearby26 => cache_and_sum_density(&Self::nearby_26(grid)),
        };

        if total_density == 0 {
            *out_size = 0;
            return false;
        }

        let mut current_found_count = 0;

        let mut sampled_out = [Point3::default(); MAX_NEIGHBORS];
        for cached in &cached_neighbors[..num_cached] {
            if current_found_count >= k {
                break;
            }

            if let Some(dist) = *cached {
                let relative_density = dist.point_count as f64 / total_density as f64;
                let mut sample_count = (relative_density * k as f64).round() as usize;

                let samples_needed = k - current_found_count;
                sample_count = sample_count.min(samples_needed);

                if sample_count == 0 {
                    continue;
                }

                Self::sample_from_distribution(dist, sample_count, &mut rng, &mut sampled_out);

                for i in 0..sample_count {
                    let p = unsafe { sampled_out.get_unchecked(i) };
                    if (query_p - p).norm_squared() < sq_max_range {
                        result[current_found_count] = PointXYZ::new(p.x, p.y, p.z);
                        current_found_count += 1;

                        if current_found_count >= k {
                            break;
                        }
                    }
                }
            }
        }

        *out_size = current_found_count;
        *out_size > 0
    }

    fn symmetric_eigen_3x3(a: &Matrix3<f32>) -> Vector3<f32> {
        let trace_a = a.trace();

        let c2 = (a[(0, 0)] * a[(1, 1)] + a[(0, 0)] * a[(2, 2)] + a[(1, 1)] * a[(2, 2)])
            - (a[(0, 1)] * a[(0, 1)] + a[(0, 2)] * a[(0, 2)] + a[(1, 2)] * a[(1, 2)]);

        let det_a = a.determinant();

        let p = -trace_a;
        let q = c2;
        let r = -det_a;

        let a_cub = (3.0 * q - p * p) / 3.0;
        let b_cub = (2.0 * p * p * p - 9.0 * p * q + 27.0 * r) / 27.0;

        let angle = if a_cub.abs() < 1e-6 {
            0.0
        } else {
            let acos_arg = (-b_cub / 2.0) / (-a_cub * a_cub * a_cub / 27.0).sqrt();
            // Clamp the argument to acos to handle floating point inaccuracies
            acos_arg.clamp(-1.0, 1.0).acos()
        };

        let two_sqrt_neg_a_div_3 = 2.0 * (-a_cub / 3.0).sqrt();
        let p_div_3 = p / 3.0;

        // Root 1
        let e1 = two_sqrt_neg_a_div_3 * (angle / 3.0).cos() - p_div_3;
        // Root 2
        let e2 =
            two_sqrt_neg_a_div_3 * ((angle + 2.0 * std::f32::consts::PI) / 3.0).cos() - p_div_3;
        // Root 3
        let e3 =
            two_sqrt_neg_a_div_3 * ((angle + 4.0 * std::f32::consts::PI) / 3.0).cos() - p_div_3;

        Vector3::new(e1, e2, e3)
    }

    pub fn plane_fit_from_distributions(&self, query: &PointXYZ) -> Option<[f32; 4]> {
        if let Some(map) = &self.multiresolution {
            let matched = map.query(&Point3::new(query.x, query.y, query.z))?;
            log::trace!("Multi-resolution correspondence at level {}", matched.level);
            return Some(matched.plane);
        }
        let mut merged_dist = VoxelDistribution::new();

        let grid = self.pos2grid(query);
        let neighbor_grids = match self.nearby_type {
            IVoxNearbyType::Center => [
                (grid.0, grid.1, grid.2),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
            ],
            IVoxNearbyType::Nearby6 => {
                let n6 = Self::nearby_6(grid);
                let mut res = [(0, 0, 0); 27];
                res[..7].copy_from_slice(&n6);
                res
            }
            IVoxNearbyType::Nearby18 => {
                let n18 = Self::nearby_18(grid);
                let mut res = [(0, 0, 0); 27];
                res[..19].copy_from_slice(&n18);
                res
            }
            IVoxNearbyType::Nearby26 => Self::nearby_26(grid),
        };

        let n_cells = match self.nearby_type {
            IVoxNearbyType::Center => 1,
            IVoxNearbyType::Nearby6 => 7,
            IVoxNearbyType::Nearby18 => 19,
            IVoxNearbyType::Nearby26 => 27,
        };

        for g in neighbor_grids[..n_cells].iter() {
            let k = Self::hash_grid(g);
            if let Some(dist) = self.livodish.find(&k) {
                merged_dist.pool(dist);
            }
        }

        if merged_dist.point_count < 3 {
            return None;
        }

        let eigenvalues = Self::symmetric_eigen_3x3(&merged_dist.covariance);
        let mut eigenvalues_vec = [eigenvalues.x, eigenvalues.y, eigenvalues.z];
        eigenvalues_vec.sort_unstable_by(|a, b| a.total_cmp(b));

        if eigenvalues_vec[0].is_nan() || eigenvalues_vec[2].is_nan() {
            return None;
        }

        let lambda_0 = eigenvalues_vec[0];
        let lambda_2 = eigenvalues_vec[2];

        if lambda_2 < 1e-9 {
            return None;
        }

        let sphericity = lambda_0 / lambda_2;

        if sphericity > 0.5 {
            return None;
        }

        let min_eigenvalue = lambda_0;

        let mut m = merged_dist.covariance;
        m[(0, 0)] -= min_eigenvalue;
        m[(1, 1)] -= min_eigenvalue;
        m[(2, 2)] -= min_eigenvalue;

        let row0 = m.row(0).transpose();
        let row1 = m.row(1).transpose();
        let row2 = m.row(2).transpose();

        let v1 = row0.cross(&row1);
        let v2 = row1.cross(&row2);
        let v3 = row2.cross(&row0);

        let mag1_sq = v1.norm_squared();
        let mag2_sq = v2.norm_squared();
        let mag3_sq = v3.norm_squared();

        let normal_vec = if mag1_sq > mag2_sq && mag1_sq > mag3_sq {
            v1
        } else if mag2_sq > mag3_sq {
            v2
        } else {
            v3
        };

        let normal = normal_vec.try_normalize(1e-12)?;

        let d = -normal.dot(&merged_dist.mean.coords);

        Some([normal.x, normal.y, normal.z, d])
    }

    #[inline(always)]
    fn nearby_6(grid_key: (i32, i32, i32)) -> [(i32, i32, i32); 7] {
        let (gx, gy, gz) = grid_key;
        [
            (gx, gy, gz),
            (gx - 1, gy, gz),
            (gx + 1, gy, gz),
            (gx, gy + 1, gz),
            (gx, gy - 1, gz),
            (gx, gy, gz - 1),
            (gx, gy, gz + 1),
        ]
    }

    #[inline(always)]
    fn nearby_18(grid_key: (i32, i32, i32)) -> [(i32, i32, i32); 19] {
        let (gx, gy, gz) = grid_key;
        [
            (gx, gy, gz),
            (gx - 1, gy, gz),
            (gx + 1, gy, gz),
            (gx, gy + 1, gz),
            (gx, gy - 1, gz),
            (gx, gy, gz - 1),
            (gx, gy, gz + 1),
            (gx + 1, gy + 1, gz),
            (gx - 1, gy + 1, gz),
            (gx + 1, gy - 1, gz),
            (gx - 1, gy - 1, gz),
            (gx + 1, gy, gz + 1),
            (gx - 1, gy, gz + 1),
            (gx + 1, gy, gz - 1),
            (gx - 1, gy, gz - 1),
            (gx, gy + 1, gz + 1),
            (gx, gy - 1, gz + 1),
            (gx, gy + 1, gz - 1),
            (gx, gy - 1, gz - 1),
        ]
    }

    #[inline(always)]
    fn nearby_26(grid_key: (i32, i32, i32)) -> [(i32, i32, i32); 27] {
        let (gx, gy, gz) = grid_key;
        [
            (gx, gy, gz),
            (gx - 1, gy, gz),
            (gx + 1, gy, gz),
            (gx, gy + 1, gz),
            (gx, gy - 1, gz),
            (gx, gy, gz - 1),
            (gx, gy, gz + 1),
            (gx + 1, gy + 1, gz),
            (gx - 1, gy + 1, gz),
            (gx + 1, gy - 1, gz),
            (gx - 1, gy - 1, gz),
            (gx + 1, gy, gz + 1),
            (gx - 1, gy, gz + 1),
            (gx + 1, gy, gz - 1),
            (gx - 1, gy, gz - 1),
            (gx, gy + 1, gz + 1),
            (gx, gy - 1, gz + 1),
            (gx, gy + 1, gz - 1),
            (gx, gy - 1, gz - 1),
            (gx + 1, gy + 1, gz + 1),
            (gx - 1, gy + 1, gz + 1),
            (gx + 1, gy - 1, gz + 1),
            (gx + 1, gy + 1, gz - 1),
            (gx - 1, gy - 1, gz + 1),
            (gx - 1, gy + 1, gz - 1),
            (gx + 1, gy - 1, gz - 1),
            (gx - 1, gy - 1, gz - 1),
        ]
    }

    #[inline(always)]
    pub fn pos2grid(&self, pos: &PointXYZ) -> (i32, i32, i32) {
        let gx = (pos.x * self.inverse_resolution).floor() as i32;
        let gy = (pos.y * self.inverse_resolution).floor() as i32;
        let gz = (pos.z * self.inverse_resolution).floor() as i32;
        (gx, gy, gz)
    }

    #[inline(always)]
    pub fn hash_grid(grid_key: &(i32, i32, i32)) -> i32 {
        const PRIME_1: u32 = 73_856_093;
        const PRIME_2: u32 = 471_943;
        const PRIME_3: u32 = 83_492_791;

        let h = (grid_key.0 as u32)
            .wrapping_mul(PRIME_1)
            .wrapping_add((grid_key.1 as u32).wrapping_mul(PRIME_2))
            .wrapping_add((grid_key.2 as u32).wrapping_mul(PRIME_3));

        h as i32 // Reinterpret bit pattern (matches GPU)
    }

    #[inline(always)]
    pub fn hash(point: &nalgebra::Point3<f32>, voxel_size: f32) -> i32 {
        const PRIME_1: u32 = 73_856_093;
        const PRIME_2: u32 = 471_943;
        const PRIME_3: u32 = 83_492_791;

        let xyz = point * (1.0 / voxel_size);

        let x = xyz.x.floor() as i32;
        let y = xyz.y.floor() as i32;
        let z = xyz.z.floor() as i32;

        let h = (x as u32)
            .wrapping_mul(PRIME_1)
            .wrapping_add((y as u32).wrapping_mul(PRIME_2))
            .wrapping_add((z as u32).wrapping_mul(PRIME_3));

        h as i32
    }

    pub fn clear(&mut self) {
        if let Some(map) = &mut self.multiresolution {
            map.clear();
            return;
        }
        self.livodish.clear();
    }

    /// Mostly for testing purposes, is not efficient
    #[must_use]
    pub fn get_entire_map(&self) -> Vec<nalgebra::Point3<f32>> {
        if let Some(map) = &self.multiresolution {
            return map
                .distributions_with_levels()
                .into_iter()
                .map(|(d, _)| d.mean)
                .collect();
        }
        #[cfg(feature = "nn_kdt")]
        {
            self.kd_points.iter().map(|rc| rc.1).collect()
        }

        #[cfg(not(feature = "nn_kdt"))]
        {
            self.livodish.get_map()
        }
    }

    #[cfg(not(feature = "nn_kdt"))]
    #[must_use]
    pub fn get_distributions(&self) -> Vec<VoxelDistribution> {
        if let Some(map) = &self.multiresolution {
            return map.distributions();
        }
        self.livodish.get_distributions()
    }

    #[cfg(not(feature = "nn_kdt"))]
    #[must_use]
    pub fn get_distributions_with_lru_scores(&self) -> Vec<localmap::VoxelDistributionLru> {
        if let Some(map) = &self.multiresolution {
            return map
                .distributions()
                .into_iter()
                .map(|distribution| localmap::VoxelDistributionLru {
                    lru_score: 1.0
                        / (1.0
                            + self
                                .scan_number
                                .saturating_sub(distribution.last_updated_at_scan)
                                as f32),
                    distribution,
                })
                .collect();
        }
        self.livodish.get_distributions_with_lru_scores()
    }

    /// All distributions tagged with the multiresolution level they live at
    /// (0 = finest, 3 = coarsest) and a relevance score in `[0, 1]`. The
    /// score ranks each cell by recency *within its level*
    /// (see [`rank_scores_by_recency`]) rather than a raw age-based decay:
    /// a LiDAR scan does not revisit every voxel every tick, so a raw
    /// `1/(1+age)` decay would make almost the entire map read as
    /// "irrelevant" within one or two scans. Ranking spreads scores evenly
    /// across `[0, 1]` regardless of scan cadence, matching how
    /// [`Self::get_distributions_with_lru_scores`]'s rank-based score
    /// behaves for the flat map. Unlike [`Self::get_distributions`], this
    /// does not collapse the pyramid to a single layer. The flat (non
    /// multiresolution) map reports level 0 for everything.
    #[cfg(not(feature = "nn_kdt"))]
    #[must_use]
    pub fn get_distributions_with_levels(&self) -> Vec<(localmap::VoxelDistributionLru, u8)> {
        if let Some(map) = &self.multiresolution {
            let per_level = map.distributions_with_levels();
            let mut result = Vec::with_capacity(per_level.len());
            for level in 0..4u8 {
                let group: Vec<VoxelDistribution> = per_level
                    .iter()
                    .filter(|(_, l)| *l == level)
                    .map(|(d, _)| *d)
                    .collect();
                let ages: Vec<u64> = group
                    .iter()
                    .map(|d| self.scan_number.saturating_sub(d.last_updated_at_scan) as u64)
                    .collect();
                let scores = rank_scores_by_recency(&ages);
                result.extend(group.into_iter().zip(scores).map(|(distribution, score)| {
                    (
                        localmap::VoxelDistributionLru {
                            distribution,
                            lru_score: score,
                        },
                        level,
                    )
                }));
            }
            return result;
        }
        self.livodish
            .get_distributions_with_lru_scores()
            .into_iter()
            .map(|d| (d, 0u8))
            .collect()
    }
}

/// Ranks `ages` (smaller = more recent) into relevance scores in `[0, 1]`:
/// the most recent entry scores 1.0, the oldest scores toward 0.0, evenly
/// spread across the population regardless of the ages' absolute scale.
/// Mirrors `Livodish::get_distributions_with_lru_scores`'s `rank/capacity`
/// score, which is why the flat map's relevance coloring doesn't suffer the
/// same "decays to irrelevant within one scan" problem a raw age ratio would.
#[must_use]
pub fn rank_scores_by_recency(ages: &[u64]) -> Vec<f32> {
    let n = ages.len();
    if n <= 1 {
        return vec![1.0; n];
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| ages[i]);
    let mut scores = vec![0.0f32; n];
    let denom = (n - 1) as f32;
    for (rank, idx) in order.into_iter().enumerate() {
        scores[idx] = 1.0 - (rank as f32 / denom);
    }
    scores
}

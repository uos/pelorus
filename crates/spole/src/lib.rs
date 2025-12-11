#[cfg(feature = "nn_kdt")]
use kiddo::SquaredEuclidean;
use localmap::Livodish;

pub mod localmap;
use rand_distr::{Distribution, Normal};

use nalgebra::{Point3, Vector3};
use rand::{Rng, SeedableRng};

use nalgebra::{Cholesky, Matrix3};
#[cfg(all(feature = "parallel", not(feature = "nn_kdt")))]
use rayon::slice::ParallelSliceMut;
use ros_pointcloud2::points::{PointXYZ, PointXYZI};

use serde::{Deserialize, Serialize};

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

        // If there are no points, return an empty distribution.
        if point_count == 0 {
            return Self::new();
        }

        // --- 1. Calculate the mean of the points ---
        // Sum all point vectors and divide by the count to get the average.
        let sum_of_points: Vector3<f32> = points.iter().map(|p| p.0.coords).sum();
        let mean = Point3::from(sum_of_points / point_count as f32);

        // --- 2. Calculate the covariance matrix ---
        // Covariance(X, Y) = E[(X - E[X]) * (Y - E[Y])^T]
        // We calculate the sum of the outer products of the deviation from the mean.
        let mut covariance = Matrix3::<f32>::zeros();
        for (point, _) in points {
            // Calculate the difference vector from the mean.
            let diff = point - mean;
            // Add the outer product of the difference vector to the sum.
            covariance += diff * diff.transpose();
        }

        // Finalize the covariance by dividing by the number of points.
        covariance /= point_count as f32;

        // --- 3. Construct and return the VoxelDistribution ---
        VoxelDistribution {
            point_count: point_count as u64,
            mean,
            covariance,
            transform: Matrix3::identity(),
            last_updated_at_scan: current_scan_number,
        }
    }

    // no cross-covariance
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
        // min_dist is not used here, as the weighting from the Mahalanobis
        // distance naturally handles far-away distributions by assigning them a
        // weight near zero, achieving the goal of "not updating too much".
        _min_dist: f32,
        current_scan_number: usize,
    ) {
        // If the other distribution is empty, there's nothing to update from.
        if other.point_count == 0 {
            return;
        }

        self.last_updated_at_scan = current_scan_number;

        // If this distribution is empty, it simply becomes the other one.
        if self.point_count == 0 {
            self.point_count = other.point_count;
            self.mean = other.mean;
            self.covariance = other.covariance;
            self.transform = other.transform.clone();
            return;
        }

        // --- 1. Calculate Mahalanobis Distance for Weighting ---

        let mean_diff = self.mean - other.mean;

        // Regularize the covariance matrix to prevent numerical instability and ensure it's invertible.
        let regularization_factor = 1e-6;
        let regularized_cov = self.covariance + Matrix3::identity() * regularization_factor;

        // Calculate the squared Mahalanobis distance.
        // D^2 = (x - μ)^T * Σ^-1 * (x - μ)
        let mahalanobis_sq = if let Some(inv_cov) = regularized_cov.try_inverse() {
            // The result of the matrix multiplication is a 1x1 matrix.
            // We extract the scalar value from it using [0].
            (mean_diff.transpose() * inv_cov * mean_diff)[0]
        } else {
            // Fallback to squared Euclidean distance if covariance is singular.
            mean_diff.norm_squared()
        };

        // --- 2. Calculate Update Weight (alpha) ---

        // As the distribution matures (gains points), it becomes less sensitive to new updates.
        // This prevents a well-established voxel from being drastically changed by a single new measurement.
        let maturity_decay = (self.point_count as f32).ln().max(1.0);
        let effective_sensitivity = sensitivity / maturity_decay;

        // The weight `alpha` is calculated using a Gaussian function.
        // A large distance results in a weight close to 0.
        let alpha = (-mahalanobis_sq / (2.0 * effective_sensitivity)).exp();

        // If alpha is negligible, skip the update to save computation.
        if alpha < 1e-5 {
            return;
        }

        // --- 3. Fuse the Mean and Covariance ---

        // The new mean is a weighted average of the two original means.
        let new_mean = Point3::from((1.0 - alpha) * self.mean.coords + alpha * other.mean.coords);

        // The new covariance is calculated using the parallel axis theorem, weighted by alpha.
        // This correctly combines the internal variances of both distributions plus the
        // variance between their means.
        let outer_product_mean_diff = mean_diff * mean_diff.transpose();
        let new_covariance = (1.0 - alpha) * self.covariance
            + alpha * other.covariance
            + alpha * (1.0 - alpha) * outer_product_mean_diff;

        // --- 4. Apply the Updates ---
        self.mean = new_mean;
        self.covariance = new_covariance;
        self.point_count += other.point_count; // Aggregate point counts

        // Optional: Update the transform based on the new covariance if needed.
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

        // --- Initial Bootstrap (Welford's Algorithm) ---
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

        // --- Weighted Update with Maturity ---

        // As point_count grows, the log term increases, making the distribution
        // less sensitive (more "skeptical") to new points.
        // We use `(point_count - 1)` because the EMA regime starts after 2 points.
        // `max(1.0)` ensures we don't increase sensitivity for low point counts.
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

        // The weight `alpha` is now calculated using the decayed `effective_sensitivity`.
        let alpha = (-distance_sq / (2.0 * effective_sensitivity)).exp();

        let new_mean = ((1.0 - alpha) * self.mean.coords + alpha * new_point.coords).into();

        let new_cov_sample = (1.0 - alpha) * &diff_from_old_mean;
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
    // The iterator yields items of type (i32, i32, i32), representing voxel coordinates.
    type Item = (i32, i32, i32);

    /// Advances the iterator and returns the next voxel coordinate.
    fn next(&mut self) -> Option<Self::Item> {
        // If we have iterated through all the voxels in the block, return None.
        if self.current_index >= self.max_index {
            return None;
        }

        // Deconstruct the 1D `current_index` into 3D offsets (dx, dy, dz)
        // relative to the origin of the encompassing voxel block.
        let i = self.current_index;
        let f = self.factor;

        let dz = i / (f * f);
        let remainder = i % (f * f);
        let dy = remainder / f;
        let dx = remainder % f;

        // Get the origin coordinate.
        let (ox, oy, oz) = self.origin;

        // Calculate the coordinate of the next voxel by adding the offset to the origin.
        let next_voxel = (ox + dx, oy + dy, oz + dz);

        // Increment the index for the next iteration.
        self.current_index += 1;

        // Return the calculated voxel coordinate.
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

    // To find the origin of the encompassing block, we use Euclidean division.
    // This correctly handles negative coordinates by rounding towards negative infinity,
    // ensuring we always get the minimum corner of the block.
    // For example, with factor 2, a key of (-1, -1, -1) belongs to the block
    // whose origin is (-2, -2, -2).
    // (-1).div_euclid(2) * 2 == -1 * 2 == -2.
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
    const PRIME_1: i32 = 73_856_093;
    const PRIME_2: i32 = 19_349_663;
    const PRIME_3: i32 = 83_492_791;
    const PRIME_4: i32 = 116_881;

    let xyz = point * (1.0 / voxel_size);

    let x = xyz.x.round() as i32;
    let y = xyz.y.round() as i32;
    let z = xyz.z.round() as i32;

    let mut hash = x.wrapping_mul(PRIME_1);
    hash = hash.wrapping_add(y.wrapping_mul(PRIME_2));
    hash = hash.wrapping_add(z.wrapping_mul(PRIME_3));
    hash = hash.wrapping_rem(PRIME_4);

    hash
}

impl IVoxHashMap {
    pub fn new(resolution: f32, capacity: usize, nbtype: IVoxNearbyType, sensitivity: f32) -> Self {
        Self {
            nearby_type: nbtype.clone(),
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
            rng: rand_pcg::Pcg64Mcg::from_os_rng(),
            #[cfg(feature = "nn_kdt")]
            pt_id: 0,
            #[cfg(feature = "nn_kdt")]
            max_map_size: capacity,
        }
    }

    pub fn is_empty(&self) -> bool {
        #[cfg(feature = "nn_kdt")]
        return self.kd_map.size() == 0;

        #[cfg(not(feature = "nn_kdt"))]
        return self.livodish.is_empty();
    }

    pub fn add(&mut self, points: &[PointXYZI], transform: &nalgebra::Isometry3<f64>) {
        self.scan_number += 1;

        #[cfg(feature = "nn_kdt")]
        {
            if self.scan_number == 1 {
                points.into_iter().for_each(|point| {
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
                            (self.resolution * self.resolution) as f32,
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
                    .into_iter() // TODO as par iter
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
                                (self.resolution * self.resolution) as f32,
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
                    let global_p = transform
                        .transform_point(&nalgebra::Point3::new(p.x, p.y, p.z).cast())
                        .cast();
                    (global_p, hash(&global_p, self.resolution))
                })
                .collect::<Vec<_>>();

            // downsample first, then update per voxel using distributions
            // After this, all points in the same voxel are adjacent
            #[cfg(feature = "parallel")]
            {
                const PAR_SORT_BOUNDARY: usize = 1; // When tested on an M3, par sort was always faster, no matter the size. TODO test on other platforms
                if keyed_points.len() > PAR_SORT_BOUNDARY {
                    keyed_points.par_sort_unstable_by_key(|(_, key)| *key);
                } else {
                    keyed_points.sort_unstable_by_key(|(_, key)| *key);
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                keyed_points.sort_unstable_by_key(|(_, key)| *key);
            }

            // TODO chunk by seems extremely fast on M3 Pro
            // #[cfg(feature = "parallel")]
            // {
            //     if keyed_points.len() > PAR_SORT_BOUNDARY {
            //         keyed_points
            //             .par_chunk_by(|a, b| a.0 == b.0)
            //             .map(|group_slice| find_medoid_in_slice(group_slice))
            //             .collect()
            //     } else {
            //         keyed_points
            //             .chunk_by(|a, b| a.0 == b.0)
            //             .map(|group_slice| find_medoid_in_slice(group_slice))
            //             .collect()
            //     }
            // }
            // #[cfg(not(feature = "parallel"))]
            // {
            keyed_points
                .chunk_by(|a, b| a.0 == b.0)
                .for_each(|group_slice| {
                    let dist = VoxelDistribution::from_points(group_slice, current_scan_number);
                    let grid = self.pos2grid(&PointXYZ {
                        x: dist.mean.x as f32,
                        y: dist.mean.y as f32,
                        z: dist.mean.z as f32,
                    });
                    let voxel_key = Self::hash_grid(&grid);
                    self.livodish.add(dist, voxel_key, self.scan_number);
                });
            // }
            self.livodish.build(self.scan_number);
        }
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

        // --- Gather Neighbors and Total Density ---
        const MAX_NEIGHBORS: usize = 27;
        let mut cached_neighbors: [Option<VoxelDistribution>; MAX_NEIGHBORS] =
            [None; MAX_NEIGHBORS];
        let mut num_cached = 0;
        let mut total_density = 0u64;

        let mut cache_and_sum_density = |grids: &[(i32, i32, i32)]| {
            for g in grids {
                let k = Self::hash_grid(g);
                if let Some(dist) = self.livodish.find(&k) {
                    if dist.point_count > 0 && num_cached < MAX_NEIGHBORS {
                        cached_neighbors[num_cached] = Some(dist);
                        total_density += dist.point_count;
                        num_cached += 1;
                    }
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

        // --- Proportional Sampling ---

        let mut current_found_count = 0;

        let mut sampled_out = [Point3::default(); MAX_NEIGHBORS];
        for i in 0..num_cached {
            if current_found_count >= k {
                break;
            }

            if let Some(dist) = cached_neighbors[i] {
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
        // The characteristic equation is λ³ - tr(A)λ² + c₂λ - det(A) = 0
        // where c₂ = 0.5 * (tr(A)² - tr(A²))
        let trace_a = a.trace();

        let c2 = (a[(0, 0)] * a[(1, 1)] + a[(0, 0)] * a[(2, 2)] + a[(1, 1)] * a[(2, 2)])
            - (a[(0, 1)] * a[(0, 1)] + a[(0, 2)] * a[(0, 2)] + a[(1, 2)] * a[(1, 2)]);

        let det_a = a.determinant();

        // Solve the cubic equation: x³ + p*x² + q*x + r = 0
        let p = -trace_a;
        let q = c2;
        let r = -det_a;

        // Convert to depressed cubic: t³ + a*t + b = 0
        let a_cub = (3.0 * q - p * p) / 3.0;
        let b_cub = (2.0 * p * p * p - 9.0 * p * q + 27.0 * r) / 27.0;

        // For symmetric matrices, eigenvalues are real, so the discriminant is <= 0.
        // We use the trigonometric solution for three real roots.
        let angle = if a_cub.abs() < 1e-6 {
            // Handle case where a is zero
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

    pub fn plane_fit_from_distributions(
        &self,
        query: &PointXYZ,
        // You might want to add an error threshold for the eigenvalue check
    ) -> Option<[f32; 4]> {
        let mut merged_dist = VoxelDistribution::new();

        // 1. Find and merge all neighboring distributions
        let grid = self.pos2grid(query);
        let neighbor_grids = Self::nearby_26(grid); // Or any other nearby type

        for g in neighbor_grids.iter() {
            let k = Self::hash_grid(g);
            if let Some(dist) = self.livodish.find(&k) {
                merged_dist.pool(dist);
            }
        }

        // 2. Ensure we have enough data to define a plane
        if merged_dist.point_count < 3 {
            return None;
        }

        let eigenvalues = Self::symmetric_eigen_3x3(&merged_dist.covariance);
        let mut eigenvalues_vec = [eigenvalues.x, eigenvalues.y, eigenvalues.z];
        eigenvalues_vec.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let lambda_0 = eigenvalues_vec[0]; // Smallest
        let lambda_2 = eigenvalues_vec[2]; // Largest

        // Ensure we don't divide by zero if the distribution is empty or a single point
        if lambda_2 < 1e-9 {
            return None;
        }

        let sphericity = lambda_0 / lambda_2;

        // The threshold now needs to be much higher. Start with something around 0.5
        // and adjust it based on your data. A lower threshold means you require
        // a "flatter" distribution to consider it a plane.
        if sphericity > 0.5 {
            return None;
        }

        let min_eigenvalue = lambda_0; // Use the sorted smallest eigenvalue

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

        // 4. Calculate the 'd' parameter of the plane equation using the merged mean
        let d = -normal.dot(&merged_dist.mean.coords);

        Some([normal.x, normal.y, normal.z, d])
    }

    pub fn plane_fit_from_distributions_lower_res(
        &self,
        query: &PointXYZ,
        // You might want to add an error threshold for the eigenvalue check
        res_factor: i32,
    ) -> Option<[f32; 4]> {
        let mut merged_dist = VoxelDistribution::new();

        // 1. Find and merge all neighboring distributions
        let grid = self.pos2grid(query);

        let lower_res_voxel = |grid: &(i32, i32, i32)| -> Option<VoxelDistribution> {
            let mut voxel: Option<VoxelDistribution> = None;

            for g in encompassing_voxel_iter(*grid, res_factor) {
                let k = Self::hash_grid(&g);
                if let Some(dist) = self.livodish.find(&k) {
                    // i += 1;
                    if let Some(vup) = voxel.as_mut() {
                        vup.pool(dist);
                    } else {
                        voxel = Some(dist);
                    }
                }
            }

            voxel
        };

        let (gx, gy, gz) = grid;
        let lr_grid = (
            gx.div_euclid(res_factor),
            gy.div_euclid(res_factor),
            gz.div_euclid(res_factor),
        );
        let neighbor_grids = Self::nearby_26(lr_grid); // Or any other nearby type

        for g in neighbor_grids.iter() {
            if let Some(dist) = lower_res_voxel(&g) {
                merged_dist.pool(dist);
            }
        }

        // 2. Ensure we have enough data to define a plane
        if merged_dist.point_count < 3 {
            return None;
        }

        // 3. Calculate the plane normal from the merged covariance matrix.
        // This logic is borrowed directly from your `plane_fit_stable` function.
        let eigenvalues = Self::symmetric_eigen_3x3(&merged_dist.covariance);
        let mut eigenvalues_vec = [eigenvalues.x, eigenvalues.y, eigenvalues.z];
        eigenvalues_vec.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let lambda_0 = eigenvalues_vec[0]; // Smallest
        let lambda_2 = eigenvalues_vec[2]; // Largest

        // Ensure we don't divide by zero if the distribution is empty or a single point
        if lambda_2 < 1e-9 {
            return None;
        }

        let sphericity = lambda_0 / lambda_2;

        // The threshold now needs to be much higher. Start with something around 0.5
        // and adjust it based on your data. A lower threshold means you require
        // a "flatter" distribution to consider it a plane.
        if sphericity > 0.5 {
            return None;
        }

        let min_eigenvalue = lambda_0; // Use the sorted smallest eigenvalue

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

        // 4. Calculate the 'd' parameter of the plane equation using the merged mean
        let d = -normal.dot(&merged_dist.mean.coords);

        Some([normal.x, normal.y, normal.z, d])
    }
    #[inline(always)]
    #[must_use]
    pub fn k_nearest_neighbor_query_lower_res(
        &self,
        query: &PointXYZ,
        k: usize,
        result: &mut [PointXYZ],
        out_size: &mut usize,
        res_factor: i32, // 2 means half the resolution
    ) -> bool {
        if k == 0 || result.is_empty() {
            *out_size = 0;
            return false;
        }

        let mut rng = self.rng.clone();

        // --- Gather Neighbors and Total Density ---
        const MAX_NEIGHBORS: usize = 27;
        let mut cached_neighbors: [Option<VoxelDistribution>; MAX_NEIGHBORS] =
            [None; MAX_NEIGHBORS];
        let mut num_cached = 0;
        let mut total_density = 0u64;

        let lower_res_voxel = |grid: &(i32, i32, i32)| -> Option<VoxelDistribution> {
            let mut voxel: Option<VoxelDistribution> = None;

            for g in encompassing_voxel_iter(*grid, res_factor) {
                let k = Self::hash_grid(&g);
                if let Some(dist) = self.livodish.find(&k) {
                    // i += 1;
                    if let Some(vup) = voxel.as_mut() {
                        vup.pool(dist);
                    } else {
                        voxel = Some(dist);
                    }
                }
            }

            voxel
        };

        let mut cache_and_sum_density = |grids: &[(i32, i32, i32)]| {
            for g in grids {
                if let Some(dist) = lower_res_voxel(g) {
                    if dist.point_count > 0 && num_cached < MAX_NEIGHBORS {
                        cached_neighbors[num_cached] = Some(dist);
                        total_density += dist.point_count;
                        num_cached += 1;
                    }
                }
            }
        };

        let hr_grid = self.pos2grid(query);
        let (gx, gy, gz) = hr_grid;
        let lr_grid = (
            gx.div_euclid(res_factor),
            gy.div_euclid(res_factor),
            gz.div_euclid(res_factor),
        );
        match self.nearby_type {
            IVoxNearbyType::Center => cache_and_sum_density(&[lr_grid]),
            IVoxNearbyType::Nearby6 => cache_and_sum_density(&Self::nearby_6(lr_grid)),
            IVoxNearbyType::Nearby18 => cache_and_sum_density(&Self::nearby_18(lr_grid)),
            IVoxNearbyType::Nearby26 => cache_and_sum_density(&Self::nearby_26(lr_grid)),
        };

        if total_density == 0 {
            *out_size = 0;
            return false;
        }

        // --- Proportional Sampling ---

        let mut current_found_count = 0;

        let mut sampled_out = [Point3::default(); MAX_NEIGHBORS];
        for i in 0..num_cached {
            if current_found_count >= k {
                break;
            }

            if let Some(dist) = cached_neighbors[i] {
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
                    result[current_found_count] = PointXYZ::new(p.x, p.y, p.z);
                    current_found_count += 1;

                    if current_found_count >= k {
                        break;
                    }
                }
            }
        }

        *out_size = current_found_count;
        *out_size > 0
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
        let gx = (pos.x * self.inverse_resolution).round() as i32;
        let gy = (pos.y * self.inverse_resolution).round() as i32;
        let gz = (pos.z * self.inverse_resolution).round() as i32;
        (gx, gy, gz)
    }

    #[inline(always)]
    pub fn hash_grid(grid_key: &(i32, i32, i32)) -> i32 {
        const PRIME_1: i32 = 73856093;
        const PRIME_2: i32 = 19349663;
        const PRIME_3: i32 = 83492791;

        let mut hash = grid_key.0.wrapping_mul(PRIME_1);
        hash = hash.wrapping_add(grid_key.1.wrapping_mul(PRIME_2));
        hash = hash.wrapping_add(grid_key.2.wrapping_mul(PRIME_3));

        hash
    }

    #[inline(always)]
    pub fn hash(point: &nalgebra::Point3<f32>, voxel_size: f32) -> i32 {
        const PRIME_1: i32 = 73_856_093;
        const PRIME_2: i32 = 19_349_663;
        const PRIME_3: i32 = 83_492_791;

        let xyz = point * (1.0 / voxel_size);

        let x = xyz.x.round() as i32;
        let y = xyz.y.round() as i32;
        let z = xyz.z.round() as i32;

        let mut hash = x.wrapping_mul(PRIME_1);
        hash = hash.wrapping_add(y.wrapping_mul(PRIME_2));
        hash = hash.wrapping_add(z.wrapping_mul(PRIME_3));

        hash
    }

    pub fn clear(&mut self) {
        self.livodish.clear();
    }

    /// Mostly for testing purposes, is not efficient
    #[must_use]
    pub fn get_entire_map(&self) -> Vec<nalgebra::Point3<f32>> {
        #[cfg(feature = "nn_kdt")]
        {
            return self.kd_points.iter().map(|rc| rc.1).collect();
        }

        #[cfg(not(feature = "nn_kdt"))]
        {
            self.livodish.get_map()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Isometry3;

    #[test]
    #[cfg(not(feature = "nn_kdt"))]
    fn basic_insert() {
        use nalgebra::Isometry3;

        let h_points = vec![
            PointXYZI::new(0.2, 0.2, 0.0, 0.0),
            PointXYZI::new(0.4, 0.4, 0.1, 0.0),
            PointXYZI::new(0.49, 0.49, 0.1, 0.0),
            PointXYZI::new(0.75001, 0.51, 0.51, 0.0),
        ];
        let mut map = IVoxHashMap::new(0.5, 1_000, IVoxNearbyType::default(), 0.3);
        map.add(&h_points, &Isometry3::identity());
        let entire_map = map.get_entire_map();
        assert_eq!(entire_map.len(), 3); // 3 because of rounding. the voxel center is the origin of the coordinate system

        {
            let mut found = false;
            for point in entire_map {
                if point.x == 0.4 && point.y == 0.4 && point.z == 0.1 {
                    found = true;
                    break;
                }
            }
            assert!(found);
        }
    }

    #[test]
    #[cfg(not(feature = "nn_kdt"))]
    fn basic_add_and_search() {
        let mut map = IVoxHashMap::new(0.5, 1_000, IVoxNearbyType::default(), 0.3);
        map.add(
            &[PointXYZI::new(3.3, 3.3, 3.0, 0.0)],
            &Isometry3::identity(),
        );
        let mut result = vec![PointXYZ::default(); 1];
        let out_size = &mut 0;
        let res = map.k_nearest_neighbor_query(
            &PointXYZ::new(3.2, 3.2, 3.0),
            1,
            5.0,
            &mut result,
            out_size,
        );
        assert!(res == true);
        assert_eq!(result.len(), 1);
    }

    #[test]
    #[cfg(not(feature = "nn_kdt"))]
    fn basic_origin() {
        let h_points = vec![
            PointXYZI::new(0.2, 0.2, 0.0, 0.0),
            PointXYZI::new(0.4, 0.4, 0.1, 0.0),
            PointXYZI::new(0.49, 0.49, 0.1, 0.0),
            PointXYZI::new(0.75001, 0.51, 0.51, 0.0),
        ];
        let mut map = IVoxHashMap::new(0.5, 1_000, IVoxNearbyType::Nearby26, 0.3);
        map.add(&h_points, &Isometry3::identity());

        let query = PointXYZ::new(0.0, 0.0, 0.0);
        let mut result = vec![PointXYZ::default(); 6];
        let out_size = &mut 0;
        let res = map.k_nearest_neighbor_query(&query, 6, 5.0, &mut result, out_size);

        assert!(res);
        assert_eq!(*out_size, 6); // fills with density sampling
    }

    #[test]
    #[cfg(not(feature = "nn_kdt"))]
    fn more_distant() {
        let distant_points = vec![
            PointXYZI::new(0.2, 0.2, 0.0, 0.0),
            PointXYZI::new(5.3, 10.7, 0.1, 0.0),
            PointXYZI::new(5.6, 10.4, 0.3, 0.0),
            PointXYZI::new(5.9, 10.1, 0.5, 0.0),
            PointXYZI::new(6.2, 9.8, 0.7, 0.0),
            PointXYZI::new(6.5, 9.5, 0.9, 0.0),
            PointXYZI::new(6.8, 9.2, 1.1, 0.0),
        ];

        let mut map = IVoxHashMap::new(0.5, 1_000, IVoxNearbyType::Nearby18, 0.3);
        map.add(&distant_points, &Isometry3::identity());

        let query = PointXYZ::new(5.5, 10.5, 0.0);
        let mut result = vec![PointXYZ::default(); 2];
        let out_size = &mut 0;

        let res = map.k_nearest_neighbor_query(&query, 2, 5.0, &mut result, out_size);

        assert!(res);
        assert_eq!(*out_size, 2);
    }
}

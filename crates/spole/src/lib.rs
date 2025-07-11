#[cfg(feature = "nn_kdt")]
use kiddo::SquaredEuclidean;
use localmap::Livodish;

pub mod localmap;
use rand_distr::{Distribution, Normal};

use nalgebra::{Point3, Vector3};
use rand::{Rng, SeedableRng};

use nalgebra::{Cholesky, Matrix3};
use ros_pointcloud2::points::PointXYZ;

use serde::{Deserialize, Serialize};

pub struct IVoxHashMap {
    nearby_type: IVoxNearbyType,
    inverse_resolution: f32,
    scan_number: usize,
    #[cfg(feature = "nn_kdt")]
    kd_map: kiddo::float::kdtree::KdTree<f32, u64, 3, 64, u32>,
    #[cfg(feature = "nn_kdt")]
    kd_points: VecDeque<Arc<(u64, nalgebra::Point3<f32>)>>,
    #[cfg(feature = "nn_kdt")]
    kd_points_lookup: HashMap<u64, Arc<(u64, nalgebra::Point3<f32>)>>,

    livodish: localmap::Livodish,
    rng: rand_pcg::Pcg64Mcg,
}

#[derive(Clone)]
pub struct VoxelDistribution {
    pub transform: Matrix3<f32>,
    pub point_count: u64,
    pub mean: Point3<f32>,
    pub covariance: Matrix3<f32>,
    pub last_updated_at_scan: usize,
}

impl VoxelDistribution {
    pub fn new(first_point: &Point3<f32>) -> Self {
        VoxelDistribution {
            point_count: 1,
            mean: *first_point,
            covariance: Matrix3::zeros(),
            transform: Matrix3::identity(),
            last_updated_at_scan: 0,
        }
    }
    pub fn update(
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

impl IVoxHashMap {
    pub fn new(resolution: f32, capacity: usize, nbtype: IVoxNearbyType, sensitivity: f32) -> Self {
        Self {
            nearby_type: nbtype.clone(),
            inverse_resolution: 1.0 / resolution,
            scan_number: 0,
            #[cfg(feature = "nn_kdt")]
            kd_map: kiddo::float::kdtree::KdTree::<f32, u64, 3, 64, u32>::with_capacity(capacity),
            #[cfg(feature = "nn_kdt")]
            kd_points: VecDeque::with_capacity(capacity),
            #[cfg(feature = "nn_kdt")]
            kd_points_lookup: HashMap::new(),
            livodish: Livodish::new(capacity, sensitivity, resolution / 4.0),
            rng: rand_pcg::Pcg64Mcg::from_os_rng(),
        }
    }

    pub fn is_empty(&self) -> bool {
        #[cfg(feature = "nn_kdt")]
        return self.kd_map.size() == 0;

        #[cfg(not(feature = "nn_kdt"))]
        return self.livodish.is_empty();
    }

    pub fn add(&mut self, points: &[PointXYZ], transform: &nalgebra::Isometry3<f64>) {
        self.scan_number += 1;

        #[cfg(feature = "nn_kdt")]
        {
            if self.scan_number == 1 {
                points.into_iter().for_each(|point| {
                    let point_global = transform.transform_point(&point.xyz_f64());
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
                        let p = Arc::new((self.pt_id, point_global.cast()));
                        self.kd_points.push_front(p.clone());
                        self.kd_points_lookup.insert(self.pt_id, p.clone());
                    }
                });
            } else {
                // check if any point is closer than 25cm, if not, add to tree
                let f = points
                    .into_iter() // TODO as par iter
                    .filter(|point| {
                        let point_global = transform.transform_point(&point.xyz_f64());
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
                    let point_global = transform.transform_point(&point.xyz_f64());
                    self.pt_id += 1;
                    self.kd_map.add(
                        &[
                            point_global.x as f32,
                            point_global.y as f32,
                            point_global.z as f32,
                        ],
                        self.pt_id,
                    );
                    let p = Arc::new((self.pt_id, point_global.cast()));
                    self.kd_points.push_front(p.clone());
                    self.kd_points_lookup.insert(self.pt_id, p.clone());
                });
            }
        }

        #[cfg(not(feature = "nn_kdt"))]
        {
            points.iter().for_each(|point| {
                let global_p = transform.transform_point(&point.xyz_f64());

                let grid = self.pos2grid(&PointXYZ {
                    x: global_p.x as f32,
                    y: global_p.y as f32,
                    z: global_p.z as f32,
                });
                let voxel_key = Self::hash_grid(&grid);
                self.livodish
                    .add(&global_p.cast(), voxel_key, self.scan_number);
            });
            self.livodish.build(self.scan_number);
        }
    }

    #[inline(always)]
    pub fn sample_from_distribution<R: Rng + ?Sized>(
        distribution: &VoxelDistribution,
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
        #[cfg(feature = "nn_kdt")]
        {
            if self.kd_map.size() == 0 {
                return false;
            }

            let neighbors = self.kd_map.nearest_n_within::<SquaredEuclidean>(
                &[query.x, query.y, query.z],
                sq_max_range,
                std::num::NonZeroUsize::new(k).unwrap(),
                true,
            );

            if neighbors.is_empty() {
                return false;
            }

            *out_size = neighbors.len();

            neighbors
                .into_iter()
                .zip(result.iter_mut())
                .for_each(|(neigh, res)| {
                    let p = self.kd_points_lookup[&neigh.item].1;
                    res.x = p.x;
                    res.y = p.y;
                    res.z = p.z;
                });

            return true;
        }

        #[cfg(not(feature = "nn_kdt"))]
        {
            if k == 0 || result.is_empty() {
                *out_size = 0;
                return false;
            }

            let mut rng = self.rng.clone();
            let query_p = nalgebra::Point3::new(query.x, query.y, query.z);

            // --- Gather Neighbors and Total Density ---
            const MAX_NEIGHBORS: usize = 27;
            let mut cached_neighbors: [Option<&VoxelDistribution>; MAX_NEIGHBORS] =
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
            PointXYZ::new(0.2, 0.2, 0.0),
            PointXYZ::new(0.4, 0.4, 0.1),
            PointXYZ::new(0.49, 0.49, 0.1),
            PointXYZ::new(0.75001, 0.51, 0.51),
        ];
        let mut map = IVoxHashMap::new(0.5, 1_000, IVoxNearbyType::default(), 0.3);
        map.add(&h_points, &Isometry3::identity());
        let entire_map = map.get_entire_map();
        assert_eq!(entire_map.len(), 4);

        {
            let mut found = false;
            for point in entire_map {
                if point.x == 0.2 && point.y == 0.2 && point.z == 0.2 {
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
        map.add(&[PointXYZ::new(3.3, 3.3, 3.0)], &Isometry3::identity());
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
            PointXYZ::new(0.2, 0.2, 0.0),
            PointXYZ::new(0.4, 0.4, 0.1),
            PointXYZ::new(0.49, 0.49, 0.1),
            PointXYZ::new(0.75001, 0.51, 0.51),
        ];
        let mut map = IVoxHashMap::new(0.5, 1_000, IVoxNearbyType::Nearby26, 0.3);
        map.add(&h_points, &Isometry3::identity());

        let query = PointXYZ::new(0.0, 0.0, 0.0);
        let mut result = vec![PointXYZ::default(); 6];
        let out_size = &mut 0;
        let res = map.k_nearest_neighbor_query(&query, 6, 5.0, &mut result, out_size);

        assert!(res == true);
        assert_eq!(*out_size, 3);

        assert_eq!(result[0].x, 0.2);
        assert_eq!(result[1].x, 0.4);
        assert_eq!(result[2].y, 0.49);
    }

    #[test]
    #[cfg(not(feature = "nn_kdt"))]
    fn more_distant() {
        let distant_points = vec![
            PointXYZ::new(0.2, 0.2, 0.0),
            PointXYZ::new(5.3, 10.7, 0.1),
            PointXYZ::new(5.6, 10.4, 0.3),
            PointXYZ::new(5.9, 10.1, 0.5),
            PointXYZ::new(6.2, 9.8, 0.7),
            PointXYZ::new(6.5, 9.5, 0.9),
            PointXYZ::new(6.8, 9.2, 1.1),
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

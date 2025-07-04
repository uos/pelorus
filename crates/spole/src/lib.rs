use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use dashmap::DashMap;
use denselocalmap::{Localmap, Point, VoxelDistribution};
#[cfg(feature = "nn_kdt")]
use kiddo::SquaredEuclidean;
use ros_pointcloud2::prelude::*;

pub mod denselocalmap;
use rand_distr::{Distribution, Normal};

use nalgebra::{Cholesky, Isometry3, Matrix3, Point3, SymmetricEigen, Vector3};
use rand::Rng; // Import the Rng trait

use serde::{Deserialize, Serialize};

pub struct IVoxHashMap {
    nearby_type: IVoxNearbyType,
    inverse_resolution: f32,
    resolution: f32,
    sensitivity: f32,
    max_map_size: usize,
    pub grid_map: Localmap,
    scan_number: usize,
    pt_id: u64,
    buffer_current_idx: usize,
    buffer_init_full: bool,
    #[cfg(feature = "nn_kdt")]
    kd_map: kiddo::float::kdtree::KdTree<f32, u64, 3, 64, u32>,
    #[cfg(not(feature = "nn_kdt"))]
    points: Vec<PointXYZ>,
    #[cfg(feature = "nn_kdt")]
    kd_points: VecDeque<Arc<(u64, nalgebra::Point3<f32>)>>,
    #[cfg(feature = "nn_kdt")]
    kd_points_lookup: HashMap<u64, Arc<(u64, nalgebra::Point3<f32>)>>,
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
            grid_map: Localmap::new(128 * 512, 10, resolution),
            max_map_size: capacity,
            scan_number: 0,
            pt_id: 0,
            resolution,
            sensitivity,
            buffer_current_idx: 0,
            buffer_init_full: false,
            #[cfg(feature = "nn_kdt")]
            kd_map: kiddo::float::kdtree::KdTree::<f32, u64, 3, 64, u32>::with_capacity(capacity),
            #[cfg(feature = "nn_kdt")]
            kd_points: VecDeque::with_capacity(capacity),
            #[cfg(feature = "nn_kdt")]
            kd_points_lookup: HashMap::new(),
            #[cfg(not(feature = "nn_kdt"))]
            points: Vec::with_capacity(capacity),
        }
    }

    pub fn is_empty(&self) -> bool {
        #[cfg(feature = "nn_kdt")]
        return self.kd_map.size() == 0;

        #[cfg(not(feature = "nn_kdt"))]
        return self.grid_map.map.is_empty();
    }

    // fn add_voxel(&mut self, hash: i32, node: IvoxNode) -> usize {
    //     let mut last_hash = None;
    //     if self.buffer_current_idx >= self.buffer.len() {
    //         if !self.buffer_init_full {
    //             self.buffer_init_full = true;
    //         }
    //         last_hash = Some(self.buffer[self.buffer_current_idx].0);
    //         self.buffer_current_idx = 0;
    //     }

    //     if let Some(last_hash) = last_hash {
    //         self.grid_map.remove(&last_hash);
    //     }

    //     let last_idx = self.buffer_current_idx;
    //     self.buffer[self.buffer_current_idx] = (hash, node);
    //     self.buffer_current_idx += 1;

    //     last_idx
    // }

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
                // self.grid_map
                //     .map
                //     .entry(voxel_key)
                //     .and_modify(|dist| {
                //         dist.update(&global_p.cast(), 100.0, self.resolution as f32 / 4.0)
                //     })
                //     .or_insert_with(|| VoxelDistribution::new(&global_p.cast()));
                let entry = self.grid_map.map.get_mut(&voxel_key);
                if let Some(entry) = entry {
                    entry.update(
                        &global_p.cast(),
                        self.sensitivity,
                        self.resolution as f32 / 4.0,
                    );
                } else {
                    // if self.scan_number < 2 {
                    self.grid_map
                        .map
                        .insert(voxel_key, VoxelDistribution::new(&global_p.cast()));
                    // }
                }
            });
            // TODO removal after simple works
        }
    }

    /// Samples a specified number of 3D points from a voxel distribution.
    ///
    /// # Arguments
    ///
    /// * `distribution` - The VoxelDistribution to sample from.
    /// * `num_samples` - The number of points to generate.
    /// * `rng` - A mutable reference to a random number generator.
    ///
    /// # Returns
    ///
    /// * A `Vec<Point3<f64>>` containing the sampled points.
    pub fn sample_from_distribution<R: Rng + ?Sized>(
        distribution: &VoxelDistribution,
        num_samples: usize,
        rng: &mut R,
    ) -> Vec<Point3<f32>> {
        let cholesky = match Cholesky::new(distribution.covariance) {
            Some(chol) => chol.l(),
            None => {
                let eigen = SymmetricEigen::new(distribution.covariance);
                Matrix3::from_diagonal(&eigen.eigenvalues.map(|e| e.max(0.0).sqrt()))
            }
        };
        let standard_normal = Normal::new(0.0, 1.0).unwrap();
        let mut samples = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let rand_vec = Vector3::new(
                standard_normal.sample(rng),
                standard_normal.sample(rng),
                standard_normal.sample(rng),
            );
            let transformed_vec = cholesky * rand_vec;
            samples.push(distribution.mean + transformed_vec);
        }

        samples
    }

    #[must_use]
    pub fn k_nearest_neighbor_query(
        &self,
        query: &PointXYZ,
        k: usize,
        max_range: f64,
        result: &mut [PointXYZ],
        out_size: &mut usize,
    ) -> bool {
        #[cfg(feature = "nn_kdt")]
        {
            if self.kd_map.size() == 0 {
                return false;
            }

            let neighbors = self.kd_map.nearest_n_within::<SquaredEuclidean>(
                &[query.x, query.y, query.z],
                (max_range * max_range) as f32,
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

            let query_p = nalgebra::Point3::new(query.x, query.y, query.z);
            let mut rng = rand::rng();

            // --- Gather Neighbors and Total Density ---

            const MAX_NEIGHBORS: usize = 27;
            let mut cached_neighbors: [Option<&VoxelDistribution>; MAX_NEIGHBORS] =
                [None; MAX_NEIGHBORS];
            let mut num_cached = 0;
            let mut total_density = 0u64;

            let mut cache_and_sum_density = |grids: &[(i32, i32, i32)]| {
                for g in grids {
                    if let Some(dist) = self.grid_map.map.get(&Self::hash_grid(g)) {
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

            // Iterate through the valid entries in our cached array.
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

                    let s = Self::sample_from_distribution(dist, sample_count, &mut rng);

                    for p in s {
                        if (query_p - p).norm_squared() < (max_range * max_range) as f32 {
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

    pub fn pos2grid(&self, pos: &PointXYZ) -> (i32, i32, i32) {
        let gx = (pos.x * self.inverse_resolution).round() as i32;
        let gy = (pos.y * self.inverse_resolution).round() as i32;
        let gz = (pos.z * self.inverse_resolution).round() as i32;
        (gx, gy, gz)
    }

    pub fn hash_grid(grid_key: &(i32, i32, i32)) -> i32 {
        const PRIME_1: i32 = 73856093;
        const PRIME_2: i32 = 19349663;
        const PRIME_3: i32 = 83492791;
        // PRIME_4 is no longer needed

        let mut hash = grid_key.0.wrapping_mul(PRIME_1);
        hash = hash.wrapping_add(grid_key.1.wrapping_mul(PRIME_2));
        hash = hash.wrapping_add(grid_key.2.wrapping_mul(PRIME_3));

        // The wrapping_rem line is removed.

        hash
    }

    pub fn hash(point: &nalgebra::Point3<f32>, voxel_size: f32) -> i32 {
        const PRIME_1: i32 = 73_856_093;
        const PRIME_2: i32 = 19_349_663;
        const PRIME_3: i32 = 83_492_791;
        // const PRIME_4: i32 = 116_881;

        let xyz = point * (1.0 / voxel_size);

        let x = xyz.x.round() as i32;
        let y = xyz.y.round() as i32;
        let z = xyz.z.round() as i32;

        let mut hash = x.wrapping_mul(PRIME_1);
        hash = hash.wrapping_add(y.wrapping_mul(PRIME_2));
        hash = hash.wrapping_add(z.wrapping_mul(PRIME_3));
        // hash = hash.wrapping_rem(PRIME_4);

        hash
    }

    pub fn clear(&mut self) {
        self.grid_map.clear();
    }

    /// Mostly for testing purposes, is not efficient
    pub fn get_entire_map(&self) -> Vec<nalgebra::Point3<f32>> {
        #[cfg(feature = "nn_kdt")]
        {
            return self.kd_points.iter().map(|rc| rc.1).collect();
        }

        #[cfg(not(feature = "nn_kdt"))]
        {
            self.grid_map
                .map
                .values()
                .map(|p| nalgebra::Point3::new(p.mean.x, p.mean.y, p.mean.z))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "nn_kdt"))]
    fn basic_insert() {
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

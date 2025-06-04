use dashmap::DashMap;
#[cfg(feature = "nn_kdt")]
use kdtree::{KdTree, distance::squared_euclidean};
use ros_pointcloud2::prelude::*;

use serde::{Deserialize, Serialize};

pub struct IVoxHashMap {
    nearby_type: IVoxNearbyType,
    inverse_resolution: f32,
    resolution: f64,
    grid_map: dashmap::DashMap<i32, ((i32, i32, i32), usize)>,
    scan_number: usize,
    buffer: Vec<(i32, IvoxNode)>,
    buffer_current_idx: usize,
    buffer_init_full: bool,
    #[cfg(feature = "nn_kdt")]
    kd_map: Vec<PointXYZ>,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub enum IVoxNearbyType {
    Center,
    Nearby6,

    #[default]
    Nearby18,

    Nearby26,
}

#[derive(Debug, Clone)]
pub struct IvoxNode {
    points: Vec<PointXYZ>,
    newest_hit: usize,
}

impl IvoxNode {
    pub fn new() -> Self {
        IvoxNode {
            points: Vec::with_capacity(100),
            newest_hit: 0,
        }
    }

    pub fn linear_knn(
        &self,
        query: &PointXYZ,
        k: usize,
        max_range: f64,
        _scan_number: usize,
    ) -> Vec<PointXYZ> {
        let max_range_sq = max_range.powi(2) as f32;

        //self.newest_hit = scan_number;
        let mut points = self.points.clone(); // TODO avoid clone

        points.sort_unstable_by(|a, b| {
            let dist_a = (a.xyz_f32() - query.xyz_f32()).norm_squared();
            let dist_b = (b.xyz_f32() - query.xyz_f32()).norm_squared();
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        points
            .iter()
            .take(k)
            .filter_map(|p| {
                let dist_p = (p.xyz_f32() - query.xyz_f32()).norm_squared();

                if dist_p > max_range_sq { None } else { Some(p) }
            })
            .cloned()
            .collect()
    }

    fn add(&mut self, points: impl IntoIterator<Item = PointXYZ>) {
        self.points.extend(points);
    }
}

impl Default for IVoxHashMap {
    fn default() -> Self {
        Self::new(0.2, 100_000, IVoxNearbyType::default())
    }
}

impl IVoxHashMap {
    pub fn new(resolution: f32, capacity: usize, nbtype: IVoxNearbyType) -> Self {
        Self {
            nearby_type: nbtype.clone(),
            inverse_resolution: 1.0 / resolution,
            grid_map: DashMap::with_capacity(capacity),
            scan_number: 0,
            resolution: resolution as f64,
            buffer: vec![(0, IvoxNode::new()); capacity],
            buffer_current_idx: 0,
            buffer_init_full: false,
            #[cfg(feature = "nn_kdt")]
            kd_map: Vec::with_capacity(capacity),
        }
    }

    pub fn is_empty(&self) -> bool {
        #[cfg(feature = "nn_kdt")]
        return self.kd_map.is_empty();

        #[cfg(not(feature = "nn_kdt"))]
        return self.grid_map.is_empty();
    }

    fn add_voxel(&mut self, hash: i32, node: IvoxNode) -> usize {
        let mut last_hash = None;
        if self.buffer_current_idx >= self.buffer.len() {
            if !self.buffer_init_full {
                self.buffer_init_full = true;
            }
            last_hash = Some(self.buffer[self.buffer_current_idx].0);
            self.buffer_current_idx = 0;
        }

        if let Some(last_hash) = last_hash {
            self.grid_map.remove(&last_hash);
        }

        let last_idx = self.buffer_current_idx;
        self.buffer[self.buffer_current_idx] = (hash, node);
        self.buffer_current_idx += 1;

        last_idx
    }

    pub fn add(&mut self, points: impl IntoIterator<Item = PointXYZ>) {
        self.scan_number += 1;

        #[cfg(feature = "nn_kdt")]
        {
            let hashes = dashmap::DashMap::new(); // TODO try out ahash
            self.kd_map.par_iter().for_each(|point| {
                let hash = Self::hash(&point.xyz_f32(), 0.50); // TODO hardcoded here, but feat only used for debugging so its ok
                if !hashes.contains_key(&hash) {
                    hashes.insert(hash, Vec::with_capacity(10));
                }
                hashes
                    .get_mut(&hash)
                    .expect("entry created the line before")
                    .push(*point);
            });

            points.into_iter().for_each(|point| {
                let hash = Self::hash(&point.xyz_f32(), 0.50); // TODO and here
                if !hashes.contains_key(&hash) {
                    hashes.insert(hash, Vec::with_capacity(10));
                }
                hashes
                    .get_mut(&hash)
                    .expect("entry created the line before")
                    .push(point);
            });

            let out = hashes
                .par_iter()
                .map(|bucket_ref| {
                    let voxel_average = bucket_ref
                        .iter()
                        .fold(nalgebra::Vector3::default(), |acc, point| {
                            point.xyz_f32().coords + acc
                        });

                    /*let voxel_average = bucket_ref
                    .par_iter()
                    .map(|point| point.xyz().coords)
                    .reduce(Vector3::default, |acc, coords| acc + coords);*/

                    let avg = voxel_average * (1.0 / bucket_ref.len() as f32);

                    let mut min_dist = f32::MAX;
                    let mut closest_point = PointXYZ::default();
                    for point in bucket_ref.iter() {
                        let dist = (point.xyz_f32().coords - avg).norm_squared();
                        if dist < min_dist {
                            min_dist = dist;
                            closest_point = (*point).clone();
                        }
                    }

                    closest_point
                })
                .collect::<Vec<PointXYZ>>();

            self.kd_map.clear();
            self.kd_map.extend(out);
        }

        #[cfg(not(feature = "nn_kdt"))]
        {
            let hashes = dashmap::DashMap::new(); // TODO try out ahash
            self.map.par_iter().for_each(|point| {
                let hash = hash(&point.xyz_f32(), 0.50);
                if !hashes.contains_key(&hash) {
                    hashes.insert(hash, Vec::with_capacity(10));
                }
                hashes
                    .get_mut(&hash)
                    .expect("entry created the line before")
                    .push(*point);
            });

            points.into_iter().for_each(|point| {
                let hash = hash(&point.xyz_f32(), 0.50);
                if !hashes.contains_key(&hash) {
                    hashes.insert(hash, Vec::with_capacity(10));
                }
                hashes
                    .get_mut(&hash)
                    .expect("entry created the line before")
                    .push(point);
            });

            let out = hashes
                .par_iter()
                .map(|bucket_ref| {
                    let voxel_average = bucket_ref
                        .iter()
                        .fold(nalgebra::Vector3::default(), |acc, point| {
                            point.xyz_f32().coords + acc
                        });

                    /*let voxel_average = bucket_ref
                    .par_iter()
                    .map(|point| point.xyz().coords)
                    .reduce(Vector3::default, |acc, coords| acc + coords);*/

                    let avg = voxel_average * (1.0 / bucket_ref.len() as f32);

                    let mut min_dist = f32::MAX;
                    let mut closest_point = PointXYZ::default();
                    for point in bucket_ref.iter() {
                        let dist = (point.xyz_f32().coords - avg).norm_squared();
                        if dist < min_dist {
                            min_dist = dist;
                            closest_point = (*point).clone();
                        }
                    }

                    closest_point
                })
                .collect::<Vec<PointXYZ>>();

            self.map.clear();
            self.map.extend(out);
        }
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
            let grid = self.pos2grid(query);

            let neighboring_buckets: &[(i32, i32, i32)] = match self.nearby_type {
                IVoxNearbyType::Center => &[grid],
                IVoxNearbyType::Nearby6 => &Self::nearby_6(grid),
                IVoxNearbyType::Nearby18 => &Self::nearby_18(grid),
                IVoxNearbyType::Nearby26 => &Self::nearby_26(grid),
            };

            /*let double_buckets = neighboring_buckets
            .iter()
            .map(|g| (Self::hash(g), g))
            .collect::<Vec<_>>();*/

            let mut neighbors = neighboring_buckets
                .iter()
                .filter_map(|g| self.grid_map.get_mut(&Self::hash_grid(g)))
                .filter_map(|hash_node| {
                    let n = self.buffer[hash_node.value().1].1.linear_knn(
                        query,
                        k,
                        max_range,
                        self.scan_number,
                    );
                    if n.is_empty() { None } else { Some(n) }
                })
                .flatten()
                .collect::<Vec<_>>();

            neighbors.sort_unstable_by(|a, b| {
                let dist_a = (a.xyz_f32() - query.xyz_f32()).norm_squared();
                let dist_b = (b.xyz_f32() - query.xyz_f32()).norm_squared();
                dist_a.partial_cmp(&dist_b).unwrap()
            });

            if neighbors.is_empty() {
                return false;
            }

            *out_size = neighbors.len();

            neighbors
                .into_iter()
                .take(k)
                .zip(result.iter_mut())
                .for_each(|(neigh, res)| *res = neigh);

            return true;
        }
        #[cfg(not(feature = "nn_kdt"))]
        {
            let mut map_kd_tree = KdTree::new(3);
            self.map.iter().for_each(|p| {
                map_kd_tree.add([p.x, p.y, p.z], *p).unwrap();
            });

            if self.map.is_empty() {
                return false;
            }

            let mut neighbors = map_kd_tree
                .nearest(&[query.x, query.y, query.z], k, &squared_euclidean)
                .unwrap();

            neighbors.sort_unstable_by(|a, b| {
                let dist_a = (a.1.xyz_f32() - query.xyz_f32()).norm_squared();
                let dist_b = (b.1.xyz_f32() - query.xyz_f32()).norm_squared();
                dist_a.partial_cmp(&dist_b).unwrap()
            });

            if neighbors.is_empty() {
                return false;
            }

            *out_size = neighbors.len();

            neighbors
                .into_iter()
                .filter_map(|n| {
                    let dist = (n.1.xyz_f32() - query.xyz_f32()).norm_squared();
                    if dist < max_range as f32 {
                        Some(n.1)
                    } else {
                        None
                    }
                })
                .take(k)
                .zip(result.iter_mut())
                .for_each(|(neigh, res)| *res = neigh.clone());

            return true;
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

    fn pos2grid(&self, pos: &PointXYZ) -> (i32, i32, i32) {
        let gx = (pos.x * self.inverse_resolution).round() as i32;
        let gy = (pos.y * self.inverse_resolution).round() as i32;
        let gz = (pos.z * self.inverse_resolution).round() as i32;
        (gx, gy, gz)
    }

    fn hash_grid(grid_key: &(i32, i32, i32)) -> i32 {
        const PRIME_1: i32 = 73856093;
        const PRIME_2: i32 = 19349663;
        const PRIME_3: i32 = 83492791;
        const PRIME_4: i32 = 116_881;

        let mut hash = grid_key.0.wrapping_mul(PRIME_1);
        hash = hash.wrapping_add(grid_key.1.wrapping_mul(PRIME_2));
        hash = hash.wrapping_add(grid_key.2.wrapping_mul(PRIME_3));
        hash = hash.wrapping_rem(PRIME_4);

        hash
    }

    fn hash(point: &nalgebra::Point3<f32>, voxel_size: f32) -> i32 {
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

    pub fn clear(&mut self) {
        self.grid_map.clear();
    }

    /// Mostly for testing purposes, is not efficient
    pub fn get_entire_map(&self) -> Vec<PointXYZ> {
        #[cfg(feature = "nn_kdt")]
        {
            return self.kd_map.clone();
        }

        #[cfg(not(feature = "nn_kdt"))]
        {
            if self.buffer_init_full {
                return self
                    .buffer
                    .iter()
                    .map(|node| node.1.points.clone())
                    .flatten()
                    .collect();
            } else {
                let mut out = Vec::with_capacity(self.buffer_current_idx);
                for i in 0..self.buffer_current_idx {
                    let node = &self.buffer[i];
                    let points = &node.1.points;
                    if !points.is_empty() {
                        out.extend(points.clone());
                    }
                }

                return out;
            }
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
        let mut map = IVoxHashMap::new(0.5, 1_000, IVoxNearbyType::default());
        map.add(h_points);
        let entire_map = map.get_entire_map();
        assert_eq!(entire_map.len(), 4);

        {
            let mut found = false;
            for point in entire_map {
                if point == PointXYZ::new(0.2, 0.2, 0.0) {
                    found = true;
                    break;
                }
            }
            assert!(found);
        }
    }

    #[test]
    #[cfg(not(feature = "nn_kdt"))]
    fn inode_linear() {
        let mut inode = IvoxNode::new();
        inode.add([
            PointXYZ::new(0.2, 0.2, 0.0),
            PointXYZ::new(0.4, 0.4, 0.1),
            PointXYZ::new(0.49, 0.49, 0.1),
        ]);

        assert_eq!(inode.points.len(), 3);
        let res = inode.linear_knn(&PointXYZ::new(0.4, 0.4, 0.1), 1, 5.0, 1);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].x, 0.4);
        assert_eq!(res[0].y, 0.4);
        assert_eq!(res[0].z, 0.1);
    }

    #[test]
    #[cfg(not(feature = "nn_kdt"))]
    fn basic_add_and_search() {
        let mut map = IVoxHashMap::new(0.5, 1_000, IVoxNearbyType::default());
        map.add([PointXYZ::new(3.3, 3.3, 3.0)]);
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
        let mut map = IVoxHashMap::new(0.5, 1_000, IVoxNearbyType::Nearby26);
        map.add(h_points);

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

        let mut map = IVoxHashMap::new(0.5, 1_000, IVoxNearbyType::Nearby18);
        map.add(distant_points);

        let query = PointXYZ::new(5.5, 10.5, 0.0);
        let mut result = vec![PointXYZ::default(); 2];
        let out_size = &mut 0;

        let res = map.k_nearest_neighbor_query(&query, 2, 5.0, &mut result, out_size);

        assert!(res);
        assert_eq!(*out_size, 2);
    }
}

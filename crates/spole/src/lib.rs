use std::ops::Deref;

use dashmap::DashMap;
use ros_pointcloud2::prelude::*;

/*pub trait ThreeDimensionalPoint: Sized {
    fn x(&self) -> f64;
    fn y(&self) -> f64;
    fn z(&self) -> f64;

    fn new(x: f64, y: f64, z: f64) -> Self;

    fn distance(&self, other: &Self) -> f64 {
        let dx = self.x() - other.x();
        let dy = self.y() - other.y();
        let dz = self.z() - other.z();
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}*/

/*pub trait HashNeighborSearch {
    fn new(resolution: f64) -> Self;
    fn add(&mut self, points: impl IntoIterator<Item = impl ThreeDimensionalPoint>);
    fn query(
        &self,
        query: &impl ThreeDimensionalPoint,
        k: usize,
        max_range: f64,
    ) -> impl IntoIterator<Item = impl ThreeDimensionalPoint>;
}*/

pub struct IVoxHashMap {
    nearby_type: IVoxNearbyType,
    inverse_resolution: f32,
    grid_map: dashmap::DashMap<i64, ((i32, i32, i32), IvoxNode)>,
    scan_number: usize,
}

#[derive(Debug, Default, Clone)]
pub enum IVoxNearbyType {
    Center,
    Nearby6,

    #[default]
    Nearby18,

    Nearby26,
}

#[derive(Debug)]
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
        &mut self,
        query: &PointXYZ,
        k: usize,
        max_range: f64,
        scan_number: usize,
    ) -> Vec<PointXYZ> {
        let max_range_sq = max_range.powi(2) as f32;

        self.newest_hit = scan_number;

        self.points.sort_unstable_by(|a, b| {
            let dist_a = ((a.x - query.x).powi(2)
                + (a.y - query.y).powi(2)
                + (a.z - query.z).powi(2)) as f64;
            let dist_b = ((b.x - query.x).powi(2)
                + (b.y - query.y).powi(2)
                + (b.z - query.z).powi(2)) as f64;
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        self.points
            .iter()
            .take(k)
            .filter_map(|p| {
                if ((p.x - query.x).powi(2) + (p.y - query.y).powi(2) + (p.z - query.z).powi(2))
                    < max_range_sq
                {
                    Some(p)
                } else {
                    None
                }
            })
            .cloned()
            .collect()
    }

    pub fn add(&mut self, points: impl IntoIterator<Item = PointXYZ>) {
        self.points.extend(points);
    }
}

impl Default for IVoxHashMap {
    fn default() -> Self {
        Self::new(0.2, 100_000, &IVoxNearbyType::default())
    }
}

impl IVoxHashMap {
    pub fn new(resolution: f32, capacity: usize, nbtype: &IVoxNearbyType) -> Self {
        Self {
            nearby_type: nbtype.clone(),
            inverse_resolution: 1.0 / resolution,
            grid_map: DashMap::with_capacity(capacity),
            scan_number: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.grid_map.is_empty()
    }

    pub fn add(&mut self, points: impl IntoIterator<Item = PointXYZ>) {
        self.scan_number += 1;

        points.into_iter().for_each(|p| {
            let grid = self.pos2grid(&p);
            let hashed = Self::hash(&grid);
            let mut grid = self
                .grid_map
                .entry(hashed)
                .or_insert_with(|| (grid, IvoxNode::new()));
            grid.1.add(std::iter::once(p));
        });
    }

    pub fn point_distance_sq(p1: &PointXYZ, p2: &PointXYZ) -> f32 {
        (p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2) + (p1.z - p2.z).powi(2)
    }

    #[must_use]
    pub fn k_nearest_neighbor_query(
        &self,
        query: &PointXYZ,
        k: usize,
        max_range: f64,
    ) -> Vec<PointXYZ> {
        let mut neighbors: Vec<PointXYZ> =
            Self::generate_nearby_grids(self.pos2grid(query), &self.nearby_type)
                .iter()
                .filter_map(|g| self.grid_map.get_mut(&(Self::hash(g))))
                .map(|mut val| {
                    val.value_mut()
                        .1
                        .linear_knn(query, k, max_range, self.scan_number)
                })
                .flatten()
                .collect();

        neighbors.sort_unstable_by(|a, b| {
            Self::point_distance_sq(a, query)
                .partial_cmp(&Self::point_distance_sq(b, query))
                .unwrap()
        });

        neighbors.into_iter().take(k).collect()
    }

    fn nearby_6(grid_key: (i32, i32, i32)) -> Vec<(i32, i32, i32)> {
        let (gx, gy, gz) = grid_key;
        vec![
            (gx, gy, gz),
            (gx - 1, gy, gz),
            (gx + 1, gy, gz),
            (gx, gy + 1, gz),
            (gx, gy - 1, gz),
            (gx, gy, gz - 1),
            (gx, gy, gz + 1),
        ]
    }

    fn nearby_18(grid_key: (i32, i32, i32)) -> Vec<(i32, i32, i32)> {
        let (gx, gy, gz) = grid_key;
        let mut first_six = Self::nearby_6(grid_key);
        first_six.extend(vec![
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
        ]);
        first_six
    }

    fn nearby_26(grid_key: (i32, i32, i32)) -> Vec<(i32, i32, i32)> {
        let (gx, gy, gz) = grid_key;
        let mut first_eighteen = Self::nearby_18(grid_key);
        first_eighteen.extend(vec![
            (gx + 1, gy + 1, gz + 1),
            (gx - 1, gy + 1, gz + 1),
            (gx + 1, gy - 1, gz + 1),
            (gx + 1, gy + 1, gz - 1),
            (gx - 1, gy - 1, gz + 1),
            (gx - 1, gy + 1, gz - 1),
            (gx + 1, gy - 1, gz - 1),
            (gx - 1, gy - 1, gz - 1),
        ]);
        first_eighteen
    }

    fn generate_nearby_grids(
        grid_key: (i32, i32, i32),
        nbtype: &IVoxNearbyType,
    ) -> Vec<(i32, i32, i32)> {
        match nbtype {
            IVoxNearbyType::Center => vec![grid_key],
            IVoxNearbyType::Nearby6 => Self::nearby_6(grid_key),
            IVoxNearbyType::Nearby18 => Self::nearby_18(grid_key),
            IVoxNearbyType::Nearby26 => Self::nearby_26(grid_key),
        }
    }

    fn pos2grid(&self, pos: &PointXYZ) -> (i32, i32, i32) {
        let gx = (pos.x * self.inverse_resolution).round() as i32;
        let gy = (pos.y * self.inverse_resolution).round() as i32;
        let gz = (pos.z * self.inverse_resolution).round() as i32;
        (gx, gy, gz)
    }

    fn hash(grid_key: &(i32, i32, i32)) -> i64 {
        const MOD: i64 = 100_000;
        (grid_key.0 as i64 * 73_856_093)
            ^ (grid_key.1 as i64 * 471_943)
            ^ (grid_key.2 as i64 * 83_492_791) % MOD
    }

    // TODO to list taht moves newest used data to the front and pointing from inode to the cache and back. if all points from the inode are removed, remove entire inode.
    pub fn remove_cold_voxels(&mut self, threshold: usize) {
        let keys = self
            .grid_map
            .iter()
            .map(|entry| *entry.key())
            .collect::<Vec<i64>>();

        keys.into_iter().for_each(|key| {
            self.grid_map
                .remove_if(&key, |_, node| node.1.newest_hit > threshold);
        });
    }

    pub fn clear(&mut self) {
        self.grid_map.clear();
    }

    /// Mostly for testing purposes, is not efficient
    pub fn get_entire_map(&self) -> Vec<PointXYZ> {
        self.grid_map
            .iter()
            .map(|entry| entry.value().1.points.clone())
            .flatten()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_insert() {
        let h_points = vec![
            PointXYZ::new(0.2, 0.2, 0.0),
            PointXYZ::new(0.4, 0.4, 0.1),
            PointXYZ::new(0.49, 0.49, 0.1),
            PointXYZ::new(0.75001, 0.51, 0.51),
        ];
        let mut map = IVoxHashMap::new(0.5, 1_000, &IVoxNearbyType::default());
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
    fn basic_add_and_search() {
        let mut map = IVoxHashMap::new(0.5, 1_000, &IVoxNearbyType::default());
        map.add([PointXYZ::new(3.3, 3.3, 3.0)]);
        let res = map.k_nearest_neighbor_query(&PointXYZ::new(3.2, 3.2, 3.0), 1, 5.0);
        assert_eq!(res.len(), 1);
    }

    #[test]
    fn basic_origin() {
        let h_points = vec![
            PointXYZ::new(0.2, 0.2, 0.0),
            PointXYZ::new(0.4, 0.4, 0.1),
            PointXYZ::new(0.49, 0.49, 0.1),
            PointXYZ::new(0.75001, 0.51, 0.51),
        ];
        let mut map = IVoxHashMap::new(0.5, 1_000, &IVoxNearbyType::default());
        map.add(h_points);

        let query = PointXYZ::new(0.0, 0.0, 0.0);
        let result = map.k_nearest_neighbor_query(&query, 6, 5.0);

        /*assert_eq!(result.len(), 3);

        assert_eq!(result[0].x, 0.2);
        assert_eq!(result[1].x, 0.4);
        assert_eq!(result[2].y, 0.49);*/
    }
}

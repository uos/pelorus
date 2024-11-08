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
    grid_map: dashmap::DashMap<i32, ((i32, i32, i32), IvoxNode)>,
    scan_number: usize,
}

#[derive(Debug, Default, Clone, Copy)]
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
            let dist_a = (a.xyz_f32() - query.xyz_f32()).norm_squared();
            let dist_b = (b.xyz_f32() - query.xyz_f32()).norm_squared();
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        self.points
            .iter()
            .take(k)
            .filter_map(|p| {
                let dist_p = (p.xyz_f32() - query.xyz_f32()).norm_squared();

                if dist_p > max_range_sq {
                    None
                } else {
                    Some(p)
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

    #[must_use]
    pub fn k_nearest_neighbor_query(
        &self,
        query: &PointXYZ,
        k: usize,
        max_range: f64,
    ) -> Option<Vec<PointXYZ>> {
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
            .filter_map(|g| self.grid_map.get_mut(&(Self::hash(g))))
            .filter_map(|mut hash_node| {
                let n = hash_node
                    .value_mut()
                    .1
                    .linear_knn(query, k, max_range, self.scan_number);
                if n.is_empty() {
                    None
                } else {
                    Some(n)
                }
            })
            .flatten()
            .collect::<Vec<_>>();

        neighbors.sort_unstable_by(|a, b| {
            let dist_a = (a.xyz_f32() - query.xyz_f32()).norm_squared();
            let dist_b = (b.xyz_f32() - query.xyz_f32()).norm_squared();
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        let neighbors: Vec<PointXYZ> = neighbors.into_iter().take(k).collect();
        if neighbors.is_empty() {
            None
        } else {
            Some(neighbors)
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

    fn hash(grid_key: &(i32, i32, i32)) -> i32 {
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

    // TODO to list taht moves newest used data to the front and pointing from inode to the cache and back. if all points from the inode are removed, remove entire inode.
    pub fn remove_cold_voxels(&mut self, threshold: usize) {
        let keys = self
            .grid_map
            .iter()
            .map(|entry| *entry.key())
            .collect::<Vec<_>>();

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
        let mut map = IVoxHashMap::new(0.5, 1_000, IVoxNearbyType::default());
        map.add([PointXYZ::new(3.3, 3.3, 3.0)]);
        let res = map.k_nearest_neighbor_query(&PointXYZ::new(3.2, 3.2, 3.0), 1, 5.0);
        assert!(res.is_some());
        let res = res.unwrap();
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
        let mut map = IVoxHashMap::new(0.5, 1_000, IVoxNearbyType::Nearby26);
        map.add(h_points);

        let query = PointXYZ::new(0.0, 0.0, 0.0);
        let result = map.k_nearest_neighbor_query(&query, 6, 5.0);

        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.len(), 3);

        assert_eq!(result[0].x, 0.2);
        assert_eq!(result[1].x, 0.4);
        assert_eq!(result[2].y, 0.49);
    }

    #[test]
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
        let result = map.k_nearest_neighbor_query(&query, 2, 5.0);

        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.len(), 2);
    }
}

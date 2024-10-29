/*
use dashmap::DashMap;
use ros_pointcloud2::prelude::*;

#[derive(Debug, Default, Clone)]
pub enum NearbyType {
    Center,
    Nearby6,

    #[default]
    Nearby18,

    Nearby26,
}

#[derive(Debug)]
pub struct IvoxNode {
    pub points: Vec<PointXYZ>,
    pub newest_hit: usize,
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
        let max_range_sq = max_range.powi(2);

        self.newest_hit = scan_number;

        let mut dists: Vec<(f64, &PointXYZ)> = self
            .points
            .iter()
            .map(|p| {
                let dist = ((p.x - query.x).powi(2)
                    + (p.y - query.y).powi(2)
                    + (p.z - query.z).powi(2)) as f64;
                (dist, p)
            })
            .collect();
        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        dists
            .into_iter()
            .enumerate()
            .take_while(|(i, (d, _))| *d <= max_range_sq && *i < k)
            .map(|(_, (_, p))| *p)
            .collect()
    }
}

pub struct Localmap {
    nearby_type: NearbyType,
    inverse_resolution: f32,
    grid_map: dashmap::DashMap<u32, ((i32, i32, i32), IvoxNode)>,
    scan_number: usize,
}

impl Default for Localmap {
    fn default() -> Self {
        Self::new(0.2, 100000, &NearbyType::Nearby6)
    }
}

impl Localmap {
    pub fn new(resolution: f32, capacity: usize, nbtype: &NearbyType) -> Self {
        Localmap {
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
            let hashed = Self::hash(&grid) as u32;
            let mut grid = self
                .grid_map
                .entry(hashed)
                .or_insert_with(|| (grid, IvoxNode::new()));
            grid.1.points.push(p);
        });
    }

    pub fn point_distance_sq(p1: &PointXYZ, p2: &PointXYZ) -> f32 {
        (p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2) + (p1.z - p2.z).powi(2)
    }

    pub fn k_nearest_neighbor_query(
        &self,
        query: &PointXYZ,
        k: usize,
        max_range: f64,
    ) -> Vec<PointXYZ> {
        let mut neighbors: Vec<PointXYZ> =
            Self::generate_nearby_grids(self.pos2grid(query), &self.nearby_type)
                .iter()
                .filter_map(|g| self.grid_map.get_mut(&(Self::hash(g) as u32)))
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
            (gx + 1, gy + 1, gz + 1),
            (gx + 1, gy - 1, gz + 1),
            (gx - 1, gy + 1, gz + 1),
            (gx - 1, gy - 1, gz + 1),
            (gx + 1, gy + 1, gz - 1),
            (gx + 1, gy - 1, gz - 1),
            (gx - 1, gy + 1, gz - 1),
            (gx - 1, gy - 1, gz - 1),
            (gx, gy, gz + 1),
            (gx, gy, gz - 1),
            (gx + 1, gy, gz),
            (gx - 1, gy, gz),
            (gx, gy + 1, gz),
            (gx, gy - 1, gz),
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
        nbtype: &NearbyType,
    ) -> Vec<(i32, i32, i32)> {
        match nbtype {
            NearbyType::Center => vec![grid_key],
            NearbyType::Nearby6 => Self::nearby_6(grid_key),
            NearbyType::Nearby18 => Self::nearby_18(grid_key),
            NearbyType::Nearby26 => Self::nearby_26(grid_key),
        }
    }

    fn pos2grid(&self, pos: &PointXYZ) -> (i32, i32, i32) {
        let gx = (pos.x * self.inverse_resolution).round() as i32;
        let gy = (pos.y * self.inverse_resolution).round() as i32;
        let gz = (pos.z * self.inverse_resolution).round() as i32;
        (gx, gy, gz)
    }

    fn hash(grid_key: &(i32, i32, i32)) -> i64 {
        const MOD: i64 = 100000;
        (grid_key.0 as i64 * 73856093)
            ^ (grid_key.1 as i64 * 471943)
            ^ (grid_key.2 as i64 * 83492791) % MOD
    }

    pub fn remove_cold_voxels(&mut self, threshold: usize) {
        let keys = self
            .grid_map
            .iter()
            .map(|entry| *entry.key())
            .collect::<Vec<u32>>();

        keys.into_iter().for_each(|key| {
            self.grid_map
                .remove_if(&key, |_, node| node.1.newest_hit > threshold);
        });
    }

    pub fn clear(&mut self) {
        self.grid_map.clear();
    }

    pub fn get_entire_map(&self) -> Vec<PointXYZ> {
        self.grid_map
            .iter()
            .map(|entry| entry.value().1.points.clone())
            .flatten()
            .collect()
    }
}
*/

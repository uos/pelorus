use rustc_hash::FxHashMap;

use ros_pointcloud2::points::PointXYZ;

pub type Point = nalgebra::Point3<f32>;

#[derive(Debug, Clone)]
pub struct VoxelGrid {
    voxels: FxHashMap<(i32, i32, i32), Point>,
    resolution: f32,
}

impl VoxelGrid {
    pub fn new(resolution: f32) -> Self {
        assert!(resolution > 0.0, "Resolution must be positive");
        VoxelGrid {
            voxels: FxHashMap::default(),
            resolution,
        }
    }

    fn get_voxel_index(&self, point: &Point) -> (i32, i32, i32) {
        (
            (point.x / self.resolution).floor() as i32,
            (point.y / self.resolution).floor() as i32,
            (point.z / self.resolution).floor() as i32,
        )
    }

    pub fn insert(&mut self, point: Point) {
        let index = self.get_voxel_index(&point);
        self.voxels.entry(index).or_insert(point);
    }

    pub fn add_pointcloud(&mut self, points: &[Point]) {
        for point in points {
            self.insert(*point);
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Point> {
        self.voxels.values()
    }

    pub fn len(&self) -> usize {
        self.voxels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }

    pub fn clear(&mut self) {
        self.voxels.clear();
    }
}

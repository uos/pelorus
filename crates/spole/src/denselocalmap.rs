use core::panic;
use std::collections::HashMap;

use nalgebra::{Cholesky, Isometry3, Matrix3, Point3, SymmetricEigen, Vector3};
use rand::RngCore;
use rand::rngs::ThreadRng;
use rayon::prelude::*;

use rand_distr::{Distribution, Normal};

use rand::Rng; // Import the Rng trait
use rand::thread_rng;
use ros_pointcloud2::points::PointXYZ; // Import the thread_rng function to get an RNG instance

pub struct VoxelDistribution {
    pub mean: Point3<f32>,
    pub covariance: Matrix3<f32>,
    pub cholesky: Matrix3<f32>,
}

/// Calculates the 3D distribution (mean and covariance) for a given set of 3D points.
///
/// # Arguments
///
/// * `points` - A slice of n 3D points.
///
/// # Returns
///
/// * `Some(VoxelDistribution)` if there are points to process.
/// * `None` if the input slice is empty.
pub fn build_voxel_distribution(points: &[(i32, Point)]) -> Option<VoxelDistribution> {
    if points.is_empty() {
        return None;
    }

    let n = points.len() as f64;

    // Calculate the mean (centroid)
    let sum_of_points = points.iter().fold(Point3::origin(), |acc, p| {
        acc + Vector3::new(p.1.0[0], p.1.0[1], p.1.0[2])
    });
    let mean = Point3::from(sum_of_points.coords / n as f32);

    // Calculate the covariance matrix
    let mut covariance = Matrix3::zeros();
    for point in points {
        let point = Point3::new(point.1.0[0], point.1.0[1], point.1.0[2]);
        let diff = point - mean;
        covariance += &diff * &diff.transpose();
    }
    covariance /= n as f32;

    // We need to be able to sample from a multivariate normal distribution.
    // This can be done by sampling from a standard normal distribution and
    // transforming the samples using the Cholesky decomposition of the covariance matrix.
    let cholesky = match Cholesky::new(covariance) {
        Some(chol) => chol.l(),
        // If the covariance matrix is not positive definite, we can fall back
        // to a diagonal matrix (no correlation between axes).
        None => {
            let eigen = SymmetricEigen::new(covariance);
            Matrix3::from_diagonal(&eigen.eigenvalues.map(|e| e.max(0.0).sqrt()))
        }
    };

    let dist = VoxelDistribution {
        mean,
        covariance,
        cholesky,
    };
    Some(dist)
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
    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    let mut samples = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let rand_vec = Vector3::new(
            standard_normal.sample(rng),
            standard_normal.sample(rng),
            standard_normal.sample(rng),
        );
        let transformed_vec = distribution.cholesky * rand_vec;
        samples.push(distribution.mean + transformed_vec);
    }

    samples
}

pub fn get_num_samples(original_point_count: usize, k: usize) -> usize {
    let max_samples = (k as f64 / 3.0).floor() as usize;
    if max_samples == 0 {
        return 0;
    }

    // A simple linear scaling based on density.
    // The `density_factor` would need to be tuned.
    let density_factor = 50.0;
    let num_to_sample = 1 + (original_point_count as f64 / density_factor).floor() as usize;

    num_to_sample.min(max_samples)
}

pub const FLOATS_PER_POINT: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Point(pub [f32; 4]);

pub struct Localmap {
    // A single, large buffer holding all point clouds in the ring buffer sequentially.
    point_clouds: Vec<f32>,

    // A buffer to store the merged and transformed point cloud for processing.
    dense_buffer: Vec<f32>,

    // Transformation for each point cloud slice in the ring buffer.
    transforms: Vec<Isometry3<f64>>,

    // Index of the next slot in the ring buffer to be written to.
    next_inactive_idx: usize,

    // Index of the most recently added point cloud.
    current_pcl_idx: usize,

    // Tracks if the ring buffer has completed a full cycle.
    was_full: bool,

    // The total number of points currently in the `dense_buffer`.
    total_dense_point_count: usize,

    max_points_per_scan: usize,
    ringbuffer_size: usize,

    pub map: HashMap<i32, ((i32, i32, i32), VoxelDistribution)>,
    voxel_resolution: f32,
}

impl Localmap {
    pub fn new(max_points_per_scan: usize, ringbuffer_size: usize, voxel_resolution: f32) -> Self {
        assert_eq!(std::mem::size_of::<f32>(), std::mem::size_of::<u32>());

        let mut map = Self {
            point_clouds: vec![0.0; max_points_per_scan * FLOATS_PER_POINT * ringbuffer_size],
            dense_buffer: vec![0.0; (max_points_per_scan * FLOATS_PER_POINT) * ringbuffer_size],
            transforms: vec![Isometry3::identity(); ringbuffer_size],
            next_inactive_idx: 0,
            current_pcl_idx: 0,
            was_full: false,
            total_dense_point_count: 0,
            voxel_resolution,
            max_points_per_scan,
            ringbuffer_size,
            map: HashMap::with_capacity(10_000), // TODO add enough for rough spatial resolution
        };

        for i in 0..ringbuffer_size {
            map.set_actual_pointcloud_size(i, 0);
        }

        map
    }

    pub fn clear(&mut self) {
        self.was_full = false;
    }

    pub fn as_merged_cloud(&self) -> &[Point] {
        let dense_floats = self.total_dense_point_count * FLOATS_PER_POINT;
        let float_slice = &self.dense_buffer[0..dense_floats];
        bytemuck::cast_slice(float_slice)
    }

    /// Prepares and returns a mutable slice for the next point cloud to be registered.
    pub fn next_register_buffer(&mut self) -> &mut [Point] {
        let idx = self.get_next_inactive_idx();
        self.current_pcl_idx = idx;

        let start = idx * self.max_points_per_scan * FLOATS_PER_POINT;
        let end = start + (self.max_points_per_scan * FLOATS_PER_POINT);
        bytemuck::cast_slice_mut(&mut self.point_clouds[start..end])
    }

    pub fn copy_to_next_buffer(&mut self, src_points: &[Point]) {
        let num_points = src_points.len();
        let dest_slice = self.next_register_buffer();

        // Unsafe is used here for a highly efficient reinterpret_cast.
        // It is safe because we know Point is repr(C) and has the same layout as [f32; 4].
        let src_floats: &[Point] =
            unsafe { std::slice::from_raw_parts(src_points.as_ptr() as *const Point, num_points) };

        dest_slice[..src_floats.len()].copy_from_slice(src_floats);
        self.set_actual_pointcloud_size(self.current_pcl_idx, num_points as u32);
    }

    pub fn copy_to_next_buffer_xyz(&mut self, src_points: &[PointXYZ]) {
        let num_points = src_points.len();
        let dest_slice = self.next_register_buffer();

        // Unsafe is used here for a highly efficient reinterpret_cast.
        // It is safe because we know Point is repr(C) and has the same layout as [f32; 4].
        let src_floats: &[Point] =
            unsafe { std::slice::from_raw_parts(src_points.as_ptr() as *const Point, num_points) };

        dest_slice[..src_floats.len()].copy_from_slice(src_floats);
        self.set_actual_pointcloud_size(self.current_pcl_idx, num_points as u32);
    }

    fn hash(point: &nalgebra::Point3<f32>, voxel_size: f32) -> i32 {
        let xyz = point * (1.0 / voxel_size);

        let x = xyz.x.round() as i32;
        let y = xyz.y.round() as i32;
        let z = xyz.z.round() as i32;

        Self::hash_grid(&(x, y, z))
    }

    fn hash_grid(grid_key: &(i32, i32, i32)) -> i32 {
        const PRIME_1: i32 = 73856093;
        const PRIME_2: i32 = 19349663;
        const PRIME_3: i32 = 83492791;
        // const PRIME_4: i32 = 116_881;

        let mut hash = grid_key.0.wrapping_mul(PRIME_1);
        hash = hash.wrapping_add(grid_key.1.wrapping_mul(PRIME_2));
        hash = hash.wrapping_add(grid_key.2.wrapping_mul(PRIME_3));
        // hash = hash.wrapping_rem(PRIME_4);

        hash
    }

    pub fn commit_and_merge(&mut self, latest_transform: Isometry3<f64>) {
        self.transforms[self.current_pcl_idx] = latest_transform;

        let mut dense_offset = 0;
        let mut total_points = 0;

        for i in 0..self.ringbuffer_size {
            if i == self.next_inactive_idx {
                continue; // Skip the next inactive buffer.
            }

            let pcl_size = self.get_actual_pointcloud_size(i) as usize;
            if pcl_size == 0 {
                continue;
            }

            // Get slices for the source and destination buffers.
            let cloud_start = i * self.max_points_per_scan * FLOATS_PER_POINT;
            let source_cloud =
                &self.point_clouds[cloud_start..cloud_start + pcl_size * FLOATS_PER_POINT];

            let dest_start = dense_offset;
            let dest_cloud =
                &mut self.dense_buffer[dest_start..dest_start + pcl_size * FLOATS_PER_POINT];

            let transform = self.transforms[i];

            source_cloud
                .par_chunks_exact(FLOATS_PER_POINT) // Process points in parallel
                .zip(dest_cloud.par_chunks_exact_mut(FLOATS_PER_POINT))
                .for_each(|(src_point, dest_point)| {
                    let pos = Point3::new(
                        src_point[0] as f64,
                        src_point[1] as f64,
                        src_point[2] as f64,
                    );

                    let transformed_pos = transform * pos;

                    dest_point[0] = transformed_pos.x as f32;
                    dest_point[1] = transformed_pos.y as f32;
                    dest_point[2] = transformed_pos.z as f32;
                    dest_point[3] = 1.0;
                });

            dense_offset += pcl_size * FLOATS_PER_POINT;
            total_points += pcl_size;
        }

        self.total_dense_point_count = total_points;
        dbg!(self.total_dense_point_count);

        // let voxel_size_inv = 1.0 / voxel_size;
        // let offset = 2i32.pow(20); // A large offset to handle negative coordinates

        // morton
        let mut keyed_points: Vec<(i32, Point)> = self
            .as_merged_cloud()
            .iter() // TODO check or iter?
            .map(|point| {
                // let ix = (point.0[0] * voxel_size_inv).floor() as i32 + offset;
                // let iy = (point.0[1] * voxel_size_inv).floor() as i32 + offset;
                // let iz = (point.0[2] * voxel_size_inv).floor() as i32 + offset;

                // let key = morton_encoding::morton_encode([ix as u32, iy as u32, iz as u32]);
                let key = Self::hash(
                    &nalgebra::Point3::new(point.0[0], point.0[1], point.0[2]),
                    self.voxel_resolution,
                );
                (key, point.clone())
            })
            .collect();

        // After this, all points in the same voxel are adjacent
        keyed_points.par_sort_unstable_by_key(|(key, _)| *key);

        if keyed_points.is_empty() {
            panic!("empty map impossible cloud");
        }

        self.map.clear();

        let mut group_start_index = 0;
        for i in 1..keyed_points.len() {
            // Check if the key has changed. If so, the previous group has ended.
            if keyed_points[i].0 != keyed_points[group_start_index].0 {
                // Process the slice for the completed group.
                let group_slice = &keyed_points[group_start_index..i];
                let dist = build_voxel_distribution(&group_slice).unwrap();

                let group_key = keyed_points[group_start_index].0;
                let representative_point = &group_slice[0].1;

                let point = nalgebra::Point3::new(
                    representative_point.0[0],
                    representative_point.0[1],
                    representative_point.0[2],
                );
                let xyz = point * (1.0 / self.voxel_resolution);

                let x = xyz.x.round() as i32;
                let y = xyz.y.round() as i32;
                let z = xyz.z.round() as i32;

                self.map.insert(group_key, ((x, y, z), dist));

                // Start a new group
                group_start_index = i;
            }
        }
        // The last group
        let last_group_slice = &keyed_points[group_start_index..];
        let dist = build_voxel_distribution(&last_group_slice).unwrap();
        let representative_point = &last_group_slice[0].1;
        let point = nalgebra::Point3::new(
            representative_point.0[0],
            representative_point.0[1],
            representative_point.0[2],
        );
        let xyz = point * (1.0 / self.voxel_resolution);

        let x = xyz.x.round() as i32;
        let y = xyz.y.round() as i32;
        let z = xyz.z.round() as i32;
        self.map.insert(last_group_slice[0].0, ((x, y, z), dist));
    }

    // --- Helper Methods ---

    /// Gets the number of points stored in a specific slot of the ring buffer.
    fn get_actual_pointcloud_size(&self, idx: usize) -> u32 {
        let float_idx = idx * (self.max_points_per_scan * FLOATS_PER_POINT)
            + ((self.max_points_per_scan * FLOATS_PER_POINT) - 1);
        let float_val = self.point_clouds[float_idx];
        // Reinterpret the bits of the float as a u32.
        u32::from_ne_bytes(float_val.to_ne_bytes())
    }

    /// Sets the number of points for a specific slot, storing it in the last float.
    fn set_actual_pointcloud_size(&mut self, idx: usize, size: u32) {
        let float_idx = idx * (self.max_points_per_scan * FLOATS_PER_POINT)
            + ((self.max_points_per_scan * FLOATS_PER_POINT) - 1);
        // Reinterpret the bits of the u32 as a float to store it.
        self.point_clouds[float_idx] = f32::from_ne_bytes(size.to_ne_bytes());
    }

    /// Calculates the next index to use in the ring buffer, handling wrapping.
    fn get_next_inactive_idx(&mut self) -> usize {
        let free_idx = self.next_inactive_idx;

        self.next_inactive_idx += 1;
        if self.next_inactive_idx == self.ringbuffer_size {
            self.next_inactive_idx = 0;
            self.was_full = true;
        }

        // Clear the size of the slot we are about to overwrite.
        self.set_actual_pointcloud_size(self.next_inactive_idx, 0);

        free_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_localmap_workflow() {
        let mut localmap = Localmap::new(1024, 10, 0.5);

        // 2. ACTION: Add the first point cloud
        let points1_count = 10;
        let points1 = vec![Point([1.0, 2.0, 3.0, 1.0]); points1_count];
        localmap.copy_to_next_buffer(&points1);

        // Define its transformation (translate by +10 on X)
        let transform1 = Isometry3::translation(10.0, 0.0, 0.0);

        localmap.commit_and_merge(transform1);

        // 3. ASSERT: Check the state after the first merge
        assert_eq!(
            localmap.total_dense_point_count, points1_count,
            "Point count should be 10 after first merge"
        );

        let merged_cloud1 = localmap.as_merged_cloud();
        assert_eq!(
            merged_cloud1.len(),
            points1_count,
            "Dense buffer size should be 10 points * 4 floats"
        );

        // Check the first point of the transformed cloud
        let first_point_cloud1 = merged_cloud1[0];
        // Original was (1, 2, 3), transform was (+10, 0, 0) -> expected (11, 2, 3)
        assert_eq!(
            first_point_cloud1,
            Point([11.0, 2.0, 3.0, 1.0]),
            "First point transformation is incorrect"
        );

        // 4. ACTION: Add a second point cloud
        let points2_count = 20;
        let points2 = vec![Point([5.0, 6.0, 7.0, 1.0]); points2_count];
        localmap.copy_to_next_buffer(&points2);

        // Define its transformation (translate by -20 on Y)
        let transform2 = Isometry3::translation(0.0, -20.0, 0.0);

        // Commit and merge. This will re-process cloud 1 and add cloud 2.
        localmap.commit_and_merge(transform2);

        // 5. ASSERT: Check the state after the second merge
        let total_expected_points = points1_count + points2_count;
        assert_eq!(
            localmap.total_dense_point_count, total_expected_points,
            "Total point count should be 30 after second merge"
        );

        let merged_cloud2 = localmap.as_merged_cloud();
        assert_eq!(
            merged_cloud2.len(),
            total_expected_points,
            "Dense buffer size should be 30 points * 4 floats"
        );

        // Check the first point of the first cloud (should still be the same)
        let first_point_cloud1_again = merged_cloud2[0];
        assert_eq!(
            first_point_cloud1_again,
            Point([11.0, 2.0, 3.0, 1.0]),
            "First cloud's point is incorrect after second merge"
        );

        // Check the first point of the second cloud. It starts after all the points of the first cloud.
        let first_point_cloud2 = merged_cloud2[points1_count];
        // Original was (5, 6, 7), transform was (0, -20, 0) -> expected (5, -14, 7)
        assert_eq!(
            first_point_cloud2,
            Point([5.0, -14.0, 7.0, 1.0]),
            "Second cloud's first point transformation is incorrect"
        );
    }
}

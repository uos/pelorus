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
    pub point_count: u64,
    pub mean: Point3<f32>,
    pub covariance: Matrix3<f32>,
}

impl VoxelDistribution {
    pub fn new(first_point: &Point3<f32>) -> Self {
        VoxelDistribution {
            point_count: 1,
            mean: *first_point,
            covariance: Matrix3::zeros(),
        }
    }
    pub fn update(
        &mut self,
        new_point: &Point3<f32>,
        // A tuning parameter. Larger values make the distribution more "skeptical" of new points.
        sensitivity: f32,
        min_dist: f32,
    ) {
        // Cannot calculate Mahalanobis distance if covariance is singular (e.g., only 1 point)
        // In this case, we fall back to a simpler, unweighted update.
        if self.point_count < 2 {
            let new_point_dist_to_mean = (self.mean - new_point).norm_squared();
            if new_point_dist_to_mean > min_dist * min_dist {
                let n = (self.point_count + 1) as f64;
                let old_mean = self.mean;
                let diff = new_point - old_mean;
                self.mean = old_mean + diff / n as f32;

                // Welford's algorithm step for Covariance
                let new_diff = new_point - self.mean;
                self.covariance = self.covariance + &diff * new_diff.transpose();
                self.point_count += 1;
            }
            return;
        }

        // dbg!("updating using mahalonobus");

        // --- Weighting Logic using Mahalanobis Distance ---
        let diff = new_point - self.mean;
        let distance_sq = if let Some(chol) = Cholesky::new(self.covariance.clone()) {
            let mut y = diff.clone();
            chol.solve_mut(&mut y);
            diff.dot(&y)
        } else {
            // Fallback only if Cholesky decomposition itself fails.
            nalgebra::distance_squared(&self.mean, new_point)
        };

        // The learning rate `alpha` is determined by the Mahalanobis distance.
        // A point that fits the geometry well (low distance) gets a high alpha.
        let alpha = (1.0 / sensitivity) * (-distance_sq / 2.0).exp();

        // --- Update Formulas ---
        self.point_count += 1;
        let old_mean = self.mean;

        // Update the mean using the weighted learning rate 'alpha'
        self.mean = old_mean + alpha * (new_point - old_mean);

        // Update the covariance using the same weighted scheme
        let new_cov_sample = &diff * &diff.transpose();
        self.covariance = (1.0 - alpha) * self.covariance + alpha * new_cov_sample;
    }
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
// pub fn build_voxel_distribution(points: &[(i32, Point)]) -> Option<VoxelDistribution> {
//     if points.is_empty() {
//         return None;
//     }

//     let n = points.len() as f64;

//     // Calculate the mean (centroid)
//     let sum_of_points = points.iter().fold(Point3::origin(), |acc, p| {
//         acc + Vector3::new(p.1.0[0], p.1.0[1], p.1.0[2])
//     });
//     let mean = Point3::from(sum_of_points.coords / n as f32);

//     // Calculate the covariance matrix
//     let mut covariance = Matrix3::zeros();
//     for point in points {
//         let point = Point3::new(point.1.0[0], point.1.0[1], point.1.0[2]);
//         let diff = point - mean;
//         covariance += &diff * &diff.transpose();
//     }
//     covariance /= n as f32;

//     // We need to be able to sample from a multivariate normal distribution.
//     // This can be done by sampling from a standard normal distribution and
//     // transforming the samples using the Cholesky decomposition of the covariance matrix.
//     let cholesky = match Cholesky::new(covariance) {
//         Some(chol) => chol.l(),
//         // If the covariance matrix is not positive definite, we can fall back
//         // to a diagonal matrix (no correlation between axes).
//         None => {
//             let eigen = SymmetricEigen::new(covariance);
//             Matrix3::from_diagonal(&eigen.eigenvalues.map(|e| e.max(0.0).sqrt()))
//         }
//     };

//     let dist = VoxelDistribution {
//         mean,
//         covariance,
//         cholesky,
//     };
//     Some(dist)
// }

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
// pub fn sample_from_distribution<R: Rng + ?Sized>(
//     distribution: &VoxelDistribution,
//     num_samples: usize,
//     rng: &mut R,
// ) -> Vec<Point3<f32>> {
//     let standard_normal = Normal::new(0.0, 1.0).unwrap();
//     let mut samples = Vec::with_capacity(num_samples);

//     for _ in 0..num_samples {
//         let rand_vec = Vector3::new(
//             standard_normal.sample(rng),
//             standard_normal.sample(rng),
//             standard_normal.sample(rng),
//         );
//         let transformed_vec = distribution.cholesky * rand_vec;
//         samples.push(distribution.mean + transformed_vec);
//     }

//     samples
// }

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

    pub map: HashMap<i32, VoxelDistribution>,
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

    // pub fn add_to_map(&mut self, points: &[PointXYZ], transform: Isometry3<f64>) {
    //     // take to global points

    //     points.iter().map(|p| {
    //         let p_global = transform * p.xyz_f64();

    //     });
    // }

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

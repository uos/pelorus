use nalgebra::{Isometry3, Point3};
use rayon::prelude::*;

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
    transforms: Vec<Isometry3<f32>>,

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
}

impl Localmap {
    pub fn new(max_points_per_scan: usize, ringbuffer_size: usize) -> Self {
        assert_eq!(std::mem::size_of::<f32>(), std::mem::size_of::<u32>());

        let mut map = Self {
            point_clouds: vec![0.0; max_points_per_scan * FLOATS_PER_POINT * ringbuffer_size],
            dense_buffer: vec![
                0.0;
                ((max_points_per_scan * FLOATS_PER_POINT) - 1) * ringbuffer_size
            ],
            transforms: vec![Isometry3::identity(); ringbuffer_size],
            next_inactive_idx: 0,
            current_pcl_idx: 0,
            was_full: false,
            total_dense_point_count: 0,
            max_points_per_scan,
            ringbuffer_size,
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
    pub fn next_register_buffer(&mut self) -> &mut [f32] {
        let idx = self.get_next_inactive_idx();
        self.current_pcl_idx = idx;

        let start = idx * self.max_points_per_scan * FLOATS_PER_POINT;
        let end = start + (self.max_points_per_scan * FLOATS_PER_POINT);
        &mut self.point_clouds[start..end]
    }

    pub fn copy_to_next_buffer(&mut self, src_points: &[Point]) {
        let num_points = src_points.len();
        let dest_slice = self.next_register_buffer();

        // Unsafe is used here for a highly efficient reinterpret_cast.
        // It is safe because we know Point is repr(C) and has the same layout as [f32; 4].
        let src_floats: &[f32] = unsafe {
            std::slice::from_raw_parts(
                src_points.as_ptr() as *const f32,
                num_points * FLOATS_PER_POINT,
            )
        };

        dest_slice[..src_floats.len()].copy_from_slice(src_floats);
        self.set_actual_pointcloud_size(self.current_pcl_idx, num_points as u32);
    }

    pub fn commit_and_merge(&mut self, latest_transform: Isometry3<f32>) {
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
                    let pos = Point3::new(src_point[0], src_point[1], src_point[2]);

                    let transformed_pos = transform * pos;

                    dest_point[0] = transformed_pos.x;
                    dest_point[1] = transformed_pos.y;
                    dest_point[2] = transformed_pos.z;
                    dest_point[3] = 1.0;
                });

            dense_offset += pcl_size * FLOATS_PER_POINT;
            total_points += pcl_size;
        }

        self.total_dense_point_count = total_points;

        // In the original code, `BuildMap` was called on a `CUDACloudHashMap`.
        // The CPU equivalent would be to build a spatial data structure (like a k-d tree or hash grid) here.
        // self.build_cpu_spatial_map();
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
        let mut localmap = Localmap::new(1024, 10);

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

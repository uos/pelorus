#![cfg(vulkan_available)]

pub use types::{Matrix3, NeighborType, Point, PointNormal, Pose, Quaternion, Vector3};

mod buffer;
mod compute_session;
mod context;
mod downsample;
mod map;
mod multiresolution;
mod pipelines;
mod planes;
mod shaders;
mod sorter;
mod stream;
mod types;

#[cfg(test)]
mod tests;

pub use buffer::{VulkanBuffer, get_used_memory};
pub use context::{
    VulkanMetadata, VulkanVersion, get_vulkan_metadata, is_available, is_multiresolution_available,
};
pub use downsample::{VulkanDownsampleTimings, downsample_points_vulkan};
pub use map::{
    add_to_map, cleanup_map, get_map, get_map_capacity, get_map_size, get_map_with_recency,
    init_map, init_map_with_mode, session_release, session_retain, shutdown,
};
pub use planes::{VulkanPlaneReadback, VulkanPlaneTimings, planes_vulkan, try_get_gpu_timings};
pub use stream::Stream;

/**
 * @file pelorus_pcl.hpp
 * @brief PCL (Point Cloud Library) integration helpers for Pelorus
 * 
 * This header provides convenience functions to convert between PCL point clouds
 * and Pelorus point cloud formats.
 * 
 * Usage:
 *   #include <pelorus.hpp>
 *   #include <pelorus_pcl.hpp>
 *   
 *   pcl::PointCloud<pelorus::pcl::PointXYZIT>::Ptr cloud = ...;
 *   pelorus::Pelorus pelorus;
 *   pelorus::pcl::add_pointcloud(pelorus, cloud, timestamp_sec, timestamp_nsec, "lidar");
 */

#ifndef PELORUS_PCL_HPP
#define PELORUS_PCL_HPP

#include "pelorus.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <cstring>

namespace pelorus {
namespace pcl {

/**
 * @brief Convert per-point timestamp offset (ns) to the float representation expected by the C-API helpers.
 *
 * The Pelorus C-API `pelorus_add_points_*_f32` interprets the `t` field as nanoseconds offset and casts
 * the provided float to `uint32_t`. Therefore we store the numeric value in the float (not bit-punning).
 */
inline float timestamp_to_float(uint32_t t_ns) {
    return static_cast<float>(t_ns);
}

// Intentionally no overloads for PointXYZ / PointXYZI.
// Per-point timestamps are required to add point clouds.

} // namespace pcl
} // namespace pelorus

// Define the custom point type in the ::pcl namespace so that PCL macros
// (which refer to pcl::Vector*Map types) resolve to the correct namespace.
namespace pcl {

/**
 * @brief Custom PCL point type with timestamp field
 */
struct EIGEN_ALIGN16 PointXYZIT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;  // Timestamp in nanoseconds (offset from header)
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace pcl

// Keep the public API name pelorus::pcl::PointXYZIT as a convenient alias.
namespace pelorus {
namespace pcl {
using PointXYZIT = ::pcl::PointXYZIT;
} // namespace pcl
} // namespace pelorus

// PCL point type registration must be at global scope (required by PCL macros)
POINT_CLOUD_REGISTER_POINT_STRUCT(pcl::PointXYZIT,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (uint32_t, t, t)
)

namespace pelorus {
namespace pcl {

/**
 * @brief Convert PCL PointXYZIT cloud (with per-point timestamps) to Pelorus format
 *
 * Per-point timestamps are required. Do not call Pelorus with point clouds that lack a `t` field.
 * 
 * @param pelorus Pelorus instance
 * @param cloud PCL point cloud with per-point timestamps
 * @param timestamp_sec Header timestamp seconds
 * @param timestamp_nanosec Header timestamp nanoseconds
 * @param frame_id Frame ID string
 */
inline void add_pointcloud(Pelorus& pelorus,
                          const ::pcl::PointCloud<::pcl::PointXYZIT>::Ptr& cloud,
                          int32_t timestamp_sec, uint32_t timestamp_nanosec,
                          const char* frame_id) {
    std::vector<float> points(cloud->size() * 5);
    
    for (size_t i = 0; i < cloud->size(); i++) {
        const auto& pt = cloud->points[i];
        points[i * 5 + 0] = pt.x;
        points[i * 5 + 1] = pt.y;
        points[i * 5 + 2] = pt.z;
        points[i * 5 + 3] = pt.intensity;
        
        // Use actual per-point timestamp (ns offset)
        points[i * 5 + 4] = static_cast<float>(pt.t);
    }
    
    pelorus.add_points_xyzit(points.data(), cloud->size(),
                            timestamp_sec, timestamp_nanosec, frame_id);
}

/**
 * @brief Convert point cloud from Pelorus odometry to PCL PointXYZI cloud
 * 
 * @param points Pointer to Pelorus point array
 * @param num_points Number of points
 * @return PCL point cloud
 */
inline ::pcl::PointCloud<::pcl::PointXYZI>::Ptr to_pcl_cloud(
    const pelorus_point_xyzi* points, size_t num_points) {
    auto cloud = ::pcl::PointCloud<::pcl::PointXYZI>::Ptr(
        new ::pcl::PointCloud<::pcl::PointXYZI>()
    );
    
    if (!points || num_points == 0) {
        return cloud;
    }
    
    cloud->points.resize(num_points);
    cloud->width = num_points;
    cloud->height = 1;
    cloud->is_dense = false;
    
    for (size_t i = 0; i < num_points; i++) {
        cloud->points[i].x = points[i].x;
        cloud->points[i].y = points[i].y;
        cloud->points[i].z = points[i].z;
        cloud->points[i].intensity = points[i].intensity;
    }
    
    return cloud;
}

} // namespace pcl
} // namespace pelorus

// Implement the template method for Odometry::get_cloud_pcl()
namespace pelorus {

template<typename PointCloudPtr>
PointCloudPtr Odometry::get_cloud_pcl() const {
    if (!valid_) {
        return PointCloudPtr(new typename PointCloudPtr::element_type());
    }
    return pelorus::pcl::to_pcl_cloud(odom_.points, odom_.num_points);
}

} // namespace pelorus

#endif // PELORUS_PCL_HPP

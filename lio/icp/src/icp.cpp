#include "../include/icp.hpp"

#include "../small_gicp/include/small_gicp/registration/registration_helper.hpp"

#include <tbb/tbb.h>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/util/normal_estimation_tbb.hpp>
#include <small_gicp/registration/reduction_tbb.hpp>
#include <small_gicp/registration/registration.hpp>

#include <iostream>

using namespace small_gicp;

typedef struct
{
    std::shared_ptr<GaussianVoxelMap> voxelmap;
    Eigen::Isometry3d global_transform;
} State;

typedef struct
{
    float x, y, z, padding;
} Point;

void *init_map(double voxel_resolution)
{
    auto voxelmap = std::make_shared<GaussianVoxelMap>(voxel_resolution);
    State *state = new State();
    state->voxelmap = voxelmap;
    return state;
}

void add_to_map(void *map, const void *point_buffer, int num_points, LioIsometry3d global_transform)
{
    State *state = reinterpret_cast<State *>(map);
    auto voxelmap = state->voxelmap;

    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.translation() << global_transform.x, global_transform.y, global_transform.z;
    Eigen::Quaterniond q(global_transform.q_w, global_transform.q_i, global_transform.q_j, global_transform.q_k);
    transform.rotate(q);

    auto points_ptr = reinterpret_cast<const Point *>(point_buffer);
    PointCloud::Ptr cloud = std::make_shared<PointCloud>();
    cloud->resize(num_points);
    for (int i = 0; i < num_points; i++)
    {
        cloud->point(i) << points_ptr[i].x, points_ptr[i].y, points_ptr[i].z, 1.0;
    }

    state->voxelmap->insert(*cloud, transform);
}

// translation: xyz
// rotation: quaternion xyzw
void align(void *map, const void *point_buffer, int num_points, LioIsometry3d init_guess, LioIsometry3d *out)
{
    State *state = reinterpret_cast<State *>(map);
    auto voxelmap = state->voxelmap;

    auto points_ptr = reinterpret_cast<const Point *>(point_buffer);

    PointCloud::Ptr cloud = std::make_shared<PointCloud>();
    cloud->resize(num_points);
    for (int i = 0; i < num_points; i++)
    {
        cloud->point(i) << points_ptr[i].x, points_ptr[i].y, points_ptr[i].z, 1.0;
    }

    Eigen::Isometry3d init_guess_eigen = Eigen::Isometry3d::Identity();
    init_guess_eigen.translation() << init_guess.x, init_guess.y, init_guess.z;
    Eigen::Quaterniond q(init_guess.q_w, init_guess.q_i, init_guess.q_j, init_guess.q_k);
    init_guess_eigen.rotate(q);

    Registration<GICPFactor, ParallelReductionTBB> registration;
    auto result = registration.align(*voxelmap, *cloud, *voxelmap, init_guess_eigen);

    out->x = result.T_target_source.translation().x();
    out->y = result.T_target_source.translation().y();
    out->z = result.T_target_source.translation().z();

    Eigen::Quaterniond q2(result.T_target_source.rotation());
    out->q_i = q2.x();
    out->q_j = q2.y();
    out->q_k = q2.z();
    out->q_w = q2.w();
}

void drop_map(void *map)
{
    State *state = reinterpret_cast<State *>(map);
    delete state;
    state = nullptr;
}

#include "../include/icp.hpp"

#include "../small_gicp/include/small_gicp/registration/registration_helper.hpp"

#include <tbb/tbb.h>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/util/normal_estimation_tbb.hpp>
#include <small_gicp/registration/reduction_tbb.hpp>
#include <small_gicp/registration/registration.hpp>

using namespace small_gicp;

typedef struct {
    std::shared_ptr<GaussianVoxelMap> voxelmap;
    Eigen::Isometry3d global_transform;
} State;

void* init_map(double voxel_resolution) {
    auto voxelmap = std::make_shared<GaussianVoxelMap>(voxel_resolution);
    State* state = new State();
    state->voxelmap = voxelmap;
    return state;
}

void add_to_map(void* map, void* point_buffer, int num_points, double* global_translate, double* global_rotate) {
    State* state = static_cast<State*>(map);
    auto voxelmap = state->voxelmap;

    if (global_translate != nullptr && global_rotate != nullptr) {
        Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
        transform.translation() = Eigen::Vector3d(global_translate[0], global_translate[1], global_translate[2]);
        Eigen::Quaterniond q(global_rotate[3], global_rotate[0], global_rotate[1], global_rotate[2]);
        transform.rotate(q);
        state->global_transform = transform;
    }

    Eigen::Vector3f* points_ptr = static_cast<Eigen::Vector3f*>(point_buffer);
    
    PointCloud::Ptr cloud = std::make_shared<PointCloud>();
    cloud->resize(num_points);
    for (int i = 0; i < num_points; i++) {
        cloud->point(i) << points_ptr[i].template cast<double>().template head<3>(), 1.0;
    }

    state->voxelmap->insert(*cloud, state->global_transform);
}

// translation: xyz
// rotation: quaternion xyzw
void align(void* map, void* point_buffer, int num_points, double* translation, double* rotation) {
    State* state = static_cast<State*>(map);
    auto voxelmap = state->voxelmap;

    Eigen::Vector3f* points_ptr = static_cast<Eigen::Vector3f*>(point_buffer);

    PointCloud::Ptr cloud = std::make_shared<PointCloud>();
    cloud->resize(num_points);
    for (int i = 0; i < num_points; i++) {
        cloud->point(i) << points_ptr[i].template cast<double>().template head<3>(), 1.0;
    }

    Registration<GICPFactor, ParallelReductionTBB> registration;
    auto result = registration.align(*voxelmap, *cloud, *voxelmap, state->global_transform);

    translation[0] = result.T_target_source.translation().x();
    translation[1] = result.T_target_source.translation().y();
    translation[2] = result.T_target_source.translation().z();

    Eigen::Quaterniond q(result.T_target_source.rotation());
    rotation[0] = q.x();
    rotation[1] = q.y();
    rotation[2] = q.z();
    rotation[3] = q.w();
}

void drop_map(void* map) {
    State* state = static_cast<State*>(map);
    delete state;
    state = nullptr;
}


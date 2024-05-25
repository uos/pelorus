#include <small_gicp/registration/registration_helper.hpp>

std::vector<Eigen::Vector3d> target_points = ...;   // Any of Eigen::Vector(3|4)(f|d) can be used
std::vector<Eigen::Vector3d> source_points = ...;   // 

RegistrationSetting setting;
setting.num_threads = 4;                    // Number of threads to be used
setting.downsampling_resolution = 0.25;     // Downsampling resolution
setting.max_correspondence_distance = 1.0;  // Maximum correspondence distance between points (e.g., triming threshold)

Eigen::Isometry3d init_T_target_source = Eigen::Isometry3d::Identity();
RegistrationResult result = align(target_points, source_points, init_T_target_source, setting);

Eigen::Isometry3d T = result.T_target_source;  // Estimated transformation
size_t num_inliers = result.num_inliers;       // Number of inlier source points
Eigen::Matrix<double, 6, 6> H = result.H;      // Final Hessian matrix (6x6)
/**
 * @file pelorus_eigen.hpp
 * @brief Eigen3 integration helpers for Pelorus
 * 
 * This header provides convenience functions to convert Pelorus odometry
 * to Eigen transformation matrices and other Eigen types.
 * 
 * Usage:
 *   #include <pelorus.hpp>
 *   #include <pelorus_eigen.hpp>
 *   
 *   pelorus::Pelorus pelorus;
 *   auto odom = pelorus.recv();
 *   Eigen::Isometry3d transform = pelorus::eigen::to_transform(odom);
 */

#ifndef PELORUS_EIGEN_HPP
#define PELORUS_EIGEN_HPP

#include "pelorus.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace pelorus {
namespace eigen {

/**
 * @brief Convert quaternion to Eigen::Quaterniond
 * 
 * @param quat Quaternion array [w, x, y, z]
 * @return Eigen quaternion
 */
inline Eigen::Quaterniond to_quaternion(const double* quat) {
    // Pelorus uses [w, x, y, z] order
    return Eigen::Quaterniond(quat[0], quat[1], quat[2], quat[3]);
}

/**
 * @brief Convert position to Eigen::Vector3d
 * 
 * @param pos Position array [x, y, z]
 * @return Eigen vector
 */
inline Eigen::Vector3d to_position(const double* pos) {
    return Eigen::Vector3d(pos[0], pos[1], pos[2]);
}

/**
 * @brief Convert linear velocity to Eigen::Vector3d
 * 
 * @param vel Velocity array [vx, vy, vz]
 * @return Eigen vector
 */
inline Eigen::Vector3d to_linear_velocity(const double* vel) {
    return Eigen::Vector3d(vel[0], vel[1], vel[2]);
}

/**
 * @brief Convert angular velocity to Eigen::Vector3d
 * 
 * @param vel Angular velocity array [wx, wy, wz]
 * @return Eigen vector
 */
inline Eigen::Vector3d to_angular_velocity(const double* vel) {
    return Eigen::Vector3d(vel[0], vel[1], vel[2]);
}

/**
 * @brief Convert Pelorus odometry to Eigen::Isometry3d transformation
 * 
 * This creates a 4x4 homogeneous transformation matrix from the position
 * and orientation in the odometry result.
 * 
 * @param odom Pelorus odometry result
 * @return Isometry3d transformation (identity if invalid)
 */
inline Eigen::Isometry3d to_transform(const Odometry& odom) {
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    
    if (!odom.valid()) {
        return transform;
    }
    
    transform.translation() = to_position(odom.position());
    transform.linear() = to_quaternion(odom.orientation()).toRotationMatrix();
    
    return transform;
}

/**
 * @brief Convert Pelorus odometry to Eigen::Affine3d transformation
 * 
 * Similar to to_transform() but returns Affine3d which is more general.
 * 
 * @param odom Pelorus odometry result
 * @return Affine3d transformation (identity if invalid)
 */
inline Eigen::Affine3d to_affine(const Odometry& odom) {
    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    
    if (!odom.valid()) {
        return transform;
    }
    
    transform.translation() = to_position(odom.position());
    transform.linear() = to_quaternion(odom.orientation()).toRotationMatrix();
    
    return transform;
}

/**
 * @brief Convert Pelorus odometry to 4x4 transformation matrix
 * 
 * @param odom Pelorus odometry result
 * @return 4x4 transformation matrix (identity if invalid)
 */
inline Eigen::Matrix4d to_matrix(const Odometry& odom) {
    return to_transform(odom).matrix();
}

/**
 * @brief Extract pose covariance as Eigen matrix (6x6)
 * 
 * @param odom Pelorus odometry result
 * @return 6x6 covariance matrix
 */
inline Eigen::Matrix<double, 6, 6> get_pose_covariance(const pelorus_odometry& odom) {
    Eigen::Matrix<double, 6, 6> cov;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            cov(i, j) = odom.pose_covariance[i * 6 + j];
        }
    }
    return cov;
}

/**
 * @brief Extract twist covariance as Eigen matrix (6x6)
 * 
 * @param odom Pelorus odometry result
 * @return 6x6 covariance matrix
 */
inline Eigen::Matrix<double, 6, 6> get_twist_covariance(const pelorus_odometry& odom) {
    Eigen::Matrix<double, 6, 6> cov;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            cov(i, j) = odom.twist_covariance[i * 6 + j];
        }
    }
    return cov;
}

/**
 * @brief Complete odometry state as Eigen types
 * 
 * Convenient struct holding all odometry information in Eigen format.
 */
struct OdometryState {
    Eigen::Isometry3d transform;           ///< Pose as transformation matrix
    Eigen::Vector3d linear_velocity;        ///< Linear velocity
    Eigen::Vector3d angular_velocity;       ///< Angular velocity
    Eigen::Matrix<double, 6, 6> pose_cov;  ///< Pose covariance
    Eigen::Matrix<double, 6, 6> twist_cov; ///< Twist covariance
    int32_t timestamp_sec;                  ///< Timestamp seconds
    uint32_t timestamp_nanosec;             ///< Timestamp nanoseconds
    
    /**
     * @brief Create OdometryState from Pelorus odometry
     */
    static OdometryState from_pelorus(const Odometry& odom) {
        OdometryState state;
        
        if (!odom.valid()) {
            state.transform = Eigen::Isometry3d::Identity();
            state.linear_velocity = Eigen::Vector3d::Zero();
            state.angular_velocity = Eigen::Vector3d::Zero();
            state.pose_cov = Eigen::Matrix<double, 6, 6>::Zero();
            state.twist_cov = Eigen::Matrix<double, 6, 6>::Zero();
            state.timestamp_sec = 0;
            state.timestamp_nanosec = 0;
            return state;
        }
        
        state.transform = to_transform(odom);
        state.linear_velocity = to_linear_velocity(odom.linear_velocity());
        state.angular_velocity = to_angular_velocity(odom.angular_velocity());
        state.pose_cov = get_pose_covariance(*odom.get());
        state.twist_cov = get_twist_covariance(*odom.get());
        state.timestamp_sec = odom.timestamp_sec();
        state.timestamp_nanosec = odom.timestamp_nanosec();
        
        return state;
    }
};

} // namespace eigen
} // namespace pelorus

#endif // PELORUS_EIGEN_HPP

/* Pelorus C Library - Complete LiDAR-Inertial Odometry System
 *
 * Includes:
 * - Point cloud preprocessing and filtering
 * - IMU integration and filtering
 * - Point cloud undistortion
 * - Kalman filter (EKF) for state estimation
 * - Point cloud registration and mapping
 *
 * Threading and async processing are handled internally.
 * The API is simple and synchronous from the caller's perspective.
 */

#ifndef PELORUS_H
#define PELORUS_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief Opaque handle to a Pelorus LIO instance.
 * Must be created with pelorus_create() and destroyed with pelorus_destroy().
 */
typedef struct PelorusHandle PelorusHandle;

/**
 * @brief IMU sensor message.
 */
typedef struct CImuMsg {
    int32_t timestamp_sec;                    /**< Timestamp seconds since epoch */
    uint32_t timestamp_nanosec;               /**< Timestamp nanoseconds part */
    const char *frame_id;                     /**< Frame ID (sensor name) */
    double orientation[4];                    /**< Quaternion [w, x, y, z] */
    double orientation_covariance[9];         /**< 3x3 covariance matrix (row-major) */
    double angular_velocity[3];               /**< Angular velocity [x, y, z] rad/s */
    double angular_velocity_covariance[9];    /**< 3x3 covariance matrix (row-major) */
    double linear_acceleration[3];            /**< Linear acceleration [x, y, z] m/s^2 */
    double linear_acceleration_covariance[9]; /**< 3x3 covariance matrix (row-major) */
} CImuMsg;

/**
 * @brief Odometry message with twist (velocity).
 */
typedef struct COdomMsg {
    int32_t timestamp_sec;       /**< Timestamp seconds since epoch */
    uint32_t timestamp_nanosec;  /**< Timestamp nanoseconds part */
    double twist_linear[3];      /**< Linear velocity [vx, vy, vz] m/s */
    double twist_angular[3];     /**< Angular velocity [wx, wy, wz] rad/s */
    double twist_covariance[36]; /**< 6x6 covariance matrix (row-major) */
} COdomMsg;

/**
 * @brief Point cloud header (legacy; prefer CPointCloud2).
 */
typedef struct CPointCloudHeader {
    int32_t timestamp_sec;      /**< Timestamp seconds since epoch */
    uint32_t timestamp_nanosec; /**< Timestamp nanoseconds part */
    const char *frame_id;       /**< Frame ID (sensor name) */
} CPointCloudHeader;

/**
 * @brief Point field descriptor for PointCloud2.
 */
typedef struct CPointField {
    const char *name; /**< Field name (e.g., "x", "y", "z", "intensity") */
    uint32_t offset;  /**< Byte offset from start of point data */
    uint8_t datatype; /**< Data type (7=FLOAT32, 8=FLOAT64, etc.) */
    uint32_t count;   /**< Number of elements (usually 1) */
} CPointField;

/**
 * @brief ROS2 PointCloud2 message.
 */
typedef struct CPointCloud2 {
    int32_t header_timestamp_sec;      /**< Timestamp seconds since epoch */
    uint32_t header_timestamp_nanosec; /**< Timestamp nanoseconds part */
    const char *header_frame_id;       /**< Frame ID (sensor name) */
    uint32_t height;                   /**< Number of rows (1 for unorganized) */
    uint32_t width;                    /**< Number of points per row */
    const CPointField *fields;         /**< Array of field descriptors */
    uintptr_t fields_len;              /**< Number of fields */
    bool is_bigendian;                 /**< Byte order (usually false) */
    uint32_t point_step;               /**< Bytes per point */
    uint32_t row_step;                 /**< Bytes per row */
    const uint8_t *data;               /**< Raw point data buffer */
    uintptr_t data_len;                /**< Length of data buffer in bytes */
    bool is_dense;                     /**< True if no invalid points */
} CPointCloud2;

/**
 * @brief Simple XYZI point structure (used in outputs).
 */
typedef struct CPointXYZI {
    float x;         /**< X coordinate in meters */
    float y;         /**< Y coordinate in meters */
    float z;         /**< Z coordinate in meters */
    float intensity; /**< Intensity/reflectivity value */
} CPointXYZI;

/**
 * @brief Complete odometry output from Pelorus.
 *
 * Memory for the points array is allocated by Pelorus and must be freed using
 * pelorus_free_odometry() after use.
 */
typedef struct CPelorusOdometry {
    int32_t timestamp_sec;       /**< Timestamp seconds since epoch */
    uint32_t timestamp_nanosec;  /**< Timestamp nanoseconds part */
    double position[3];          /**< Position [x, y, z] in meters */
    double orientation[4];       /**< Quaternion [w, x, y, z] */
    double pose_covariance[36];  /**< 6x6 pose covariance (row-major) */
    double linear_velocity[3];   /**< Linear velocity [vx, vy, vz] m/s */
    double angular_velocity[3];  /**< Angular velocity [wx, wy, wz] rad/s */
    double twist_covariance[36]; /**< 6x6 twist covariance (row-major) */
    CPointXYZI *points;          /**< Registered point cloud (must free) */
    uintptr_t num_points;        /**< Number of points in cloud */
} CPelorusOdometry;

/**
 * @brief Create a new Pelorus LIO instance.
 * @param config_path Path to .rl configuration file, or NULL for defaults.
 * @return Opaque handle to Pelorus instance, or NULL on error.
 */
PelorusHandle *pelorus_create(const char *config_path);

/**
 * @brief Add an IMU measurement to the pipeline.
 * @return 0 on success, -1 on error.
 */
int32_t pelorus_add_imu(PelorusHandle *handle, const CImuMsg *imu);

/**
 * @brief Add an odometry measurement to the pipeline.
 * @return 0 on success, -1 on error.
 */
int32_t pelorus_add_odom(PelorusHandle *handle, const COdomMsg *odom);

/**
 * @brief Add a point cloud to the pipeline.
 * @return 0 on success, -1 on error.
 */
int32_t pelorus_add_pointcloud(PelorusHandle *handle, const CPointCloud2 *cloud);

/**
 * @brief Try to receive odometry output (non-blocking).
 * @return 1 if odometry received, 0 if no data, -1 on error.
 */
int32_t pelorus_try_recv(PelorusHandle *handle, CPelorusOdometry *result);

/**
 * @brief Free memory allocated for odometry result.
 */
void pelorus_free_odometry(CPelorusOdometry *odom);

/**
 * @brief Destroy a Pelorus instance and free all resources.
 */
void pelorus_destroy(PelorusHandle *handle);

/**
 * @brief Get the library version string (static storage; do not free).
 */
const char *pelorus_version(void);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif /* PELORUS_H */
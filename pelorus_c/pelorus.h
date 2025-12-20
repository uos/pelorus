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
typedef struct pelorus_imu_msg {
    int32_t timestamp_sec;                    /**< Timestamp seconds since epoch */
    uint32_t timestamp_nanosec;               /**< Timestamp nanoseconds part */
    const char *frame_id;                     /**< Frame ID (sensor name) */
    double orientation[4];                    /**< Quaternion [w, x, y, z] */
    double orientation_covariance[9];         /**< 3x3 covariance matrix (row-major) */
    double angular_velocity[3];               /**< Angular velocity [x, y, z] rad/s */
    double angular_velocity_covariance[9];    /**< 3x3 covariance matrix (row-major) */
    double linear_acceleration[3];            /**< Linear acceleration [x, y, z] m/s^2 */
    double linear_acceleration_covariance[9]; /**< 3x3 covariance matrix (row-major) */
} pelorus_imu_msg;

/**
 * @brief Odometry message with twist (velocity).
 */
typedef struct pelorus_odom_msg {
    int32_t timestamp_sec;       /**< Timestamp seconds since epoch */
    uint32_t timestamp_nanosec;  /**< Timestamp nanoseconds part */
    double twist_linear[3];      /**< Linear velocity [vx, vy, vz] m/s */
    double twist_angular[3];     /**< Angular velocity [wx, wy, wz] rad/s */
    double twist_covariance[36]; /**< 6x6 covariance matrix (row-major) */
} pelorus_odom_msg;

/**
 * @brief Point field descriptor for PointCloud2.
 */
typedef struct pelorus_point_field {
    const char *name; /**< Field name (e.g., "x", "y", "z", "intensity") */
    uint32_t offset;  /**< Byte offset from start of point data */
    uint8_t datatype; /**< Data type (7=FLOAT32, 8=FLOAT64, etc.) */
    uint32_t count;   /**< Number of elements (usually 1) */
} pelorus_point_field;

/**
 * @brief ROS2 PointCloud2 message.
 *
 * Timestamp semantics:
 * - `header_timestamp_sec/nanosec` is an absolute reference time for the scan.
 * - If your fields include a `t` field, it is interpreted as a **non-negative** (`uint32`)
 *   nanoseconds offset **forward in time** relative to the scan header timestamp.
 *   (So `point_time_abs = header_time_abs + t`.)
 *
 * Recommended field layout for non-ROS point sources:
 * - x (FLOAT32) offset 0
 * - y (FLOAT32) offset 4
 * - z (FLOAT32) offset 8
 * - intensity (FLOAT32) offset 12 (optional)
 * - t (UINT32) offset 16 (nanoseconds offset)
 * - point_step = 20
 */
typedef struct pelorus_pointcloud2 {
    int32_t header_timestamp_sec;      /**< Timestamp seconds since epoch */
    uint32_t header_timestamp_nanosec; /**< Timestamp nanoseconds part */
    const char *header_frame_id;       /**< Frame ID (sensor name) */
    uint32_t height;                   /**< Number of rows (1 for unorganized) */
    uint32_t width;                    /**< Number of points per row */
    const pelorus_point_field *fields; /**< Array of field descriptors */
    uintptr_t fields_len;              /**< Number of fields */
    bool is_bigendian;                 /**< Byte order (usually false) */
    uint32_t point_step;               /**< Bytes per point */
    uint32_t row_step;                 /**< Bytes per row */
    const uint8_t *data;               /**< Raw point data buffer */
    uintptr_t data_len;                /**< Length of data buffer in bytes */
    bool is_dense;                     /**< True if no invalid points */
} pelorus_pointcloud2;

/**
 * @brief Simple XYZI point structure (used in outputs).
 */
typedef struct pelorus_point_xyzi {
    float x;         /**< X coordinate in meters */
    float y;         /**< Y coordinate in meters */
    float z;         /**< Z coordinate in meters */
    float intensity; /**< Intensity/reflectivity value */
} pelorus_point_xyzi;

/**
 * @brief Complete odometry output from Pelorus.
 *
 * Memory for the points array is allocated by Pelorus and must be freed using
 * pelorus_free_odometry() after use.
 */
typedef struct pelorus_odometry {
    int32_t timestamp_sec;       /**< Timestamp seconds since epoch */
    uint32_t timestamp_nanosec;  /**< Timestamp nanoseconds part */
    double position[3];          /**< Position [x, y, z] in meters */
    double orientation[4];       /**< Quaternion [w, x, y, z] */
    double pose_covariance[36];  /**< 6x6 pose covariance (row-major) */
    double linear_velocity[3];   /**< Linear velocity [vx, vy, vz] m/s */
    double angular_velocity[3];  /**< Angular velocity [wx, wy, wz] rad/s */
    double twist_covariance[36]; /**< 6x6 twist covariance (row-major) */
    pelorus_point_xyzi *points; /**< Registered point cloud (must free) */
    uintptr_t num_points;        /**< Number of points in cloud */
} pelorus_odometry;

/**
 * @brief Create a new Pelorus LIO instance.
 * @param config_path Path to .rl configuration file, or NULL for defaults.
 *
 * Default behavior:
 * - Uses the embedded default configuration if `config_path == NULL`.
 * - Treats `pelorus_pointcloud2` inputs as **device-specific** point clouds.
 *   (See `pelorus_create_with_options()` to override.)
 * @return Opaque handle to Pelorus instance, or NULL on error.
 */
PelorusHandle *pelorus_create(const char *config_path);

/**
 * @brief Options for creating a Pelorus instance.
 *
 * This struct lets you control optional behaviors without having to modify the on-disk `.rl`.
 * For tri-state fields, use -1 to select the library default.
 */
typedef struct pelorus_create_options {
    const char *config_path; /**< Path to .rl configuration file, or NULL for defaults. */
    int8_t device_specific_pointcloud2;
} pelorus_create_options;

/**
 * @brief Default initializer for pelorus_create_options.
 *
 * Usage:
 * `pelorus_create_options opts = PELORUS_CREATE_OPTIONS_DEFAULT;`
 */
#define PELORUS_CREATE_OPTIONS_DEFAULT { .config_path = NULL, .device_specific_pointcloud2 = 1 }

/**
 * @brief Create a new Pelorus LIO instance with options.
 *
 * Defaults:
 * - If `options == NULL`, behaves like `pelorus_create(NULL)`.
 * - If `options->device_specific_pointcloud2 == -1`, the default is `true`.
 */
PelorusHandle *pelorus_create_with_options(const pelorus_create_options *options);

/**
 * @brief Add an IMU measurement to the pipeline.
 * @return 0 on success, -1 on error.
 */
int32_t pelorus_add_imu(PelorusHandle *handle, const pelorus_imu_msg *imu);

/**
 * @brief Add an odometry measurement to the pipeline.
 * @return 0 on success, -1 on error.
 */
int32_t pelorus_add_odom(PelorusHandle *handle, const pelorus_odom_msg *odom);

/**
 * @brief Add a point cloud to the pipeline.
 * @return 0 on success, -1 on error.
 */
int32_t pelorus_add_pointcloud(PelorusHandle *handle, const pelorus_pointcloud2 *cloud);

/**
 * @brief Add raw points from a packed `float` array (Nx4): [x,y,z,t].
 *
 * `t` is interpreted as nanoseconds offset (relative to the scan header timestamp) and is cast
 * to `uint32_t`.
 * Internally this is converted to a PointCloud2 with fields: x,y,z,intensity,t.
 *
 * @return 0 on success, -1 on error.
 */
int32_t pelorus_add_points_xyzt_f32(
    PelorusHandle *handle,
    int32_t header_timestamp_sec,
    uint32_t header_timestamp_nanosec,
    const char *frame_id,
    const float *xyzt,
    uintptr_t num_points);

/**
 * @brief Add raw points from a packed `float` array (Nx5): [x,y,z,intensity,t].
 *
 * `t` is interpreted as nanoseconds offset (relative to the scan header timestamp) and is cast
 * to `uint32_t`.
 * Internally this is converted to a PointCloud2 with fields: x,y,z,intensity,t.
 *
 * @return 0 on success, -1 on error.
 */
int32_t pelorus_add_points_xyzit_f32(
    PelorusHandle *handle,
    int32_t header_timestamp_sec,
    uint32_t header_timestamp_nanosec,
    const char *frame_id,
    const float *xyzit,
    uintptr_t num_points);

/**
 * @brief Try to receive odometry output (non-blocking).
 * @return 1 if odometry received, 0 if no data, -1 on error.
 */
int32_t pelorus_try_recv(PelorusHandle *handle, pelorus_odometry *result);

/**
 * @brief Blocking receive odometry output (synchronous).
 * 
 * This function blocks until odometry data is available or an error occurs.
 * Use this for synchronous processing where you want to wait for each result.
 * 
 * @param handle The Pelorus handle.
 * @param result Pointer to store the odometry result.
 * @return 1 if odometry received, -1 on error (e.g., channel disconnected).
 */
int32_t pelorus_recv(PelorusHandle *handle, pelorus_odometry *result);

/**
 * @brief Callback function type for asynchronous odometry reception.
 * 
 * @param odom Pointer to odometry data. Valid only during the callback.
 * @param user_data User-provided context pointer passed to pelorus_set_callback.
 */
typedef void (*PelorusOdometryCallback)(const pelorus_odometry *odom, void *user_data);

/**
 * @brief Set a callback for asynchronous odometry reception.
 * 
 * When a callback is set, odometry results will be delivered via the callback
 * instead of being queued for pelorus_try_recv() or pelorus_recv().
 * The callback is invoked from a background thread.
 * 
 * @param handle The Pelorus handle.
 * @param callback The callback function to invoke, or NULL to disable callbacks.
 * @param user_data User-provided context pointer passed to the callback.
 * @return 0 on success, -1 on error.
 */
int32_t pelorus_set_callback(PelorusHandle *handle, 
                              PelorusOdometryCallback callback,
                              void *user_data);

/**
 * @brief Free memory allocated for odometry result.
 */
void pelorus_free_odometry(pelorus_odometry *odom);

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
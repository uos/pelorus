/* Pelorus C Library - Efficient Robot Odometry.
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
  int32_t timestamp_sec;            /**< Timestamp seconds since epoch */
  uint32_t timestamp_nanosec;       /**< Timestamp nanoseconds part */
  const char *frame_id;             /**< Frame ID (sensor name) - optional (may be NULL) */
  double orientation[4];            /**< Quaternion [w, x, y, z] */
  double orientation_covariance[9]; /**< 3x3 covariance matrix (row-major) */
  double angular_velocity[3];       /**< Angular velocity [x, y, z] rad/s */
  double
      angular_velocity_covariance[9]; /**< 3x3 covariance matrix (row-major) */
  double linear_acceleration[3]; /**< Linear acceleration [x, y, z] m/s^2 */
  double linear_acceleration_covariance[9]; /**< 3x3 covariance matrix
                                               (row-major) */
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
  const char *name; /**< Field name (must be non-NULL, e.g., "x", "y", "z", "intensity") */
  uint32_t offset;  /**< Byte offset from start of point data */
  uint8_t datatype; /**< Data type (7=FLOAT32, 8=FLOAT64, etc.) */
} pelorus_point_field;

/**
 * @brief ROS2 PointCloud2 message.
 *
 * Interpretation and timestamp semantics depend on how the point cloud is parsed:
 * - When Pelorus is configured to use *device-specific* parsing (the default; see
 *   `device_specific_pointcloud2`), field meanings (including any `t` field) are
 *   interpreted according to the device/vendor configuration and may **not** be a
 *   simple header-relative nanosecond offset.
 * - When using the *generic* parsing path (`device_specific_pointcloud2 == false`),
 *   Pelorus expects per-point `t` to be a **non-negative** (`uint32`) nanoseconds
 *   offset **forward in time** relative to the scan header timestamp.
 *   (So `point_time_abs = header_time_abs + t`.)
 *
 * Recommended field layout for generic (non-device-specific) inputs:
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
  uint32_t width;                    /**< Number of points (unstructured cloud; height is implicitly 1) */
  const pelorus_point_field *fields; /**< Pointer to single field descriptor (must be non-NULL) */
  uint32_t point_step;               /**< Bytes per point (recommended 20 for x,y,z,intensity,t) */
  uint32_t row_step;                 /**< Bytes per row (point_step * width) */
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
  float intensity; /**< Intensity/reflectivity value (0..255 scaled as float) */
} pelorus_point_xyzi;

/**
 * @brief Complete odometry output from Pelorus.
 *
 * **Coordinate Frame:**
 * - `position`, `orientation`, `linear_velocity`, `angular_velocity`, and
 * `points`: Expressed in the **Lidar World frame**.
 * - `pose_covariance` and `twist_covariance`: Expressed in the **IMU World
 * frame**.
 *
 * The world frame is established at initialization with the Z-axis pointing
 * upward (anti-gravity), aligned with the IMU orientation at startup.
 *
 * Memory for the points array is allocated by Pelorus and must be freed using
 * pelorus_free_odometry() after use.
 */
typedef struct pelorus_odometry {
  int32_t timestamp_sec;      /**< Timestamp seconds since epoch */
  uint32_t timestamp_nanosec; /**< Timestamp nanoseconds part */
  double position[3]; /**< Position [x, y, z] in meters in Lidar World frame */
  double orientation[4]; /**< Quaternion [w, x, y, z] in Lidar World frame */
  double pose_covariance[36];  /**< 6x6 pose covariance (row-major) in IMU World
                                  frame */
  double linear_velocity[3];   /**< Linear velocity [vx, vy, vz] m/s in Lidar
                                  World frame */
  double angular_velocity[3];  /**< Angular velocity [wx, wy, wz] rad/s in Lidar
                                  World frame */
  double twist_covariance[36]; /**< 6x6 twist covariance (row-major) in IMU
                                  World frame */
  uint64_t scan_id; /**< Scan id assigned when the pointcloud was added */
  pelorus_point_xyzi *points; /**< Registered point cloud in Lidar World frame (must free when `points_memory==0`) */
  uintptr_t num_points; /**< Number of points in cloud */
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
 * This struct lets you control optional behaviors without having to modify the
 * on-disk `.rl`. For tri-state fields, use -1 to select the library default.
 */
typedef struct pelorus_create_options {
  const char *
      config_path; /**< Path to .rl configuration file, or NULL for defaults. */
  int8_t device_specific_pointcloud2;
} pelorus_create_options;

/**
 * @brief Default initializer for pelorus_create_options.
 *
 * Usage:
 * `pelorus_create_options opts = PELORUS_CREATE_OPTIONS_DEFAULT;`
 */
#define PELORUS_CREATE_OPTIONS_DEFAULT                                         \
  { .config_path = NULL, .device_specific_pointcloud2 = 1 }

/**
 * @brief Create a new Pelorus LIO instance with options.
 *
 * Defaults:
 * - If `options == NULL`, behaves like `pelorus_create(NULL)`.
 * - If `options->device_specific_pointcloud2 == -1`, the default is `true`.
 */
PelorusHandle *
pelorus_create_with_options(const pelorus_create_options *options);

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
 * @return Assigned scan id (>0) on success, 0 on error.
 */
uint64_t pelorus_add_pointcloud(PelorusHandle *handle,
                                const pelorus_pointcloud2 *cloud);

/**
 * @brief Add raw points from a packed `float` array (Nx4): [x,y,z,t].
 *
 * `t` is interpreted as nanoseconds offset (relative to the scan header
 * timestamp) and is cast to `uint32_t`. Internally this is converted to a
 * PointCloud2 with fields: x,y,z,intensity,t.
 *
 * @return Assigned scan id (>0) on success, 0 on error.
 */
uint64_t pelorus_add_points_xyzt_f32(PelorusHandle *handle,
                                     int32_t header_timestamp_sec,
                                     uint32_t header_timestamp_nanosec,
                                     const char *frame_id, const float *xyzt,
                                     uintptr_t num_points);

/**
 * @brief Add raw points from a packed `float` array (Nx5): [x,y,z,intensity,t].
 *
 * `t` is interpreted as nanoseconds offset (relative to the scan header
 * timestamp) and is cast to `uint32_t`. Internally this is converted to a
 * PointCloud2 with fields: x,y,z,intensity,t.
 *
 * @return Assigned scan id (>0) on success, 0 on error.
 */
uint64_t pelorus_add_points_xyzit_f32(PelorusHandle *handle,
                                      int32_t header_timestamp_sec,
                                      uint32_t header_timestamp_nanosec,
                                      const char *frame_id, const float *xyzit,
                                      uintptr_t num_points);

/**
 * @brief Status codes for Pelorus functions.
 *
 * Use these to interpret return values from functions that return an `int32_t`
 * status.
 */
typedef enum pelorus_status {
  PELORUS_STATUS_OK =
      0, /**< Operation succeeded and data (if applicable) is available */
  PELORUS_STATUS_NO_DATA = 1, /**< Non-blocking call had no data available */
  PELORUS_STATUS_ERROR = -1,  /**< Generic error */
  PELORUS_STATUS_CALLBACK_SET =
      -2, /**< Operation unavailable because callback mode is active */
  PELORUS_STATUS_DISCONNECTED = -3, /**< Underlying channel disconnected */
} pelorus_status;

/**
 * @brief Try to receive odometry output (non-blocking).
 *
 * This function attempts to retrieve odometry without blocking. The return
 * value is a status code from `pelorus_status`. When `PELORUS_STATUS_OK` is
 * returned, `result` is populated and must be freed with
 * `pelorus_free_odometry()` when no longer needed.
 */
int32_t pelorus_try_recv(PelorusHandle *handle, pelorus_odometry *result);

/**
 * @brief Blocking receive odometry output (synchronous).
 *
 * This function blocks until odometry data is available or an error occurs.
 * The return value is a status code from `pelorus_status`.
 * When `PELORUS_STATUS_OK` is returned, `result` is populated and must be freed
 * with `pelorus_free_odometry()` when no longer needed.
 */
int32_t pelorus_recv(PelorusHandle *handle, pelorus_odometry *result);

/**
 * @brief Blocking receive odometry for a specific scan id.
 *
 * This function blocks until an odometry result with the requested `scan_id` is
 * available. Non-matching outputs are cached for subsequent calls.
 *
 * @param handle The Pelorus handle.
 * @param scan_id The scan id to wait for.
 * @param result Pointer to store the odometry result.
 * @return A `pelorus_status` code (e.g., `PELORUS_STATUS_OK` on success, negative on error).
 */

/**
 * @brief Callback function type for asynchronous odometry reception.
 *
 * @param odom Pointer to odometry data. Valid only during the callback.
 * @param user_data User-provided context pointer passed to
 * pelorus_set_callback.
 */
typedef void (*PelorusOdometryCallback)(const pelorus_odometry *odom,
                                        void *user_data);

/**
 * @brief Set a callback for asynchronous odometry reception.
 *
 * When a callback is set, odometry results will be delivered via the callback
 * instead of being queued for pelorus_try_recv() or pelorus_recv().
 * The callback is invoked from a background thread.
 *
 * @param handle The Pelorus handle.
 * @param callback The callback function to invoke, or NULL to disable
 * callbacks.
 * @param user_data User-provided context pointer passed to the callback.
 * @return A `pelorus_status` code (e.g., `PELORUS_STATUS_OK` on success,
 * negative on error).
 */
int32_t pelorus_set_callback(PelorusHandle *handle,
                             PelorusOdometryCallback callback, void *user_data);

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

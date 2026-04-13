#ifndef PELORUS_PICO_CONFIG_H
#define PELORUS_PICO_CONFIG_H

/* stdint.h is available from the compiler (freestanding) on GCC/Clang */
#include <stdint.h>

/* =========================================================
 * pelorus_pico_config.h — compile-time tunables
 *
 * All memory is statically allocated.  Adjust these values
 * before compilation to match your hardware constraints.
 * ========================================================= */

/* Hash table slots for the voxel map.
 * Must be a power-of-two for the modulo to be cheap (use & mask).
 * Keep load factor < 50%: max useful voxels ≈ CAPACITY/2.
 * 8192 → ~720 KB  |  4096 → ~360 KB  |  2048 → ~180 KB */
#define PELORUS_VOXEL_CAPACITY        8192

/* Single point buffer that holds both raw input and downsampled output.
 * After pp_downsample(), work_n <= n_raw tracks the actual valid count.
 * 16384 pts × 20 B = 320 KB */
#define PELORUS_CLOUD_MAX_POINTS      16384

/* Maximum IMU messages buffered between successive cloud calls.
 * 200 × 56 B = 11 KB */
#define PELORUS_IMU_BUFFER_MAX        200

/* Duration of stationary IMU data used to estimate initial gravity & gyro bias.
 * Matches the default config: imu.initial_setup = 100ms. */
#define PELORUS_INIT_IMU_DURATION_SEC 0.1

/* Maximum number of IMU samples retained while collecting the init window. */
#define PELORUS_INIT_IMU_MAX_SAMPLES  100

/* IEKF maximum correction iterations per scan.
 * Matches default.rl: registration.max_iter = 4 */
#define PELORUS_EKF_MAX_ITER          4

/* Voxel grid resolution in metres (must match the system that provides data).
 * Matches default.rl: registration.map.resolution = 50cm */
#define PELORUS_VOXEL_SIZE            0.5f

/* Voxel neighborhood type for plane fitting: 6, 18, or 26.
 * Matches Rust: IVoxNearbyType::Nearby26 */
#define PELORUS_NEARBY_MODE           26

/* Sensitivity for weighted voxel distribution update (Mahalanobis decay).
 * Matches default.rl: registration.map.distribution_update_sensitivity = 10.0 */
#define PELORUS_SENSITIVITY           10.0f

/* Minimum distance threshold inside a voxel (metres, squared used internally). */
#define PELORUS_VOXEL_MIN_DIST        0.05f

/* LRU eviction trigger: evict when occupied_count >= CAPACITY * this fraction. */
#define PELORUS_LRU_TRIGGER_FRACTION  0.9f

/* IMU measurement noise covariances (tune per-sensor). */
#define PELORUS_COV_ACC               0.1
#define PELORUS_COV_GYR               0.1
#define PELORUS_COV_ACC_BIAS          0.0001
#define PELORUS_COV_GYR_BIAS          0.0001

/* Laser point noise covariance for the IEKF measurement model. */
#define PELORUS_LASER_POINT_NOISE     0.001

/* Gravity magnitude (m/s²). */
#define PELORUS_GRAVITY               9.81

/* Initial lidar-to-IMU extrinsic (identity = co-located).
 * Rotation expressed as a 3×3 row-major matrix (9 doubles). */
#define PELORUS_LIDAR_IMU_R { \
    1.0, 0.0, 0.0,            \
    0.0, 1.0, 0.0,            \
    0.0, 0.0, 1.0             \
}
#define PELORUS_LIDAR_IMU_T { -0.006253, 0.011775, 0.028535 }

/* Initial base-to-lidar extrinsic (identity = co-located).
 * Used only to set the initial world-frame origin at base_link. */
#define PELORUS_BASE_LIDAR_R { \
    1.0, 0.0, 0.0,             \
    0.0, 1.0, 0.0,             \
    0.0, 0.0, 1.0              \
}
#define PELORUS_BASE_LIDAR_T { 0.0, 0.0, 0.0 }

/* Initial EKF covariance diagonals (from imu.rs:COV1/COV2/COV3). */
#define PELORUS_COV_INIT_BG   0.0001
#define PELORUS_COV_INIT_BA   0.001
#define PELORUS_COV_INIT_GRAV 0.00001

/* Convergence threshold for IEKF iterations (same unit as noise cov). */
#define PELORUS_IEKF_EPSI     PELORUS_LASER_POINT_NOISE

/* Plane validity threshold (same as Rust: s > 0.9). */
#define PELORUS_PLANE_SCORE_THRESHOLD 0.9f

/* Sphericity threshold for plane rejection (lambda0/lambda2 <= 0.5). */
#define PELORUS_PLANE_SPHERICITY_MAX  0.5f

#endif /* PELORUS_PICO_CONFIG_H */

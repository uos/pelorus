#ifndef PELORUS_PICO_H
#define PELORUS_PICO_H

/* =========================================================
 * pelorus_pico.h — public API
 *
 * Bare-metal LiDAR-Inertial Odometry library.
 * No OS, no libc heap, no threads.
 * All state lives in a single static context.
 *
 * Usage:
 *   pelorus_pico_init();
 *   while(running) {
 *       // feed IMU at sensor rate
 *       pelorus_odometry_t odom = pelorus_pico_add_imu(t, gx,gy,gz, ax,ay,az);
 *       // feed a cloud when a full scan arrives (BLOCKING until registered)
 *       odom = pelorus_pico_add_cloud(pts, n, t_start, t_end);
 *       if (odom.valid) { use(odom); }
 *   }
 * ========================================================= */

#ifdef __cplusplus
extern "C" {
#endif

/* Single LiDAR point: x, y, z in metres (LiDAR frame), intensity [0,1],
 * timestamp in seconds relative to the start of the scan. */
typedef struct {
    float x;
    float y;
    float z;
    float intensity;
    float timestamp; /* seconds, relative to cloud_start_time */
} pelorus_pico_point_t;

/* Odometry output.
 * position    : metres, world frame
 * orientation : quaternion [w, x, y, z], world frame
 * velocity    : m/s, body (IMU) frame
 * valid       : 0 = system not yet initialised, 1 = valid estimate */
typedef struct {
    double timestamp;       /* seconds (absolute) */
    double position[3];     /* x, y, z */
    double orientation[4];  /* w, x, y, z */
    double linear_velocity[3];
    int    valid;
} pelorus_odometry_t;

/* Initialise the library.  Must be called once before any other function.
 * Zeroes all static state and sets up the initial EKF covariance. */
void pelorus_pico_init(void);

/* Feed a single IMU sample.
 * gyro_* : rad/s  (angular velocity, body frame)
 * accel_*: m/s²   (linear acceleration, body frame, including gravity)
 * Returns the current best-estimate odometry.  Before the system is
 * initialised (needs the configured IMU init time window), .valid == 0. */
pelorus_odometry_t pelorus_pico_add_imu(
    double timestamp,
    float  gyro_x,  float gyro_y,  float gyro_z,
    float  accel_x, float accel_y, float accel_z
);

/* Feed a full point cloud.  BLOCKS until downsampling, undistortion, and
 * scan-to-map registration are complete.
 * pts            : array of n points in LiDAR frame
 * n              : number of points (capped at PELORUS_CLOUD_MAX_POINTS)
 * cloud_start_time / cloud_end_time : absolute timestamps in seconds
 * Returns the odometry at the end of this scan.  .valid == 0 on the
 * first scan (map is empty; the cloud is used to build the initial map). */
pelorus_odometry_t pelorus_pico_add_cloud(
    const pelorus_pico_point_t *pts,
    int    n,
    double cloud_start_time,
    double cloud_end_time
);

#ifdef __cplusplus
}
#endif

#endif /* PELORUS_PICO_H */

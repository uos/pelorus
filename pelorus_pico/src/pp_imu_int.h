#ifndef PP_IMU_INT_H
#define PP_IMU_INT_H

/* =========================================================
 * pp_imu_int.h — IMU ring buffer, integration, gravity init
 *
 * Port of pelorus_reg/src/imu.rs: ImuProcessing
 * ========================================================= */

#include "../include/pelorus_pico_config.h"
#include "pp_iekf.h"

/* A single IMU measurement. */
typedef struct {
    double t;
    double gyro[3];  /* rad/s */
    double acc[3];   /* m/s² */
} pp_imu_msg_t;

/* A pose snapshot for undistortion (one per IMU step between scans). */
typedef struct {
    double offset_time; /* seconds from cloud_start_time */
    double pose_rot[3][3];
    double pose_pos[3];
    double vel[3];
    double acc[3];  /* world-frame acc + gravity, for quadratic motion model */
    double gyr[3];  /* bias-corrected angular velocity */
} pp_imu_pose_t;

/* All IMU state lives here. */
typedef struct {
    pp_imu_msg_t  buf[PELORUS_IMU_BUFFER_MAX];  /* ring buffer */
    int           head;                          /* oldest entry */
    int           tail;                          /* next write position */
    int           count;

    /* Initialisation accumulators */
    pp_imu_msg_t  init_buf[PELORUS_INIT_IMU_MAX_SAMPLES];
    int           init_count;
    double        init_start_time;

    double        mean_acc[3];
    double        mean_gyr[3];
    double        mean_acc_norm; /* |mean_acc|, used for gravity scaling */

    double        angvel_last[3]; /* bias-corrected ang vel from last IMU */
    double        acc_s_last[3];  /* world-frame acc+gravity from last IMU */

    /* Noise Q matrix (12×12 diagonal, built at init) */
    double        Q[12][12];

    /* Poses accumulated during last add_cloud integration pass */
    pp_imu_pose_t imu_poses[PELORUS_IMU_BUFFER_MAX];
    int           n_imu_poses;

    /* Extrinsics (lidar→IMU), initialised from config */
    double        lidar_to_imu_r[3][3];
    double        lidar_to_imu_t[3];
    double        lidar_to_base_r[3][3];
    double        lidar_to_base_t[3];
} pp_imu_state_t;

/* ---- lifecycle ---- */
void imu_state_init(pp_imu_state_t *st,
                    const double lidar_to_imu_r[3][3],
                    const double lidar_to_imu_t[3],
                    const double base_to_lidar_r[3][3],
                    const double base_to_lidar_t[3]);

/* ---- ring buffer ---- */
void imu_push(pp_imu_state_t *st, const pp_imu_msg_t *msg);
/* Returns -1 if empty, otherwise fills *msg and returns 0. */
int  imu_peek_oldest(const pp_imu_state_t *st, pp_imu_msg_t *msg);
int  imu_peek_newest(const pp_imu_state_t *st, pp_imu_msg_t *msg);

/* ---- initialisation (gravity + bias from stationary samples) ----
 * Accumulates samples over the configured init time window; returns 1 when
 * ready, 0 while still collecting.
 * On success: initialises ekf state, covariance, and st->mean_acc_norm. */
int imu_init_gravity(pp_imu_state_t *st, const pp_imu_msg_t *msg, pp_iekf_t *ekf);

/* ---- IMU integration over a time range ----
 * Reads from ring buffer, calls iekf_predict() for each step.
 * Also builds st->imu_poses[] for subsequent undistortion.
 * t_start / t_end: absolute times of cloud begin/end.
 * last_cloud_end: kept for API compatibility; current integration uses
 *                 the cloud-local [t_start, t_end] span. */
void imu_integrate_range(pp_imu_state_t *st,
                         pp_iekf_t *ekf,
                         double last_cloud_end,
                         double t_start,
                         double t_end);

#endif /* PP_IMU_INT_H */

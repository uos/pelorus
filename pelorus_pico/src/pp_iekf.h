#ifndef PP_IEKF_H
#define PP_IEKF_H

/* =========================================================
 * pp_iekf.h — Indirect Extended Kalman Filter on Lie manifold
 *
 * 24-dimensional state (double precision):
 *   pos[0-2]          position in world frame
 *   rot_vec[3-5]      SO(3) stored as rotation matrix, serialised
 *                     via box_plus/box_minus using Rodrigues
 *   ext_r_vec[6-8]    lidar-to-IMU rotation extrinsic
 *   ext_t[9-11]       lidar-to-IMU translation extrinsic
 *   vel[12-14]        velocity in world frame
 *   bg[15-17]         gyro bias
 *   ba[18-20]         accel bias
 *   grav[21-23]       gravity vector in world frame
 *
 * Port of pelorus_reg/src/iekf.rs
 * ========================================================= */

#include "pp_linalg.h"
#include "pp_voxmap.h"
#include "../include/pelorus_pico.h"
#include "../include/pelorus_pico_config.h"

/* State: rotation parts stored as 3×3 rotation matrices (double).
 * All other parts are plain 3-vectors (double). */
typedef struct {
    double rot[3][3];     /* R_W_I: world to IMU */
    double ext_r[3][3];   /* R_I_L: lidar to IMU rotation extrinsic */
    double pos[3];
    double ext_t[3];
    double vel[3];
    double bg[3];
    double ba[3];
    double grav[3];
} pp_state_t;

/* Plane measurement: {nx, ny, nz, d, px_local, py_local, pz_local}
 * The IEKF model function fills an array of these. */
typedef struct {
    float nx, ny, nz, d;           /* plane coefficients (world frame) */
    float px, py, pz;              /* point coordinates (LiDAR frame, undistorted) */
} pp_plane_meas_t;

/* All large scratch matrices live here — NEVER allocate 24×24 on the stack. */
typedef struct {
    pp_state_t state;
    mat24       P;          /* 24×24 covariance */

    /* prediction scratch */
    mat24       Fx;         /* I + df_dx * dt */
    mat24       Fw_Q_Fw;    /* Fw*Q*Fw^T*dt^2 */
    mat24       tmp24a;
    mat24       tmp24b;

    /* correction scratch */
    mat24       P_propagated;
    mat24       P_inv;
    double      HTH12[12][12];
    double      lhs12[12][12];
    double      lhs12_inv[12][12];
    double      kf24x12[24][12];   /* Kalman gain K (24×12) */
    double      KH24x12[24][12];   /* K*H  (24×12, 12 non-zero cols of KH) */
    mat24       KH24;               /* full 24×24 KH */
    double      H_m_x12[PELORUS_CLOUD_MAX_POINTS][12]; /* H (m×12) */
    double      h_m[PELORUS_CLOUD_MAX_POINTS];           /* residual h (m) */
    double      dx_new[24];         /* box_minus(state, state_prop) */
    double      dx_[24];            /* correction delta */

    /* misc */
    int initialized;
    double last_time;
    double mean_acc_norm; /* used in undistort gravity scaling */
} pp_iekf_t;

/* ---- lifecycle ---- */
void iekf_init(pp_iekf_t *ekf);
void iekf_state_init_default(pp_iekf_t *ekf, double gravity);

/* ---- box_plus / box_minus (Lie manifold) ---- */
/* state = state ⊕ dx[24] */
void box_plus(pp_state_t *out, const pp_state_t *x, const double dx[24]);
/* result[24] = x1 ⊖ x2 */
void box_minus(double result[24], const pp_state_t *x1, const pp_state_t *x2);

/* ---- state helper: get world-frame pose ----
 * T_W_L = T_W_I * T_I_L
 * pos_world: world position of LiDAR origin
 * R_world: rotation of LiDAR in world */
void iekf_get_lidar_pose(const pp_iekf_t *ekf,
                         double pos_world[3], double R_world[3][3]);

/* ---- predict (IMU time update) ----
 * Q[12][12] noise matrix (diagonal, built once at init)
 * acc[3], gyro[3] in body frame */
void iekf_predict(pp_iekf_t *ekf,
                  double dt,
                  const double Q[12][12],
                  const double acc[3],
                  const double gyro[3]);

/* ---- correct (LiDAR scan-to-map measurement update) ----
 * Plane correspondences are re-extracted on every IEKF iteration so that
 * the linearisation point tracks the evolving state estimate.
 * (Matches Rust after "revert: no plane cache" commit.)
 *
 * pts / n_pts       : undistorted LiDAR-frame points
 * vm                : voxel map (read-only for plane queries)
 * planes_scratch    : caller-provided scratch buffer (≥ PELORUS_CLOUD_MAX_POINTS)
 * measurement_time  : absolute timestamp of the scan end */
void iekf_correct(pp_iekf_t *ekf,
                  const pelorus_pico_point_t *pts,
                  int n_pts,
                  pp_voxmap_t *vm,
                  pp_plane_meas_t *planes_scratch,
                  double measurement_time);

#endif /* PP_IEKF_H */

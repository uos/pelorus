#ifndef PP_UNDISTORT_H
#define PP_UNDISTORT_H

/* =========================================================
 * pp_undistort.h — point cloud deskew using IMU poses
 *
 * Port of the deskew closure in imu.rs:undistort
 * ========================================================= */

#include "../include/pelorus_pico.h"
#include "pp_imu_int.h"
#include "pp_iekf.h"

/* Deskew all n points in pts[] using the accumulated imu_poses.
 * After this call each point is expressed in the IMU frame at
 * the END of the scan (the reference frame for registration).
 *
 * imu_state: read-only, supplies imu_poses[] and n_imu_poses
 * ekf:       read-only, supplies final-frame rotation/translation for
 *            the inverse transform back to LiDAR frame */
void pp_undistort(pelorus_pico_point_t *pts,
                  int n,
                  const pp_imu_state_t *st,
                  const pp_iekf_t *ekf);

#endif /* PP_UNDISTORT_H */

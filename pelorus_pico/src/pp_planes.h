#ifndef PP_PLANES_H
#define PP_PLANES_H

/* =========================================================
 * pp_planes.h — extract plane measurements for the IEKF
 *
 * Port of the CPU plane-extraction path in
 * pelorus_reg/src/registration.rs:register_and_map (Cpu branch)
 * ========================================================= */

#include "../include/pelorus_pico.h"
#include "pp_voxmap.h"
#include "pp_iekf.h"

/* For each downsampled point, transform to world frame using the current
 * EKF state, query the voxel map, compute the plane, score it, and
 * if valid emit a pp_plane_meas_t into planes[].
 *
 * Returns the number of valid plane measurements written.
 *
 * pts[0..n_work-1]: undistorted points in LiDAR frame
 * vm:              voxel map (read-only)
 * ekf:             current EKF state (read-only, f32 transform as in Rust)
 * planes:          output array, caller provides PELORUS_CLOUD_MAX_POINTS slots */
int pp_planes_extract(const pelorus_pico_point_t *pts,
                      int n_work,
                      pp_voxmap_t *vm,
                      const pp_iekf_t *ekf,
                      pp_plane_meas_t *planes);

#endif /* PP_PLANES_H */

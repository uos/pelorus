/* pp_planes.c — plane measurement extraction
 * Port of the CPU path in registration.rs (Cpu backend branch)
 */
#include "pp_planes.h"
#include "pp_linalg.h"
#include "../include/pelorus_pico_config.h"

#define PP_SQRTF(x) __builtin_sqrtf(x)
#define PP_FABSF(x) __builtin_fabsf(x)

int pp_planes_extract(const pelorus_pico_point_t *pts,
                      int n_work,
                      pp_voxmap_t *vm,
                      const pp_iekf_t *ekf,
                      pp_plane_meas_t *planes) {
    /* Compute f32 rotation matrices for lidar→imu→world transform.
     * Matches the Rust CPU path which does UnitQuaternion<f32> cast
     * from the f64 rotation matrices (bit-equivalent to f32 quaternion). */

    /* ext_q (lidar→IMU): quaternion from ext_r */
    double ext_qd[4]; so3_to_quat(ext_qd, ekf->state.ext_r);
    float ext_qf[4] = {(float)ext_qd[0],(float)ext_qd[1],(float)ext_qd[2],(float)ext_qd[3]};
    float ext_tf[3] = {(float)ekf->state.ext_t[0],(float)ekf->state.ext_t[1],(float)ekf->state.ext_t[2]};

    /* pose_q (IMU→world): quaternion from rot */
    double pose_qd[4]; so3_to_quat(pose_qd, ekf->state.rot);
    float pose_qf[4] = {(float)pose_qd[0],(float)pose_qd[1],(float)pose_qd[2],(float)pose_qd[3]};

    /* pose translation: world position of the IMU origin.
     * Matches Rust: pose_t_f32 = ekf_state.pose().translation = pos (IMU world pos).
     * The extrinsic translation is already baked into p_imu via the first transform step,
     * so adding R_WI * t_ext here would double-count it. */
    float pose_tf[3] = {
        (float)ekf->state.pos[0],
        (float)ekf->state.pos[1],
        (float)ekf->state.pos[2]
    };

    int n_out = 0;
    for (int i = 0; i < n_work && n_out < PELORUS_CLOUD_MAX_POINTS; i++) {
        float px = pts[i].x, py = pts[i].y, pz = pts[i].z;

        /* Transform: p_imu = ext_q * p_local + ext_t */
        float qw=ext_qf[0],qx=ext_qf[1],qy=ext_qf[2],qz=ext_qf[3];
        float p_imu_x = (1-2*(qy*qy+qz*qz))*px + 2*(qx*qy-qw*qz)*py + 2*(qx*qz+qw*qy)*pz + ext_tf[0];
        float p_imu_y = 2*(qx*qy+qw*qz)*px + (1-2*(qx*qx+qz*qz))*py + 2*(qy*qz-qw*qx)*pz + ext_tf[1];
        float p_imu_z = 2*(qx*qz-qw*qy)*px + 2*(qy*qz+qw*qx)*py + (1-2*(qx*qx+qy*qy))*pz + ext_tf[2];

        /* p_world = pose_q * p_imu + pose_t */
        float qw2=pose_qf[0],qx2=pose_qf[1],qy2=pose_qf[2],qz2=pose_qf[3];
        float p_wx = (1-2*(qy2*qy2+qz2*qz2))*p_imu_x + 2*(qx2*qy2-qw2*qz2)*p_imu_y + 2*(qx2*qz2+qw2*qy2)*p_imu_z + pose_tf[0];
        float p_wy = 2*(qx2*qy2+qw2*qz2)*p_imu_x + (1-2*(qx2*qx2+qz2*qz2))*p_imu_y + 2*(qy2*qz2-qw2*qx2)*p_imu_z + pose_tf[1];
        float p_wz = 2*(qx2*qz2-qw2*qy2)*p_imu_x + 2*(qy2*qz2+qw2*qx2)*p_imu_y + (1-2*(qx2*qx2+qy2*qy2))*p_imu_z + pose_tf[2];

        /* Query voxel map for plane at this world position */
        float plane[4];
        if (!voxmap_plane_fit(vm, p_wx, p_wy, p_wz, PELORUS_NEARBY_MODE, plane)) continue;

        /* Point-to-plane distance */
        float pd2 = plane[0]*p_wx + plane[1]*p_wy + plane[2]*p_wz + plane[3];

        /* Scoring: s = 1 - 0.9 * |pd2| / sqrt(sqrt(l2_sq))
         * matches registration.rs: l2_sq is the squared distance of the LOCAL point */
        float l2_sq = px*px + py*py + pz*pz;
        float denom = PP_SQRTF(PP_SQRTF(l2_sq < 1e-9f ? 1e-9f : l2_sq));
        float s = 1.0f - 0.9f * PP_FABSF(pd2) / denom;
        if (s <= PELORUS_PLANE_SCORE_THRESHOLD) continue;

        planes[n_out].nx = plane[0];
        planes[n_out].ny = plane[1];
        planes[n_out].nz = plane[2];
        planes[n_out].d  = plane[3];
        planes[n_out].px = px;
        planes[n_out].py = py;
        planes[n_out].pz = pz;
        n_out++;
    }
    return n_out;
}

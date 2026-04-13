/* pp_undistort.c — point cloud deskew
 * Port of the deskew closure in pelorus_reg/src/imu.rs:undistort
 */
#include "pp_undistort.h"
#include "pp_linalg.h"

void pp_undistort(pelorus_pico_point_t *pts,
                  int n,
                  const pp_imu_state_t *st,
                  const pp_iekf_t *ekf) {
    if (st->n_imu_poses < 2) return;

    /* Final-frame quantities (at scan end = current ekf state) */
    /* imu_state_pose_translation = ekf->state.pos */
    /* imu_state_pose_rotation_t  = ekf->state.rot^T */
    /* R_I_L = ext_r, t_I_L = ext_t */

    double R_WI_end_T[3][3]; /* rotation_T of end pose */
    so3_transpose(R_WI_end_T, ekf->state.rot);

    double R_IL[3][3]; /* = ext_r */
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) R_IL[i][j] = ekf->state.ext_r[i][j];
    double R_LI[3][3]; /* = R_IL^T */
    so3_transpose(R_LI, R_IL);

    const double *p_end = ekf->state.pos;    /* world position at scan end */
    const double *t_il  = ekf->state.ext_t;  /* lidar-to-IMU translation */

    for (int pi = 0; pi < n; pi++) {
        float pt_ts = pts[pi].timestamp;

        /* Binary search for interval bracketing pt.timestamp in imu_poses */
        int lo = 0, hi = st->n_imu_poses - 1;
        while (lo < hi - 1) {
            int mid = (lo + hi) / 2;
            if (st->imu_poses[mid].offset_time <= (double)pt_ts) lo = mid;
            else hi = mid;
        }
        int head_idx = lo;
        int tail_idx = lo + 1;
        if (tail_idx >= st->n_imu_poses) continue;

        const pp_imu_pose_t *head = &st->imu_poses[head_idx];
        const pp_imu_pose_t *tail = &st->imu_poses[tail_idx];

        double dt = (double)pt_ts - head->offset_time;
        if (dt < 0.0) dt = 0.0;

        /* R_i = head.pose_rot * exp(tail.gyr * dt)  (matches Rust: head.pose * SO3::exp(tail.gyr * dt)) */
        double gyr_dt[3] = { tail->gyr[0]*dt, tail->gyr[1]*dt, tail->gyr[2]*dt };
        double dR[3][3];
        so3_exp(dR, gyr_dt);
        double R_i[3][3];
        so3_mul(R_i, head->pose_rot, dR);

        /* t_ei = head.pos + head.vel*dt + 0.5*tail.acc*dt^2 - p_end */
        double t_ei[3];
        for (int i=0;i<3;i++)
            t_ei[i] = head->pose_pos[i] + head->vel[i]*dt + 0.5*tail->acc[i]*dt*dt - p_end[i];

        /* p_imu_frame = R_IL * p_local + t_IL */
        double p_loc[3] = { pts[pi].x, pts[pi].y, pts[pi].z };
        double p_imu[3];
        mat3d_mulv(p_imu, R_IL, p_loc);
        vec3d_add(p_imu, p_imu, t_il);

        /* p_compensate = R_LI * (R_WI_end^T * (R_i * p_imu + t_ei) - t_IL)  */
        double R_i_p[3];
        mat3d_mulv(R_i_p, R_i, p_imu);
        double sum[3]; vec3d_add(sum, R_i_p, t_ei);
        double in_world_ref[3]; mat3d_mulv(in_world_ref, R_WI_end_T, sum);
        double sub[3]; vec3d_sub(sub, in_world_ref, t_il);
        double p_comp[3]; mat3d_mulv(p_comp, R_LI, sub);

        pts[pi].x = (float)p_comp[0];
        pts[pi].y = (float)p_comp[1];
        pts[pi].z = (float)p_comp[2];
    }
}

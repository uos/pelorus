/* pp_iekf.c — Indirect Extended Kalman Filter
 * Port of pelorus_reg/src/iekf.rs
 */
#include "pp_iekf.h"
#include "pp_planes.h"
#include "pp_linalg.h"
#include "../include/pelorus_pico_config.h"

#define PP_SQRT(x)  __builtin_sqrt(x)
#define PP_FABS(x)  __builtin_fabs(x)

/* ================================================================
 * Lifecycle
 * ================================================================ */
void iekf_init(pp_iekf_t *ekf) {
    /* Zero everything */
    int bytes = sizeof(pp_iekf_t);
    char *p = (char *)ekf;
    for (int i = 0; i < bytes; i++) p[i] = 0;

    ekf->initialized = 0;
    ekf->last_time   = 0.0;
}

void iekf_state_init_default(pp_iekf_t *ekf, double gravity) {
    /* Identity rotations */
    ekf->state.rot[0][0]=1; ekf->state.rot[1][1]=1; ekf->state.rot[2][2]=1;
    ekf->state.ext_r[0][0]=1; ekf->state.ext_r[1][1]=1; ekf->state.ext_r[2][2]=1;
    /* Gravity: Z-down in IMU frame initially */
    ekf->state.grav[2] = -gravity;
    /* Identity covariance */
    mat24_identity(ekf->P);
}

/* ================================================================
 * box_plus: state ⊕ dx[24]
 * Port of iekf.rs:box_plus
 * ================================================================ */
void box_plus(pp_state_t *out, const pp_state_t *x, const double dx[24]) {
    /* pos += dx[0:3] */
    out->pos[0] = x->pos[0] + dx[0];
    out->pos[1] = x->pos[1] + dx[1];
    out->pos[2] = x->pos[2] + dx[2];

    /* rot = rot * exp(dx[3:6]) */
    mat3d dR;
    so3_exp(dR, &dx[3]);
    so3_mul(out->rot, x->rot, dR);

    /* ext_r = ext_r * exp(dx[6:9]) */
    mat3d dR2;
    so3_exp(dR2, &dx[6]);
    so3_mul(out->ext_r, x->ext_r, dR2);

    /* ext_t += dx[9:12] */
    out->ext_t[0] = x->ext_t[0] + dx[9];
    out->ext_t[1] = x->ext_t[1] + dx[10];
    out->ext_t[2] = x->ext_t[2] + dx[11];

    /* vel += dx[12:15] */
    out->vel[0] = x->vel[0] + dx[12];
    out->vel[1] = x->vel[1] + dx[13];
    out->vel[2] = x->vel[2] + dx[14];

    /* bg += dx[15:18] */
    out->bg[0] = x->bg[0] + dx[15];
    out->bg[1] = x->bg[1] + dx[16];
    out->bg[2] = x->bg[2] + dx[17];

    /* ba += dx[18:21] */
    out->ba[0] = x->ba[0] + dx[18];
    out->ba[1] = x->ba[1] + dx[19];
    out->ba[2] = x->ba[2] + dx[20];

    /* grav += dx[21:24] */
    out->grav[0] = x->grav[0] + dx[21];
    out->grav[1] = x->grav[1] + dx[22];
    out->grav[2] = x->grav[2] + dx[23];
}

/* ================================================================
 * box_minus: result[24] = x1 ⊖ x2
 * Port of iekf.rs:box_minus
 * ================================================================ */
void box_minus(double result[24], const pp_state_t *x1, const pp_state_t *x2) {
    /* pos difference */
    result[0] = x1->pos[0] - x2->pos[0];
    result[1] = x1->pos[1] - x2->pos[1];
    result[2] = x1->pos[2] - x2->pos[2];

    /* rot difference: log(x2^{-1} * x1) */
    mat3d R2t, tmp;
    so3_transpose(R2t, x2->rot);
    so3_mul(tmp, R2t, x1->rot);
    so3_log(&result[3], tmp);

    /* ext_r difference */
    so3_transpose(R2t, x2->ext_r);
    so3_mul(tmp, R2t, x1->ext_r);
    so3_log(&result[6], tmp);

    /* ext_t difference */
    result[9]  = x1->ext_t[0] - x2->ext_t[0];
    result[10] = x1->ext_t[1] - x2->ext_t[1];
    result[11] = x1->ext_t[2] - x2->ext_t[2];

    /* vel, bg, ba, grav — all linear */
    for (int i=0;i<3;i++) result[12+i] = x1->vel[i]  - x2->vel[i];
    for (int i=0;i<3;i++) result[15+i] = x1->bg[i]   - x2->bg[i];
    for (int i=0;i<3;i++) result[18+i] = x1->ba[i]   - x2->ba[i];
    for (int i=0;i<3;i++) result[21+i] = x1->grav[i] - x2->grav[i];
}

/* ================================================================
 * Helper: get LiDAR pose in world frame
 * T_W_L = T_W_I * T_I_L
 * ================================================================ */
void iekf_get_lidar_pose(const pp_iekf_t *ekf,
                         double pos_world[3], double R_world[3][3]) {
    /* p_W_L = R_W_I * t_I_L + p_W_I */
    double tmp[3];
    mat3d_mulv(tmp, ekf->state.rot, ekf->state.ext_t);
    pos_world[0] = ekf->state.pos[0] + tmp[0];
    pos_world[1] = ekf->state.pos[1] + tmp[1];
    pos_world[2] = ekf->state.pos[2] + tmp[2];

    /* R_W_L = R_W_I * R_I_L */
    so3_mul(R_world, ekf->state.rot, ekf->state.ext_r);
}

/* ================================================================
 * get_f: state transition (24×1 column, written to f[24])
 * Port of iekf.rs:get_f
 * ================================================================ */
static void get_f(const pp_state_t *s, const double acc[3], const double gyro[3],
                  double f[24]) {
    for (int i=0;i<24;i++) f[i] = 0.0;

    /* omega = gyro - bg */
    double omega[3] = { gyro[0]-s->bg[0], gyro[1]-s->bg[1], gyro[2]-s->bg[2] };

    /* acc_diff = acc - ba */
    double acc_d[3] = { acc[0]-s->ba[0], acc[1]-s->ba[1], acc[2]-s->ba[2] };

    /* acc_inertial = R * acc_diff */
    double acc_i[3];
    mat3d_mulv(acc_i, s->rot, acc_d);

    /* f[0:3]   = vel  (position derivative) */
    f[0]=s->vel[0]; f[1]=s->vel[1]; f[2]=s->vel[2];

    /* f[3:6]   = omega (rotation derivative) */
    f[3]=omega[0]; f[4]=omega[1]; f[5]=omega[2];

    /* f[12:15] = acc_inertial + grav (velocity derivative) */
    f[12] = acc_i[0]+s->grav[0];
    f[13] = acc_i[1]+s->grav[1];
    f[14] = acc_i[2]+s->grav[2];
}

/* ================================================================
 * df_dx: Jacobian of f w.r.t. state (24×24)
 * Port of iekf.rs:df_dx
 * ================================================================ */
static void df_dx(const pp_state_t *s, const double acc[3],
                  double Fx[24][24]) {
    for (int i=0;i<24;i++) for (int j=0;j<24;j++) Fx[i][j]=0.0;

    /* d(pos)/d(vel) = I  -> block (0,12) */
    Fx[0][12]=1; Fx[1][13]=1; Fx[2][14]=1;

    /* d(rot)/d(bg) = -I  -> block (3,15) */
    Fx[3][15]=-1; Fx[4][16]=-1; Fx[5][17]=-1;

    /* d(vel)/d(rot) = -R * skew(acc - ba)  -> block (12,3) */
    double acc_d[3] = { acc[0]-s->ba[0], acc[1]-s->ba[1], acc[2]-s->ba[2] };
    mat3d K;
    skew3d(K, acc_d);
    /* vel_rot_block = -R * K */
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
        double v = 0.0;
        for (int k=0;k<3;k++) v += s->rot[i][k]*K[k][j];
        Fx[12+i][3+j] = -v;
    }

    /* d(vel)/d(ba) = -R  -> block (12,18) */
    for (int i=0;i<3;i++) for (int j=0;j<3;j++)
        Fx[12+i][18+j] = -s->rot[i][j];

    /* d(vel)/d(grav) = I  -> block (12,21) */
    Fx[12][21]=1; Fx[13][22]=1; Fx[14][23]=1;
}

/* ================================================================
 * df_dw: Jacobian of f w.r.t. noise (24×12)
 * Port of iekf.rs:df_dw
 * ================================================================ */
static void df_dw(const pp_state_t *s, double Fw[24][12]) {
    for (int i=0;i<24;i++) for (int j=0;j<12;j++) Fw[i][j]=0.0;

    /* d(rot)/d(n_g) = -I  -> block (3,0) */
    Fw[3][0]=-1; Fw[4][1]=-1; Fw[5][2]=-1;

    /* d(vel)/d(n_a) = -R  -> block (12,3) */
    for (int i=0;i<3;i++) for (int j=0;j<3;j++)
        Fw[12+i][3+j] = -s->rot[i][j];

    /* d(bg)/d(n_bg) = I  -> block (15,6) */
    Fw[15][6]=1; Fw[16][7]=1; Fw[17][8]=1;

    /* d(ba)/d(n_ba) = I  -> block (18,9) */
    Fw[18][9]=1; Fw[19][10]=1; Fw[20][11]=1;
}

/* ================================================================
 * iekf_predict
 * Port of Iekf::predict in iekf.rs
 * ================================================================ */
void iekf_predict(pp_iekf_t *ekf,
                  double dt,
                  const double Q[12][12],
                  const double acc[3],
                  const double gyro[3]) {
    ekf->last_time += dt;

    /* Rust computes f, df_dx and df_dw at the same linearisation point:
     * the pre-propagation state. Keep a local copy so covariance propagation
     * uses the same state as the nominal derivative evaluation. */
    pp_state_t state_before = ekf->state;

    double f_vec[24];
    get_f(&state_before, acc, gyro, f_vec);

    /* State update: x = box_plus(x, f*dt) */
    double f_dt[24];
    for (int i=0;i<24;i++) f_dt[i] = f_vec[i]*dt;
    pp_state_t new_state;
    box_plus(&new_state, &state_before, f_dt);
    ekf->state = new_state;

    /* Fx = I + df_dx * dt  (24×24) */
    double raw_Fx[24][24];
    df_dx(&state_before, acc, raw_Fx);

    mat24_identity(ekf->Fx);
    for (int i=0;i<24;i++) for (int j=0;j<24;j++)
        ekf->Fx[i][j] += raw_Fx[i][j]*dt;

    /* Fw = df_dw * dt  (24×12) */
    double Fw[24][12], Fw_raw[24][12];
    df_dw(&state_before, Fw_raw);
    for (int i=0;i<24;i++) for (int j=0;j<12;j++)
        Fw[i][j] = Fw_raw[i][j]*dt;

    /* P = Fx * P * Fx^T + Fw * Q * Fw^T */
    /* tmp = P * Fx^T */
    double FxT[24][24];
    for (int i=0;i<24;i++) for (int j=0;j<24;j++) FxT[i][j]=ekf->Fx[j][i];
    mat24_mul(ekf->tmp24a, ekf->P, FxT);   /* tmp24a = P * Fx^T */
    mat24_mul(ekf->tmp24b, ekf->Fx, ekf->tmp24a); /* tmp24b = Fx*P*Fx^T */

    /* Fw_Q_Fw = Fw * Q * Fw^T  (24×24) */
    /* First: (Q * Fw^T) -> tmp_12x24[12][24] */
    double QFwT[12][24];
    for (int i=0;i<12;i++) for (int j=0;j<24;j++) {
        double s=0.0;
        for (int k=0;k<12;k++) s += Q[i][k]*Fw[j][k]; /* Fw^T[k][j] = Fw[j][k] */
        QFwT[i][j] = s;
    }
    /* Fw * QFwT -> Fw_Q_Fw[24][24] */
    for (int i=0;i<24;i++) for (int j=0;j<24;j++) {
        double s=0.0;
        for (int k=0;k<12;k++) s += Fw[i][k]*QFwT[k][j];
        ekf->Fw_Q_Fw[i][j] = s;
    }

    mat24_add(ekf->P, ekf->tmp24b, ekf->Fw_Q_Fw);
}

/* ================================================================
 * iekf_correct
 * Port of Iekf::correct in iekf.rs
 * Only first 12 columns of H are non-zero (sparse H optimization).
 * ================================================================ */
void iekf_correct(pp_iekf_t *ekf,
                  const pelorus_pico_point_t *pts,
                  int n_pts,
                  pp_voxmap_t *vm,
                  pp_plane_meas_t *planes_scratch,
                  double measurement_time) {
    ekf->last_time = measurement_time;

    /* Save propagated state and covariance */
    pp_state_t state_prop = ekf->state;
    mat24_copy(ekf->P_propagated, ekf->P);

    double r_inv = 1.0 / PELORUS_LASER_POINT_NOISE;

    int converge_count = 0;
    int converge_flag  = 1;

    for (int iter = 0; iter <= PELORUS_EKF_MAX_ITER; iter++) {
        /* Re-extract plane correspondences at the current linearisation point.
         * Matches Rust after "revert: no plane cache" — correspondence set is
         * refreshed every iteration as the state estimate changes. */
        int n_planes = pp_planes_extract(pts, n_pts, vm, ekf, planes_scratch);

        /* Build H (m×12) and h (m) for current state estimate */
        int m = 0;

        for (int pi = 0; pi < n_planes && m < PELORUS_CLOUD_MAX_POINTS; pi++) {
            pp_plane_meas_t *pl = &planes_scratch[pi];
            double point_local[3] = {(double)pl->px, (double)pl->py, (double)pl->pz};
            double p_imu[3], p_world[3];
            mat3d_mulv(p_imu, ekf->state.ext_r, point_local);
            vec3d_add(p_imu, p_imu, ekf->state.ext_t);
            mat3d_mulv(p_world, ekf->state.rot, p_imu);
            vec3d_add(p_world, p_world, ekf->state.pos);

            /* Residual: h = -(n·p_world + d)  — matches Rust: h[i] = -pd2 */
            double norm[3] = {(double)pl->nx, (double)pl->ny, (double)pl->nz};
            double pd2 = norm[0] * p_world[0] + norm[1] * p_world[1] + norm[2] * p_world[2] + (double)pl->d;
            ekf->h_m[m] = -pd2;

            /* Jacobian H row (1×12):
             * H[0:3]  = n           (d(h)/d(pos))
             * H[3:6]  = skew(p_imu) * R_WI^T * n  (d(h)/d(delta_rot))
             * H[6:12] = 0           (extrinsic Jacobians omitted, matching Rust) */
            double *H_row = ekf->H_m_x12[m];
            for (int k=0;k<12;k++) H_row[k]=0.0;

            /* H[0:3] = n */
            H_row[0] = norm[0];
            H_row[1] = norm[1];
            H_row[2] = norm[2];

            /* H[3:6] = skew(p_imu) * (R_WI^T * n) */
            double c[3];
            c[0] = ekf->state.rot[0][0] * norm[0] + ekf->state.rot[1][0] * norm[1] + ekf->state.rot[2][0] * norm[2];
            c[1] = ekf->state.rot[0][1] * norm[0] + ekf->state.rot[1][1] * norm[1] + ekf->state.rot[2][1] * norm[2];
            c[2] = ekf->state.rot[0][2] * norm[0] + ekf->state.rot[1][2] * norm[1] + ekf->state.rot[2][2] * norm[2];

            mat3d Kp;
            skew3d(Kp, p_imu);
            H_row[3] = Kp[0][0] * c[0] + Kp[0][1] * c[1] + Kp[0][2] * c[2];
            H_row[4] = Kp[1][0] * c[0] + Kp[1][1] * c[1] + Kp[1][2] * c[2];
            H_row[5] = Kp[2][0] * c[0] + Kp[2][1] * c[1] + Kp[2][2] * c[2];

            m++;
        }

        if (m == 0) continue;

        /* dx_new = box_minus(state, state_prop) */
        box_minus(ekf->dx_new, &ekf->state, &state_prop);

        /* HTH[12][12] = H^T * H */
        for (int i=0;i<12;i++) for (int j=0;j<12;j++) {
            double s=0.0;
            for (int k=0;k<m;k++) s += ekf->H_m_x12[k][i]*ekf->H_m_x12[k][j];
            ekf->HTH12[i][j] = s;
        }

        /* P_inv = P_propagated^{-1} */
        if (mat24_lup_inverse(ekf->P_inv, ekf->P_propagated) != 0) break;

        /* Build full 24×24 lhs = HTH * r_inv + P_inv
         * HTH is only non-zero in the top-left 12×12 block. */
        mat24_zero(ekf->tmp24a);
        for (int i=0;i<12;i++) for (int j=0;j<12;j++)
            ekf->tmp24a[i][j] = ekf->HTH12[i][j]*r_inv;
        mat24_add(ekf->tmp24a, ekf->tmp24a, ekf->P_inv);

        /* K_front = lhs^{-1} (full 24×24 inverse, stored in tmp24b) */
        if (mat24_lup_inverse(ekf->tmp24b, ekf->tmp24a) != 0) break;

        /* kf24x12 = first 12 columns of K_front */
        for (int i=0;i<24;i++) for (int j=0;j<12;j++)
            ekf->kf24x12[i][j] = ekf->tmp24b[i][j];

        /* K = K_front_12 * H^T * r_inv  (24×12 × 12×m → need (24×m) intermediate)
         * But we only need K*h (24×1) and K*H (24×12).
         * K*H[24][12] = kf24x12 * HTH12 * r_inv */
        for (int i=0;i<24;i++) for (int j=0;j<12;j++) {
            double s=0.0;
            for (int k=0;k<12;k++) s += ekf->kf24x12[i][k]*ekf->HTH12[k][j];
            ekf->KH24x12[i][j] = s*r_inv;
        }

        /* KH24: full 24×24 matrix (zero-extend columns 12-23) */
        mat24_zero(ekf->KH24);
        for (int i=0;i<24;i++) for (int j=0;j<12;j++)
            ekf->KH24[i][j] = ekf->KH24x12[i][j];

        /* K * h (24×1): first compute (kf24x12 * H^T) then multiply by h */
        /* H_T_h[12] = H^T * h_m (m×12 transposed × m = 12) */
        double HT_h[12];
        for (int j=0;j<12;j++) {
            double s=0.0;
            for (int k=0;k<m;k++) s += ekf->H_m_x12[k][j]*ekf->h_m[k];
            HT_h[j] = s;
        }
        /* K_h[24] = kf24x12 * HT_h * r_inv */
        double K_h_vec[24];
        for (int i=0;i<24;i++) {
            double s=0.0;
            for (int k=0;k<12;k++) s += ekf->kf24x12[i][k]*HT_h[k];
            K_h_vec[i] = s*r_inv;
        }

        /* dx_ = K_h + (KH - I) * dx_new */
        for (int i=0;i<24;i++) {
            double kh_i_dx = 0.0;
            for (int j=0;j<24;j++) kh_i_dx += ekf->KH24[i][j]*ekf->dx_new[j];
            kh_i_dx -= ekf->dx_new[i]; /* (KH - I) * dx_new */
            ekf->dx_[i] = K_h_vec[i] + kh_i_dx;
        }

        /* NaN check */
        int has_nan = 0;
        for (int i=0;i<24;i++) if (ekf->dx_[i] != ekf->dx_[i]) { has_nan=1; break; }
        if (has_nan) { ekf->state = state_prop; mat24_copy(ekf->P, ekf->P_propagated); return; }

        pp_state_t new_s;
        box_plus(&new_s, &ekf->state, ekf->dx_);
        ekf->state = new_s;

        /* Convergence check */
        converge_flag = 1;
        for (int j=0;j<24;j++) {
            if (PP_FABS(ekf->dx_[j]) > PELORUS_LASER_POINT_NOISE) { converge_flag=0; break; }
        }
        if (converge_flag) converge_count++;
        if (converge_count == 0 && iter == PELORUS_EKF_MAX_ITER-1) converge_flag=1;

        if (converge_count > 1 || iter == PELORUS_EKF_MAX_ITER) break;
    }

    /* Covariance update: P = (I - KH) * P_propagated */
    mat24 I_KH;
    mat24_identity(I_KH);
    mat24_sub(I_KH, I_KH, ekf->KH24);
    mat24_mul(ekf->P, I_KH, ekf->P_propagated);
}

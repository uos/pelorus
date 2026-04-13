/* pp_imu_int.c — IMU ring buffer, gravity init, integration
 * Port of pelorus_reg/src/imu.rs
 */
#include "pp_imu_int.h"
#include "pp_linalg.h"
#include "../include/pelorus_pico_config.h"

#define PP_SQRT(x) __builtin_sqrt(x)

/* ================================================================
 * Lifecycle
 * ================================================================ */
void imu_state_init(pp_imu_state_t *st,
                    const double lidar_to_imu_r[3][3],
                    const double lidar_to_imu_t[3],
                    const double base_to_lidar_r[3][3],
                    const double base_to_lidar_t[3]) {
    /* Zero everything */
    int sz = sizeof(pp_imu_state_t);
    char *p = (char *)st;
    for (int i=0;i<sz;i++) p[i]=0;

    /* Copy extrinsics */
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
        st->lidar_to_imu_r[i][j] = lidar_to_imu_r[i][j];
    }
    for (int i=0;i<3;i++) {
        st->lidar_to_imu_t[i] = lidar_to_imu_t[i];
    }

    /* lidar_to_base_r = base_to_lidar_r^T  (R_B_L = R_L_B^T) */
    for (int i=0;i<3;i++) for (int j=0;j<3;j++)
        st->lidar_to_base_r[i][j] = base_to_lidar_r[j][i];
    /* lidar_to_base_t: t_B_L in base frame = -R_L_B^T * t_B_L_in_lidar
     * We store the result of the inverse transform:
     * T_LB = T_BL^{-1}: t_L_B = -R_BL^T * t_BL_pos = R_LB * (-t_BL)
     * For simplicity store raw base_to_lidar_t negated in base frame.
     * pelorus_reg uses base_to_lidar.inverse().translation.vector */
    {
        double neg_t[3] = {-base_to_lidar_t[0],-base_to_lidar_t[1],-base_to_lidar_t[2]};
        mat3d_mulv(st->lidar_to_base_t, st->lidar_to_base_r, neg_t);
    }

    /* Build Q matrix (12×12 diagonal)
     * [n_g(3), n_a(3), n_bg(3), n_ba(3)] */
    st->Q[0][0]=st->Q[1][1]=st->Q[2][2]   = PELORUS_COV_GYR;
    st->Q[3][3]=st->Q[4][4]=st->Q[5][5]   = PELORUS_COV_ACC;
    st->Q[6][6]=st->Q[7][7]=st->Q[8][8]   = PELORUS_COV_GYR_BIAS;
    st->Q[9][9]=st->Q[10][10]=st->Q[11][11]= PELORUS_COV_ACC_BIAS;
}

/* ================================================================
 * Ring buffer
 * ================================================================ */
void imu_push(pp_imu_state_t *st, const pp_imu_msg_t *msg) {
    st->buf[st->tail] = *msg;
    st->tail = (st->tail + 1) % PELORUS_IMU_BUFFER_MAX;
    if (st->count < PELORUS_IMU_BUFFER_MAX) {
        st->count++;
    } else {
        /* Overwrite: advance head (drop oldest) */
        st->head = (st->head + 1) % PELORUS_IMU_BUFFER_MAX;
    }
}

int imu_peek_oldest(const pp_imu_state_t *st, pp_imu_msg_t *msg) {
    if (st->count == 0) return -1;
    *msg = st->buf[st->head];
    return 0;
}

int imu_peek_newest(const pp_imu_state_t *st, pp_imu_msg_t *msg) {
    if (st->count == 0) return -1;
    int idx = (st->tail - 1 + PELORUS_IMU_BUFFER_MAX) % PELORUS_IMU_BUFFER_MAX;
    *msg = st->buf[idx];
    return 0;
}

/* ================================================================
 * IMU initialisation
 * Port of ImuProcessing::initialize in imu.rs
 * ================================================================ */
int imu_init_gravity(pp_imu_state_t *st, const pp_imu_msg_t *msg, pp_iekf_t *ekf) {
    if (st->init_count == 0) {
        st->init_start_time = msg->t;
    } else if (msg->t < st->init_start_time) {
        /* Time jumped backwards; restart the init window like the higher-level
         * Rust pipeline does when it sees earlier timestamps. */
        st->init_count = 0;
        st->init_start_time = msg->t;
    }

    if ((msg->t - st->init_start_time) < PELORUS_INIT_IMU_DURATION_SEC) {
        if (st->init_count < PELORUS_INIT_IMU_MAX_SAMPLES) {
            st->init_buf[st->init_count++] = *msg;
        }
        return 0;
    }

    if (st->init_count <= 0) {
        return 0;
    }

    /* Compute running mean of acc and gyro (Welford) */
    double mean_acc[3] = {0,0,0};
    double mean_gyr[3] = {0,0,0};
    int n = st->init_count;
    for (int k=0;k<n;k++) {
        for (int i=0;i<3;i++) {
            mean_acc[i] += (st->init_buf[k].acc[i] - mean_acc[i]) / (k+1.0);
            mean_gyr[i] += (st->init_buf[k].gyro[i] - mean_gyr[i]) / (k+1.0);
        }
    }
    vec3d_copy(st->mean_acc, mean_acc);
    vec3d_copy(st->mean_gyr, mean_gyr);
    st->mean_acc_norm = vec3d_norm(mean_acc);
    ekf->mean_acc_norm = st->mean_acc_norm;

    /* Set gyro bias */
    vec3d_copy(ekf->state.bg, mean_gyr);

    /* Set extrinsics in state */
    for (int i=0;i<3;i++) for (int j=0;j<3;j++)
        ekf->state.ext_r[i][j] = st->lidar_to_imu_r[i][j];
    vec3d_copy(ekf->state.ext_t, st->lidar_to_imu_t);

    /* R_W_I: world frame aligns with base frame initially.
     * R_W_I = (R_I_L * R_L_B)^{-1} = R_B_L * R_L_I
     * R_L_I = R_I_L^T = lidar_to_imu_r^T */
    double R_LI[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++)
        R_LI[i][j] = st->lidar_to_imu_r[j][i];
    /* R_W_I = R_B_L * R_L_I */
    double R_WI[3][3];
    so3_mul(R_WI, st->lidar_to_base_r, R_LI);
    for (int i=0;i<3;i++) for (int j=0;j<3;j++)
        ekf->state.rot[i][j] = R_WI[i][j];

    /* Initial position: base origin at world origin.
     * p_W_I = -R_W_I * (R_I_L * p_L_B + t_I_L) */
    double p_LB[3]; vec3d_copy(p_LB, st->lidar_to_base_t);
    double R_IL_pLB[3];
    mat3d_mulv(R_IL_pLB, st->lidar_to_imu_r, p_LB);
    double p_IB[3];
    vec3d_add(p_IB, R_IL_pLB, st->lidar_to_imu_t);
    double R_WI_p[3];
    mat3d_mulv(R_WI_p, R_WI, p_IB);
    for (int i=0;i<3;i++) ekf->state.pos[i] = -R_WI_p[i];

    /* Gravity in world frame: g_W = R_W_I * g_IMU
     * g_IMU = -mean_acc / |mean_acc| * 9.81 */
    double g_imu[3], norm = st->mean_acc_norm;
    if (norm < 1e-9) norm = 9.81;
    for (int i=0;i<3;i++) g_imu[i] = -mean_acc[i] / norm * PELORUS_GRAVITY;
    double g_W[3];
    mat3d_mulv(g_W, R_WI, g_imu);
    vec3d_copy(ekf->state.grav, g_W);

    /* Initial covariance */
    mat24_identity(ekf->P);
    ekf->P[15][15] = ekf->P[16][16] = ekf->P[17][17] = PELORUS_COV_INIT_BG;
    ekf->P[18][18] = ekf->P[19][19] = ekf->P[20][20] = PELORUS_COV_INIT_BA;
    ekf->P[21][21] = ekf->P[22][22] = ekf->P[23][23] = PELORUS_COV_INIT_GRAV;

    ekf->initialized = 1;
    return 1;
}

/* ================================================================
 * IMU integration over [t_start, t_end]
 * Port of ImuProcessing::undistort (the prediction loop) in imu.rs
 * ================================================================ */
void imu_integrate_range(pp_imu_state_t *st,
                         pp_iekf_t *ekf,
                         double last_cloud_end,
                         double t_start,
                         double t_end) {
    (void)last_cloud_end;
    st->n_imu_poses = 0;

    /* Initial pose */
    {
        pp_imu_pose_t *ip = &st->imu_poses[st->n_imu_poses++];
        ip->offset_time = 0.0;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) ip->pose_rot[i][j]=ekf->state.rot[i][j];
        vec3d_copy(ip->pose_pos, ekf->state.pos);
        vec3d_copy(ip->vel, ekf->state.vel);
        vec3d_copy(ip->acc, st->acc_s_last);
        vec3d_copy(ip->gyr, st->angvel_last);
    }

    if (st->count == 0) return;

    /* Match pelorus_c behavior in equality flow: integrate strictly over
     * this cloud's [t_start, t_end] IMU span. */
    double effective_start = t_start;
    ekf->last_time = effective_start;

    /* Find the starting entry: most recent buffered message at or before
     * effective_start.  If none qualify, start from the oldest message. */
    pp_imu_msg_t prev_msg;
    {
        prev_msg = st->buf[st->head];
        int ci = st->head;
        for (int k=0;k<st->count;k++) {
            if (st->buf[ci].t <= effective_start) prev_msg = st->buf[ci];
            ci = (ci+1) % PELORUS_IMU_BUFFER_MAX;
        }
    }

    /* Keep last raw acc/gyro passed to iekf_predict so the final step can
     * use the same values as the Rust code (avr_x / angvel_avr from last pair). */
    double last_acc_avr[3] = {0,0,PELORUS_GRAVITY};
    double last_gyro_avr[3] = {0,0,0};

    /* Trapezoidal integration over buffered messages */
    int ci = st->head;
    double acc_norm = ekf->mean_acc_norm > 1e-9 ? ekf->mean_acc_norm : PELORUS_GRAVITY;
    for (int k=0;k<st->count;k++) {
        pp_imu_msg_t *tail = &st->buf[ci];
        double tail_t = tail->t;
        double head_t = prev_msg.t;

        if (head_t == tail_t) { prev_msg = *tail; ci=(ci+1)%PELORUS_IMU_BUFFER_MAX; continue; }
        if (tail_t <= effective_start) { prev_msg = *tail; ci=(ci+1)%PELORUS_IMU_BUFFER_MAX; continue; }

        /* Trapezoidal averages — same normalization as Rust imu.rs */
        double gyro_avr[3], acc_avr[3];
        for (int i=0;i<3;i++) {
            gyro_avr[i] = (prev_msg.gyro[i] + tail->gyro[i]) * 0.5;
            acc_avr[i]  = (prev_msg.acc[i]  + tail->acc[i])  * 0.5 * PELORUS_GRAVITY / acc_norm;
        }

        double dt = (head_t < effective_start) ? (tail_t - effective_start)
                                               : (tail_t - head_t);
        if (dt > 0.0) {
            iekf_predict(ekf, dt, (const double (*)[12])st->Q, acc_avr, gyro_avr);
            for (int i=0;i<3;i++) { last_acc_avr[i] = acc_avr[i]; last_gyro_avr[i] = gyro_avr[i]; }
        }

        /* Compute world-frame acc+grav and bias-corrected gyro for imu_poses.
         * Match Rust exactly: acc_s_last uses the tail raw acceleration
         * (normalized), not the trapezoidal average used for predict(). */
        double angvel_bc[3];
        for (int i=0;i<3;i++) angvel_bc[i] = tail->gyro[i] - ekf->state.bg[i];
        double acc_bc[3];
        for (int i=0;i<3;i++) {
            double tail_acc_norm = tail->acc[i] * PELORUS_GRAVITY / acc_norm;
            acc_bc[i] = tail_acc_norm - ekf->state.ba[i];
        }
        double acc_world[3];
        mat3d_mulv(acc_world, ekf->state.rot, acc_bc);
        double acc_s[3];
        for (int i=0;i<3;i++) acc_s[i] = acc_world[i] + ekf->state.grav[i];

        vec3d_copy(st->angvel_last, angvel_bc);
        vec3d_copy(st->acc_s_last, acc_s);

        /* Store pose snapshot for undistortion */
        if (st->n_imu_poses < PELORUS_IMU_BUFFER_MAX) {
            pp_imu_pose_t *ip = &st->imu_poses[st->n_imu_poses++];
            ip->offset_time = tail_t - t_start;
            for (int i=0;i<3;i++) for (int j=0;j<3;j++) ip->pose_rot[i][j]=ekf->state.rot[i][j];
            vec3d_copy(ip->pose_pos, ekf->state.pos);
            vec3d_copy(ip->vel, ekf->state.vel);
            vec3d_copy(ip->acc, acc_s);
            vec3d_copy(ip->gyr, angvel_bc);
        }

        prev_msg = *tail;
        if (tail_t >= t_end) break;
        ci = (ci+1) % PELORUS_IMU_BUFFER_MAX;
    }

    /* Final step from last processed message to t_end, using the last
     * raw acc/gyro averages (mirrors Rust imu.rs final predict). */
    {
        double dt = t_end - ekf->last_time;
        if (dt > 0.0 && dt < 2.0) { /* sanity cap: skip if dt is unreasonable */
            double gyro_a[3], acc_a[3];
            for (int i=0;i<3;i++) {
                gyro_a[i] = last_gyro_avr[i];
                acc_a[i]  = last_acc_avr[i];
            }
            iekf_predict(ekf, dt, (const double (*)[12])st->Q, acc_a, gyro_a);
        }
    }
}

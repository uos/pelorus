/* pelorus_pico.c — top-level API and wiring
 *
 * All state lives in a single static g_ctx.  No heap.
 * The three Rust workers (filter_downsample, sync_lidar_imu, registration)
 * collapse into the single blocking call sequence inside add_cloud.
 */
#include "../include/pelorus_pico.h"
#include "../include/pelorus_pico_config.h"
#include "pp_linalg.h"
#include "pp_voxmap.h"
#include "pp_iekf.h"
#include "pp_imu_int.h"
#include "pp_downsample.h"
#include "pp_undistort.h"

/* ------------------------------------------------------------------ */
/* Static context — single allocation, no heap                         */
/* ------------------------------------------------------------------ */

typedef struct {
    pp_iekf_t      ekf;
    pp_imu_state_t imu;
    pp_voxmap_t    voxmap;

    /* Shared raw/work point buffer.
     * After pp_downsample work_n tracks the valid count. */
    pelorus_pico_point_t pts[PELORUS_CLOUD_MAX_POINTS];

    /* Scratch for downsample pass (must not be on the stack). */
    pp_ds_acc_t ds_acc[PELORUS_CLOUD_MAX_POINTS];

    /* Scratch for grouped voxel-map updates. */
    pp_vox_acc_t map_acc[PELORUS_CLOUD_MAX_POINTS];

    /* Plane measurements output (one slot per downsampled point at most). */
    pp_plane_meas_t planes[PELORUS_CLOUD_MAX_POINTS];

    uint32_t scan_id;
    double   last_cloud_end;
    int      initialized;      /* 1 once imu_init_gravity() succeeds */
} pp_ctx_t;

static pp_ctx_t g_ctx;

/* ------------------------------------------------------------------ */
/* Internal helpers                                                    */
/* ------------------------------------------------------------------ */

/* Fill a pelorus_odometry_t from the current EKF state. */
static pelorus_odometry_t ctx_to_odom(double ts, int valid)
{
    pelorus_odometry_t o;
    o.timestamp = ts;
    o.valid     = valid;

    const pp_state_t *s = &g_ctx.ekf.state;
    double R_WL[3][3], p_WL[3];
    double R_WB[3][3], p_WB[3];

    /* Match pelorus_c: publish the pose of the output frame (base when a
     * base-to-lidar transform is configured, otherwise the lidar frame). */
    so3_mul(R_WL, s->rot, s->ext_r);
    mat3d_mulv(p_WL, s->rot, s->ext_t);
    vec3d_add(p_WL, p_WL, s->pos);

    so3_mul(R_WB, R_WL, g_ctx.imu.lidar_to_base_r);
    mat3d_mulv(p_WB, R_WL, g_ctx.imu.lidar_to_base_t);
    vec3d_add(p_WB, p_WB, p_WL);

    o.position[0] = p_WB[0];
    o.position[1] = p_WB[1];
    o.position[2] = p_WB[2];

    /* pelorus_c exposes linear velocity in the published body frame. */
    {
        double R_BW[3][3];
        so3_transpose(R_BW, R_WB);
        mat3d_mulv(o.linear_velocity, R_BW, s->vel);
    }

    /* Convert rotation matrix to quaternion [w, x, y, z]. */
    {
        double qd[4];
        so3_to_quat(qd, R_WB);
        o.orientation[0] = qd[0];
        o.orientation[1] = qd[1];
        o.orientation[2] = qd[2];
        o.orientation[3] = qd[3];
    }

    return o;
}

static pelorus_odometry_t odom_invalid(void)
{
    pelorus_odometry_t o;
    int i;
    o.timestamp = 0.0;
    o.valid = 0;
    for (i = 0; i < 3; i++) { o.position[i] = 0.0; o.linear_velocity[i] = 0.0; }
    o.orientation[0] = 1.0;
    o.orientation[1] = o.orientation[2] = o.orientation[3] = 0.0;
    return o;
}

static void ctx_transform_points_to_world(int n_pts)
{
    const pp_state_t *s = &g_ctx.ekf.state;

    for (int i = 0; i < n_pts; i++) {
        double p_l[3] = { g_ctx.pts[i].x, g_ctx.pts[i].y, g_ctx.pts[i].z };
        double p_imu[3], p_w[3];
        mat3d_mulv(p_imu, s->ext_r, p_l);
        vec3d_add(p_imu, p_imu, s->ext_t);
        mat3d_mulv(p_w, s->rot, p_imu);
        vec3d_add(p_w, p_w, s->pos);
        g_ctx.pts[i].x = (float)p_w[0];
        g_ctx.pts[i].y = (float)p_w[1];
        g_ctx.pts[i].z = (float)p_w[2];
    }
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

void pelorus_pico_init(void)
{
    /* Zero the entire context first. */
    int i;
    unsigned char *p = (unsigned char *)&g_ctx;
    for (i = 0; i < (int)sizeof(g_ctx); i++) p[i] = 0;

    /* Build extrinsic matrices from config defines. */
    double lidar_imu_r_flat[9] = PELORUS_LIDAR_IMU_R;
    double lidar_imu_t_arr[3]  = PELORUS_LIDAR_IMU_T;
    double base_lidar_r_flat[9] = PELORUS_BASE_LIDAR_R;
    double base_lidar_t_arr[3]  = PELORUS_BASE_LIDAR_T;

    double lidar_imu_r[3][3];
    double base_lidar_r[3][3];
    for (i = 0; i < 3; i++) {
        int j;
        for (j = 0; j < 3; j++) {
            lidar_imu_r[i][j]  = lidar_imu_r_flat[i*3+j];
            base_lidar_r[i][j] = base_lidar_r_flat[i*3+j];
        }
    }

    iekf_init(&g_ctx.ekf);
    imu_state_init(&g_ctx.imu,
                   (const double(*)[3])lidar_imu_r,
                   lidar_imu_t_arr,
                   (const double(*)[3])base_lidar_r,
                   base_lidar_t_arr);
    voxmap_init(&g_ctx.voxmap);

    g_ctx.scan_id        = 0;
    g_ctx.last_cloud_end = 0.0;
    g_ctx.initialized    = 0;
}

/* ------------------------------------------------------------------ */

pelorus_odometry_t pelorus_pico_add_imu(
    double timestamp,
    float  gyro_x,  float gyro_y,  float gyro_z,
    float  accel_x, float accel_y, float accel_z)
{
    pp_imu_msg_t msg;
    msg.t       = timestamp;
    msg.gyro[0] = (double)gyro_x;
    msg.gyro[1] = (double)gyro_y;
    msg.gyro[2] = (double)gyro_z;
    msg.acc[0]  = (double)accel_x;
    msg.acc[1]  = (double)accel_y;
    msg.acc[2]  = (double)accel_z;

    if (!g_ctx.initialized) {
        /* Try to accumulate enough samples for gravity init. */
        if (imu_init_gravity(&g_ctx.imu, &msg, &g_ctx.ekf)) {
            g_ctx.initialized = 1;
        }
        imu_push(&g_ctx.imu, &msg);
        return odom_invalid();
    }

    imu_push(&g_ctx.imu, &msg);
    /* EKF predict is NOT called here.
     * All IMU integration happens exclusively in imu_integrate_range()
     * which is called from add_cloud.  This matches the Rust pipeline where
     * the sync_lidar_imu worker is the only path that advances the EKF state.
     * add_imu just accumulates samples; it returns the last committed pose. */

    return ctx_to_odom(timestamp, g_ctx.initialized);
}

/* ------------------------------------------------------------------ */

pelorus_odometry_t pelorus_pico_add_cloud(
    const pelorus_pico_point_t *pts,
    int    n,
    double cloud_start_time,
    double cloud_end_time)
{
    int i;

    /* 1. Copy input into shared buffer (bounds-check). */
    int n_raw = n;
    if (n_raw > PELORUS_CLOUD_MAX_POINTS) n_raw = PELORUS_CLOUD_MAX_POINTS;
    for (i = 0; i < n_raw; i++) g_ctx.pts[i] = pts[i];

    /* 2. Voxel-grid downsample (in-place, work_n ≤ n_raw). */
    int work_n = 0;
    pp_downsample(g_ctx.pts, n_raw, &work_n, g_ctx.ds_acc);

    /* 3. System not yet initialised → return invalid. */
    if (!g_ctx.initialized) {
        return odom_invalid();
    }

    /* 4. First scan ever: bootstrap the map.
     * Like the Rust/Prelio pipeline, the bootstrap scan is map-only, but it
     * still advances the previous lidar end timestamp so the next real scan
     * integrates the full IMU span since this scan ended. */
    if (voxmap_is_empty(&g_ctx.voxmap)) {
        if (g_ctx.voxmap.occupied_count >=
            (uint32_t)(PELORUS_VOXEL_CAPACITY * PELORUS_LRU_TRIGGER_FRACTION)) {
            voxmap_lru_evict(&g_ctx.voxmap, g_ctx.scan_id);
        }
        ctx_transform_points_to_world(work_n);
        voxmap_add_points(&g_ctx.voxmap, g_ctx.pts, work_n, g_ctx.scan_id, g_ctx.map_acc);
        /* Prelio updates prev_lidar_end_time after every processed scan,
         * including map-only bootstrap scans. Mirror that here so the first
         * real scan integrates from this scan's end time. */
        g_ctx.last_cloud_end = cloud_end_time;
        g_ctx.ekf.last_time = cloud_end_time;
        g_ctx.scan_id++;
        return ctx_to_odom(cloud_end_time, 0); /* first scan: not yet valid */
    }

    /* 5. Integrate IMU over the cloud interval, build imu_poses[]. */
    imu_integrate_range(&g_ctx.imu, &g_ctx.ekf,
                        g_ctx.last_cloud_end,
                        cloud_start_time,
                        cloud_end_time);
    /* 6. Deskew (undistort) points using accumulated IMU poses. */
    pp_undistort(g_ctx.pts, work_n, &g_ctx.imu, &g_ctx.ekf);

    /* 7. IEKF measurement update.
     * Plane correspondences are re-extracted on every IEKF iteration inside
     * iekf_correct (matching Rust after "revert: no plane cache"). */
    iekf_correct(&g_ctx.ekf,
                 g_ctx.pts, work_n,
                 &g_ctx.voxmap,
                 g_ctx.planes,
                 cloud_end_time);

    /* 9. Insert this scan's points into the map (world-frame transform). */
    if (g_ctx.voxmap.occupied_count >=
        (uint32_t)(PELORUS_VOXEL_CAPACITY * PELORUS_LRU_TRIGGER_FRACTION)) {
        voxmap_lru_evict(&g_ctx.voxmap, g_ctx.scan_id);
    }

    ctx_transform_points_to_world(work_n);
    voxmap_add_points(&g_ctx.voxmap, g_ctx.pts, work_n, g_ctx.scan_id, g_ctx.map_acc);

    g_ctx.last_cloud_end = cloud_end_time;
    g_ctx.ekf.last_time  = cloud_end_time;
    g_ctx.scan_id++;

    return ctx_to_odom(cloud_end_time, 1);
}

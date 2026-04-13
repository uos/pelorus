/* equality_test.c — compare pelorus_pico output against pelorus_c (Rust)
 *
 * Build (from repo root):
 *   cargo build -p pelorus_c --release
 *
 *   gcc -O2 -std=c99 \
 *       -Ipelorus_pico/include \
 *       -Ipelorus_c \
 *       pelorus_pico/src/pp_linalg.c \
 *       pelorus_pico/src/pp_voxmap.c \
 *       pelorus_pico/src/pp_iekf.c \
 *       pelorus_pico/src/pp_imu_int.c \
 *       pelorus_pico/src/pp_downsample.c \
 *       pelorus_pico/src/pp_undistort.c \
 *       pelorus_pico/src/pp_planes.c \
 *       pelorus_pico/src/pelorus_pico.c \
 *       pelorus_pico/tests/equality_test.c \
 *       -Ltarget/release -lpelorus_c \
 *       -lpthread -ldl -lm \
 *       -Wl,-rpath,target/release \
 *       -o eq_test
 *
 *   ./eq_test recorded_sequence.bin
 *
 * Binary sequence format (written by record_sequence.c):
 *   Each record starts with a uint8_t tag:
 *     TAG_IMU   = 1
 *     TAG_CLOUD = 2
 *
 *   IMU record  (32 bytes after tag):
 *     double  timestamp    (absolute, seconds)
 *     float   gx, gy, gz  (rad/s)
 *     float   ax, ay, az  (m/s²)
 *
 *   CLOUD record:
 *     double   t_start       (absolute, seconds)
 *     double   t_end         (absolute, seconds)
 *     int32_t  n_pts
 *     n_pts × 5 floats: x, y, z, intensity, timestamp_rel
 *                        (timestamp_rel in seconds relative to t_start)
 *
 * Pass criteria:
 *   position error per scan  < POS_THRESH (0.01 m)
 *   orientation quat dot     > ORI_THRESH (0.9999)
 *   velocity error           < VEL_THRESH (0.1 m/s)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "../include/pelorus_pico.h"
#include "../../pelorus_c/pelorus.h"

/* ---- pass/fail thresholds ---- */
#define POS_THRESH  0.01   /* metres */
#define ORI_THRESH  0.9999 /* quaternion dot product */
#define VEL_THRESH  0.10   /* m/s */

/* ---- binary record tags ---- */
#define TAG_IMU   1
#define TAG_CLOUD 2

/* ---- helpers ---- */
static double pos_error(const double a[3], const double b[3])
{
    double dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
    return sqrt(dx*dx + dy*dy + dz*dz);
}

static double quat_dot(const double a[4], const double b[4])
{
    return fabs(a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]);
}

static double vel_error(const double a[3], const double b[3])
{
    double dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
    return sqrt(dx*dx + dy*dy + dz*dz);
}

/* Split absolute seconds into (sec, nsec) for pelorus_c */
static void split_ts(double t, int32_t *sec, uint32_t *nsec)
{
    *sec  = (int32_t)t;
    *nsec = (uint32_t)((t - (double)*sec) * 1e9 + 0.5);
}

/* ---- pelorus_c synchronous wrapper ---- */
typedef struct {
    struct PelorusHandle *handle;
    /* last odometry from recv */
    double position[3];
    double orientation[4];  /* w,x,y,z */
    double linear_velocity[3];
    int    valid;
} pc_ctx_t;

static void pc_init(pc_ctx_t *ctx, const char *config_path)
{
    ctx->handle = pelorus_create(config_path);
    if (!ctx->handle) {
        fprintf(stderr, "pelorus_create() failed\n");
        exit(1);
    }
    ctx->valid = 0;
}

static void pc_add_imu(pc_ctx_t *ctx, double t,
                       float gx, float gy, float gz,
                       float ax, float ay, float az)
{
    pelorus_imu_msg msg;
    memset(&msg, 0, sizeof(msg));
    split_ts(t, &msg.timestamp_sec, &msg.timestamp_nanosec);
    msg.angular_velocity[0]    = (double)gx;
    msg.angular_velocity[1]    = (double)gy;
    msg.angular_velocity[2]    = (double)gz;
    msg.linear_acceleration[0] = (double)ax;
    msg.linear_acceleration[1] = (double)ay;
    msg.linear_acceleration[2] = (double)az;
    pelorus_add_imu(ctx->handle, &msg);
}

/* Returns 1 if a result was received, 0 if no result yet. */
static int pc_add_cloud(pc_ctx_t *ctx,
                        const pelorus_pico_point_t *pts, int n,
                        double t_start, double t_end)
{
    /* Pack into xyzit float array.
     * pelorus_c expects t field in nanoseconds offset from header timestamp. */
    float *buf = (float *)malloc((size_t)n * 5 * sizeof(float));
    if (!buf) { fprintf(stderr, "malloc failed\n"); exit(1); }

    int i;
    for (i = 0; i < n; i++) {
        buf[i*5+0] = pts[i].x;
        buf[i*5+1] = pts[i].y;
        buf[i*5+2] = pts[i].z;
        buf[i*5+3] = pts[i].intensity;
        /* timestamp_rel is seconds from t_start; convert to nanoseconds */
        buf[i*5+4] = pts[i].timestamp * 1e9f;
    }

    int32_t  hdr_sec;
    uint32_t hdr_nsec;
    split_ts(t_start, &hdr_sec, &hdr_nsec);

    uint64_t scan_id = pelorus_add_points_xyzit_f32(
        ctx->handle, hdr_sec, hdr_nsec, "lidar",
        buf, (size_t)n);
    free(buf);

    if (scan_id == 0) return 0; /* not yet processed */

    /* Block until the result for this scan is available. */
    pelorus_odometry odom;
    memset(&odom, 0, sizeof(odom));
    int32_t r = pelorus_recv(ctx->handle, &odom);
    if (r != 1) return 0;

    ctx->position[0]      = odom.position[0];
    ctx->position[1]      = odom.position[1];
    ctx->position[2]      = odom.position[2];
    /* pelorus_c orientation: [w, x, y, z] — same as pelorus_pico */
    ctx->orientation[0]   = odom.orientation[0]; /* w */
    ctx->orientation[1]   = odom.orientation[1]; /* x */
    ctx->orientation[2]   = odom.orientation[2]; /* y */
    ctx->orientation[3]   = odom.orientation[3]; /* z */
    ctx->linear_velocity[0] = odom.linear_velocity[0];
    ctx->linear_velocity[1] = odom.linear_velocity[1];
    ctx->linear_velocity[2] = odom.linear_velocity[2];
    ctx->valid = 1;

    pelorus_free_odometry(&odom);
    return 1;
}

/* ---- main ---- */
int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <recorded_sequence.bin> [pelorus_config.rl]\n", argv[0]);
        return 1;
    }
    const char *seq_path    = argv[1];
    const char *config_path = (argc >= 3) ? argv[2] : NULL;

    FILE *f = fopen(seq_path, "rb");
    if (!f) { perror("fopen"); return 1; }

    /* Initialise both libraries. */
    pelorus_pico_init();

    pc_ctx_t pc;
    pc_init(&pc, config_path);

    int n_imu = 0, n_cloud = 0, n_valid = 0, n_fail = 0;
    int any_fail = 0;

    /* ---- replay loop ---- */
    for (;;) {
        uint8_t tag;
        if (fread(&tag, 1, 1, f) != 1) break; /* EOF */

        if (tag == TAG_IMU) {
            double t;
            float  gx, gy, gz, ax, ay, az;
            if (fread(&t,  sizeof(double), 1, f) != 1) break;
            if (fread(&gx, sizeof(float),  1, f) != 1) break;
            if (fread(&gy, sizeof(float),  1, f) != 1) break;
            if (fread(&gz, sizeof(float),  1, f) != 1) break;
            if (fread(&ax, sizeof(float),  1, f) != 1) break;
            if (fread(&ay, sizeof(float),  1, f) != 1) break;
            if (fread(&az, sizeof(float),  1, f) != 1) break;

            pelorus_pico_add_imu(t, gx, gy, gz, ax, ay, az);
            pc_add_imu(&pc, t, gx, gy, gz, ax, ay, az);
            n_imu++;

        } else if (tag == TAG_CLOUD) {
            double  t_start, t_end;
            int32_t n_pts;
            if (fread(&t_start, sizeof(double),  1, f) != 1) break;
            if (fread(&t_end,   sizeof(double),  1, f) != 1) break;
            if (fread(&n_pts,   sizeof(int32_t), 1, f) != 1) break;

            pelorus_pico_point_t *pts =
                (pelorus_pico_point_t *)malloc((size_t)n_pts * sizeof(pelorus_pico_point_t));
            if (!pts) { fprintf(stderr, "malloc failed\n"); break; }

            int i;
            for (i = 0; i < n_pts; i++) {
                float v[5];
                if (fread(v, sizeof(float), 5, f) != 5) { n_pts = i; break; }
                pts[i].x         = v[0];
                pts[i].y         = v[1];
                pts[i].z         = v[2];
                pts[i].intensity = v[3];
                pts[i].timestamp = v[4];
            }

            /* Feed to pelorus_pico */
            pelorus_odometry_t po =
                pelorus_pico_add_cloud(pts, n_pts, t_start, t_end);

            /* Feed to pelorus_c */
            int got_pc = pc_add_cloud(&pc, pts, n_pts, t_start, t_end);

            free(pts);
            n_cloud++;

            /* Only compare when both libraries have produced a valid result. */
            if (!po.valid || !got_pc || !pc.valid) continue;
            n_valid++;

            double pe = pos_error(po.position, pc.position);
            double od = quat_dot(po.orientation, pc.orientation);
            double ve = vel_error(po.linear_velocity, pc.linear_velocity);

            int pass = (pe < POS_THRESH) && (od > ORI_THRESH) && (ve < VEL_THRESH);
            if (!pass) {
                n_fail++;
                any_fail = 1;
                fprintf(stderr,
                    "[FAIL] scan %d  pos_err=%.4f m  ori_dot=%.6f  vel_err=%.4f m/s\n"
                    "       pico  pos=(%.3f %.3f %.3f)  q=(%.4f %.4f %.4f %.4f)\n"
                    "       pc    pos=(%.3f %.3f %.3f)  q=(%.4f %.4f %.4f %.4f)\n",
                    n_cloud,
                    pe, od, ve,
                    po.position[0], po.position[1], po.position[2],
                    po.orientation[0], po.orientation[1], po.orientation[2], po.orientation[3],
                    pc.position[0], pc.position[1], pc.position[2],
                    pc.orientation[0], pc.orientation[1], pc.orientation[2], pc.orientation[3]);
            } else {
                printf("[pass] scan %3d  pos_err=%.4f m  ori_dot=%.6f  vel_err=%.4f m/s\n",
                       n_cloud, pe, od, ve);
            }

        } else {
            fprintf(stderr, "Unknown tag 0x%02x at scan %d — aborting\n", tag, n_cloud);
            break;
        }
    }

    fclose(f);
    pelorus_destroy(pc.handle);

    /* ---- summary ---- */
    printf("\n=== Equality Test Summary ===\n");
    printf("IMU records   : %d\n", n_imu);
    printf("Cloud records : %d\n", n_cloud);
    printf("Valid scans   : %d\n", n_valid);
    printf("Failures      : %d\n", n_fail);
    printf("Result        : %s\n", any_fail ? "FAIL" : "PASS");

    return any_fail ? 1 : 0;
}

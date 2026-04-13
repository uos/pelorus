/* pico_equality.c — feed identical data to pelorus_c and pelorus_pico,
 * compare odometry outputs.  Data is the exact same sequence as sync_blocking.c.
 *
 * Build via the examples Makefile (from pelorus_c/examples/):
 *   make pico_equality
 *   ./pico_equality
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "../pelorus.h"
#include "../../pelorus_pico/include/pelorus_pico.h"

/* ---- pass/fail thresholds ---- */
#define POS_THRESH  0.001    /* metres */
#define ORI_THRESH  0.999999 /* |dot(q1,q2)| */
#define VEL_THRESH  0.01     /* m/s */

/* ---- helpers ---- */
static void fill_cov(double *cov, int n, double diag) {
    for (int i = 0; i < n*n; i++) cov[i] = 0.0;
    for (int i = 0; i < n;   i++) cov[i*n+i] = diag;
}

static double pos_err(const double a[3], const double b[3]) {
    double dx=a[0]-b[0], dy=a[1]-b[1], dz=a[2]-b[2];
    return sqrt(dx*dx + dy*dy + dz*dz);
}
static double quat_dot(const double a[4], const double b[4]) {
    return fabs(a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]);
}
static double vel_err(const double a[3], const double b[3]) {
    double dx=a[0]-b[0], dy=a[1]-b[1], dz=a[2]-b[2];
    return sqrt(dx*dx + dy*dy + dz*dz);
}

#define NUM_POINTS 64

/* Build a single IMU message and feed it to both libraries.
 * i is the global IMU index (0-based). */
static void feed_imu(PelorusHandle *pc, int i) {
    const int32_t base_sec = 1000;

    double t_abs = base_sec + i * 0.005;
    float gx, gy, gz, ax, ay, az;
    if (i < 200) {
        gx = 0.004f + 0.001f * (i%3 - 1);
        gy = -0.002f + 0.001f * ((i+1)%3 - 1);
        gz = 0.010f + 0.002f * ((i+2)%3 - 1);
        ax = 0.03f  + 0.01f * (i%5 - 2);
        ay = -0.015f + 0.01f * ((i+1)%5 - 2);
        az = 9.81f  + 0.02f + 0.01f * ((i+2)%5 - 2);
    } else {
        gx = 0.050f + 0.003f * (i%3 - 1);
        gy = -0.020f + 0.003f * ((i+1)%3 - 1);
        gz = 0.070f + 0.004f * ((i+2)%3 - 1);
        ax = 0.50f + 0.02f * (i%5 - 2);
        ay = -0.25f + 0.02f * ((i+1)%5 - 2);
        az = 9.81f + 0.35f + 0.02f * ((i+2)%5 - 2);
    }

    /* pelorus_pico */
    pelorus_pico_add_imu(t_abs, gx, gy, gz, ax, ay, az);

    /* pelorus_c */
    pelorus_imu_msg imu = {0};
    /* Compute (sec, nsec) without overflow: each step is 5ms */
    uint64_t t_ns_total = (uint64_t)base_sec * 1000000000ULL + (uint64_t)i * 5000000ULL;
    imu.timestamp_sec     = (int32_t)(t_ns_total / 1000000000ULL);
    imu.timestamp_nanosec = (uint32_t)(t_ns_total % 1000000000ULL);
    imu.orientation[0]    = 1.0;
    imu.angular_velocity[0]    = gx;
    imu.angular_velocity[1]    = gy;
    imu.angular_velocity[2]    = gz;
    imu.linear_acceleration[0] = ax;
    imu.linear_acceleration[1] = ay;
    imu.linear_acceleration[2] = az;
    fill_cov(imu.orientation_covariance,        3, 0.01);
    fill_cov(imu.angular_velocity_covariance,   3, 0.001);
    fill_cov(imu.linear_acceleration_covariance,3, 0.01);
    pelorus_add_imu(pc, &imu);
}

int main(int argc, char **argv)
{
    printf("pelorus_pico equality test\n");
    printf("==========================\n\n");

    /* ---- setup ---- */
    const char *config_path = (argc >= 2) ? argv[1] : "./pico-equality.rl";
    PelorusHandle *pc = pelorus_create(config_path);
    if (!pc) { fprintf(stderr, "pelorus_create() failed\n"); return 1; }
    pelorus_pico_init();

    const int32_t base_sec = 1000;

    /* ---- Phase 1: 200 IMU for initialisation (i=0..199) ---- */
    printf("Sending 200 IMU messages for initialisation...\n");
    for (int i = 0; i < 200; i++) feed_imu(pc, i);
    printf("  done.\n\n");

    /* ---- Build scan 1 data (same geometry as sync_blocking) ---- */
    /* pelorus_c: xyzt (4 floats, t in nanoseconds relative to scan start) */
    float scan1_xyzt[NUM_POINTS * 4];
    /* pico: xyzit (timestamp in seconds relative to cloud_start_time) */
    pelorus_pico_point_t scan1_pico[NUM_POINTS];
    for (int i = 0; i < NUM_POINTS; i++) {
        float angle = (float)i / NUM_POINTS * 6.28318f;
        float range = 5.0f + 0.5f * (i % 8);
        float x = range * cosf(angle);
        float y = range * sinf(angle);
        float z = 0.2f * sinf(angle * 3.0f);
        float t_rel_ns  = (float)i / NUM_POINTS * 100000000.0f; /* 0..100 ms in ns */
        float t_rel_sec = (float)i / NUM_POINTS * 0.1f;         /* 0..0.1 s */

        scan1_xyzt[i*4+0] = x;
        scan1_xyzt[i*4+1] = y;
        scan1_xyzt[i*4+2] = z;
        scan1_xyzt[i*4+3] = t_rel_ns;

        scan1_pico[i].x         = x;
        scan1_pico[i].y         = y;
        scan1_pico[i].z         = z;
        scan1_pico[i].intensity = 0.0f;
        scan1_pico[i].timestamp = t_rel_sec;
    }

    /* ---- Add scan 1 to pelorus_c (same call as sync_blocking) ---- */
    uint64_t scan1_id = pelorus_add_points_xyzt_f32(
        pc, base_sec, 600000000, "lidar", scan1_xyzt, NUM_POINTS);
    if (scan1_id == 0)
        fprintf(stderr, "Warning: pelorus_c scan1 rejected\n");

    /* ---- Add scan 1 to pico (bootstraps map; returns valid=0) ---- */
    pelorus_odometry_t pp1 = pelorus_pico_add_cloud(
        scan1_pico, NUM_POINTS, 1000.6, 1000.7);
    printf("[scan 1] pico valid=%d (expected 0 — bootstraps map)\n\n", pp1.valid);

    /* ---- Phase 2: 100 more IMU (i=200..299) ---- */
    printf("Sending 100 more IMU messages...\n");
    for (int i = 200; i < 300; i++) feed_imu(pc, i);
    printf("  done.\n\n");

    /* ---- Build scan 2 data (same geometry as sync_blocking) ---- */
    float scan2_xyzit[NUM_POINTS * 5];
    pelorus_pico_point_t scan2_pico[NUM_POINTS];
    for (int i = 0; i < NUM_POINTS; i++) {
        float angle = (float)i / NUM_POINTS * 6.28318f + 0.22f;
        float range = 5.0f + 0.5f * (i % 8) + 0.20f;
        float x2 = range * cosf(angle) + 0.60f;
        float y2 = range * sinf(angle) - 0.30f;
        float z2 = 0.2f * sinf(angle * 3.0f) + 0.08f;
        float t_rel_ns  = (float)i / NUM_POINTS * 100000000.0f;
        float t_rel_sec = (float)i / NUM_POINTS * 0.1f;

        scan2_xyzit[i*5+0] = x2;
        scan2_xyzit[i*5+1] = y2;
        scan2_xyzit[i*5+2] = z2;
        scan2_xyzit[i*5+3] = 0.0f;        /* intensity */
        scan2_xyzit[i*5+4] = t_rel_ns;

        scan2_pico[i].x         = x2;
        scan2_pico[i].y         = y2;
        scan2_pico[i].z         = z2;
        scan2_pico[i].intensity = 0.0f;
        scan2_pico[i].timestamp = t_rel_sec;
    }

    /* ---- Add scan 2 to pelorus_c (same call as sync_blocking) ---- */
    uint64_t scan2_id = pelorus_add_points_xyzit_f32(
        pc, base_sec + 1, 200000000, NULL, scan2_xyzit, NUM_POINTS);
    if (scan2_id == 0)
        fprintf(stderr, "Warning: pelorus_c scan2 rejected\n");

    /* ---- Add scan 2 to pico (does registration; should return valid=1) ---- */
    pelorus_odometry_t pp2 = pelorus_pico_add_cloud(
        scan2_pico, NUM_POINTS, 1001.2, 1001.3);

    /* ---- Poll pelorus_c for scan2 result (up to 5 s) ---- */
    printf("Waiting for pelorus_c scan2 result (up to 5s, scan2_id=%llu)...\n",
           (unsigned long long)scan2_id);
    pelorus_odometry pc_odom = {0};
    int got_pc = 0;
    if (scan2_id > 0) {
        for (int try = 0; try < 1000 && !got_pc; try++) {
            pelorus_odometry tmp = {0};
            int32_t r = pelorus_try_recv(pc, &tmp);
            if (r == PELORUS_STATUS_OK) {
                printf("  got scan_id=%llu pos=(%.6f %.6f %.6f)\n",
                       (unsigned long long)tmp.scan_id,
                       tmp.position[0], tmp.position[1], tmp.position[2]);
                if (tmp.scan_id == scan2_id) {
                    pc_odom = tmp;
                    got_pc = 1;
                } else {
                    pelorus_free_odometry(&tmp);
                }
            } else if (r == PELORUS_STATUS_NO_DATA) {
                usleep(5000); /* 5 ms — queue empty, wait */
            } else {
                fprintf(stderr, "pelorus_try_recv error %d\n", r);
                break;
            }
        }
    }

    /* ---- Report raw outputs ---- */
    printf("\npico  valid=%d  pos=(%.6f %.6f %.6f)  q=(%.6f %.6f %.6f %.6f)\n",
           pp2.valid,
           pp2.position[0], pp2.position[1], pp2.position[2],
           pp2.orientation[0], pp2.orientation[1],
           pp2.orientation[2], pp2.orientation[3]);

    if (got_pc) {
        printf("pc    valid=1  pos=(%.6f %.6f %.6f)  q=(%.6f %.6f %.6f %.6f)\n",
               pc_odom.position[0], pc_odom.position[1], pc_odom.position[2],
               pc_odom.orientation[0], pc_odom.orientation[1],
               pc_odom.orientation[2], pc_odom.orientation[3]);
    } else {
        printf("pc    TIMEOUT — no result for scan2_id=%llu\n",
               (unsigned long long)scan2_id);
    }

    /* ---- Compare ---- */
    int result = 0; /* 0=pass */
    if (!pp2.valid) {
        printf("\nRESULT: FAIL — pico did not return valid odometry for scan 2\n");
        result = 1;
    } else if (!got_pc) {
        printf("\nRESULT: FAIL — pelorus_c timed out\n");
        result = 1;
    } else {
        double pe = pos_err(pp2.position, pc_odom.position);
        double od = quat_dot(pp2.orientation, pc_odom.orientation);
        double ve = vel_err(pp2.linear_velocity, pc_odom.linear_velocity);

        printf("\npos_err=%.6f m  ori_dot=%.9f  vel_err=%.6f m/s\n", pe, od, ve);
        printf("pico vel=(%.3f %.3f %.3f)  pc vel=(%.3f %.3f %.3f)\n",
               pp2.linear_velocity[0], pp2.linear_velocity[1], pp2.linear_velocity[2],
               pc_odom.linear_velocity[0], pc_odom.linear_velocity[1], pc_odom.linear_velocity[2]);

        int pass = (pe < POS_THRESH) && (od > ORI_THRESH) && (ve < VEL_THRESH);
        printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
        if (!pass) {
            if (pe >= POS_THRESH) printf("  position error %.4f >= %.4f m\n", pe, POS_THRESH);
            if (od <= ORI_THRESH) printf("  orientation dot %.6f <= %.6f\n", od, ORI_THRESH);
            if (ve >= VEL_THRESH) printf("  velocity error %.4f >= %.4f m/s\n", ve, VEL_THRESH);
        }
        result = pass ? 0 : 1;
    }

    if (got_pc) pelorus_free_odometry(&pc_odom);
    pelorus_destroy(pc);
    return result;
}

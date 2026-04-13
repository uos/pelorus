/* gen_sequence.c — generate a synthetic IMU + point cloud sequence
 *
 * Simulates a robot moving in a straight line (constant velocity on X axis)
 * with a flat ground plane visible to the LiDAR.
 *
 * Build:
 *   gcc -O2 -std=c99 pelorus_pico/tests/gen_sequence.c -lm -o gen_sequence
 *   ./gen_sequence > sequence.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define TAG_IMU   ((uint8_t)1)
#define TAG_CLOUD ((uint8_t)2)

/* Motion parameters */
#define VX          0.5f     /* m/s forward velocity */
#define GRAVITY     9.81f    /* m/s² */
#define IMU_RATE    200.0    /* Hz */
#define CLOUD_RATE  10.0     /* Hz */
#define N_SCANS     30       /* total clouds */

/* Point cloud: a flat ground plane z=-1.5m + some walls */
#define GRID_DIM    20       /* points per side on ground plane */
#define N_GROUND    (GRID_DIM*GRID_DIM)
#define N_WALL      (GRID_DIM*GRID_DIM)
#define N_PTS       (N_GROUND + N_WALL)

static void write_u8(FILE *f, uint8_t v)  { fwrite(&v, 1, 1, f); }
static void write_f64(FILE *f, double v)  { fwrite(&v, sizeof(double), 1, f); }
static void write_f32(FILE *f, float  v)  { fwrite(&v, sizeof(float),  1, f); }
static void write_i32(FILE *f, int32_t v) { fwrite(&v, sizeof(int32_t), 1, f); }

int main(void)
{
    FILE *f = stdout;
    /* write to stdout as binary */
#ifdef _WIN32
    _setmode(_fileno(stdout), 0x8000); /* O_BINARY */
#endif

    double t = 100.0; /* absolute start time (seconds) */
    double scan_dt = 1.0 / CLOUD_RATE;
    double imu_dt  = 1.0 / IMU_RATE;
    int    imu_per_scan = (int)(scan_dt / imu_dt);

    int scan;
    for (scan = 0; scan < N_SCANS; scan++) {
        double t_scan_start = t;
        double t_scan_end   = t + scan_dt;

        /* ---- IMU records for this interval ---- */
        int i;
        for (i = 0; i < imu_per_scan; i++) {
            double ti = t + i * imu_dt;
            write_u8(f, TAG_IMU);
            write_f64(f, ti);
            /* gyro: all zero (no rotation) */
            write_f32(f, 0.0f); write_f32(f, 0.0f); write_f32(f, 0.0f);
            /* accel: gravity pointing down (-Z body = +Z world for upright IMU) */
            write_f32(f, 0.0f); write_f32(f, 0.0f); write_f32(f, GRAVITY);
        }

        /* ---- CLOUD record ---- */
        /* Robot position at scan start: x = VX * (t_scan_start - 100.0) */
        float robot_x = VX * (float)(t_scan_start - 100.0);

        write_u8(f, TAG_CLOUD);
        write_f64(f, t_scan_start);
        write_f64(f, t_scan_end);
        write_i32(f, (int32_t)N_PTS);

        /* Ground plane at z = -1.5 m (relative to robot) */
        int gx, gy;
        int pt_idx = 0;
        for (gx = 0; gx < GRID_DIM; gx++) {
            for (gy = 0; gy < GRID_DIM; gy++) {
                float px = -5.0f + (float)gx * 0.5f;
                float py = -5.0f + (float)gy * 0.5f;
                float pz = -1.5f;
                /* timestamp spread across scan */
                float ts_rel = (float)pt_idx / (float)N_PTS * (float)scan_dt;
                write_f32(f, px);
                write_f32(f, py);
                write_f32(f, pz);
                write_f32(f, 0.5f);  /* intensity */
                write_f32(f, ts_rel);
                pt_idx++;
            }
        }

        /* Front wall at x = 8m (relative to robot) */
        for (gx = 0; gx < GRID_DIM; gx++) {
            for (gy = 0; gy < GRID_DIM; gy++) {
                float px = 8.0f;
                float py = -2.5f + (float)gx * 0.25f;
                float pz = -1.5f + (float)gy * 0.25f;
                float ts_rel = (float)pt_idx / (float)N_PTS * (float)scan_dt;
                write_f32(f, px);
                write_f32(f, py);
                write_f32(f, pz);
                write_f32(f, 0.8f);
                write_f32(f, ts_rel);
                pt_idx++;
            }
        }

        t = t_scan_end;
    }

    fflush(f);
    return 0;
}

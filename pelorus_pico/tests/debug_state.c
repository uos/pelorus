/* debug_state.c — minimal standalone test that prints EKF state */
#include <stdio.h>
#include <math.h>
#include "../../pelorus_pico/include/pelorus_pico.h"

int main(void) {
    pelorus_pico_init();

    /* Feed 100 init IMU (same as equality test) */
    for (int i = 0; i < 100; i++) {
        double t = 1000.0 + i * 5e-3;
        pelorus_pico_add_imu(t,
            0.001f*(i%3-1), 0.001f*((i+1)%3-1), 0.002f*((i+2)%3-1),
            0.01f*(i%5-2),  0.01f*((i+1)%5-2),  9.81f+0.01f*((i+2)%5-2));
    }
    printf("After 100 init IMU: initialized\n");

    /* Feed 20 more IMU */
    for (int i = 100; i < 220; i++) {
        double t = 1000.0 + i * 5e-3;
        pelorus_odometry_t o = pelorus_pico_add_imu(t,
            0.001f*(i%3-1), 0.001f*((i+1)%3-1), 0.002f*((i+2)%3-1),
            0.01f*(i%5-2),  0.01f*((i+1)%5-2),  9.81f+0.01f*((i+2)%5-2));
        if (i == 219) {
            printf("After IMU i=219 (t=%.3f): pos=(%.4f %.4f %.4f) vel=(%.4f %.4f %.4f)\n",
                t, o.position[0], o.position[1], o.position[2],
                o.linear_velocity[0], o.linear_velocity[1], o.linear_velocity[2]);
            printf("  orient=(%.4f %.4f %.4f %.4f) valid=%d\n",
                o.orientation[0], o.orientation[1], o.orientation[2], o.orientation[3], o.valid);
        }
    }

    /* Feed first cloud */
#define SCAN_GRID 15
#define SCAN_N_PTS (2*SCAN_GRID*SCAN_GRID)
    float scan_data[SCAN_N_PTS * 5];
    int idx = 0;
    for (int gx = 0; gx < SCAN_GRID; gx++)
        for (int gy = 0; gy < SCAN_GRID; gy++) {
            scan_data[idx*5+0] = -3.5f + gx*0.5f;
            scan_data[idx*5+1] = -3.5f + gy*0.5f;
            scan_data[idx*5+2] = -1.5f;
            scan_data[idx*5+3] = 0.6f;
            scan_data[idx*5+4] = (float)idx/SCAN_N_PTS * 0.1f * 1e9f;
            idx++;
        }
    for (int gx = 0; gx < SCAN_GRID; gx++)
        for (int gy = 0; gy < SCAN_GRID; gy++) {
            scan_data[idx*5+0] = 8.0f;
            scan_data[idx*5+1] = -3.5f + gx*0.35f;
            scan_data[idx*5+2] = -1.5f + gy*0.35f;
            scan_data[idx*5+3] = 0.8f;
            scan_data[idx*5+4] = (float)idx/SCAN_N_PTS * 0.1f * 1e9f;
            idx++;
        }

    double t_start = 1000.0 + 220*5e-3; /* 1001.1 */
    double t_end   = t_start + 0.1;

    printf("\nBefore cloud 1: t_start=%.3f t_end=%.3f\n", t_start, t_end);
    pelorus_odometry_t o1 = pelorus_pico_add_cloud(
        (const pelorus_pico_point_t *)scan_data, SCAN_N_PTS, t_start, t_end);
    printf("After cloud 1: pos=(%.4f %.4f %.4f) vel=(%.4f %.4f %.4f) valid=%d\n",
        o1.position[0], o1.position[1], o1.position[2],
        o1.linear_velocity[0], o1.linear_velocity[1], o1.linear_velocity[2], o1.valid);
    printf("  orient=(%.4f %.4f %.4f %.4f)\n",
        o1.orientation[0], o1.orientation[1], o1.orientation[2], o1.orientation[3]);

    /* 20 more IMU */
    for (int i = 220; i < 240; i++) {
        double t = 1000.0 + i * 5e-3;
        pelorus_pico_add_imu(t,
            0.001f*(i%3-1), 0.001f*((i+1)%3-1), 0.002f*((i+2)%3-1),
            0.01f*(i%5-2),  0.01f*((i+1)%5-2),  9.81f+0.01f*((i+2)%5-2));
    }

    double t2_start = 1000.0 + 260*5e-3;
    double t2_end   = t2_start + 0.1;
    printf("\nBefore cloud 2: t_start=%.3f t_end=%.3f\n", t2_start, t2_end);
    pelorus_odometry_t o2 = pelorus_pico_add_cloud(
        (const pelorus_pico_point_t *)scan_data, SCAN_N_PTS, t2_start, t2_end);
    printf("After cloud 2: pos=(%.4f %.4f %.4f) vel=(%.4f %.4f %.4f) valid=%d\n",
        o2.position[0], o2.position[1], o2.position[2],
        o2.linear_velocity[0], o2.linear_velocity[1], o2.linear_velocity[2], o2.valid);

    return 0;
}

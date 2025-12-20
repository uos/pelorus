#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "../pelorus.h"

static void fill_identity_cov(double *cov, double diag) {
    for (size_t i = 0; i < 9; i++) {
        cov[i] = (i % 4 == 0) ? diag : 0.0;
    }
}

int main(void) {
    printf("Version: %s\n\n", pelorus_version());

    // Create Pelorus handle with default configuration.
    PelorusHandle *pelorus = pelorus_create(NULL);
    if (!pelorus) {
        fprintf(stderr, "Failed to create Pelorus handle\n");
        return 1;
    }
    printf("Created Pelorus handle with default config\n\n");

    // The library handles threading internally; this example stays single-threaded.
    // We just feed IMU + point clouds in sequence and then call `pelorus_recv()`.

    const int32_t base_sec = 1000;

    // Send IMU messages for initialization (default config requires ~500ms).
    printf("Sending 200 IMU messages for initialization...\n");
    for (int i = 0; i < 200; i++) {
        pelorus_imu_msg imu = {0};
        imu.timestamp_sec = base_sec;
        imu.timestamp_nanosec = i * 5000000;  // 5ms intervals
        imu.frame_id = "imu";

        imu.orientation[0] = 1.0;
        imu.angular_velocity[0] = 0.001 * (i % 3 - 1);
        imu.angular_velocity[1] = 0.001 * ((i + 1) % 3 - 1);
        imu.angular_velocity[2] = 0.002 * ((i + 2) % 3 - 1);
        imu.linear_acceleration[0] = 0.01 * (i % 5 - 2);
        imu.linear_acceleration[1] = 0.01 * ((i + 1) % 5 - 2);
        imu.linear_acceleration[2] = 9.81 + 0.01 * ((i + 2) % 5 - 2);

        fill_identity_cov(imu.orientation_covariance, 0.01);
        fill_identity_cov(imu.angular_velocity_covariance, 0.001);
        fill_identity_cov(imu.linear_acceleration_covariance, 0.01);

        pelorus_add_imu(pelorus, &imu);
    }

    #define NUM_POINTS 64

    // Scan 1
    float scan1_data[NUM_POINTS * 4];
    for (int i = 0; i < NUM_POINTS; i++) {
        float angle = (float)i / NUM_POINTS * 6.28318f;
        float range = 5.0f + 0.5f * (i % 8);

        scan1_data[i * 4 + 0] = range * cosf(angle);
        scan1_data[i * 4 + 1] = range * sinf(angle);
        scan1_data[i * 4 + 2] = 0.2f * sinf(angle * 3);
        scan1_data[i * 4 + 3] = (float)i / NUM_POINTS * 100000000.0f;
    }
    pelorus_add_points_xyzt_f32(
        pelorus,
        base_sec,
        600000000,  // 600ms - after 500ms initialization
        "lidar",
        scan1_data,
        NUM_POINTS);

    // Some more IMU
    for (int i = 200; i < 300; i++) {
        pelorus_imu_msg imu = {0};
        imu.timestamp_sec = base_sec;
        imu.timestamp_nanosec = i * 5000000;
        imu.frame_id = "imu";
        imu.orientation[0] = 1.0;
        imu.angular_velocity[0] = 0.001 * (i % 3 - 1);
        imu.angular_velocity[1] = 0.001 * ((i + 1) % 3 - 1);
        imu.angular_velocity[2] = 0.002 * ((i + 2) % 3 - 1);
        imu.linear_acceleration[0] = 0.01 * (i % 5 - 2);
        imu.linear_acceleration[1] = 0.01 * ((i + 1) % 5 - 2);
        imu.linear_acceleration[2] = 9.81 + 0.01 * ((i + 2) % 5 - 2);
        fill_identity_cov(imu.orientation_covariance, 0.01);
        fill_identity_cov(imu.angular_velocity_covariance, 0.001);
        fill_identity_cov(imu.linear_acceleration_covariance, 0.01);
        pelorus_add_imu(pelorus, &imu);
    }

    // Scan 2 (slightly moved)
    float scan2_data[NUM_POINTS * 4];
    for (int i = 0; i < NUM_POINTS; i++) {
        float angle = (float)i / NUM_POINTS * 6.28318f + 0.05f;
        float range = 5.0f + 0.5f * (i % 8) + 0.05f;

        scan2_data[i * 4 + 0] = range * cosf(angle);
        scan2_data[i * 4 + 1] = range * sinf(angle);
        scan2_data[i * 4 + 2] = 0.2f * sinf(angle * 3);
        scan2_data[i * 4 + 3] = (float)i / NUM_POINTS * 100000000.0f;
    }
    pelorus_add_points_xyzt_f32(
        pelorus,
        base_sec + 1,
        200000000,
        "lidar",
        scan2_data,
        NUM_POINTS);

    // Now block waiting for an odometry result.
    printf("Waiting for odometry (blocking)...\n\n");
    
    pelorus_odometry odom = {0};
    const int recv_status = pelorus_recv(pelorus, &odom);
    if (recv_status == 1) {
        printf("  Position: [%.3f, %.3f, %.3f]\n",
            odom.position[0], odom.position[1], odom.position[2]);
        printf("  Orientation: [%.3f, %.3f, %.3f, %.3f]\n",
            odom.orientation[0], odom.orientation[1],
            odom.orientation[2], odom.orientation[3]);
        printf("  Linear velocity: [%.3f, %.3f, %.3f]\n",
            odom.linear_velocity[0], odom.linear_velocity[1], odom.linear_velocity[2]);
        printf("  Angular velocity: [%.3f, %.3f, %.3f]\n",
            odom.angular_velocity[0], odom.angular_velocity[1], odom.angular_velocity[2]);
        printf("  Point cloud: %zu points\n", odom.num_points);

        for (size_t j = 0; j < odom.num_points && j < 3; j++) {
        const pelorus_point_xyzi *p = &odom.points[j];
            printf("    Point %zu: [%.2f, %.2f, %.2f] I=%.1f\n",
                j, p->x, p->y, p->z, p->intensity);
        }

        pelorus_free_odometry(&odom);
    } else {
        fprintf(stderr, "Error receiving odometry (status: %d)\n", recv_status);
    }

    // Clean up
    pelorus_destroy(pelorus);
    printf("Destroyed Pelorus handle\n");
    
    return 0;
}

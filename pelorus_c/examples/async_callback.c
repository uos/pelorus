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

// User context to track received odometry count
typedef struct {
    int count;
} UserContext;

// Callback function that gets invoked when odometry is available
void odometry_callback(const pelorus_odometry *odom, void *user_data) {
    UserContext *ctx = (UserContext *)user_data;
    ctx->count++;
    
    printf("Callback invoked! Odometry #%d received\n", ctx->count);
    printf("  Timestamp: %d.%09u\n", odom->timestamp_sec, odom->timestamp_nanosec);
    printf("  Position: [%.3f, %.3f, %.3f]\n", 
           odom->position[0], odom->position[1], odom->position[2]);
    printf("  Linear velocity: [%.3f, %.3f, %.3f]\n", 
           odom->linear_velocity[0], odom->linear_velocity[1], odom->linear_velocity[2]);
    printf("  Point cloud: %zu points\n", odom->num_points);
    
    // Print first few points
    for (size_t i = 0; i < odom->num_points && i < 3; i++) {
        const pelorus_point_xyzi *p = &odom->points[i];
        printf("    Point %zu: [%.2f, %.2f, %.2f] I=%.1f\n", 
               i, p->x, p->y, p->z, p->intensity);
    }
    printf("\n");
}

int main(void) {
    printf("Version: %s\n\n", pelorus_version());

    // Create Pelorus handle with default configuration.
    PelorusHandle *pelorus = pelorus_create(NULL);
    if (!pelorus) {
        fprintf(stderr, "Failed to create Pelorus handle\n");
        return 1;
    }
    printf("Created Pelorus handle with default config\n");

    // Set up user context
    UserContext ctx = {.count = 0};

    // Register callback for asynchronous odometry reception
    if (pelorus_set_callback(pelorus, odometry_callback, &ctx) != 0) {
        fprintf(stderr, "Failed to set callback\n");
        pelorus_destroy(pelorus);
        return 1;
    }
    printf("Registered callback for async odometry reception\n\n");

    // The library handles threading internally; this example stays single-threaded.
    // We just feed IMU + point clouds in sequence and then wait for callbacks.

    // Base timestamp for simulation
    const int32_t base_sec = 1000;
    
    // Send many IMU messages (needed for initialization - 500ms default)
    printf("Sending 200 IMU messages for initialization...\n");
    for (int i = 0; i < 200; i++) {
        pelorus_imu_msg imu = {0};
        // IMU at ~200Hz (5ms intervals)
        imu.timestamp_sec = base_sec;
        imu.timestamp_nanosec = i * 5000000;  // 5ms intervals
        imu.frame_id = "imu";
        
        // Stationary orientation (identity quaternion)
        imu.orientation[0] = 1.0;  // w
        imu.orientation[1] = 0.0;  // x
        imu.orientation[2] = 0.0;  // y
        imu.orientation[3] = 0.0;  // z
        
        // Small angular velocity (nearly stationary)
        imu.angular_velocity[0] = 0.001 * (i % 3 - 1);
        imu.angular_velocity[1] = 0.001 * ((i + 1) % 3 - 1);
        imu.angular_velocity[2] = 0.002 * ((i + 2) % 3 - 1);
        
        // Gravity in z direction
        imu.linear_acceleration[0] = 0.01 * (i % 5 - 2);
        imu.linear_acceleration[1] = 0.01 * ((i + 1) % 5 - 2);
        imu.linear_acceleration[2] = 9.81 + 0.01 * ((i + 2) % 5 - 2);
        
        fill_identity_cov(imu.orientation_covariance, 0.01);
        fill_identity_cov(imu.angular_velocity_covariance, 0.001);
        fill_identity_cov(imu.linear_acceleration_covariance, 0.01);

        pelorus_add_imu(pelorus, &imu);
    }
    printf("Added 200 IMU samples\n");

    // Raw point helper API (mirrors Python numpy path):
    // - packed float array Nx4: [x,y,z,t]
    // - scan header timestamp is absolute
    // - per-point `t` is nanoseconds offset relative to the header timestamp (cast to uint32)

    // Create first scan with multiple points distributed in space.
    // Each point has per-point timestamp relative to scan start.
    #define NUM_POINTS_SCAN 64
    float scan1_data[NUM_POINTS_SCAN * 4];
    
    printf("Generating first scan with %d points...\n", NUM_POINTS_SCAN);
    for (int i = 0; i < NUM_POINTS_SCAN; i++) {
        float angle = (float)i / NUM_POINTS_SCAN * 6.28318f;  // Full circle
        float range = 5.0f + 0.5f * (i % 8);  // Range 5-9m

        scan1_data[i * 4 + 0] = range * cosf(angle);  // x
        scan1_data[i * 4 + 1] = range * sinf(angle);  // y
        scan1_data[i * 4 + 2] = 0.2f * sinf(angle * 3);  // z
        scan1_data[i * 4 + 3] = (float)i / NUM_POINTS_SCAN * 100000000.0f;  // t (ns offset)
    }

    if (pelorus_add_points_xyzt_f32(
            pelorus,
            base_sec,
            600000000,  // 600ms - after 500ms initialization
            "lidar",
            scan1_data,
            NUM_POINTS_SCAN) != 0) {
        fprintf(stderr, "Failed to add first point cloud\n");
    } else {
        printf("Added first scan with %d points\n", NUM_POINTS_SCAN);
    }

    // Send more IMU messages between scans (need to cover up to 1.2+ seconds)
    printf("Sending 100 more IMU messages...\n");
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
    printf("Added 100 more IMU samples\n");

    // Create second scan (slightly different to enable registration)
    float scan2_data[NUM_POINTS_SCAN * 4];
    
    printf("Generating second scan with %d points...\n", NUM_POINTS_SCAN);
    for (int i = 0; i < NUM_POINTS_SCAN; i++) {
        float angle = (float)i / NUM_POINTS_SCAN * 6.28318f + 0.1f;  // Slight rotation
        float range = 5.0f + 0.5f * (i % 8) + 0.05f;  // Slight movement

        scan2_data[i * 4 + 0] = range * cosf(angle);
        scan2_data[i * 4 + 1] = range * sinf(angle);
        scan2_data[i * 4 + 2] = 0.2f * sinf(angle * 3);
        scan2_data[i * 4 + 3] = (float)i / NUM_POINTS_SCAN * 100000000.0f;
    }

    if (pelorus_add_points_xyzt_f32(
            pelorus,
            base_sec + 1,  // 1.2 seconds from start
            200000000,
            "lidar",
            scan2_data,
            NUM_POINTS_SCAN) != 0) {
        fprintf(stderr, "Failed to add second point cloud\n");
    } else {
        printf("Added second scan with %d points\n\n", NUM_POINTS_SCAN);
    }

    // Wait for callbacks to be invoked (processing happens asynchronously)
    printf("Waiting for odometry callback...\n");
    for (int i = 0; i < 2000 && ctx.count == 0; i++) {
        usleep(1 * 1000);
    }

    // Disable callback before destroying
    pelorus_set_callback(pelorus, NULL, NULL);

    // Clean up
    pelorus_destroy(pelorus);
    printf("\nDestroyed Pelorus handle\n");
    printf("Total odometry results received: %d\n", ctx.count);
   
    
    return 0;
}

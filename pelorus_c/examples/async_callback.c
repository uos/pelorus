#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
#include "../pelorus.h"

static void fill_identity_cov(double *cov, double diag)
{
    for (size_t i = 0; i < 9; i++)
    {
        cov[i] = (i % 4 == 0) ? diag : 0.0;
    }
}

// User context to track received odometry count
typedef struct
{
    int count;
    uint64_t scan2_id; // 0 until set by the main thread
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} UserContext;

// Callback function that gets invoked when odometry is available
void odometry_callback(const pelorus_odometry *odom, void *user_data)
{
    UserContext *ctx = (UserContext *)user_data;

    // If scan2_id is set, only react to odometry matching that scan id.
    if (ctx->scan2_id != 0)
    {
        if (odom->scan_id != ctx->scan2_id)
        {
            return; // ignore unrelated odometry
        }
    }

    // We will update count and notify the main thread.
    pthread_mutex_lock(&ctx->mutex);
    ctx->count++;
    pthread_cond_signal(&ctx->cond);
    pthread_mutex_unlock(&ctx->mutex);

    printf("Callback invoked! Odometry #%d received (scan_id=%llu)\n", ctx->count, (unsigned long long)odom->scan_id);
    printf("  Timestamp: %d.%09u\n", odom->timestamp_sec, odom->timestamp_nanosec);
    printf("  Position: [%.6f, %.6f, %.6f]\n",
           odom->position[0], odom->position[1], odom->position[2]);
    printf("  Linear velocity: [%.6f, %.6f, %.6f]\n",
           odom->linear_velocity[0], odom->linear_velocity[1], odom->linear_velocity[2]);
    printf("  Point cloud: %zu points\n", odom->num_points);

    // Print points. When PELORUS_DUMP_ALL_POINTS=1, dump full cloud (for regression tests)
    if (getenv("PELORUS_DUMP_ALL_POINTS") != NULL)
    {
        for (size_t i = 0; i < odom->num_points; i++)
        {
            const pelorus_point_xyzi *p = &odom->points[i];
            printf("    Point %zu: [%.6g, %.6g, %.6g] I=%.6g\n",
                   i, p->x, p->y, p->z, p->intensity);
        }
    }
    else
    {
        for (size_t i = 0; i < odom->num_points && i < 3; i++)
        {
            const pelorus_point_xyzi *p = &odom->points[i];
            printf("    Point %zu: [%.2f, %.2f, %.2f] I=%.1f\n",
                   i, p->x, p->y, p->z, p->intensity);
        }
    }
    printf("\n");
}

int main(void)
{
    printf("Version: %s\n\n", pelorus_version());

    // Use the comparison config so binding/backend comparisons all bypass the
    // optional imu-fusion preprocessing in the same way.
    PelorusHandle *pelorus = pelorus_create("./pico-equality.rl");
    if (!pelorus)
    {
        fprintf(stderr, "Failed to create Pelorus handle\n");
        return 1;
    }
    printf("Created Pelorus handle with comparison config\n");

    // Set up user context
    UserContext ctx = {.count = 0};

    // Initialize synchronization primitives
    pthread_mutex_init(&ctx.mutex, NULL);
    pthread_cond_init(&ctx.cond, NULL);

    // Register callback for asynchronous odometry reception
    if (pelorus_set_callback(pelorus, odometry_callback, &ctx) != 0)
    {
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
    for (int i = 0; i < 200; i++)
    {
        pelorus_imu_msg imu = {0};
        // IMU at ~200Hz (5ms intervals)
        imu.timestamp_sec = base_sec;
        imu.timestamp_nanosec = i * 5000000; // 5ms intervals
        imu.frame_id = NULL;

        // Stationary orientation (identity quaternion)
        imu.orientation[0] = 1.0; // w
        imu.orientation[1] = 0.0; // x
        imu.orientation[2] = 0.0; // y
        imu.orientation[3] = 0.0; // z

        // Small angular velocity (nearly stationary)
        imu.angular_velocity[0] = 0.004 + 0.001 * (i % 3 - 1);
        imu.angular_velocity[1] = -0.002 + 0.001 * ((i + 1) % 3 - 1);
        imu.angular_velocity[2] = 0.010 + 0.002 * ((i + 2) % 3 - 1);

        // Gravity in z direction
        imu.linear_acceleration[0] = 0.03 + 0.01 * (i % 5 - 2);
        imu.linear_acceleration[1] = -0.015 + 0.01 * ((i + 1) % 5 - 2);
        imu.linear_acceleration[2] = 9.81 + 0.02 + 0.01 * ((i + 2) % 5 - 2);

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
#define NUM_POINTS_SCAN 512
    float scan1_data[NUM_POINTS_SCAN * 4];

    printf("Generating first scan with %d points...\n", NUM_POINTS_SCAN);
    for (int i = 0; i < NUM_POINTS_SCAN; i++)
    {
        float x, y, z;
        if (i < NUM_POINTS_SCAN / 2)
        {
            int u = i % 16;
            int v = i / 16;
            x = 4.0f + 0.12f * u;
            y = -1.0f + 0.12f * v;
            z = 0.01f * ((u + v) % 2);
        }
        else
        {
            int j = i - NUM_POINTS_SCAN / 2;
            int u = j % 16;
            int v = j / 16;
            x = 6.0f + 0.005f * ((u + v) % 2);
            y = -1.0f + 0.12f * u;
            z = 0.10f + 0.12f * v;
        }

        scan1_data[i * 4 + 0] = x;
        scan1_data[i * 4 + 1] = y;
        scan1_data[i * 4 + 2] = z;
        scan1_data[i * 4 + 3] = (float)i / NUM_POINTS_SCAN * 100000000.0f; // t (ns offset)
    }

    uint64_t scan1_id = pelorus_add_points_xyzt_f32(
        pelorus,
        base_sec,
        600000000, // 600ms - after 500ms initialization
        NULL,
        scan1_data,
        NUM_POINTS_SCAN);
    if (scan1_id == 0)
    {
        fprintf(stderr, "Failed to add first point cloud\n");
    }
    else
    {
        printf("Added first scan id=%llu with %d points\n", (unsigned long long)scan1_id, NUM_POINTS_SCAN);
    }

    // Send more IMU messages between scans (need to cover up to 1.2+ seconds)
    printf("Sending 100 more IMU messages...\n");
    for (int i = 200; i < 300; i++)
    {
        pelorus_imu_msg imu = {0};
        imu.timestamp_sec = base_sec;
        imu.timestamp_nanosec = i * 5000000;
        imu.frame_id = NULL;

        imu.orientation[0] = 1.0;
        imu.angular_velocity[0] = 0.004 + 0.001 * (i % 3 - 1);
        imu.angular_velocity[1] = -0.002 + 0.001 * ((i + 1) % 3 - 1);
        imu.angular_velocity[2] = 0.010 + 0.002 * ((i + 2) % 3 - 1);
        imu.linear_acceleration[0] = 0.03 + 0.01 * (i % 5 - 2);
        imu.linear_acceleration[1] = -0.015 + 0.01 * ((i + 1) % 5 - 2);
        imu.linear_acceleration[2] = 9.81 + 0.02 + 0.01 * ((i + 2) % 5 - 2);

        fill_identity_cov(imu.orientation_covariance, 0.01);
        fill_identity_cov(imu.angular_velocity_covariance, 0.001);
        fill_identity_cov(imu.linear_acceleration_covariance, 0.01);

        pelorus_add_imu(pelorus, &imu);
    }
    printf("Added 100 more IMU samples\n");

    // Create second scan (slightly different to enable registration)
    float scan2_data[NUM_POINTS_SCAN * 5]; // Nx5: x,y,z,intensity,t

    printf("Generating second scan with %d points...\n", NUM_POINTS_SCAN);
    for (int i = 0; i < NUM_POINTS_SCAN; i++)
    {
        float x, y, z;
        if (i < NUM_POINTS_SCAN / 2)
        {
            int u = i % 16;
            int v = i / 16;
            x = 4.0f + 0.12f * u;
            y = -1.0f + 0.12f * v;
            z = 0.01f * ((u + v) % 2);
        }
        else
        {
            int j = i - NUM_POINTS_SCAN / 2;
            int u = j % 16;
            int v = j / 16;
            x = 6.0f + 0.005f * ((u + v) % 2);
            y = -1.0f + 0.12f * u;
            z = 0.10f + 0.12f * v;
        }

        const float c = 0.996802f; // cos(0.08 rad)
        const float s = 0.079915f; // sin(0.08 rad)
        float x2 = c * x - s * y + 0.18f;
        float y2 = s * x + c * y - 0.12f;
        float z2 = z + 0.03f;

        scan2_data[i * 5 + 0] = x2;
        scan2_data[i * 5 + 1] = y2;
        scan2_data[i * 5 + 2] = z2;
        scan2_data[i * 5 + 3] = 0.0f; // intensity
        scan2_data[i * 5 + 4] = (float)i / NUM_POINTS_SCAN * 100000000.0f;
    }

    uint64_t scan2_id = pelorus_add_points_xyzit_f32(
        pelorus,
        base_sec + 1, // 1.2 seconds from start
        200000000,
        NULL,
        scan2_data,
        NUM_POINTS_SCAN);
    if (scan2_id == 0)
    {
        fprintf(stderr, "Failed to add second point cloud\n");
    }
    else
    {
        printf("Added second scan id=%llu with %d points\n\n", (unsigned long long)scan2_id, NUM_POINTS_SCAN);
        /* inform callback about the scan id we care about */
        ctx.scan2_id = scan2_id;
    }

    // Wait for the registered scan odometry (signal from callback)
    printf("Waiting for registered odometry callback (scan id %llu)...\n", (unsigned long long)ctx.scan2_id);
    pthread_mutex_lock(&ctx.mutex);
    while (ctx.count < 1)
    {
        pthread_cond_wait(&ctx.cond, &ctx.mutex);
    }
    pthread_mutex_unlock(&ctx.mutex);

    // Disable callback before destroying
    pelorus_set_callback(pelorus, NULL, NULL);

    // Clean up synchronization primitives
    pthread_cond_destroy(&ctx.cond);
    pthread_mutex_destroy(&ctx.mutex);

    // Clean up
    pelorus_destroy(pelorus);
    printf("\nDestroyed Pelorus handle\n");
    printf("Total odometry results received: %d\n", ctx.count);

    return 0;
}

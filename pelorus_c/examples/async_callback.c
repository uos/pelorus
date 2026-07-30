#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "../pelorus.h"
#include "synthetic_dataset.h"

typedef struct {
    int count;
    uint64_t target_scan_id;
    uint64_t latest_scan_id;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} UserContext;

static void odometry_callback(const pelorus_odometry *odom, void *user_data)
{
    UserContext *ctx = (UserContext *)user_data;

    pthread_mutex_lock(&ctx->mutex);
    ctx->count++;
    const int callback_count = ctx->count;
    if (odom->scan_id > ctx->latest_scan_id) {
        ctx->latest_scan_id = odom->scan_id;
    }
    pthread_cond_broadcast(&ctx->cond);
    pthread_mutex_unlock(&ctx->mutex);

    printf("Callback invoked! Odometry #%d received (scan_id=%llu)\n",
           callback_count, (unsigned long long)odom->scan_id);
    printf("  Timestamp: %d.%09u\n", odom->timestamp_sec, odom->timestamp_nanosec);
    printf("  Position: [%.6f, %.6f, %.6f]\n",
           odom->position[0], odom->position[1], odom->position[2]);
    printf("  Linear velocity: [%.6f, %.6f, %.6f]\n",
           odom->linear_velocity[0], odom->linear_velocity[1], odom->linear_velocity[2]);
    printf("  Point cloud: %zu points\n", odom->num_points);

    size_t limit = getenv("PELORUS_DUMP_ALL_POINTS") ? odom->num_points : (odom->num_points < 3 ? odom->num_points : 3);
    for (size_t i = 0; i < limit; i++) {
        const pelorus_point_xyzi *p = &odom->points[i];
        if (getenv("PELORUS_DUMP_ALL_POINTS")) {
            printf("    Point %zu: [%.6g, %.6g, %.6g] I=%.6g\n", i, p->x, p->y, p->z, p->intensity);
        } else {
            printf("    Point %zu: [%.2f, %.2f, %.2f] I=%.1f\n", i, p->x, p->y, p->z, p->intensity);
        }
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    const char *dataset_path = argc >= 2 ? argv[1] : "./synthetic_dataset.bin";
    printf("Version: %s\n\n", pelorus_version());

    pelorus_create_options options = {
        .config_path = "./no-accum.rl",
        .device_specific_pointcloud2 = 0,
    };
    PelorusHandle *pelorus = pelorus_create_with_options(&options);
    if (!pelorus) {
        fprintf(stderr, "Failed to create Pelorus handle\n");
        return 1;
    }
    printf("Created Pelorus handle with comparison config\n");

    UserContext ctx = {0};
    pthread_mutex_init(&ctx.mutex, NULL);
    pthread_cond_init(&ctx.cond, NULL);

    if (pelorus_set_callback(pelorus, odometry_callback, &ctx) != 0) {
        fprintf(stderr, "Failed to set callback\n");
        pelorus_destroy(pelorus);
        return 1;
    }
    printf("Registered callback for async odometry reception\n\n");

    FILE *f = fopen(dataset_path, "rb");
    if (!f) {
        perror(dataset_path);
        pelorus_destroy(pelorus);
        return 1;
    }

    synthetic_record_t rec;
    while (synthetic_read_record(f, &rec) > 0) {
        if (rec.tag == SYNTH_TAG_IMU) {
            pelorus_imu_msg imu;
            synthetic_fill_imu_msg(&imu, &rec.imu);
            pelorus_add_imu(pelorus, &imu);
        } else if (rec.tag == SYNTH_TAG_CLOUD) {
            int32_t sec;
            uint32_t nsec;
            synthetic_split_time(rec.cloud.start_time, &sec, &nsec);
            float *points = synthetic_cloud_to_xyzit_ns(&rec.cloud);
            if (!points) {
                fprintf(stderr, "Failed to allocate cloud buffer\n");
                synthetic_free_record(&rec);
                fclose(f);
                pelorus_destroy(pelorus);
                return 1;
            }
            uint64_t scan_id = pelorus_add_points_xyzit_f32(
                pelorus, sec, nsec, NULL, points, (size_t)rec.cloud.n_points);
            free(points);
            if (scan_id == 0) {
                fprintf(stderr, "Failed to add point cloud\n");
            } else {
                pthread_mutex_lock(&ctx.mutex);
                ctx.target_scan_id = scan_id;
                pthread_mutex_unlock(&ctx.mutex);
                printf("Added scan id=%llu\n", (unsigned long long)scan_id);
            }
        }
        synthetic_free_record(&rec);
    }
    fclose(f);

    pthread_mutex_lock(&ctx.mutex);
    const uint64_t target_scan_id = ctx.target_scan_id;
    pthread_mutex_unlock(&ctx.mutex);

    if (target_scan_id == 0) {
        fprintf(stderr, "Dataset did not contain a valid LiDAR scan\n");
        pelorus_set_callback(pelorus, NULL, NULL);
        pthread_cond_destroy(&ctx.cond);
        pthread_mutex_destroy(&ctx.mutex);
        pelorus_destroy(pelorus);
        return 1;
    }

    printf("Waiting for registered odometry callback (scan id %llu)...\n",
           (unsigned long long)target_scan_id);
    pthread_mutex_lock(&ctx.mutex);
    while (ctx.latest_scan_id < target_scan_id) {
        pthread_cond_wait(&ctx.cond, &ctx.mutex);
    }
    pthread_mutex_unlock(&ctx.mutex);

    pelorus_set_callback(pelorus, NULL, NULL);
    pthread_cond_destroy(&ctx.cond);
    pthread_mutex_destroy(&ctx.mutex);
    pelorus_destroy(pelorus);
    printf("\nDestroyed Pelorus handle\n");
    printf("Total odometry results received: %d\n", ctx.count);
    return 0;
}

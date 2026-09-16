#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "../pelorus.h"
#include "synthetic_dataset.h"

static void print_odom(const pelorus_odometry *odom)
{
    printf("  Timestamp: %d.%09u\n", odom->timestamp_sec, odom->timestamp_nanosec);
    printf("  Position: [%.6f, %.6f, %.6f] (scan_id=%llu)\n",
           odom->position[0], odom->position[1], odom->position[2],
           (unsigned long long)odom->scan_id);
    printf("  Orientation: [%.6f, %.6f, %.6f, %.6f]\n",
           odom->orientation[0], odom->orientation[1], odom->orientation[2], odom->orientation[3]);
    printf("  Linear velocity: [%.6f, %.6f, %.6f]\n",
           odom->linear_velocity[0], odom->linear_velocity[1], odom->linear_velocity[2]);
    printf("  Angular velocity: [%.6f, %.6f, %.6f]\n",
           odom->angular_velocity[0], odom->angular_velocity[1], odom->angular_velocity[2]);
    printf("  Point cloud: %zu points\n", odom->num_points);

    size_t limit = getenv("PELORUS_DUMP_ALL_POINTS") ? odom->num_points : (odom->num_points < 3 ? odom->num_points : 3);
    for (size_t j = 0; j < limit; j++) {
        const pelorus_point_xyzi *p = &odom->points[j];
        if (getenv("PELORUS_DUMP_ALL_POINTS")) {
            printf("    Point %zu: [%.6g, %.6g, %.6g] I=%.6g\n", j, p->x, p->y, p->z, p->intensity);
        } else {
            printf("    Point %zu: [%.2f, %.2f, %.2f] I=%.1f\n", j, p->x, p->y, p->z, p->intensity);
        }
    }
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
    printf("Created Pelorus handle with comparison config\n\n");

    FILE *f = fopen(dataset_path, "rb");
    if (!f) {
        perror(dataset_path);
        pelorus_destroy(pelorus);
        return 1;
    }

    uint64_t target_scan_id = 0;
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
                target_scan_id = scan_id;
                printf("Added scan id=%llu\n", (unsigned long long)scan_id);
            }
        }
        synthetic_free_record(&rec);
    }
    fclose(f);

    if (target_scan_id == 0) {
        fprintf(stderr, "Dataset did not contain a valid Lidar scan\n");
        pelorus_destroy(pelorus);
        return 1;
    }

    printf("Waiting for odometry for scan id %llu (blocking)...\n\n", (unsigned long long)target_scan_id);
    while (1) {
        pelorus_odometry odom = {0};
        const int recv_status = pelorus_recv(pelorus, &odom);
        if (recv_status != PELORUS_STATUS_OK) {
            fprintf(stderr, "Error receiving odometry (status: %d)\n", recv_status);
            break;
        }
        if (odom.scan_id == target_scan_id) {
            print_odom(&odom);
            pelorus_free_odometry(&odom);
            break;
        }
        pelorus_free_odometry(&odom);
    }

    pelorus_destroy(pelorus);
    printf("Destroyed Pelorus handle\n");
    return 0;
}

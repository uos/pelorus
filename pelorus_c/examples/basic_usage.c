#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../pelorus.h"

static void fill_identity_cov(double *cov, double diag) {
    for (size_t i = 0; i < 9; i++) {
        cov[i] = (i % 4 == 0) ? diag : 0.0;
    }
}

int main(void) {
    printf("Pelorus C example\n");
    printf("version: %s\n", pelorus_version());

    PelorusHandle *pelorus = pelorus_create(NULL);
    if (!pelorus) {
        fprintf(stderr, "failed to create pelorus\n");
        return 1;
    }
    printf("created pelorus handle\n");

    CImuMsg imu = {0};
    imu.timestamp_sec = 1;
    imu.timestamp_nanosec = 0;
    imu.frame_id = "imu";
    imu.orientation[0] = 1.0;
    imu.angular_velocity[0] = 0.01;
    imu.angular_velocity[1] = 0.02;
    imu.angular_velocity[2] = 0.03;
    imu.linear_acceleration[2] = 9.81;
    fill_identity_cov(imu.orientation_covariance, 0.01);
    fill_identity_cov(imu.angular_velocity_covariance, 0.001);
    fill_identity_cov(imu.linear_acceleration_covariance, 0.01);

    if (pelorus_add_imu(pelorus, &imu) != 0) {
        fprintf(stderr, "failed to add imu\n");
    } else {
        printf("added imu sample\n");
    }

    // Add odometry message
    COdomMsg odom_input = {0};
    odom_input.timestamp_sec = 1;
    odom_input.timestamp_nanosec = 500000000;
    odom_input.twist_linear[0] = 0.1;  // vx
    odom_input.twist_linear[1] = 0.0;  // vy
    odom_input.twist_linear[2] = 0.0;  // vz
    odom_input.twist_angular[2] = 0.05; // wz
    for (size_t i = 0; i < 36; i++) {
        odom_input.twist_covariance[i] = (i % 7 == 0) ? 0.01 : 0.0;
    }

    if (pelorus_add_odom(pelorus, &odom_input) != 0) {
        fprintf(stderr, "failed to add odometry\n");
    } else {
        printf("added odometry message\n");
    }

    static const CPointField fields[] = {
        {.name = "x", .offset = 0, .datatype = 7, .count = 1},
        {.name = "y", .offset = 4, .datatype = 7, .count = 1},
        {.name = "z", .offset = 8, .datatype = 7, .count = 1},
        {.name = "intensity", .offset = 12, .datatype = 7, .count = 1},
    };

    static const float point_data[] = {
        0.0f, 0.0f, 0.0f, 50.0f,
        1.0f, 0.0f, 0.0f, 80.0f,
        0.0f, 1.0f, 0.0f, 90.0f,
        0.0f, 0.0f, 1.0f, 110.0f,
    };
    const size_t num_points = 4;
    const size_t point_step = 16;

    CPointCloud2 cloud = {0};
    cloud.header_timestamp_sec = 2;
    cloud.header_timestamp_nanosec = 100;
    cloud.header_frame_id = "lidar";
    cloud.height = 1;
    cloud.width = num_points;
    cloud.fields = fields;
    cloud.fields_len = 4;
    cloud.is_bigendian = false;
    cloud.point_step = point_step;
    cloud.row_step = num_points * point_step;
    cloud.data = (const uint8_t *)point_data;
    cloud.data_len = sizeof(point_data);
    cloud.is_dense = true;

    if (pelorus_add_pointcloud(pelorus, &cloud) != 0) {
        fprintf(stderr, "failed to add point cloud\n");
    } else {
        printf("added point cloud with %u points\n", cloud.width);
    }

    CPelorusOdometry odom = {0};
    const int recv_status = pelorus_try_recv(pelorus, &odom);
    if (recv_status == 1) {
        printf("received odometry\n");
        printf("pose: [%.3f, %.3f, %.3f]\n", odom.position[0], odom.position[1], odom.position[2]);
        printf("linear velocity: [%.3f, %.3f, %.3f]\n", odom.linear_velocity[0], odom.linear_velocity[1], odom.linear_velocity[2]);
        printf("cloud points: %zu\n", odom.num_points);
        for (size_t i = 0; i < odom.num_points && i < 3; i++) {
            const CPointXYZI *p = &odom.points[i];
            printf("  p%zu: [%.2f, %.2f, %.2f] I=%.1f\n", i, p->x, p->y, p->z, p->intensity);
        }
        pelorus_free_odometry(&odom);
    } else if (recv_status == 0) {
        printf("no odometry available yet\n");
    } else {
        fprintf(stderr, "error receiving odometry\n");
    }

    pelorus_destroy(pelorus);
    printf("destroyed pelorus handle\n");
    return 0;
}

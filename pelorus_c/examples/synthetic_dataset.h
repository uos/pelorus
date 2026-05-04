#ifndef PELORUS_SYNTHETIC_DATASET_H
#define PELORUS_SYNTHETIC_DATASET_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../pelorus.h"

#define SYNTH_TAG_IMU 1
#define SYNTH_TAG_CLOUD 2

typedef struct {
    float x;
    float y;
    float z;
    float intensity;
    float timestamp;
} synthetic_point_t;

typedef struct {
    double timestamp;
    float gyro[3];
    float accel[3];
} synthetic_imu_t;

typedef struct {
    double start_time;
    double end_time;
    int32_t n_points;
    synthetic_point_t *points;
} synthetic_cloud_t;

typedef struct {
    uint8_t tag;
    synthetic_imu_t imu;
    synthetic_cloud_t cloud;
} synthetic_record_t;

static void synthetic_split_time(double t, int32_t *sec, uint32_t *nsec)
{
    uint64_t total_ns = (uint64_t)(t * 1000000000.0 + 0.5);
    *sec = (int32_t)(total_ns / 1000000000ULL);
    *nsec = (uint32_t)(total_ns % 1000000000ULL);
}

static void synthetic_fill_identity_cov(double *cov, double diag)
{
    for (size_t i = 0; i < 9; i++) {
        cov[i] = (i % 4 == 0) ? diag : 0.0;
    }
}

static void synthetic_fill_imu_msg(pelorus_imu_msg *msg, const synthetic_imu_t *imu)
{
    int32_t sec;
    uint32_t nsec;
    synthetic_split_time(imu->timestamp, &sec, &nsec);
    *msg = (pelorus_imu_msg){0};
    msg->timestamp_sec = sec;
    msg->timestamp_nanosec = nsec;
    msg->orientation[0] = 1.0;
    msg->angular_velocity[0] = imu->gyro[0];
    msg->angular_velocity[1] = imu->gyro[1];
    msg->angular_velocity[2] = imu->gyro[2];
    msg->linear_acceleration[0] = imu->accel[0];
    msg->linear_acceleration[1] = imu->accel[1];
    msg->linear_acceleration[2] = imu->accel[2];
    synthetic_fill_identity_cov(msg->orientation_covariance, 0.01);
    synthetic_fill_identity_cov(msg->angular_velocity_covariance, 0.001);
    synthetic_fill_identity_cov(msg->linear_acceleration_covariance, 0.01);
}

static int synthetic_read_record(FILE *f, synthetic_record_t *record)
{
    record->cloud.points = NULL;
    if (fread(&record->tag, 1, 1, f) != 1) {
        return 0;
    }
    if (record->tag == SYNTH_TAG_IMU) {
        if (fread(&record->imu.timestamp, sizeof(double), 1, f) != 1) return -1;
        if (fread(record->imu.gyro, sizeof(float), 3, f) != 3) return -1;
        if (fread(record->imu.accel, sizeof(float), 3, f) != 3) return -1;
        return 1;
    }
    if (record->tag == SYNTH_TAG_CLOUD) {
        if (fread(&record->cloud.start_time, sizeof(double), 1, f) != 1) return -1;
        if (fread(&record->cloud.end_time, sizeof(double), 1, f) != 1) return -1;
        if (fread(&record->cloud.n_points, sizeof(int32_t), 1, f) != 1) return -1;
        if (record->cloud.n_points <= 0) return -1;
        record->cloud.points = (synthetic_point_t *)malloc((size_t)record->cloud.n_points * sizeof(synthetic_point_t));
        if (!record->cloud.points) return -1;
        if (fread(record->cloud.points, sizeof(synthetic_point_t), (size_t)record->cloud.n_points, f)
            != (size_t)record->cloud.n_points) {
            free(record->cloud.points);
            record->cloud.points = NULL;
            return -1;
        }
        return 1;
    }
    return -1;
}

static void synthetic_free_record(synthetic_record_t *record)
{
    free(record->cloud.points);
    record->cloud.points = NULL;
}

static float *synthetic_cloud_to_xyzit_ns(const synthetic_cloud_t *cloud)
{
    float *buf = (float *)malloc((size_t)cloud->n_points * 5 * sizeof(float));
    if (!buf) return NULL;
    for (int32_t i = 0; i < cloud->n_points; i++) {
        buf[i * 5 + 0] = cloud->points[i].x;
        buf[i * 5 + 1] = cloud->points[i].y;
        buf[i * 5 + 2] = cloud->points[i].z;
        buf[i * 5 + 3] = cloud->points[i].intensity;
        buf[i * 5 + 4] = cloud->points[i].timestamp * 1000000000.0f;
    }
    return buf;
}

#endif

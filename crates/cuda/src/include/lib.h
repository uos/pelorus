#ifndef PELORUS_CUDA_H
#define PELORUS_CUDA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

// Use extern "C" to prevent C++ name mangling, which ensures
// that the function name in the compiled library matches what
// bindgen and the linker expect.
#ifdef __cplusplus
extern "C" {
#endif

// ------------------- Types -------------------

// TODO CHECK ALIGN 16
typedef struct {
  float x;
  float y;
  float z;
  float w;
} Point;

// TODO CHECK ALIGN 16
typedef struct {
  float x;
  float y;
  float z;
  float intensity;
  float normal_x;
  float normal_y;
  float normal_z;
} PointNormal;

typedef struct {       // TODO add more parameters if needed. prefer parameter.h
  double imu_acc_cov;  // default 0.01
  double imu_gyro_cov; // default 0.01
  double imu_acc_bias_cov;  // default 0.0001
  double imu_gyro_bias_cov; // default 0.0001
} Parameter;

typedef struct {
  uint32_t seq;
  double stamp;
  char frame_id[100];
} jlio_header_t;

typedef struct {
  double x;
  double y;
  double z;
} Vector3;

typedef struct {
  double w;
  double x;
  double y;
  double z;
} Quaternion;

typedef struct {
  Vector3 position;
  Quaternion orientation;
} Pose;

typedef struct {
  double data[3][3];
} Matrix3;

typedef struct {
  double timestamp_sec;
  Quaternion orientation;
  double orientation_covariance[9];
  Vector3 angular_velocity;
  double angular_velocity_covariance[9];
  Vector3 linear_acceleration;
  double linear_acceleration_covariance[9];
} ImuMsg;

typedef struct {
  Pose pose;
  Vector3 linear_velocity;
  Matrix3
      lidar_imu_extrinsic_rotation; // When applying offset_R_L_I to a vector in
                                    // the LiDAR frame, you rotate the vector to
                                    // align it with the IMU frame.
  Vector3
      lidar_imu_extrinsic_translation; // When adding offset_T_L_I to a point in
                                       // the LiDAR frame, you effectively
                                       // translate the point to the IMU frame.
} ImuEkfPrediction;

void *cupcl_create_stream();
float *cupcl_create_pointcloud(void *stream, float *point_buffer_cpu,
                               unsigned int n_points);
void *cupcl_create_buffer(void *stream, void *buffer_cpu,
                          unsigned int element_size, unsigned int n_points);
void *cupcl_create_empty_buffer(void *stream, unsigned int element_size,
                                unsigned int n);
void cupcl_upload_buffer(void *stream, void *gpu_buffer, void *cpu_buffer,
                         unsigned int element_size, unsigned int n);
void cupcl_free_buffer(void *buffer);
float *cupcl_create_empty_pointcloud(void *stream, unsigned int n_points);
void cupcl_free_pointcloud(float *cloud);
void cupcl_sync_stream(void *stream);
void cupcl_destroy_stream(void *stream);
void cupcl_download_buffer(void *stream, void *cpu_buffer, void *gpu_buffer,
                           unsigned int element_size, unsigned int n);
void cupcl_copy_buffer(void *stream, void *src, void *target,
                       unsigned int element_size, unsigned int n);

u_int32_t planes(void *stream, Point *neighbor_cache,
                 u_int32_t *neighbor_size_cache, Pose ekf_lidar_to_imu,
                 Pose ekf_pose, u_int32_t k, float neighbor_max_range,
                 double plane_point_error_threshold, bool first_it_local,
                 PointNormal *out_buff);

void add_to_map(void *stream, Point *points, u_int32_t size,
                Pose registered_pose);

void init_map(void *stream, u_int32_t map_capacity, float map_resolution);

#ifdef __cplusplus
}
#endif

#endif // PELORUS_CUDA_H

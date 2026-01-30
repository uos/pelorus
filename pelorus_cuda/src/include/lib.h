#ifndef PELORUS_CUDA_H
#define PELORUS_CUDA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#if defined(__CUDACC__)
#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__)
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)
#define MY_ALIGN(n) __declspec(align(n))
#else
#error "Provide MY_ALIGN"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ------------------- Types -------------------
typedef enum { Center, Nearby6, Nearby18, Nearby26 } NearbyType;

typedef struct MY_ALIGN(4) {
  float x;
  float y;
  float z;
  float i;
  float t;
} Point;

typedef struct MY_ALIGN(16) {
  float x;
  float y;
  float z;
  float intensity;
  float normal_x;
  float normal_y;
  float normal_z;
} PointNormal;

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

void *cupcl_create_stream();
float *cupcl_create_pointcloud(void *stream, Point *point_buffer_cpu,
                               unsigned int n_points);
void *cupcl_create_buffer(void *stream, void *buffer_cpu,
                          unsigned int element_size, unsigned int n_points);
void *cupcl_create_empty_buffer(void *stream, unsigned int element_size,
                                unsigned int n);
void cupcl_upload_buffer(void *stream, void *gpu_buffer, void *cpu_buffer,
                         unsigned int element_size, unsigned int n);
void cupcl_free_buffer(void *buffer);
Point *cupcl_create_empty_pointcloud(void *stream, unsigned int n_points);
void cupcl_free_pointcloud(Point *cloud);
void cupcl_sync_stream(void *stream);
void cupcl_destroy_stream(void *stream);
void cupcl_download_buffer(void *stream, void *cpu_buffer, void *gpu_buffer,
                           unsigned int element_size, unsigned int n);
void cupcl_copy_buffer(void *stream, void *src, void *target,
                       unsigned int element_size, unsigned int n);

u_int32_t planes(void *stream, Point *cloud, uint32_t cloud_size,
                 Pose ekf_lidar_to_imu, Pose ekf_pose, PointNormal *out_buff);

void add_to_map(void *stream, Point *points, u_int32_t size,
                Pose registered_pose);

uint32_t get_map_capacity();
void get_map(Point *points, uint32_t *points_size);
uint32_t get_map_size();

void init_map(void *stream, uint32_t map_capacity, float sensitivity,
              float min_dist, float map_resolution, NearbyType nbtype);

void cleanup_map();

#ifdef __cplusplus
}
#endif

#endif // PELORUS_CUDA_H

#include "lib.h"
#include <cuda_runtime.h>
#include <stdio.h>

#include <core/cuda_debug.h>
#include <core/cuda_stream.h>
#include <core/memory_cuda.h>
#include <math/iso3.h>
#include <sys/types.h>

extern "C" {

void *cupcl_create_empty_buffer(void *stream, unsigned int element_size,
                                unsigned int n) {
  cudaStream_t stream_ = (cudaStream_t)stream;
  void *buff = NULL;
  RM_CUDA_CHECK(cudaMallocAsync(&buff, element_size * n, stream_));
  RM_CUDA_CHECK(cudaMemsetAsync(buff, 0, element_size * n, stream_));
  // RM_CUDA_CHECK(cudaStreamSynchronize(stream_));
  return buff;
}

void cupcl_upload_buffer(void *stream, void *gpu_buffer, void *cpu_buffer,
                         unsigned int element_size, unsigned int n) {
  cudaStream_t stream_ = (cudaStream_t)stream;
  RM_CUDA_CHECK(cudaMemcpyAsync(gpu_buffer, cpu_buffer, element_size * n,
                                cudaMemcpyHostToDevice, stream_));
  // RM_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void cupcl_download_buffer(void *stream, void *cpu_buffer, void *gpu_buffer,
                           unsigned int element_size, unsigned int n) {
  cudaStream_t stream_ = (cudaStream_t)stream;
  RM_CUDA_CHECK(cudaMemcpyAsync(cpu_buffer, gpu_buffer, element_size * n,
                                cudaMemcpyDeviceToHost, stream_));
  // RM_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void cupcl_copy_buffer(void *stream, void *src, void *target,
                       unsigned int element_size, unsigned int n) {
  cudaStream_t stream_ = (cudaStream_t)stream;
  RM_CUDA_CHECK(cudaMemcpyAsync(target, src, element_size * n,
                                cudaMemcpyDeviceToDevice, stream_));
  // RM_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void *cupcl_create_buffer(void *stream, void *buffer_cpu,
                          unsigned int element_size, unsigned int n) {
  cudaStream_t stream_ = (cudaStream_t)stream;
  void *buff = NULL;
  RM_CUDA_CHECK(cudaMallocAsync(&buff, element_size * n, stream_));
  // RM_CUDA_CHECK(cudaStreamAttachMemAsync(stream_, buff));
  RM_CUDA_CHECK(cudaMemcpyAsync(buff, buffer_cpu, element_size * n,
                                cudaMemcpyHostToDevice, stream_));
  // RM_CUDA_CHECK(cudaStreamSynchronize(stream_));
  return buff;
}

void cupcl_free_buffer(void *buffer) {
  if (buffer != NULL) {
    cudaFree(buffer);
    buffer = NULL;
  }
}

void cupcl_sync_stream(void *stream) {
  cudaStream_t stream_ = (cudaStream_t)stream;
  RM_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void *cupcl_create_stream() {
  cudaStream_t stream = NULL;
  RM_CUDA_CHECK(cudaStreamCreate(&stream));
  return (void *)stream;
}

float *cupcl_create_pointcloud(void *stream, float *point_buffer_cpu,
                               unsigned int n_points) {
  return (float *)cupcl_create_buffer(stream, point_buffer_cpu,
                                      sizeof(float) * 4, n_points);
}

float *cupcl_create_empty_pointcloud(void *stream, unsigned int n_points) {
  return (float *)cupcl_create_empty_buffer(stream, sizeof(float) * 4,
                                            n_points);
}

void cupcl_free_pointcloud(float *cloud) { cupcl_free_buffer(cloud); }

void cupcl_destroy_stream(void *stream) {
  if (stream != NULL) {
    RM_CUDA_CHECK(cudaStreamDestroy((cudaStream_t)stream));
    stream = NULL;
  }
}

// TODO static state with params

// TODO sync stream before doing stuff on memory
void add_to_map(void *stream, Point *points, u_int32_t size,
                Pose registered_pose) {
  // TODO
}

void init_map(void *stream, u_int32_t map_capacity, float map_resolution) {
  // TODO
}

// returns: size of out_buff
u_int32_t planes(void *stream, Point *neighbor_cache,
                 u_int32_t *neighbor_size_cache, Pose ekf_lidar_to_imu,
                 Pose ekf_pose, u_int32_t k, float neighbor_max_range,
                 double plane_point_error_threshold, bool first_it_local,
                 PointNormal *out_buff) {
  // TODO

  return 0;
}
}

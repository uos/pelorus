#include "lib.h"
#include <cuda_runtime.h>
#include <stdio.h>

#include <core/cuda_debug.h>
#include <core/cuda_stream.h>
#include <core/memory_cuda.h>

extern "C" {
int allocate_memory_on_gpu(int size) {
  int *d_array = NULL; // Pointer to device memory

  // Calculate the total size in bytes
  size_t byte_size = size * sizeof(int);

  // Allocate memory on the GPU device
  // cudaMalloc takes a pointer to a pointer and the size in bytes.
  cudaError_t malloc_status = cudaMalloc((void **)&d_array, byte_size);

  // Check for errors during memory allocation
  if (malloc_status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed: %s\n",
            cudaGetErrorString(malloc_status));
    return -1; // Return an error code
  }

  printf("Successfully allocated %zu bytes on the GPU.\n", byte_size);

  // In a real application, you would now launch a kernel to use this memory.
  // For this example, we'll just free it immediately.

  // Free the allocated memory on the device
  cudaError_t free_status = cudaFree(d_array);

  // Check for errors during memory deallocation
  if (free_status != cudaSuccess) {
    fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(free_status));
    return -2; // Return a different error code
  }

  printf("Successfully freed memory on the GPU.\n");

  return 0; // Success
}

void *cupcl_create_empty_buffer(void *stream, unsigned int element_size,
                                unsigned int n) {
  cudaStream_t stream_ = (cudaStream_t)stream;
  void *buff = NULL;
  RM_CUDA_CHECK(cudaMallocManaged(&buff, element_size * n, cudaMemAttachHost));
  RM_CUDA_CHECK(cudaStreamAttachMemAsync(stream_, buff));
  RM_CUDA_CHECK(cudaMemsetAsync(buff, 0, element_size * n, stream_));
  RM_CUDA_CHECK(cudaStreamSynchronize(stream_));
  return buff;
}

void cupcl_upload_buffer(void *stream, void *gpu_buffer, void *cpu_buffer,
                         unsigned int element_size, unsigned int n) {
  cudaStream_t stream_ = (cudaStream_t)stream;
  RM_CUDA_CHECK(cudaMemcpyAsync(gpu_buffer, cpu_buffer, element_size * n,
                                cudaMemcpyHostToDevice, stream_));
  RM_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void *cupcl_create_buffer(void *stream, void *buffer_cpu,
                          unsigned int element_size, unsigned int n) {
  cudaStream_t stream_ = (cudaStream_t)stream;
  void *buff = NULL;
  RM_CUDA_CHECK(cudaMallocManaged(&buff, element_size * n, cudaMemAttachHost));
  RM_CUDA_CHECK(cudaStreamAttachMemAsync(stream_, buff));
  RM_CUDA_CHECK(cudaMemcpyAsync(buff, buffer_cpu, element_size * n,
                                cudaMemcpyHostToDevice, stream_));
  RM_CUDA_CHECK(cudaStreamSynchronize(stream_));
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
    cudaStreamDestroy((cudaStream_t)stream);
    stream = NULL;
  }
}
}

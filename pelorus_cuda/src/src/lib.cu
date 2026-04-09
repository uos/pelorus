#include <cassert>
#include <core/common.cuh>
#include <core/cuda_debug.cuh>
#include <core/cuda_stream.cuh>
#include <core/memory.cuh>
#include <core/memory_cuda.cuh>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <lib.h>
#include <localmap.cuh>
#include <memory>

namespace rm = rmagine;

static_assert(sizeof(Point) % 4 == 0, "Point is not aligned properly");

// Convert C NearbyType enum (0/1/2/3) to neighbor count (0/6/18/26)
static uint32_t nbtype_to_count(NearbyType t) {
    switch (t) {
    case Nearby6:  return 6;
    case Nearby18: return 18;
    case Nearby26: return 26;
    default:       return 0;
    }
}

extern "C" {

bool cupcl_is_available() {
  CUresult r = cuInit(0);
  if (r != CUDA_SUCCESS) return false;
  CUdevice dev;
  return cuDeviceGet(&dev, 0) == CUDA_SUCCESS;
}

void *cupcl_create_empty_buffer(void *stream, unsigned int element_size,
                                unsigned int n) {
  try {
    rm::CudaStream *stream_ = (rm::CudaStream *)stream;
    void *buff = NULL;
    RM_CUDA_CHECK(cudaMallocAsync(&buff, element_size * n, stream_->handle()));
    RM_CUDA_CHECK(
        cudaMemsetAsync(buff, 0, element_size * n, stream_->handle()));
    return buff;
  } catch (const std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    std::exit(1);
  } catch (...) {
    std::cerr << "Caught unknown exception." << std::endl;
    std::exit(1);
  }
}

void cupcl_upload_buffer(void *stream, void *gpu_buffer, void *cpu_buffer,
                         unsigned int element_size, unsigned int n) {
  try {
    rm::CudaStream *stream_ = (rm::CudaStream *)stream;
    RM_CUDA_CHECK(cudaMemcpyAsync(gpu_buffer, cpu_buffer, element_size * n,
                                  cudaMemcpyHostToDevice, stream_->handle()));
  } catch (const std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    std::exit(1);
  } catch (...) {
    std::cerr << "Caught unknown exception." << std::endl;
    std::exit(1);
  }
}

void cupcl_download_buffer(void *stream, void *cpu_buffer, void *gpu_buffer,
                           unsigned int element_size, unsigned int n) {
  try {
    rm::CudaStream *stream_ = (rm::CudaStream *)stream;
    RM_CUDA_CHECK(cudaMemcpyAsync(cpu_buffer, gpu_buffer, element_size * n,
                                  cudaMemcpyDeviceToHost, stream_->handle()));
  } catch (const std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    std::exit(1);
  } catch (...) {
    std::cerr << "Caught unknown exception." << std::endl;
    std::exit(1);
  }
}

void cupcl_copy_buffer(void *stream, void *src, void *target,
                       unsigned int element_size, unsigned int n) {
  try {
    rm::CudaStream *stream_ = (rm::CudaStream *)stream;
    RM_CUDA_CHECK(cudaMemcpyAsync(target, src, element_size * n,
                                  cudaMemcpyDeviceToDevice, stream_->handle()));
  } catch (const std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    std::exit(1);
  } catch (...) {
    std::cerr << "Caught unknown exception." << std::endl;
    std::exit(1);
  }
}

void *cupcl_create_buffer(void *stream, void *buffer_cpu,
                          unsigned int element_size, unsigned int n) {
  try {
    rm::CudaStream *stream_ = (rm::CudaStream *)stream;
    void *buff = NULL;
    RM_CUDA_CHECK(cudaMallocAsync(&buff, element_size * n, stream_->handle()));
    RM_CUDA_CHECK(cudaMemcpyAsync(buff, buffer_cpu, element_size * n,
                                  cudaMemcpyHostToDevice, stream_->handle()));
    return buff;
  } catch (const std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    std::exit(1);
  } catch (...) {
    std::cerr << "Caught unknown exception." << std::endl;
    std::exit(1);
  }
}

void cupcl_free_buffer(void *buffer) {
  try {
    if (buffer != NULL) {
      cudaError_t err = cudaFree(buffer);
      if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
        std::cerr << "CUDA free failed: " << cudaGetErrorString(err) << std::endl;
      }
      buffer = NULL;
    }
  } catch (const std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    std::exit(1);
  } catch (...) {
    std::cerr << "Caught unknown exception." << std::endl;
    std::exit(1);
  }
}

void cupcl_sync_stream(void *stream) {
  try {
    rm::CudaStream *stream_ = (rm::CudaStream *)stream;
    stream_->synchronize();
  } catch (const std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    std::exit(1);
  } catch (...) {
    std::cerr << "Caught unknown exception." << std::endl;
    std::exit(1);
  }
}

void *cupcl_create_stream() {
  try {
    rm::CudaStream *stream = new rm::CudaStream();
    return (void *)stream;
  } catch (const std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    std::exit(1);
  } catch (...) {
    std::cerr << "Caught unknown exception." << std::endl;
    std::exit(1);
  }
}

float *cupcl_create_pointcloud(void *stream, Point *point_buffer_cpu,
                               unsigned int n_points) {
  return (float *)cupcl_create_buffer(stream, point_buffer_cpu, sizeof(Point),
                                      n_points);
}

Point *cupcl_create_empty_pointcloud(void *stream, unsigned int n_points) {
  return (Point *)cupcl_create_empty_buffer(stream, sizeof(Point), n_points);
}

void cupcl_free_pointcloud(Point *cloud) { cupcl_free_buffer(cloud); }

void cupcl_destroy_stream(void *stream) {
  if (stream != NULL) {
    static_cast<rm::CudaStream *>(stream)->synchronize();
    delete (rm::CudaStream *)stream;
    stream = NULL;
  }
}

void add_to_map(void *stream, Point *points, u_int32_t size,
                Pose registered_pose, uint32_t scan_id) {
  try {
    assert(stream != NULL);
    rm::CudaStream *stream_ = (rm::CudaStream *)stream;
    add_to_map_impl(stream_->handle(), points, size, registered_pose, scan_id);
  } catch (const std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    std::exit(1);
  } catch (...) {
    std::cerr << "Caught unknown exception." << std::endl;
    std::exit(1);
  }
}

void init_map(void *stream, u_int32_t map_capacity, float sensitivity,
              float min_dist, float map_resolution, NearbyType nbtype) {
  try {
    assert(stream != NULL);
    rm::CudaStream *stream_ = (rm::CudaStream *)stream;
    stream_->context().get()->use();
    init_map_impl(stream_->handle(), map_capacity, map_resolution,
                  sensitivity, min_dist, nbtype_to_count(nbtype));
  } catch (const std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    std::exit(1);
  } catch (...) {
    std::cerr << "Caught unknown exception." << std::endl;
    std::exit(1);
  }
}

void get_map(Point *points, uint32_t *points_size) {
  get_map_impl(points, points_size);
}

uint32_t get_map_capacity() { return get_map_capacity_impl(); }
uint32_t get_map_size()     { return get_map_size_impl(); }

void cleanup_map() { cleanup_map_impl(); }

u_int32_t planes(void *stream, Point *cloud, uint32_t cloud_size,
                 Pose ekf_lidar_to_imu, Pose ekf_pose, PointNormal *out_buff) {
  try {
    rm::CudaStream *stream_ = (rm::CudaStream *)stream;
    return planes_impl(stream_->handle(), cloud, cloud_size,
                       ekf_lidar_to_imu, ekf_pose, out_buff);
  } catch (const std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    std::exit(1);
  } catch (...) {
    std::cerr << "Caught unknown exception." << std::endl;
    std::exit(1);
  }
}

// CUDA Event timing functions
void *cupcl_event_create() {
  cudaEvent_t *event = new cudaEvent_t;
  RM_CUDA_CHECK(cudaEventCreate(event));
  return (void *)event;
}

void cupcl_event_destroy(void *event) {
  if (event != NULL) {
    cudaEvent_t *evt = (cudaEvent_t *)event;
    cudaEventDestroy(*evt);
    delete evt;
  }
}

void cupcl_event_record(void *event, void *stream) {
  rm::CudaStream *stream_ = (rm::CudaStream *)stream;
  cudaEvent_t *evt = (cudaEvent_t *)event;
  RM_CUDA_CHECK(cudaEventRecord(*evt, stream_->handle()));
}

float cupcl_event_elapsed(void *start, void *stop) {
  cudaEvent_t *start_evt = (cudaEvent_t *)start;
  cudaEvent_t *stop_evt = (cudaEvent_t *)stop;
  float ms = 0.0f;
  RM_CUDA_CHECK(cudaEventElapsedTime(&ms, *start_evt, *stop_evt));
  return ms;
}

int cupcl_event_query(void *event) {
  cudaEvent_t *evt = (cudaEvent_t *)event;
  cudaError_t err = cudaEventQuery(*evt);
  return (err == cudaSuccess) ? 1 : 0;
}

} // extern "C"

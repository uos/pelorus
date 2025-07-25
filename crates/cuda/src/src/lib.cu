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
#include <hashmap.cuh>
#include <lib.h>
#include <localmap.cuh>
#include <math/iso3.cuh>
#include <math/quaternion.cuh>
#include <math/vector3.cuh>
#include <memory>
#include <stdio.h>
#include <sys/types.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

extern "C" {

void *cupcl_create_empty_buffer(void *stream, unsigned int element_size,
                                unsigned int n) {
  try {
    rm::CudaStream *stream_ = (rm::CudaStream *)stream;
    void *buff = NULL;
    RM_CUDA_CHECK(cudaMallocAsync(&buff, element_size * n, stream_->handle()));
    RM_CUDA_CHECK(
        cudaMemsetAsync(buff, 0, element_size * n, stream_->handle()));
    // RM_CUDA_CHECK(cudaStreamSynchronize(stream_));
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
    // RM_CUDA_CHECK(cudaStreamSynchronize(stream_));
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
    // RM_CUDA_CHECK(cudaStreamSynchronize(stream_));
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
    // RM_CUDA_CHECK(cudaStreamSynchronize(stream_));
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
    // RM_CUDA_CHECK(cudaStreamAttachMemAsync(stream_, buff));
    RM_CUDA_CHECK(cudaMemcpyAsync(buff, buffer_cpu, element_size * n,
                                  cudaMemcpyHostToDevice, stream_->handle()));
    // RM_CUDA_CHECK(cudaStreamSynchronize(stream_));
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
      cudaFree(buffer);
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
    delete (rm::CudaStream *)stream;
    stream = NULL;
  }
}

class State {
public:
  Localmap localmap;
  uint32_t scan_number;

  State(size_t capacity, float sensitivity, float min_dist, float resolution,
        NearbyType nbtype, void *stream)
      : localmap(capacity, sensitivity, min_dist, resolution, nbtype,
                 *((rm::CudaStream *)stream)),
        scan_number(0) {}
};

std::shared_ptr<State> state;

void add_to_map(void *stream, Point *points, u_int32_t size,
                Pose registered_pose) {

  try {
    assert((state != NULL && stream != NULL));

    rm::CudaStream *stream_ = (rm::CudaStream *)stream;
    state->scan_number++;
    state->localmap.add(points, size, state->scan_number, registered_pose,
                        *stream_);
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
    assert((state == NULL && stream != NULL));
    rm::CudaStream *stream_ = (rm::CudaStream *)stream;

    // NOTE globally set context. is this global process wide?
    stream_->context().get()->use();
    state = std::make_shared<State>(map_capacity, sensitivity, min_dist,
                                    map_resolution, nbtype, stream);

  } catch (const std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    std::exit(1);
  } catch (...) {
    std::cerr << "Caught unknown exception." << std::endl;
    std::exit(1);
  }
}

void get_map(Point *points, uint32_t *points_size) {
  auto cpu_map = state->localmap.get_map();
  *points_size = cpu_map.size();
  for (uint32_t i = 0; i < cpu_map.size(); i++) {
    points[i] = cpu_map[i];
  }
}

uint32_t get_map_capacity() { return state->localmap.capacity; }
uint32_t get_map_size() { return state->localmap.size(); }

__global__ void plane_krnl(
    Point *cloud, uint32_t cloud_size, Point *neighbor_cache,
    uint32_t *neighbor_size_cache, Pose ekf_lidar_to_imu, Pose ekf_pose,
    uint32_t k, float neighbor_max_range, double plane_point_error_threshold,
    bool first_it_local, bool converged, KeyValuePair *map, uint32_t capacity,
    thrust::pair<uint32_t, VoxelDistribution> *buffer, float map_resolution,
    time_t time, NearbyType nbtype, PointNormal *out_buff, uint32_t *out_size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= cloud_size) {
    return;
  }

  rm::Quaternionf ekf_rot(ekf_pose.orientation.w, ekf_pose.orientation.x,
                          ekf_pose.orientation.y, ekf_pose.orientation.z);
  rm::Vector3f ekf_trans(ekf_pose.position.x, ekf_pose.position.y,
                         ekf_pose.position.z);
  rm::Isometry3f global_pose{ekf_rot, ekf_trans};

  rm::Quaternionf ekf_lidar_to_imu_rot(
      ekf_lidar_to_imu.orientation.w, ekf_lidar_to_imu.orientation.x,
      ekf_lidar_to_imu.orientation.y, ekf_lidar_to_imu.orientation.z);
  rm::Vector3f ekf_lidar_to_imu_trans(ekf_lidar_to_imu.position.x,
                                      ekf_lidar_to_imu.position.y,
                                      ekf_lidar_to_imu.position.z);
  rm::Isometry3f lidar_to_imu_pose{ekf_lidar_to_imu_rot,
                                   ekf_lidar_to_imu_trans};

  auto p_raw = cloud[idx];
  rm::Pointf p_local(p_raw.x, p_raw.y, p_raw.z);

  rm::Pointf p_imu = lidar_to_imu_pose * p_local;
  rm::Pointf p_global = global_pose * p_imu;

  rm::Pointf neighbors[MAX_NEIGHBORS];
  uint32_t neighbors_size = 0;

  if (first_it_local || converged) {
    if (!k_nearest_neighbor_query(p_global, k, nbtype, neighbor_max_range, map,
                                  capacity, buffer, map_resolution, neighbors,
                                  neighbors_size, time, idx)) {
      return;
    }
  }

  // Write neighbors to cache if set
  if (neighbors_size > 0) {
    Point *cache_offset = &neighbor_cache[idx * k];
    for (uint32_t i = 0; i < neighbors_size; i++) {
      rm::Pointf p = neighbors[i];
      cache_offset[i] = Point{p.x, p.y, p.z};
    }
    neighbor_size_cache[idx] = neighbors_size;
  } else {
    // Or get from cache if already populated in last iteration
    neighbors_size = neighbor_size_cache[idx];
    Point *cache_offset = &neighbor_cache[idx * k];
    for (uint32_t i = 0; i < neighbors_size; i++) {
      Point p = cache_offset[i];
      neighbors[i] = rm::Pointf{p.x, p.y, p.z};
    }
  }

  // Plane
  float4 plane;
  if (!pca_eigen(neighbors, neighbors_size, 0.1, plane)) {
    return;
  }

  // if (idx < 100) {
  //   for (uint32_t i = 0; i < neighbors_size; i++) {
  //     rm::Pointf p = neighbors[i];
  //     printf(
  //         "  neighbor for point(%.6f, %.6f, %.6f): point(%.6f, %.6f,
  //         %.6f)\n", p_global.x, p_global.y, p_global.z, p.x, p.y, p.z);
  //   }
  // }

  auto pd2 = plane.x * p_global.x + plane.y * p_global.y +
             plane.z * p_global.z + plane.w;
  auto s = 1.0 - 0.9 * abs(pd2) / sqrtf(p_local.l2norm());

  if (s <= 0.9) {
    return;
  }

  PointNormal n = {p_local.x,
                   p_local.y,
                   p_local.z,
                   static_cast<float>(pd2),
                   static_cast<float>(plane.x),
                   static_cast<float>(plane.y),
                   static_cast<float>(plane.z)};

  uint32_t out_idx = atomicAdd(out_size, 1);
  out_buff[out_idx] = n;
}

// returns: size of out_buff, out_buff is on gpu and has enough space
u_int32_t planes(void *stream, Point *cloud, uint32_t cloud_size,
                 Point *neighbor_cache, uint32_t *neighbor_size_cache,
                 Pose ekf_lidar_to_imu, Pose ekf_pose, uint32_t k,
                 float neighbor_max_range, double plane_point_error_threshold,
                 bool first_it_local, PointNormal *out_buff, bool converged) {
  try {
    rm::CudaStream *stream_ = (rm::CudaStream *)stream;
    rm::Memory<uint32_t, rmagine::RAM> out_size_cpu(1);
    *out_size_cpu = 0;
    rm::Memory<uint32_t, rmagine::VRAM_CUDA> out_size_gpu = out_size_cpu;

    int threads_per_block = 256;
    int blocks_per_grid =
        (cloud_size + threads_per_block - 1) / threads_per_block;
    plane_krnl<<<blocks_per_grid, threads_per_block, 0, stream_->handle()>>>(
        cloud, cloud_size, neighbor_cache, neighbor_size_cache,
        ekf_lidar_to_imu, ekf_pose, k, neighbor_max_range,
        plane_point_error_threshold, first_it_local, converged,
        thrust::raw_pointer_cast(state->localmap.lookup->d_table.data()),
        state->localmap.lookup->capacity,
        thrust::raw_pointer_cast(state->localmap.buffer.data()),
        state->localmap.resolution, time(0), state->localmap.nbtype, out_buff,
        out_size_gpu.raw());

    stream_->synchronize();

    checkLastCudaError(__FILE__, __LINE__);

    out_size_cpu = out_size_gpu;
    return *out_size_cpu;
  } catch (const std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    std::exit(1);
  } catch (...) {
    std::cerr << "Caught unknown exception." << std::endl;
    std::exit(1);
  }
}
} // End "C"

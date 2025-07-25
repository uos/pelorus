#ifndef LOCALMAP_H
#define LOCALMAP_H

#include <core/cuda_stream.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <hashmap.h>
#include <lib.h>
#include <math/matrix3x3.h>
#include <math/vector3.h>
#include <memory>
#include <thrust/device_vector.h>
#include <vector>

namespace rm = rmagine;

constexpr uint8_t MAX_NEIGHBORS = 27;

struct VoxelDistribution {
  int lock_flag;
  rm::Matrix3x3f transform;
  uint32_t point_count;
  rm::Vector3f mean;
  rm::Matrix3x3f covariance;
  uint32_t last_updated_at_scan;

  __host__ __device__ VoxelDistribution()
      : mean(rm::Vector3f(0.0f, 0.0f, 0.0f)), point_count(0),
        last_updated_at_scan(0), lock_flag(0) {}

  __host__ __device__ VoxelDistribution(const rm::Vector3f &point)
      : mean(point), point_count(1), last_updated_at_scan(0), lock_flag(0) {}

  __device__ void update(const rm::Vector3f &point, float sensitivity,
                         float min_dist, size_t scan_number);

  __device__ void lock();
  __device__ void unlock();
};
using Value = thrust::pair<uint32_t, VoxelDistribution>;

class Localmap {
public:
  Localmap(size_t capacity, float sensitivity, float min_dist, float resolution,
           NearbyType nbtype, rm::CudaStream &stream);
  ~Localmap();

  /// Adds or updates a voxel in the map.
  void add(Point *points, uint32_t point_size, uint32_t scan_number,
           Pose registered_pose, rm::CudaStream &stream);

  /// Finds a voxel distribution by its hash.
  /// Note: This copies data from device to host. Returns false if not found.
  bool find(uint32_t key, const VoxelDistribution &out_dist) const;

  /// Sorts the internal mapping to prepare for LRU eviction.
  void build(size_t scan_number, rm::CudaStream &stream);

  /// Clears the map of all entries.
  void clear();

  /// Returns a host vector of the means of all stored voxel distributions.
  std::vector<Point> get_map() const;

  /// Returns the number of active elements in the map.
  uint32_t size() const { return std::min(this->add_cursor, this->capacity); }

  /// Checks if the map is empty.
  bool is_empty() const { return add_cursor == 0; }

  thrust::device_vector<Value> buffer;

  std::unique_ptr<GpuHashMap> lookup;

  thrust::device_vector<uint32_t> lru_mapping;

  uint32_t add_cursor;
  uint32_t lru_cursor;

  // Configuration parameters
  uint32_t capacity;
  float resolution;
  float sensitivity;
  float min_dist;
  NearbyType nbtype;
};

__device__ bool pca_constant(const rm::Vector3f *cluster, uint32_t n,
                             float normal_threshold, double4 &plane_equation);
__device__ bool pca_eigen_robust(const rm::Vector3f *cluster, uint32_t n,
                                 float normal_threshold,
                                 float4 &plane_equation);

__device__ bool
k_nearest_neighbor_query(const rm::Pointf &query, uint32_t k, NearbyType nbtype,
                         float max_range, KeyValuePair *map, uint32_t capacity,
                         thrust::pair<uint32_t, VoxelDistribution> *buffer,
                         float resolution, rm::Pointf *result,
                         uint32_t &out_size, time_t time, int thread_id);

#endif

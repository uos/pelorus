#ifndef LOCALMAP_H
#define LOCALMAP_H

#include <core/cuda_stream.h>
#include <cuda_runtime.h>
#include <hashmap.h>
#include <lib.h>
#include <math/matrix3x3.h>
#include <math/vector3.h>
#include <thrust/device_vector.h>
#include <utility>
#include <vector>

namespace rm = rmagine;

struct VoxelDistribution {
  rm::Matrix3x3f transform;
  u_int64_t point_count;
  rm::Vector3f mean;
  rm::Matrix3x3f covariance;
  size_t last_updated_at_scan;

  __host__ __device__ VoxelDistribution()
      : mean(rm::Vector3f(0.0f, 0.0f, 0.0f)), point_count(0),
        last_updated_at_scan(0) {}

  __host__ __device__ VoxelDistribution(const rm::Vector3f &point)
      : mean(point), point_count(1), last_updated_at_scan(0) {}

  __device__ void update(const rm::Vector3f &point, float sensitivity,
                         float min_dist, size_t scan_number);
};
using Value = thrust::pair<uint32_t, VoxelDistribution>;

class Localmap {
public:
  Localmap(size_t capacity, float sensitivity, float min_dist, float resolution,
           rm::CudaStream &stream);
  ~Localmap();

  void add(Point *points, size_t point_size, size_t scan_number);

  /// Adds or updates a voxel in the map.
  void add(Point *points, size_t point_size, size_t scan_number,
           rm::CudaStream &stream);

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
  size_t size() const { return add_cursor; }

  /// Checks if the map is empty.
  bool is_empty() const { return add_cursor == 0; }

private:
  thrust::device_vector<Value> buffer;

  thrust::device_vector<uint32_t> lru_mapping;

  std::shared_ptr<GpuHashMap> lookup;

  size_t add_cursor;
  size_t lru_cursor;

  // Configuration parameters
  size_t capacity;
  size_t resolution;
  float sensitivity;
  float min_dist;
};

#endif

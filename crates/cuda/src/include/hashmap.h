#ifndef HASHMAP_H
#define HASHMAP_H

#include <core/cuda_stream.h>
#include <memory>
#include <optional>
constexpr int32_t EMPTY_KEY = INT32_MIN;
constexpr int32_t DELETED_SLOT = INT32_MIN + 1;

struct KeyValuePair {
  int32_t key;
  int32_t value;
};

// Host-side handle for the GPU hash map.
class GpuHashMap {
public:
  KeyValuePair *d_table;
  size_t capacity;
};

std::shared_ptr<GpuHashMap> create_map(size_t capacity,
                                       rmagine::CudaStream &stream);

void clear_map(std::shared_ptr<GpuHashMap> map);

void destroy_map(std::shared_ptr<GpuHashMap> &map);

__device__ void insert_item(KeyValuePair *table, size_t capacity, int32_t key,
                            uint64_t value);

__device__ bool remove_item(KeyValuePair *table, size_t capacity, int32_t key);

__device__ uint32_t hash_function(int32_t key);

__device__ uint32_t lookup_item(const KeyValuePair *table, size_t capacity,
                                int32_t key, bool &found);

#endif

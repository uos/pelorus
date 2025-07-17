#include <core/common.h>
#include <core/cuda_stream.h>
#include <core/memory_cuda.h>

#include <core/cuda_debug.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

constexpr int32_t EMPTY_KEY = INT32_MIN;

// The key-value pair stored in the hash table.
// Maps a 32-bit integer to a 64-bit unsigned integer (size_t).
struct KeyValuePair {
  int32_t key;
  uint64_t value;
};

// Host-side handle for the GPU hash map.
class GpuHashMap {
public:
  KeyValuePair *d_table;
  size_t capacity;
};

// Murmur3
__device__ uint32_t hash_function(int32_t key) {
  uint32_t h = key;
  h = (h ^ (h >> 16)) * 0x85ebca6b;
  h = (h ^ (h >> 13)) * 0xc2b2ae35;
  h = h ^ (h >> 16);
  return h;
}

__device__ void insert_item(KeyValuePair *table, size_t capacity, int32_t key,
                            uint64_t value) {
  if (key == EMPTY_KEY)
    return;

  uint32_t slot = hash_function(key) & (capacity - 1);

  while (true) {
    int32_t prev_key = atomicCAS(&table[slot].key, EMPTY_KEY, key);

    if (prev_key == EMPTY_KEY || prev_key == key) {
      table[slot].value = value;
      return;
    }

    // Collision: the slot was occupied by a different key. Move to the next.
    slot = (slot + 1) & (capacity - 1);
  }
}

__device__ uint64_t lookup_item(const KeyValuePair *table, size_t capacity,
                                int32_t key, uint64_t not_found_value) {
  if (key == EMPTY_KEY)
    return not_found_value;

  uint32_t slot = hash_function(key) & (capacity - 1);

  while (true) {
    int32_t current_key = table[slot].key;

    // If we find the key, return its associated value.
    if (current_key == key) {
      return table[slot].value;
    }

    // If we find an empty slot, the key cannot be in the table,
    // because linear probing does not leave gaps.
    if (current_key == EMPTY_KEY) {
      return not_found_value;
    }

    // Otherwise, continue probing.
    slot = (slot + 1) & (capacity - 1);
  }
}

// --- Kernels ---

__global__ void initialize_kernel(KeyValuePair *table, size_t capacity) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < capacity) {
    table[idx].key = EMPTY_KEY;
    table[idx].value = 0;
  }
}

__global__ void insert_kernel(KeyValuePair *table, size_t capacity,
                              const int32_t *keys, const uint64_t *values,
                              size_t num_items) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_items) {
    insert_item(table, capacity, keys[idx], values[idx]);
  }
}

__global__ void lookup_kernel(const KeyValuePair *table, size_t capacity,
                              const int32_t *keys_to_find, uint64_t *results,
                              size_t num_items, uint64_t not_found_value) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_items) {
    results[idx] =
        lookup_item(table, capacity, keys_to_find[idx], not_found_value);
  }
}

// --- Host-Side API Functions ---

// Creates and initializes a hash map on the GPU.
std::shared_ptr<GpuHashMap> create_map(size_t capacity,
                                       rmagine::CudaStream &stream) {
  // Ensure capacity is a power of two for efficient modulo via bitwise AND.
  if (capacity == 0 || (capacity & (capacity - 1)) != 0) {
    throw std::invalid_argument("Capacity must be a power of two.");
  }

  auto map = std::make_shared<GpuHashMap>();
  map->capacity = capacity;

  RM_CUDA_CHECK(cudaMallocAsync(&map->d_table, capacity * sizeof(KeyValuePair),
                                stream.handle()));
  stream.synchronize();

  // Initialize the table with the EMPTY_KEY sentinel.
  int threads_per_block = 256;
  int blocks_per_grid = (capacity + threads_per_block - 1) / threads_per_block;
  initialize_kernel<<<blocks_per_grid, threads_per_block>>>(map->d_table,
                                                            capacity);

  checkLastCudaError(__FILE__, __LINE__);

  return map;
}

// Frees the GPU memory associated with the hash map.
void destroy_map(std::shared_ptr<GpuHashMap> &map) {
  if (map.get()) {
    RM_CUDA_CHECK(cudaFree(map->d_table));
  }
}

// Inserts a batch of key-value pairs into the hash map.
void insert_batch(GpuHashMap *map, const std::vector<int32_t> &h_keys,
                  const std::vector<uint64_t> &h_values,
                  rmagine::CudaStream &stream) {
  if (h_keys.size() != h_values.size() || h_keys.empty()) {
    return;
  }

  size_t num_items = h_keys.size();
  int32_t *d_keys;
  uint64_t *d_values;

  RM_CUDA_CHECK(
      cudaMallocAsync(&d_keys, num_items * sizeof(int32_t), stream.handle()));
  RM_CUDA_CHECK(cudaMallocAsync(&d_values, num_items * sizeof(uint64_t),
                                stream.handle()));

  RM_CUDA_CHECK(cudaMemcpyAsync(d_keys, h_keys.data(),
                                num_items * sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream.handle()));
  RM_CUDA_CHECK(cudaMemcpyAsync(d_values, h_values.data(),
                                num_items * sizeof(uint64_t),
                                cudaMemcpyHostToDevice, stream.handle()));

  stream.synchronize();

  int threads_per_block = 256;
  int blocks_per_grid = (num_items + threads_per_block - 1) / threads_per_block;
  insert_kernel<<<blocks_per_grid, threads_per_block>>>(
      map->d_table, map->capacity, d_keys, d_values, num_items);

  checkLastCudaError(__FILE__, __LINE__);

  cudaFree(d_keys);
  cudaFree(d_values);
}

// Looks up a batch of keys and returns the corresponding values.
std::vector<uint64_t> lookup_batch(GpuHashMap *map,
                                   const std::vector<int32_t> &h_keys_to_find,
                                   uint64_t not_found_value) {
  if (h_keys_to_find.empty()) {
    return {};
  }

  size_t num_items = h_keys_to_find.size();
  int32_t *d_keys_to_find;
  uint64_t *d_results;

  cudaMalloc(&d_keys_to_find, num_items * sizeof(int32_t));
  cudaMalloc(&d_results, num_items * sizeof(uint64_t));

  cudaMemcpy(d_keys_to_find, h_keys_to_find.data(), num_items * sizeof(int32_t),
             cudaMemcpyHostToDevice);

  int threads_per_block = 256;
  int blocks_per_grid = (num_items + threads_per_block - 1) / threads_per_block;
  lookup_kernel<<<blocks_per_grid, threads_per_block>>>(
      map->d_table, map->capacity, d_keys_to_find, d_results, num_items,
      not_found_value);

  std::vector<uint64_t> h_results(num_items);
  cudaMemcpy(h_results.data(), d_results, num_items * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);

  cudaFree(d_keys_to_find);
  cudaFree(d_results);

  return h_results;
}

// --- Main Function (Example Usage) ---

// int main() {
//   try {
//     const size_t capacity = 1 << 20;            // 1,048,576 slots
//     const size_t num_items_to_insert = 500'000; // ~47.6% load factor
//     const uint64_t not_found_sentinel = UINT64_MAX;

//     rmagine::CudaStream stream;
//     auto map = create_map(capacity, stream);
//     std::cout << "GPU Hash Map created with capacity " << capacity <<
//     std::endl;

//     // Prepare host data for insertion
//     std::vector<int32_t> h_keys(num_items_to_insert);
//     std::vector<uint64_t> h_values(num_items_to_insert);
//     for (size_t i = 0; i < num_items_to_insert; ++i) {
//       h_keys[i] = static_cast<int32_t>(i * 3); // Use non-consecutive keys
//       h_values[i] = static_cast<uint64_t>(i);
//     }

//     std::cout << "Inserting " << num_items_to_insert << " items..."
//               << std::endl;
//     insert_batch(map.get(), h_keys, h_values, stream);
//     cudaDeviceSynchronize(); // Wait for insertion to complete

//     // Prepare keys for lookup (some exist, some don't)
//     std::vector<int32_t> h_keys_to_find = {0, 5, 6, 9, 999999, 12};
//     std::cout << "Looking up " << h_keys_to_find.size() << " keys..."
//               << std::endl;
//     std::vector<uint64_t> h_results =
//         lookup_batch(map.get(), h_keys_to_find, not_found_sentinel);

//     for (size_t i = 0; i < h_keys_to_find.size(); ++i) {
//       std::cout << "Key: " << h_keys_to_find[i] << " -> Value: ";
//       if (h_results[i] == not_found_sentinel) {
//         std::cout << "NOT FOUND" << std::endl;
//       } else {
//         std::cout << h_results[i] << std::endl;
//       }
//     }

//     destroy_map(map);
//     std::cout << "GPU Hash Map destroyed." << std::endl;

//   } catch (const std::exception &e) {
//     std::cerr << "Error: " << e.what() << std::endl;
//     return 1;
//   }

//   return 0;
// }

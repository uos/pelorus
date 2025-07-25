#include <cassert>
#include <core/common.h>
#include <core/cuda_stream.h>
#include <core/memory_cuda.h>

#include <core/cuda_debug.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

#include <hashmap.h>

// Murmur3
__device__ uint32_t hash_function(int32_t key) {
  uint32_t h = key;
  h = (h ^ (h >> 16)) * 0x85ebca6b;
  h = (h ^ (h >> 13)) * 0xc2b2ae35;
  h = h ^ (h >> 16);
  return h;
}

__device__ void insert_item(KeyValuePair *table, size_t capacity, int32_t key,
                            uint32_t value) {
  if (key == EMPTY_KEY || key == DELETED_SLOT)
    return;

  uint32_t slot = hash_function(key) & (capacity - 1);

  while (true) {
    int32_t current_key = table[slot].key;

    // If slot is available (empty or deleted), try to claim it.
    if (current_key == EMPTY_KEY || current_key == DELETED_SLOT) {
      int32_t prev_key = atomicCAS(&table[slot].key, current_key, key);

      // If we successfully swapped (we got the slot), write the value and exit.
      if (prev_key == current_key) {
        table[slot].value = value;
        return;
      }
      // If we failed, another thread beat us. We must re-evaluate the slot in
      // the next loop iteration. The slot now holds a new key, so the next
      // check will treat it as a collision.
      continue;
    }

    // If the key already exists, update the value and exit.
    if (current_key == key) {
      table[slot].value = value; // Note: Not atomic, assumes one writer per key
                                 // or last-write-wins is OK.
      return;
    }

    // Collision: Move to the next slot.
    slot = (slot + 1) & (capacity - 1);
  }
}

__device__ uint32_t lookup_item(const KeyValuePair *table, size_t capacity,
                                int32_t key, bool &found) {
  if (key == EMPTY_KEY || key == DELETED_SLOT) {
    found = false;
    return 0;
  }

  uint32_t slot = hash_function(key) & (capacity - 1);

  while (true) {
    int32_t current_key = table[slot].key;

    if (current_key == key) {
      found = true;
      // assert((false));
      return table[slot].value;
    }

    if (current_key == EMPTY_KEY) {
      // An empty slot breaks the chain. Key not found.
      found = false;
      return 0;
    }

    // If it's a DELETED_SLOT or a different key, continue probing.
    slot = (slot + 1) & (capacity - 1);
  }
}

__device__ bool remove_item(KeyValuePair *table, size_t capacity, int32_t key) {
  if (key == EMPTY_KEY || key == DELETED_SLOT) {
    return false;
  }

  uint32_t slot = hash_function(key) & (capacity - 1);

  while (true) {
    int32_t current_key = table[slot].key;

    if (current_key == key) {
      // Found it. Atomically swap with the tombstone.
      atomicExch(&table[slot].key, DELETED_SLOT);
      return true;
    }

    if (current_key == EMPTY_KEY) {
      // Empty slot breaks the chain. Key not found.
      return false;
    }

    // If it's a DELETED_SLOT or a different key, continue probing.
    slot = (slot + 1) & (capacity - 1);
  }
}

// --- Kernels ---

__global__ void initialize_kernel(KeyValuePair *table, uint32_t capacity) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < capacity) {
    table[idx].key = EMPTY_KEY;
    table[idx].value = 0;
  }
}

std::unique_ptr<GpuHashMap> create_map(size_t capacity) {
  // Ensure capacity is a power of two for efficient modulo via bitwise AND.
  if (capacity == 0 || (capacity & (capacity - 1)) != 0) {
    throw std::invalid_argument("Capacity must be a power of two.");
  }

  auto map = std::make_unique<GpuHashMap>();
  map->capacity = capacity * 2;
  map->d_table = thrust::device_vector<KeyValuePair>(map->capacity);

  KeyValuePair *lookup = thrust::raw_pointer_cast(map->d_table.data());
  // Initialize the table with the EMPTY_KEY sentinel.
  int threads_per_block = 256;
  int blocks_per_grid =
      (map->capacity + threads_per_block - 1) / threads_per_block;
  uint32_t capacity_krnl = map->capacity;
  initialize_kernel<<<blocks_per_grid, threads_per_block>>>(lookup,
                                                            capacity_krnl);

  RM_CUDA_CHECK(cudaDeviceSynchronize());
  checkLastCudaError(__FILE__, __LINE__);

  return map;
}

void clear_map(std::unique_ptr<GpuHashMap> &map) {

  // Initialize the table with the EMPTY_KEY sentinel.
  int threads_per_block = 256;
  int blocks_per_grid =
      (map->capacity + threads_per_block - 1) / threads_per_block;
  initialize_kernel<<<blocks_per_grid, threads_per_block>>>(
      thrust::raw_pointer_cast(map->d_table.data()), map->capacity);

  checkLastCudaError(__FILE__, __LINE__);
}

void destroy_map(std::unique_ptr<GpuHashMap> &map) { map.reset(); }

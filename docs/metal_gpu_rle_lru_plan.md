# Metal GPU RLE and LRU Implementation Plan

This document captures the changes needed to move RLE (run-length encoding) and LRU eviction from CPU to GPU in the Metal backend.

## Status

The implementation compiles but causes trajectory divergence (APE RMSE ~0.035m vs threshold 0.01m). The root cause needs further debugging.

## Files to Modify

### 1. `pelorus_metal/src/shaders/localmap.metal`

Add three new kernels:

```metal
// RLE kernel - finds run boundaries in sorted keys
kernel void rle(
    device const int* sorted_keys [[buffer(0)]],
    device int* unique_keys [[buffer(1)]],
    device uint* run_starts [[buffer(2)]],
    device uint* run_ends [[buffer(3)]],
    device atomic_uint* unique_count [[buffer(4)]],
    constant BuildVoxelsParams& params [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= params.n_points) return;
    
    int key = sorted_keys[idx];
    if (key == EMPTY_KEY) return;
    
    bool is_start = (idx == 0) || (sorted_keys[idx - 1] != key);
    if (!is_start) return;
    
    // Binary search for run end
    uint left = idx;
    uint right = params.n_points;
    while (left < right) {
        uint mid = left + (right - left) / 2;
        if (sorted_keys[mid] <= key) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    uint slot = atomic_fetch_add_explicit(unique_count, 1, memory_order_relaxed);
    run_starts[slot] = idx;
    run_ends[slot] = left;
    unique_keys[slot] = key;
}

// Assign indices kernel - hashmap lookup + LRU eviction
kernel void assign_indices(
    device const int* unique_keys [[buffer(0)]],
    device atomic_int* table_keys [[buffer(1)]],
    device uint* table_vals [[buffer(2)]],
    device int* dist_keys [[buffer(3)]],
    device const uint* lru_mapping [[buffer(4)]],
    device uint* target_indices [[buffer(5)]],
    device uint* update_flags [[buffer(6)]],
    device atomic_uint* counters [[buffer(7)]],  // {unique_count, add_cursor, lru_cursor}
    constant BuildVoxelsParams& params [[buffer(8)]],
    uint idx [[thread_position_in_grid]])
{
    uint unique_count = atomic_load_explicit(&counters[0], memory_order_relaxed);
    if (idx >= unique_count) return;
    
    int key = unique_keys[idx];
    if (key == EMPTY_KEY) return;
    
    // Hashmap lookup
    uint slot = hash_fn(key) & (params.table_cap - 1);
    int found_idx = -1;
    
    for (uint i = 0; i < params.table_cap; i++) {
        int cur = atomic_load_explicit(&table_keys[slot], memory_order_relaxed);
        if (cur == key) {
            found_idx = int(table_vals[slot]);
            break;
        }
        if (cur == EMPTY_KEY) break;
        slot = (slot + 1) & (params.table_cap - 1);
    }
    
    if (found_idx != -1) {
        target_indices[idx] = uint(found_idx);
        update_flags[idx] = 1;
    } else {
        update_flags[idx] = 0;
        uint old_add = atomic_fetch_add_explicit(&counters[1], 1u, memory_order_relaxed);
        uint assign_idx;
        
        if (old_add < params.dist_cap) {
            assign_idx = old_add;
        } else {
            uint lru_idx = atomic_fetch_add_explicit(&counters[2], 1u, memory_order_relaxed);
            assign_idx = lru_mapping[lru_idx % params.dist_cap];
        }
        
        target_indices[idx] = assign_idx;
        dist_keys[assign_idx] = key;
    }
}

// LRU score kernel
kernel void lru_score(
    device const VoxelDistribution* dists [[buffer(0)]],
    device uint* lru_keys [[buffer(1)]],
    device uint* lru_vals [[buffer(2)]],
    constant BuildVoxelsParams& params [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= params.dist_cap) return;
    
    uint pt_count = dists[idx].point_count;
    uint temporal_dist = params.scan_number - dists[idx].last_scan;
    
    uint key;
    if (pt_count == 0) {
        key = 0;  // Empty voxels evict first
    } else {
        float score = float(pt_count) / float(temporal_dist + 1u);
        key = (1u << 30) | uint(clamp(score * 1000.0f, 0.0f, float((1u << 30) - 1u)));
    }
    
    lru_keys[idx] = key;
    lru_vals[idx] = idx;
}
```

Update `BuildVoxelsParams` struct to use `n_points` instead of `unique_key_count`.

Update `calculate_keys`, `reorder_points`, and `build_voxels` to read `unique_count` from atomic counter buffer.

### 2. `pelorus_metal/src/pipeline.rs`

Add pipeline states:

```rust
pub struct Pipelines {
    // ... existing fields ...
    pub rle: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub assign_indices: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub lru_score: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    // ...
}

// In compile():
rle: make("rle"),
assign_indices: make("assign_indices"),
lru_score: make("lru_score"),
```

### 3. `pelorus_metal/src/map.rs`

#### Struct Changes

```rust
pub struct ScratchBuffers {
    // ... existing fields ...
    pub counter: crate::buffer::MetalBuffer,  // {unique_count, add_cursor, lru_cursor} - 3 elements
    pub lru_keys: crate::buffer::MetalBuffer,
    pub lru_mapping: crate::buffer::MetalBuffer,
}

pub struct MetalMapState {
    // Remove: add_cursor_buf, key_to_index, lru_mapping (Vec), lru_mapping_keys, pending_lru_rebuild
    // Add:
    pub add_cursor: u32,  // CPU-side tracking, synced after each frame
}
```

#### `init_map_impl` Changes

- Allocate `lru_keys` and `lru_mapping` buffers with size `dist_cap`
- Initialize `lru_mapping` to identity: [0, 1, 2, ..., dist_cap-1]
- Allocate `counter` buffer with 3 elements (unique_count, add_cursor, lru_cursor)

#### `add_to_map_impl` Rewrite

Single command buffer with all kernels:

1. **calculate_keys** - transform points, compute keys
2. **Bitonic sort** - pack_keys, bitonic_step, unpack_keys
3. **rle** - run-length encoding
4. **assign_indices** - hashmap lookup + LRU eviction
5. **reorder_points** - reorder by sorted indices
6. **build_voxels** - update voxel distributions
7. **hashmap_init** - clear hash table
8. **rebuild_hashmap_keyed** - rebuild from dist_keys
9. **(conditional) lru_score + sort** - if add_cursor >= dist_cap

Key synchronization points:
- Use `enc.memoryBarrierWithScope(MTLBarrierScope::Buffers)` between kernels
- After `cb.waitUntilCompleted()`, sync `state.add_cursor` from `counters[1]`

#### Remove CPU Functions

- `assign_target_indices_with_cpu_lru`
- `rebuild_cpu_lru_order`

## Buffer Layout

| Buffer | Size | Purpose |
|--------|------|---------|
| `counter` | 3 | {unique_count, add_cursor, lru_cursor} |
| `lru_keys` | dist_cap | LRU eviction scores |
| `lru_mapping` | dist_cap | Voxel indices sorted by LRU score |
| `unique_keys` | n_points | Unique voxel keys from RLE |
| `run_starts` | n_points | Run start indices |
| `run_ends` | n_points | Run end indices |
| `target_indices` | n_points | Assigned distribution indices |
| `update_flags` | n_points | 1=update, 0=insert |

## Kernel Dispatch Order

```
calculate_keys(n_points)
  barrier
pack_keys(n_points)
[init_pad if needed]
  barrier
bitonic_step(padded_n) x log(padded_n)
  barrier
unpack_keys(n_points)
  barrier
rle(n_points)
  barrier
assign_indices(n_points)  -- reads unique_count from counter[0]
  barrier
reorder_points(n_points)
  barrier
build_voxels(n_points)
  barrier
hashmap_init(table_cap)
  barrier
rebuild_hashmap_keyed(add_cursor)
  barrier
[if add_cursor >= dist_cap:]
  lru_score(dist_cap)
    barrier
  pack_keys(dist_cap)  -- reuse packed buffer
    barrier
  bitonic_step(dist_cap_padded)
    barrier
  unpack_keys(dist_cap)
```

## Debugging Notes

### Potential Issues

1. **Counter buffer ordering**: Ensure `unique_count` is read correctly by `assign_indices` after `rle` writes it.

2. **LRU condition timing**: The LRU rebuild decision uses CPU-side `add_cursor` from PREVIOUS frame, which is correct (matches Vulkan approach).

3. **Hashmap lookup race**: Multiple threads may lookup same key simultaneously - this is fine as they'll all find the same entry.

4. **Dist_keys overwrites**: When evicting, `dist_keys[slot]` is overwritten with new key. Hashmap rebuild iterates over `dist_keys`, so old key is not re-inserted.

5. **Packed buffer sizing**: Must be at least `max(n_points, dist_cap)` for both main sort and LRU sort.

### Testing Approach

1. Add debug logging to print:
   - `unique_count` after RLE
   - `add_cursor` after assign_indices
   - `lru_cursor` to track evictions

2. Compare CPU vs GPU RLE output for a small point cloud

3. Unit test the binary search in RLE kernel

4. Verify LRU score ordering (empty voxels first, then by score ascending)

## Reverting

If issues persist, revert to CPU-based RLE/LRU:

```bash
git checkout HEAD -- pelorus_metal/src/map.rs pelorus_metal/src/lib.rs pelorus_metal/src/pipeline.rs pelorus_metal/src/shaders/localmap.metal
```

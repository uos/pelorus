// hashmap.metal — GPU hash table init and insert kernels
// Included after common.metal in the compiled source.

// Initialize hash table: set all keys to EMPTY_KEY
kernel void hashmap_init(
    device atomic_int* table_keys   [[buffer(0)]],
    constant uint&     capacity     [[buffer(1)]],
    uint               idx          [[thread_position_in_grid]])
{
    if (idx >= capacity) return;
    atomic_store_explicit(&table_keys[idx], EMPTY_KEY, memory_order_relaxed);
}

// Rebuild the hashmap from the current distribution buffer.
// One thread per valid distribution entry (idx in 0..valid_size).
kernel void rebuild_hashmap(
    device atomic_int*         table_keys   [[buffer(0)]],
    device uint*               table_vals   [[buffer(1)]],
    device const VoxelDistribution* dists  [[buffer(2)]],
    constant uint&             valid_size   [[buffer(3)]],
    constant uint&             table_cap    [[buffer(4)]],
    uint                       idx          [[thread_position_in_grid]])
{
    if (idx >= valid_size) return;

    int key = int(dists[idx].point_count == 0 ? uint(EMPTY_KEY) : 0);
    // We stored the voxel key separately; read from the dist_keys companion buffer.
    // (See build_voxels which writes dist_keys[idx] = voxel_key.)
    // Here we receive the key through a dedicated dist_keys buffer.
    // This variant is called with dist_keys provided at buffer(5).
    // (Overloaded signature handled in pipeline.rs by selecting the right fn.)
    return; // Placeholder: actual work in rebuild_hashmap_keyed below.
}

// Rebuild using a companion key buffer (int dist_keys[valid_size]).
kernel void rebuild_hashmap_keyed(
    device atomic_int*              table_keys [[buffer(0)]],
    device uint*                    table_vals [[buffer(1)]],
    device const int*               dist_keys  [[buffer(2)]],
    constant uint&                  valid_size [[buffer(3)]],
    constant uint&                  table_cap  [[buffer(4)]],
    uint                            idx        [[thread_position_in_grid]])
{
    if (idx >= valid_size) return;

    int key = dist_keys[idx];
    if (key == EMPTY_KEY) return;

    uint slot = hash_fn(key) & (table_cap - 1);
    for (;;) {
        int expected = EMPTY_KEY;
        bool claimed = atomic_compare_exchange_weak_explicit(
            &table_keys[slot], &expected, key,
            memory_order_relaxed, memory_order_relaxed);
        if (claimed) {
            table_vals[slot] = idx;
            return;
        }
        // Slot taken by someone else (should not collide in rebuild if keys are unique)
        slot = (slot + 1) & (table_cap - 1);
    }
}

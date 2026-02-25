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
        if (atomic_compare_exchange_weak_explicit(
            &table_keys[slot], &expected, key,
            memory_order_relaxed, memory_order_relaxed)) {
            table_vals[slot] = idx;
            return;
        }
        // Slot taken by someone else (should not collide in rebuild if keys are unique)
        slot = (slot + 1) & (table_cap - 1);
    }
}

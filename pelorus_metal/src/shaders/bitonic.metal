#include <metal_stdlib>
using namespace metal;

// ── Helpers ──────────────────────────────────────────────────────────────────

kernel void pack_keys(
    device const int*  keys    [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device ulong*      packed  [[buffer(2)]],
    constant uint&     n       [[buffer(3)]],
    uint               idx     [[thread_position_in_grid]])
{
    if (idx >= n) {
        // Pad with max value so they sort to the end
        if (idx < ((n + 255) & ~255)) { // Heuristic padding visibility
             // Actually, we should pad up to power of 2 in the sort dispatch, 
             // but writing here is safe if buffer is large enough.
        }
        return; 
    }
    
    // We treat the int32 key as uint32 for sorting. 
    // This preserves grouping of identical keys, which is all we need.
    uint k = as_type<uint>(keys[idx]);
    uint i = indices[idx];
    
    packed[idx] = ((ulong)k << 32) | i;
}

kernel void unpack_keys(
    device const ulong* packed  [[buffer(0)]],
    device int*         keys    [[buffer(1)]],
    device uint*        indices [[buffer(2)]],
    constant uint&      n       [[buffer(3)]],
    uint                idx     [[thread_position_in_grid]])
{
    if (idx >= n) return;
    
    ulong p = packed[idx];
    uint k_u = (uint)(p >> 32);
    uint i   = (uint)(p & 0xFFFFFFFF);
    
    keys[idx] = as_type<int>(k_u);
    indices[idx] = i;
}

kernel void init_pad(
    device ulong*  packed [[buffer(0)]],
    constant uint& n      [[buffer(1)]],
    constant uint& cap    [[buffer(2)]],
    uint           idx    [[thread_position_in_grid]])
{
    uint i = n + idx;
    if (i < cap) {
        packed[i] = 0xFFFFFFFFFFFFFFFFUL; // Max ulong
    }
}

// ── Bitonic Sort ─────────────────────────────────────────────────────────────

// Single step of bitonic sort
kernel void bitonic_step(
    device ulong*  data [[buffer(0)]],
    constant uint& j    [[buffer(1)]],
    constant uint& k    [[buffer(2)]],
    uint           idx  [[thread_position_in_grid]])
{
    uint i = idx;
    uint ixj = i ^ j;
    
    if (ixj > i) {
        if ((i & k) == 0) {
            // Ascending
            if (data[i] > data[ixj]) {
                ulong tmp = data[i];
                data[i] = data[ixj];
                data[ixj] = tmp;
            }
        } else {
            // Descending
            if (data[i] < data[ixj]) {
                ulong tmp = data[i];
                data[i] = data[ixj];
                data[ixj] = tmp;
            }
        }
    }
}

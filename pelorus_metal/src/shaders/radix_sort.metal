#include <metal_stdlib>
using namespace metal;

constant uint RADIX_BITS = 8u;
constant uint RADIX_SIZE = 1u << RADIX_BITS;
constant uint RADIX_MASK = RADIX_SIZE - 1u;
constant uint RADIX_BLOCK_SIZE = 256u;

kernel void radix_zero_hist(
    device uint* histogram [[buffer(0)]],
    constant uint& total_entries [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= total_entries) return;
    histogram[idx] = 0u;
}

kernel void radix_block_histogram(
    device const uint* keys [[buffer(0)]],
    device uint* block_histograms [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& shift [[buffer(3)]],
    constant uint& key_xor [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint hist[RADIX_SIZE];
    if (tid < RADIX_SIZE) {
        atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint digit = 0u;
    bool active = gid < n;
    if (active) {
        digit = ((keys[gid] ^ key_xor) >> shift) & RADIX_MASK;
        atomic_fetch_add_explicit(&hist[digit], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < RADIX_SIZE) {
        block_histograms[group_id * RADIX_SIZE + tid] =
            atomic_load_explicit(&hist[tid], memory_order_relaxed);
    }
}

kernel void radix_prefix_blocks(
    device const uint* block_histograms [[buffer(0)]],
    device uint* block_offsets [[buffer(1)]],
    device uint* bucket_counts [[buffer(2)]],
    constant uint& num_blocks [[buffer(3)]],
    uint digit [[thread_position_in_grid]])
{
    if (digit >= RADIX_SIZE) return;

    uint running = 0u;
    for (uint block = 0u; block < num_blocks; ++block) {
        uint idx = block * RADIX_SIZE + digit;
        block_offsets[idx] = running;
        running += block_histograms[idx];
    }
    bucket_counts[digit] = running;
}

kernel void radix_bucket_offsets(
    device const uint* bucket_counts [[buffer(0)]],
    device uint* bucket_offsets [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx != 0u) return;

    uint running = 0u;
    for (uint digit = 0u; digit < RADIX_SIZE; ++digit) {
        bucket_offsets[digit] = running;
        running += bucket_counts[digit];
    }
}

kernel void radix_scatter_pairs(
    device const uint* in_keys [[buffer(0)]],
    device const uint* in_values [[buffer(1)]],
    device uint* out_keys [[buffer(2)]],
    device uint* out_values [[buffer(3)]],
    device const uint* block_offsets [[buffer(4)]],
    device const uint* bucket_offsets [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& shift [[buffer(7)]],
    constant uint& key_xor [[buffer(8)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]])
{
    threadgroup uint digits[RADIX_BLOCK_SIZE];

    bool active = gid < n;
    uint digit = 0u;
    if (active) {
        digit = ((in_keys[gid] ^ key_xor) >> shift) & RADIX_MASK;
    }
    digits[tid] = digit;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!active) return;

    uint local_rank = 0u;
    for (uint i = 0u; i < tid; ++i) {
        local_rank += digits[i] == digit ? 1u : 0u;
    }

    uint dst = bucket_offsets[digit]
        + block_offsets[group_id * RADIX_SIZE + digit]
        + local_rank;
    out_keys[dst] = in_keys[gid];
    out_values[dst] = in_values[gid];
}

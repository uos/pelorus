#include <metal_stdlib>
using namespace metal;

struct DownsampleParams {
    float voxel_size;
    uint selected_count;
    uint point_skip;
    uint unique_count;
};

kernel void downsample_build_keys(
    device const Point* input [[buffer(0)]],
    device Point* selected [[buffer(1)]],
    device uint* keys [[buffer(2)]],
    device uint* indices [[buffer(3)]],
    constant DownsampleParams& params [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= params.selected_count) return;

    uint src_idx = idx * max(params.point_skip, 1u);
    Point p = input[src_idx];
    selected[idx] = p;

    int key = hash_grid(grid_pos(params.voxel_size, float3(p.x, p.y, p.z)));
    keys[idx] = uint(key);
    indices[idx] = idx;
}

kernel void downsample_rle(
    device const uint* sorted_keys [[buffer(0)]],
    device uint* run_starts [[buffer(1)]],
    device uint* run_ends [[buffer(2)]],
    device atomic_uint* counter [[buffer(3)]],
    constant DownsampleParams& params [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= params.selected_count) return;

    uint key = sorted_keys[idx];
    uint sort_key = key ^ 0x80000000u;
    bool is_start = (idx == 0u) || (sorted_keys[idx - 1u] != key);
    if (!is_start) return;

    uint left = idx;
    uint right = params.selected_count;
    while (left < right) {
        uint mid = left + (right - left) / 2u;
        if ((sorted_keys[mid] ^ 0x80000000u) <= sort_key) {
            left = mid + 1u;
        } else {
            right = mid;
        }
    }

    uint slot = atomic_fetch_add_explicit(&counter[0], 1u, memory_order_relaxed);
    run_starts[slot] = idx;
    run_ends[slot] = left;
}

kernel void downsample_medoids(
    device const Point* selected [[buffer(0)]],
    device const uint* sorted_indices [[buffer(1)]],
    device const uint* run_starts [[buffer(2)]],
    device const uint* run_ends [[buffer(3)]],
    device Point* medoids [[buffer(4)]],
    device const atomic_uint* counter [[buffer(5)]],
    constant DownsampleParams& params [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    uint unique_count = atomic_load_explicit(&counter[0], memory_order_relaxed);
    if (idx >= unique_count) return;

    uint start = run_starts[idx];
    uint end = run_ends[idx];
    if (end <= start) return;

    float3 sum = float3(0.0f);
    for (uint i = start; i < end; ++i) {
        Point p = selected[sorted_indices[i]];
        sum += float3(p.x, p.y, p.z);
    }

    float3 center = sum / float(end - start);
    Point best = selected[sorted_indices[start]];
    float3 best_delta = float3(best.x, best.y, best.z) - center;
    float best_dist = dot(best_delta, best_delta);

    for (uint i = start + 1u; i < end; ++i) {
        Point p = selected[sorted_indices[i]];
        float3 delta = float3(p.x, p.y, p.z) - center;
        float dist = dot(delta, delta);
        if (dist < best_dist) {
            best_dist = dist;
            best = p;
        }
    }

    medoids[idx] = best;
}

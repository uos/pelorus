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

// Rust computes the CPU medoid distance as separate f32 subtracts, multiplies
// and adds, associated left to right. `dot(delta, delta)` is a fused dot
// product, and plain `dx*dx + dy*dy + dz*dz` can still be contracted into
// FMAs, either of which lands a different last bit. That is enough to break an
// exact distance tie — a two-point voxel puts both points equidistant from
// their own midpoint — and pick the wrong medoid. `#pragma clang fp
// contract(off)` keeps the operation boundaries the CPU has; the CUDA backend
// needs the same thing and spells it with __fadd_rn/__fmul_rn intrinsics.
inline float medoid_sq_dist(Point p, float3 center) {
#pragma clang fp contract(off)
    float dx = p.x - center.x;
    float dy = p.y - center.y;
    float dz = p.z - center.z;
    return (dx * dx + dy * dy) + dz * dz;
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
    float best_dist = medoid_sq_dist(best, center);

    for (uint i = start + 1u; i < end; ++i) {
        Point p = selected[sorted_indices[i]];
        float dist = medoid_sq_dist(p, center);
        // Comparing distance alone leaves the winner of a tie decided by the
        // order the points happen to sit in this run, which is not the order
        // the CPU reference walks them in. `find_medoid_in_slice` in
        // pelorus_sync breaks ties lexicographically on
        // (dist, timestamp, x, y, z, intensity) — a total order, so its result
        // does not depend on traversal order at all. Mirror it exactly.
        bool better = dist < best_dist;
        if (dist == best_dist) {
            if (p.t != best.t)      better = p.t < best.t;
            else if (p.x != best.x) better = p.x < best.x;
            else if (p.y != best.y) better = p.y < best.y;
            else if (p.z != best.z) better = p.z < best.z;
            else                    better = p.i < best.i;
        }
        if (better) {
            best_dist = dist;
            best = p;
        }
    }

    medoids[idx] = best;
}

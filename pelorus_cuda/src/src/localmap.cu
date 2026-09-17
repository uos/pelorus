// CUDA voxel-map kernels.
//
// The map update, LRU ranking, and plane-selection stages mirror the Metal and
// Vulkan backends while using CUB for sorting and scans.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>

#include <cub/cub.cuh>

#include <lib.h>
#include <localmap.cuh>

static constexpr int      EMPTY_KEY   = INT_MIN;
static constexpr uint32_t INVALID_IDX = UINT32_MAX;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_e));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

__device__ __forceinline__ float3 quat_rotate(float4 q, float3 v) {
    // t = 2 * cross(q.xyz, v)
    float tx = 2.0f * (q.y * v.z - q.z * v.y);
    float ty = 2.0f * (q.z * v.x - q.x * v.z);
    float tz = 2.0f * (q.x * v.y - q.y * v.x);
    // result = v + q.w*t + cross(q.xyz, t)
    return make_float3(v.x + q.w * tx + (q.y * tz - q.z * ty),
                       v.y + q.w * ty + (q.z * tx - q.x * tz),
                       v.z + q.w * tz + (q.x * ty - q.y * tx));
}

__device__ __forceinline__ float3 iso_transform(float4 quat, float3 trans, float3 p) {
    float3 r = quat_rotate(quat, p);
    return make_float3(r.x + trans.x, r.y + trans.y, r.z + trans.z);
}

__device__ __forceinline__ int3 grid_pos(float resolution, float3 p) {
    float inv = 1.0f / resolution;
    return make_int3((int)floorf(p.x * inv),
                     (int)floorf(p.y * inv),
                     (int)floorf(p.z * inv));
}

__device__ __forceinline__ int hash_grid(int3 g) {
    // Match Rust/Metal wrapping i32 arithmetic exactly.
    uint32_t h = (uint32_t)g.x * 73856093u;
    h += (uint32_t)g.y * 471943u;
    h += (uint32_t)g.z * 83492791u;
    return (int32_t)h;
}

__device__ __forceinline__ uint32_t hash_fn(int key) {
    uint32_t h = (uint32_t)key;
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

// Flat row-major 3×3 Cholesky: A = L L^T
// Indices: L[r*3+c], lower-triangular entries L[0]=L00, L[3]=L10, L[4]=L11, etc.
__device__ bool cholesky_3x3(const float* __restrict__ A, float* __restrict__ L) {
    for (int i = 0; i < 9; i++) L[i] = 0.0f;

    float l00_sq = A[0];
    if (l00_sq <= 0.0f) return false;
    float l00 = sqrtf(l00_sq);
    L[0] = l00;

    L[3] = A[3] / l00;
    L[6] = A[6] / l00;

    float l11_sq = A[4] - L[3] * L[3];
    if (l11_sq <= 0.0f) return false;
    float l11 = sqrtf(l11_sq);
    L[4] = l11;

    L[7] = (A[7] - L[6] * L[3]) / l11;

    float l22_sq = A[8] - (L[6] * L[6] + L[7] * L[7]);
    if (l22_sq <= 0.0f) return false;
    L[8] = sqrtf(l22_sq);

    return true;
}

__device__ float3 cholesky_solve_3x3(const float* __restrict__ L, float3 b) {
    float3 y;
    y.x = b.x / L[0];
    y.y = (b.y - L[3] * y.x) / L[4];
    y.z = (b.z - L[6] * y.x - L[7] * y.y) / L[8];
    float3 x;
    x.z = y.z / L[8];
    x.y = (y.y - L[7] * x.z) / L[4];
    x.x = (y.x - L[3] * x.y - L[6] * x.z) / L[0];
    return x;
}

__device__ float mahalanobis_sq(float3 diff, const float* __restrict__ L) {
    float3 sol = cholesky_solve_3x3(L, diff);
    return diff.x * sol.x + diff.y * sol.y + diff.z * sol.z;
}

__device__ bool invert_3x3(const float* __restrict__ A, float* __restrict__ inv) {
    const float a = A[0], b = A[1], c = A[2];
    const float d = A[3], e = A[4], f = A[5];
    const float g = A[6], h = A[7], i = A[8];

    const float m0 = e*i - f*h;
    const float m1 = d*i - f*g;
    const float m2 = d*h - e*g;

    const float det = a*m0 - b*m1 + c*m2;
    if (!isfinite(det) || fabsf(det) < 1e-30f) return false;
    const float inv_det = 1.0f / det;

    inv[0] =  m0 * inv_det;
    inv[1] = -(b*i - c*h) * inv_det;
    inv[2] =  (b*f - c*e) * inv_det;
    inv[3] = -m1 * inv_det;
    inv[4] =  (a*i - c*g) * inv_det;
    inv[5] = -(a*f - c*d) * inv_det;
    inv[6] =  m2 * inv_det;
    inv[7] = -(a*h - b*g) * inv_det;
    inv[8] =  (a*e - b*d) * inv_det;
    return true;
}

__device__ float mahalanobis_sq_from_inv(float3 diff, const float* __restrict__ inv_cov) {
    float3 v = make_float3(
        inv_cov[0]*diff.x + inv_cov[1]*diff.y + inv_cov[2]*diff.z,
        inv_cov[3]*diff.x + inv_cov[4]*diff.y + inv_cov[5]*diff.z,
        inv_cov[6]*diff.x + inv_cov[7]*diff.y + inv_cov[8]*diff.z
    );
    return diff.x*v.x + diff.y*v.y + diff.z*v.z;
}

__device__ void regularize_cov(float* __restrict__ cov) {
    const float eps = 1e-6f;
    cov[0] += eps;
    cov[4] += eps;
    cov[8] += eps;
}

__device__ float3 symmetric_eigen_3x3(const float* __restrict__ A) {
    const float a00 = A[0], a01 = A[1], a02 = A[2];
    const float a11 = A[4], a12 = A[5];
    const float a22 = A[8];

    const float trA  = a00 + a11 + a22;
    const float c2   = (a00*a11 + a00*a22 + a11*a22) - (a01*a01 + a02*a02 + a12*a12);
    const float detA = a00*(a11*a22 - a12*a12)
                     - a01*(a01*a22 - a12*a02)
                     + a02*(a01*a12 - a11*a02);

    const float p  = -trA;
    const float q  = c2;
    const float r  = -detA;
    const float ad = (3.0f*q - p*p) / 3.0f;
    const float bd = (2.0f*p*p*p - 9.0f*p*q + 27.0f*r) / 27.0f;

    float angle = 0.0f;
    if (ad < -1e-7f) {
        float arg = (-bd * 0.5f) / sqrtf(-ad * ad * ad / 27.0f);
        arg   = fmaxf(-1.0f, fminf(1.0f, arg));
        angle = acosf(arg);
    }

    const float two_s = 2.0f * sqrtf(fmaxf(0.0f, -ad / 3.0f));
    const float p3    = p / 3.0f;
    const float pi2   = 6.28318530718f;
    float3 ev;
    ev.x = two_s * cosf(angle / 3.0f) - p3;
    ev.y = two_s * cosf((angle + pi2) / 3.0f) - p3;
    ev.z = two_s * cosf((angle + 2.0f*pi2) / 3.0f) - p3;
    return ev;
}

// returns INVALID_IDX if key not found
__device__ uint32_t hashmap_lookup(const int*      __restrict__ table_keys,
                                   const uint32_t* __restrict__ table_vals,
                                   uint32_t                     table_cap,
                                   int                          key) {
    if (key == EMPTY_KEY) return INVALID_IDX;
    uint32_t slot = hash_fn(key) & (table_cap - 1);
    for (;;) {
        int cur = table_keys[slot];
        if (cur == key)       return table_vals[slot];
        if (cur == EMPTY_KEY) return INVALID_IDX;
        slot = (slot + 1) & (table_cap - 1);
    }
}

// Neighbor offsets (ported from common.metal neighbor_cells)
__device__ uint32_t neighbor_cells(uint32_t nbtype, int3 g, int3* out) {
    if (nbtype == 6) {
        out[0] = g;
        out[1] = make_int3(g.x-1, g.y,   g.z  );
        out[2] = make_int3(g.x+1, g.y,   g.z  );
        out[3] = make_int3(g.x,   g.y+1, g.z  );
        out[4] = make_int3(g.x,   g.y-1, g.z  );
        out[5] = make_int3(g.x,   g.y,   g.z-1);
        out[6] = make_int3(g.x,   g.y,   g.z+1);
        return 7;
    } else if (nbtype == 18) {
        out[0]  = g;
        out[1]  = make_int3(g.x-1, g.y,   g.z  ); out[2]  = make_int3(g.x+1, g.y,   g.z  );
        out[3]  = make_int3(g.x,   g.y+1, g.z  ); out[4]  = make_int3(g.x,   g.y-1, g.z  );
        out[5]  = make_int3(g.x,   g.y,   g.z-1); out[6]  = make_int3(g.x,   g.y,   g.z+1);
        out[7]  = make_int3(g.x+1, g.y+1, g.z  ); out[8]  = make_int3(g.x-1, g.y+1, g.z  );
        out[9]  = make_int3(g.x+1, g.y-1, g.z  ); out[10] = make_int3(g.x-1, g.y-1, g.z  );
        out[11] = make_int3(g.x+1, g.y,   g.z+1); out[12] = make_int3(g.x-1, g.y,   g.z+1);
        out[13] = make_int3(g.x+1, g.y,   g.z-1); out[14] = make_int3(g.x-1, g.y,   g.z-1);
        out[15] = make_int3(g.x,   g.y+1, g.z+1); out[16] = make_int3(g.x,   g.y-1, g.z+1);
        out[17] = make_int3(g.x,   g.y+1, g.z-1); out[18] = make_int3(g.x,   g.y-1, g.z-1);
        return 19;
    } else { // 26 (default)
        out[0]  = g;
        out[1]  = make_int3(g.x-1, g.y,   g.z  ); out[2]  = make_int3(g.x+1, g.y,   g.z  );
        out[3]  = make_int3(g.x,   g.y+1, g.z  ); out[4]  = make_int3(g.x,   g.y-1, g.z  );
        out[5]  = make_int3(g.x,   g.y,   g.z-1); out[6]  = make_int3(g.x,   g.y,   g.z+1);
        out[7]  = make_int3(g.x+1, g.y+1, g.z  ); out[8]  = make_int3(g.x-1, g.y+1, g.z  );
        out[9]  = make_int3(g.x+1, g.y-1, g.z  ); out[10] = make_int3(g.x-1, g.y-1, g.z  );
        out[11] = make_int3(g.x+1, g.y,   g.z+1); out[12] = make_int3(g.x-1, g.y,   g.z+1);
        out[13] = make_int3(g.x+1, g.y,   g.z-1); out[14] = make_int3(g.x-1, g.y,   g.z-1);
        out[15] = make_int3(g.x,   g.y+1, g.z+1); out[16] = make_int3(g.x,   g.y-1, g.z+1);
        out[17] = make_int3(g.x,   g.y+1, g.z-1); out[18] = make_int3(g.x,   g.y-1, g.z-1);
        out[19] = make_int3(g.x+1, g.y+1, g.z+1); out[20] = make_int3(g.x-1, g.y+1, g.z+1);
        out[21] = make_int3(g.x+1, g.y-1, g.z+1); out[22] = make_int3(g.x+1, g.y+1, g.z-1);
        out[23] = make_int3(g.x-1, g.y-1, g.z+1); out[24] = make_int3(g.x-1, g.y+1, g.z-1);
        out[25] = make_int3(g.x+1, g.y-1, g.z-1); out[26] = make_int3(g.x-1, g.y-1, g.z-1);
        return 27;
    }
}

__device__ void pool_dist(uint32_t&           acc_count,
                          float*              acc_mean,
                          float*              acc_cov,
                          const GpuVoxelDist& other) {
    if (other.point_count == 0) return;

    if (acc_count == 0) {
        acc_count   = other.point_count;
        acc_mean[0] = other.mean[0];
        acc_mean[1] = other.mean[1];
        acc_mean[2] = other.mean[2];
        for (int i = 0; i < 9; i++) acc_cov[i] = other.cov[i];
        return;
    }

    float n1    = (float)acc_count;
    float n2    = (float)other.point_count;
    float total = n1 + n2;

    float nm0 = (n1*acc_mean[0] + n2*other.mean[0]) / total;
    float nm1 = (n1*acc_mean[1] + n2*other.mean[1]) / total;
    float nm2 = (n1*acc_mean[2] + n2*other.mean[2]) / total;

    float d[3] = { acc_mean[0] - other.mean[0],
                   acc_mean[1] - other.mean[1],
                   acc_mean[2] - other.mean[2] };
    float w = (n1 * n2) / total;

    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            acc_cov[r*3+c] = (n1*acc_cov[r*3+c] + n2*other.cov[r*3+c] + d[r]*d[c]*w) / total;

    acc_mean[0] = nm0; acc_mean[1] = nm1; acc_mean[2] = nm2;
    acc_count   = (uint32_t)total;
}

__global__ void hashmap_init_kernel(int* table_keys, uint32_t table_cap) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < table_cap) table_keys[idx] = EMPTY_KEY;
}

__global__ void rebuild_hashmap_keyed_kernel(int*        table_keys,
                                             uint32_t*   table_vals,
                                             const int*  dist_keys,
                                             uint32_t    valid_size,
                                             uint32_t    table_cap) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= valid_size) return;

    int key = dist_keys[idx];
    if (key == EMPTY_KEY) return;

    uint32_t slot = hash_fn(key) & (table_cap - 1);
    for (;;) {
        int prev = atomicCAS(&table_keys[slot], EMPTY_KEY, key);
        if (prev == EMPTY_KEY || prev == key) {
            table_vals[slot] = idx;
            return;
        }
        slot = (slot + 1) & (table_cap - 1);
    }
}


__global__ void classify_keys_kernel(const int*      unique_keys,
                                     const uint32_t* n_unique_ptr,
                                     const int*      table_keys,
                                     const uint32_t* table_vals,
                                     uint32_t        table_cap,
                                     uint32_t*       out_target_indices,
                                     uint32_t*       out_update_flags,
                                     uint32_t*       out_is_new) {
    uint32_t n_unique = *n_unique_ptr;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_unique) return;

    int key = unique_keys[idx];
    if (key == EMPTY_KEY) {
        out_target_indices[idx] = INVALID_IDX;
        out_update_flags[idx]   = 0;
        out_is_new[idx]         = 0;
        return;
    }

    uint32_t found_idx = hashmap_lookup(table_keys, table_vals, table_cap, key);
    if (found_idx != INVALID_IDX) {
        out_target_indices[idx] = found_idx;
        out_update_flags[idx]   = 1;
        out_is_new[idx]         = 0;
    } else {
        out_target_indices[idx] = INVALID_IDX; // resolved by assign_new_targets_kernel
        out_update_flags[idx]   = 0;
        out_is_new[idx]         = 1;
    }
}

__global__ void assign_new_targets_kernel(const int*      unique_keys,
                                          const uint32_t* n_unique_ptr,
                                          const uint32_t* is_new,
                                          const uint32_t* new_offset,
                                          const uint32_t* d_add_cursor_in,
                                          int*            dist_keys,
                                          const uint32_t* d_lru_mapping,
                                          uint32_t        dist_cap,
                                          uint32_t*       out_target_indices) {
    uint32_t n_unique = *n_unique_ptr;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_unique) return;
    if (is_new[idx] == 0u) return;

    int key = unique_keys[idx];
    uint32_t prev_add = *d_add_cursor_in;
    uint32_t cum      = new_offset[idx];
    uint64_t abs_pos  = (uint64_t)prev_add + cum;

    if (abs_pos < (uint64_t)dist_cap) {
        uint32_t slot = (uint32_t)abs_pos;
        dist_keys[slot] = key;
        out_target_indices[idx] = slot;
    } else {
        uint64_t lru_pos = abs_pos - dist_cap;
        if (lru_pos < (uint64_t)dist_cap) {
            uint32_t victim = d_lru_mapping[(uint32_t)lru_pos];
            dist_keys[victim] = key;
            out_target_indices[idx] = victim;
        } else {
            out_target_indices[idx] = INVALID_IDX;
        }
    }
}

__global__ void finalize_cursors_kernel(const uint32_t* n_unique_ptr,
                                        const uint32_t* is_new,
                                        const uint32_t* new_offset,
                                        uint32_t*       d_add_cursor,
                                        uint32_t*       d_lru_cursor,
                                        uint32_t        dist_cap) {
    uint32_t n_unique = *n_unique_ptr;
    uint32_t total_new = 0u;
    if (n_unique > 0u) {
        total_new = is_new[n_unique - 1u] + new_offset[n_unique - 1u];
    }

    uint32_t prev = *d_add_cursor;
    uint64_t end  = (uint64_t)prev + total_new;
    uint32_t new_add = (end > (uint64_t)dist_cap) ? dist_cap : (uint32_t)end;
    *d_add_cursor = new_add;

    uint32_t consumed_new = new_add - prev;
    uint32_t evictions    = total_new - consumed_new;
    if (evictions > dist_cap) evictions = dist_cap;
    *d_lru_cursor = evictions;
}

__global__ void initialize_lru_mapping_kernel(uint32_t* d_lru_mapping,
                                              uint32_t  dist_cap) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dist_cap) return;

    d_lru_mapping[idx] = idx;
}

struct LruIndexLess {
    const GpuVoxelDist* dist_buf;
    const uint32_t* d_add_cursor;
    uint32_t dist_cap;
    uint32_t scan_number;

    __device__ bool operator()(uint32_t lhs_idx, uint32_t rhs_idx) const {
        if (*d_add_cursor < dist_cap) return lhs_idx < rhs_idx;

        const GpuVoxelDist& lhs = dist_buf[lhs_idx];
        const GpuVoxelDist& rhs = dist_buf[rhs_idx];

        if (lhs.point_count == 0 && rhs.point_count == 0) return false;
        if (lhs.point_count == 0) return true;
        if (rhs.point_count == 0) return false;

        uint32_t lhs_age =
            (scan_number >= lhs.last_scan) ? (scan_number - lhs.last_scan) : 0u;
        uint32_t rhs_age =
            (scan_number >= rhs.last_scan) ? (scan_number - rhs.last_scan) : 0u;

        uint64_t lhs_ratio_numerator =
            (uint64_t)lhs_age * (uint64_t)rhs.point_count;
        uint64_t rhs_ratio_numerator =
            (uint64_t)rhs_age * (uint64_t)lhs.point_count;
        return lhs_ratio_numerator > rhs_ratio_numerator;
    }
};

__global__ void calculate_keys_kernel(const Point*      points,
                                      int*              out_keys,
                                      uint32_t*         out_indices,
                                      BuildVoxelsParams params) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.unique_key_count) return;

    float4 q = make_float4(params.pose_quat[0], params.pose_quat[1],
                           params.pose_quat[2], params.pose_quat[3]);
    float3 t = make_float3(params.pose_trans[0], params.pose_trans[1], params.pose_trans[2]);

    Point  p_raw    = points[idx];
    float3 p_local  = make_float3(p_raw.x, p_raw.y, p_raw.z);
    float3 p_global = iso_transform(q, t, p_local);

    int3 grid    = grid_pos(params.resolution, p_global);
    out_keys[idx]    = hash_grid(grid);
    out_indices[idx] = idx;
}

__global__ void reorder_points_kernel(const Point*      src,
                                      Point*            dst,
                                      const uint32_t*   sorted_indices,
                                      BuildVoxelsParams params) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.unique_key_count) return;

    float4 q = make_float4(params.pose_quat[0], params.pose_quat[1],
                           params.pose_quat[2], params.pose_quat[3]);
    float3 t = make_float3(params.pose_trans[0], params.pose_trans[1], params.pose_trans[2]);

    Point  p_raw    = src[sorted_indices[idx]];
    float3 p_local  = make_float3(p_raw.x, p_raw.y, p_raw.z);
    float3 p_global = iso_transform(q, t, p_local);

    Point p_out = p_raw;
    p_out.x = p_global.x;
    p_out.y = p_global.y;
    p_out.z = p_global.z;
    dst[idx] = p_out;
}

__global__ void build_voxels_kernel(const Point*      cloud,
                                    const uint32_t*   run_starts,
                                    const uint32_t*   run_counts,
                                    const int*        unique_keys,
                                    const uint32_t*   target_indices,
                                    const uint32_t*   update_flags,
                                    GpuVoxelDist*     dists,
                                    const uint32_t*   n_unique_ptr,
                                    BuildVoxelsParams params) {
    uint32_t n_unique = *n_unique_ptr;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_unique) return;

    int      voxel_key = unique_keys[idx];
    uint32_t start     = run_starts[idx];
    uint32_t count     = run_counts[idx];
    uint32_t end       = start + count;

    if (count == 0 || voxel_key == EMPTY_KEY || target_indices[idx] == INVALID_IDX) return;

    // ── 1. Compute scan mean/covariance (two-pass, points already in global frame) ──
    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    for (uint32_t i = start; i < end; i++) {
        sx += cloud[i].x; sy += cloud[i].y; sz += cloud[i].z;
    }
    float fn = (float)count;
    float3 scan_mean = make_float3(sx/fn, sy/fn, sz/fn);

    float s_xx=0,s_xy=0,s_xz=0,s_yy=0,s_yz=0,s_zz=0;
    for (uint32_t i = start; i < end; i++) {
        float px = cloud[i].x - scan_mean.x;
        float py = cloud[i].y - scan_mean.y;
        float pz = cloud[i].z - scan_mean.z;
        s_xx += px*px; s_xy += px*py; s_xz += px*pz;
        s_yy += py*py; s_yz += py*pz; s_zz += pz*pz;
    }
    float scan_cov[9];
    scan_cov[0] = s_xx/fn; scan_cov[1] = s_xy/fn; scan_cov[2] = s_xz/fn;
    scan_cov[3] = scan_cov[1];
    scan_cov[4] = s_yy/fn; scan_cov[5] = s_yz/fn;
    scan_cov[6] = scan_cov[2];
    scan_cov[7] = scan_cov[5];
    scan_cov[8] = s_zz/fn;

    uint32_t buf_idx  = target_indices[idx];
    bool     is_update = (update_flags[idx] != 0u);

    if (is_update) {
        // ── Alpha-blend fusion ──
        GpuVoxelDist& d = dists[buf_idx];
        d.last_scan = params.scan_number;

        float3 d_mean    = make_float3(d.mean[0], d.mean[1], d.mean[2]);
        float3 mean_diff = make_float3(d_mean.x - scan_mean.x,
                                       d_mean.y - scan_mean.y,
                                       d_mean.z - scan_mean.z);
        float  fn_d           = (float)d.point_count;
        float  maturity_decay = fmaxf(1.0f, logf(fn_d));
        float  eff_sens       = params.sensitivity / maturity_decay;

        float reg_cov[9];
        for (int i = 0; i < 9; i++) reg_cov[i] = d.cov[i];
        regularize_cov(reg_cov);

        float L_reg[9] = {};
        float m_sq;
        if (cholesky_3x3(reg_cov, L_reg)) {
            m_sq = mahalanobis_sq(mean_diff, L_reg);
        } else {
            m_sq = mean_diff.x*mean_diff.x + mean_diff.y*mean_diff.y + mean_diff.z*mean_diff.z;
        }

        float alpha = expf(-m_sq / (2.0f * eff_sens));

        if (alpha >= 1e-5f) {
            float oma = 1.0f - alpha;
            d.mean[0] = d_mean.x*oma + scan_mean.x*alpha;
            d.mean[1] = d_mean.y*oma + scan_mean.y*alpha;
            d.mean[2] = d_mean.z*oma + scan_mean.z*alpha;

            float dv[3] = { mean_diff.x, mean_diff.y, mean_diff.z };
            float aw    = alpha * oma;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    d.cov[r*3+c] = d.cov[r*3+c]*oma + scan_cov[r*3+c]*alpha
                                   + dv[r]*dv[c]*aw;

            float next_L[9] = {};
            if (!cholesky_3x3(d.cov, next_L)) {
                float3 ev = symmetric_eigen_3x3(d.cov);
                next_L[0] = sqrtf(fmaxf(0.0f, ev.x));
                next_L[4] = sqrtf(fmaxf(0.0f, ev.y));
                next_L[8] = sqrtf(fmaxf(0.0f, ev.z));
            }
            for (int i = 0; i < 9; i++) d.transform[i] = next_L[i];

            d.point_count += count;
        }
    } else {
        // ── Insert: slot pre-selected by CPU LRU ──
        GpuVoxelDist& d = dists[buf_idx];
        d.mean[0] = scan_mean.x; d.mean[1] = scan_mean.y; d.mean[2] = scan_mean.z;
        for (int i = 0; i < 9; i++) d.cov[i] = scan_cov[i];
        d.point_count = count;
        d.last_scan   = params.scan_number;

        float scan_L[9] = {};
        if (!cholesky_3x3(scan_cov, scan_L)) {
            float3 ev = symmetric_eigen_3x3(scan_cov);
            scan_L[0] = sqrtf(fmaxf(0.0f, ev.x));
            scan_L[4] = sqrtf(fmaxf(0.0f, ev.y));
            scan_L[8] = sqrtf(fmaxf(0.0f, ev.z));
        }
        for (int i = 0; i < 9; i++) d.transform[i] = scan_L[i];
        d.lock_flag = 0;
    }
}

// planes.metal: plane_krnl
__global__ void plane_kernel(const Point*        cloud,
                             const int*          table_keys,
                             const uint32_t*     table_vals,
                             const GpuVoxelDist* dists,
                             uint8_t*            valid_flags,
                             PointNormal*        out_buff,
                             PlanesParams        params) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.cloud_size) return;

    // Sentinel: zero this slot. Any early-exit below leaves it as invalid so
    // CPU-side compaction (which walks slots in order) skips it. This keeps
    // GPU plane output deterministic and aligned with cloud-input order.
    {
        PointNormal inv;
        inv.x = 0.0f; inv.y = 0.0f; inv.z = 0.0f;
        inv.intensity = 0.0f;
        inv.normal_x = 0.0f; inv.normal_y = 0.0f; inv.normal_z = 0.0f;
        out_buff[idx] = inv;
        valid_flags[idx] = 0;
    }

    float4 ext_q = make_float4(params.lidar_to_imu_quat[0], params.lidar_to_imu_quat[1],
                                params.lidar_to_imu_quat[2], params.lidar_to_imu_quat[3]);
    float3 ext_t = make_float3(params.lidar_to_imu_pos[0], params.lidar_to_imu_pos[1],
                                params.lidar_to_imu_pos[2]);
    float4 q     = make_float4(params.pose_quat[0], params.pose_quat[1],
                                params.pose_quat[2], params.pose_quat[3]);
    float3 t     = make_float3(params.pose_pos[0], params.pose_pos[1], params.pose_pos[2]);

    Point  p_raw    = cloud[idx];
    float3 p_local  = make_float3(p_raw.x, p_raw.y, p_raw.z);
    float3 p_imu    = iso_transform(ext_q, ext_t, p_local);
    float3 p_global = iso_transform(q, t, p_imu);

    // Collect neighbor voxels and pool their distributions
    int3     grid = grid_pos(params.resolution, p_global);
    int3     neighbors[27];
    uint32_t nb_count = neighbor_cells(params.nbtype, grid, neighbors);

    uint32_t merged_count  = 0;
    float    merged_mean[3] = {0,0,0};
    float    merged_cov[9]  = {0,0,0,0,0,0,0,0,0};

    for (uint32_t ni = 0; ni < nb_count; ni++) {
        int      nb_key  = hash_grid(neighbors[ni]);
        uint32_t buf_idx = hashmap_lookup(table_keys, table_vals, params.table_cap, nb_key);
        if (buf_idx == INVALID_IDX) continue;
        const GpuVoxelDist& nd = dists[buf_idx];
        if (nd.point_count == 0) continue;
        pool_dist(merged_count, merged_mean, merged_cov, nd);
    }

    if (merged_count < 3) return;

    // ── Plane fit (planes.metal plane_fit) ──
    float3 ev = symmetric_eigen_3x3(merged_cov);

    // Sort eigenvalues ascending
    float v0=ev.x, v1=ev.y, v2=ev.z;
    if (v0 > v1) { float tmp=v0; v0=v1; v1=tmp; }
    if (v1 > v2) { float tmp=v1; v1=v2; v2=tmp; }
    if (v0 > v1) { float tmp=v0; v0=v1; v1=tmp; }

    if (!isfinite(v0) || !isfinite(v1) || !isfinite(v2)) return;
    if (v2 < 1e-9f) return;
    if (v0 / v2 > params.sphericity_threshold) return;

    // Null space of (cov - lambda_0 * I) via cross-product of rows
    float M[9];
    for (int i = 0; i < 9; i++) M[i] = merged_cov[i];
    M[0] -= v0; M[4] -= v0; M[8] -= v0;

    float3 row0 = make_float3(M[0], M[1], M[2]);
    float3 row1 = make_float3(M[3], M[4], M[5]);
    float3 row2 = make_float3(M[6], M[7], M[8]);

    float3 v01 = make_float3(row0.y*row1.z - row0.z*row1.y,
                             row0.z*row1.x - row0.x*row1.z,
                             row0.x*row1.y - row0.y*row1.x);
    float3 v12 = make_float3(row1.y*row2.z - row1.z*row2.y,
                             row1.z*row2.x - row1.x*row2.z,
                             row1.x*row2.y - row1.y*row2.x);
    float3 v20 = make_float3(row2.y*row0.z - row2.z*row0.y,
                             row2.z*row0.x - row2.x*row0.z,
                             row2.x*row0.y - row2.y*row0.x);

    float m01 = v01.x*v01.x + v01.y*v01.y + v01.z*v01.z;
    float m12 = v12.x*v12.x + v12.y*v12.y + v12.z*v12.z;
    float m20 = v20.x*v20.x + v20.y*v20.y + v20.z*v20.z;

    float3 normal;
    if      (m01 > m12 && m01 > m20) normal = v01;
    else if (m12 > m20)               normal = v12;
    else                              normal = v20;

    float mag = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
    if (!isfinite(mag) || mag < 1e-9f) return;
    normal.x /= mag; normal.y /= mag; normal.z /= mag;

    float3 mean_p = make_float3(merged_mean[0], merged_mean[1], merged_mean[2]);
    float  d_val  = -(normal.x*mean_p.x + normal.y*mean_p.y + normal.z*mean_p.z);

    // Score filter (same formula as Metal/CPU)
    float pd2   = normal.x*p_global.x + normal.y*p_global.y + normal.z*p_global.z + d_val;
    float l2_sq = p_local.x*p_local.x + p_local.y*p_local.y + p_local.z*p_local.z;
    float s     = 1.0f - 0.9f * fabsf(pd2) / sqrtf(sqrtf(fmaxf(l2_sq, 1e-9f)));
    if (!isfinite(s) || s <= 0.9f) return;

    PointNormal pn;
    pn.x        = p_local.x;
    pn.y        = p_local.y;
    pn.z        = p_local.z;
    pn.intensity = d_val;
    pn.normal_x  = normal.x;
    pn.normal_y  = normal.y;
    pn.normal_z  = normal.z;
    out_buff[idx] = pn;
    valid_flags[idx] = 1;
}

// ── CPU state ─────────────────────────────────────────────────────────────────

struct CudaMapState {
    // GPU-only (device memory)
    int*      table_keys;   // size: table_cap
    uint32_t* table_vals;   // size: table_cap

    // Device buffers. Do not use cudaMallocManaged here: on the Jetson CUDA
    // driver a host load from a newly allocated managed page can fault even
    // though cudaMallocManaged reported success. Host observations of these
    // buffers are made explicitly with cudaMemcpy below.
    GpuVoxelDist* dist_buf;     // size: dist_cap
    int*          dist_keys;    // size: dist_cap
    uint32_t*     counter;
    uint32_t*     d_add_cursor;
    uint32_t*     d_lru_cursor;
    uint32_t*     d_num_unique;

    // Scratch (device memory)
    int*      scratch_keys;
    int*      scratch_keys_alt;
    uint32_t* scratch_indices;
    uint32_t* scratch_indices_alt;
    uint32_t* scratch_run_starts;
    uint32_t* scratch_run_counts;
    int*      scratch_unique_keys;
    uint32_t* scratch_target_indices;
    uint32_t* scratch_update_flags;
    uint32_t* scratch_is_new;
    uint32_t* scratch_new_offset;
    Point*    scratch_sorted_points;
    uint8_t*  scratch_plane_flags;
    PointNormal* scratch_compacted_planes;
    uint32_t* d_selected_planes;

    // LRU on GPU (device memory)
    uint32_t* d_lru_mapping;

    // CUB temp storage
    void*     d_cub_temp_storage;
    size_t    cub_temp_storage_bytes;

    // Config
    uint32_t dist_cap;
    uint32_t table_cap;
    float    resolution;
    float    sensitivity;
    float    min_dist;
    uint32_t scan_number;
    uint32_t nbtype;

    size_t scratch_cap;  // current allocated scratch capacity (max n_points)
    bool   trace_map_debug;

    CudaMapState()
        : table_keys(nullptr), table_vals(nullptr),
          dist_buf(nullptr), dist_keys(nullptr), counter(nullptr),
          d_add_cursor(nullptr), d_lru_cursor(nullptr), d_num_unique(nullptr),
          scratch_keys(nullptr), scratch_keys_alt(nullptr),
          scratch_indices(nullptr), scratch_indices_alt(nullptr),
          scratch_run_starts(nullptr), scratch_run_counts(nullptr),
          scratch_unique_keys(nullptr), scratch_target_indices(nullptr),
          scratch_update_flags(nullptr), scratch_is_new(nullptr),
          scratch_new_offset(nullptr), scratch_sorted_points(nullptr),
          scratch_plane_flags(nullptr), scratch_compacted_planes(nullptr),
          d_selected_planes(nullptr),
          d_lru_mapping(nullptr),
          d_cub_temp_storage(nullptr), cub_temp_storage_bytes(0),
          dist_cap(0), table_cap(0), resolution(0), sensitivity(0),
          min_dist(0), scan_number(0), nbtype(26),
          scratch_cap(0), trace_map_debug(false) {}

    ~CudaMapState() {
        if (table_keys)             cudaFree(table_keys);
        if (table_vals)             cudaFree(table_vals);
        if (dist_buf)               cudaFree(dist_buf);
        if (dist_keys)              cudaFree(dist_keys);
        if (counter)                cudaFree(counter);
        if (d_add_cursor)           cudaFree(d_add_cursor);
        if (d_lru_cursor)           cudaFree(d_lru_cursor);
        if (d_num_unique)           cudaFree(d_num_unique);
        if (scratch_keys)           cudaFree(scratch_keys);
        if (scratch_keys_alt)       cudaFree(scratch_keys_alt);
        if (scratch_indices)        cudaFree(scratch_indices);
        if (scratch_indices_alt)    cudaFree(scratch_indices_alt);
        if (scratch_run_starts)     cudaFree(scratch_run_starts);
        if (scratch_run_counts)     cudaFree(scratch_run_counts);
        if (scratch_unique_keys)    cudaFree(scratch_unique_keys);
        if (scratch_target_indices) cudaFree(scratch_target_indices);
        if (scratch_update_flags)   cudaFree(scratch_update_flags);
        if (scratch_is_new)         cudaFree(scratch_is_new);
        if (scratch_new_offset)     cudaFree(scratch_new_offset);
        if (scratch_sorted_points)  cudaFree(scratch_sorted_points);
        if (scratch_plane_flags)    cudaFree(scratch_plane_flags);
        if (scratch_compacted_planes) cudaFree(scratch_compacted_planes);
        if (d_selected_planes)      cudaFree(d_selected_planes);
        if (d_lru_mapping)          cudaFree(d_lru_mapping);
        if (d_cub_temp_storage)     cudaFree(d_cub_temp_storage);
    }
};

static CudaMapState* g_state = nullptr;

// ── CPU helpers ───────────────────────────────────────────────────────────────

static inline int grid_dim(uint32_t n) { return (n + 255) / 256; }

static void alloc_scratch(CudaMapState* s, size_t new_cap) {
    if (s->scratch_keys)           CUDA_CHECK(cudaFree(s->scratch_keys));
    if (s->scratch_keys_alt)       CUDA_CHECK(cudaFree(s->scratch_keys_alt));
    if (s->scratch_indices)        CUDA_CHECK(cudaFree(s->scratch_indices));
    if (s->scratch_indices_alt)    CUDA_CHECK(cudaFree(s->scratch_indices_alt));
    if (s->scratch_run_starts)     CUDA_CHECK(cudaFree(s->scratch_run_starts));
    if (s->scratch_run_counts)     CUDA_CHECK(cudaFree(s->scratch_run_counts));
    if (s->scratch_unique_keys)    CUDA_CHECK(cudaFree(s->scratch_unique_keys));
    if (s->scratch_target_indices) CUDA_CHECK(cudaFree(s->scratch_target_indices));
    if (s->scratch_update_flags)   CUDA_CHECK(cudaFree(s->scratch_update_flags));
    if (s->scratch_is_new)         CUDA_CHECK(cudaFree(s->scratch_is_new));
    if (s->scratch_new_offset)     CUDA_CHECK(cudaFree(s->scratch_new_offset));
    if (s->scratch_sorted_points)  CUDA_CHECK(cudaFree(s->scratch_sorted_points));
    if (s->scratch_plane_flags)    CUDA_CHECK(cudaFree(s->scratch_plane_flags));
    if (s->scratch_compacted_planes) CUDA_CHECK(cudaFree(s->scratch_compacted_planes));

    CUDA_CHECK(cudaMalloc(&s->scratch_keys,           new_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s->scratch_keys_alt,       new_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s->scratch_indices,        new_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s->scratch_indices_alt,    new_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s->scratch_run_starts,     new_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s->scratch_run_counts,     new_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s->scratch_unique_keys,    new_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s->scratch_target_indices, new_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s->scratch_update_flags,   new_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s->scratch_is_new,         new_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s->scratch_new_offset,     new_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s->scratch_sorted_points,  new_cap * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&s->scratch_plane_flags,    new_cap * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&s->scratch_compacted_planes, new_cap * sizeof(PointNormal)));

    s->scratch_cap = new_cap;
}

static void ensure_cub_temp_storage(CudaMapState* s, size_t required_bytes) {
    if (s->cub_temp_storage_bytes < required_bytes) {
        if (s->d_cub_temp_storage) CUDA_CHECK(cudaFree(s->d_cub_temp_storage));
        CUDA_CHECK(cudaMalloc(&s->d_cub_temp_storage, required_bytes));
        s->cub_temp_storage_bytes = required_bytes;
    }
}

void init_map_impl(cudaStream_t stream, uint32_t capacity, float voxel_size,
                   float sensitivity, float min_dist, uint32_t nbtype) {
    assert(g_state == nullptr && "init_map_impl called twice");

    uint32_t dist_cap  = capacity;      // caller ensures power-of-two
    uint32_t table_cap = dist_cap * 2;

    g_state = new CudaMapState();
    CudaMapState* s = g_state;

    // GPU-only buffers
    CUDA_CHECK(cudaMalloc(&s->table_keys, table_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s->table_vals, table_cap * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(&s->dist_buf,     dist_cap * sizeof(GpuVoxelDist)));
    CUDA_CHECK(cudaMalloc(&s->dist_keys,    dist_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s->counter,      sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s->d_add_cursor, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s->d_lru_cursor, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s->d_num_unique, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s->d_selected_planes, sizeof(uint32_t)));

    // LRU buffers
    CUDA_CHECK(cudaMalloc(&s->d_lru_mapping,     dist_cap * sizeof(uint32_t)));
    // Zero dist_buf. Fill dist_keys with EMPTY_KEY.
    CUDA_CHECK(cudaMemsetAsync(s->dist_buf, 0, dist_cap * sizeof(GpuVoxelDist), stream));
    std::vector<int> h_keys(dist_cap, EMPTY_KEY);
    CUDA_CHECK(cudaMemcpyAsync(s->dist_keys, h_keys.data(), dist_cap * sizeof(int), cudaMemcpyHostToDevice, stream));

    // Initial LRU mapping: 0, 1, 2, ...
    std::vector<uint32_t> h_lru(dist_cap);
    for (uint32_t i = 0; i < dist_cap; i++) h_lru[i] = i;
    CUDA_CHECK(cudaMemcpyAsync(s->d_lru_mapping, h_lru.data(), dist_cap * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMemsetAsync(s->counter, 0, sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(s->d_add_cursor, 0, sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(s->d_lru_cursor, 0, sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(s->d_num_unique, 0, sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(s->d_selected_planes, 0, sizeof(uint32_t), stream));

    // Scratch buffers start at minimum size and grow on first add_to_map_impl.
    alloc_scratch(s, 1);

    s->dist_cap     = dist_cap;
    s->table_cap    = table_cap;
    s->resolution   = voxel_size;
    s->sensitivity  = sensitivity;
    s->min_dist     = min_dist;
    s->scan_number  = 0;
    s->nbtype       = nbtype;
    s->trace_map_debug = (std::getenv("PELORUS_CUDA_TRACE_MAP") != nullptr);

    // Init hashmap
    hashmap_init_kernel<<<grid_dim(table_cap), 256, 0, stream>>>(s->table_keys, table_cap);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void add_to_map_impl(cudaStream_t stream, Point* points, uint32_t n_points, Pose pose, uint32_t scan_id) {
    if (multiresolution_active_impl()) { add_multiresolution_impl(stream, points, n_points, pose, scan_id); return; }
    CudaMapState* s = g_state;
    assert(s != nullptr);

    if (n_points == 0) return;

    s->scan_number = scan_id;
    uint32_t scan_number = s->scan_number;

    // Scratch must hold at least n_points elements
    if (s->scratch_cap < (size_t)n_points) {
        alloc_scratch(s, n_points + n_points / 2);
    }

    // ── 1. Calculate temp storage requirements for CUB ──
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
        s->scratch_keys, s->scratch_keys_alt, s->scratch_indices, s->scratch_indices_alt, n_points);
    size_t sort_bytes = temp_storage_bytes;

    temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, temp_storage_bytes,
        s->scratch_keys_alt, s->scratch_unique_keys, s->scratch_run_counts, s->d_num_unique, n_points);
    size_t encode_bytes = temp_storage_bytes;

    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
        s->scratch_run_counts, s->scratch_run_starts, n_points);
    size_t scan_bytes = temp_storage_bytes;

    temp_storage_bytes = 0;
    LruIndexLess lru_less{
        s->dist_buf, s->d_add_cursor, s->dist_cap, scan_number};
    cub::DeviceMergeSort::SortKeys(nullptr, temp_storage_bytes,
        s->d_lru_mapping, s->dist_cap, lru_less);
    size_t lru_bytes = temp_storage_bytes;

    size_t max_cub_bytes = std::max({sort_bytes, encode_bytes, scan_bytes, lru_bytes});
    ensure_cub_temp_storage(s, max_cub_bytes);

    BuildVoxelsParams params;
    params.pose_quat[0]  = (float)pose.orientation.x;
    params.pose_quat[1]  = (float)pose.orientation.y;
    params.pose_quat[2]  = (float)pose.orientation.z;
    params.pose_quat[3]  = (float)pose.orientation.w;
    params.pose_trans[0] = (float)pose.position.x;
    params.pose_trans[1] = (float)pose.position.y;
    params.pose_trans[2] = (float)pose.position.z;
    params._pad0          = 0;
    params.resolution     = s->resolution;
    params.scan_number    = scan_number;
    params.sensitivity    = s->sensitivity;
    params.min_dist       = s->min_dist;
    params.unique_key_count = n_points;  // cloud_size
    params.table_cap      = s->table_cap;
    params.dist_cap       = s->dist_cap;
    params._pad1          = 0;

    // Calculate_keys
    calculate_keys_kernel<<<grid_dim(n_points), 256, 0, stream>>>(
        points, s->scratch_keys, s->scratch_indices, params);

    // Sort (key, index) pairs
    temp_storage_bytes = s->cub_temp_storage_bytes;
    cub::DeviceRadixSort::SortPairs(s->d_cub_temp_storage, temp_storage_bytes,
        s->scratch_keys, s->scratch_keys_alt, s->scratch_indices, s->scratch_indices_alt, n_points, 0, 32, stream);

    // Run-length encode
    CUDA_CHECK(cudaMemsetAsync(s->scratch_run_counts, 0, n_points * sizeof(uint32_t), stream));
    temp_storage_bytes = s->cub_temp_storage_bytes;
    cub::DeviceRunLengthEncode::Encode(s->d_cub_temp_storage, temp_storage_bytes,
        s->scratch_keys_alt, s->scratch_unique_keys, s->scratch_run_counts, s->d_num_unique, n_points, stream);

    // Exclusive scan for run starts
    temp_storage_bytes = s->cub_temp_storage_bytes;
    cub::DeviceScan::ExclusiveSum(s->d_cub_temp_storage, temp_storage_bytes,
        s->scratch_run_counts, s->scratch_run_starts, n_points, stream);

    // Assign target indices (classify → scan → assign → finalize)
    classify_keys_kernel<<<grid_dim(n_points), 256, 0, stream>>>(
        s->scratch_unique_keys, s->d_num_unique, s->table_keys, s->table_vals, s->table_cap,
        s->scratch_target_indices, s->scratch_update_flags, s->scratch_is_new);

    temp_storage_bytes = s->cub_temp_storage_bytes;
    cub::DeviceScan::ExclusiveSum(s->d_cub_temp_storage, temp_storage_bytes,
        s->scratch_is_new, s->scratch_new_offset, n_points, stream);

    assign_new_targets_kernel<<<grid_dim(n_points), 256, 0, stream>>>(
        s->scratch_unique_keys, s->d_num_unique, s->scratch_is_new, s->scratch_new_offset,
        s->d_add_cursor, s->dist_keys, s->d_lru_mapping, s->dist_cap,
        s->scratch_target_indices);

    finalize_cursors_kernel<<<1, 1, 0, stream>>>(
        s->d_num_unique, s->scratch_is_new, s->scratch_new_offset,
        s->d_add_cursor, s->d_lru_cursor, s->dist_cap);

    // Reorder_points
    reorder_points_kernel<<<grid_dim(n_points), 256, 0, stream>>>(
        points, s->scratch_sorted_points, s->scratch_indices_alt, params);

    // Build_voxels
    build_voxels_kernel<<<grid_dim(n_points), 256, 0, stream>>>(
        s->scratch_sorted_points, s->scratch_run_starts, s->scratch_run_counts,
        s->scratch_unique_keys, s->scratch_target_indices, s->scratch_update_flags,
        s->dist_buf, s->d_num_unique, params);

    // Rebuild hashmap
    hashmap_init_kernel<<<grid_dim(s->table_cap), 256, 0, stream>>>(
        s->table_keys, s->table_cap);
    rebuild_hashmap_keyed_kernel<<<grid_dim(s->dist_cap), 256, 0, stream>>>(
        s->table_keys, s->table_vals, s->dist_keys, s->dist_cap, s->table_cap);

    // LRU ranking
    initialize_lru_mapping_kernel<<<grid_dim(s->dist_cap), 256, 0, stream>>>(
        s->d_lru_mapping, s->dist_cap);
    temp_storage_bytes = s->cub_temp_storage_bytes;
    cub::DeviceMergeSort::SortKeys(s->d_cub_temp_storage, temp_storage_bytes,
        s->d_lru_mapping, s->dist_cap, lru_less, stream);

    if (s->trace_map_debug && ((scan_number % 100u) == 0u)) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        uint32_t unique = 0;
        uint32_t add_cursor = 0;
        uint32_t lru_cursor = 0;
        CUDA_CHECK(cudaMemcpy(&unique, s->d_num_unique, sizeof(unique), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&add_cursor, s->d_add_cursor, sizeof(add_cursor), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&lru_cursor, s->d_lru_cursor, sizeof(lru_cursor), cudaMemcpyDeviceToHost));
        fprintf(stderr,
                "[CUDA-MAP] scan=%u points=%u unique=%u add_cursor=%u lru_cursor=%u\n",
                scan_number, n_points, unique, add_cursor, lru_cursor);
    }
}

uint32_t planes_impl(cudaStream_t stream, Point* cloud, uint32_t cloud_size,
                     Pose lidar_to_imu, Pose pose, PointNormal* out_buff) {
    if (multiresolution_active_impl()) return planes_multiresolution_impl(stream, cloud, cloud_size, lidar_to_imu, pose, out_buff);
    CudaMapState* s = g_state;
    assert(s != nullptr);

    if (cloud_size == 0) return 0;

    if (s->scratch_cap < (size_t)cloud_size) {
        alloc_scratch(s, cloud_size + cloud_size / 2);
    }

    PlanesParams params;
    params.lidar_to_imu_quat[0] = (float)lidar_to_imu.orientation.x;
    params.lidar_to_imu_quat[1] = (float)lidar_to_imu.orientation.y;
    params.lidar_to_imu_quat[2] = (float)lidar_to_imu.orientation.z;
    params.lidar_to_imu_quat[3] = (float)lidar_to_imu.orientation.w;
    params.lidar_to_imu_pos[0]  = (float)lidar_to_imu.position.x;
    params.lidar_to_imu_pos[1]  = (float)lidar_to_imu.position.y;
    params.lidar_to_imu_pos[2]  = (float)lidar_to_imu.position.z;
    params._pad0 = 0;
    params.pose_quat[0] = (float)pose.orientation.x;
    params.pose_quat[1] = (float)pose.orientation.y;
    params.pose_quat[2] = (float)pose.orientation.z;
    params.pose_quat[3] = (float)pose.orientation.w;
    params.pose_pos[0]  = (float)pose.position.x;
    params.pose_pos[1]  = (float)pose.position.y;
    params.pose_pos[2]  = (float)pose.position.z;
    params._pad1 = 0;
    params.cloud_size           = cloud_size;
    params.table_cap            = s->table_cap;
    params.resolution           = s->resolution;
    params.sphericity_threshold = 0.5f;
    params.nbtype               = s->nbtype;

    plane_kernel<<<grid_dim(cloud_size), 256, 0, stream>>>(
        cloud, s->table_keys, s->table_vals, s->dist_buf,
        s->scratch_plane_flags, out_buff, params);

    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes,
        out_buff, s->scratch_plane_flags, s->scratch_compacted_planes,
        s->d_selected_planes, cloud_size, stream);
    ensure_cub_temp_storage(s, temp_storage_bytes);

    cub::DeviceSelect::Flagged(s->d_cub_temp_storage, temp_storage_bytes,
        out_buff, s->scratch_plane_flags, s->scratch_compacted_planes,
        s->d_selected_planes, cloud_size, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    uint32_t selected = 0;
    CUDA_CHECK(cudaMemcpy(&selected, s->d_selected_planes, sizeof(selected),
                          cudaMemcpyDeviceToHost));
    if (selected > 0) {
        CUDA_CHECK(cudaMemcpyAsync(out_buff, s->scratch_compacted_planes,
                                   selected * sizeof(PointNormal),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    return selected;
}

void get_map_impl(Point* points, uint32_t* size) {
    if (multiresolution_active_impl()) { get_multiresolution_impl(points, size); return; }
    if (!g_state) { *size = 0; return; }
    CudaMapState* s = g_state;

    uint32_t add_cursor = 0;
    CUDA_CHECK(cudaMemcpy(&add_cursor, s->d_add_cursor, sizeof(add_cursor),
                          cudaMemcpyDeviceToHost));
    uint32_t valid = (add_cursor < s->dist_cap) ? add_cursor : s->dist_cap;
    std::vector<GpuVoxelDist> dist_buf(valid);
    if (valid > 0) {
        CUDA_CHECK(cudaMemcpy(dist_buf.data(), s->dist_buf,
                              valid * sizeof(GpuVoxelDist), cudaMemcpyDeviceToHost));
    }
    uint32_t count = 0;
    for (uint32_t i = 0; i < valid; i++) {
        if (dist_buf[i].point_count > 0) {
            points[count].x = dist_buf[i].mean[0];
            points[count].y = dist_buf[i].mean[1];
            points[count].z = dist_buf[i].mean[2];
            points[count].i = 0.0f;
            points[count].t = 0.0f;
            count++;
        }
    }
    *size = count;
}

uint32_t get_map_capacity_impl() {
    if (multiresolution_active_impl()) return multiresolution_capacity_impl();
    return g_state ? g_state->dist_cap : 0;
}

uint32_t get_map_size_impl() {
    if (multiresolution_active_impl()) return multiresolution_size_impl();
    if (!g_state) return 0;
    uint32_t add_cursor = 0;
    CUDA_CHECK(cudaMemcpy(&add_cursor, g_state->d_add_cursor, sizeof(add_cursor),
                          cudaMemcpyDeviceToHost));
    return (add_cursor < g_state->dist_cap) ? add_cursor : g_state->dist_cap;
}

void cleanup_map_impl() {
    cleanup_multiresolution_impl();
    if (g_state) {
        delete g_state;
        g_state = nullptr;
    }
}

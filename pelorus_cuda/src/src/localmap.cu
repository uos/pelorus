// localmap.cu — CUDA port of pelorus_metal (shaders/ + map.rs)
//
// Matches the Metal backend one-to-one:
//   - Same GPU kernels (common.metal, bitonic.metal, localmap.metal, planes.metal, hashmap.metal)
//   - Same CPU-side LRU policy (map.rs: assign_target_indices_with_cpu_lru / rebuild_cpu_lru_order)
//   - cudaMallocManaged for buffers the CPU accesses; cudaMalloc for GPU-only buffers

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

#include <lib.h>
#include <localmap.cuh>

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr int      EMPTY_KEY   = INT_MIN;
static constexpr uint32_t INVALID_IDX = UINT32_MAX;

// ── CUDA error check ──────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_e));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ── Device helpers (ported from common.metal) ─────────────────────────────────

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
    return (g.x * 73856093) + (g.y * 471943) + (g.z * 83492791);
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

__device__ void regularize_cov(float* __restrict__ cov) {
    const float eps = 1e-6f;
    cov[0] += eps;
    cov[4] += eps;
    cov[8] += eps;
}

// Symmetric 3×3 eigenvalues (trigonometric method, ported from common.metal)
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
    if (ad != 0.0f) {
        float arg = (-bd * 0.5f) / sqrtf(-ad * ad * ad / 27.0f);
        arg   = fmaxf(-1.0f, fminf(1.0f, arg));
        angle = acosf(arg);
    }

    const float two_s = 2.0f * sqrtf(-ad / 3.0f);
    const float p3    = p / 3.0f;
    const float pi2   = 6.28318530718f;
    float3 ev;
    ev.x = two_s * cosf(angle / 3.0f) - p3;
    ev.y = two_s * cosf((angle + pi2) / 3.0f) - p3;
    ev.z = two_s * cosf((angle + 2.0f*pi2) / 3.0f) - p3;
    return ev;
}

// Hashmap lookup — returns INVALID_IDX if key not found
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

// Pool (merge) two distributions using the parallel-mean formula (common.metal pool_dist)
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

// ── GPU Kernels ────────────────────────────────────────────────────────────────

// hashmap.metal: hashmap_init
__global__ void hashmap_init_kernel(int* table_keys, uint32_t table_cap) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < table_cap) table_keys[idx] = EMPTY_KEY;
}

// hashmap.metal: rebuild_hashmap_keyed
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

// bitonic.metal: pack_keys — (int32 key, uint32 index) → uint64
__global__ void pack_keys_kernel(const int*      keys,
                                 const uint32_t* indices,
                                 uint64_t*       packed,
                                 uint32_t        n_points) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;
    uint32_t ku   = (uint32_t)keys[idx];
    packed[idx]   = ((uint64_t)ku << 32) | (uint64_t)indices[idx];
}

// bitonic.metal: init_pad — fill [start, total_cap) with 0xFFFFFFFFFFFFFFFF
__global__ void init_pad_kernel(uint64_t* packed, uint32_t start, uint32_t total_cap) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pos = start + idx;
    if (pos < total_cap) packed[pos] = 0xFFFFFFFFFFFFFFFFULL;
}

// bitonic.metal: bitonic_step
__global__ void bitonic_step_kernel(uint64_t* data, uint32_t j, uint32_t k, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t l = idx ^ j;
    if (l <= idx) return;

    bool ascending = ((idx & k) == 0);
    if ((data[idx] > data[l]) == ascending) {
        uint64_t tmp = data[idx];
        data[idx]    = data[l];
        data[l]      = tmp;
    }
}

// bitonic.metal: unpack_keys
__global__ void unpack_keys_kernel(const uint64_t* packed,
                                   int*            keys,
                                   uint32_t*       indices,
                                   uint32_t        n_points) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;
    uint64_t p   = packed[idx];
    keys[idx]    = (int)(uint32_t)(p >> 32);
    indices[idx] = (uint32_t)(p & 0xFFFFFFFFULL);
}

// localmap.metal: calculate_keys
// unique_key_count is used here as cloud_size
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

// localmap.metal: reorder_points — reorder AND transform to global frame
// unique_key_count is used here as cloud_size
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

// localmap.metal: build_voxels — one thread per unique voxel
// unique_key_count is used here as n_assigned
__global__ void build_voxels_kernel(const Point*      cloud,
                                    const uint32_t*   run_starts,
                                    const uint32_t*   run_ends,
                                    const int*        unique_keys,
                                    const uint32_t*   target_indices,
                                    const uint32_t*   update_flags,
                                    GpuVoxelDist*     dists,
                                    BuildVoxelsParams params) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.unique_key_count) return;

    int      voxel_key = unique_keys[idx];
    uint32_t start     = run_starts[idx];
    uint32_t end       = run_ends[idx];
    uint32_t count     = end - start;

    if (count == 0 || voxel_key == EMPTY_KEY) return;

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
                             uint32_t*           out_size,
                             PointNormal*        out_buff,
                             PlanesParams        params) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.cloud_size) return;

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
    if (mag < 1e-9f) return;
    normal.x /= mag; normal.y /= mag; normal.z /= mag;

    float3 mean_p = make_float3(merged_mean[0], merged_mean[1], merged_mean[2]);
    float  d_val  = -(normal.x*mean_p.x + normal.y*mean_p.y + normal.z*mean_p.z);

    // Score filter (same formula as Metal/CPU)
    float pd2   = normal.x*p_global.x + normal.y*p_global.y + normal.z*p_global.z + d_val;
    float l2_sq = p_local.x*p_local.x + p_local.y*p_local.y + p_local.z*p_local.z;
    float s     = 1.0f - 0.9f * fabsf(pd2) / sqrtf(sqrtf(fmaxf(l2_sq, 1e-9f)));
    if (s <= 0.9f) return;

    uint32_t out_idx = atomicAdd(out_size, 1u);
    PointNormal pn;
    pn.x        = p_local.x;
    pn.y        = p_local.y;
    pn.z        = p_local.z;
    pn.intensity = pd2;
    pn.normal_x  = normal.x;
    pn.normal_y  = normal.y;
    pn.normal_z  = normal.z;
    out_buff[out_idx] = pn;
}

// ── CPU state ─────────────────────────────────────────────────────────────────

struct SortKey {
    uint32_t index;
    uint64_t temporal_dist;
    uint64_t point_count;
};

struct CudaMapState {
    // GPU-only (device memory)
    int*      table_keys;   // size: table_cap
    uint32_t* table_vals;   // size: table_cap
    // CPU+GPU shared (managed memory)
    GpuVoxelDist* dist_buf;  // size: dist_cap
    int*          dist_keys; // size: dist_cap
    // Scratch (managed: CPU reads keys after sort, writes run arrays)
    int*      scratch_keys;
    uint32_t* scratch_indices;
    uint64_t* scratch_packed;
    uint32_t* scratch_run_starts;
    uint32_t* scratch_run_ends;
    int*      scratch_unique_keys;
    uint32_t* scratch_target_indices;
    uint32_t* scratch_update_flags;
    Point*    scratch_sorted_points;
    uint32_t* counter;  // managed: atomic planes output counter

    // Config
    uint32_t dist_cap;
    uint32_t table_cap;
    float    resolution;
    float    sensitivity;
    float    min_dist;
    uint32_t scan_number;
    uint32_t nbtype;

    // CPU LRU (mirrors Metal map.rs MetalMapState)
    uint32_t add_cursor;
    uint32_t lru_cursor;
    std::vector<uint32_t> lru_mapping;
    std::vector<SortKey>  lru_mapping_keys;
    std::unordered_map<int32_t, uint32_t> key_to_index;
    bool     pending_lru_rebuild;
    uint32_t pending_lru_scan_number;

    size_t scratch_cap;  // current allocated scratch capacity

    CudaMapState()
        : table_keys(nullptr), table_vals(nullptr),
          dist_buf(nullptr), dist_keys(nullptr),
          scratch_keys(nullptr), scratch_indices(nullptr),
          scratch_packed(nullptr), scratch_run_starts(nullptr),
          scratch_run_ends(nullptr), scratch_unique_keys(nullptr),
          scratch_target_indices(nullptr), scratch_update_flags(nullptr),
          scratch_sorted_points(nullptr), counter(nullptr),
          dist_cap(0), table_cap(0), resolution(0), sensitivity(0),
          min_dist(0), scan_number(0), nbtype(26),
          add_cursor(0), lru_cursor(0),
          pending_lru_rebuild(false), pending_lru_scan_number(0),
          scratch_cap(0) {}

    ~CudaMapState() {
        // Ignore errors during shutdown
        if (table_keys)             cudaFree(table_keys);
        if (table_vals)             cudaFree(table_vals);
        if (dist_buf)               cudaFree(dist_buf);
        if (dist_keys)              cudaFree(dist_keys);
        if (scratch_keys)           cudaFree(scratch_keys);
        if (scratch_indices)        cudaFree(scratch_indices);
        if (scratch_packed)         cudaFree(scratch_packed);
        if (scratch_run_starts)     cudaFree(scratch_run_starts);
        if (scratch_run_ends)       cudaFree(scratch_run_ends);
        if (scratch_unique_keys)    cudaFree(scratch_unique_keys);
        if (scratch_target_indices) cudaFree(scratch_target_indices);
        if (scratch_update_flags)   cudaFree(scratch_update_flags);
        if (scratch_sorted_points)  cudaFree(scratch_sorted_points);
        if (counter)                cudaFree(counter);
    }
};

static CudaMapState* g_state = nullptr;

// ── CPU helpers ───────────────────────────────────────────────────────────────

static inline int grid_dim(uint32_t n) { return (n + 255) / 256; }

static void alloc_scratch(CudaMapState* s, size_t new_cap) {
    if (s->scratch_keys)           CUDA_CHECK(cudaFree(s->scratch_keys));
    if (s->scratch_indices)        CUDA_CHECK(cudaFree(s->scratch_indices));
    if (s->scratch_packed)         CUDA_CHECK(cudaFree(s->scratch_packed));
    if (s->scratch_run_starts)     CUDA_CHECK(cudaFree(s->scratch_run_starts));
    if (s->scratch_run_ends)       CUDA_CHECK(cudaFree(s->scratch_run_ends));
    if (s->scratch_unique_keys)    CUDA_CHECK(cudaFree(s->scratch_unique_keys));
    if (s->scratch_target_indices) CUDA_CHECK(cudaFree(s->scratch_target_indices));
    if (s->scratch_update_flags)   CUDA_CHECK(cudaFree(s->scratch_update_flags));
    if (s->scratch_sorted_points)  CUDA_CHECK(cudaFree(s->scratch_sorted_points));

    CUDA_CHECK(cudaMallocManaged(&s->scratch_keys,           new_cap * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&s->scratch_indices,        new_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged(&s->scratch_packed,         new_cap * sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocManaged(&s->scratch_run_starts,     new_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged(&s->scratch_run_ends,       new_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged(&s->scratch_unique_keys,    new_cap * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&s->scratch_target_indices, new_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged(&s->scratch_update_flags,   new_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged(&s->scratch_sorted_points,  new_cap * sizeof(Point)));

    s->scratch_cap = new_cap;
}

// Mirrors Metal map.rs assign_target_indices_with_cpu_lru
static void assign_target_indices_with_cpu_lru(
    CudaMapState*         s,
    const int*            unique_keys,
    uint32_t              n_unique,
    std::vector<uint32_t>& target_indices,
    std::vector<uint32_t>& update_flags)
{
    target_indices.clear();
    update_flags.clear();
    target_indices.reserve(n_unique);
    update_flags.reserve(n_unique);

    uint32_t dist_cap = s->dist_cap;

    for (uint32_t i = 0; i < n_unique; i++) {
        int key = unique_keys[i];

        auto it = s->key_to_index.find(key);
        if (it != s->key_to_index.end()) {
            target_indices.push_back(it->second);
            update_flags.push_back(1);
            continue;
        }

        if (s->add_cursor < dist_cap) {
            uint32_t idx = s->add_cursor++;
            s->dist_keys[idx] = key;
            s->key_to_index[key] = idx;
            target_indices.push_back(idx);
            update_flags.push_back(0);
            continue;
        }

        if (s->lru_cursor >= dist_cap) {
            fprintf(stderr, "[CUDA] WARNING: LRU evicted all slots — increase map capacity\n");
            break;
        }

        uint32_t victim  = s->lru_mapping[s->lru_cursor++];
        int      old_key = s->dist_keys[victim];
        if (old_key != EMPTY_KEY) s->key_to_index.erase(old_key);
        s->dist_keys[victim] = key;
        s->key_to_index[key] = victim;

        target_indices.push_back(victim);
        update_flags.push_back(0);
    }

    if (s->add_cursor > dist_cap) s->add_cursor = dist_cap;
}

// Mirrors Metal map.rs rebuild_cpu_lru_order — sort descending by temporal_dist/point_count
static void rebuild_cpu_lru_order(CudaMapState* s, uint32_t scan_number) {
    if (s->add_cursor < s->dist_cap) return;

    for (size_t i = 0; i < s->lru_mapping.size(); i++) {
        uint32_t idx = s->lru_mapping[i];
        const GpuVoxelDist& item = s->dist_buf[idx];
        uint64_t td = (scan_number >= item.last_scan)
                    ? (uint64_t)(scan_number - item.last_scan) : 0ULL;
        s->lru_mapping_keys[i] = SortKey{ idx, td, (uint64_t)item.point_count };
    }

    std::sort(s->lru_mapping_keys.begin(), s->lru_mapping_keys.end(),
        [](const SortKey& a, const SortKey& b) {
            // a < b means a is a better eviction candidate
            if (a.point_count == 0 && b.point_count == 0) return false;
            if (a.point_count == 0) return true;
            if (b.point_count == 0) return false;
            // descending by td/pc: a < b iff a.td*b.pc > b.td*a.pc
            unsigned __int128 left  = (unsigned __int128)a.temporal_dist * b.point_count;
            unsigned __int128 right = (unsigned __int128)b.temporal_dist * a.point_count;
            return left > right;
        });

    for (size_t i = 0; i < s->lru_mapping_keys.size(); i++)
        s->lru_mapping[i] = s->lru_mapping_keys[i].index;
    s->lru_cursor = 0;
}

// ── Public impl functions (declared in localmap.cuh) ─────────────────────────

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

    // Managed (CPU+GPU) buffers
    CUDA_CHECK(cudaMallocManaged(&s->dist_buf,  dist_cap * sizeof(GpuVoxelDist)));
    CUDA_CHECK(cudaMallocManaged(&s->dist_keys, dist_cap * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&s->counter,   sizeof(uint32_t)));

    // Zero dist_buf; fill dist_keys with EMPTY_KEY
    CUDA_CHECK(cudaMemsetAsync(s->dist_buf, 0, dist_cap * sizeof(GpuVoxelDist), stream));
    for (uint32_t i = 0; i < dist_cap; i++) s->dist_keys[i] = EMPTY_KEY;
    *s->counter = 0;

    // Scratch buffers (minimum size; will grow on first add_to_map_impl)
    alloc_scratch(s, 1);

    s->dist_cap  = dist_cap;
    s->table_cap = table_cap;
    s->resolution   = voxel_size;
    s->sensitivity  = sensitivity;
    s->min_dist     = min_dist;
    s->scan_number  = 0;
    s->nbtype       = nbtype;
    s->add_cursor   = 0;
    s->lru_cursor   = 0;

    s->lru_mapping.resize(dist_cap);
    for (uint32_t i = 0; i < dist_cap; i++) s->lru_mapping[i] = i;
    s->lru_mapping_keys.resize(dist_cap);
    s->key_to_index.reserve(dist_cap);
    s->pending_lru_rebuild     = false;
    s->pending_lru_scan_number = 0;

    // Init hashmap
    hashmap_init_kernel<<<grid_dim(table_cap), 256, 0, stream>>>(s->table_keys, table_cap);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void add_to_map_impl(cudaStream_t stream, Point* points, uint32_t n_points, Pose pose) {
    CudaMapState* s = g_state;
    assert(s != nullptr);

    // Deferred LRU rebuild: sync GPU work from previous scan, then read dist_buf
    if (s->pending_lru_rebuild) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        rebuild_cpu_lru_order(s, s->pending_lru_scan_number);
        s->pending_lru_rebuild = false;
    }

    if (n_points == 0) return;

    s->scan_number++;
    uint32_t scan_number = s->scan_number;

    // Compute padded_n for bitonic sort BEFORE scratch resize check
    uint32_t padded_n = 1;
    while (padded_n < n_points) padded_n <<= 1;

    // Scratch must hold at least padded_n elements (for packed buffer)
    if (s->scratch_cap < (size_t)padded_n) {
        // Use at least 1.5× padded_n to reduce future reallocations
        size_t new_cap = padded_n + padded_n / 2;
        alloc_scratch(s, new_cap);
    }

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
    params.unique_key_count = n_points;  // cloud_size for calculate_keys
    params.table_cap      = s->table_cap;
    params.dist_cap       = s->dist_cap;
    params._pad1          = 0;

    // ── Step 1: calculate_keys ────────────────────────────────────────────────
    calculate_keys_kernel<<<grid_dim(n_points), 256, 0, stream>>>(
        points, s->scratch_keys, s->scratch_indices, params);

    // ── Step 2: Bitonic sort of (key, index) pairs ────────────────────────────
    pack_keys_kernel<<<grid_dim(n_points), 256, 0, stream>>>(
        s->scratch_keys, s->scratch_indices, s->scratch_packed, n_points);

    if (n_points < padded_n) {
        uint32_t diff = padded_n - n_points;
        init_pad_kernel<<<grid_dim(diff), 256, 0, stream>>>(
            s->scratch_packed, n_points, (uint32_t)s->scratch_cap);
    }

    for (uint32_t k = 2; k <= padded_n; k <<= 1)
        for (uint32_t j = k >> 1; j > 0; j >>= 1)
            bitonic_step_kernel<<<grid_dim(padded_n), 256, 0, stream>>>(
                s->scratch_packed, j, k, padded_n);

    unpack_keys_kernel<<<grid_dim(n_points), 256, 0, stream>>>(
        s->scratch_packed, s->scratch_keys, s->scratch_indices, n_points);

    // Sync: CPU needs to read sorted keys
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Step 3: CPU run computation ───────────────────────────────────────────
    std::vector<int>      unique_keys;
    std::vector<uint32_t> run_starts, run_ends;

    if (n_points > 0) {
        int      cur_key   = s->scratch_keys[0];
        uint32_t start_idx = 0;
        for (uint32_t i = 1; i < n_points; i++) {
            if (s->scratch_keys[i] != cur_key) {
                unique_keys.push_back(cur_key);
                run_starts.push_back(start_idx);
                run_ends.push_back(i);
                cur_key   = s->scratch_keys[i];
                start_idx = i;
            }
        }
        unique_keys.push_back(cur_key);
        run_starts.push_back(start_idx);
        run_ends.push_back(n_points);
    }

    // ── Step 4: CPU LRU assignment ────────────────────────────────────────────
    std::vector<uint32_t> target_indices, update_flags;
    assign_target_indices_with_cpu_lru(s, unique_keys.data(),
                                       (uint32_t)unique_keys.size(),
                                       target_indices, update_flags);
    uint32_t n_assigned = (uint32_t)target_indices.size();
    if (n_assigned == 0) return;

    unique_keys.resize(n_assigned);
    run_starts.resize(n_assigned);
    run_ends.resize(n_assigned);

    // Copy CPU arrays into managed scratch buffers (GPU will read them)
    memcpy(s->scratch_run_starts,     run_starts.data(),     n_assigned * sizeof(uint32_t));
    memcpy(s->scratch_run_ends,       run_ends.data(),       n_assigned * sizeof(uint32_t));
    memcpy(s->scratch_unique_keys,    unique_keys.data(),    n_assigned * sizeof(int));
    memcpy(s->scratch_target_indices, target_indices.data(), n_assigned * sizeof(uint32_t));
    memcpy(s->scratch_update_flags,   update_flags.data(),   n_assigned * sizeof(uint32_t));

    // ── Step 5: reorder_points ────────────────────────────────────────────────
    BuildVoxelsParams params_reorder = params;
    params_reorder.unique_key_count  = n_points;  // cloud_size
    reorder_points_kernel<<<grid_dim(n_points), 256, 0, stream>>>(
        points, s->scratch_sorted_points, s->scratch_indices, params_reorder);

    // ── Step 6: build_voxels (one thread per unique voxel) ───────────────────
    BuildVoxelsParams params_vox = params;
    params_vox.unique_key_count  = n_assigned;
    build_voxels_kernel<<<grid_dim(n_assigned), 256, 0, stream>>>(
        s->scratch_sorted_points,
        s->scratch_run_starts,
        s->scratch_run_ends,
        s->scratch_unique_keys,
        s->scratch_target_indices,
        s->scratch_update_flags,
        s->dist_buf,
        params_vox);

    // ── Step 7: Rebuild hashmap ───────────────────────────────────────────────
    hashmap_init_kernel<<<grid_dim(s->table_cap), 256, 0, stream>>>(
        s->table_keys, s->table_cap);

    uint32_t valid_size = (s->add_cursor < s->dist_cap) ? s->add_cursor : s->dist_cap;
    if (valid_size > 0) {
        rebuild_hashmap_keyed_kernel<<<grid_dim(valid_size), 256, 0, stream>>>(
            s->table_keys, s->table_vals, s->dist_keys, valid_size, s->table_cap);
    }

    // Defer sync to start of next scan (mirrors Metal's deferred design)
    s->pending_lru_rebuild     = true;
    s->pending_lru_scan_number = scan_number;
}

uint32_t planes_impl(cudaStream_t stream, Point* cloud, uint32_t cloud_size,
                     Pose lidar_to_imu, Pose pose, PointNormal* out_buff) {
    CudaMapState* s = g_state;
    assert(s != nullptr);

    // Ensure previous add_to_map GPU work (hashmap rebuild) is visible
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Zero atomic counter (CPU write safe now — stream is idle)
    *s->counter = 0;

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
        s->counter, out_buff, params);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    return *s->counter;
}

void get_map_impl(Point* points, uint32_t* size) {
    if (!g_state) { *size = 0; return; }
    CudaMapState* s = g_state;

    uint32_t valid = (s->add_cursor < s->dist_cap) ? s->add_cursor : s->dist_cap;
    uint32_t count = 0;
    for (uint32_t i = 0; i < valid; i++) {
        if (s->dist_buf[i].point_count > 0) {
            points[count].x = s->dist_buf[i].mean[0];
            points[count].y = s->dist_buf[i].mean[1];
            points[count].z = s->dist_buf[i].mean[2];
            points[count].i = 0.0f;
            points[count].t = 0.0f;
            count++;
        }
    }
    *size = count;
}

uint32_t get_map_capacity_impl() {
    return g_state ? g_state->dist_cap : 0;
}

uint32_t get_map_size_impl() {
    if (!g_state) return 0;
    return (g_state->add_cursor < g_state->dist_cap) ? g_state->add_cursor : g_state->dist_cap;
}

void cleanup_map_impl() {
    if (g_state) {
        // Do NOT call cudaDeviceSynchronize() here: this is called from an atexit
        // handler registered via libc::atexit() in Rust. CUDA's own atexit handlers
        // are registered AFTER ours (LIFO order), so they run BEFORE ours during
        // process exit — meaning CUDA may already be torn down when we get here.
        // cudaFree inside ~CudaMapState already ignores cudaErrorCudartUnloading.
        delete g_state;
        g_state = nullptr;
    }
}

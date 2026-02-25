#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// ── Constants ────────────────────────────────────────────────────────────────

constant int EMPTY_KEY = INT_MIN;

// ── Shared data types ────────────────────────────────────────────────────────

// Must match Rust Point { x, y, z, i, t: f32 } = 20 bytes
struct Point {
    float x, y, z, i, t;
};

// Must match Rust PointNormal (repr(C, align(16)), 8 floats = 32 bytes)
struct PointNormal {
    float x, y, z, intensity, nx, ny, nz, _pad;
};

// VoxelDistribution layout (must match Rust GpuVoxelDist exactly)
struct VoxelDistribution {
    atomic_int lock_flag;
    float transform[9];  // Cholesky factor L
    uint  point_count;
    float mean[3];
    float cov[9];        // row-major 3x3 covariance
    uint  last_scan;
};

// ── Math helpers ─────────────────────────────────────────────────────────────

// Rotate vector v by quaternion q = (x, y, z, w)
inline float3 quat_rotate(float4 q, float3 v) {
    float3 u = q.xyz;
    float3 t = 2.0 * cross(u, v);
    return v + q.w * t + cross(u, t);
}

// Apply isometry (rotate then translate)
inline float3 iso_transform(float4 quat, float3 trans, float3 point) {
    return quat_rotate(quat, point) + trans;
}

// Convert world position → voxel grid cell (match CUDA floor)
inline int3 grid_pos(float resolution, float3 p) {
    float inv = 1.0 / resolution;
    return int3(int(floor(p.x * inv)),
                int(floor(p.y * inv)),
                int(floor(p.z * inv)));
}

// Map grid cell → int32 hash key (aligned with CUDA hash_grid)
inline int hash_grid(int3 g) {
    uint64_t h = (g.x * 73856093) ^
                 (g.y * 471943) ^
                 (g.z * 83492791);
    return int(h);
}

// Murmur3-inspired hash for open-addressing (same as CUDA hash_function)
inline uint hash_fn(int key) {
    uint h = uint(key);
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

// ── Cholesky 3×3 ─────────────────────────────────────────────────────────────

// Returns true on success; fills L (lower-triangular) such that A = L * L^T
template<typename T>
inline bool cholesky_3x3_impl(T A, thread float* L) {
    // Zero out L
    for (int i = 0; i < 9; i++) L[i] = 0.0;

    float l00_sq = A[0];
    if (l00_sq <= 0.0) return false;
    float l00 = sqrt(l00_sq);
    L[0] = l00;

    L[3] = A[3] / l00;   // L(1,0)
    L[6] = A[6] / l00;   // L(2,0)

    float l11_sq = A[4] - L[3] * L[3];
    if (l11_sq <= 0.0) return false;
    float l11 = sqrt(l11_sq);
    L[4] = l11;

    L[7] = (A[7] - L[6] * L[3]) / l11; // L(2,1)

    float l22_sq = A[8] - (L[6] * L[6] + L[7] * L[7]);
    if (l22_sq <= 0.0) return false;
    L[8] = sqrt(l22_sq);

    return true;
}

inline bool cholesky_3x3(thread const float* A, thread float* L) {
    return cholesky_3x3_impl(A, L);
}
inline bool cholesky_3x3_device(device const float* A, thread float* L) {
    return cholesky_3x3_impl(A, L);
}

// Forward + backward substitution: solve A x = b using Cholesky factor L
inline float3 cholesky_solve_3x3(thread const float* L, float3 b) {
    // Forward: L y = b
    float3 y;
    y.x = b.x / L[0];
    y.y = (b.y - L[3] * y.x) / L[4];
    y.z = (b.z - L[6] * y.x - L[7] * y.y) / L[8];

    // Backward: L^T x = y
    float3 x;
    x.z = y.z / L[8];
    x.y = (y.y - L[7] * x.z) / L[4];
    x.x = (y.x - L[3] * x.y - L[6] * x.z) / L[0];
    return x;
}

// Compute (mu1 - mu2)^T * inv(Cov) * (mu1 - mu2)
inline float mahalanobis_sq(float3 diff, thread const float* L) {
    float3 sol = cholesky_solve_3x3(L, diff);
    return dot(diff, sol);
}

inline void regularize_cov(thread float* cov) {
    float regularization_factor = 1e-6f;
    cov[0] += regularization_factor;
    cov[4] += regularization_factor;
    cov[8] += regularization_factor;
}

// ── Symmetric 3×3 eigenvalues (trigonometric method) ─────────────────────────

template<typename T>
inline float3 symmetric_eigen_3x3_impl(T A) {
    const float a00 = A[0], a01 = A[1], a02 = A[2];
    const float a11 = A[4], a12 = A[5];
    const float a22 = A[8];

    const float traceA = a00 + a11 + a22;
    const float c2 = (a00 * a11 + a00 * a22 + a11 * a22)
                   - (a01 * a01 + a02 * a02 + a12 * a12);
    const float detA = a00 * (a11 * a22 - a12 * a12)
                     - a01 * (a01 * a22 - a12 * a02)
                     + a02 * (a01 * a12 - a11 * a02);

    const float p = -traceA;
    const float q = c2;
    const float r = -detA;

    const float a_d = (3.0 * q - p * p) / 3.0;
    const float b_d = (2.0 * p * p * p - 9.0 * p * q + 27.0 * r) / 27.0;

    float angle = 0.0;
    if (a_d != 0.0) {
        float arg = (-b_d / 2.0) / sqrt(-a_d * a_d * a_d / 27.0);
        arg = clamp(arg, -1.0f, 1.0f);
        angle = acos(arg);
    }

    const float two_s = 2.0 * sqrt(-a_d / 3.0);
    const float p3 = p / 3.0;
    const float pi2 = 6.28318530718f;

    float3 ev;
    ev.x = two_s * cos(angle / 3.0) - p3;
    ev.y = two_s * cos((angle + pi2) / 3.0) - p3;
    ev.z = two_s * cos((angle + 2.0 * pi2) / 3.0) - p3;
    return ev;
}

inline float3 symmetric_eigen_3x3(thread const float* A) {
    return symmetric_eigen_3x3_impl(A);
}
inline float3 symmetric_eigen_3x3_device(device const float* A) {
    return symmetric_eigen_3x3_impl(A);
}

// ── Hashmap lookup ────────────────────────────────────────────────────────────

// Returns buffer index for key, or UINT_MAX if not found.
inline uint hashmap_lookup(device const atomic_int* table_keys,
                           device const uint* table_vals,
                           uint table_capacity,
                           int key) {
    if (key == EMPTY_KEY) return UINT_MAX;
    uint slot = hash_fn(key) & (table_capacity - 1);
    for (;;) {
        int cur = atomic_load_explicit(&table_keys[slot], memory_order_relaxed);
        if (cur == key)       return table_vals[slot];
        if (cur == EMPTY_KEY) return UINT_MAX;
        slot = (slot + 1) & (table_capacity - 1);
    }
}

// ── Neighbor offsets ──────────────────────────────────────────────────────────

// Writes up to 27 neighbor grid offsets into out[]; returns count.
inline uint neighbor_cells(uint nbtype, int3 g, thread int3* out) {
    uint n = 0;
    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        out[n++] = g + int3(dx, dy, dz);
    }
    return n;
}

// ── Pool neighbor VoxelDistributions (parallel combination) ──────────────────

// Merge `other` into `acc` using parallel-mean formula (unbiased covariance matching CUDA)
inline void pool_dist(thread uint& acc_point_count,
                      thread float* acc_mean,
                      thread float* acc_cov,
                      device const VoxelDistribution& other) {
    if (other.point_count == 0) return;

    if (acc_point_count == 0) {
        acc_point_count  = other.point_count;
        acc_mean[0]      = other.mean[0];
        acc_mean[1]      = other.mean[1];
        acc_mean[2]      = other.mean[2];
        for (int i = 0; i < 9; i++) acc_cov[i] = other.cov[i];
        return;
    }

    float fn1 = float(acc_point_count);
    float fn2 = float(other.point_count);
    float fn_t = fn1 + fn2;

    if (fn_t <= 1.0f) {
        // Degenerate case, just average
        acc_mean[0] = (acc_mean[0] * fn1 + other.mean[0] * fn2) / fn_t;
        acc_mean[1] = (acc_mean[1] * fn1 + other.mean[1] * fn2) / fn_t;
        acc_mean[2] = (acc_mean[2] * fn1 + other.mean[2] * fn2) / fn_t;
        acc_point_count = (uint)fn_t;
        for(int i=0; i<9; i++) acc_cov[i] = 0;
        return;
    }

    float3 delta;
    delta.x = other.mean[0] - acc_mean[0];
    delta.y = other.mean[1] - acc_mean[1];
    delta.z = other.mean[2] - acc_mean[2];

    float3 new_mean;
    new_mean.x = acc_mean[0] + delta.x * (fn2 / fn_t);
    new_mean.y = acc_mean[1] + delta.y * (fn2 / fn_t);
    new_mean.z = acc_mean[2] + delta.z * (fn2 / fn_t);

    // Combine Covariance (matching CUDA logic):
    // Treat input covariances as if they came from Biased storage, 
    // effectively converting to scatter matrix by * (N-1).
    
    float fn1_m1 = (fn1 > 1.0f) ? (fn1 - 1.0f) : 0.0f;
    float fn2_m1 = (fn2 > 1.0f) ? (fn2 - 1.0f) : 0.0f;
    
    // Weight for the mean-separation term
    float delta_weight = (fn1 * fn2 / fn_t);
    
    float d[3] = { delta.x, delta.y, delta.z };

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            // S1 = Cov1 * (n1 - 1)
            float s1 = acc_cov[r * 3 + c] * fn1_m1;
            // S2 = Cov2 * (n2 - 1)
            float s2 = other.cov[r * 3 + c] * fn2_m1;
            
            // New Scatter = S1 + S2 + delta_outer * weight
            float scatter = s1 + s2 + d[r] * d[c] * delta_weight;
            
            // New Cov = Scatter / (n_total - 1)
            acc_cov[r * 3 + c] = scatter / (fn_t - 1.0f);
        }
    }

    acc_mean[0] = new_mean.x;
    acc_mean[1] = new_mean.y;
    acc_mean[2] = new_mean.z;
    acc_point_count = (uint)fn_t;
}

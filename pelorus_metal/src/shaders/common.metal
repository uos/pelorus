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
// mean[3]:     12 bytes (offset 0)
// point_count:  4 bytes (offset 12)
// cov[9]:      36 bytes (offset 16)
// last_scan:    4 bytes (offset 52)
// L[9]:        36 bytes (offset 56)
// Total: 92 bytes
struct VoxelDistribution {
    float mean[3];
    uint  point_count;
    float cov[9];       // row-major 3x3 covariance
    uint  last_scan;
    float L[9];         // row-major 3x3 Cholesky factor
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

// Convert world position → voxel grid cell
inline int3 pos2grid(float resolution, float3 p) {
    float inv = 1.0 / resolution;
    return int3(int(floor(p.x * inv)),
                int(floor(p.y * inv)),
                int(floor(p.z * inv)));
}

// Map grid cell → int32 hash key (same as CUDA hash_grid)
inline int hash_grid(int3 g) {
    return int(((uint)(g.x * 73856093)) ^
               ((uint)(g.y * 471943))   ^
               ((uint)(g.z * 83492791)));
}

// Murmur3-inspired hash for open-addressing (same as CUDA hash_function)
inline uint hash_fn(int key) {
    uint h = uint(key);
    h = (h ^ (h >> 16)) * 0x85ebca6bu;
    h = (h ^ (h >> 13)) * 0xc2b2ae35u;
    h = h ^ (h >> 16);
    return h;
}

// ── Cholesky 3×3 ─────────────────────────────────────────────────────────────

// Returns true on success; fills L (lower-triangular) such that A = L * L^T
inline bool cholesky_3x3(thread const float* A, thread float* L) {
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

// ── Symmetric 3×3 eigenvalues (trigonometric method) ─────────────────────────

inline float3 symmetric_eigen_3x3(thread const float* A) {
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

// ── Hashmap lookup ────────────────────────────────────────────────────────────

// Returns buffer index for key, or UINT_MAX if not found.
// table_keys: flat int array of size table_capacity
// table_vals: flat uint array of size table_capacity
inline uint hashmap_lookup(device const int* table_keys,
                           device const uint* table_vals,
                           uint table_capacity,
                           int key) {
    if (key == EMPTY_KEY) return UINT_MAX;
    uint slot = hash_fn(key) & (table_capacity - 1);
    for (;;) {
        int cur = table_keys[slot];
        if (cur == key)       return table_vals[slot];
        if (cur == EMPTY_KEY) return UINT_MAX;
        slot = (slot + 1) & (table_capacity - 1);
    }
}

// ── Neighbor offsets ──────────────────────────────────────────────────────────

// Writes up to 27 neighbor grid offsets into out[]; returns count.
inline uint neighbor_cells(uint nbtype, int3 g, thread int3* out) {
    // nbtype: 0=center, 6=nearby6, 18=nearby18, 26=nearby26
    // We always output nearby_26 (27 cells) as that is what plane_fit uses.
    // nbtype parameter reserved for future use.
    uint n = 0;
    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        out[n++] = g + int3(dx, dy, dz);
    }
    return n; // always 27
}

// ── Pool neighbor VoxelDistributions (parallel combination) ──────────────────

// Merge `other` into `acc` using parallel-mean formula (no Mahalanobis weighting)
inline void pool_dist(thread VoxelDistribution& acc,
                      device const VoxelDistribution& other) {
    if (other.point_count == 0) return;

    uint n1 = acc.point_count;
    uint n2 = other.point_count;

    if (n1 == 0) {
        acc.point_count  = n2;
        acc.mean[0]      = other.mean[0];
        acc.mean[1]      = other.mean[1];
        acc.mean[2]      = other.mean[2];
        for (int i = 0; i < 9; i++) acc.cov[i] = other.cov[i];
        return;
    }

    uint n_total = n1 + n2;
    float fn1 = float(n1), fn2 = float(n2), fn = float(n_total);

    float3 delta;
    delta.x = other.mean[0] - acc.mean[0];
    delta.y = other.mean[1] - acc.mean[1];
    delta.z = other.mean[2] - acc.mean[2];

    float3 new_mean;
    new_mean.x = acc.mean[0] + delta.x * (fn2 / fn);
    new_mean.y = acc.mean[1] + delta.y * (fn2 / fn);
    new_mean.z = acc.mean[2] + delta.z * (fn2 / fn);

    // Welford parallel covariance: S_combined = S1 + S2 + delta*delta^T * n1*n2/n
    float w = fn1 * fn2 / fn;
    float d[3] = { delta.x, delta.y, delta.z };

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float s1 = acc.cov[r * 3 + c] * (fn1 > 1.0 ? (fn1 - 1.0) : 0.0);
            float s2 = other.cov[r * 3 + c] * (fn2 > 1.0 ? (fn2 - 1.0) : 0.0);
            acc.cov[r * 3 + c] = (s1 + s2 + d[r] * d[c] * w)
                                   / (fn > 1.0 ? (fn - 1.0) : 1.0);
        }
    }

    acc.mean[0] = new_mean.x;
    acc.mean[1] = new_mean.y;
    acc.mean[2] = new_mean.z;
    acc.point_count = n_total;
}

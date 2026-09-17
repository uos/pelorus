// multiresolution.metal — GPU-accelerated correspondence query against the
// CPU-maintained multiresolution distribution pyramid (see
// pelorus_metal::multiresolution). Building the pyramid itself (folding in
// new samples, merging, LRU eviction, neighborhood-fit caching) runs on the
// CPU in plain Rust — those touch only a few thousand cells per scan, so
// they are not worth an unverifiable custom GPU sort/scan network. This
// kernel handles only the hot, per-point plane lookup, called every ICP
// iteration for every point in a scan, mirroring `plane_krnl` in planes.metal.
//
// Metal has no double-precision floating point, so this runs in float
// throughout — the same precision tier the rest of this Metal backend
// already uses. The Vulkan/CUDA multiresolution backends use double for
// pose composition and eigendecomposition instead.
// Included after common.metal (iso_transform, symmetric_eigen_3x3) and
// planes.metal in the compiled source.

// Layout below must match `pelorus_metal::multiresolution::Cell` exactly
// (72 bytes, no padding: 3×i32 + u32 + u64 + 3×f32 + 9×f32).
struct MRCell {
    int key[3];
    uint scan;
    ulong count;
    float mean[3];
    float cov[9];
};

// Must match `pelorus_metal::multiresolution::Fit` exactly (68 bytes).
struct MRFit {
    uint status;
    float mean[3];
    float normal[3];
    float tangent[6];
    float extent[2];
    float quality;
    float d;
};

// Must match `pelorus_metal::multiresolution::QueryParams` exactly (100 bytes).
struct MRQueryParams {
    float lidar_to_imu_quat[4];
    float lidar_to_imu_pos[3];
    float _pad0;
    float pose_quat[4];
    float pose_trans[3];
    float _pad1;
    float resolution;
    uint level_sizes[4];
    uint n;
    uint _pad2[3];
};

template <typename A, typename B>
inline bool mr_key_less(A a, B b) {
    return a[0] != b[0] ? a[0] < b[0] : a[1] != b[1] ? a[1] < b[1] : a[2] < b[2];
}
template <typename A, typename B>
inline bool mr_key_equal(A a, B b) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

inline int mr_coordinate(float p, float r) {
    float v = floor(p / r);
    if (v >= 2147483648.0) return 2147483647;
    if (v <= -2147483648.0) return int(-2147483648.0);
    return int(v);
}

inline void mr_grid(thread const float* p, float r, thread int* out) {
    out[0] = mr_coordinate(p[0], r);
    out[1] = mr_coordinate(p[1], r);
    out[2] = mr_coordinate(p[2], r);
}

inline int mr_neighbor(int x, int d) {
    if (d > 0 && x == 2147483647) return x;
    if (d < 0 && x == (-2147483647 - 1)) return x;
    return x + d;
}

inline void mr_empty_cell(thread MRCell& c) {
    c.key[0] = 0; c.key[1] = 0; c.key[2] = 0;
    c.scan = 0;
    c.count = 0;
    for (int i = 0; i < 3; i++) c.mean[i] = 0.0;
    for (int i = 0; i < 9; i++) c.cov[i] = 0.0;
}

// `b` is taken by value so callers can pass a `device` cell directly. A
// by-value struct parameter copies out of whatever address space the
// argument lives in. A reference parameter would need to match exactly.
inline void mr_pool(thread MRCell& a, MRCell b) {
    if (b.count == 0) return;
    if (a.count == 0) { a = b; return; }
    float n1 = float(a.count), n2 = float(b.count), n = n1 + n2, w = n1 * n2 / n;
    float delta[3];
    for (int i = 0; i < 3; i++) delta[i] = a.mean[i] - b.mean[i];
    for (int i = 0; i < 3; i++) a.mean[i] = (n1 * a.mean[i] + n2 * b.mean[i]) / n;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            a.cov[3 * r + c] =
                (n1 * a.cov[3 * r + c] + n2 * b.cov[3 * r + c] + delta[r] * delta[c] * w) / n;
    a.count = ulong(n);
}

// Binary search a key-sorted level (cells are kept sorted by the CPU side).
inline int mr_find_cell(device const MRCell* cells, uint n, thread const int* key) {
    uint lo = 0, hi = n;
    while (lo < hi) {
        uint mid = lo + (hi - lo) / 2;
        if (mr_key_less(cells[mid].key, key)) lo = mid + 1; else hi = mid;
    }
    if (lo >= n) return -1;
    return mr_key_equal(cells[lo].key, key) ? int(lo) : -1;
}

// Null-space of (cov - lambda*I) via cross products of the shifted rows,
// picking the largest-magnitude candidate — the same technique `plane_fit`
// in planes.metal already uses to recover the smallest-eigenvalue
// eigenvector from a closed-form eigenvalue solve (Metal has no
// closed-form eigenvector solver of its own).
inline float3 mr_eigenvector(thread const float* cov, float lambda) {
    float m[9];
    for (int i = 0; i < 9; i++) m[i] = cov[i];
    m[0] -= lambda; m[4] -= lambda; m[8] -= lambda;
    float3 row0 = float3(m[0], m[1], m[2]);
    float3 row1 = float3(m[3], m[4], m[5]);
    float3 row2 = float3(m[6], m[7], m[8]);
    float3 v01 = cross(row0, row1), v12 = cross(row1, row2), v20 = cross(row2, row0);
    float m01 = dot(v01, v01), m12 = dot(v12, v12), m20 = dot(v20, v20);
    float3 v = (m01 > m12 && m01 > m20) ? v01 : (m12 > m20 ? v12 : v20);
    float mag = length(v);
    return mag > 1e-9f ? v / mag : float3(0, 0, 1);
}

// Fit a plane to a pooled cell. Mirrors `mr_fit` in the Vulkan/CUDA kernels:
// status 0 = insufficient support, 1 = conflicting/degenerate (stop the
// caller's level search), 2 = usable plane.
inline MRFit mr_fit(MRCell cell, float resolution) {
    MRFit f;
    f.status = 0;
    for (int i = 0; i < 3; i++) { f.mean[i] = 0; f.normal[i] = 0; }
    for (int i = 0; i < 6; i++) f.tangent[i] = 0;
    f.extent[0] = 0; f.extent[1] = 0; f.quality = 0; f.d = 0;
    if (cell.count < 6) return f;
    f.status = 1;

    float cov[9];
    for (int i = 0; i < 9; i++) {
        float x = cell.cov[i];
        if (isnan(x) || isinf(x)) return f;
        cov[i] = x;
    }

    float3 ev = symmetric_eigen_3x3(cov);
    int idx[3] = {0, 1, 2};
    float val[3] = {ev.x, ev.y, ev.z};
    for (int i = 0; i < 3; i++)
        for (int j = i + 1; j < 3; j++)
            if (val[idx[j]] < val[idx[i]]) { int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }

    float small = max(val[idx[0]], 0.0f);
    float middle = val[idx[1]];
    float large = val[idx[2]];
    if (isnan(middle) || isinf(middle) || isnan(large) || isinf(large)) return f;
    if (middle < 1e-4) { f.status = 0; return f; }
    float thickness = resolution * .15;
    if (small > thickness * thickness || small > .1 * middle) return f;
    f.status = 2;

    float3 normal = mr_eigenvector(cov, val[idx[0]]);
    float3 tangent0_raw = mr_eigenvector(cov, val[idx[1]]);
    // Re-orthogonalize against `normal`. The two null-space directions are
    // only exactly orthogonal for an exact shift, so guard numerical drift
    // and near-degenerate (repeated-eigenvalue) cases before completing the
    // right-handed basis.
    float3 t0 = tangent0_raw - normal * dot(tangent0_raw, normal);
    float t0len = length(t0);
    float3 tangent0 = t0len > 1e-6f ? t0 / t0len : normalize(cross(normal, float3(0, 0, 1)));
    float3 tangent1 = cross(normal, tangent0);

    f.mean[0] = cell.mean[0]; f.mean[1] = cell.mean[1]; f.mean[2] = cell.mean[2];
    f.normal[0] = normal.x; f.normal[1] = normal.y; f.normal[2] = normal.z;
    f.tangent[0] = tangent0.x; f.tangent[1] = tangent0.y; f.tangent[2] = tangent0.z;
    f.tangent[3] = tangent1.x; f.tangent[4] = tangent1.y; f.tangent[5] = tangent1.z;
    f.extent[0] = 3.0 * sqrt(max(middle, 0.0f)) + resolution * .5;
    f.extent[1] = 3.0 * sqrt(max(large, 0.0f)) + resolution * .5;
    float noise = resolution * .04;
    f.quality = (small + noise * noise) *
        (1.0 / max(middle - small, 1e-6f) + 1.0 / max(large - small, 1e-6f));
    f.d = -(normal.x * f.mean[0] + normal.y * f.mean[1] + normal.z * f.mean[2]);
    return f;
}

// Live 3x3x3 neighborhood pool + fit, for query points whose grid cell has
// no cached fit (i.e. it fell in a key with no cell of its own at this level).
inline MRFit mr_neighborhood(device const MRCell* cells, uint n, thread const int* key, float resolution) {
    MRCell pooled;
    mr_empty_cell(pooled);
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++) {
                int k[3] = {mr_neighbor(key[0], dx), mr_neighbor(key[1], dy), mr_neighbor(key[2], dz)};
                int i = mr_find_cell(cells, n, k);
                if (i >= 0) mr_pool(pooled, cells[i]);
            }
    return mr_fit(pooled, resolution);
}

kernel void mr_query(
    device const Point*    cloud    [[buffer(0)]],
    device const MRCell*   cells0   [[buffer(1)]],
    device const MRCell*   cells1   [[buffer(2)]],
    device const MRCell*   cells2   [[buffer(3)]],
    device const MRCell*   cells3   [[buffer(4)]],
    device const MRFit*    fits0    [[buffer(5)]],
    device const MRFit*    fits1    [[buffer(6)]],
    device const MRFit*    fits2    [[buffer(7)]],
    device const MRFit*    fits3    [[buffer(8)]],
    device PointNormal*    out_buff [[buffer(9)]],
    constant MRQueryParams& p       [[buffer(10)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= p.n) return;

    PointNormal rejected;
    rejected.x = 0; rejected.y = 0; rejected.z = 0; rejected.intensity = 0;
    rejected.nx = 0; rejected.ny = 0; rejected.nz = 0; rejected._pad = 0;
    out_buff[idx] = rejected;

    Point pt = cloud[idx];
    float3 local3 = float3(pt.x, pt.y, pt.z);
    float4 ext_q = float4(p.lidar_to_imu_quat[0], p.lidar_to_imu_quat[1], p.lidar_to_imu_quat[2], p.lidar_to_imu_quat[3]);
    float3 ext_t = float3(p.lidar_to_imu_pos[0], p.lidar_to_imu_pos[1], p.lidar_to_imu_pos[2]);
    float4 q = float4(p.pose_quat[0], p.pose_quat[1], p.pose_quat[2], p.pose_quat[3]);
    float3 t = float3(p.pose_trans[0], p.pose_trans[1], p.pose_trans[2]);
    float3 imu = iso_transform(ext_q, ext_t, local3);
    float3 world3 = iso_transform(q, t, imu);
    if (any(isnan(world3)) || any(isinf(world3))) return;
    float world[3] = {world3.x, world3.y, world3.z};
    float local[3] = {local3.x, local3.y, local3.z};

    MRFit selected;
    selected.status = 0;

    for (uint l = 0; l < 4; l++) {
        float r = p.resolution * (.5 * float(1u << l));
        int key[3];
        mr_grid(world, r, key);
        uint n = p.level_sizes[l];

        int cell = -1;
        if (l == 0) cell = mr_find_cell(cells0, n, key);
        else if (l == 1) cell = mr_find_cell(cells1, n, key);
        else if (l == 2) cell = mr_find_cell(cells2, n, key);
        else cell = mr_find_cell(cells3, n, key);

        MRFit f;
        if (cell >= 0) {
            if (l == 0) f = fits0[cell];
            else if (l == 1) f = fits1[cell];
            else if (l == 2) f = fits2[cell];
            else f = fits3[cell];
        } else {
            if (l == 0) f = mr_neighborhood(cells0, n, key, p.resolution);
            else if (l == 1) f = mr_neighborhood(cells1, n, key, p.resolution);
            else if (l == 2) f = mr_neighborhood(cells2, n, key, p.resolution);
            else f = mr_neighborhood(cells3, n, key, p.resolution);
        }

        if (f.status == 0) continue;
        if (f.status == 1) break;

        float delta[3], t0[3], t1[3];
        for (int k = 0; k < 3; k++) {
            delta[k] = world[k] - f.mean[k];
            t0[k] = f.tangent[k];
            t1[k] = f.tangent[k + 3];
        }
        float proj_t0 = delta[0] * t0[0] + delta[1] * t0[1] + delta[2] * t0[2];
        float proj_t1 = delta[0] * t1[0] + delta[1] * t1[1] + delta[2] * t1[2];
        if (fabs(proj_t0) > f.extent[0] || fabs(proj_t1) > f.extent[1]) continue;

        if (selected.status == 2) {
            float cosine = min(fabs(selected.normal[0] * f.normal[0] + selected.normal[1] * f.normal[1] +
                                     selected.normal[2] * f.normal[2]), 1.0f);
            float d2[3];
            for (int k = 0; k < 3; k++) d2[k] = f.mean[k] - selected.mean[k];
            float dn_sel = selected.normal[0] * d2[0] + selected.normal[1] * d2[1] + selected.normal[2] * d2[2];
            float dn_f = f.normal[0] * d2[0] + f.normal[1] * d2[1] + f.normal[2] * d2[2];
            if (cosine < .98 || 1.0 - cosine * cosine > selected.quality + f.quality ||
                fabs(dn_sel) > p.resolution * .15 || fabs(dn_f) > p.resolution * .15)
                break;
            if (f.quality >= selected.quality) continue;
        }
        selected = f;
    }

    if (selected.status != 2) return;
    float distance = selected.normal[0] * world[0] + selected.normal[1] * world[1] +
        selected.normal[2] * world[2] + selected.d;
    float l2 = local[0] * local[0] + local[1] * local[1] + local[2] * local[2];
    float s = 1.0 - .9 * fabs(distance) / sqrt(sqrt(max(l2, 1e-9f)));
    if (s <= .9) return;

    PointNormal outp;
    outp.x = pt.x; outp.y = pt.y; outp.z = pt.z;
    outp.intensity = selected.d;
    outp.nx = selected.normal[0]; outp.ny = selected.normal[1]; outp.nz = selected.normal[2];
    outp._pad = 0;
    out_buff[idx] = outp;
}

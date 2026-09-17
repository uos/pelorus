// Must match `pelorus_metal::multiresolution::Cell`.
struct MRCell {
    int key[3];
    uint scan;
    ulong count;
    float mean[3];
    float cov[9];
};

// Must match `pelorus_metal::multiresolution::Fit`.
struct MRFit {
    uint status;
    float mean[3];
    float normal[3];
    float tangent[6];
    float extent[2];
    float quality;
    float d;
};

// Must match `pelorus_metal::multiresolution::QueryParams`.
struct MRQueryParams {
    float lidar_to_imu_quat[4];
    float lidar_to_imu_pos[3];
    float _pad0;
    float pose_quat[4];
    float pose_trans[3];
    float _pad1;
    float resolution;
    uint table_cap;
    uint n;
    uint level_sizes[4];
    uint fitcache_cap;
    uint fitcache_gen;
    uint fitcache_epoch;
    uint _pad2;
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

#define MR_EMPTY_SLOT 0xffffffffu

inline uint mr_slot(thread const int* key, uint cap) {
    return hash_fn(hash_grid(int3(key[0], key[1], key[2]))) & (cap - 1u);
}

// Locate a cell by key through a per-level open-addressing index.
//
// This replaced a binary search over the key-sorted cell array. The search
// was correct but pathologically slow here: a scan is only ~5k points, far
// too few to fill the GPU, so `mr_query` is bound by the latency of
// dependent random loads rather than by throughput
inline int mr_find_cell(device const MRCell* cells, device const uint* table, uint cap,
                        thread const int* key) {
    uint slot = mr_slot(key, cap);
    for (uint probe = 0; probe < cap; ++probe) {
        uint v = table[slot];
        if (v == MR_EMPTY_SLOT) return -1;
        if (mr_key_equal(cells[v].key, key)) return int(v);
        slot = (slot + 1u) & (cap - 1u);
    }
    return -1;
}

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
inline MRFit mr_fit(thread const MRCell& cell, float resolution) {
    MRFit f;
    f.status = 0;
    for (int i = 0; i < 3; i++) { f.mean[i] = 0; f.normal[i] = 0; }
    for (int i = 0; i < 6; i++) f.tangent[i] = 0;
    f.extent[0] = 0; f.extent[1] = 0; f.quality = 0; f.d = 0;
    if (cell.count < 6) return f;
    f.status = 1;

    for (int i = 0; i < 9; i++) {
        float x = cell.cov[i];
        if (isnan(x) || isinf(x)) return f;
    }

    float3 ev = symmetric_eigen_3x3(cell.cov);
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

    float3 normal = mr_eigenvector(cell.cov, val[idx[0]]);
    float3 tangent0_raw = mr_eigenvector(cell.cov, val[idx[1]]);
    // Re-orthogonalize against `normal`. The two null-space directions are
    // only exactly orthogonal for an exact shift, so guard numerical drift
    // and near-degenerate (repeated-eigenvalue) cases before completing the
    // right-handed basis.
    float3 t0 = tangent0_raw - normal * dot(tangent0_raw, normal);
    float t0len = length(t0);
    // Choose the least-aligned axis: crossing a horizontal plane's normal
    // with z would produce zero and normalize it into NaNs.
    float3 nabs = abs(normal);
    float3 axis = nabs.x <= nabs.y && nabs.x <= nabs.z ? float3(1, 0, 0)
        : nabs.y <= nabs.z ? float3(0, 1, 0) : float3(0, 0, 1);
    float3 tangent0 = t0len > 1e-6f ? t0 / t0len : normalize(cross(normal, axis));
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
inline MRFit mr_neighborhood(device const MRCell* cells, device const uint* table, uint cap,
                             uint n, thread const int* key, float resolution) {
    MRCell pooled;
    mr_empty_cell(pooled);

    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++) {
            int idx = -1;
            for (int dz = -1; dz <= 1; dz++) {
                int k[3] = {mr_neighbor(key[0], dx), mr_neighbor(key[1], dy), mr_neighbor(key[2], dz)};
                if (idx >= 0 && uint(idx) + 1u < n && mr_key_equal(cells[idx + 1].key, k)) {
                    idx = idx + 1;
                } else {
                    idx = mr_find_cell(cells, table, cap, k);
                }
                if (idx >= 0) mr_pool(pooled, cells[idx]);
            }
        }
    return mr_fit(pooled, resolution);
}

// Must match `pelorus_metal::multiresolution::CacheParams` exactly (16 bytes).
struct MRCacheParams {
    uint n;
    float resolution;
    uint cap;
    uint _pad;
};

constant uint MR_FC_FREE    = 0u;
constant uint MR_FC_WRITING = 1u;
constant uint MR_FC_GEN_BIAS = 2u;

inline uint mr_fc_slot(uint level, thread const int* key, uint cap) {
    uint h = hash_fn(hash_grid(int3(key[0], key[1], key[2])));
    return (h * 2654435761u + level * 2246822519u) & (cap - 1u);
}

inline bool mr_fc_key_eq(device const int* k, uint level, thread const int* key) {
    return k[0] == int(level) && k[1] == key[0] && k[2] == key[1] && k[3] == key[2];
}

kernel void mr_table_init(
    device uint*             table [[buffer(0)]],
    constant MRCacheParams&  p     [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (i < p.cap) table[i] = MR_EMPTY_SLOT;
}

kernel void mr_table_build(
    device atomic_uint*      table [[buffer(0)]],
    device const MRCell*     cells [[buffer(1)]],
    constant MRCacheParams&  p     [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n) return;
    int key[3] = {cells[i].key[0], cells[i].key[1], cells[i].key[2]};
    uint slot = mr_slot(key, p.cap);
    for (uint probe = 0; probe < p.cap; ++probe) {
        uint expected = MR_EMPTY_SLOT;
        if (atomic_compare_exchange_weak_explicit(&table[slot], &expected, i,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) {
            return;
        }
        slot = (slot + 1u) & (p.cap - 1u);
    }
}

kernel void mr_cache_fits(
    device const MRCell*     cells [[buffer(0)]],
    device MRFit*            fits  [[buffer(1)]],
    device const uint*       table [[buffer(2)]],
    constant MRCacheParams&  p     [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n) return;
    // mr_neighborhood takes a thread-address-space key.
    int key[3] = {cells[i].key[0], cells[i].key[1], cells[i].key[2]};
    fits[i] = mr_neighborhood(cells, table, p.cap, p.n, key, p.resolution);
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
    device const uint*     table0   [[buffer(9)]],
    device const uint*     table1   [[buffer(10)]],
    device const uint*     table2   [[buffer(11)]],
    device const uint*     table3   [[buffer(12)]],
    device atomic_uint*    fc_flags [[buffer(13)]],
    device int*            fc_keys  [[buffer(14)]],
    device MRFit*          fc_fits  [[buffer(15)]],
    device PointNormal*    out_buff [[buffer(16)]],
    constant MRQueryParams& p       [[buffer(17)]],
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
        uint cap = p.table_cap;
        uint ln = p.level_sizes[l];
        
        // Registration queries before the first scan is added, so a level that
        // has no cells yet also has a hash table that `mr_table_init` has never
        // touched. Probing it would read whatever the allocator returned and
        // could report a cell that does not exist.
        if (ln == 0) continue;

        int cell = -1;
        if (l == 0) cell = mr_find_cell(cells0, table0, cap, key);
        else if (l == 1) cell = mr_find_cell(cells1, table1, cap, key);
        else if (l == 2) cell = mr_find_cell(cells2, table2, cap, key);
        else cell = mr_find_cell(cells3, table3, cap, key);

        MRFit f;
        if (cell >= 0) {
            if (l == 0) f = fits0[cell];
            else if (l == 1) f = fits1[cell];
            else if (l == 2) f = fits2[cell];
            else f = fits3[cell];
        } else {
            uint fcap = p.fitcache_cap;
            bool hit = false;
            uint free_slot = 0xffffffffu;
            if (fcap > 0u) {
                uint slot = mr_fc_slot(l, key, fcap);
                for (uint probe = 0; probe < 16u; ++probe) {
                    uint fl = atomic_load_explicit(&fc_flags[slot], memory_order_relaxed);
                    if (fl == MR_FC_FREE) {
                        if (free_slot == 0xffffffffu) free_slot = slot;
                        break;
                    }
                    if (fl != MR_FC_WRITING) {
                        uint gen = fl - MR_FC_GEN_BIAS;
                        if (gen < p.fitcache_epoch) {
                            // Left over from an earlier scan: reusable, but
                            // keep probing in case this key is further along.
                            if (free_slot == 0xffffffffu) free_slot = slot;
                        } else if (gen < p.fitcache_gen
                                   && mr_fc_key_eq(&fc_keys[slot * 4u], l, key)) {
                            f = fc_fits[slot];
                            hit = true;
                            break;
                        }
                    }
                    slot = (slot + 1u) & (fcap - 1u);
                }
            }
            if (!hit) {
                if (l == 0) f = mr_neighborhood(cells0, table0, cap, ln, key, p.resolution);
                else if (l == 1) f = mr_neighborhood(cells1, table1, cap, ln, key, p.resolution);
                else if (l == 2) f = mr_neighborhood(cells2, table2, cap, ln, key, p.resolution);
                else f = mr_neighborhood(cells3, table3, cap, ln, key, p.resolution);
                if (free_slot != 0xffffffffu) {
                    uint expected = atomic_load_explicit(&fc_flags[free_slot],
                                                         memory_order_relaxed);
                    bool claimable = expected == MR_FC_FREE
                        || (expected != MR_FC_WRITING
                            && expected - MR_FC_GEN_BIAS < p.fitcache_epoch);
                    if (claimable && atomic_compare_exchange_weak_explicit(
                            &fc_flags[free_slot], &expected, MR_FC_WRITING,
                            memory_order_relaxed, memory_order_relaxed)) {
                        fc_keys[free_slot * 4u + 0u] = int(l);
                        fc_keys[free_slot * 4u + 1u] = key[0];
                        fc_keys[free_slot * 4u + 2u] = key[1];
                        fc_keys[free_slot * 4u + 3u] = key[2];
                        fc_fits[free_slot] = f;
                        // Tagged with this dispatch's generation, so only a
                        // later dispatch will read it.
                        atomic_store_explicit(&fc_flags[free_slot],
                                              p.fitcache_gen + MR_FC_GEN_BIAS,
                                              memory_order_relaxed);
                    }
                }
            }
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

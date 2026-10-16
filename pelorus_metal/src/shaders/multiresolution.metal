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
    uint miss_cap;
    uint miss_version;
    uint work_cap;
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
// status 0 = insufficient support, 1 = conflicting/degenerate, 2 = usable
// plane. The caller moves on to the next coarser level unless the status is 2.
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
// `prev_n` is only read by `mr_refresh_fits`.
struct MRCacheParams {
    uint n;
    float resolution;
    uint cap;
    uint prev_n;
};

// Miss cache: neighborhood fits of cells that are absent from a level, valid
// for one map version. Mirrors the CUDA backend. Per slot, `miss_state` holds
// version << 1 | ready; a slot of an older version counts as empty, so the
// cache never has to be cleared. `miss_keys` holds (level, x, y, z).
constant uint MR_MISS_PROBES = 32u;

// Must match `pelorus_metal::multiresolution::MissWork` (20 bytes).
struct MRMissWork {
    uint level;
    uint slot;
    int key[3];
};

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

// A fit only depends on the 27 cells around its key. Marks every cell of the
// new level version next to a key that was updated, inserted or evicted in
// this scan. `changed` holds 4 ints (x, y, z, unused) per key; p.n = #keys.
kernel void mr_mark_dirty(
    device const int*        changed [[buffer(0)]],
    device const MRCell*     cells   [[buffer(1)]],
    device const uint*       table   [[buffer(2)]],
    device uint*             dirty   [[buffer(3)]],
    constant MRCacheParams&  p       [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n) return;
    int key[3] = {changed[i * 4u + 0u], changed[i * 4u + 1u], changed[i * 4u + 2u]};
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++) {
                int k[3] = {mr_neighbor(key[0], dx), mr_neighbor(key[1], dy), mr_neighbor(key[2], dz)};
                int j = mr_find_cell(cells, table, p.cap, k);
                if (j >= 0) dirty[j] = 1u;
            }
}

// Carries unchanged fits over from the previous level version and refits the
// rest. Gives the same fits as fitting every cell of the new version.
kernel void mr_refresh_fits(
    device const MRCell*     prev_cells [[buffer(0)]],
    device const MRFit*      prev_fits  [[buffer(1)]],
    device const uint*       prev_table [[buffer(2)]],
    device const MRCell*     cells      [[buffer(3)]],
    device MRFit*            fits       [[buffer(4)]],
    device const uint*       table      [[buffer(5)]],
    device const uint*       dirty      [[buffer(6)]],
    constant MRCacheParams&  p          [[buffer(7)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n) return;
    int key[3] = {cells[i].key[0], cells[i].key[1], cells[i].key[2]};
    if (dirty[i] == 0u && p.prev_n > 0u) {
        int j = mr_find_cell(prev_cells, prev_table, p.cap, key);
        if (j >= 0) {
            fits[i] = prev_fits[j];
            return;
        }
    }
    fits[i] = mr_neighborhood(cells, table, p.cap, p.n, key, p.resolution);
}

// Miss pass 1: claims a miss-cache slot for every absent cell a point could
// query on any level and lists it once, so pass 2 computes each fit a single
// time. The point transform and level search mirror `mr_query` exactly.
kernel void mr_collect_misses(
    device const Point*    cloud      [[buffer(0)]],
    device const MRCell*   cells0     [[buffer(1)]],
    device const MRCell*   cells1     [[buffer(2)]],
    device const MRCell*   cells2     [[buffer(3)]],
    device const MRCell*   cells3     [[buffer(4)]],
    device const uint*     table0     [[buffer(5)]],
    device const uint*     table1     [[buffer(6)]],
    device const uint*     table2     [[buffer(7)]],
    device const uint*     table3     [[buffer(8)]],
    device atomic_uint*    miss_state [[buffer(9)]],
    device int*            miss_keys  [[buffer(10)]],
    device MRMissWork*     work       [[buffer(11)]],
    device atomic_uint*    work_count [[buffer(12)]],
    constant MRQueryParams& p         [[buffer(13)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= p.n || p.miss_cap == 0u) return;

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

    for (uint l = 0; l < 4; l++) {
        float r = p.resolution * (.5 * float(1u << l));
        int key[3];
        mr_grid(world, r, key);
        uint cap = p.table_cap;
        if (p.level_sizes[l] == 0) continue;

        int cell = -1;
        if (l == 0) cell = mr_find_cell(cells0, table0, cap, key);
        else if (l == 1) cell = mr_find_cell(cells1, table1, cap, key);
        else if (l == 2) cell = mr_find_cell(cells2, table2, cap, key);
        else cell = mr_find_cell(cells3, table3, cap, key);
        if (cell >= 0) continue;

        uint slot = mr_fc_slot(l, key, p.miss_cap);
        for (uint probe = 0; probe < MR_MISS_PROBES; ++probe) {
            uint st = atomic_load_explicit(&miss_state[slot], memory_order_relaxed);
            if ((st >> 1) != p.miss_version) {
                // Empty for this map version: claim it, still not ready.
                if (atomic_compare_exchange_weak_explicit(&miss_state[slot], &st,
                                                          p.miss_version << 1,
                                                          memory_order_relaxed,
                                                          memory_order_relaxed)) {
                    miss_keys[slot * 4u + 0u] = int(l);
                    miss_keys[slot * 4u + 1u] = key[0];
                    miss_keys[slot * 4u + 2u] = key[1];
                    miss_keys[slot * 4u + 3u] = key[2];
                    uint w = atomic_fetch_add_explicit(work_count, 1u, memory_order_relaxed);
                    // A full list leaves the slot unready; mr_query then fits inline.
                    if (w < p.work_cap) {
                        MRMissWork item;
                        item.level = l;
                        item.slot = slot;
                        item.key[0] = key[0]; item.key[1] = key[1]; item.key[2] = key[2];
                        work[w] = item;
                    }
                    break;
                }
                continue; // lost the race for this slot, look at it again
            }
            // Claimed in this version. A slot claimed in this dispatch may
            // still show an older key, which at worst lists a key twice or
            // leaves it to mr_query's inline fallback.
            if (mr_fc_key_eq(&miss_keys[slot * 4u], l, key)) break;
            slot = (slot + 1u) & (p.miss_cap - 1u);
        }
    }
}

// Miss pass 2: one fit per listed absent cell.
kernel void mr_compute_misses(
    device const MRMissWork* work       [[buffer(0)]],
    device const uint*       work_count [[buffer(1)]],
    device const MRCell*     cells0     [[buffer(2)]],
    device const MRCell*     cells1     [[buffer(3)]],
    device const MRCell*     cells2     [[buffer(4)]],
    device const MRCell*     cells3     [[buffer(5)]],
    device const uint*       table0     [[buffer(6)]],
    device const uint*       table1     [[buffer(7)]],
    device const uint*       table2     [[buffer(8)]],
    device const uint*       table3     [[buffer(9)]],
    device atomic_uint*      miss_state [[buffer(10)]],
    device MRFit*            miss_fits  [[buffer(11)]],
    constant MRQueryParams&  p          [[buffer(12)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= min(*work_count, p.work_cap)) return;
    MRMissWork w = work[i];
    int key[3] = {w.key[0], w.key[1], w.key[2]};
    uint cap = p.table_cap;
    uint ln = p.level_sizes[w.level];
    MRFit f;
    if (w.level == 0) f = mr_neighborhood(cells0, table0, cap, ln, key, p.resolution);
    else if (w.level == 1) f = mr_neighborhood(cells1, table1, cap, ln, key, p.resolution);
    else if (w.level == 2) f = mr_neighborhood(cells2, table2, cap, ln, key, p.resolution);
    else f = mr_neighborhood(cells3, table3, cap, ln, key, p.resolution);
    miss_fits[w.slot] = f;
    atomic_store_explicit(&miss_state[w.slot], (p.miss_version << 1) | 1u, memory_order_relaxed);
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
    device atomic_uint*    miss_state [[buffer(13)]],
    device const int*      miss_keys  [[buffer(14)]],
    device const MRFit*    miss_fits  [[buffer(15)]],
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
            // Read the fit of the absent cell from the miss cache filled by
            // mr_collect_misses/mr_compute_misses; fit inline if not cached,
            // so the result is always exact.
            bool hit = false;
            uint mcap = p.miss_cap;
            if (mcap > 0u) {
                uint slot = mr_fc_slot(l, key, mcap);
                for (uint probe = 0; probe < MR_MISS_PROBES; ++probe) {
                    uint st = atomic_load_explicit(&miss_state[slot], memory_order_relaxed);
                    // Slots are claimed front to back along the probe chain,
                    // so an empty one ends the search.
                    if ((st >> 1) != p.miss_version) break;
                    if ((st & 1u) != 0u && mr_fc_key_eq(&miss_keys[slot * 4u], l, key)) {
                        f = miss_fits[slot];
                        hit = true;
                        break;
                    }
                    slot = (slot + 1u) & (mcap - 1u);
                }
            }
            if (!hit) {
                if (l == 0) f = mr_neighborhood(cells0, table0, cap, ln, key, p.resolution);
                else if (l == 1) f = mr_neighborhood(cells1, table1, cap, ln, key, p.resolution);
                else if (l == 2) f = mr_neighborhood(cells2, table2, cap, ln, key, p.resolution);
                else f = mr_neighborhood(cells3, table3, cap, ln, key, p.resolution);
            }
        }

        if (f.status != 2) continue;

        float delta[3], t0[3], t1[3];
        for (int k = 0; k < 3; k++) {
            delta[k] = world[k] - f.mean[k];
            t0[k] = f.tangent[k];
            t1[k] = f.tangent[k + 3];
        }
        float proj_t0 = delta[0] * t0[0] + delta[1] * t0[1] + delta[2] * t0[2];
        float proj_t1 = delta[0] * t1[0] + delta[1] * t1[1] + delta[2] * t1[2];
        if (fabs(proj_t0) > f.extent[0] || fabs(proj_t1) > f.extent[1]) continue;

        // The finest level with a valid plane supplies the residual.
        selected = f;
        break;
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

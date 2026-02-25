// localmap.metal — build_voxels kernel: one thread per unique voxel group.
// CPU pre-sorts points by voxel key and provides run_starts/run_ends.
// Included after hashmap.metal in the compiled source.

// Params for build_voxels, passed as a single constant buffer to minimise bindings.
struct BuildVoxelsParams {
    float4 pose_quat;      // quaternion (x,y,z,w) of the pose
    float3 pose_trans;     // translation of the pose
    float  resolution;
    uint   scan_number;
    float  sensitivity;    // unused in this simplified port
    float  min_dist;       // unused in this simplified port
    uint   unique_key_count;
    uint   table_cap;      // hash table capacity = dist_cap * 2
    uint   dist_cap;       // distribution buffer capacity
};

// build_voxels: one thread per unique voxel.
// Sorted_indices[run_starts[idx]..run_ends[idx]] contains the original cloud
// indices for all points that belong to this voxel.
kernel void build_voxels(
    device const Point*          cloud          [[buffer(0)]],
    device const uint*           sorted_indices [[buffer(1)]],
    device const uint*           run_starts     [[buffer(2)]],
    device const uint*           run_ends       [[buffer(3)]],
    device const int*            unique_keys    [[buffer(4)]],
    device atomic_int*           table_keys     [[buffer(5)]],
    device uint*                 table_vals     [[buffer(6)]],
    device VoxelDistribution*    dists          [[buffer(7)]],
    device int*                  dist_keys      [[buffer(8)]],  // companion: dist key per slot
    device atomic_uint*          add_cursor     [[buffer(9)]],
    constant BuildVoxelsParams&  params         [[buffer(10)]],
    uint                         idx            [[thread_position_in_grid]])
{
    if (idx >= params.unique_key_count) return;

    int   voxel_key  = unique_keys[idx];
    uint  start      = run_starts[idx];
    uint  end        = run_ends[idx];
    uint  count      = end - start;

    if (count == 0 || voxel_key == EMPTY_KEY) return;

    // ── 1. Compute mean and covariance for this voxel's points ──────────────
    float3 sum = float3(0, 0, 0);
    for (uint i = start; i < end; i++) {
        uint pi = sorted_indices[i];
        float3 p_local  = float3(cloud[pi].x, cloud[pi].y, cloud[pi].z);
        float3 p_global = iso_transform(params.pose_quat, params.pose_trans, p_local);
        sum += p_global;
    }
    float  fn     = float(count);
    float3 mean_v = sum / fn;

    // Covariance (scatter / n)
    float cov[9] = { 0,0,0, 0,0,0, 0,0,0 };
    for (uint i = start; i < end; i++) {
        uint pi = sorted_indices[i];
        float3 p_local  = float3(cloud[pi].x, cloud[pi].y, cloud[pi].z);
        float3 p_global = iso_transform(params.pose_quat, params.pose_trans, p_local);
        float d[3] = { p_global.x - mean_v.x,
                       p_global.y - mean_v.y,
                       p_global.z - mean_v.z };
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                cov[r * 3 + c] += d[r] * d[c];
    }
    for (int i = 0; i < 9; i++) cov[i] /= fn;

    // ── 2. Compute Cholesky factor L ─────────────────────────────────────────
    float L[9] = { 0,0,0, 0,0,0, 0,0,0 };
    bool chol_ok = cholesky_3x3(cov, L);
    if (!chol_ok) {
        // Fallback: diagonal from eigenvalues
        float3 ev = symmetric_eigen_3x3(cov);
        L[0] = sqrt(max(0.0f, ev.x));
        L[4] = sqrt(max(0.0f, ev.y));
        L[8] = sqrt(max(0.0f, ev.z));
    }

    // ── 3. Look up or insert into the hash table ─────────────────────────────
    uint table_cap = params.table_cap;
    uint slot = hash_fn(voxel_key) & (table_cap - 1);
    uint buf_idx = UINT_MAX;
    bool found_existing = false;

    for (;;) {
        int cur = atomic_load_explicit(&table_keys[slot], memory_order_relaxed);
        if (cur == voxel_key) {
            buf_idx = table_vals[slot];
            found_existing = true;
            break;
        }
        if (cur == EMPTY_KEY) {
            // Try to claim this slot
            int expected = EMPTY_KEY;
            bool claimed = atomic_compare_exchange_weak_explicit(
                &table_keys[slot], &expected, voxel_key,
                memory_order_relaxed, memory_order_relaxed);
            if (claimed) {
                // Claim a distribution buffer slot
                uint new_idx = atomic_fetch_add_explicit(add_cursor, 1u, memory_order_relaxed);
                if (new_idx >= params.dist_cap) {
                    // Buffer full — overwrite the oldest (LRU not implemented: just wrap)
                    new_idx = new_idx % params.dist_cap;
                }
                table_vals[slot] = new_idx;
                buf_idx = new_idx;
                found_existing = false;
                break;
            }
            // Someone else claimed it; retry this slot
            continue;
        }
        slot = (slot + 1) & (table_cap - 1);
    }

    if (buf_idx == UINT_MAX) return;

    // ── 4. Write the distribution to the buffer ───────────────────────────────
    device VoxelDistribution& d = dists[buf_idx];
    dist_keys[buf_idx] = voxel_key;

    if (found_existing && d.point_count > 0) {
        // Merge: weighted update with the new scan's distribution
        // Simple approach: pool (Welford parallel combination)
        uint n1 = d.point_count;
        uint n2 = count;
        uint n_total = n1 + n2;
        float fn1 = float(n1), fn2 = float(n2), fn_t = float(n_total);

        float3 old_mean = float3(d.mean[0], d.mean[1], d.mean[2]);
        float3 delta    = mean_v - old_mean;
        float3 new_mean = old_mean + delta * (fn2 / fn_t);

        float w = fn1 * fn2 / fn_t;
        float dv[3] = { delta.x, delta.y, delta.z };

        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                float s1 = d.cov[r*3+c] * (fn1 > 1.0 ? (fn1-1.0) : 0.0);
                float s2 = cov[r*3+c]   * (fn2 > 1.0 ? (fn2-1.0) : 0.0);
                d.cov[r*3+c] = (s1 + s2 + dv[r]*dv[c]*w)
                               / (fn_t > 1.0 ? (fn_t-1.0) : 1.0);
            }
        }

        d.mean[0] = new_mean.x;
        d.mean[1] = new_mean.y;
        d.mean[2] = new_mean.z;
        d.point_count = n_total;
    } else {
        // Fresh entry
        d.mean[0] = mean_v.x;
        d.mean[1] = mean_v.y;
        d.mean[2] = mean_v.z;
        d.point_count = count;
        for (int i = 0; i < 9; i++) d.cov[i] = cov[i];
    }

    for (int i = 0; i < 9; i++) d.L[i] = L[i];
    d.last_scan = params.scan_number;
}

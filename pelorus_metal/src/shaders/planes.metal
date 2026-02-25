// planes.metal — per-point plane fitting via voxel distribution lookup.
// Included after localmap.metal in the compiled source.

struct PlanesParams {
    float4 lidar_to_imu_quat; // (x,y,z,w)
    float3 lidar_to_imu_pos;
    float  _pad0;
    float4 pose_quat;         // (x,y,z,w)
    float3 pose_pos;
    float  _pad1;
    uint   cloud_size;
    uint   table_cap;
    float  resolution;
    float  sphericity_threshold;  // typically 0.5
};

// Helper: plane normal–point distance (signed)
inline float plane_point_dist(float4 plane, float3 p) {
    return plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w;
}

// Fit a plane from a merged VoxelDistribution.
// Returns true on success; sets plane_eq = (nx,ny,nz,d) such that n·p+d=0.
inline bool plane_fit(thread VoxelDistribution& merged,
                      float sphericity_threshold,
                      thread float4& plane_eq) {
    if (merged.point_count < 3) return false;

    float3 ev = symmetric_eigen_3x3(merged.cov);

    // Sort eigenvalues ascending
    float v0 = ev.x, v1 = ev.y, v2 = ev.z;
    if (v0 > v1) { float t = v0; v0 = v1; v1 = t; }
    if (v1 > v2) { float t = v1; v1 = v2; v2 = t; }
    if (v0 > v1) { float t = v0; v0 = v1; v1 = t; }

    float lambda_0 = v0; // smallest
    float lambda_2 = v2; // largest

    if (lambda_2 < 1e-9) return false;
    if (lambda_0 / lambda_2 > sphericity_threshold) return false;

    // Eigenvector for lambda_0: null space of (cov - lambda_0 * I)
    float M[9];
    for (int i = 0; i < 9; i++) M[i] = merged.cov[i];
    M[0] -= lambda_0; M[4] -= lambda_0; M[8] -= lambda_0;

    float3 row0 = float3(M[0], M[1], M[2]);
    float3 row1 = float3(M[3], M[4], M[5]);
    float3 row2 = float3(M[6], M[7], M[8]);

    float3 v01 = cross(row0, row1);
    float3 v12 = cross(row1, row2);
    float3 v20 = cross(row2, row0);

    float m01 = dot(v01, v01);
    float m12 = dot(v12, v12);
    float m20 = dot(v20, v20);

    float3 normal;
    if (m01 > m12 && m01 > m20)      normal = v01;
    else if (m12 > m20)               normal = v12;
    else                              normal = v20;

    float mag = length(normal);
    if (mag < 1e-9) return false;
    normal /= mag;

    float3 mean_p = float3(merged.mean[0], merged.mean[1], merged.mean[2]);
    float  d = -(normal.x * mean_p.x + normal.y * mean_p.y + normal.z * mean_p.z);
    plane_eq = float4(normal, d);
    return true;
}

kernel void plane_krnl(
    device const Point*          cloud      [[buffer(0)]],
    device const int*            table_keys [[buffer(1)]],
    device const uint*           table_vals [[buffer(2)]],
    device const VoxelDistribution* dists   [[buffer(3)]],
    device atomic_uint*          out_size   [[buffer(4)]],
    device PointNormal*          out_buff   [[buffer(5)]],
    constant PlanesParams&       params     [[buffer(6)]],
    uint                         idx        [[thread_position_in_grid]])
{
    if (idx >= params.cloud_size) return;

    Point p_raw = cloud[idx];
    float3 p_local  = float3(p_raw.x, p_raw.y, p_raw.z);

    // Transform: local → IMU → global
    float3 p_imu    = iso_transform(params.lidar_to_imu_quat,
                                    params.lidar_to_imu_pos, p_local);
    float3 p_global = iso_transform(params.pose_quat,
                                    params.pose_pos, p_imu);

    // Collect 27 neighbor voxels
    int3 grid = pos2grid(params.resolution, p_global);
    int3 neighbors[27];
    uint nb_count = neighbor_cells(26, grid, neighbors);

    // Pool distributions from found neighbors
    VoxelDistribution merged;
    merged.point_count = 0;
    for (int i = 0; i < 9; i++) { merged.cov[i] = 0.0; merged.L[i] = 0.0; }
    merged.mean[0] = merged.mean[1] = merged.mean[2] = 0.0;

    for (uint ni = 0; ni < nb_count; ni++) {
        int nb_key = hash_grid(neighbors[ni]);
        uint buf_idx = hashmap_lookup(table_keys, table_vals,
                                      params.table_cap, nb_key);
        if (buf_idx == UINT_MAX) continue;

        device const VoxelDistribution& nd = dists[buf_idx];
        if (nd.point_count == 0) continue;

        // Welford parallel pool
        uint n1 = merged.point_count;
        uint n2 = nd.point_count;
        if (n1 == 0) {
            merged.point_count = n2;
            merged.mean[0] = nd.mean[0];
            merged.mean[1] = nd.mean[1];
            merged.mean[2] = nd.mean[2];
            for (int i = 0; i < 9; i++) merged.cov[i] = nd.cov[i];
            continue;
        }
        uint  n_t = n1 + n2;
        float fn1 = float(n1), fn2 = float(n2), fn_t = float(n_t);

        float3 old_mean = float3(merged.mean[0], merged.mean[1], merged.mean[2]);
        float3 nd_mean  = float3(nd.mean[0], nd.mean[1], nd.mean[2]);
        float3 delta    = nd_mean - old_mean;
        float3 new_mean = old_mean + delta * (fn2 / fn_t);

        float w = fn1 * fn2 / fn_t;
        float dv[3] = { delta.x, delta.y, delta.z };
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                float s1 = merged.cov[r*3+c] * (fn1 > 1.0 ? (fn1-1.0) : 0.0);
                float s2 = nd.cov[r*3+c]     * (fn2 > 1.0 ? (fn2-1.0) : 0.0);
                merged.cov[r*3+c] = (s1 + s2 + dv[r]*dv[c]*w)
                                    / (fn_t > 1.0 ? (fn_t-1.0) : 1.0);
            }
        }
        merged.mean[0] = new_mean.x;
        merged.mean[1] = new_mean.y;
        merged.mean[2] = new_mean.z;
        merged.point_count = n_t;
    }

    // Fit plane
    float4 plane;
    if (!plane_fit(merged, params.sphericity_threshold, plane)) return;

    // Score filter (same formula as CUDA)
    float pd2 = plane_point_dist(plane, p_global);
    float l2  = sqrt(p_local.x*p_local.x + p_local.y*p_local.y + p_local.z*p_local.z);
    float s   = 1.0 - 0.9 * fabs(pd2) / max(l2, 1e-9f);
    if (s <= 0.9) return;

    // Atomic output write
    uint out_idx = atomic_fetch_add_explicit(out_size, 1u, memory_order_relaxed);

    PointNormal pn;
    pn.x         = p_local.x;
    pn.y         = p_local.y;
    pn.z         = p_local.z;
    pn.intensity = pd2;
    pn.nx        = plane.x;
    pn.ny        = plane.y;
    pn.nz        = plane.z;
    pn._pad      = 0.0;

    out_buff[out_idx] = pn;
}

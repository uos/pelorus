// planes.metal — per-point plane fitting via voxel distribution lookup.
// Included after localmap.metal in the compiled source.

struct PlanesParams {
    float lidar_to_imu_quat[4]; // (x,y,z,w)
    float lidar_to_imu_pos[3];
    float _pad0;
    float pose_quat[4];         // (x,y,z,w)
    float pose_pos[3];
    float _pad1;
    uint  cloud_size;
    uint  table_cap;
    float resolution;
    float sphericity_threshold;  // typically 0.5
    uint  _pad2;
};

// Helper: plane normal–point distance (signed)
inline float plane_point_dist(float4 plane, float3 p) {
    return plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w;
}

// Fit a plane from a merged accumulation.
// Returns true on success; sets plane_eq = (nx,ny,nz,d) such that n·p+d=0.
inline bool plane_fit(uint point_count, thread const float* mean, thread const float* cov,
                      float sphericity_threshold,
                      thread float4& plane_eq) {
    if (point_count < 3) return false;

    float3 ev = symmetric_eigen_3x3(cov);

    // Sort eigenvalues ascending
    float v0 = ev.x, v1 = ev.y, v2 = ev.z;
    if (v0 > v1) { float t = v0; v0 = v1; v1 = t; }
    if (v1 > v2) { float t = v1; v1 = v2; v2 = t; }
    if (v0 > v1) { float t = v0; v0 = v1; v1 = t; }

    const float lambda_0 = v0; // smallest
    const float lambda_2 = v2; // largest

    if (lambda_2 < 1e-9) return false;
    if (lambda_0 / lambda_2 > sphericity_threshold) return false;

    // Eigenvector for lambda_0: null space of (cov - lambda_0 * I)
    float M[9];
    for (int i = 0; i < 9; i++) M[i] = cov[i];
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

    float3 mean_p = float3(mean[0], mean[1], mean[2]);
    float  d = -(normal.x * mean_p.x + normal.y * mean_p.y + normal.z * mean_p.z);
    plane_eq = float4(normal, d);
    return true;
}

kernel void plane_krnl(
    device const Point*          cloud      [[buffer(0)]],
    device const atomic_int*     table_keys [[buffer(1)]],
    device const uint*           table_vals [[buffer(2)]],
    device const VoxelDistribution* dists   [[buffer(3)]],
    device atomic_uint*          out_size   [[buffer(4)]],
    device PointNormal*          out_buff   [[buffer(5)]],
    constant PlanesParams&       params     [[buffer(6)]],
    uint                         idx        [[thread_position_in_grid]])
{
    if (idx >= params.cloud_size) return;

    float4 ext_q = float4(params.lidar_to_imu_quat[0], params.lidar_to_imu_quat[1], params.lidar_to_imu_quat[2], params.lidar_to_imu_quat[3]);
    float3 ext_t = float3(params.lidar_to_imu_pos[0], params.lidar_to_imu_pos[1], params.lidar_to_imu_pos[2]);
    float4 q = float4(params.pose_quat[0], params.pose_quat[1], params.pose_quat[2], params.pose_quat[3]);
    float3 t = float3(params.pose_pos[0], params.pose_pos[1], params.pose_pos[2]);

    Point p_raw = cloud[idx];
    float3 p_local  = float3(p_raw.x, p_raw.y, p_raw.z);

    // Transform: local → IMU → global
    float3 p_imu    = iso_transform(ext_q, ext_t, p_local);
    float3 p_global = iso_transform(q, t, p_imu);

    // Collect 27 neighbor voxels
    int3 grid = grid_pos(params.resolution, p_global);
    int3 neighbors[27];
    uint nb_count = neighbor_cells(26, grid, neighbors);

    // Local accumulation fields
    uint  merged_point_count = 0;
    float merged_mean[3] = {0,0,0};
    float merged_cov[9]  = {0,0,0, 0,0,0, 0,0,0};

    for (uint ni = 0; ni < nb_count; ni++) {
        int nb_key = hash_grid(neighbors[ni]);
        uint buf_idx = hashmap_lookup(table_keys, table_vals,
                                      params.table_cap, nb_key);
        if (buf_idx == UINT_MAX) continue;

        device const VoxelDistribution& nd = dists[buf_idx];
        if (nd.point_count == 0) continue;

        pool_dist(merged_point_count, merged_mean, merged_cov, nd);
    }

    // Fit plane
    float4 plane;
    if (!plane_fit(merged_point_count, merged_mean, merged_cov, params.sphericity_threshold, plane)) return;

    // Score filter (same formula as CUDA)
    float pd2 = plane_point_dist(plane, p_global);
    float l2_sq = p_local.x*p_local.x + p_local.y*p_local.y + p_local.z*p_local.z;
    float s   = 1.0 - 0.9 * fabs(pd2) / sqrt(max(l2_sq, 1e-9f));
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

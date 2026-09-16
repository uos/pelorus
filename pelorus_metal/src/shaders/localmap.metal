// localmap.metal — build_voxels kernel: one thread per unique voxel group.
// Included after hashmap.metal in the compiled source.

struct BuildVoxelsParams {
    float pose_quat[4];
    float pose_trans[3];
    float _pad0;
    float resolution;
    uint  scan_number;
    float sensitivity;
    float min_dist;
    uint  unique_key_count;
    uint  table_cap;
    uint  dist_cap;
    uint  n_points;
};

// calculate_keys: Transform points to global frame and compute voxel keys.
kernel void calculate_keys(
    device const Point*          points         [[buffer(0)]],
    device int*                  out_keys       [[buffer(1)]],
    device uint*                 out_indices    [[buffer(2)]],
    constant BuildVoxelsParams&  params         [[buffer(3)]],
    uint                         idx            [[thread_position_in_grid]])
{
    if (idx >= params.n_points) return;

    float4 q = float4(params.pose_quat[0], params.pose_quat[1], params.pose_quat[2], params.pose_quat[3]);
    float3 t = float3(params.pose_trans[0], params.pose_trans[1], params.pose_trans[2]);

    Point p_raw = points[idx];
    float3 p_local = float3(p_raw.x, p_raw.y, p_raw.z);
    float3 p_global = iso_transform(q, t, p_local);

    int3 grid = grid_pos(params.resolution, p_global);
    out_keys[idx] = hash_grid(grid);
    out_indices[idx] = idx;
}

kernel void reorder_points(
    device const Point*          src            [[buffer(0)]],
    device Point*                dst            [[buffer(1)]],
    device const uint*           sorted_indices [[buffer(2)]],
    constant BuildVoxelsParams&  params         [[buffer(3)]],
    uint                         idx            [[thread_position_in_grid]])
{
    if (idx >= params.n_points) return;

    float4 q = float4(params.pose_quat[0], params.pose_quat[1], params.pose_quat[2], params.pose_quat[3]);
    float3 t = float3(params.pose_trans[0], params.pose_trans[1], params.pose_trans[2]);

    Point p_raw = src[sorted_indices[idx]];
    float3 p_local = float3(p_raw.x, p_raw.y, p_raw.z);
    float3 p_global = iso_transform(q, t, p_local);

    Point p_out = p_raw;
    p_out.x = p_global.x; p_out.y = p_global.y; p_out.z = p_global.z;
    dst[idx] = p_out;
}

kernel void rle(
    device const int*            sorted_keys   [[buffer(0)]],
    device int*                  unique_keys   [[buffer(1)]],
    device uint*                 run_starts    [[buffer(2)]],
    device uint*                 run_ends      [[buffer(3)]],
    device atomic_uint*          counters      [[buffer(4)]],
    constant BuildVoxelsParams&  params        [[buffer(5)]],
    uint                         idx           [[thread_position_in_grid]])
{
    if (idx >= params.n_points) return;

    int key = sorted_keys[idx];
    if (key == EMPTY_KEY) return;

    bool is_start = (idx == 0) || (sorted_keys[idx - 1] != key);
    if (!is_start) return;

    uint left = idx;
    uint right = params.n_points;
    while (left < right) {
        uint mid = left + (right - left) / 2;
        if (sorted_keys[mid] <= key) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    uint slot = atomic_fetch_add_explicit(&counters[0], 1u, memory_order_relaxed);
    run_starts[slot] = idx;
    run_ends[slot] = left;
    unique_keys[slot] = key;
}

kernel void assign_indices(
    device const int*            unique_keys     [[buffer(0)]],
    device const atomic_int*     table_keys      [[buffer(1)]],
    device const uint*           table_vals      [[buffer(2)]],
    device int*                  dist_keys       [[buffer(3)]],
    device const uint*           lru_mapping     [[buffer(4)]],
    device uint*                 target_indices  [[buffer(5)]],
    device uint*                 update_flags    [[buffer(6)]],
    device atomic_uint*          counters        [[buffer(7)]],
    constant BuildVoxelsParams&  params          [[buffer(8)]],
    uint                         idx             [[thread_position_in_grid]])
{
    uint unique_count = atomic_load_explicit(&counters[0], memory_order_relaxed);
    if (idx >= unique_count) return;

    int key = unique_keys[idx];
    if (key == EMPTY_KEY) {
        target_indices[idx] = UINT_MAX;
        update_flags[idx] = 0u;
        return;
    }

    uint found_idx = hashmap_lookup(table_keys, table_vals, params.table_cap, key);
    if (found_idx != UINT_MAX) {
        target_indices[idx] = found_idx;
        update_flags[idx] = 1u;
        atomic_fetch_add_explicit(&counters[3], 1u, memory_order_relaxed);
        return;
    }

    update_flags[idx] = 0u;
    atomic_fetch_add_explicit(&counters[4], 1u, memory_order_relaxed);
    uint old_add = atomic_fetch_add_explicit(&counters[1], 1u, memory_order_relaxed);
    uint assign_idx = old_add;
    if (old_add >= params.dist_cap) {
        uint lru_idx = atomic_fetch_add_explicit(&counters[2], 1u, memory_order_relaxed);
        assign_idx = lru_mapping[lru_idx % params.dist_cap];
    }

    target_indices[idx] = assign_idx;
    dist_keys[assign_idx] = key;
}

kernel void lru_score(
    device const VoxelDistribution* dists        [[buffer(0)]],
    device uint*                    lru_keys     [[buffer(1)]],
    device uint*                    lru_mapping  [[buffer(2)]],
    constant BuildVoxelsParams&     params       [[buffer(3)]],
    uint                            idx          [[thread_position_in_grid]])
{
    if (idx >= params.dist_cap) return;

    uint point_count = dists[idx].point_count;
    uint temporal_dist = params.scan_number - dists[idx].last_scan;

    uint key;
    if (point_count == 0u) {
        key = 0u;
    } else {
        float score = float(point_count) / float(temporal_dist + 1u);
        key = (1u << 30) | uint(clamp(score * 1000.0f, 0.0f, float((1u << 30) - 1u)));
    }

    lru_keys[idx] = key;
    lru_mapping[idx] = idx;
}

// build_voxels: one thread per unique voxel.
// Host-side code pre-assigns each voxel to either an existing slot (update)
// or a replacement slot (CPU LRU policy), then this kernel only applies math updates.
kernel void build_voxels(
    device const Point*          cloud          [[buffer(0)]],
    device const uint*           sorted_indices [[buffer(1)]],
    device const uint*           run_starts     [[buffer(2)]],
    device const uint*           run_ends       [[buffer(3)]],
    device const int*            unique_keys    [[buffer(4)]],
    device const uint*           target_indices [[buffer(5)]],
    device const uint*           update_flags   [[buffer(6)]],
    device VoxelDistribution*    dists          [[buffer(7)]],
    device const atomic_uint*    counters       [[buffer(8)]],
    constant BuildVoxelsParams&  params         [[buffer(9)]],
    uint                         idx            [[thread_position_in_grid]])
{
    uint unique_count = atomic_load_explicit(&counters[0], memory_order_relaxed);
    if (idx >= unique_count) return;

    int   voxel_key  = unique_keys[idx];
    uint  start      = run_starts[idx];
    uint  end        = run_ends[idx];
    uint  count      = end - start;

    if (count == 0 || voxel_key == EMPTY_KEY || target_indices[idx] == UINT_MAX) return;

    // ── 1. Create scan_voxel (two-pass biased covariance for stability) ──
    float3 sum = float3(0, 0, 0);
    for (uint i = start; i < end; i++) {
        sum += float3(cloud[i].x, cloud[i].y, cloud[i].z);
    }
    
    float fn = float(count);
    float3 scan_mean = sum / fn;

    float  s_xx=0, s_xy=0, s_xz=0, s_yy=0, s_yz=0, s_zz=0;
    for (uint i = start; i < end; i++) {
        float3 p = float3(cloud[i].x, cloud[i].y, cloud[i].z) - scan_mean;
        s_xx += p.x * p.x; s_xy += p.x * p.y; s_xz += p.x * p.z;
        s_yy += p.y * p.y; s_yz += p.y * p.z; s_zz += p.z * p.z;
    }

    float scan_cov[9];
    scan_cov[0] = s_xx/fn; scan_cov[1] = s_xy/fn; scan_cov[2] = s_xz/fn;
    scan_cov[3] = scan_cov[1];
    scan_cov[4] = s_yy/fn; scan_cov[5] = s_yz/fn;
    scan_cov[6] = scan_cov[2];
    scan_cov[7] = scan_cov[5];
    scan_cov[8] = s_zz/fn;

    uint buf_idx = target_indices[idx];
    bool is_update = (update_flags[idx] != 0u);

	if (is_update) {
		// Update existing voxel - use local copy for atomic writeback (matches Vulkan pattern)
		VoxelDistribution d = dists[buf_idx];
		d.last_scan = params.scan_number;

		float3 d_mean = float3(d.mean[0], d.mean[1], d.mean[2]);
		float3 mean_diff = d_mean - scan_mean;
		float fn_d = float(d.point_count);
		float maturity_decay = max(1.0f, log(fn_d));
		float effective_sensitivity = params.sensitivity / maturity_decay;

		float reg_cov[9];
		for (int i=0; i<9; i++) reg_cov[i] = d.cov[i];
		regularize_cov(reg_cov);

		float L_reg[9];
		float m_sq;
		if (cholesky_3x3(reg_cov, L_reg)) {
			m_sq = mahalanobis_sq(mean_diff, L_reg);
		} else {
			m_sq = dot(mean_diff, mean_diff);
		}

		float alpha = exp(-m_sq / (2.0f * effective_sensitivity));

		if (alpha >= 1e-5f) {
			float3 new_mean = d_mean * (1.0f - alpha) + scan_mean * alpha;
			float outer[9];
			float dv[3] = { mean_diff.x, mean_diff.y, mean_diff.z };
			for (int r=0; r<3; r++)
			for (int c=0; c<3; c++)
				outer[r*3+c] = dv[r] * dv[c];

			for (int i=0; i<9; i++) {
				d.cov[i] = d.cov[i] * (1.0f - alpha) + scan_cov[i] * alpha +
				           outer[i] * (alpha * (1.0f - alpha));
			}
			d.mean[0] = new_mean.x; d.mean[1] = new_mean.y; d.mean[2] = new_mean.z;

			float next_L[9] = {0};
			if (!cholesky_3x3(d.cov, next_L)) {
				float3 ev = symmetric_eigen_3x3(d.cov);
				next_L[0] = sqrt(max(0.0f, ev.x)); next_L[4] = sqrt(max(0.0f, ev.y)); next_L[8] = sqrt(max(0.0f, ev.z));
			}
			for (int i=0; i<9; i++) d.transform[i] = next_L[i];

			d.point_count += count;
		}
		dists[buf_idx] = d; // Always write back: at minimum persists last_scan (matches CUDA/pelorus_localmap)
	} else {
		// Insert new voxel - use local copy for atomic writeback (matches Vulkan pattern)
		VoxelDistribution d;
		d.lock_flag = 0;
		for (int i=0; i<9; i++) {
			d.transform[i] = 0.0f;
			d.cov[i] = scan_cov[i];
		}
		d.point_count = count;
		d.mean[0] = scan_mean.x; d.mean[1] = scan_mean.y; d.mean[2] = scan_mean.z;
		d.last_scan = params.scan_number;

		float scan_L[9] = {0};
		if (!cholesky_3x3(scan_cov, scan_L)) {
			float3 ev = symmetric_eigen_3x3(scan_cov);
			scan_L[0] = sqrt(max(0.0f, ev.x)); scan_L[4] = sqrt(max(0.0f, ev.y)); scan_L[8] = sqrt(max(0.0f, ev.z));
		}
		for (int i=0; i<9; i++) d.transform[i] = scan_L[i];

		dists[buf_idx] = d; // Single writeback (matches Vulkan)
	}
}

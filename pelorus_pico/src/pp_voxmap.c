/* pp_voxmap.c — fixed-size open-addressing voxel map
 * Algorithm: direct port of spole/src/lib.rs + spole/src/localmap.rs
 */
#include "pp_voxmap.h"
#include "pp_linalg.h"

#define PP_SQRT(x)  __builtin_sqrt(x)
#define PP_SQRTF(x) __builtin_sqrtf(x)
#define PP_EXPF(x)  __builtin_expf(x)
#define PP_LOGF(x)  __builtin_logf(x)
#define PP_FABSF(x) __builtin_fabsf(x)

/* ---- spole hash constants ---- */
#define PRIME_1  73856093u
#define PRIME_2    471943u
#define PRIME_3  83492791u

/* ================================================================
 * Lifecycle
 * ================================================================ */
void voxmap_init(pp_voxmap_t *vm) {
    for (int i = 0; i < PELORUS_VOXEL_CAPACITY; i++) {
        vm->slots[i].occupied = 0;
    }
    vm->occupied_count = 0;
    vm->scan_number = 0;
}

void voxmap_clear(pp_voxmap_t *vm) {
    voxmap_init(vm);
}

int voxmap_is_empty(const pp_voxmap_t *vm) {
    return vm->occupied_count == 0;
}

/* ================================================================
 * Hash function — identical to spole/src/lib.rs
 * ================================================================ */
void voxmap_pos2grid(float x, float y, float z, int *gx, int *gy, int *gz) {
    float inv = 1.0f / PELORUS_VOXEL_SIZE;
    *gx = (int)__builtin_floorf(x * inv);
    *gy = (int)__builtin_floorf(y * inv);
    *gz = (int)__builtin_floorf(z * inv);
}

int32_t voxmap_hash_grid(int gx, int gy, int gz) {
    uint32_t h = (uint32_t)gx * PRIME_1
               + (uint32_t)gy * PRIME_2
               + (uint32_t)gz * PRIME_3;
    return (int32_t)h;
}

/* ================================================================
 * Open-addressing lookup (linear probing)
 * ================================================================ */
/* Returns slot index of existing entry with this key, or -1. */
int voxmap_find(pp_voxmap_t *vm, int32_t key) {
    uint32_t start = (uint32_t)(key) & (PELORUS_VOXEL_CAPACITY - 1);
    for (uint32_t i = 0; i < (uint32_t)PELORUS_VOXEL_CAPACITY; i++) {
        uint32_t idx = (start + i) & (PELORUS_VOXEL_CAPACITY - 1);
        if (!vm->slots[idx].occupied) return -1;
        if (vm->slots[idx].key == key) return (int)idx;
    }
    return -1;
}

/* Returns slot index for insertion (may be occupied with same key or empty). */
static int voxmap_find_or_empty(pp_voxmap_t *vm, int32_t key) {
    uint32_t start = (uint32_t)(key) & (PELORUS_VOXEL_CAPACITY - 1);
    for (uint32_t i = 0; i < (uint32_t)PELORUS_VOXEL_CAPACITY; i++) {
        uint32_t idx = (start + i) & (PELORUS_VOXEL_CAPACITY - 1);
        if (!vm->slots[idx].occupied) return (int)idx;
        if (vm->slots[idx].key == key) return (int)idx;
    }
    return -1;
}

/* ================================================================
 * Voxel distribution update (Welford's online + weighted EMA)
 * Direct port of VoxelDistribution::update_from_point in spole/src/lib.rs
 * ================================================================ */
static void voxslot_update_from_point(pp_voxel_slot_t *s, float px, float py, float pz,
                                       uint32_t scan_n) {
    s->last_scan = scan_n;

    if (s->point_count < 2) {
        s->point_count++;
        float n = (float)s->point_count;
        float old_mean[3] = { s->mean[0], s->mean[1], s->mean[2] };
        float diff[3] = { px - old_mean[0], py - old_mean[1], pz - old_mean[2] };
        /* Welford update of mean */
        s->mean[0] += diff[0] / n;
        s->mean[1] += diff[1] / n;
        s->mean[2] += diff[2] / n;

        if (s->point_count > 1) {
            float diff2[3] = { px - s->mean[0], py - s->mean[1], pz - s->mean[2] };
            /* scatter matrix update: cov = ((n-2)/(n-1))*cov + (1/(n-1)) * outer(diff,diff2) */
            float f1 = (n - 2.0f) / (n - 1.0f);
            float f2 = 1.0f / (n - 1.0f);
            for (int i = 0; i < 3; i++) {
                float di = (i==0?diff[0]:i==1?diff[1]:diff[2]);
                for (int j = 0; j < 3; j++) {
                    float dj = (j==0?diff2[0]:j==1?diff2[1]:diff2[2]);
                    s->cov[i*3+j] = f1 * s->cov[i*3+j] + f2 * di * dj;
                }
            }
        }

        /* Update Cholesky / normal */
        mat3f C;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) C[i][j] = s->cov[i*3+j];
        mat3f_eigenvec_smallest(C, s->normal);
        return;
    }

    /* Weighted EMA regime */
    float maturity = PP_LOGF((float)(s->point_count - 1));
    if (maturity < 1.0f) maturity = 1.0f;
    float eff_sens = PELORUS_SENSITIVITY / maturity;

    float d[3] = { px - s->mean[0], py - s->mean[1], pz - s->mean[2] };

    /* Mahalanobis distance via regularised covariance */
    mat3f C, Cinv;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) C[i][j] = s->cov[i*3+j];
    C[0][0] += 1e-6f; C[1][1] += 1e-6f; C[2][2] += 1e-6f;
    float dist_sq;
    if (mat3f_inv(Cinv, C) == 0) {
        float Cd[3];
        mat3f_mulv(Cd, Cinv, d);
        dist_sq = vec3f_dot(d, Cd);
    } else {
        dist_sq = vec3f_dot(d, d);
    }

    float alpha = PP_EXPF(-dist_sq / (2.0f * eff_sens));

    float new_m[3] = {
        (1.0f - alpha)*s->mean[0] + alpha*px,
        (1.0f - alpha)*s->mean[1] + alpha*py,
        (1.0f - alpha)*s->mean[2] + alpha*pz
    };

    /* Covariance: (1-α)*cov + α*(1-α)*outer(d,d) */
    float f_cov = (1.0f - alpha);
    float f_out = alpha * (1.0f - alpha);
    for (int i = 0; i < 3; i++) {
        float di = (i==0?d[0]:i==1?d[1]:d[2]);
        for (int j = 0; j < 3; j++) {
            float dj = (j==0?d[0]:j==1?d[1]:d[2]);
            s->cov[i*3+j] = f_cov * s->cov[i*3+j] + f_out * di * dj;
        }
    }

    s->mean[0] = new_m[0];
    s->mean[1] = new_m[1];
    s->mean[2] = new_m[2];
    s->point_count++;

    /* Update cached plane normal */
    mat3f C2;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) C2[i][j] = s->cov[i*3+j];
    mat3f_eigenvec_smallest(C2, s->normal);
}

static void voxslot_recompute_normal(pp_voxel_slot_t *s) {
    mat3f C;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            C[i][j] = s->cov[i*3+j];
        }
    }
    if (mat3f_eigenvec_smallest(C, s->normal) != 0) {
        s->normal[0] = 0.0f;
        s->normal[1] = 0.0f;
        s->normal[2] = 1.0f;
    }
}

static void voxslot_set_from_dist(pp_voxel_slot_t *s,
                                  int32_t key,
                                  const pp_vox_dist_t *dist,
                                  uint32_t scan_n) {
    s->mean[0] = dist->mean[0];
    s->mean[1] = dist->mean[1];
    s->mean[2] = dist->mean[2];
    for (int i = 0; i < 9; i++) {
        s->cov[i] = dist->cov[i];
    }
    s->point_count = dist->point_count;
    s->last_scan = scan_n;
    s->key = key;
    s->occupied = 1;
    voxslot_recompute_normal(s);
}

static void voxslot_update_from_dist(pp_voxel_slot_t *s,
                                     const pp_vox_dist_t *other,
                                     uint32_t scan_n) {
    if (other->point_count == 0) return;

    s->last_scan = scan_n;

    if (s->point_count == 0) {
        voxslot_set_from_dist(s, s->key, other, scan_n);
        return;
    }

    float mean_diff[3] = {
        s->mean[0] - other->mean[0],
        s->mean[1] - other->mean[1],
        s->mean[2] - other->mean[2],
    };

    mat3f C, Cinv;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            C[i][j] = s->cov[i*3+j];
        }
    }
    C[0][0] += 1e-6f;
    C[1][1] += 1e-6f;
    C[2][2] += 1e-6f;

    float mahalanobis_sq;
    if (mat3f_inv(Cinv, C) == 0) {
        float tmp[3];
        mat3f_mulv(tmp, Cinv, mean_diff);
        mahalanobis_sq = vec3f_dot(mean_diff, tmp);
    } else {
        mahalanobis_sq = vec3f_dot(mean_diff, mean_diff);
    }

    float maturity_decay = PP_LOGF((float)s->point_count);
    if (maturity_decay < 1.0f) maturity_decay = 1.0f;
    float effective_sensitivity = PELORUS_SENSITIVITY / maturity_decay;
    float alpha = PP_EXPF(-mahalanobis_sq / (2.0f * effective_sensitivity));
    if (alpha < 1e-5f) return;

    float new_mean[3] = {
        (1.0f - alpha) * s->mean[0] + alpha * other->mean[0],
        (1.0f - alpha) * s->mean[1] + alpha * other->mean[1],
        (1.0f - alpha) * s->mean[2] + alpha * other->mean[2],
    };

    float outer[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            outer[i*3+j] = mean_diff[i] * mean_diff[j];
        }
    }

    for (int i = 0; i < 9; i++) {
        s->cov[i] = (1.0f - alpha) * s->cov[i]
                  + alpha * other->cov[i]
                  + alpha * (1.0f - alpha) * outer[i];
    }

    s->mean[0] = new_mean[0];
    s->mean[1] = new_mean[1];
    s->mean[2] = new_mean[2];
    s->point_count += other->point_count;
    voxslot_recompute_normal(s);
}

static void voxacc_reset(pp_vox_acc_t *acc) {
    for (int i = 0; i < PELORUS_CLOUD_MAX_POINTS; i++) {
        acc[i].used = 0;
    }
}

static void voxacc_add(pp_vox_acc_t *acc, float wx, float wy, float wz) {
    int gx, gy, gz;
    voxmap_pos2grid(wx, wy, wz, &gx, &gy, &gz);
    int32_t key = voxmap_hash_grid(gx, gy, gz);
    uint32_t start = (uint32_t)key & (PELORUS_CLOUD_MAX_POINTS - 1);

    for (uint32_t k = 0; k < (uint32_t)PELORUS_CLOUD_MAX_POINTS; k++) {
        uint32_t idx = (start + k) & (PELORUS_CLOUD_MAX_POINTS - 1);
        if (!acc[idx].used) {
            acc[idx].sum[0] = wx;
            acc[idx].sum[1] = wy;
            acc[idx].sum[2] = wz;
            acc[idx].sum_outer[0] = wx * wx;
            acc[idx].sum_outer[1] = wx * wy;
            acc[idx].sum_outer[2] = wx * wz;
            acc[idx].sum_outer[3] = wy * wx;
            acc[idx].sum_outer[4] = wy * wy;
            acc[idx].sum_outer[5] = wy * wz;
            acc[idx].sum_outer[6] = wz * wx;
            acc[idx].sum_outer[7] = wz * wy;
            acc[idx].sum_outer[8] = wz * wz;
            acc[idx].count = 1;
            acc[idx].key = key;
            acc[idx].used = 1;
            return;
        }
        if (acc[idx].key == key) {
            acc[idx].sum[0] += wx;
            acc[idx].sum[1] += wy;
            acc[idx].sum[2] += wz;
            acc[idx].sum_outer[0] += wx * wx;
            acc[idx].sum_outer[1] += wx * wy;
            acc[idx].sum_outer[2] += wx * wz;
            acc[idx].sum_outer[3] += wy * wx;
            acc[idx].sum_outer[4] += wy * wy;
            acc[idx].sum_outer[5] += wy * wz;
            acc[idx].sum_outer[6] += wz * wx;
            acc[idx].sum_outer[7] += wz * wy;
            acc[idx].sum_outer[8] += wz * wz;
            acc[idx].count++;
            return;
        }
    }
}

/* ================================================================
 * LRU eviction
 * ================================================================ */
void voxmap_lru_evict(pp_voxmap_t *vm, uint32_t scan_n) {
    /* Find the slot with the highest temporal_dist / point_count score
     * (i.e., oldest and least populated — same logic as Livodish::build). */
    float best_score = -1.0f;
    int best_idx = -1;
    for (int i = 0; i < PELORUS_VOXEL_CAPACITY; i++) {
        if (!vm->slots[i].occupied) continue;
        uint32_t age = scan_n - vm->slots[i].last_scan;
        float score = (float)age / (float)(vm->slots[i].point_count > 0 ? vm->slots[i].point_count : 1);
        if (score > best_score) { best_score = score; best_idx = i; }
    }
    if (best_idx >= 0) {
        vm->slots[best_idx].occupied = 0;
        vm->occupied_count--;
    }
}

/* ================================================================
 * Add point to map
 * ================================================================ */
void voxmap_add_point(pp_voxmap_t *vm, float wx, float wy, float wz, uint32_t scan_n) {
    int gx, gy, gz;
    voxmap_pos2grid(wx, wy, wz, &gx, &gy, &gz);
    int32_t key = voxmap_hash_grid(gx, gy, gz);

    /* Evict if load factor too high */
    if ((float)vm->occupied_count >= PELORUS_LRU_TRIGGER_FRACTION * (float)PELORUS_VOXEL_CAPACITY) {
        voxmap_lru_evict(vm, scan_n);
    }

    int idx = voxmap_find_or_empty(vm, key);
    if (idx < 0) {
        /* Table full even after eviction — evict one more slot at the probe start */
        uint32_t start = (uint32_t)key & (PELORUS_VOXEL_CAPACITY - 1);
        vm->slots[start].occupied = 0;
        vm->occupied_count--;
        idx = (int)start;
    }

    pp_voxel_slot_t *s = &vm->slots[idx];
    if (!s->occupied) {
        /* New slot */
        s->mean[0] = wx; s->mean[1] = wy; s->mean[2] = wz;
        for (int i=0;i<9;i++) s->cov[i] = 0.0f;
        s->normal[0]=0.0f; s->normal[1]=0.0f; s->normal[2]=1.0f;
        s->point_count = 1;
        s->last_scan = scan_n;
        s->key = key;
        s->occupied = 1;
        vm->occupied_count++;
    } else {
        voxslot_update_from_point(s, wx, wy, wz, scan_n);
    }
}

void voxmap_add_points(pp_voxmap_t *vm,
                       const pelorus_pico_point_t *pts_world,
                       int n_pts,
                       uint32_t scan_n,
                       pp_vox_acc_t *acc) {
    voxacc_reset(acc);

    for (int i = 0; i < n_pts; i++) {
        voxacc_add(acc, pts_world[i].x, pts_world[i].y, pts_world[i].z);
    }

    for (int i = 0; i < PELORUS_CLOUD_MAX_POINTS; i++) {
        if (!acc[i].used || acc[i].count == 0) continue;

        pp_vox_dist_t dist;
        float inv_count = 1.0f / (float)acc[i].count;
        dist.mean[0] = acc[i].sum[0] * inv_count;
        dist.mean[1] = acc[i].sum[1] * inv_count;
        dist.mean[2] = acc[i].sum[2] * inv_count;
        dist.point_count = acc[i].count;

        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                dist.cov[r*3+c] =
                    acc[i].sum_outer[r*3+c] * inv_count
                    - dist.mean[r] * dist.mean[c];
            }
        }

        if ((float)vm->occupied_count >=
            PELORUS_LRU_TRIGGER_FRACTION * (float)PELORUS_VOXEL_CAPACITY) {
            voxmap_lru_evict(vm, scan_n);
        }

        int idx = voxmap_find_or_empty(vm, acc[i].key);
        if (idx < 0) {
            uint32_t start = (uint32_t)acc[i].key & (PELORUS_VOXEL_CAPACITY - 1);
            vm->slots[start].occupied = 0;
            vm->occupied_count--;
            idx = (int)start;
        }

        pp_voxel_slot_t *s = &vm->slots[idx];
        if (!s->occupied) {
            voxslot_set_from_dist(s, acc[i].key, &dist, scan_n);
            vm->occupied_count++;
        } else {
            voxslot_update_from_dist(s, &dist, scan_n);
        }
    }
}

/* ================================================================
 * Neighbourhood helpers (mirrors spole::nearby_6/18/26)
 * ================================================================ */

/* Fill out[3*n_out] with grid offsets for the selected mode.
 * Returns number of cells (including center). */
static int nearby_offsets(int mode, int offsets[27][3]) {
    static const int nb6[7][3] = {
        {0,0,0},{-1,0,0},{1,0,0},{0,1,0},{0,-1,0},{0,0,-1},{0,0,1}
    };
    static const int nb18[19][3] = {
        {0,0,0},{-1,0,0},{1,0,0},{0,1,0},{0,-1,0},{0,0,-1},{0,0,1},
        {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
        {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
        {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
    };
    static const int nb26[27][3] = {
        {0,0,0},{-1,0,0},{1,0,0},{0,1,0},{0,-1,0},{0,0,-1},{0,0,1},
        {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
        {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
        {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1},
        {1,1,1},{-1,1,1},{1,-1,1},{1,1,-1},
        {-1,-1,1},{-1,1,-1},{1,-1,-1},{-1,-1,-1}
    };
    int n;
    const int (*src)[3];
    if (mode <= 6)  { n=7;  src=nb6; }
    else if (mode<=18){ n=19; src=nb18; }
    else            { n=27; src=nb26; }
    for (int i=0;i<n;i++) { offsets[i][0]=src[i][0]; offsets[i][1]=src[i][1]; offsets[i][2]=src[i][2]; }
    return n;
}

/* Pool distributions: weighted merge (no cross-covariance) */
static void pool_two(pp_vox_dist_t *dst, const pp_voxel_slot_t *s) {
    if (s->point_count == 0) return;
    float n1 = (float)dst->point_count;
    float n2 = (float)s->point_count;
    if (n1 == 0.0f) {
        dst->mean[0]=s->mean[0]; dst->mean[1]=s->mean[1]; dst->mean[2]=s->mean[2];
        for (int i=0;i<9;i++) dst->cov[i]=s->cov[i];
        dst->point_count = s->point_count;
        return;
    }
    float total = n1 + n2;
    float md[3] = {
        dst->mean[0]-s->mean[0],
        dst->mean[1]-s->mean[1],
        dst->mean[2]-s->mean[2]
    };
    float outer[9];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++)
        outer[i*3+j] = md[i]*md[j];

    for (int k=0;k<9;k++)
        dst->cov[k] = (n1*dst->cov[k] + n2*s->cov[k] + outer[k]*(n1*n2/total)) / total;

    dst->mean[0] = (n1*dst->mean[0] + n2*s->mean[0]) / total;
    dst->mean[1] = (n1*dst->mean[1] + n2*s->mean[1]) / total;
    dst->mean[2] = (n1*dst->mean[2] + n2*s->mean[2]) / total;
    dst->point_count = (uint32_t)(total + 0.5f);
}

void voxmap_pool_neighbors(pp_voxmap_t *vm, int gx, int gy, int gz,
                           int nearby_mode, pp_vox_dist_t *dst) {
    dst->point_count = 0;
    for (int i=0;i<9;i++) dst->cov[i]=0.0f;
    dst->mean[0]=dst->mean[1]=dst->mean[2]=0.0f;

    int offs[27][3];
    int n = nearby_offsets(nearby_mode, offs);

    for (int i=0;i<n;i++) {
        int32_t k = voxmap_hash_grid(gx+offs[i][0], gy+offs[i][1], gz+offs[i][2]);
        int idx = voxmap_find(vm, k);
        if (idx < 0) continue;
        pool_two(dst, &vm->slots[idx]);
    }
}

int voxmap_plane_fit(pp_voxmap_t *vm, float wx, float wy, float wz,
                     int nearby_mode, float plane[4]) {
    int gx, gy, gz;
    voxmap_pos2grid(wx, wy, wz, &gx, &gy, &gz);

    pp_vox_dist_t dist;
    voxmap_pool_neighbors(vm, gx, gy, gz, nearby_mode, &dist);

    if (dist.point_count < 3) return 0;

    mat3f C;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) C[i][j] = dist.cov[i*3+j];

    /* Eigenvalues to check sphericity */
    float ev[3];
    mat3f_eigenvalues(C, ev); /* sorted ascending */

    if (ev[0] != ev[0] || ev[2] != ev[2]) return 0; /* NaN check */
    if (ev[2] < 1e-9f) return 0;
    float sphericity = ev[0] / ev[2];
    if (sphericity > PELORUS_PLANE_SPHERICITY_MAX) return 0;

    float normal[3];
    if (mat3f_eigenvec_smallest(C, normal) != 0) return 0;

    float d = -(normal[0]*dist.mean[0] + normal[1]*dist.mean[1] + normal[2]*dist.mean[2]);
    plane[0] = normal[0];
    plane[1] = normal[1];
    plane[2] = normal[2];
    plane[3] = d;
    return 1;
}

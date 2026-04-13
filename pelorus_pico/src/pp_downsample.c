/* pp_downsample.c — voxel-grid medoid downsampling (in-place)
 * Port of prelio/src/lib.rs:filter_downsample (medoid path)
 */
#include "pp_downsample.h"

#define PRIME_1  73856093u
#define PRIME_2    471943u
#define PRIME_3  83492791u

/* Reuse the same hash function as the voxel map */
static inline int32_t ds_hash(float x, float y, float z) {
    float inv = 1.0f / PELORUS_VOXEL_SIZE;
    int gx = (int)__builtin_floorf(x * inv);
    int gy = (int)__builtin_floorf(y * inv);
    int gz = (int)__builtin_floorf(z * inv);
    uint32_t h = (uint32_t)gx * PRIME_1
               + (uint32_t)gy * PRIME_2
               + (uint32_t)gz * PRIME_3;
    return (int32_t)h;
}

void pp_downsample(pelorus_pico_point_t *pts,
                   int n_raw,
                   int *work_n,
                   pp_ds_acc_t *acc) {
    /* Zero accumulator */
    for (int i = 0; i < PELORUS_CLOUD_MAX_POINTS; i++) acc[i].used = 0;

    /* Pass 1: accumulate centroids per voxel */
    for (int i = 0; i < n_raw; i++) {
        int32_t key = ds_hash(pts[i].x, pts[i].y, pts[i].z);
        uint32_t start = (uint32_t)key & (PELORUS_CLOUD_MAX_POINTS - 1);
        /* Linear probe in accumulator */
        for (uint32_t k = 0; k < (uint32_t)PELORUS_CLOUD_MAX_POINTS; k++) {
            uint32_t idx = (start + k) & (PELORUS_CLOUD_MAX_POINTS - 1);
            if (!acc[idx].used) {
                acc[idx].sum[0] = pts[i].x;
                acc[idx].sum[1] = pts[i].y;
                acc[idx].sum[2] = pts[i].z;
                acc[idx].count  = 1;
                acc[idx].key    = key;
                acc[idx].used   = 1;
                break;
            }
            if (acc[idx].key == key) {
                acc[idx].sum[0] += pts[i].x;
                acc[idx].sum[1] += pts[i].y;
                acc[idx].sum[2] += pts[i].z;
                acc[idx].count++;
                break;
            }
        }
    }

    /* Pass 2: for each occupied accumulator slot, find the medoid
     * (point closest to centroid) and write into output */
    int out = 0;
    for (int s = 0; s < PELORUS_CLOUD_MAX_POINTS && out < PELORUS_CLOUD_MAX_POINTS; s++) {
        if (!acc[s].used) continue;
        float cx = acc[s].sum[0] / (float)acc[s].count;
        float cy = acc[s].sum[1] / (float)acc[s].count;
        float cz = acc[s].sum[2] / (float)acc[s].count;
        int32_t key = acc[s].key;

        /* Find medoid: scan all raw points with matching hash */
        float best_dist2 = 1e30f;
        int   best_idx   = -1;
        for (int i = 0; i < n_raw; i++) {
            int32_t ik = ds_hash(pts[i].x, pts[i].y, pts[i].z);
            if (ik != key) continue;
            float dx = pts[i].x - cx;
            float dy = pts[i].y - cy;
            float dz = pts[i].z - cz;
            float d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < best_dist2) { best_dist2 = d2; best_idx = i; }
        }
        if (best_idx >= 0) {
            pts[out++] = pts[best_idx];
        }
    }
    *work_n = out;
}

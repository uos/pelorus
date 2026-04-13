#ifndef PP_VOXMAP_H
#define PP_VOXMAP_H

/* =========================================================
 * pp_voxmap.h — fixed-size open-addressing voxel hash map
 *
 * Stores VoxelDistribution per voxel.  Identical hash function
 * to spole/src/lib.rs (floor + same primes).
 * ========================================================= */

#include "../include/pelorus_pico_config.h"
#include "../include/pelorus_pico.h"
#include "pp_linalg.h"

/* One entry in the hash table. */
typedef struct {
    float    mean[3];       /* voxel centroid (world frame, f32) */
    float    cov[9];        /* 3×3 row-major covariance (f32) */
    float    normal[3];     /* plane normal (cached from eigenvec) */
    uint32_t point_count;
    uint32_t last_scan;
    int32_t  key;           /* hash_grid result — used for collision resolution */
    uint8_t  occupied;      /* 0 = empty, 1 = valid */
} pp_voxel_slot_t;

/* Global voxel map state (lives in g_ctx). */
typedef struct {
    pp_voxel_slot_t slots[PELORUS_VOXEL_CAPACITY];
    uint32_t        occupied_count;
    uint32_t        scan_number;
} pp_voxmap_t;

/* Scratch accumulator used to batch points by voxel before updating the map.
 * This mirrors spole's per-voxel distribution update path instead of inserting
 * each point independently. */
typedef struct {
    float    sum[3];
    float    sum_outer[9];
    uint32_t count;
    int32_t  key;
    uint8_t  used;
} pp_vox_acc_t;

/* ---- lifecycle ---- */
void voxmap_init(pp_voxmap_t *vm);
void voxmap_clear(pp_voxmap_t *vm);
int  voxmap_is_empty(const pp_voxmap_t *vm);

/* ---- hashing (mirrors spole exactly) ----
 * pos2grid: floor(pos / voxel_size)
 * hash_grid: u32 arithmetic, cast to i32 */
void voxmap_pos2grid(float x, float y, float z, int *gx, int *gy, int *gz);
int32_t voxmap_hash_grid(int gx, int gy, int gz);

/* ---- lookup / insert ---- */
/* Find slot index for key; returns -1 if not found. */
int voxmap_find(pp_voxmap_t *vm, int32_t key);
/* Insert or update distribution for the voxel that contains (wx,wy,wz).
 * Uses update_from_point (Welford online algorithm). */
void voxmap_add_point(pp_voxmap_t *vm, float wx, float wy, float wz, uint32_t scan_n);
/* Insert or update a full batch of world-frame points using grouped per-voxel
 * distributions, matching spole::IVoxHashMap::add more closely. */
void voxmap_add_points(pp_voxmap_t *vm,
                       const pelorus_pico_point_t *pts_world,
                       int n_pts,
                       uint32_t scan_n,
                       pp_vox_acc_t *acc);

/* ---- neighbourhood query (returns up to n_out distributions by copy) ----
 * Fills out[] with pooled VoxelDistribution for Nearby6/18/26 + centre. */
typedef struct {
    float    mean[3];
    float    cov[9];
    uint32_t point_count;
} pp_vox_dist_t;

/* Merge all occupied neighbor voxels around grid pos (gx,gy,gz) into dst.
 * nearby_mode: 6, 18, or 26 */
void voxmap_pool_neighbors(pp_voxmap_t *vm, int gx, int gy, int gz,
                           int nearby_mode, pp_vox_dist_t *dst);

/* ---- plane fit (returns [nx,ny,nz,d] or 0 if not enough points/not planar) ---- */
int voxmap_plane_fit(pp_voxmap_t *vm, float wx, float wy, float wz,
                     int nearby_mode, float plane[4]);

/* ---- LRU eviction (O(N) scan) ---- */
void voxmap_lru_evict(pp_voxmap_t *vm, uint32_t scan_n);

#endif /* PP_VOXMAP_H */

#ifndef PP_DOWNSAMPLE_H
#define PP_DOWNSAMPLE_H

#include "../include/pelorus_pico.h"
#include "../include/pelorus_pico_config.h"

/* Voxel-grid downsampling with medoid selection.
 * Operates in-place: pts[] is both input and output.
 * *work_n set to the number of valid output points (<= n_raw).
 *
 * Uses the same hash function as pp_voxmap (floor + primes).
 * Scratch buffer (accumulator) must be caller-provided to avoid
 * stack overflow; it lives in g_ctx. */

/* Per-slot accumulator for centroid computation */
typedef struct {
    float  sum[3];
    int    count;
    int32_t key;
    uint8_t used;
} pp_ds_acc_t;

/* acc must point to an array of PELORUS_CLOUD_MAX_POINTS pp_ds_acc_t */
void pp_downsample(pelorus_pico_point_t *pts,
                   int n_raw,
                   int *work_n,
                   pp_ds_acc_t *acc);

#endif /* PP_DOWNSAMPLE_H */

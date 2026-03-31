#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <lib.h>

// ── GPU voxel distribution — matches Metal's GpuVoxelDist (96 bytes) ─────────

struct GpuVoxelDist {
    int      lock_flag;      //  4 bytes, offset  0
    float    transform[9];   // 36 bytes, offset  4
    uint32_t point_count;    //  4 bytes, offset 40
    float    mean[3];        // 12 bytes, offset 44
    float    cov[9];         // 36 bytes, offset 56
    uint32_t last_scan;      //  4 bytes, offset 92
};
static_assert(sizeof(GpuVoxelDist) == 96, "GpuVoxelDist must be 96 bytes");

// ── Kernel parameter structs — must match Metal exactly ───────────────────────

struct BuildVoxelsParams {
    float    pose_quat[4];      // 16 bytes, offset  0
    float    pose_trans[3];     // 12 bytes, offset 16
    float    _pad0;             //  4 bytes, offset 28
    float    resolution;        //  4 bytes, offset 32
    uint32_t scan_number;       //  4 bytes, offset 36
    float    sensitivity;       //  4 bytes, offset 40
    float    min_dist;          //  4 bytes, offset 44
    uint32_t unique_key_count;  //  4 bytes, offset 48
    uint32_t table_cap;         //  4 bytes, offset 52
    uint32_t dist_cap;          //  4 bytes, offset 56
    uint32_t _pad1;             //  4 bytes, offset 60  → 64 total
};

struct PlanesParams {
    float    lidar_to_imu_quat[4]; // 16 bytes, offset  0
    float    lidar_to_imu_pos[3];  // 12 bytes, offset 16
    float    _pad0;                //  4 bytes, offset 28
    float    pose_quat[4];         // 16 bytes, offset 32
    float    pose_pos[3];          // 12 bytes, offset 48
    float    _pad1;                //  4 bytes, offset 60
    uint32_t cloud_size;           //  4 bytes, offset 64
    uint32_t table_cap;            //  4 bytes, offset 68
    float    resolution;           //  4 bytes, offset 72
    float    sphericity_threshold; //  4 bytes, offset 76
    uint32_t nbtype;               //  4 bytes, offset 80  → 84 total
};

// ── C API exposed to lib.cu ───────────────────────────────────────────────────

#ifdef __cplusplus
extern "C++" {
#endif

void     init_map_impl(cudaStream_t stream, uint32_t capacity, float voxel_size,
                       float sensitivity, float min_dist, uint32_t nbtype);
void     add_to_map_impl(cudaStream_t stream, Point* points, uint32_t n_points, Pose pose);
uint32_t planes_impl(cudaStream_t stream, Point* cloud, uint32_t cloud_size,
                     Pose lidar_to_imu, Pose pose, PointNormal* out_buff);
void     get_map_impl(Point* points, uint32_t* size);
uint32_t get_map_capacity_impl();
uint32_t get_map_size_impl();
void     cleanup_map_impl();

#ifdef __cplusplus
} // extern "C++"
#endif

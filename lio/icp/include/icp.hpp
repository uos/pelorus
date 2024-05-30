#ifndef ICP_HPP
#define ICP_HPP

typedef struct
{
    double x;
    double y;
    double z;

    double q_i;
    double q_j;
    double q_k;
    double q_w;
} LioIsometry3d;

void *init_map(double voxel_resolution);

void add_to_map(void *map, const void *point_buffer, int num_points, LioIsometry3d global_transform);

void align(void *map, const void *point_buffer, int num_points, LioIsometry3d init_guess, LioIsometry3d *out);

void drop_map(void *map);

#endif

/*#[repr(C)]
struct LioIsometry3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,

    pub q_i: f64,
    pub q_j: f64,
    pub q_k: f64,
    pub q_w: f64,
}*/
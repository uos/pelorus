#ifndef ICP_HPP
#define ICP_HPP

void* init_map(double voxel_resolution);

void add_to_map(void* map, void* point_buffer, int num_points, double* global_translate, double* global_rotate);

void align(void* map, void* point_buffer, int num_points, double* translation, double* rotation);

void drop_map(void *map);

#endif
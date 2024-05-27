void *init_map();

void add_to_map(void *map, void *point_buffer, int num_points);

void align(void *map, void *point_buffer, int num_points, double *translation, double *rotation);

void drop_map(void *map);

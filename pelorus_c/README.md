# pelorus_c

C library bindings for Pelorus.

## Configuration

The library uses `.rl` configuration files (same format as the ROS2 version):

```c
// Use custom configuration file to override defaults for your setup
struct PelorusHandle* pelorus = pelorus_create("/path/to/my_config.rl");
```

You can generate a default config file with:
```bash
pelorus config init > my_config.rl
```

All parameters (map resolution, IMU covariances, LiDAR vendor, filtering options, etc.) are configurable.

## Threading Model

The API is designed to be simple and synchronous from the caller's perspective:
- `pelorus_add_imu()` and `pelorus_add_pointcloud()` return immediately
- Processing happens asynchronously in background threads
- `pelorus_try_recv()` is non-blocking and returns immediately
- All threading complexity is hidden from the user

## License

Dual-licensed under MIT and Apache 2.0, consistent with the Pelorus project.

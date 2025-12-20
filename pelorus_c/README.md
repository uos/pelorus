# pelorus_c

C library bindings for Pelorus.

## Configuration

The library uses `.rl` configuration files (same format as the ROS2 version):

```c
// Use custom configuration file to override defaults for your setup
struct PelorusHandle* pelorus = pelorus_create("/path/to/my_config.rl");

// Or: configure creation-time behavior without editing the on-disk .rl
pelorus_create_options opts = PELORUS_CREATE_OPTIONS_DEFAULT;
opts.config_path = "/path/to/my_config.rl";
// opts.device_specific_pointcloud2 = 0; // disable device-specific interpretation
struct PelorusHandle* pelorus2 = pelorus_create_with_options(&opts);
```

You can generate a default config file with:
```bash
pelorus config init > my_config.rl
```

All parameters (map resolution, IMU covariances, LiDAR vendor, filtering options, etc.) are configurable.

## API Design

Pelorus provides three ways to receive odometry results:

### 1. Non-blocking (Polling)
Use `pelorus_try_recv()` for non-blocking checks:
```c
pelorus_odometry odom;
int status = pelorus_try_recv(pelorus, &odom);
if (status == 1) {
    // Process odometry
    pelorus_free_odometry(&odom);
} else if (status == 0) {
    // No data available yet
}
```

### 2. Blocking (Synchronous)
Use `pelorus_recv()` to wait for results:
```c
pelorus_odometry odom;
int status = pelorus_recv(pelorus, &odom);  // Blocks until data available
if (status == 1) {
    // Process odometry
    pelorus_free_odometry(&odom);
}
```

### 3. Callback (Asynchronous)
Use `pelorus_set_callback()` for event-driven processing:
```c
void odometry_callback(const pelorus_odometry *odom, void *user_data) {
    // Process odometry (called from background thread)
    // No need to call pelorus_free_odometry() - handled automatically
}

// Register callback
pelorus_set_callback(pelorus, odometry_callback, user_data);

// Disable callback
pelorus_set_callback(pelorus, NULL, NULL);
```

**Note**: When using callbacks, `pelorus_try_recv()` and `pelorus_recv()` will not receive data. Callbacks and polling/blocking APIs are mutually exclusive.

## Examples

See the `examples/` directory for complete usage examples:
- `basic_usage.c` - Simple polling example using `pelorus_try_recv()`
- `async_callback.c` - Callback-based asynchronous example
- `sync_blocking.c` - Synchronous blocking example

Build and run examples:
```bash
cd examples
make
./async_callback
./sync_blocking
```

### Raw point helpers

If you already have raw point buffers (similar to the Python numpy input), you can avoid
building a full `pelorus_pointcloud2` and instead call one of:

- `pelorus_add_points_xyzt_f32(...)` for Nx4 `[x,y,z,t]` (`t` is ns offset, cast to `uint32_t`)
- `pelorus_add_points_xyzit_f32(...)` for Nx5 `[x,y,z,intensity,t]`

All of these are internally converted into a PointCloud2 with `x,y,z,intensity,t` fields.

## Threading Model

The API is designed to be simple from the caller's perspective:
- `pelorus_add_imu()`, `pelorus_add_odom()`, and `pelorus_add_pointcloud()` return immediately
- Processing happens asynchronously in background threads
- `pelorus_try_recv()` is non-blocking and returns immediately
- `pelorus_recv()` blocks until data is available
- Callbacks are invoked from a background thread
- All threading complexity is hidden from the user

## Memory Management

- Results from `pelorus_try_recv()` and `pelorus_recv()` must be freed with `pelorus_free_odometry()`
- Callback results are automatically freed after the callback returns
- The `PelorusHandle` must be destroyed with `pelorus_destroy()`

## Timestamp semantics

Pelorus uses two timestamp levels:

1. **IMU/Odom timestamps** (`timestamp_sec` + `timestamp_nanosec`)
    - Absolute timestamps (typically epoch / ROS time) used for synchronization.

2. **LiDAR timestamps**
    - `pelorus_pointcloud2.header_timestamp_sec/nanosec`: absolute reference time for the scan.
    - Per-point field `t` (if provided): `uint32` **nanoseconds offset**, **non-negative** and **forward in time** relative to the scan header timestamp.

The effective point time is:

`point_time_abs = header_time_abs + (t * 1e-9)`

### Recommended PointCloud2 field layout for per-point timing

Provide points as a `pelorus_pointcloud2` with:

- `x` (FLOAT32) offset 0
- `y` (FLOAT32) offset 4
- `z` (FLOAT32) offset 8
- `intensity` (FLOAT32) offset 12 (optional; may be ignored depending on vendor)
- `t` (UINT32) offset 16 (nanoseconds offset)

This corresponds to `point_step = 20`.

## License

Dual-licensed under MIT and Apache 2.0, consistent with the Pelorus project.

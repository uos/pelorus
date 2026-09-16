# pelorus_c

C library bindings for Pelorus.

## Building and Installation

### CMake

The library can be built and installed using CMake:

```bash
cd pelorus_c
mkdir build && cd build
cmake ..
make
sudo make install
```

#### CMake Options

- `PELORUS_BUILD_SHARED` - Build shared library (default: ON)
- `PELORUS_BUILD_STATIC` - Build static library (default: ON)
- `PELORUS_BUILD_EXAMPLES` - Build C/C++ examples (default: OFF)
- `PELORUS_INSTALL` - Install library and headers (default: ON)

#### Feature Flags

- `PELORUS_FEATURE_RAYON` - Enable Rayon CPU parallelism (default: ON)
- `PELORUS_FEATURE_METAL` - Enable Metal acceleration (default: OFF)
- `PELORUS_FEATURE_CUDA` - Enable CUDA acceleration (default: OFF)
- `PELORUS_FEATURE_VULKAN` - Enable Vulkan acceleration (default: OFF)
- `PELORUS_FEATURE_NO_IMU_SYNC` - Disable IMU synchronization (default: OFF)
- `PELORUS_FEATURE_LARGE_PROCESSING_QUEUE` - Use larger internal processing queues (default: OFF)

Select at most one GPU backend. Metal is available on macOS; CUDA requires a
supported NVIDIA CUDA installation; Vulkan requires a compatible Vulkan device
and driver. If no GPU backend is selected, Pelorus uses the CPU backend.

Example with CUDA:

```bash
cmake .. -DPELORUS_FEATURE_CUDA=ON
```

### Using Cargo (Direct)

You can also build directly with Cargo:

```bash
cargo build --package pelorus_c --release

# Select one GPU backend when needed:
cargo build --package pelorus_c --release --features cuda
cargo build --package pelorus_c --release --features vulkan
cargo build --package pelorus_c --release --features metal
```

## Using the Library

### In CMake Projects

After installation, use `find_package`:

```cmake
find_package(pelorus REQUIRED)
add_executable(my_app main.c)
target_link_libraries(my_app pelorus::pelorus)
```

See [examples/CMAKE_USAGE.md](examples/CMAKE_USAGE.md) for more details.

### With pkg-config

```bash
gcc my_app.c $(pkg-config --cflags --libs pelorus) -o my_app
```

### C++ Wrapper

A modern C++ wrapper is available in `pelorus.hpp`.

```cpp
#include <pelorus.hpp>
#include <iostream>

int main() {
    pelorus::Pelorus pelorus;
    
    // Add measurements
    pelorus::ImuMsg imu;
    // ... fill IMU data ...
    pelorus.add_imu(imu);
    
    // Receive odometry
    auto res = pelorus.try_recv();
    if (res.status == pelorus::Pelorus::STATUS_OK) {
        const auto &odom = res.odom;
        std::cout << "Position: " << odom.position()[0] << "\n";
    }
    
    return 0;
}
```

### PCL Integration

For projects using the Point Cloud Library (PCL):

```cpp
#include <pelorus.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

pelorus::Pelorus pelorus;

// Convert and add PCL cloud to Pelorus
pcl::PointCloud<pelorus::pcl::PointXYZIT>::Ptr cloud(
    new pcl::PointCloud<pelorus::pcl::PointXYZIT>());
// ... fill cloud ...
pelorus::pcl::add_pointcloud(pelorus, cloud, timestamp_sec, timestamp_nsec);

// Get result
auto result = pelorus.recv();
if (result.status == pelorus::Pelorus::STATUS_OK) {
    auto result_cloud =
        result.odom.get_cloud_pcl<pcl::PointCloud<pcl::PointXYZI>::Ptr>();

    // Or use the standalone function
    auto result_cloud2 = pelorus::pcl_helpers::to_pcl_cloud(
        result.odom.points(), result.odom.num_points());
}
```

### Eigen3 Integration

For projects using Eigen for transformations:

```cpp
#include <pelorus.hpp>
#include <iostream>

pelorus::Pelorus pelorus;
auto result = pelorus.recv();
if (result.status == pelorus::Pelorus::STATUS_OK) {
    const auto &odom = result.odom;

    // Get odometry as a transformation matrix
    Eigen::Isometry3d transform = pelorus::eigen::to_transform(odom);

    // Transform a point from sensor to world coordinates
    Eigen::Vector3d point_sensor(1.0, 0.0, 0.0);
    Eigen::Vector3d point_world = transform * point_sensor;

    // Get all odometry data as Eigen types
    auto state = pelorus::eigen::OdometryState::from_pelorus(odom);
    std::cout << "Position: "
              << state.transform.translation().transpose() << "\n";
    std::cout << "Velocity: " << state.linear_velocity.transpose() << "\n";
    std::cout << "Accelerometer bias: " << state.accel_bias.transpose() << "\n";
    std::cout << "Gyroscope bias: " << state.gyro_bias.transpose() << "\n";

    // Extract individual components
    Eigen::Vector3d position = pelorus::eigen::to_position(odom.position());
    Eigen::Quaterniond orientation =
        pelorus::eigen::to_quaternion(odom.orientation());
    Eigen::Matrix<double, 6, 6> pose_cov =
        pelorus::eigen::get_pose_covariance(*odom.get());
}
```

## Configuration

The library uses `.rl` configuration files:

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
pelorus config generate recommended > my_config.rl

# Or generate the complete default configuration:
pelorus config generate full > my_config.rl
```

All parameters (map resolution, IMU covariances, Lidar vendor, filtering options, etc.) are configurable.

## API Design

Pelorus provides three ways to receive odometry results:

Bias-aware callers can use `pelorus_try_recv_with_bias()`,
`pelorus_recv_with_bias()`, or `pelorus_set_callback_with_bias()`. These
additive APIs return `pelorus_imu_bias` without changing the layout of the
existing `pelorus_odometry` structure. The C++ wrapper uses these APIs
automatically and exposes `Odometry::accel_bias()` and
`Odometry::gyro_bias()`.

### 1. Non-blocking (Polling)
Use `pelorus_try_recv()` for non-blocking checks:
```c
pelorus_odometry odom;
int status = pelorus_try_recv(pelorus, &odom);
if (status == PELORUS_STATUS_OK) {
    // Process odometry
    pelorus_free_odometry(&odom);
} else if (status == PELORUS_STATUS_NO_DATA) {
    // No data available yet
} else if (status == PELORUS_STATUS_CALLBACK_SET) {
    // Callback mode is active; polling APIs are unavailable
}
```

### 2. Blocking (Synchronous)
Use `pelorus_recv()` to wait for results:
```c
pelorus_odometry odom;
int status = pelorus_recv(pelorus, &odom);  // Blocks until data available
if (status == PELORUS_STATUS_OK) {
    // Process odometry
    pelorus_free_odometry(&odom);
} else if (status == PELORUS_STATUS_CALLBACK_SET) {
    // Callback mode is active; cannot use blocking APIs
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

- `async_callback.c` - Callback-based asynchronous example
- `sync_blocking.c` - Synchronous blocking example
- `sync_blocking_pcl_eigen.cpp` - C++ wrapper example with optional PCL/Eigen support
- `eigen_example.cpp` - Eigen conversion example

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

## Memory Management

- Results from `pelorus_try_recv()` and `pelorus_recv()` must be freed with `pelorus_free_odometry()`
- Callback results are automatically freed after the callback returns
- The `PelorusHandle` must be destroyed with `pelorus_destroy()`

## Timestamp semantics

Pelorus uses two timestamp levels:

1. **IMU/Odom timestamps** (`timestamp_sec` + `timestamp_nanosec`)
    - Absolute timestamps (typically epoch / ROS time) used for synchronization.

2. **Lidar timestamps**
    - `pelorus_pointcloud2.header_timestamp_sec/nanosec`: absolute reference time for the scan.
    - Per-point field `t` (if provided): `uint32` **nanoseconds offset**, **non-negative** and **forward in time** relative to the scan header timestamp.

The effective point time is:

`point_time_abs = header_time_abs + (t * 1e-9)`

### Pelorus expects a simplified, single-row (unstructured) little-endian `PointCloud2`.

Interpretation of per-point timing depends on the parsing mode:

- When Pelorus uses *device-specific* parsing (the default), field meanings — including any `t` field — are interpreted according to the vendor/device configuration and may vary.
- When using the generic parsing path (`device_specific_pointcloud2 == false`), Pelorus expects `t` to be a **non-negative** (`uint32`) nanoseconds offset relative to the scan header timestamp.

For generic point input, prefer `pelorus_add_points_xyzt_f32()` or `pelorus_add_points_xyzit_f32()`.

## License

Dual-licensed under MIT and Apache 2.0, consistent with the Pelorus project.

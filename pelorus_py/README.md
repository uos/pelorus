# Pelorus Python Bindings

Python bindings for Pelorus, a resource-aware Lidar-Inertial Odometry.

## Installation

### From source (requires Rust toolchain)

```bash
cd pelorus_py
pip install maturin
maturin develop --release
```

Select at most one GPU backend when building:

```bash
maturin develop --release --features cuda
maturin develop --release --features vulkan
maturin develop --release --features metal
```

Metal is available on macOS. CUDA requires a supported NVIDIA CUDA
installation. Vulkan requires a compatible device and driver. The default
features enable Rayon CPU parallelism and include `processing_time_ns` in
odometry results. Additional features are `no-imu-sync` and
`large-processing-queue`. The legacy `downsample` feature is retained for
compatibility but does not control the current configuration-based downsampling.

Or build a wheel:

```bash
maturin build --release --out dist
pip install dist/pelorus-*.whl
```

## Usage

### Basic API

```python
import pelorus
import numpy as np

# Create Pelorus instance
lio = pelorus.Handle() # Uses embedded default config

# Or with custom config:
# lio = pelorus.Handle("path/to/config.rl")
#
# If you want to control PointCloud2 parsing (like `pelorus_create_with_options` in C):
# - device_specific_pointcloud2=True  (default)
# - device_specific_pointcloud2=False (generic parsing path)
# lio = pelorus.Handle("path/to/config.rl", device_specific_pointcloud2=False)

# Add IMU data
lio.add_imu(
    timestamp_sec=1000,
    timestamp_nanosec=0,
    orientation=[1.0, 0.0, 0.0, 0.0],  # w, x, y, z
    orientation_covariance=[0.01] + [0.0]*8,
    angular_velocity=[0.0, 0.0, 0.0],
    angular_velocity_covariance=[0.001] + [0.0]*8,
    linear_acceleration=[0.0, 0.0, 9.81],
    linear_acceleration_covariance=[0.01] + [0.0]*8,
)

# Add point cloud (recommended):
# - If you already have ROS2 messages, pass them directly:
#   lio.add_pointcloud2_ros(cloud_msg)
#
# - For non-ROS point sources, use NumPy/sequence inputs:
# - Nx3: [x,y,z] (no per-point timing)
# - Nx4: [x,y,z,t] where t is uint32 nanosecond offset
# - Nx5: [x,y,z,intensity,t]
# NumPy arrays may use float32 or float64. Inputs must be non-empty. The t value
# must be finite and in the uint32 range; fractional values are rounded to the
# nearest nanosecond. The scan header timestamp is required:
points = np.array([[1.0, 2.0, 3.0, 0.0]], dtype=np.float32)
scan_id = lio.add_pointcloud(
    points,
    timestamp_sec=1000,
    timestamp_nanosec=0,
)

# After feeding enough synchronized IMU and Lidar data, receive odometry:
odom = lio.try_recv()
if odom:
    print(f"Position: {odom['position']}")
    print(f"Orientation: {odom['orientation']}")
    print(f"Linear velocity: {odom['linear_velocity']}")
    print(f"Accelerometer bias: {odom['accel_bias']}")
    print(f"Gyroscope bias: {odom['gyro_bias']}")
    print(f"Points: {len(odom['points'])}")

# Or block until the next odometry result is available. Do this only while
# another producer is feeding data or when an output is expected.
# odom = lio.recv()


# Or receive odometry asynchronously via callback (like pelorus_c)
def on_odom(odom: dict) -> None:
    print("Odom:", odom["timestamp_sec"], odom["timestamp_nanosec"], odom["position"])

lio.set_callback(on_odom)
# ... feed IMU / pointclouds ...
lio.clear_callback()  # equivalent to lio.set_callback(None)
```

### Using with MCAP Bagfiles

See `examples/bagfile_example.py` for a complete example using real sensor data from an MCAP bagfile.


```bash
python -m pip install mcap mcap-ros2-support
```

## API Reference

### `Handle`

Main interface to the Pelorus system.

#### Constructor
- `Handle(config_path: str | None = None, device_specific_pointcloud2: bool = True)`: Create a new Pelorus instance

    `device_specific_pointcloud2=False` selects the generic parsing path for PointCloud2 inputs.

#### Methods
- `add_imu(...)`: Add an IMU measurement
- `add_imu_ros(imu_msg)`: Add a ROS2 `sensor_msgs/msg/Imu` object directly
- `add_odom(...)`: Add an odometry measurement
- `add_odom_ros(odom_msg)`: Add a ROS2 `nav_msgs/msg/Odometry` object directly
- `add_pointcloud(cloud_dict)`: Add a point cloud (PointCloud2 format)
- `add_pointcloud(cloud_np, timestamp_sec=..., timestamp_nanosec=...)`: Add a point cloud from a non-empty float32/float64 NumPy array
- `add_pointcloud(cloud_sequence, timestamp_sec=..., timestamp_nanosec=...)`: Add a point cloud from lists, tuples, or other Python sequences
- `add_pointcloud2_ros(cloud_msg)`: Add a ROS2 `sensor_msgs/msg/PointCloud2` message directly
- `try_recv()`: Try to receive odometry (non-blocking), returns `None` if no data
- `recv()`: Receive odometry (blocks until available)
- `set_callback(callback)`: Deliver odometry via callback from a background thread. Disables `try_recv()`/`recv()` queuing while enabled
- `clear_callback()`: Disable callback delivery (equivalent to `set_callback(None)`)
- `version()`: Get library version (static method)

## Timestamp semantics

Pelorus uses two timestamp levels:

1. **IMU/Odom timestamps** (`timestamp_sec` + `timestamp_nanosec`)
    - Absolute timestamps (typically epoch / ROS time) used for synchronization.

2. **Lidar timestamps**
    - **Scan header timestamp**: absolute (sec/nanosec)
    - **Per-point `t` field** (if provided): `uint32` nanoseconds offset, **non-negative** and **forward in time** relative to the scan header time.

The effective point time is:

`point_time_abs = header_time_abs + (t * 1e-9)`

Odometry outputs include `timestamp_sec`/`timestamp_nanosec` (and `timestamp`
as float for convenience), `scan_id`, pose/twist data and covariances,
`accel_bias` and `gyro_bias` in the IMU frame, a `points` NumPy array, and
`frame_id`. Builds using the default `processing-time-export` feature also
include `processing_time_ns`.

## Development

```bash
# Install development dependencies
pip install maturin pytest black ruff

# Build and test
maturin develop
pytest

# Format code
black .
ruff check .
```

## License

Dual-licensed under MIT or Apache-2.0, consistent with the Pelorus project.

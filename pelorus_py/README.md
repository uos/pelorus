# Pelorus Python Bindings

Python bindings for Pelorus - a LiDAR-Inertial Odometry system.

## Installation

### From source (requires Rust toolchain)

```bash
cd pelorus_py
pip install maturin
maturin develop --release
```

Or build a wheel:

```bash
maturin build --release
pip install target/wheels/pelorus-*.whl
```

## Usage

### Basic API

```python
import pelorus
import numpy as np

# Create Pelorus instance
lio = pelorus.Handle()  # Uses embedded default config

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
    frame_id="imu",
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
# - For non-ROS point sources, use numpy/list inputs:
# - Nx3: [x,y,z] (no per-point timing)
# - Nx4: [x,y,z,t] where t is uint32 nanosecond offset
# - Nx5: [x,y,z,intensity,t]
# For numpy/list inputs, you must also provide the scan header timestamp:
# lio.add_pointcloud(points_np, timestamp_sec=..., timestamp_nanosec=..., frame_id="lidar")

# Try to receive odometry (non-blocking)
odom = lio.try_recv()
if odom:
    print(f"Position: {odom['position']}")
    print(f"Orientation: {odom['orientation']}")
    print(f"Linear velocity: {odom['linear_velocity']}")
    print(f"Points: {len(odom['points'])}")

# Or block until odometry is available
odom = lio.recv()


# Or receive odometry asynchronously via callback (like pelorus_c)
def on_odom(odom: dict) -> None:
    print("Odom:", odom["timestamp_sec"], odom["timestamp_nanosec"], odom["position"])

lio.set_callback(on_odom)
# ... feed IMU / pointclouds ...
lio.clear_callback()  # equivalent to lio.set_callback(None)
```

### Using with MCAP Bagfiles

See `examples/bagfile_example.py` for a complete example using real sensor data from an MCAP bagfile.

## API Reference

### `Handle`

Main interface to the Pelorus system.

#### Constructor
- `Handle(config_path: str | None = None, device_specific_pointcloud2: bool = True)`: Create a new Pelorus instance

    `device_specific_pointcloud2=False` selects the generic parsing path for PointCloud2 inputs.

#### Methods
- `add_imu(...)`: Add an IMU measurement
- `add_odom(...)`: Add an odometry measurement  
- `add_pointcloud(cloud_dict)`: Add a point cloud (PointCloud2 format)
- `add_pointcloud(cloud_np, timestamp_sec=..., timestamp_nanosec=..., frame_id="lidar")`: Add a point cloud from numpy
- `add_pointcloud2_ros(cloud_msg)`: Add a ROS2 `sensor_msgs/msg/PointCloud2` message directly
- `try_recv()`: Try to receive odometry (non-blocking), returns `None` if no data
- `recv()`: Receive odometry (blocks until available)
- `set_callback(callback)`: Deliver odometry via callback from a background thread; disables `try_recv()`/`recv()` queuing while enabled
- `clear_callback()`: Disable callback delivery (equivalent to `set_callback(None)`)
- `version()`: Get library version (static method)

## Timestamp semantics

Pelorus uses two timestamp levels:

1. **IMU/Odom timestamps** (`timestamp_sec` + `timestamp_nanosec`)
    - Absolute timestamps (typically epoch / ROS time) used for synchronization.

2. **LiDAR timestamps**
    - **Scan header timestamp**: absolute (sec/nanosec)
    - **Per-point `t` field** (if provided): `uint32` nanoseconds offset, **non-negative** and **forward in time** relative to the scan header time.

The effective point time is:

`point_time_abs = header_time_abs + (t * 1e-9)`

Odometry outputs include `timestamp_sec`/`timestamp_nanosec` (and `timestamp` as float for convenience).

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

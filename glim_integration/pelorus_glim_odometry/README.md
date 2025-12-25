# pelorus_glim_odometry

GLIM odometry backend plugin that runs **Pelorus as the odometry estimation module** (implements `glim::OdometryEstimationBase`).

This is an alternative to GLIM's built-in `odometry_estimation_cpu/gpu/ct` modules.

## Quick start

### 1) Prerequisites

- GLIM installed (with headers and CMake config)
- Pelorus installed (CMake package `pelorus`), or Rust toolchain (to build `pelorus_c` from source)

### 2) Build and install

```sh
cd pelorus/glim_integration/pelorus_glim_odometry
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make
sudo make install
```

Notes:
- If `find_package(pelorus)` succeeds, the build uses your installed Pelorus.
- Otherwise it falls back to building `pelorus_c` from `../../pelorus_c` (requires Rust).

This installs `libodometry_estimation_pelorus.so` (or `.dylib` on macOS) to `/usr/local/lib`.

## Enable Pelorus odometry in GLIM

GLIM selects the odometry module via `global.config_odometry` → a JSON file containing:

```json
{
  "odometry_estimation": {
    "so_name": "libodometry_estimation_pelorus.so",
    "pelorus_rl_path": "/abs/path/to/pelorus.rl",
    "max_wait_ms": 200
  }
}
```

Typical setup:
1. Create a config file (e.g., `config_odometry_pelorus.json`) in your GLIM config root
2. Set `so_name` to:
   - Linux: `libodometry_estimation_pelorus.so`
   - macOS: `libodometry_estimation_pelorus.dylib`
3. In GLIM's `config.json`, set:
   - `"config_odometry": "config_odometry_pelorus.json"`
4. Run GLIM as usual

## Configuration

- `so_name` (required): Path or name of the plugin library
- `pelorus_rl_path` (optional): Path to Pelorus config file (empty = defaults)
- `max_wait_ms` (optional): Timeout when polling Pelorus (default: 200ms)

Example with all options:
```json
{
  "odometry_estimation": {
    "so_name": "libodometry_estimation_pelorus.so",
    "pelorus_rl_path": "/path/to/pelorus.rl",
    "max_wait_ms": 300
  }
}
```

## How it works

The backend:
- Receives IMU and LiDAR frames from GLIM's preprocessing pipeline
- Feeds them into Pelorus via `add_imu()` and `add_points_xyzit()`
- Polls Pelorus for odometry results and populates `EstimationFrame` fields
- Returns pose, velocity, and point cloud to GLIM for factor graph construction

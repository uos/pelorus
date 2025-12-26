# pelorus_glim_frontend

GLIM extension module that feeds Pelorus and injects **additional odometry constraints** into GLIM's odometry factor graph (similar to `orb_slam_frontend` in `glim_ext`).

This **does not replace** GLIM odometry. It only adds extra `BetweenFactor<Pose3>` constraints.

## Quick start

### 1) Prerequisites

- GLIM installed (with headers and CMake config)
- GTSAM installed (provides `gtsam` headers/libs)
- Pelorus installed (CMake package `pelorus`), or Rust toolchain (to build `pelorus_c` from source)

### 2) Build and install

```sh
cd pelorus/glim_integration/pelorus_glim_frontend
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make
sudo -E make install
```

Notes:
- If `find_package(pelorus)` succeeds, the build uses your installed Pelorus.
- Otherwise it falls back to building `pelorus_c` from `../../pelorus_c` (requires Rust).

This installs `libpelorus_glim_frontend.so` (or `.dylib` on macOS) to `/usr/local/lib`.

### 3) Enable the extension in GLIM

Add the plugin library name to `glim_ros.extension_modules` (in your GLIM config):

- Linux: `libpelorus_glim_frontend.so`
- macOS: `libpelorus_glim_frontend.dylib`

Then run GLIM as usual.

## Configuration

The extension reads optional parameters from a config file named `config_pelorus_frontend.json` in your GLIM config root:

```json
{
  "pelorus_frontend": {
    "pelorus_rl_path": "/abs/path/to/pelorus.rl",

    "stamp_match_tolerance_sec": 0.02,

    "between_sigma_trans": 0.2,
    "between_sigma_rot_rad": 0.15,

    "max_frame_buffer": 400
  }
}
```

Notes:
- `pelorus_rl_path` may be empty to use Pelorus defaults.
- `between_sigma_*` controls how "strong" Pelorus constraints are.

## How it works

- The module subscribes to GLIM odometry callbacks (`on_insert_imu`, `on_insert_frame`, `on_new_frame`).
- It feeds IMU + points into Pelorus via `add_imu()` and `add_points_xyzit()`.
- When Pelorus emits an odometry estimate, it is matched to the closest GLIM frame timestamp.
- For consecutive matched frames, a `BetweenFactor<Pose3>` is added to GLIM's odometry graph.

<h1 align="center">
  Pelorus<br>
  <a href="https://github.com/stelzo/pelorus/blob/main/LICENSE-MIT">    <img alt="MIT"            src="https://img.shields.io/badge/license-MIT-blue.svg">                        </a>
  <a href="https://github.com/stelzo/pelorus/blob/main/LICENSE-APACHE"> <img alt="Apache"         src="https://img.shields.io/badge/license-Apache-blue.svg">                     </a>
  <a href="https://www.rust-lang.org">                                  <img alt="Rust"         src="https://img.shields.io/badge/rust-1.85+-blue.svg">                         </a>
</h1>

<p align="center">
Ressource-aware 3D Lidar-based Odometry for robots.
</p>

Supporting various 3D Lidar sensors, IMUs, odometry sources and kinematic models, Pelorus provides real-time performance on embedded hardware while exceeding State-of-the-Art accuracy.

Scale your compute down to single-core CPUs or up to full parallel compute on CPU and GPU with CUDA acceleration.


## 🤖 ROS 2 Support

First, install **[Rust](https://www.rust-lang.org/tools/install)** if you haven't already.

Pelorus integrates seamlessly with your standard **ROS 2 CMake** build system.

```sh
cd ~/ros2_ws/src
git clone https://github.com/stelzo/pelorus
cd ..

rosdep install --from-paths src -y --ignore-src

colcon build --packages-select pelorus
```

Building might take a while.

> [!TIP]
> Pelorus offers various configurations at compile time. You can set them with CMake arguments via uppercasing and setting the prefix `PELORUS_` like so:
> ```sh
> colcon build --packages-select pelorus --cmake-args -D PELORUS_CUDA=ON
> ```
> Building with CUDA acceleration needs the `CUDA_HOME` environment variable to be set. Some installation do this already. If not, add `export CUDA_HOME=/usr/local/cuda` to your `~/.bashrc`, `~/.zshrc` etc.

Available flags for the ROS 2 version are:

**Default flags** (enabled by default, disable all with `-D PELORUS_ALL_DEFAULTS=OFF`):
- `downsample`
- `parallel`
- `mimalloc`
- `no-viz`

**Optional flags**:
- `cuda` (requires `CUDA_HOME` environment variable)
- `rerun` (set `RERUN_SERVER_ADDR` for remote instances, e.g. `127.0.0.1:9876`.)
- `rat`
- `mapping` (experimental)
- `map`
- `evaluation`

Explanations for all flags are available in the [Cargo.toml](./pelorus/Cargo.toml) under `[features]`.

### ⚙️ Configuration

When running for the first time, generate a configuration file with default settings to customize them later.
```sh
ros2 run pelorus pelorus config init > pelorus.rl
```
Open `pelorus.rl` in your preferred text editor and adjust the parameters for your specific use case. You can even delete lines where you're happy with the default values.

Typical parameters to modify are:

- `lidar.topic`
- `lidar.vendor`
- `imu.topic`
- `wheel.kinematic_model`
- `preprocessing.downsample.resolution`
- `registration.map.resolution`

For basics of the configuration language itself visit the [Ratslang](https://github.com/stelzo/ratslang) project.

### 🚀 Launching

Source your workspace and run the node. If your configuration file, (e.g. `pelorus.rl`) file is in the same directory where you are running the command, you can use a relative path:
```sh
ros2 run pelorus pelorus run online --config-path pelorus.rl
```

## Offline Bagfiles

TODO minot, export tum format usage

You can also run Pelorus on recorded bagfiles for offline evaluation:
```sh
ros2 run pelorus pelorus run offline --config-path pelorus.rl --bag-path /path/to/my_bagfile.bag
```

## C Bindings

Pelorus provides a C library for using the system outside of ROS pipelines.

TODO examples

### Configuration

Both libraries use `.rl` configuration files (same format as the ROS2 version):
- Pass `NULL` to use default configuration from `default.rl`
- Pass a path to load custom configuration

Generate a default config file:
```bash
pelorus config init > my_config.rl
```

All parameters are configurable: map resolution, IMU covariances, LiDAR vendor, filtering options, etc.

See the library [README](./pelorus_c/README.md) for more detail.

## GLIM integration

Pelorus can be integrated into [GLIM](https://github.com/koide3/glim) in two ways:

1. **As a GLIM extension module (constraint source)**
  - Adds additional `BetweenFactor<Pose3>` constraints to GLIM’s odometry factor graph.
  - Does **not** replace GLIM odometry (similar to ORB-SLAM frontend constraints in `glim_ext`).
  - Setup guide: [pelorus_glim_frontend/README.md](./pelorus_glim_frontend/README.md)

2. **As a full GLIM odometry backend**
  - Replaces GLIM’s odometry module by implementing `glim::OdometryEstimationBase`.
  - Setup guide: [pelorus_glim_odometry/README.md](./pelorus_glim_odometry/README.md)

## 🔧 Development

We use `mask` to simplify common development commands and save you some typing.
```sh
cargo install mask cargo-mommy # ;)
```

In the project directory (where `./maskfile.md` is located), you can see all available commands:
```sh
mask --help
```

For proper visualisations, the development environment uses `rerun`. It's recommended to run it in the background.
```sh
cargo install rerun

tmux new -s rerun
rerun # then detach
```


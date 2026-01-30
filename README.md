# Pelorus

Ressource-aware 3D Lidar-based Odometry for robots.

Supporting various 3D Lidar sensors, IMUs, odometry sources and kinematic models, Pelorus provides real-time performance on embedded hardware while exceeding State-of-the-Art accuracy.

Scale your compute down to single-core CPUs or up to full parallel compute on CPU and GPU with CUDA acceleration.


## ROS 2

First, install **[Rust](https://www.rust-lang.org/tools/install)** if you haven't already.

Pelorus integrates seamlessly with your standard **ROS 2 CMake** build system.

```sh
cd ~/ros2_ws/src
git clone git@gitlabti.informatik.uni-osnabrueck.de:chsieh/pelorus.git
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
- `parallel` (use all available CPU cores)
- `mimalloc` (use mimalloc as memory allocator)
- `evaluation` (build the `evaluation` command for comparing trajectories and generating metrics similar to the `evo` tool)

**Optional flags**:
- `cuda` (use all available CUDA accelerations; requires `CUDA_HOME` environment variable to be set to your `/usr/local/cuda` or similar)
- `rerun` (visualization with timeline; set `RERUN_SERVER_ADDR` for remote instances, e.g. `127.0.0.1:9876`. Requires v0.24)
- `map` (publish localmap; only meant for debug purposes)
- `no-viz` (avoid any visualization code; useful for headless systems)


After building and sourcing, you can use Pelorus as a ROS node.

```sh
ros2 run pelorus pelorus run online -c pelorus.rl
```

In the following examples, the `ros2 run` prefix will be omitted. Just preprend it again when using ROS.

### Configuration

Pelorus uses `ratslang` for proper configuration. The defaults are validated at compile time. Every change overwrites the default.

Generate a minimal configuration with typical required parameters for robots or datasets by:

```sh
# typical settings that often change
pelorus config generate recommended > pelorus.rl

# alternatively, output all settings
pelorus config generate full > pelorus.rl
```

Adapt the values and validate the config to check you did everything correctly:

```sh
pelorus config verify pelorus.rl
```

For basics of the configuration language itself visit the [Ratslang](https://github.com/stelzo/ratslang) project.

### 🚀 Launching

Source your workspace and run the node. If your configuration file, (e.g. `pelorus.rl`) file is in the same directory where you are running the command, you can use a relative path:

```sh
ros2 run pelorus pelorus run online -c pelorus.rl
```


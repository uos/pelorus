# ROS

We support ROS 2 for fast prototyping.

First, install [Rust](https://www.rust-lang.org/tools/install) if you haven't already.

Pelorus integrates seamlessly with your standard **ROS 2 CMake** build system.

```sh
cd ~/ros2_ws/src
git clone ssh://git.steado.tech/pelorus.git
cd ..

rosdep install --from-paths src -y --ignore-src

colcon build --packages-select pelorus
```

Building might take a while.

!!! note "GPU Support"

    Pelorus offers various configurations at compile time. You can set them with CMake arguments via uppercasing and setting the prefix `PELORUS_` like so:
    ```sh
    colcon build --packages-select pelorus --cmake-args -D PELORUS_CUDA=ON
    ```

Available flags for the ROS 2 version are:

**Default flags** (enabled by default, disable all with `-D PELORUS_ALL_DEFAULTS=OFF`):
- `rayon` use all available CPU cores
- `rat` synchronize input/output for use with [Minot](https://codeberg.org/stelzo/minot) using the `-r` flag

**Highlighted flags**:
- `cuda` use all available CUDA accelerations; searches for your CUDA directory - if not found, please set the `CUDA_HOME` environment variable.
- `metal` use Apple Metal acceleration on macOS (Apple Silicon)
- `vulkan` use Vulkan acceleration
- `rerun` visualization with timeline; set `RERUN_SERVER_ADDR` for remote instances, e.g. `127.0.0.1:9876`. Requires a recent Rust toolchain.

Now you can use the launch file to start the node but we need a configuration first. For basics of the configuration language itself visit the [Ratslang](https://github.com/stelzo/ratslang) project.

```sh
# make a config and change the values to your needs, e.g. topic names, frame ids, sensor parameters etc.
ros2 run pelorus pelorus config generate recommended > pelorus.rl

# now launch
ros2 launch pelorus pelorus.launch.py config_path:=pelorus.rl rviz:=false
```

## Binaries

The Pelorus beta (0.0.x) is currently available as precompiled binaries for Linux on x86_64 and aarch64 (ARM64).

Since the binaries may have problems during the current beta, we recommend building from source. Follow the related instructions from the other pages for that.

You can [setup the UOS PPA](https://codeberg.org/uos-robotics/ppa/src/branch/pages/README.md) for the Ubuntu LTS versions to install Pelorus and get updates automatically.

```sh title="ROS 2 package for Humble and Jazzy"
sudo apt install ros-$ROS_DISTRO-pelorus

# or with CUDA support
sudo apt install ros-$ROS_DISTRO-pelorus-cuda12-8
sudo apt install ros-$ROS_DISTRO-pelorus-cuda13-1
```

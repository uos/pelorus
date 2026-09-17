# GLIM

Pelorus can act as the odometry module inside [GLIM](https://koide3.github.io/glim/), which then handles submapping and global optimization on top of it. This page assumes GLIM and [Rust](https://www.rust-lang.org/tools/install) are already installed, and uses a ROS 2 Jazzy layout for the configuration paths.

The plugin links against the Pelorus C library dynamically, so changing the acceleration backend later is a matter of rebuilding `pelorus_c` alone.

## Install the C library

```sh
git clone ssh://git.steado.tech/pelorus
cd pelorus/pelorus_c
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
sudo cmake --install build
cd ..
```

Add `-DPELORUS_FEATURE_CUDA=ON`, `-DPELORUS_FEATURE_VULKAN=ON` or `-DPELORUS_FEATURE_METAL=ON` to that configure step for GPU acceleration. The [standalone page](standalone.md#c-and-c-library) lists the rest of the options.

## Build the plugin

GLIM has to be installed and sourced in the shell you build this in.

```sh
cd glim_integration/pelorus_glim_odometry
cmake -S . -B build
cmake --build build --parallel
sudo cmake --install build
```

## Point GLIM at it

Symlink the installed plugin configuration into GLIM's configuration directory:

```sh
sudo ln -s /usr/local/share/pelorus_glim_odometry/config/config_odometry_pelorus.json \
  /opt/ros/jazzy/share/glim/config/config_odometry_pelorus.json
```

Then select it in `/opt/ros/jazzy/share/glim/config/config.json`:

```json
"config_odometry": "config_odometry_pelorus.json",
// simplify the other modules while getting started
"config_sub_mapping": "config_sub_mapping_passthrough.json",
"config_global_mapping": "config_global_mapping_pose_graph.json"
```

## Run

```sh
ros2 run glim_ros glim_rosnode
```

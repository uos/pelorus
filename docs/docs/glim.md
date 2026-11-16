# GLIM

Pelorus can act as the odometry module inside [GLIM](https://koide3.github.io/glim/), which then handles submapping and global optimization on top of it. This page assumes GLIM and [Rust](https://www.rust-lang.org/tools/install) are already installed, and uses a ROS 2 Jazzy layout for the configuration paths.

The plugin links against the Pelorus C library dynamically, so changing the acceleration backend later is a matter of rebuilding `pelorus_c` alone.

If you want to install with existing binaries, use the existing [release channels](./install.md).

Otherwise, install from source.

```sh
cd ros2_ws/src
git clone https://codeberg.org/stelzo/pelorus.git

colcon build --packages-select pelorus --cmake-args -DPELORUS_BUILD_GLIM=ON
source install/setup.bash
```
 
Add `-DPELORUS_FEATURE_CUDA=ON`, `-DPELORUS_FEATURE_VULKAN=ON` or `-DPELORUS_FEATURE_METAL=ON` to that configure step for GPU acceleration. The [standalone page](standalone.md#c-and-c-library) lists the rest of the options.

We need to tell GLIM to use our plugin now. 

When built from source, the json is in the install folder.
```sh
sudo ln -s "$(pwd)/install/pelorus/share/pelorus_glim_odometry/config/config_odometry_pelorus.json" \
  /opt/ros/jazzy/share/glim/config/config_odometry_pelorus.json
```


If you installed the system library, the json will be in the system folders instead.
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

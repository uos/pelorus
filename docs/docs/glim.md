# GLIM

We assume [GLIM](https://koide3.github.io/glim/) and [Rust](https://www.rust-lang.org/tools/install) are already installed on your system.

First install the Pelorus C library.

```sh
git clone ssh://git.steado.tech/pelorus
cd pelorus/pelorus_c
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
sudo cmake --install build
cd ..
```

You can enable acceleration backends with cmake flags here, so e.g. in the `pelorus_c` folder:
```sh
cmake -S . -B build -DPELORUS_FEATURE_CUDA=ON
```

Other backends can be activated like this:

- `-DPELORUS_FEATURE_VULKAN=ON`
- `-DPELORUS_FEATURE_METAL=ON`

Now the odometry plugin for GLIM. You'll need to have GLIM installed/sourced in your shell.

```sh
cd glim_integration/pelorus_glim_odometry/
mkdir build
cd build
cmake ..
cmake --build . --parallel
sudo cmake --install .
```

Now we need to tell GLIM to use Pelorus for odometry. We will symlink the plugin config from our install to GLIM. We assume a ROS2 Jazzy installation here for the config path.

```sh
sudo ln -s /usr/local/share/pelorus_glim_odometry/config/config_odometry_pelorus.json /opt/ros/jazzy/share/glim/config/config_odometry_pelorus.json
```

Now edit the GLIM config `/opt/ros/jazzy/share/glim/config/config.json` to activate our new plugin.

```json
"config_odometry": "config_odometry_pelorus.json",
// we will also simplify other modules for starting
"config_sub_mapping": "config_sub_mapping_passthrough.json",
"config_global_mapping": "config_global_mapping_pose_graph.json"
```

Now run GLIM. For example here online, waiting for input:

```sh
ros2 run glim_ros glim_rosnode
```

Changing the backend via feature flags can be done by just re-compiling `pelorus_c`. The odometry plugin links to the library dynamically.

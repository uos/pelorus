# Pelorus

Resource-aware Lidar-Inertial Odometry.

Supporting various 3D Lidar sensors, IMUs, odometry sources and kinematic models, Pelorus provides real-time performance on embedded hardware while pushing state-of-the-art accuracy.

Scale your compute down to single-core CPU usage or up to full parallel compute on CPU and **GPU** supporting NVIDIA, AMD, Intel and Apple Silicon.


## ROS 2

First, install **[Rust](https://www.rust-lang.org/tools/install)** if you haven't already.

Pelorus integrates seamlessly with your standard **ROS 2 CMake** build system.

```sh
cd ~/ros2_ws/src
git clone git@gitlabti.informatik.uni-osnabrueck.de:chsieh/pelorus.git
cd ..

rosdep install --from-paths src -y --ignore-src

colcon build --packages-select pelorus

source install/setup.bash

# make a config and change the values to your needs, e.g. topic names, frame ids, sensor parameters etc.
ros2 run pelorus pelorus config generate recommended > pelorus.rl

# now launch
ros2 launch pelorus pelorus.launch.py config_path:=pelorus.rl rviz:=false
```


Please see the [docs](https://stelzo.codeberg.page/pelorus/) for more info about ROS, standalone usage, libraries and other tooling or information.

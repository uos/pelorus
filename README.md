# lio - LiDAR Inertial Odometry

A simple LiDAR Odometry using IMU data as inertial estimation for GICP on Gaussian Voxel Grids.

## Setup

You need to have ROS2 installed on your sytem. We assume an Ubuntu installation from here on.

Dependencies
```shell
sudo apt-get install libeigen3-dev libtbb-dev libclang-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

pip install --break-system-package colcon-ros-cargo cargo-ament-build
```


```shell
git clone --recurse-submodules git://gitlab.informatik.uni-osnabrueck.de/csieh/lio.git src/lio
colcon build --clean-build --cargo-args --release
ros2 launch src/lio/lio/launch/lio.py
```

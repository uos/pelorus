# lio - LiDAR Inertial Odometry

## Setup

You need to have ROS2 Humble installed on your sytem. We assume an Ubuntu installation from here on.

Install `Eigen3` and `TBB`.
```shell
sudo apt-get install libeigen3-dev libtbb-dev
```

And Rust if you do not have it already.
```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Clone the repository and build the project. We use nightly for interacting with a C library and SIMD acceleration, which is not stable yet.
```shell
git clone --recurse-submodules git://gitlab.informatik.uni-osnabrueck.de/csieh/lio.git
cd lio
rustup override set nightly
cargo build
```

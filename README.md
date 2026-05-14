# Pelorus

Ressource-aware 3D Lidar-based Odometry for robots.

Supporting various 3D Lidar sensors, IMUs, odometry sources and kinematic models, Pelorus provides real-time performance on embedded hardware while exceeding State-of-the-Art accuracy.

Scale your compute down to single-core CPU usage or up to full parallel compute on CPU and **GPU** supporting NVIDIA, AMD, Intel and Apple Silicon.


## ROS 2

### Binary

Use the UOS PPA:

```sh
curl -fsSL "https://uos-robotics.codeberg.page/ppa/ubuntu/key.gpg" | gpg --dearmor \
  | sudo tee /usr/share/keyrings/uos-archive-keyring.gpg >/dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/uos-archive-keyring.gpg] https://uos-robotics.codeberg.page/ppa/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/uos.list
sudo apt update
```

And install:

```sh
# Jazzy
sudo apt install ros-jazzy-pelorus

# Humble
sudo apt install ros-humble-pelorus
```

### Source

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

Available flags for the ROS 2 version are:

**Default flags** (enabled by default, disable all with `-D PELORUS_ALL_DEFAULTS=OFF`):
- `rayon` use all available CPU cores
- `rat` synchronize input/output for use with [Minot](https://codeberg.org/stelzo/minot) using the `-r` flag

**Highlighted flags**:
- `cuda` use all available CUDA accelerations; searches for your CUDA directory - if not found, please set the `CUDA_HOME` environment variable.
- `metal` use Apple Metal acceleration on macOS (Apple Silicon)
- `vulkan` use Vulkan acceleration
- `rerun` visualization with timeline; set `RERUN_SERVER_ADDR` for remote instances, e.g. `127.0.0.1:9876`. Requires a recent Rust toolchain.

After building and sourcing, you can use Pelorus as a ROS node.

## Configuration

Pelorus uses `ratslang` for proper configuration. The defaults are validated at compile time. Every change overwrites the default.

Generate a minimal configuration with typical required parameters for robots or datasets with the `recommended` preset or a full configuration with all parameters and their defaults with the `full` preset.

For basics of the configuration language itself visit the [Ratslang](https://github.com/stelzo/ratslang) project.

## Run

```sh
# generate a config file with settings that are recommended to change for your use case
ros2 run pelorus pelorus config generate recommended > pelorus.rl
# edit pelorus.rl to your needs, e.g. set the topic names, frame ids, sensor parameters etc.

# (optional) verify the config file for correctness
ros2 run pelorus pelorus config verify pelorus.rl

# run Pelorus with the config file
ros2 run pelorus pelorus online -c pelorus.rl
```

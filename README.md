<h1 align="center">
  Pelorus<br>
  <a href="https://github.com/stelzo/pelorus/blob/main/LICENSE-MIT">    <img alt="MIT"            src="https://img.shields.io/badge/license-MIT-blue.svg">                        </a>
  <a href="https://github.com/stelzo/pelorus/blob/main/LICENSE-APACHE"> <img alt="Apache"         src="https://img.shields.io/badge/license-Apache-blue.svg">                     </a>
  <a href="https://www.rust-lang.org">                                  <img alt="Rust"         src="https://img.shields.io/badge/rust-1.85+-blue.svg">                         </a>
</h1>

<p align="center">
Efficient Lidar Inertial Odometry with optional Cuda acceleration.
</p>

## ROS 2 (Humble/Jazzy)
First, install **[Rust](https://www.rust-lang.org/tools/install)** if you haven't already.

Pelorus integrates seamlessly with your standard **ROS 2 CMake** build system.

```sh
cd ~/ros2_ws/src
git clone https://github.com/stelzo/pelorus
cd ..

rosdep install --from-paths src -y --ignore-src

colcon build
```

Building might take a while 

💡 Tip: Pelorus offers various configurations at compile time. You can set them with CMake arguments via uppercasing and setting the prefix `PELORUS_` like so:
```sh
colcon build --cmake-args -D PELORUS_CUDA=ON -D PELORUS_RERUN=ON
```

Available flags for the ROS 2 version are:

- `downsample` (enabled by default)
- `parallel` (enabled by default)
- `cuda`
- `rerun`
- `rat`
- `mapping`
- `map`
- `trajectory`
- `mimalloc`

Explanations for all flags are available in the [Cargo.toml](./crates/pelorus/Cargo.toml) under `[features]`.

## ⚙️ Configuration

When running for the first time, generate a configuration file with default settings to customize them later.
```sh
ros2 run pelorus pelorus config init > pelorus.rl
```
Open `pelorus.rl` in your preferred text editor and adjust the parameters for your specific use case. You can even delete lines where you're happy with the default values.

Typical parameters to modify are:

- `lidar.topic`
- `lidar.vendor`
- `imu.topic`
- `preprocessing.downsample.resolution`
- `registration.map.resolution`

For basics of the configuration language itself visit the [Ratslang](https://github.com/stelzo/ratslang) project.

Once configured, source your workspace and run the node with your new config file:
```sh
ros2 run pelorus pelorus run -c pelorus.rl
```

## 🔧 Development

We use `mask` to simplify common development commands and save you some typing.
```sh
cargo install mask
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

If you're using a remote `rerun` instance, simply set the `RERUN_SERVER_ADDR` environment variable. The default is `127.0.0.1:9876`.


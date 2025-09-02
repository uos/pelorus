# Pelorus

Efficient Lidar Inertial Odometry with optional Cuda acceleration.

## Usage

Install [Rust](https://www.rust-lang.org/tools/install) first if not done already.

### ROS2

Integrates with your typical ROS2 CMake build system.

Build.
```sh
cd ~/ros2_ws/src
git clone https://github.com/stelzo/pelorus
cd ..

colcon build
```

Pelorus provides configurations at compile time. You can set them with CMake arguments.

- downsample (enabled by default)
- cuda
- rerun
- rat
- mapping
- map
- trajectory
- mimalloc

Explanations for all flags are available in the [Cargo.toml](./crates/pelorus/Cargo.toml) under [features].

You can enable or disable them by uppercase like this:
```sh
colcon build --cmake-args -D RERUN=ON -D CUDA=ON
```

When running the first time, create a configuration with defaults to override them.
```sh
ros2 run pelorus pelorus config init > pelorus.rl
```
Open pelorus.rl in a text editor and adapt to your use case. You can also delete any line where you want to keep the defaults.

Typical parameters to change are:

- lidar.topic
- lidar.vendor
- imu.topic
- preprocessing.downsample.resolution
- registration.map.resolution

For basics of the configuration language itself visit [Ratslang](https://github.com/stelzo/ratslang).

Now source and run with the config.
```sh
ros2 run pelorus pelorus run -c pelorus.rl
```

## Development

We use mask to save some time typing out usual commands.
```sh
cargo install mask
```

In the project directory, where the `./maskfile.md` lives:
```sh
mask --help
```

Dev uses rerun for proper visualisations. Run it in background.
```sh
cargo install rerun

tmux new -s rerun
rerun # then detach
```
For remote rerun instances, set the `RERUN_SERVER_ADDR` environment variable. The default is `127.0.0.1:9876`.


# ROS 2

Pelorus builds as an ordinary ament package, so it fits into a colcon workspace without anything special. Humble, Jazzy and Lyrical are the supported distributions. Prebuilt packages are listed under [installation](install.md).

## Building from source

Install [Rust](https://www.rust-lang.org/tools/install) first, then:

```sh
cd ~/ros2_ws/src
git clone ssh://git.steado.tech/pelorus.git
cd ..

rosdep install --from-paths src -y --ignore-src
colcon build --packages-select pelorus
```

The first build takes a while.

## Running

```sh
source install/setup.bash

# generate a configuration and set topics, frame IDs and sensor parameters
ros2 run pelorus pelorus config generate recommended > pelorus.rl

ros2 launch pelorus pelorus.launch.py config_path:=pelorus.rl rviz:=false
```

See [configuration](configuration/index.md) for what to put in that file.

## Build options

Compile-time options are CMake arguments, named after the Cargo feature in upper case with a `PELORUS_` prefix:

```sh
colcon build --packages-select pelorus --cmake-args -D PELORUS_CUDA=ON
```

`PELORUS_CUDA`, `PELORUS_METAL`, `PELORUS_VULKAN` and `PELORUS_LARGE_PROCESSING_QUEUE` can also be set as environment variables of the same name.

| Option | Effect |
| --- | --- |
| `PELORUS_CUDA` | CUDA acceleration. The CUDA directory is searched for, and `CUDA_HOME` overrides the result. |
| `PELORUS_VULKAN` | Vulkan acceleration. See the [Vulkan prerequisites](install.md#vulkan-prerequisites). |
| `PELORUS_METAL` | Metal acceleration on macOS with Apple Silicon. |
| `PELORUS_RERUN` | Compile [Rerun](https://rerun.io/) visualization support, activated at runtime with `--rerun`. |
| `PELORUS_MIMALLOC` | Use mimalloc as the allocator. |
| `PELORUS_LARGE_PROCESSING_QUEUE` | Larger internal processing queues. |
| `PELORUS_NO_VIZ` | Drop visualization support entirely. |

`rat` and `rayon` are on by default. `rat` synchronizes input and output for use with [Minot](https://codeberg.org/stelzo/minot) through the `-r` flag, and `rayon` uses all available CPU cores. Turn both off with `-D PELORUS_ALL_DEFAULTS=OFF`.

Rerun's server address and map settings live under `output.rerun` in the configuration file and can be overridden per run, for example with `-o output.rerun.server_addr="192.168.1.2:9876"`.

# Standalone

Pelorus runs on Linux and macOS without ROS. The only thing it needs is a source of data, which can be a recording, a live sensor, or your own application feeding it through the C, C++ or Python bindings.

This page gets Pelorus running on a recording first, then covers the library bindings.

## Running a recording

The quickest path uses the SeaKit tools, which we built during Pelorus' development to fix determinism and efficiency problems we had with ROS. [Installing Pelorus with uv](./index.md#how-to-run) gives you everything you need at once. This includes:

- [`marina`](https://marina.steado.tech/docs/) for managing bags
- `dedrunk` for IMU intrinsics and lidar, camera and IMU extrinsics
- [`minot`](https://minot.steado.tech/docs/) for synchronized playback of bags

First we get our dataset:

```sh
marina pull wachsbleiche/parking_25:hunter
```

!!! tip
    Just run `marina` without any additional command for a TUI of available datasets and `marina completions` to install autocompletions in your shell.

Now run it:

```sh
pelorus offline wachsbleiche/parking_25:hunter -c $(marina resolve wachsbleiche/parking_25:hunter)/pelorus.rl
```

### Shell completions

```sh
pelorus completions
```

This detects the current shell and installs self-refreshing completions for the current user. Restart the shell afterwards.

## Running with the ROS 2 Zenoh middleware

Thanks to [hiroz](https://github.com/ZettaScaleLabs/hiroz) we can connect to other ROS 2 nodes without needing a ROS installation.

The default `pelorus` and `minot` builds talk Minot's own publish/subscribe, not ROS 2. For Zenoh, replace them with builds that have the hiroz features. Remove the default ones the way you installed them, for example with `uv pip uninstall pelorus-cli minot-cli` in the environment of the quick start. Building the replacements needs [Rust](https://www.rust-lang.org/tools/install).

```sh
cargo install minot --no-default-features --features default-hiroz

cargo install --git https://codeberg.org/stelzo/pelorus pelorus_cli --no-default-features --features rayon,offline-hiroz
```

Start a Zenoh router:

```sh
cargo install zenohd@1.10.1
zenohd
```

Run Pelorus in online mode...

```sh
pelorus online -c $(marina resolve wachsbleiche/parking_25:hunter)/pelorus.rl
```

... and play a bag into it
```sh
minot async $(marina resolve wachsbleiche/parking_25:hunter) --clock --rate 1.0
```

RViz, started from a ROS-enabled shell, then shows the result with the Pelorus view:

```sh
curl -fsSLO https://codeberg.org/stelzo/pelorus/raw/branch/main/pelorus_cli/rviz/lio.rviz
rviz2 -d lio.rviz
```

## C and C++ library

The C library is built and installed with CMake and ships a C++ header alongside the C one.

```sh
cd pelorus_c
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
sudo cmake --install build
```

Acceleration backends are CMake options on this build:

| Option | Effect |
| --- | --- |
| `PELORUS_FEATURE_CUDA` | CUDA acceleration |
| `PELORUS_FEATURE_VULKAN` | Vulkan acceleration, see the [prerequisites](install.md#vulkan-from-source) |
| `PELORUS_FEATURE_METAL` | Metal acceleration, macOS only |
| `PELORUS_FEATURE_RAYON` | CPU parallelism, on by default |
| `PELORUS_FEATURE_LARGE_PROCESSING_QUEUE` | Larger internal processing queues |

```sh
cmake -S . -B build -DPELORUS_FEATURE_CUDA=ON
```

In your own project, link against the installed package:

```cmake
find_package(pelorus REQUIRED)

add_executable(my_app main.c)
target_link_libraries(my_app pelorus::pelorus)
```

`pkg-config` and `add_subdirectory` both work as well. Include `pelorus.h` from C and `pelorus.hpp` from C++, where the header adds helpers for Eigen and PCL. Working examples for the synchronous and the callback-driven API are in `pelorus_c/examples`, including one that feeds Eigen and PCL types directly.

## Python

The Python package is on [PyPI](install.md#pypi) and takes the same configuration file as everything else. It is the fastest way to try an idea against a recording.

```sh
uv venv
uv pip install "pelorus[all]"
```

`pelorus` is a library, so it goes into an environment. The `pelorus` command
comes from a separate package, `pelorus-cli`, which the `all` extra pulls in.

Examples covering the blocking API, the callback API and reading from an MCAP bag are in `pelorus_py/examples`.


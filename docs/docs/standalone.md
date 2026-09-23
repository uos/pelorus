# Standalone

Pelorus runs on Linux and macOS without ROS. The only thing it needs is a source of data, which can be a recording, a live sensor, or your own application feeding it through the C, C++ or Python bindings.

This page gets Pelorus running on a recording first, then covers the library bindings.

## Running a recording

The quickest path uses the SeaKit tools, which were built during Pelorus development to fix determinism and efficiency problems we had with ROS. Everything below is installed from source with `cargo install`, so install [Rust](https://www.rust-lang.org/tools/install) first.

**Marina** fetches recordings:

```sh
cargo install --git git://git.steado.tech/marina --locked

marina pull wachsbleiche/parking_25:hunter
```

We recommend adding the [tab completions](https://stelzo.codeberg.page/marina/features/shell/completions.html). As with all of SeaKit, the documentation is also available over SSH with `ssh marina@steado.tech`.

**Minot** is what Pelorus uses to read bags and process them synchronously:

```sh
cargo install --git git://git.steado.tech/minot minot --tag v0.11.1 --locked
```

**Rerun** is optional and gives you a GUI for visualization:

```sh
cargo install rerun-cli@0.38.0
```

Then the CLI itself:

```sh
cargo install --git ssh://git.steado.tech/pelorus pelorus_cli --locked
```

That is everything. Run it:

```sh
pelorus offline wachsbleiche/parking_25:hunter -c $(marina resolve wachsbleiche/parking_25:hunter)/pelorus.rl
```

### Shell completions

```sh
pelorus completions
```

This detects the current shell and installs self-refreshing completions for the current user. Restart the shell afterwards.

??? note "Generating completion files for packaging or a system-wide install"

    Pass the shell explicitly to write the file yourself.

    === "Bash"

        === "Linux"

            ~~~bash
            pelorus completions bash | sudo tee /etc/bash_completion.d/pelorus > /dev/null
            ~~~

        === "macOS (Homebrew)"

            ~~~bash
            pelorus completions bash > $(brew --prefix)/etc/bash_completion.d/pelorus
            ~~~

        Reload your shell afterwards, or source the file:

        ~~~bash
        source /etc/bash_completion.d/pelorus
        ~~~

    === "Zsh"

        === "Linux"

            ~~~bash
            mkdir -p ~/.zsh/completions
            pelorus completions zsh > ~/.zsh/completions/_pelorus
            ~~~

        === "macOS (Homebrew)"

            ~~~bash
            pelorus completions zsh > $(brew --prefix)/share/zsh/site-functions/_pelorus
            ~~~

        Add the following to `~/.zshrc` **before** the `compinit` call:

        ~~~bash
        fpath=(~/.zsh/completions $fpath)
        autoload -U compinit && compinit
        ~~~

    === "Fish"

        === "Linux"

            ~~~bash
            pelorus completions fish > ~/.config/fish/completions/pelorus.fish
            ~~~

        === "macOS (Homebrew)"

            ~~~bash
            pelorus completions fish > $(brew --prefix)/share/fish/completions/pelorus.fish
            ~~~

        Fish picks completions up automatically.

## Running against ROS 2 without installing ROS 2

Both Minot and Pelorus can speak ROS 2 over the Zenoh middleware, which lets an existing ROS stack talk to a Pelorus build that has no ROS dependency at all. Install both with the Hiroz feature set. Minot is the same release as above, built with different features:

```sh
cargo install --git git://git.steado.tech/minot minot --tag v0.11.1 --no-default-features --features default-hiroz

cargo install --git ssh://git.steado.tech/pelorus pelorus_cli --no-default-features --features rayon,offline-hiroz
```

Start a Zenoh router:

```sh
cargo install zenohd@1.9.0
zenohd
```

Run Pelorus in online mode and play a bag into it:

```sh
pelorus online -c $(marina resolve wachsbleiche/parking_25:hunter)/pelorus.rl
```

```sh
minot async $(marina resolve wachsbleiche/parking_25:hunter) --clock --rate 1.0
```

RViz, started from a ROS-enabled shell, then shows the result:

```sh
rviz2 -d ./pelorus_cli/rviz/lio.rviz
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
| `PELORUS_FEATURE_VULKAN` | Vulkan acceleration, see the [prerequisites](install.md#vulkan-prerequisites) |
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

The Python package is on [PyPI](install.md#python) and takes the same configuration file as everything else. It is the fastest way to try an idea against a recording.

```sh
uv venv
uv pip install "pelorus[all]"
```

`pelorus` is a library, so it goes into an environment. The `pelorus` command
comes from a separate package, `pelorus-cli`, which the `all` extra pulls in.

Examples covering the blocking API, the callback API and reading from an MCAP bag are in `pelorus_py/examples`.

## Where to look next

- [Configuration](configuration/index.md) for the file every mode reads.
- [Calibration](calibration/index.md) once the pipeline runs and you want the drift gone.

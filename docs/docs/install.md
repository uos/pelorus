# Installation

## Ubuntu packages

Install Pelorus from the [steado PPA](https://ppa.steado.tech/) for your Ubuntu LTS version and receive updates automatically. Every package comes in a plain CPU build and in CUDA and Vulkan variants.

??? note "Set up the PPA"

    Run this once on Ubuntu 22.04, 24.04 or 26.04.

    ~~~sh
    curl -fsSL "https://ppa.steado.tech/ubuntu/key.gpg" | gpg --dearmor \
      | sudo tee /usr/share/keyrings/steado-archive-keyring.gpg >/dev/null
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/steado-archive-keyring.gpg] https://ppa.steado.tech/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
      | sudo tee /etc/apt/sources.list.d/steado.list
    sudo apt update
    ~~~

=== "ROS 2 node"

    Available for Humble, Jazzy and Lyrical.

    ~~~sh
    sudo apt install ros-$ROS_DISTRO-pelorus

    # or with CUDA support on Humble and Jazzy
    sudo apt install ros-$ROS_DISTRO-pelorus-cuda12-6
    sudo apt install ros-$ROS_DISTRO-pelorus-cuda12-8
    sudo apt install ros-$ROS_DISTRO-pelorus-cuda13-1

    # or with CUDA support on Lyrical
    sudo apt install ros-lyrical-pelorus-cuda13-4

    # or with Vulkan support
    sudo apt install ros-$ROS_DISTRO-pelorus-vulkan
    ~~~

=== "CLI and C library"

    ~~~sh
    sudo apt install pelorus

    # or with CUDA support
    sudo apt install pelorus-cuda12-6
    sudo apt install pelorus-cuda12-8
    sudo apt install pelorus-cuda13-1

    # or with Vulkan support
    sudo apt install pelorus-vulkan
    ~~~

=== "GLIM plugin"

    GLIM upstream supports CUDA 12.6 and 12.8, so the plugin is offered for
    those two only, even though the library itself also builds for 13.1.

    ~~~sh
    sudo apt install pelorus-glim-odometry

    # or with CUDA support
    sudo apt install pelorus-glim-odometry-cuda12-6
    sudo apt install pelorus-glim-odometry-cuda12-8

    # or with Vulkan support
    sudo apt install libpelorus-glim-odometry-vulkan
    ~~~

## PyPI

Two packages are published on PyPI, and which one you want depends on what you
are doing. `pelorus` is the library you import. `pelorus-cli` carries the
`pelorus` executable. Every index below holds both, built with the acceleration
backend its name says.

We recommend `uv`:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

=== "The `pelorus` command"

    `uv tool` installs applications, so the package to name here is
    `pelorus-cli`. The `rerun` extra adds the Rerun viewer, so we can use
    `output.rerun = true`.

    ~~~sh
    uv tool install "pelorus-cli[rerun]"

    # or with CUDA support
    uv tool install "pelorus-cli[rerun]" --extra-index-url https://ppa.steado.tech/whl/cu126/
    uv tool install "pelorus-cli[rerun]" --extra-index-url https://ppa.steado.tech/whl/cu128/
    uv tool install "pelorus-cli[rerun]" --extra-index-url https://ppa.steado.tech/whl/cu131/

    # or with Vulkan support
    uv tool install "pelorus-cli[rerun]" --extra-index-url https://ppa.steado.tech/whl/vulkan/
    ~~~

=== "The Python library"

    This one goes into a project or a virtual environment. The `all` extra adds
    MCAP bag reading and pulls in the CLI as well, so the `pelorus` command ends
    up on the environment's path too.

    ~~~sh
    uv venv
    uv pip install "pelorus[all]"

    # or with CUDA support
    uv pip install "pelorus[all]" --extra-index-url https://ppa.steado.tech/whl/cu126/
    uv pip install "pelorus[all]" --extra-index-url https://ppa.steado.tech/whl/cu128/
    uv pip install "pelorus[all]" --extra-index-url https://ppa.steado.tech/whl/cu131/

    # or with Vulkan support
    uv pip install "pelorus[all]" --extra-index-url https://ppa.steado.tech/whl/vulkan/

    # without uv
    pip install "pelorus[all]"
    ~~~

    Inside a uv project, `uv add "pelorus[all]"` does the same and records the
    dependency. The extras are `mcap` for bag reading, `cli` for the executable,
    and `all` for both.

## macOS

The [uos Homebrew tap](https://github.com/uos/homebrew-tools) carries the CLI
and the C library for Apple Silicon, built with Metal support. It pulls in
[`minot`](https://minot.steado.tech/docs/), `dedrunk` and [`marina`](https://marina.steado.tech/docs/) as well.

```sh
brew install uos/tools/pelorus
```

## Arch Linux

The [`pelorus-bin`](https://aur.archlinux.org/packages/pelorus-bin) package on
the AUR installs the CPU build of the CLI and the C library for x86_64 and
aarch64. Install it with an AUR helper of your choice.

```sh
yay -S pelorus-bin
```

## Prebuilt archives

Every [release on Codeberg](https://codeberg.org/stelzo/pelorus/releases)
comes with `.tar.gz` archives of the CLI and the C library, for amd64 and arm64
where the platform offers both.

| Archive | Contents |
| --- | --- |
| `pelorus-<version>-linux-<series>-<arch>` | CLI and C library for Ubuntu `resolute`, `noble` or `jammy` |
| `pelorus-<version>-linux-<series>-<arch>-cuda<cuda>` | The same with CUDA 12.6, 12.8 or 13.1, for `noble` and `jammy` |
| `pelorus-<version>-linux-<series>-<arch>-vulkan` | The same with Vulkan |
| `pelorus-<version>-linux-musl-<arch>` | Statically linked CLI for any Linux distribution |
| `pelorus-<version>-linux-musl-dds-<arch>` | Static CLI that runs as a ROS 2 node over DDS |
| `pelorus-<version>-linux-musl-hiroz-<arch>` | Static CLI that talks to ROS 2 over Zenoh without a ROS installation, see [standalone](standalone.md) |
| `pelorus-<version>-macos-arm64` | CLI, C library, headers and CMake files for Apple Silicon with Metal |
| `pelorus-glim-<version>-linux-<series>-<arch>` | GLIM plugin for `noble` and `jammy` |

The musl and macOS archives are mirrored at
[ppa.steado.tech/bin](https://ppa.steado.tech/bin/).

## Vulkan from source

Vulkan needs some runtime dependencies to run on your GPU. A working CUDA installation, or `nvidia-smi` producing output, does not automatically provide Vulkan support.

```sh
sudo apt update
sudo apt install libvulkan1 vulkan-tools
```

When building Pelorus from source, install the development packages as well:

```sh
sudo apt install build-essential pkg-config libshaderc-dev libvulkan-dev libvulkan1 vulkan-tools
```

Then install a GPU driver:

- **NVIDIA.** Install the `libnvidia-gl-<branch>` package matching your installed driver. Despite the name, this package contains the Vulkan ICD, which is the vendor driver the loader uses. Compute-only installations often lack it. Check your driver branch and the available package versions first, and do not copy a version number from elsewhere:

    ```sh
    nvidia-smi
    apt-cache policy libnvidia-gl-570   # replace 570 with your branch
    ```

    See the [Ubuntu package description](https://packages.ubuntu.com/noble/amd64/libnvidia-gl-570) and the [NVIDIA package repository](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/) for details.

- **AMD and Intel.** Install `mesa-vulkan-drivers`.

### Verifying

```sh
vulkaninfo --summary
```

Under `Devices`, check that `deviceName` names your intended GPU and that `deviceType` is a discrete or integrated GPU. Pelorus requires Vulkan 1.2 or newer.

If the only device is `llvmpipe` with `PHYSICAL_DEVICE_TYPE_CPU`, Vulkan is running through Mesa's Lavapipe software implementation, which is known to cause problems with Pelorus.

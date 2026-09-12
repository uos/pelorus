# Binaries

The Pelorus beta is currently available as precompiled binaries for Linux on x86_64 and aarch64 (ARM64) but highly experimental.

Since the binaries may have problems during the current beta, we recommend building from source. Follow the related instructions from the other pages for that.

You can [setup the UOS PPA](https://codeberg.org/uos-robotics/ppa/src/branch/pages/README.md) for the Ubuntu LTS versions to install Pelorus and get updates automatically.

```sh title="ROS 2 package for Humble and Jazzy"
sudo apt install ros-$ROS_DISTRO-pelorus

# or with CUDA support
sudo apt install ros-$ROS_DISTRO-pelorus-cuda12-8
sudo apt install ros-$ROS_DISTRO-pelorus-cuda13-1
```

```sh title="Standalone CLI and C library"
sudo apt install pelorus

# or with CUDA support
sudo apt install pelorus-cuda12-8
sudo apt install pelorus-cuda13-1
```

```sh title="GLIM odometry plugin"
sudo apt install pelorus-glim-odometry

# or with CUDA support
sudo apt install pelorus-glim-odometry-cuda12-8
sudo apt install pelorus-glim-odometry-cuda13-1
```

---

Python packages are available on PyPI. We recommend `uv` to manage the installation. Get `uv` first if you don't have it already:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh

# create a virtual environment for Pelorus
uv venv .venv
```

```sh title="Python packages"
# With rerun for visualization and standalone cli
uv tool install "pelorus[all]"

# or with CUDA support
uv tool install "pelorus[all]" --extra-index-url https://uos-robotics.codeberg.page/ppa/pelorus/pypi/cuda12-8/simple
uv tool install "pelorus[all]" --extra-index-url https://uos-robotics.codeberg.page/ppa/pelorus/pypi/cuda13-1/simple

# Without uv (not recommended)
pip install "pelorus[all]"
```

## Vulkan

Vulkan needs both the Vulkan loader and a working Vulkan driver for your GPU. A working CUDA installation or successful `nvidia-smi` output alone does not establish Vulkan support.

### Ubuntu prerequisites

Install the build dependencies and the tool used to check device detection:

```sh
sudo apt update
sudo apt install build-essential pkg-config libshaderc-dev libvulkan-dev libvulkan1 vulkan-tools
```

Install a GPU driver as well:

- **NVIDIA:** install the `libnvidia-gl-<branch>` package matching your installed NVIDIA driver. Despite its name, this package includes the Vulkan ICD (the vendor driver used by the Vulkan loader). Compute-only installations can lack it.
- **AMD / Intel:** install `mesa-vulkan-drivers` and verify that your GPU is detected below.

For NVIDIA, inspect the current driver and available package versions first:

```sh
nvidia-smi
# Example for driver branch 570; replace 570 with your installed branch.
apt-cache policy libnvidia-gl-570
```

Do not blindly use this historical version on a different driver installation. See the [Ubuntu package description](https://packages.ubuntu.com/noble/amd64/libnvidia-gl-570) and [NVIDIA package repository](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/) for package details.

### Verify the GPU before running Pelorus

```sh
vulkaninfo --summary
```

Under `Devices`, check that `deviceName` names your intended GPU and `deviceType` is a discrete or integrated GPU. Pelorus requires Vulkan 1.2 or newer.

If the only device is `llvmpipe` with `PHYSICAL_DEVICE_TYPE_CPU`, Vulkan is running through Mesa's Lavapipe software implementation, which has been known to cause issues with Pelorus.


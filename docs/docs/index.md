![Alt text](images/pelorus_banner.png)

# Introduction

**Pelorus** is a resource-aware *Lidar-Inertial Odometry*, written in Rust, CUDA, Vulkan and Metal. It is developed by the Computer Engineering Group at the Osnabrück University.

The online mode provides more than 20 Hz odometry on an Jetson Orin Nano with an 128-layer Ouster Lidar while keeping resource consumption to a minimum. Pelorus aims to *just run* in the background so the main application can use the resources for the actual task at hand.

You can run Pelorus on embedded devices (like the NVIDIA Jetson series) as well as on desktop systems with or without GPUs, supporting NVIDIA, AMD, Apple and Intel.

## First look

Gifs und so

## Getting started

Pelorus is a fully functional LIO system that is meant to be used in real-world applications.

We provide multiple ways to use Pelorus for different use-cases. Each one supports CPU-only systems as well as GPU-accelerated setups:

- **[ROS 2 package]()** for easy integration in existing ROS 2 systems
- **[Standalone C library]()** with CMake integration and C++ bindings (including helpers for Eigen and PCL) for custom applications
- **[Python package]()** published on PyPI for easy prototyping and offline processing
- **[Standalone CLI]()** with offline and online modes
- **[GLIM odometry plugin]()** for easy and accurate mapping


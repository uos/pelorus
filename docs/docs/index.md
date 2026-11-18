---
hide:
  - toc
---

<div class="sk-intro">
  <div class="sk-intro-copy">

<div class="sk-wordmark"><img src="images/pelorus.svg" alt="" width="72" height="72"><h1>Pelorus</h1></div>

<p class="sk-tagline">Resource-Aware<br>Lidar-Inertial Odometry</p>

<p>Accurate motion estimation that leaves room for the rest of your application.
From embedded robots to desktop systems, with optional GPU acceleration.</p>

<div class="sk-actions">
  <a class="md-button md-button--primary" href="https://pelorus.steado.tech">Try in your browser ↗</a>
  <a class="md-button" href="#how-to-run">Get started</a>
</div>

<p class="sk-note">Run Pelorus on a recording, directly in your browser.</p>

  </div>
  <figure class="sk-intro-visual">
    <img src="images/pelorus-header.webp" alt="Pelorus lidar reconstruction of the Institute of Computer Science at Osnabrück University, with the trajectory shown in yellow." width="1346" height="1020">
    <figcaption>Institute of Computer Science · Osnabrück University</figcaption>
  </figure>
</div>

<div class="sk-highlights">
  <div><strong>Accurate odometry</strong><span>Lidar and IMU motion estimation.</span></div>
  <div><strong>Lightweight &amp; fast</strong><span>20+ Hz on Jetson Orin Nano with a 128-layer Ouster lidar.</span></div>
  <div><strong>GPU accelerated</strong><span>CUDA, Vulkan and Metal, alongside the CPU pipeline.</span></div>
</div>

<p class="sk-links">Runs as: <a href="ros/">ROS 2</a> · <a href="standalone/">Standalone CLI &amp; bag playback</a> · <a href="standalone/#c-and-c-library">C / C++</a> · <a href="standalone/#python">Python</a> · <a href="glim/">GLIM</a> · <a href="nostd/">Embedded</a></p>

## About

**Pelorus**{ .small-caps } is a resource-aware lidar-inertial odometry written in Rust with optional GPU support for NVIDIA, AMD, Apple and Intel. It is developed and maintained by the [Computer Engineering Group](https://www.uni-osnabrueck.de/fb6/informatik/en/research-groups/computer-engineering) at Osnabrück University.

The online mode delivers more than 20 Hz odometry on a Jetson Orin Nano with a 128-layer Ouster lidar while keeping resource consumption to a minimum. Pelorus is meant to run in the background so that the main application can use the machine for the task at hand. It runs on embedded devices such as the NVIDIA Jetson series and on desktop systems with or without a GPU, supporting NVIDIA, AMD, Apple and Intel.

## How to run

For a quick start, install the following package containing the CLI, Python library, playback and visualization tools.

```bash
uv pip install "pelorus[all]"
```

In a real scenario, Pelorus can run in several ways: 

- **[ROS 2 package](ros.md)**: drop-in integration for existing ROS 2 systems
- **[Standalone CLI](standalone.md)**: offline and online modes, no ROS required
- **[C and C++ library](standalone.md#c-and-c-library)**: CMake integration, with helpers for Eigen and PCL
- **[Python package](standalone.md#python)**: on PyPI, for prototyping and offline processing
- **[GLIM plugin](glim.md)**: odometry source for [GLIM](https://koide3.github.io/glim/) mapping
- **[Embedded](nostd.md)**: `no_std` Rust library for microcontrollers

Prebuilt packages for Ubuntu, PyPI, Homebrew and the AUR, and release archives, are listed under [installation](install.md).

All you need to run Pelorus now is the dataset and a configuration for your sensors and topics. For this demo, we will provide everything for you. Pelorus uses SeaKit by default for all its tooling. 

The dataset can be downloaded with [`marina`](https://marina.steado.tech/docs/), which was installed with the Pelorus package.

```bash
marina pull wachsbleiche/parking_25:hunter
```

Now we can run the method.
```bash
# optional gui visualization
rerun

# in other terminal
pelorus offline wachsbleiche/parking_25:hunter \
    -c $(marina resolve wachsbleiche/parking_25:hunter)/pelorus.rl \
    -o output.rerun=true
```

Pelorus will run the odometry as fast as possible through the offline pipeline. You can go back in time in the GUI to view real-time results. You can get there by selecting "sensor_time" and choosing "1x" speed.

Sometimes you'll want to see more than just the odometry. You can enable the local map visualization by appending `-o output.rerun.map=true` to the above command. Mind that this will slow down the process. Even more so when running with GPU acceleration.

The local map visuals are meant for debugging [local map sizes](configuration/reference/registration.md#registration-map-capacity), since Pelorus' local map (and Pelorus in general) is intended for localization, not mapping. This is odometry after all. For pretty maps with global consistency, you can use the [GLIM plugin](glim.md). A full SLAM is in preparation.


## Architecture

<!-- Two rules keep this diagram readable.

     The highlight fill is translucent. The theme colours mermaid labels with
     .nodeLabel { color: var(--md-mermaid-label-fg-color) }, which is dark in
     the light scheme and light in the dark one, and it wins over anything a
     classDef sets. A translucent fill lets that label colour show through in
     both schemes. A solid fill would need a fixed label colour, which is
     unreadable in one of them. The stroke carries the emphasis.

     Keep the diagram plain mermaid. A YAML frontmatter config block keeps it
     from rendering and shows the source as a code block. -->

```mermaid
flowchart TB
    IN["Lidar + IMU"] --> PRE["Filter<br/>and deskew"]
    PRE --> DS{{"Downsample"}}
    DS --> NN{{"Neighbour search<br/>and plane fit"}}
    NN --> EQF["Equivariant<br/>filter update"]
    EQF --> OUT["Odometry, TF<br/>registered cloud"]
    EQF --> INS{{"Map insert<br/>and eviction"}}
    INS --> MAP[("Distribution Map")]
    MAP --> NN
    classDef gpu fill:#e5c65c40,stroke:#b68e24,stroke-width:3px
    class DS,NN,INS,MAP gpu
```

Everything highlighted can run on the GPU. Each of those stages has a CUDA, a Vulkan and a Metal implementation next to its CPU one. A GPU binary runs the same pipeline at the same accuracy.

The map holds a [fixed number of voxels](configuration/reference/registration.md#registration-map-capacity)
and evicts using a least-relevance criterion. Memory is therefore bounded by configuration. That is what makes a long run on a Jetson predictable.

The state is estimated through an [iterated equivariant Kalman filter](https://arxiv.org/abs/2409.06948). Wheel odometry or kinematic constraints can be added for additional constraints, which helps most where the geometry is degenerate, such as a long corridor. Wheel odometry is still unstable due to a lack of test data.


## Getting started

1. Install Pelorus one of the ways above.
2. Generate a [configuration](configuration/index.md) and set the topics, frame IDs and sensor parameters for your hardware.
3. Run it against a recording before running it on a robot.
4. Once it works, [calibrate the IMU](calibration/index.md). It removes drift that tuning cannot.

## Citation

The paper is under review at IEEE Robotics and Automation Letters (RA-L). Please cite the [arXiv preprint](https://arxiv.org/abs/2611.22850) when you use Pelorus:

```bibtex
@misc{sieh2026pelorus,
  title         = {Pelorus: Resource-Aware Lidar-Inertial Odometry},
  author        = {Sieh, Charlotte and Porrmann, Mario},
  year          = {2026},
  eprint        = {2611.22850},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2611.22850},
}
```

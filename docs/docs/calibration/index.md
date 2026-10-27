# Calibration

Calibration is optional. Pelorus produces a usable trajectory on the defaults with a well mounted sensor pair. Calibrating the IMU removes drift that tuning cannot.

| Measurement | Recording | Fit | Pelorus setting |
| --- | --- | --- | --- |
| Scale, axis misalignment, bias | Minutes, sensor held still in many orientations | `dedrunk intrinsics imu` | `imu.calibration.parameter_file` |
| Noise densities and bias random walks | Hours, sensor completely undisturbed | `dedrunk imu noise` | `imu.covariance_*` |
| Lidar to IMU transform and clock offset | Minutes of driving, turning and tilting | `dedrunk extrinsics lidar-imu` | `imu.from_lidar_transform`, `imu.calibration.time_offset` |

[Dedrunk](https://stelzo.codeberg.page/dedrunk/) does the fitting. It reads MCAP recordings and ROS 2 bags and writes Kalibr format YAML. It ships with `pelorus[all]` and the Homebrew package, and installs on its own with `cargo install dedrunk`. Its documentation covers the motion each fit needs, how to read the results and what to do when a fit refuses. This page covers the Pelorus side: recording and applying the results.

The steps can be done in any order. The extrinsic fit is cleaner with the intrinsics already applied.

## Recording

`pelorus online raw` runs the native sensor drivers without the odometry and records their output unchanged. It publishes on the normal topics while recording, so the rest of your stack keeps working.

```sh
# Both sensors, for the intrinsics and the extrinsics
pelorus online raw --output-rosbag calib_drive

# IMU only, for the noise measurement
pelorus online raw imu --output-rosbag imu_static
```

One drive with both sensors covers intrinsics and extrinsics if it follows [Dedrunk's recording guide](https://stelzo.codeberg.page/dedrunk/recording/). The noise measurement needs its own recording of at least three hours. `dedrunk topics <bag>` lists what was recorded. `--output-mcap` works as well.

!!! warning "Calibrate against raw IMU output"

    `pelorus online raw` records the driver output without AHRS filtering. Never
    calibrate against `imu/filtered` or any other topic that went through an
    orientation filter.

For the intrinsics, `dedrunk imu record` listens while `pelorus online raw imu` publishes, prompts for each orientation, and fits the recording once all are captured:

```sh
pelorus online raw imu &
dedrunk imu record /ouster/imu --bag calib_intrinsics.mcap -O imu_intrinsics.yaml
```

## IMU intrinsics and noise

```sh
dedrunk intrinsics imu calib_drive -O imu_intrinsics.yaml
dedrunk imu noise imu_static -O imu_noise.yaml
dedrunk imu merge imu_intrinsics.yaml imu_noise.yaml -O imu.yaml --pelorus imu.rl
```

`--pelorus imu.rl` writes the Pelorus side of the result as Ratslang: `imu.calibration.parameter_file` pointing at `imu.yaml`, and the four `imu.covariance_*` values. Include it from the main configuration instead of copying values over:

```ratslang
= ./imu.rl
```

```ratslang
imu.calibration {
    parameter_file = "./imu.yaml"

    # Measure the gyroscope bias at start-up instead of trusting the file.
    gyroscope = true
    gyroscope.samples = 100
}
```

Pelorus applies the scale, misalignment and bias in the file to every IMU sample. Keep the robot still for the first second after start-up while the gyroscope bias is measured.

`dedrunk imu noise` also prints the four `imu.covariance_*` values derived from the measurement, ready to paste, and writes them with `--pelorus PATH`. The defaults are tuning values, not a noise model of any sensor, so the measured ones are optional. They come from a still sensor, and Dedrunk inflates them by `noise.inflation` (10 by default, `-o noise.inflation=20` to change it) to account for vibration and temperature changes on a moving robot. Compare trajectories on a recording you know before keeping them.

## Lidar to IMU extrinsics

The extrinsic fit needs a trajectory produced without the IMU, since a lidar-inertial trajectory already assumes an extrinsic:

```sh
pelorus offline calib_drive -c pelorus.rl -o imu.lidar_only=true -o output.mcap.rosbag_dir=lidar_traj
dedrunk extrinsics lidar-imu lidar_traj --imu-bag calib_drive --pelorus lidar_imu.rl
```

[`imu.lidar_only`](../configuration/reference/imu.md#imu-lidar-only) ignores the IMU and `imu.from_lidar_transform`. Roll and pitch are relative to the start pose in this mode, so use it only for calibration.

Read [Dedrunk's observability report](https://stelzo.codeberg.page/dedrunk/extrinsics/#results) before accepting the output. A level loop drive does not constrain every parameter, and an existing calibration is better than an unconstrained replacement. Dedrunk prints a block with `imu.from_lidar_transform` and `imu.calibration.time_offset` to paste into the configuration, and `--pelorus lidar_imu.rl` writes the same block to a file to include with `= ./lidar_imu.rl`. Includes resolve relative to the including file, and a setting assigned after an include overrides it.

[`imu.calibration.time_offset`](../configuration/reference/imu.md#imu-calibration-time-offset) is how far the IMU clock runs ahead of the Lidar clock. Pelorus subtracts it from every IMU timestamp. The effect grows with turn rate. At 1 rad/s, 10 ms of skew rotates the scan by about half a degree.

!!! warning "Paste the time offset exactly as Dedrunk prints it, sign included"

    A flipped sign doubles the skew. It shows up as smeared scans while turning
    and is easily mistaken for a registration problem.

### Refining online

With a good starting extrinsic, from CAD or from Dedrunk, the filter can refine it while running:

```ratslang
imu.from_lidar_transform.estimate_online = true
```

It needs a close initial guess and enough excitation. On a robot that mostly drives straight it drifts, so compare the result against the starting value.

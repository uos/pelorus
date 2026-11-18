# Calibration

Calibration is optional. With a well mounted lidar and IMU, Pelorus runs fine on the defaults. Calibrating the IMU removes drift that tuning cannot.

You can calibrate three things, in any order:

- **[IMU intrinsics](#imu-intrinsics)**: scale, axis misalignment and bias of the IMU. Needs a few minutes of recording.
- **[IMU noise](#imu-noise)**: how noisy the IMU is. Needs a few hours with the sensor completely still.
- **[Lidar to IMU](#lidar-to-imu)**: where the IMU sits relative to the lidar, and how far their clocks are apart. Needs a few minutes of driving.

If you do both, apply the intrinsics before fitting the lidar to IMU transform. The fit comes out cleaner.

Dedrunk does the fitting. It comes with `pelorus[all]` and the Homebrew package, or on its own with `cargo install dedrunk`. Its documentation explains the motion each fit needs and how to read the results. This page covers the Pelorus side: recording and applying the results.

## Recording

`pelorus online raw` runs the sensor drivers without the odometry and records what they publish. The topics stay up while it records, so the rest of your stack keeps working.

```sh
# Lidar and IMU, for the intrinsics and the lidar to IMU fit
pelorus online raw --output-rosbag calib_drive

# IMU only, for the noise measurement
pelorus online raw imu --output-rosbag imu_static
```

One drive that follows Dedrunk's recording guide covers both the intrinsics and the lidar to IMU fit. The noise measurement needs its own recording of at least three hours. Use `--output-mcap` for MCAP instead of a ROS 2 bag, and `dedrunk topics <bag>` to see what was recorded.

!!! warning "Calibrate against raw IMU output"

    `pelorus online raw` records the IMU without AHRS filtering. Never calibrate
    against `imu/filtered` or any other topic that went through an orientation
    filter.

## IMU intrinsics

Fit them from the drive:

```sh
dedrunk intrinsics imu calib_drive -O imu_intrinsics.yaml
```

Or let Dedrunk guide you through the orientations while the IMU publishes. It records and fits once all orientations are captured:

```sh
pelorus online raw imu &
dedrunk imu record /ouster/imu --bag calib_intrinsics.mcap -O imu_intrinsics.yaml
```

## IMU noise

```sh
dedrunk imu noise imu_static -O imu_noise.yaml
```

The noise values are optional. The defaults are tuning values, not a model of any particular sensor. Measured on a still sensor, the noise is lower than on a moving robot, so Dedrunk multiplies it by `noise.inflation`, 10 by default. Change it with `-o noise.inflation=20`. Compare trajectories on a recording you know before keeping the measured values.

## Applying the IMU calibration

Merge the two results and let Dedrunk write the Pelorus settings:

```sh
dedrunk imu merge imu_intrinsics.yaml imu_noise.yaml -O imu.yaml --pelorus imu.rl
```

`imu.rl` is a [Ratslang](https://ratslang.steado.tech/docs/) file that points `imu.calibration.parameter_file` at `imu.yaml` and sets the four `imu.covariance_*` values. Include it from your configuration:

```ratslang
= ./imu.rl
```

To measure the gyroscope bias at start-up instead of taking it from the file, add:

```ratslang
imu.calibration {
    gyroscope = true
    gyroscope.samples = 100
}
```

Then keep the robot still for the first second after start-up.

## Lidar to IMU

This fit needs a trajectory from the lidar alone, because a lidar-inertial trajectory already assumes a transform. Run Pelorus with [`imu.lidar_only`](../configuration/reference/imu.md#imu-lidar-only) on the drive, using the configuration you recorded it with, then fit:

```sh
pelorus offline calib_drive -c pelorus.rl -o imu.lidar_only=true -o output.mcap.rosbag_dir=lidar_traj
dedrunk extrinsics lidar-imu lidar_traj --imu-bag calib_drive --pelorus lidar_imu.rl
```

`imu.lidar_only` ignores the IMU and its transform, and roll and pitch are relative to the start pose. Use it only for calibration.

Read Dedrunk's observability report before you accept the result. A flat loop does not pin down every parameter, and a known calibration is better than a poorly constrained new one.

`lidar_imu.rl` sets `imu.from_lidar_transform` and [`imu.calibration.time_offset`](../configuration/reference/imu.md#imu-calibration-time-offset). Include it like the IMU file:

```ratslang
= ./lidar_imu.rl
```

Includes resolve relative to the including file. A setting after the include overrides it.

The time offset is how far the IMU clock runs ahead of the lidar clock. Its effect grows with turn rate: at 1 rad/s, 10 ms of offset rotates a scan by about half a degree.
### Refining while running

With a good starting transform, from CAD or from Dedrunk, Pelorus can refine it while it runs:

```ratslang
imu.from_lidar_transform.estimate_online = true
```

This needs a close starting guess and enough rotation in all axes. On a robot that mostly drives straight it drifts, so compare the result with the starting value.

# Scale, misalignment and bias

Two minutes of holding the sensor still in different orientations. dedrunk fits the IMU model from [Rehder et al., ICRA 2016](https://doi.org/10.1109/ICRA.2016.7487628) using the multi-position procedure from [Tedaldi et al., ICRA 2014](https://doi.org/10.1109/ICRA.2014.6907297).

## Guided recording

```sh
pelorus online raw imu intrinsics --output-rosbag calib_intrinsics
```

The CLI prompts for each orientation, detects when the sensor has settled and records it. It asks for twelve orientations by default and `--poses` changes the count. Nine is the minimum.

- Spread the orientations over all octants.
- Rotate about a different axis between each one.

If `dedrunk` is on your `PATH`, the CLI offers to run the fit at the end. Otherwise it prints the command.

## Fitting an existing recording

Any recording with raw IMU data works, including the [shared calibration drive](index.md#recording), as long as it starts at rest. The first four seconds measure the noise floor.

```sh
dedrunk intrinsics calib_drive -o output.path='"imu_intrinsics.yaml"'
```

dedrunk finds the stationary stretches itself and prints the residuals:

```
poses: 12   duration: 42.5 s   rate: 200.0 Hz
accelerometer: rms 0.0021 m/s^2, max 0.0043 m/s^2
gyroscope:     rms 0.021°, max 0.055° over 11 segments
```

The accelerometer rms should be a few mm/s² for a decent MEMS part. Much higher values mean the orientations were too similar or the surface was not solid.

## Sensor that cannot be moved freely

If only the six axis aligned faces are reachable, use the reduced model. It skips misalignment and needs six orientations.

```sh
dedrunk intrinsics calib_drive -o intrinsics.accel_model='"scale-bias"'
```

## Applying it

Merge the file with the [noise measurement](imu-noise.md) and set [`imu.calibration.parameter_file`](../configuration/reference/imu.md#imu-calibration-parameter-file), see [applying the result](index.md#applying-the-result).

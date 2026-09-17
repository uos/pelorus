# Scale, misalignment and bias

This is the measurement that changes the numbers Pelorus feeds into the filter, and it is the cheapest one to record. Two minutes of holding the sensor still in different orientations is enough.

## Guided recording

The CLI can walk you through it. It prompts for each orientation, detects when the sensor has settled, captures the samples, and offers to run the fit at the end.

```sh
pelorus online raw imu intrinsics --output-rosbag calib_intrinsics
```

Twelve orientations are requested by default. Nine is the minimum for the full model, and `--poses` changes the count. Rotate about a different axis between each one and spread the orientations over all octants. The gyroscope scale is recovered from the rotations between the static poses, so those rotations have to cover every axis.

If `dedrunk` is on your `PATH`, the CLI offers to fit the recording when it finishes. Otherwise it prints the command to run later.

## Fitting an existing recording

Any bag that contains the raw IMU works, including the [shared calibration drive](index.md#recording), as long as it starts with the sensor already at rest. The first four seconds measure the noise floor that tells a hold apart from a move.

```sh
dedrunk intrinsics calib_drive -o output.path='"imu_intrinsics.yaml"'
```

dedrunk finds the stationary stretches on its own, so there is nothing to mark up. It reports how many it found together with the residuals:

```
poses: 12   duration: 42.5 s   rate: 200.0 Hz
accelerometer: rms 0.0021 m/s^2, max 0.0043 m/s^2
gyroscope:     rms 0.021°, max 0.055° over 11 segments
```

The accelerometer residual should land near the sensor's own noise level, which is a few mm/s² for a decent MEMS part. Anything much worse means the orientations were not spread out enough, or the surface was not solid.

## When the sensor cannot be moved freely

If the IMU is bolted down and you can only reach the six axis aligned faces, use the reduced model. It drops the misalignment terms and needs six orientations instead of nine.

```sh
dedrunk intrinsics calib_drive -o intrinsics.accel_model='"scale-bias"'
```

## Applying it

The result is a Kalibr format file. Merge it with the [noise measurement](imu-noise.md) and point [`imu.calibration.parameter_file`](../configuration/reference/imu.md#imu-calibration-parameter-file) at it, as described under [applying the result](index.md#applying-the-result).

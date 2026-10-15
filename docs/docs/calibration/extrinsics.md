# Lidar to IMU extrinsics

dedrunk estimates the transform between Lidar and IMU, the time offset, gravity and both biases from a recording of the two sensors moving together. It follows [Zhu, Ren and Zhang, "Robust Real-time Lidar-inertial Initialization", IROS 2022](https://arxiv.org/abs/2202.11006).

It needs a Lidar-only trajectory, because Lidar-inertial poses already assume an extrinsic:

```sh
pelorus offline calib_drive -c pelorus.rl -o imu.lidar_only=true -o output.mcap.rosbag_dir=lidar_traj
dedrunk extrinsics lidar_traj --imu-bag calib_drive
```

[`imu.lidar_only`](../configuration/reference/imu.md#imu-lidar-only) ignores the IMU and `imu.from_lidar_transform`. Roll and pitch are relative to the start pose in this mode, so use it only for calibration.

dedrunk prints the result ready to paste:

```ratslang
imu.calibration.time_offset = -0.005900s
imu {
  from_lidar_transform {
    translation = [0.0379m, -0.0480m, 0.1972m]
    rotation = [ [+0.999995, -0.001268, +0.003056],
                 [-0.001272, -0.999998, +0.001479],
                 [+0.003055, -0.001483, -0.999994] ]
  }
}
```

## Recording

For good calibration results, you'll need to move the sensors around together.

- **Rotation**: turn about at least two axes. An ordinary loop drive is enough.
- **Translation**: rotate briskly. Sharp direction changes help more than long turns.
- **Gravity and accelerometer bias**: tilt the platform. A robot that stays level cannot separate them.

A wheeled robot on flat ground gets the rotation and time offset but not the translation. For the translation, lift the sensor head if you can and roll, pitch and yaw it for a couple of minutes. Keep enough structure in view for the Lidar odometry and stay still for the first few seconds.

## Results

Check these before trusting the numbers:

```
gravity:      [+0.723, +0.239, -6.267] m/s^2  (|g| = 6.313)
residuals:    acceleration 4.126 m/s^2
attitude spread:                       2.1°  (separates gravity from accel bias)
angular excitation (rms, Lidar frame): [0.030, 0.037, 0.143] rad/s
warning: the fitted gravity is 6.31 m/s^2 against an expected 9.81, off by 36%...
warning: the platform's attitude varies by only 2.1° over the recording, so gravity
and the accelerometer bias cannot be told apart...
```

If `|g|` is far from 9.81, the translation, gravity and accelerometer bias are unusable. The rotation and time offset come from the gyroscope alone and are still valid. Aim for `|g|` near 9.81 and no warnings.

## Time offset

[`imu.calibration.time_offset`](../configuration/reference/imu.md#imu-calibration-time-offset) is how far the IMU clock runs ahead of the Lidar clock. Pelorus subtracts it from every IMU timestamp.

!!! warning "Paste the value exactly as dedrunk prints it, sign included"

    A flipped sign doubles the skew. It shows up as smeared scans while turning
    and is easily mistaken for a registration problem.

The effect grows with turn rate. At 1 rad/s, 10 ms of skew rotates the scan by about half a degree.

## Refining online

With a good starting extrinsic, from CAD or from dedrunk, the filter can refine it while running:

```ratslang
imu.from_lidar_transform.estimate_online = true
```

It needs a close initial guess and enough excitation. On a robot that mostly drives straight it drifts, so compare the result against the starting value.

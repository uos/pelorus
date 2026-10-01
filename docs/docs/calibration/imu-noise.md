# Noise densities and random walks

The white noise densities and bias random walks of the accelerometer and gyroscope. They tell the filter how far to trust the IMU between Lidar corrections. dedrunk reads them off the Allan deviation like Kalibr and `allan_variance_ros` do, so the values are directly comparable. See the [Kalibr IMU noise model](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model) and [El-Sheimy et al., 2008](https://doi.org/10.1109/TIM.2007.908635) for the background.

## Recording

Leave the sensor completely undisturbed for at least three hours. Longer is better. Fans, compressors or people walking nearby spoil the result, so pick a solid floor in a quiet room.

```sh
pelorus online raw imu --output-rosbag imu_static
```

The `imu` argument records only the IMU and keeps the Lidar off, which keeps the recording small. Stop with ++ctrl+c++ and check with `dedrunk topics imu_static` that the whole recording is there.

## Computing

```sh
dedrunk noise imu_static -o output.path='"imu_noise.yaml"'
```

```
/imu/data: 4320000 samples over 10800.0 s (400.0 Hz)

accelerometer_noise_density: 2.027539e-3
accelerometer_random_walk:   9.985073e-4
gyroscope_noise_density:     3.053876e-4
gyroscope_random_walk:       2.169208e-4
update_rate:                 400.00 Hz

written to /home/you/imu_noise.yaml
Allan deviation curves in allan.csv
```

## When it refuses

dedrunk warns below three hours and refuses when the curve has no usable random walk slope:

```
error: the Allan deviation of the accelerometer x axis has no clear slope +0.5 region
(0 point(s) spanning 0.0x in tau, need 4 spanning 1.5x). Either the recording is too
short to reach the bias random walk, or this sensor does not show one over the cluster
times covered
```

Record longer, or extend the cluster range:

```sh
dedrunk noise imu_static -o noise.cluster.max=3000s
```

Keep `noise.cluster.max` below a tenth of the recording length.

## Checking the curve

`allan.csv` has one row per cluster time:

```
tau,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,pairs
```

Plot the axes against `tau` on log-log axes. A healthy curve falls, flattens and rises again. Bumps or an early rise mean the sensor was disturbed. All three axes should look similar.

## Using datasheet values

Without time for a long recording, take the values from the datasheet. Angular random walk and velocity random walk are the white noise densities. In-run bias instability stands in for the random walk. Write them into a Kalibr file by hand and merge as usual.

Datasheet values are optimistic. Inflate them by a factor of ten or more for low-cost MEMS parts.

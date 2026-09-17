# Calibration

Pelorus assumes three things about your IMU that are not true out of the box: that its axes are square and correctly scaled, that you know how noisy it is, and that you know where it sits relative to the Lidar. Each one is a separate measurement, and each has its own way of failing quietly.

[dedrunk](https://github.com/stelzo/dedrunk) does the measuring. It reads MCAP recordings and writes Kalibr-format YAML, which Pelorus reads back through `imu.calibration.parameter_file` and `imu.from_lidar_transform`.

```sh
cargo install --path <checkout>/dedrunk
```

| What | Command | What you record |
| --- | --- | --- |
| Scale, axis misalignment, biases | `dedrunk intrinsics` | minutes, sensor held still in many orientations |
| Noise densities, bias random walks | `dedrunk noise` | hours, sensor completely undisturbed |
| Lidar-to-IMU transform | `dedrunk extrinsics` | minutes, rotating and tilting both sensors together |

The first two are properties of the IMU alone. The third is about the pair. You can do them in any order, but the extrinsic fit is cleaner once the intrinsics are applied.

## Getting the data into a bag

Two of the three need a bag with both sensors in it, and your robot already writes one — dedrunk picks the topics apart by message type. Noise is the exception and wants its own recording. [See below](#noise).

```sh
pelorus online raw --output-rosbag calib_drive
```

That publishes on the normal topics *and* records, so nothing else in your stack has to change. `sensor_msgs/msg/Imu` must be the **raw** device output: `pelorus online raw` publishes exactly what the driver reported, with no AHRS filtering, which is what you want here. Never calibrate against `imu/filtered`.

Check what landed:

```sh
dedrunk topics calib_drive
```

Both a `.mcap` file and a ROS 2 bag directory work as input, so `--output-rosbag` and `--output-mcap` are equally fine.

## Scale, misalignment and biases

This is the one that changes the numbers Pelorus feeds into the filter, and it is the cheapest to record.

Hold the sensor still in **at least nine well-spread orientations**, rotating about a different axis between each one. Start with it already at rest — the first four seconds measure the noise floor that tells a hold apart from a move. Two minutes is enough.

```sh
dedrunk intrinsics calib_drive -o output.path='"imu_intrinsics.yaml"'
```

dedrunk finds the stationary stretches itself. There is nothing to mark up. It prints how many it found and the residuals:

```
poses: 12   duration: 42.5 s   rate: 200.0 Hz
accelerometer: rms 0.0021 m/s^2, max 0.0043 m/s^2
gyroscope:     rms 0.021°, max 0.055° over 11 segments
```

The accelerometer residual should land near the sensor's own noise level — a few mm/s² for a decent MEMS part. Much worse than that means the orientations were not spread out, or the surface was not solid.

If the IMU is bolted down and you can only manage the six axis-aligned faces, use the reduced model:

```sh
dedrunk intrinsics calib_drive -o intrinsics.accel_model='"scale-bias"'
```

That drops the misalignment terms and needs six orientations instead of nine.

## Noise

These four numbers say how much the IMU makes up: two white-noise densities and two bias random walks. They are **not** applied to measurements — noise is irreducible, and you cannot subtract a value you do not know sample by sample. What they do is tell the filter how far to trust the IMU between Lidar corrections. Too small and it rides the IMU through a corner and ignores the scan match. Too large and it throws away good inertial information.

### What to record

Park the robot, start the recording, and **leave the room**. Three hours is the practical minimum, longer is better, and the sensor must be genuinely undisturbed for all of it — a nearby fan, a compressor cycling, or somebody walking past on a suspended floor is enough to spoil the long end of the curve. A basement slab beats a workshop bench.

Record the IMU **only**:

```sh
pelorus online raw imu --output-rosbag imu_static
```

The `imu` argument matters here. Three hours of IMU at 400 Hz is a couple of gigabytes. Three hours with the Lidar running is closer to a hundred, and none of it would be used. `pelorus online raw imu` also leaves the Lidar socket closed, so the machine is doing nothing that could vibrate the sensor.

It records until you stop it with `Ctrl-C`. Confirm afterwards that you got the whole thing:

```sh
dedrunk topics imu_static
```

### Computing

```sh
dedrunk noise imu_static -o output.path='"imu_noise.yaml"'
```

That is the whole procedure. The result:

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

dedrunk reads the values off the Allan deviation exactly the way Kalibr and `allan_variance_ros` do — white noise from the `−1/2` slope at `τ = 1 s`, bias random walk from the `+1/2` slope at `τ = 3 s` — so the numbers are directly comparable with anything else in that ecosystem. The estimator is streaming, keeping running accumulators per cluster time rather than the samples, so a 24-hour recording costs no more memory than a ten-minute one.

It warns if you gave it less than three hours, and it **refuses rather than guessing** when the curve does not actually contain a region of the slope it needs:

```
error: the Allan deviation of the accelerometer x axis has no clear slope +0.5 region
(0 point(s) spanning 0.0x in tau, need 4 spanning 1.5x). Either the recording is too
short to reach the bias random walk, or this sensor does not show one over the cluster
times covered
```

Two things cause that. Either the recording is too short — the random walk only emerges at cluster times of tens of seconds, so you need hours to average enough of them — or the cluster range does not reach far enough:

```sh
dedrunk noise imu_static -o noise.cluster.max=3000s
```

Keep `noise.cluster.max` **below a tenth of the recording length**. Past that there are too few independent clusters for the estimate to mean anything, and dedrunk says so.

### Checking the curve

`allan.csv` has one row per cluster time, with a column per axis:

```
tau,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,pairs
```

Plot the axis columns against `tau` on log-log axes in whatever you normally use, and look at the shape rather than the numbers.

A healthy curve falls at slope `−1/2`, flattens through a minimum — the bias instability — and rises again at `+1/2`. What you are checking is that both sloped regions are actually there and reasonably long. A curve that only ever falls means no measurable random walk over the range covered. A curve with bumps or a rise that starts early usually means the sensor was disturbed, and the recording is not usable.

The three axes should look similar. One that stands out is worth investigating before you use the averaged value.

### If you cannot record for hours

Datasheet figures are a legitimate fallback. Manufacturers usually quote white noise as **angular random walk** for the gyroscope and **velocity random walk** for the accelerometer, which are the same quantities under different names. Bias figures are rarer, and **in-run bias instability** is the usual stand-in. Write them into a Kalibr file by hand and merge as below.

Expect them to be optimistic. The Kalibr wiki suggests inflating measured densities by 10× or more for low-cost MEMS parts, and datasheet values need at least as much.

## Combining and applying

```sh
dedrunk merge imu_intrinsics.yaml imu_noise.yaml -o output.path='"imu.yaml"'
```

Then point Pelorus at the result:

```ratslang
imu.calibration {
    parameter_file = "./imu.yaml"

    # Bias moves with temperature and between power cycles, so the stored value
    # is the weakest part of the file. Re-measuring beats trusting it.
    gyroscope = true
    samples = 100
}
```

Keep the robot stationary for the first second after start-up. Those samples measure the gyroscope bias and are dropped rather than published.

### Covariances from the file

Once the file carries the noise half, Pelorus can derive `imu.covariance_*` from it instead of the tuned constants:

```ratslang
imu.calibration {
    parameter_file.use_noise = true
    noise_inflation = 10.0
}
```

This is **off by default**, and deliberately so. `Q` is sandwiched with `dt²` in the propagation, so it holds discrete-time variances — `(density × inflation)² × update_rate` — and the inflation is a tuning choice, not a derivable one: Allan variance is measured on a still sensor at constant temperature, and a moving robot has vibration, thermal drift and timing jitter on top.

Pelorus logs the derived and the configured values side by side at start-up so you can see the ratios for your sensor before committing:

```
IMU covariances from ./imu.yaml (inflation 10x): gyroscope 1.800e-3 (was 1.000e-1),
acceleration 8.000e-2 (was 1.000e-1), gyroscope bias 8.000e-6 (was 1.000e-4),
acceleration bias 3.200e-3 (was 1.000e-4)
```

Note that the four ratios there are nothing like each other — the gyroscope term lands 55× below the shipped constant while the accelerometer bias lands 32× above it. The defaults Pelorus inherits are FAST-LIO tuning values, not one consistent noise model, so no single inflation reproduces them. Switch this on against a bag you know the answer for and compare trajectories before adopting it.

## Lidar-to-IMU extrinsics

The transform has to be measured against a trajectory that did **not** already assume it. Pelorus's own odometry is a Lidar-inertial one, so feeding its poses back would return the assumption instead of measuring it. Run it Lidar-only first:

```sh
pelorus offline calib_drive -c pelorus.rl -o imu.lidar_only=true -o output.mcap.rosbag_dir=lidar_traj
dedrunk extrinsics lidar_traj --imu-bag calib_drive
```

`imu.lidar_only` ignores the IMU and propagates the filter at a constant twist between scans instead. Everything else — deskewing, the voxel map, the registration — is unchanged, so the trajectory is as good as your normal tuning makes it. It also ignores `imu.from_lidar_transform`, because without an IMU there is nothing for that transform to point at.

Two notes on the plumbing: this is not a navigation mode (with no IMU the gravity direction is unobservable, so roll and pitch are relative to wherever the sensor started), and the covariances stop describing sensor noise and start describing how far the constant-twist model can be trusted, so raise them.

dedrunk prints the result as configuration you can paste straight in:

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

### The time offset

`imu.calibration.time_offset` is the constant skew between the two sensors' clocks — how far the IMU's timestamps run ahead of the Lidar's for the same physical instant. Pelorus subtracts it from every IMU timestamp as the sample enters, before anything reads that time, so the integration step, the undistortion window and the scan pairing all agree.

**Paste the number exactly as dedrunk prints it, sign included.** Getting the sign backwards doubles the skew instead of removing it, and the symptom — slightly smeared scans while turning — looks like a registration problem rather than a timing one.

It matters in proportion to how fast you turn: at 1 rad/s, 10 ms of skew drags the scan through about half a degree, which the registration then has to absorb every frame. Standing still it costs nothing, which is why a bad offset can hide in a gentle dataset and bite on an aggressive one.

Unlike the transform, this comes out of the gyroscope alone, so it is trustworthy from the same ordinary drive that gives you the rotation — no tilting required.

### What the recording has to contain

This is where an extrinsic calibration goes wrong, and the two halves of the transform need different things:

* **Rotation** needs the platform to *turn*, about at least two axes. An ordinary drive around a loop is enough — this half works on data you already have.
* **Translation** — the lever arm — enters the model only through `S(ẇ) + S(w)²`. Without rotation the IMU and the Lidar accelerate identically and their separation leaves no trace at all. Rotation about an axis makes the lever-arm components *perpendicular* to it observable and leaves the one along it invisible. Rotate briskly: the term goes with `w²`, so sharp direction changes are worth more than sustained turning.
* **Gravity and the accelerometer bias** need something else again. They enter as `−R·Cᵀ·g` and `+b_a`, so separating them needs the platform's *attitude* to change enough that `Cᵀ·g` swings around the body frame. **Tilting, not turning.** A robot that stays level holds `Cᵀ·g` almost constant, and then the two are one unknown wearing two hats.

That last point is the trap, because a wheeled robot on flat ground satisfies the first requirement and fails the third.

### Reading the diagnostics

Look at these before the numbers:

```
gravity:      [+0.723, +0.239, -6.267] m/s^2  (|g| = 6.313)
residuals:    acceleration 4.126 m/s^2
attitude spread:                       2.1°  (separates gravity from accel bias)
angular excitation (rms, Lidar frame): [0.030, 0.037, 0.143] rad/s
```

**Gravity is the check, not a result.** dedrunk fits it as a free vector precisely so its magnitude can be compared against 9.81. If it is off, everything the accelerometer contributed — translation, gravity, accelerometer bias — is worthless together, and dedrunk says so:

```
warning: the fitted gravity is 6.31 m/s^2 against an expected 9.81, off by 36%...
warning: the platform's attitude varies by only 2.1° over the recording, so gravity
and the accelerometer bias cannot be told apart...
```

The rotation and the time offset come from the gyroscope alone and survive this untouched — both are worth taking from a recording whose accelerometer half failed. On a real loop drive with a wheeled robot, the rotation came out within 0.2° of the surveyed mounting while the translation was unusable — a normal and correct outcome for that recording.

To get the translation, record a couple of minutes with the sensor head **tilted through a range**: lifted off the robot if you can, rolled and pitched as well as yawed. Aim for `|g|` near 9.81 and no warnings.

### Refining online

If you already have a good extrinsic from CAD or from the above, the filter can keep refining it while it runs:

```ratslang
imu.from_lidar_transform.estimate_online = true
```

Off by default. This is refinement, not calibration: it needs a starting guess close enough to converge, and the extrinsic is only observable while the platform is being excited. On a robot that mostly drives straight it will drift along the unobservable directions rather than improve, so check the result against what you started with.

//! WebAssembly bindings for Pelorus Lidar-Inertial Odometry.
//!
//! Wraps the thread-free [`pelorus::Odometer`] for JavaScript. Every call runs
//! on the calling thread and returns once its work is done.
//!
//! # Building
//!
//! ```sh
//! cargo build -p pelorus_wasm --release --target wasm32-unknown-unknown
//! wasm-bindgen --target web --out-dir pkg \
//!     target/wasm32-unknown-unknown/release/pelorus_wasm.wasm
//! ```
//!
//! # Example
//!
//! ```js
//! import init, { Odometer } from "./pkg/pelorus_wasm.js";
//! await init();
//!
//! const odometer = new Odometer(configSource);
//! odometer.addImu(sec, nanosec, orientation, angularVelocity, linearAcceleration);
//! odometer.addPointsXyzit(sec, nanosec, points); // Float32Array, 5 values per point
//! for (let scan = odometer.nextScan(); scan; scan = odometer.nextScan()) {
//!   console.log(scan.position, odometer.points().length / 5);
//! }
//! ```
//!
//! # Replaying an MCAP bag
//!
//! [`McapBag`] reads a bag the user picked, a range at a time:
//!
//! ```js
//! const read = async (bag) => bag.supply(new Uint8Array(
//!   await file.slice(bag.readOffset(), bag.readOffset() + bag.readLength()).arrayBuffer()));
//!
//! const bag = new McapBag(file.size);
//! while (!bag.loadSummary()) await read(bag);
//! bag.play("/hesai/pandar", "/alphasense_driver_ros/imu");
//! for (let step; (step = bag.next(odometer)) !== BagStep.End; ) {
//!   if (step === BagStep.NeedData) await read(bag);
//!   else for (let scan = odometer.nextScan(); scan; scan = odometer.nextScan()) draw(scan);
//! }
//! ```

use pelorus::ImuMeasurement;
use wasm_bindgen::prelude::*;

mod bag;
pub use bag::{BagStep, McapBag, TopicInfo};

/// Numbers in [`Scan::to_array`].
pub const SCAN_ARRAY_LEN: usize = 20;

/// Values per point in [`Odometer::points`]: x, y, z, intensity and the time
/// within the scan.
pub const POINT_STRIDE: usize = 5;

fn js_error(error: impl core::fmt::Display) -> JsError {
    JsError::new(&format!("{error:#}"))
}

/// `values` as a fixed-size array, or an error naming `what`.
fn exact<const N: usize>(values: &[f64], what: &str) -> Result<[f64; N], JsError> {
    values
        .try_into()
        .map_err(|_| js_error(format!("{what} needs {N} values, got {}", values.len())))
}

/// Lidar-inertial odometry for JavaScript.
///
/// Add sensor input with the `add*` methods, then call
/// [`nextScan`](Odometer::next_scan) until it returns `undefined`.
#[wasm_bindgen]
pub struct Odometer {
    inner: pelorus::Odometer,
    /// Reused by [`world_points`](Odometer::world_points).
    world_points: Vec<f32>,
    occupied_voxels: rustc_hash::FxHashSet<(i32, i32, i32)>,
}

#[wasm_bindgen]
impl Odometer {
    /// Create an odometer. `config` is `.rl` source layered over the embedded
    /// defaults. Without `config` the defaults are used. The execution mode is
    /// always single-core, since the browser runs one thread.
    ///
    /// With `deviceSpecificPointcloud2 = false` the per-point time is a
    /// nanosecond offset from the scan stamp. It defaults to `true`, which
    /// reads the time the way the configured Lidar vendor writes it.
    #[wasm_bindgen(constructor)]
    pub fn new(
        config: Option<String>,
        device_specific_pointcloud2: Option<bool>,
    ) -> Result<Odometer, JsError> {
        console_error_panic_hook::set_once();
        let device_specific = device_specific_pointcloud2.unwrap_or(true);
        // The browser runs one thread, so configs written for Rayon run single-core.
        let source = format!(
            "{}\nparallelism.execution_mode = SingleCore\n",
            config.unwrap_or_default()
        );
        let inner =
            pelorus::Odometer::from_config_source(&source, device_specific).map_err(js_error)?;
        Ok(Odometer {
            inner,
            world_points: Vec::new(),
            occupied_voxels: Default::default(),
        })
    }

    /// Add an IMU measurement. `orientation` is a quaternion `[w, x, y, z]`.
    ///
    /// `covariances` holds the row-major 3×3 orientation, angular velocity and
    /// linear acceleration covariances, 27 values. Without it they are zero.
    #[wasm_bindgen(js_name = addImu)]
    pub fn add_imu(
        &mut self,
        sec: i32,
        nanosec: u32,
        orientation: &[f64],
        angular_velocity: &[f64],
        linear_acceleration: &[f64],
        covariances: Option<Box<[f64]>>,
    ) -> Result<(), JsError> {
        let covariances: [f64; 27] = match covariances {
            Some(values) => exact(&values, "covariances")?,
            None => [0.0; 27],
        };
        let covariance = |block: usize| -> [f64; 9] {
            covariances[block * 9..(block + 1) * 9]
                .try_into()
                .expect("the covariances hold three 3×3 blocks")
        };
        self.inner.add_imu(ImuMeasurement {
            timestamp_sec: sec,
            timestamp_nanosec: nanosec,
            orientation: exact(orientation, "orientation")?,
            orientation_covariance: covariance(0),
            angular_velocity: exact(angular_velocity, "angularVelocity")?,
            angular_velocity_covariance: covariance(1),
            linear_acceleration: exact(linear_acceleration, "linearAcceleration")?,
            linear_acceleration_covariance: covariance(2),
        });
        Ok(())
    }

    /// Add a wheel-odometry twist in the IMU frame. `covariance` is the
    /// row-major 6×6 twist covariance.
    #[wasm_bindgen(js_name = addOdom)]
    pub fn add_odom(
        &mut self,
        sec: i32,
        nanosec: u32,
        linear: &[f64],
        angular: &[f64],
        covariance: &[f64],
    ) -> Result<(), JsError> {
        self.inner.add_odom(
            sec,
            nanosec,
            exact(linear, "linear")?,
            exact(angular, "angular")?,
            exact(covariance, "covariance")?,
        );
        Ok(())
    }

    /// Add a point cloud with 5 values per point: x, y, z, intensity and the
    /// time `t` in nanoseconds after the scan stamp.
    ///
    /// Returns the scan ID, or `0n` for a dropped cloud.
    #[wasm_bindgen(js_name = addPointsXyzit)]
    pub fn add_points_xyzit(
        &mut self,
        sec: i32,
        nanosec: u32,
        points: &[f32],
    ) -> Result<u64, JsError> {
        let (points, []) = points.as_chunks::<5>() else {
            return Err(js_error("points need 5 values per point"));
        };
        Ok(self.inner.add_points_xyzit(sec, nanosec, points))
    }

    /// Add a point cloud with 4 values per point: x, y, z and the time `t` in
    /// nanoseconds after the scan stamp. Intensity is zero.
    ///
    /// Returns the scan ID, or `0n` for a dropped cloud.
    #[wasm_bindgen(js_name = addPointsXyzt)]
    pub fn add_points_xyzt(
        &mut self,
        sec: i32,
        nanosec: u32,
        points: &[f32],
    ) -> Result<u64, JsError> {
        let (points, []) = points.as_chunks::<4>() else {
            return Err(js_error("points need 4 values per point"));
        };
        Ok(self.inner.add_points_xyzt(sec, nanosec, points))
    }

    /// Register the next scan whose IMU interval is complete. Returns
    /// `undefined` once no queued scan can be completed with the input added
    /// so far.
    #[wasm_bindgen(js_name = nextScan)]
    pub fn next_scan(&mut self) -> Option<Scan> {
        self.inner.next_scan().map(|scan| Scan::from(&scan))
    }

    /// Enable GPU correspondence search. Rejects unsupported configurations;
    /// callers can catch that rejection and continue using CPU processing.
    #[cfg(feature = "webgpu")]
    #[wasm_bindgen(js_name = enableWebGpu)]
    pub async fn enable_webgpu(&mut self) -> Result<(), JsError> {
        #[cfg(target_arch = "wasm32")]
        {
            use std::future::{Future, poll_fn};
            use std::task::Poll;
            let mut initialization = std::pin::pin!(self.inner.enable_webgpu());
            let mut timeout = std::pin::pin!(gloo_timers::future::TimeoutFuture::new(15_000));
            // Cancel inside Rust so the mutable odometer borrow is released before
            // JavaScript catches the rejection and resumes CPU processing.
            poll_fn(|cx| {
                if let Poll::Ready(result) = initialization.as_mut().poll(cx) {
                    return Poll::Ready(result.map_err(js_error));
                }
                if timeout.as_mut().poll(cx).is_ready() {
                    return Poll::Ready(Err(js_error(
                        "WebGPU initialization timed out after 15 seconds",
                    )));
                }
                Poll::Pending
            })
            .await
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.inner.enable_webgpu().await.map_err(js_error)
    }

    #[cfg(feature = "webgpu")]
    #[wasm_bindgen(js_name = webGpuStatus)]
    pub fn webgpu_status(&self) -> Result<String, JsError> {
        self.inner
            .webgpu_status()
            .map(str::to_owned)
            .map_err(js_error)
    }

    #[cfg(feature = "webgpu")]
    #[wasm_bindgen(js_name = nextScanAsync)]
    pub async fn next_scan_async(&mut self) -> Option<Scan> {
        self.inner
            .next_scan_async()
            .await
            .map(|scan| Scan::from(&scan))
    }

    /// The deskewed points of the scan the last [`nextScan`](Self::next_scan)
    /// returned, in the Lidar frame, with [`POINT_STRIDE`] values per point:
    /// x, y, z, intensity and the time within the scan.
    ///
    /// The array is a view into the odometer's memory, not a copy. It is only
    /// valid until the next call to any odometer method. Use `.slice()` to keep
    /// the points.
    pub fn points(&self) -> Option<js_sys::Float32Array> {
        let scan = self.inner.last_scan()?;
        let values: &[f32] = bytemuck::cast_slice(scan.points());
        // SAFETY: the view is documented to be valid only until the next call
        // into the odometer, which is the only way the buffer can change or the
        // wasm memory can grow.
        Some(unsafe { js_sys::Float32Array::view(values) })
    }

    /// The points of the scan the last [`nextScan`](Self::next_scan) returned,
    /// in the world frame, with 4 values per point: x, y, z and intensity.
    ///
    /// With `voxelSize` above zero only the first point in each cube of that
    /// edge length in metres is kept. Unlike [`points`](Self::points) this is a
    /// copy, which stays valid and can be posted to another thread.
    #[wasm_bindgen(js_name = worldPoints)]
    pub fn world_points(&mut self, voxel_size: f32) -> Option<js_sys::Float32Array> {
        let scan = self.inner.last_scan()?;
        self.world_points.clear();
        self.occupied_voxels.clear();
        let inverse = if voxel_size > 0.0 {
            1.0 / voxel_size
        } else {
            0.0
        };
        for p in scan.world_points() {
            if inverse > 0.0 {
                let voxel = (
                    (p.x * inverse).floor() as i32,
                    (p.y * inverse).floor() as i32,
                    (p.z * inverse).floor() as i32,
                );
                if !self.occupied_voxels.insert(voxel) {
                    continue;
                }
            }
            self.world_points
                .extend_from_slice(&[p.x, p.y, p.z, p.intensity]);
        }
        Some(js_sys::Float32Array::from(&self.world_points[..]))
    }

    /// Number of scans waiting for IMU data.
    #[wasm_bindgen(js_name = pendingScans)]
    pub fn pending_scans(&self) -> usize {
        self.inner.pending_scans()
    }

    /// The Lidar vendor the configuration sets, for example `Hesai`.
    #[wasm_bindgen(js_name = lidarVendor)]
    pub fn lidar_vendor(&self) -> String {
        format!("{:?}", self.inner.config().sync_params.lidar_vendor)
    }

    /// The Pelorus version.
    pub fn version() -> String {
        env!("CARGO_PKG_VERSION").to_owned()
    }
}

/// Odometry of one registered scan.
///
/// The world frame is Z-up and aligned with the IMU at initialization. The
/// body frame is the configured base link, or the Lidar frame without one.
#[wasm_bindgen]
pub struct Scan {
    scan_id: u64,
    sec: i32,
    nanosec: u32,
    position: [f64; 3],
    orientation: [f64; 4],
    pose_covariance: [f64; 36],
    linear_velocity: [f64; 3],
    angular_velocity: [f64; 3],
    twist_covariance: [f64; 36],
    accel_bias: [f64; 3],
    gyro_bias: [f64; 3],
    frame_id: String,
}

impl From<&pelorus::Scan<'_>> for Scan {
    fn from(scan: &pelorus::Scan<'_>) -> Self {
        let (sec, nanosec) = scan.timestamp();
        let (linear_velocity, angular_velocity) = scan.twist();
        Scan {
            scan_id: scan.scan_id(),
            sec,
            nanosec,
            position: scan.position(),
            orientation: scan.orientation(),
            pose_covariance: *scan.pose_covariance(),
            linear_velocity,
            angular_velocity,
            twist_covariance: *scan.twist_covariance(),
            accel_bias: scan.accel_bias(),
            gyro_bias: scan.gyro_bias(),
            frame_id: scan.frame_id().to_owned(),
        }
    }
}

#[wasm_bindgen]
impl Scan {
    /// Scan ID matching the input point cloud.
    #[wasm_bindgen(getter, js_name = scanId)]
    pub fn scan_id(&self) -> u64 {
        self.scan_id
    }

    /// Seconds of the end of the scan.
    #[wasm_bindgen(getter)]
    pub fn sec(&self) -> i32 {
        self.sec
    }

    /// Nanoseconds of the end of the scan.
    #[wasm_bindgen(getter)]
    pub fn nanosec(&self) -> u32 {
        self.nanosec
    }

    /// Position `[x, y, z]` of the body frame in world frame (metres).
    #[wasm_bindgen(getter)]
    pub fn position(&self) -> js_sys::Float64Array {
        js_sys::Float64Array::from(&self.position[..])
    }

    /// Orientation quaternion `[w, x, y, z]` of the body frame.
    #[wasm_bindgen(getter)]
    pub fn orientation(&self) -> js_sys::Float64Array {
        js_sys::Float64Array::from(&self.orientation[..])
    }

    /// 6×6 pose covariance (row-major).
    #[wasm_bindgen(getter, js_name = poseCovariance)]
    pub fn pose_covariance(&self) -> js_sys::Float64Array {
        js_sys::Float64Array::from(&self.pose_covariance[..])
    }

    /// Linear velocity `[vx, vy, vz]` in body frame (m/s).
    #[wasm_bindgen(getter, js_name = linearVelocity)]
    pub fn linear_velocity(&self) -> js_sys::Float64Array {
        js_sys::Float64Array::from(&self.linear_velocity[..])
    }

    /// Angular velocity `[wx, wy, wz]` in body frame (rad/s).
    #[wasm_bindgen(getter, js_name = angularVelocity)]
    pub fn angular_velocity(&self) -> js_sys::Float64Array {
        js_sys::Float64Array::from(&self.angular_velocity[..])
    }

    /// 6×6 twist covariance (row-major).
    #[wasm_bindgen(getter, js_name = twistCovariance)]
    pub fn twist_covariance(&self) -> js_sys::Float64Array {
        js_sys::Float64Array::from(&self.twist_covariance[..])
    }

    /// Accelerometer bias `[bx, by, bz]` in IMU frame (m/s²).
    #[wasm_bindgen(getter, js_name = accelBias)]
    pub fn accel_bias(&self) -> js_sys::Float64Array {
        js_sys::Float64Array::from(&self.accel_bias[..])
    }

    /// Gyroscope bias `[bx, by, bz]` in IMU frame (rad/s).
    #[wasm_bindgen(getter, js_name = gyroBias)]
    pub fn gyro_bias(&self) -> js_sys::Float64Array {
        js_sys::Float64Array::from(&self.gyro_bias[..])
    }

    /// Frame whose pose is reported.
    #[wasm_bindgen(getter, js_name = frameId)]
    pub fn frame_id(&self) -> String {
        self.frame_id.clone()
    }

    /// The scan as [`SCAN_ARRAY_LEN`] numbers, for example to post it to
    /// another thread: end time in seconds, position (3), orientation
    /// `[w, x, y, z]` (4), linear velocity (3), angular velocity (3),
    /// accelerometer bias (3) and gyroscope bias (3).
    #[wasm_bindgen(js_name = toArray)]
    pub fn to_array(&self) -> js_sys::Float64Array {
        let mut values = [0.0; SCAN_ARRAY_LEN];
        values[0] = self.sec as f64 + self.nanosec as f64 * 1e-9;
        values[1..4].copy_from_slice(&self.position);
        values[4..8].copy_from_slice(&self.orientation);
        values[8..11].copy_from_slice(&self.linear_velocity);
        values[11..14].copy_from_slice(&self.angular_velocity);
        values[14..17].copy_from_slice(&self.accel_bias);
        values[17..20].copy_from_slice(&self.gyro_bias);
        js_sys::Float64Array::from(&values[..])
    }
}

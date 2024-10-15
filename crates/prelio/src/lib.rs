pub mod config;
pub mod imu;
mod naive_preint;
mod naive_preint_eth;
pub mod timing;

use imu_fusion::FusionAhrs;
use nalgebra::*;
use timing::RuntimeEvaluator;

use std::str::FromStr;
use std::{collections::VecDeque, future::Future};

use anyhow::Result;
use ros_pointcloud2::prelude::*;

use imu::*;

pub type ImuPredictor = dyn FnMut(&[imu::ImuMsg], f64, f64) -> ImuPosePrediction;

struct ImuMadgwick {
    fusion: imu_fusion::FusionAhrs,
    gravity: f64,
}

impl ImuMadgwick {
    pub fn new(gravity: f64) -> Self {
        ImuMadgwick {
            fusion: FusionAhrs::new(),
            gravity,
        }
    }
    /// Update the AHRS with new sensor data.
    /// gyroscope units are radians/s
    /// accelerometer units are m/s^2
    pub fn update_no_magnetometer_ros(&mut self, dt: f64, gyro: [f64; 3], accel: [f64; 3]) {
        let gyro = gyro.map(|v| v.to_degrees());
        let gyro = imu_fusion::FusionVector::new(gyro[0] as f32, gyro[1] as f32, gyro[2] as f32);

        let accel = accel.map(|a| a / self.gravity);
        let accel =
            imu_fusion::FusionVector::new(accel[0] as f32, accel[1] as f32, accel[2] as f32);

        self.fusion.update_no_mag(gyro, accel, dt as f32)
    }

    /// Get orientation in world frame as a quaternion.
    /// The quaternion is in the form [w, x, y, z].
    pub fn get_quaternion(&self) -> [f64; 4] {
        [
            self.fusion.quaternion.w as f64,
            self.fusion.quaternion.x as f64,
            self.fusion.quaternion.y as f64,
            self.fusion.quaternion.z as f64,
        ]
    }

    /// Get linear acceleration in sensor frame.
    /// The acceleration is in the form [x, y, z].
    /// Units are g.
    pub fn get_linear_acceleration(&self) -> [f64; 3] {
        let acc = self.fusion.linear_acc();
        [acc.x as f64, acc.y as f64, acc.z as f64]
    }

    pub fn get_linear_acceleration_ros(&self) -> [f64; 3] {
        self.get_linear_acceleration().map(|a| a * self.gravity)
    }
}

#[inline]
fn safe_angle(angle: f64) -> f64 {
    let factor = (angle / (2.0 * std::f64::consts::PI)).floor() as i32;
    angle - factor as f64 * 2.0 * std::f64::consts::PI
}

#[inline]
fn angle2d(v1: Vector2<f32>, v2: Vector2<f32>) -> f32 {
    v1.y.atan2(v1.x) - v2.y.atan2(v2.x)
}

#[inline]
fn inside_box(p: Point3<f32>, min: Vector3<f32>, max: Vector3<f32>) -> bool {
    p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y && p.z >= min.z && p.z <= max.z
}

#[inline]
fn inside_horizontal_fov(
    p: Point3<f32>,
    fov_right: f32,
    fov_left: f32,
    forward: Vector2<f32>,
) -> bool {
    let angle = angle2d(forward, Vector2::new(p.x, p.y));
    let fov_angle = safe_angle((fov_right - fov_left).into());
    let local_angle = safe_angle((angle - fov_left).into());
    local_angle <= fov_angle
}

#[inline]
fn point_inside_polygon_winding_number(
    p: Point3<f32>,
    polygon: &[Point2<f32>],
    polygon_size: usize,
) -> bool {
    let mut winding_number = 0;
    for i in 0..polygon_size {
        let v1 = polygon[i];
        let v2 = polygon[(i + 1) % polygon_size];

        let cont =
            p.y < v1.y.min(v2.y) || p.y > v1.y.max(v2.y) || p.x > v1.x.max(v2.x) || v1.y == v2.y;

        if cont {
            continue;
        }

        let y_slope = (v2.x - v1.x) / (v2.y - v1.y);
        let x_intercept = (p.y - v1.y) * y_slope + v1.y;
        if v1.x == v2.x || p.x <= x_intercept {
            winding_number += if v2.y > v1.y { 1 } else { -1 };
        }
    }

    winding_number != 0
}

pub fn timestamp_to_secs(t: &TimeMsg) -> f64 {
    t.sec as f64 + t.nanosec as f64 / 1_000_000_000.0
}

// TODO check correctness
pub fn secs_to_timestamp(secs: f64) -> TimeMsg {
    let sec = secs as i32;
    let nanosec = ((secs - sec as f64) * 1_000_000_000.0) as u32;
    TimeMsg { sec, nanosec }
}

pub trait PointXYZITimestamp<const N: usize>: PointConvertible<N> + Send + Sync + Sized {
    fn xyz(&self) -> Point3<f32>;
    fn intensity(&self) -> f32;
    fn timestamp(&self) -> f64; // in seconds
}

#[derive(Debug, Clone, Default)]
pub struct MeasureGroup {
    cloud: Vec<PointXYZT>,
    lidar_start_stamp: f64,
    lidar_end_stamp: f64,
    imu_buffer: Vec<ImuMsg>,
}

#[derive(Debug, Clone, Default)]
struct PointXYZIT {
    x: f32,
    y: f32,
    z: f32,
    intensity: f32,
    timestamp: f64,
}

impl std::ops::Add for PointXYZIT {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            intensity: self.intensity + other.intensity,
            timestamp: self.timestamp + other.timestamp,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PointXYZT {
    x: f32,
    y: f32,
    z: f32,
    timestamp: f64,
}

impl PointXYZT {
    fn from_p(point: Point3<f32>, timestamp: f64) -> Self {
        Self {
            x: point.x,
            y: point.y,
            z: point.z,
            timestamp,
        }
    }
}

use std::sync::Arc;
use std::sync::Mutex;

fn hash(point: &Point3<f32>, voxel_size: f32) -> i64 {
    let xyz = point * (1.0 / voxel_size);
    let key = (xyz.x.round() as i64 * 73856093)
        ^ (xyz.y.round() as i64 * 471943)
        ^ (xyz.z.round() as i64 * 83492791) % 100000;
    key as i64
}

async fn filter_downsample<const N: usize, P: PointXYZITimestamp<N>>(
    cloud: PointCloud2Msg,
    lifis_params: config::LifisParameters,
) -> Result<Option<(Vec<PointXYZT>, f64, f64, HeaderMsg)>> {
    let header = cloud.header.clone();
    let res = cloud.try_into_par_iter()?.filter(move |point: &P| {
        let params = &lifis_params.passthrough_filter;
        let p_t = params.filter_frame_transform * point.xyz();

        let dist = p_t.coords.norm();
        let is_within_range = if params.invert_distance {
            dist < params.min_dist || dist > params.max_dist
        } else {
            dist >= params.min_dist && dist <= params.max_dist
        };

        let is_within_intensity = if params.invert_intensity {
            point.intensity() < params.min_intensity || point.intensity() > params.max_intensity
        } else {
            point.intensity() >= params.min_intensity && point.intensity() <= params.max_intensity
        };

        let is_within_box = if params.invert_bounding_box {
            !inside_box(p_t, params.min, params.max)
        } else {
            inside_box(p_t, params.min, params.max)
        };

        let is_within_fov = if params.enable_horizontal_fov {
            if params.invert_fov {
                !inside_horizontal_fov(p_t, params.fov_right, params.fov_left, params.forward)
            } else {
                inside_horizontal_fov(p_t, params.fov_right, params.fov_left, params.forward)
            }
        } else {
            true
        };

        let is_within_polygon = match &params.polygon {
            None => true,
            Some(polygon) => {
                let is_within_polygon = if params.invert_polygon {
                    !point_inside_polygon_winding_number(p_t, polygon, polygon.len())
                } else {
                    point_inside_polygon_winding_number(p_t, polygon, polygon.len())
                };
                polygon.len() != 0 && is_within_polygon
            }
        };

        is_within_range
            && is_within_intensity
            && is_within_box
            && is_within_fov
            && is_within_polygon
    });

    // Downsample
    let hashes = dashmap::DashMap::new(); // TODO try out ahash
    res.for_each(|point: P| {
        let hash = hash(&point.xyz(), lifis_params.downsample.voxel_size);
        if !hashes.contains_key(&hash) {
            hashes.insert(hash, Vec::with_capacity(10));
        }
        hashes
            .get_mut(&hash)
            .expect("entry created the line before")
            .push(point);
    });

    let mut out = hashes
        .par_iter()
        .map(|bucket_ref| {
            let voxel_average = bucket_ref
                .iter()
                .fold(Vector3::default(), |acc, point| point.xyz().coords + acc);

            /*let voxel_average = bucket_ref
            .par_iter()
            .map(|point| point.xyz().coords)
            .reduce(Vector3::default, |acc, coords| acc + coords);*/

            let avg = voxel_average * (1.0 / bucket_ref.len() as f32);

            let mut min_dist = f32::MAX;
            let mut closest_point = PointXYZT::default();
            for point in bucket_ref.iter() {
                let dist = (point.xyz().coords - avg).norm_squared();
                if dist < min_dist {
                    min_dist = dist;
                    closest_point = PointXYZT::from_p(point.xyz(), point.timestamp());
                }
            }

            closest_point
        })
        .collect::<Vec<PointXYZT>>();

    out.par_sort_by(|a, b| {
        if (a.timestamp - b.timestamp).abs() < 1e-6 {
            a.z.partial_cmp(&b.z).unwrap()
        } else {
            a.timestamp.partial_cmp(&b.timestamp).unwrap()
        }
    });

    let (first, last) = match (&out.first(), &out.last()) {
        (Some(first), Some(last)) => (first.timestamp, last.timestamp),
        _ => return Ok(None),
    };

    Ok(Some((out, first, last, header)))
}

struct PointCloudEoSJoin {
    inner:
        tokio::task::JoinSet<Result<Option<(Vec<PointXYZT>, f64, f64, HeaderMsg)>, anyhow::Error>>,
}

impl PointCloudEoSJoin {
    fn new() -> Self {
        Self {
            inner: tokio::task::JoinSet::new(),
        }
    }

    fn spawn<F>(&mut self, f: F)
    where
        F: Future<Output = Result<Option<(Vec<PointXYZT>, f64, f64, HeaderMsg)>, anyhow::Error>>
            + Send
            + 'static,
    {
        self.inner.spawn(f);
    }

    // Note: waiting for async drop to exist
    async fn drop(
        &mut self,
        mut buffer: std::sync::MutexGuard<'_, VecDeque<(Vec<PointXYZT>, f64, f64, HeaderMsg)>>,
    ) -> Result<()> {
        if self.inner.is_empty() {
            return Ok(());
        }

        let got_all = {
            while let Some(res) = self.inner.join_next().await {
                if let Some(cloud) = res?? {
                    buffer.push_front(cloud);
                }
            }
            Ok(())
        };

        match got_all {
            Err(e) => {
                self.inner.abort_all();
                Err(e)
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug)]
pub struct Twist {
    pub linear: Vector3<f64>,
    pub angular: Vector3<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct ImuPosePrediction {
    pub pose: Isometry3<f64>,
    pub linear_velocity: nalgebra::Vector3<f64>,
    pub angular_velocity: nalgebra::Vector3<f64>,
    pub lidar_imu_extrinsic_rotation: nalgebra::Matrix3<f64>,
    pub lidar_imu_extrinsic_translation: nalgebra::Vector3<f64>,
}

#[derive(Debug)]
pub struct LifisProcessedCloud {
    pub cloud: Vec<PointXYZ>,
    pub header: HeaderMsg,
    pub end_of_frame_guess: Isometry3<f64>,
    pub end_of_frame_time: TimeMsg,
    pub end_of_frame_twist: Twist,
    pub pose_covariance: [f64; 36],
    pub twist_covariance: [f64; 36],
    pub runtime_eval: RuntimeEvaluator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum MadgwickSetting {
    CreateOrientationOnly,
    FilterAccelerationOnly,
    #[default]
    FilterAccelerationAndCreateOrientation,
    Disable,
}

impl FromStr for MadgwickSetting {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "CreateOrientationOnly" => Ok(Self::CreateOrientationOnly),
            "FilterAccelerationOnly" => Ok(Self::FilterAccelerationOnly),
            "FilterAccelerationAndCreateOrientation" => {
                Ok(Self::FilterAccelerationAndCreateOrientation)
            }
            "Disable" => Ok(Self::Disable),
            _ => Err(anyhow::anyhow!("Invalid MadgwickSetting")),
        }
    }
}

pub struct Lifis {
    imu_buffer: VecDeque<ImuMsg>,
    imu_integrate_buffer: VecDeque<ImuMsg>,
    preproc_buffer: Arc<Mutex<VecDeque<Option<PointCloud2Msg>>>>,
    pointcloud_buffer: Arc<Mutex<VecDeque<(Vec<PointXYZT>, f64, f64, HeaderMsg)>>>,
    undistort: ImuProcess,
    last_cloud_time: f64,
    imu_filter_fusion: ImuMadgwick,
    last_imu_time: f64,
    runtime_eval: RuntimeEvaluator,
    imu_calibration_apply: Option<imu_calib::CalibrationApplication>,
    madgwick_filter: MadgwickSetting,
}

impl Lifis {
    pub fn create(
        gravity: f64,
        madgwick_filter: MadgwickSetting,
        imu_intrinsic_calibration_params: Option<config::ImuCalibrationParameters>,
    ) -> anyhow::Result<Self> {
        const CLOUDS_BUFFER_SIZE: usize = 2;
        let imu_intrinsic_calibration = match imu_intrinsic_calibration_params {
            Some(imu_intrinsic_calibration) => {
                let imu_calib = imu_calib::CalibrationApplication::create(
                    imu_intrinsic_calibration.imu_calibrate_gyros,
                    imu_intrinsic_calibration.gyro_calibration_samples,
                    imu_intrinsic_calibration.imu_intrinsics_calib_file_path,
                    gravity,
                )?;
                Some(imu_calib)
            }
            None => None,
        };
        Ok(Lifis {
            pointcloud_buffer: Arc::new(Mutex::new(VecDeque::with_capacity(CLOUDS_BUFFER_SIZE))),
            preproc_buffer: Arc::new(Mutex::new(VecDeque::with_capacity(CLOUDS_BUFFER_SIZE))),
            imu_buffer: VecDeque::with_capacity(20 * CLOUDS_BUFFER_SIZE),
            imu_integrate_buffer: VecDeque::with_capacity(20 * CLOUDS_BUFFER_SIZE),
            last_cloud_time: 0.0,
            undistort: ImuProcess::new(gravity, imu_intrinsic_calibration.is_some()),
            imu_filter_fusion: ImuMadgwick::new(gravity),
            last_imu_time: 0.0,
            runtime_eval: RuntimeEvaluator::new(),
            imu_calibration_apply: imu_intrinsic_calibration,
            madgwick_filter,
        })
    }

    // TODO: debouncing - make sure we don't get too many clouds by buffer length and only accepting every nth cloud, then every nth+1 etc after certain buffer lengths each
    // also, this should be configurable with defaults and toggleable
    /// Add a point cloud to the buffer.
    pub fn add_pointcloud<const N: usize, P: PointXYZITimestamp<N>>(
        &mut self,
        cloud: PointCloud2Msg,
    ) {
        let mut preproc_buffer = self.preproc_buffer.lock().unwrap();

        let cloud_stamp = timestamp_to_secs(&cloud.header.stamp);
        let buff_cloud = if cloud_stamp < self.last_cloud_time {
            None
        } else {
            Some(cloud)
        };

        self.last_cloud_time = cloud_stamp;
        preproc_buffer.push_front(buff_cloud);
    }

    /// Filter raw IMU message and add to the buffer for synchronization and integration.
    /// Returns the filtered IMU message for republishing.
    pub fn add_imu(&mut self, mut imu: ImuMsg) -> Option<ImuMsg> {
        let imu_time = timestamp_to_secs(&imu.stamp);
        if imu_time < self.last_imu_time {
            return None; // wrapping time (mostly bag file restarts), wait for lidar wrap to reset
        }

        self.runtime_eval.start("total").unwrap();
        self.runtime_eval.start("imu_pre").unwrap();

        let dt = imu_time - self.last_imu_time;
        self.last_imu_time = imu_time;

        let mut gyro = [
            imu.angular_velocity[0],
            imu.angular_velocity[1],
            imu.angular_velocity[2],
        ];

        let mut accel = [
            imu.linear_acceleration[0],
            imu.linear_acceleration[1],
            imu.linear_acceleration[2],
        ];

        // apply calibration if available
        if let Some(imu_calib) = &mut self.imu_calibration_apply {
            let calibrated = imu_calib.imu_callback(imu_calib::ImuMsg::new(&accel, &gyro))?;
            accel = calibrated.linear_acceleration.clone();
            gyro = calibrated.angular_velocity.clone();
        }

        // madgwick filter for global orientation
        match self.madgwick_filter {
            MadgwickSetting::CreateOrientationOnly => {
                self.imu_filter_fusion
                    .update_no_magnetometer_ros(dt, gyro, accel);
                imu.orientation = self.imu_filter_fusion.get_quaternion();
            }
            MadgwickSetting::FilterAccelerationOnly => {
                self.imu_filter_fusion
                    .update_no_magnetometer_ros(dt, gyro, accel);
                let filtered_acc = self.imu_filter_fusion.get_linear_acceleration_ros();
                imu.linear_acceleration[0] = filtered_acc[0];
                imu.linear_acceleration[1] = filtered_acc[1];
                imu.linear_acceleration[2] = filtered_acc[2];
            }
            MadgwickSetting::FilterAccelerationAndCreateOrientation => {
                self.imu_filter_fusion
                    .update_no_magnetometer_ros(dt, gyro, accel);
                let filtered_acc = self.imu_filter_fusion.get_linear_acceleration_ros();
                imu.orientation = self.imu_filter_fusion.get_quaternion();

                // update imu message
                imu.linear_acceleration[0] = filtered_acc[0];
                imu.linear_acceleration[1] = filtered_acc[1];
                imu.linear_acceleration[2] = filtered_acc[2];
            }
            MadgwickSetting::Disable => {}
        }

        // add to buffer
        self.imu_buffer.push_front(imu.clone());
        self.imu_integrate_buffer.push_front(imu.clone());
        self.runtime_eval.stop("imu_pre").unwrap();
        self.runtime_eval.stop("total").unwrap();

        Some(imu)
    }

    /// Process the buffered data.
    /// returns Result<Option<(PointCloud2Msg, Affine3<f32>)>> but is async
    /// imu_predictor is a fn [ImuMsg] -> ImuPosePrediction
    pub async fn tick<const N: usize, P: PointXYZITimestamp<N>>(
        &mut self,
        params: &config::LifisParameters,
        imu_pose_prediction_undistort: Option<Box<ImuPredictor>>,
        imu_pose_prediction_init_guess: Option<Box<ImuPredictor>>,
    ) -> Result<Option<LifisProcessedCloud>> {
        let too_few_imu_error = anyhow::anyhow!("Too few IMU data for integration after scan. The sensors are probably not synchronized or the IMU is too slow.");
        self.runtime_eval.start("total").unwrap(); // TODO crashes when bagfile is too fast
        self.runtime_eval.start("scan_pre").unwrap();
        let mut immediate_handle = None; // gets assigned if we have nothing processed yet because we must wait for the first cloud
        let to_buffer = {
            let mut preproc_buffer = self.preproc_buffer.lock().unwrap();

            let mut to_buffer = Vec::with_capacity(preproc_buffer.len());
            while let Some(cloud) = preproc_buffer.pop_back() {
                match cloud {
                    Some(cloud) => {
                        let pointcloud_buffer = self.pointcloud_buffer.lock().unwrap();
                        if pointcloud_buffer.is_empty() && immediate_handle.is_none() {
                            let params = params.clone();
                            immediate_handle = Some(tokio::spawn(async move {
                                filter_downsample::<N, P>(cloud, params).await
                            }));
                        } else {
                            to_buffer.push(cloud);
                        }
                    }
                    None => {
                        self.imu_buffer.clear();
                        {
                            let mut pointcloud_buffer = self.pointcloud_buffer.lock().unwrap();
                            pointcloud_buffer.clear();
                        }
                        self.last_cloud_time = 0.0;
                        self.last_imu_time = 0.0;
                        self.undistort.reset();
                        to_buffer.clear();
                    }
                }
            }

            to_buffer
        };

        let mut bg_filtering = PointCloudEoSJoin::new();
        to_buffer.into_iter().for_each(|cloud| {
            let params = params.clone();
            bg_filtering.spawn(async move { filter_downsample::<N, P>(cloud, params).await });
        });

        let mut g_cloud_buffer = self.pointcloud_buffer.lock().unwrap();

        if let Some(immediate_cloud) = immediate_handle {
            if let Some(cloud) = immediate_cloud.await?? {
                g_cloud_buffer.push_front(cloud);
            }
        }

        if g_cloud_buffer.is_empty() {
            bg_filtering.drop(g_cloud_buffer).await?;
            self.runtime_eval.reset("total").unwrap();
            self.runtime_eval.reset("scan_pre").unwrap();
            return Ok(None);
        }

        self.runtime_eval.stop("scan_pre").unwrap();

        if P::default().timestamp() < 0.0 {
            // we don't have a timestamp so no reason to continue
            let cloud = g_cloud_buffer.pop_back().unwrap();
            let cloud_xyz = cloud
                .0
                .iter()
                .map(|point| PointXYZ::new(point.x, point.y, point.z)) // TODO, integrate within "out". was hardcoded 1.0
                .collect::<Vec<_>>();
            let out = LifisProcessedCloud {
                cloud: cloud_xyz,
                header: cloud.3.clone(),
                end_of_frame_guess: Isometry3::identity(),
                end_of_frame_time: TimeMsg { sec: 0, nanosec: 0 },
                end_of_frame_twist: Twist {
                    linear: Vector3::new(0.0, 0.0, 0.0),
                    angular: Vector3::new(0.0, 0.0, 0.0),
                },
                pose_covariance: [0.0; 36],
                twist_covariance: [0.0; 36],
                runtime_eval: self.runtime_eval.clone(), // TODO avoid clone
            };
            bg_filtering.drop(g_cloud_buffer).await?;
            self.runtime_eval.reset("total").unwrap();
            return Ok(Some(out));
        }

        if self.imu_buffer.is_empty() {
            bg_filtering.drop(g_cloud_buffer).await?;
            self.runtime_eval.reset("total").unwrap();
            return Ok(None);
        }

        let oldest_imu_time = match self.imu_buffer.back() {
            Some(imu) => timestamp_to_secs(&imu.stamp),
            None => {
                bg_filtering.drop(g_cloud_buffer).await?;
                self.runtime_eval.reset("total").unwrap();
                return Ok(None);
            }
        };

        while let Some(cloud) = g_cloud_buffer.back() {
            let (_, cloud_end_time, _) = {
                let cloud_stamp = timestamp_to_secs(&cloud.3.stamp);
                (cloud_stamp, cloud_stamp + cloud.2, cloud.3.clone())
            };
            if cloud_end_time < oldest_imu_time {
                g_cloud_buffer.pop_back();
            } else {
                break;
            }
        }

        if g_cloud_buffer.is_empty() {
            bg_filtering.drop(g_cloud_buffer).await?;
            self.runtime_eval.reset("total").unwrap();
            return Ok(None);
        }

        let (cloud_begin_time, cloud_end_time, _) = match g_cloud_buffer.back() {
            Some(cloud) => {
                let cloud_stamp = timestamp_to_secs(&cloud.3.stamp);
                (cloud_stamp, cloud_stamp + cloud.2, cloud.3.clone())
            }
            None => {
                bg_filtering.drop(g_cloud_buffer).await?;
                self.runtime_eval.reset("total").unwrap();
                return Ok(None);
            }
        };

        // remove all imu data that are younger than the cloud
        while let Some(imu) = self.imu_buffer.back() {
            let imu_time = timestamp_to_secs(&imu.stamp);
            if imu_time < cloud_begin_time {
                self.imu_buffer.pop_back();
            } else {
                break;
            }
        }

        // removed all imu, wait for more
        if self.imu_buffer.is_empty() {
            bg_filtering.drop(g_cloud_buffer).await?;
            return Ok(None); // TODO: too_few_imu_error warning
        }

        let oldest_imu_time = match self.imu_buffer.back() {
            Some(imu) => timestamp_to_secs(&imu.stamp),
            None => {
                bg_filtering.drop(g_cloud_buffer).await?;
                self.runtime_eval.reset("total").unwrap();
                return Ok(None);
            }
        };

        if g_cloud_buffer.is_empty() || cloud_end_time < oldest_imu_time {
            bg_filtering.drop(g_cloud_buffer).await?;
            self.runtime_eval.reset("total").unwrap();
            return Ok(None);
        }
        let mut cloud_imu_buffer = Vec::with_capacity(20);

        let mut buffer_full = false;
        while let Some(imu) = self.imu_buffer.back() {
            let imu_time = timestamp_to_secs(&imu.stamp);
            debug_assert!(imu_time >= cloud_begin_time);

            if imu_time <= cloud_end_time {
                cloud_imu_buffer.push(imu.clone());
            } else {
                buffer_full = true;
                break;
            }

            self.imu_buffer.pop_back();
        }

        if !buffer_full {
            // we need more imu data to match to the clouds, give back the packets by iterating from the back
            for imu in cloud_imu_buffer.into_iter().rev() {
                self.imu_buffer.push_back(imu);
            }
            bg_filtering.drop(g_cloud_buffer).await?;
            self.runtime_eval.reset("total").unwrap();
            return Ok(None);
        }

        if cloud_imu_buffer.is_empty() {
            bg_filtering.drop(g_cloud_buffer).await?;
            self.runtime_eval.reset("total").unwrap();
            return Err(too_few_imu_error);
        }

        let cloud_data = match g_cloud_buffer.pop_back() {
            Some(cloud_data) => cloud_data,
            None => {
                bg_filtering.drop(g_cloud_buffer).await?;
                self.runtime_eval.reset("total").unwrap();
                return Ok(None);
            }
        };
        // we have imu data and clouds, now we can do the matching

        // ------------ imu integration since last scan ------------

        while let Some(imu) = self.imu_integrate_buffer.back() {
            let imu_time = timestamp_to_secs(&imu.stamp);
            if imu_time <= cloud_begin_time {
                self.imu_integrate_buffer.pop_back();
            } else {
                break;
            }
        }

        /*let last_integrate_imu_time = match self.imu_integrate_buffer.back() {
            Some(last) => timestamp_to_secs(&last.stamp),
            _ => {
                bg_filtering.drop(g_cloud_buffer).await?;
                self.runtime_eval.reset("total").unwrap();
                self.runtime_eval.reset("imu_int").unwrap();
                return Err(too_few_imu_error);
            }
        };*/

        // ------------ matching measuregroup for deskewing points ------------
        let (cloud_begin_time, cloud_end_time, header) = {
            let cloud_stamp = timestamp_to_secs(&cloud_data.3.stamp);
            (
                cloud_stamp,
                cloud_stamp + cloud_data.2,
                cloud_data.3.clone(),
            )
        };

        let meas = MeasureGroup {
            cloud: cloud_data.0,
            lidar_start_stamp: cloud_begin_time,
            lidar_end_stamp: cloud_end_time,
            imu_buffer: cloud_imu_buffer,
        };

        let given_imu_calib = self.undistort.intrinsic_calibration_given;

        match self
            .undistort
            .process(meas, &params.undistort, imu_pose_prediction_undistort)
        {
            Some(imu_cloud) => {
                // ------------ imu integration ------------
                self.runtime_eval.start("imu_int").unwrap();
                let imu_int = match imu_pose_prediction_init_guess {
                    Some(mut imu_pose_prediction_init_guess) => imu_pose_prediction_init_guess(
                        &Vec::from_iter(self.imu_integrate_buffer.iter().rev().cloned()),
                        cloud_begin_time,
                        cloud_end_time,
                    ),
                    None => {
                        let estimated_pose = if given_imu_calib {
                            // given a calibration, we trust the IMU data and use it for the pose
                            #[cfg(feature = "gtsam")]
                            {
                                ImuProcess::integrate_imu_gtsam(
                                    &Vec::from_iter(
                                        self.imu_integrate_buffer.iter().rev().cloned(),
                                    ),
                                    self.last_cloud_time,
                                    Some(&params.undistort),
                                    params.imu.clone(), // TODO maybe not clone but changes inside the function?
                                )
                            }

                            #[cfg(not(feature = "gtsam"))]
                            {
                                naive_preint_eth::integrate_imu(
                                    &Vec::from_iter(
                                        self.imu_integrate_buffer.iter().rev().cloned(),
                                    ),
                                    self.last_cloud_time,
                                    Some(&params.undistort),
                                    false, // TODO check ahrs orientation via config or if enabled
                                    params.imu.gravity,
                                )
                            }
                        } else {
                            // no calibration, gyro is used for rotation but nothing else
                            Isometry3::from_parts(
                                Translation::identity(),
                                imu_cloud.end_of_frame_guess.rotation,
                            )
                        };

                        ImuPosePrediction {
                            pose: estimated_pose,
                            linear_velocity: Vector3::zeros(),
                            angular_velocity: Vector3::new(
                                imu_cloud.last_twist[3],
                                imu_cloud.last_twist[4],
                                imu_cloud.last_twist[5],
                            ),
                            lidar_imu_extrinsic_rotation: Matrix3::identity(), // TODO never used?
                            lidar_imu_extrinsic_translation: Vector3::zeros(), // TODO never used?
                        }
                    }
                };

                self.runtime_eval.stop("imu_int").unwrap();

                bg_filtering.drop(g_cloud_buffer).await?;
                self.runtime_eval.stop("total").unwrap();

                let mut out_runtime = self.runtime_eval.clone();
                out_runtime.merge(&imu_cloud.runtime_eval); // TODO avoid clone, do new object with both evaluations
                Ok(Some(LifisProcessedCloud {
                    cloud: imu_cloud.cloud,
                    header,
                    end_of_frame_guess: imu_int.pose,
                    end_of_frame_time: self.imu_integrate_buffer.back().unwrap().stamp.clone(),
                    pose_covariance: imu_cloud.integrated_covariance,
                    twist_covariance: imu_cloud.last_covariance,
                    end_of_frame_twist: Twist {
                        linear: Vector3::new(
                            imu_cloud.last_twist[0],
                            imu_cloud.last_twist[1],
                            imu_cloud.last_twist[2],
                        ),
                        angular: Vector3::new(
                            imu_cloud.last_twist[3],
                            imu_cloud.last_twist[4],
                            imu_cloud.last_twist[5],
                        ),
                    },
                    runtime_eval: out_runtime,
                }))
            }
            None => {
                bg_filtering.drop(g_cloud_buffer).await?;
                self.runtime_eval.stop("total").unwrap();
                return Ok(None);
            }
        }
    }
}

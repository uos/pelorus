pub mod config;
pub mod imu;
mod naive_preint;
mod naive_preint_eth;
pub mod timing;

use imu_fusion::FusionAhrs;
use nalgebra::*;
use timing::RuntimeEvaluator;

use std::{collections::VecDeque, str::FromStr};

use anyhow::Result;
use ros_pointcloud2::prelude::*;

use imu::*;

pub type ImuPredictor = dyn for<'a> Fn(&'a [imu::ImuMsg], f64, f64) -> ImuPosePrediction + Send;

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

fn hash(point: &Point3<f32>, voxel_size: f32) -> i32 {
    const PRIME_1: i32 = 73_856_093;
    const PRIME_2: i32 = 19_349_663;
    const PRIME_3: i32 = 83_492_791;
    const PRIME_4: i32 = 116_881;

    let xyz = point * (1.0 / voxel_size);

    let x = xyz.x.round() as i32;
    let y = xyz.y.round() as i32;
    let z = xyz.z.round() as i32;

    let mut hash = x.wrapping_mul(PRIME_1);
    hash = hash.wrapping_add(y.wrapping_mul(PRIME_2));
    hash = hash.wrapping_add(z.wrapping_mul(PRIME_3));
    hash = hash.wrapping_rem(PRIME_4);

    hash
}

async fn filter_downsample<const N: usize, P: PointXYZITimestamp<N>>(
    cloud: PointCloud2Msg,
    lifis_params: &config::LifisParameters,
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
    preproc_sender: tokio::sync::mpsc::Sender<PointCloud2Msg>,
    imu_sender: tokio::sync::mpsc::Sender<ImuMsg>,
    synced_receiver: Option<tokio::sync::mpsc::Receiver<LifisProcessedCloud>>,
    last_cloud_time: f64,
    imu_filter_fusion: ImuMadgwick,
    last_imu_time: f64,
    runtime_eval: RuntimeEvaluator,
    imu_calibration_apply: Option<imu_calib::CalibrationApplication>,
    madgwick_filter: MadgwickSetting,
}

impl Lifis {
    pub fn create<const N: usize, P: PointXYZITimestamp<N>>(
        gravity: f64,
        madgwick_filter: MadgwickSetting,
        imu_intrinsic_calibration_params: Option<config::ImuCalibrationParameters>,
        params: &config::LifisParameters,
        imu_pose_prediction_undistort: Option<Box<ImuPredictor>>,
        imu_pose_prediction_init_guess: Option<Box<ImuPredictor>>,
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

        let (preproc_tx, mut preproc_rx) = tokio::sync::mpsc::channel(CLOUDS_BUFFER_SIZE);
        let (imu_tx, imu_rx) = tokio::sync::mpsc::channel(20 * CLOUDS_BUFFER_SIZE);
        let (filtered_cloud_tx, filtered_cloud_rx) = tokio::sync::mpsc::channel(CLOUDS_BUFFER_SIZE);
        let (synced_cloud_imu_tx, synced_cloud_imu_rx) =
            tokio::sync::mpsc::channel(CLOUDS_BUFFER_SIZE);

        // sync and imu integration worker
        let intrinsic_given = imu_intrinsic_calibration.is_some();
        let undistort_params = params.undistort.clone();
        tokio::spawn(async move {
            let mut last_lidar_frame_time = 0.0;
            let imu_process = ImuProcess::new(gravity, intrinsic_given);

            Self::sync_lidar_imu_worker(
                &mut last_lidar_frame_time,
                gravity,
                filtered_cloud_rx,
                imu_rx,
                &synced_cloud_imu_tx,
                intrinsic_given,
                imu_process,
                &undistort_params,
                imu_pose_prediction_undistort,
                imu_pose_prediction_init_guess,
            )
            .await;
        });

        // preprocessing worker
        tokio::spawn({
            let params_clone = params.clone();
            async move {
                while let Some(cloud) = preproc_rx.recv().await {
                    match filter_downsample::<N, P>(cloud, &params_clone).await {
                        Ok(Some(cloud_meta)) => {
                            //dbg!("Pointcloud was processed");
                            if let Err(e) = filtered_cloud_tx.send(cloud_meta).await {
                                dbg!("Error sending pointcloud: {}", e);
                            }
                        }
                        Ok(None) => {
                            dbg!("Pointcloud was filtered out");
                        }
                        Err(e) => {
                            dbg!("Error processing pointcloud: {}", e);
                        }
                    }
                }
            }
        });

        Ok(Lifis {
            imu_sender: imu_tx,
            synced_receiver: Some(synced_cloud_imu_rx),
            preproc_sender: preproc_tx,
            last_cloud_time: 0.0,
            imu_filter_fusion: ImuMadgwick::new(gravity),
            last_imu_time: 0.0,
            runtime_eval: RuntimeEvaluator::new(),
            imu_calibration_apply: imu_intrinsic_calibration,
            madgwick_filter,
        })
    }

    pub fn get_chan(&mut self) -> Option<tokio::sync::mpsc::Receiver<LifisProcessedCloud>> {
        self.synced_receiver.take()
    }

    // TODO: debouncing - make sure we don't get too many clouds by buffer length and only accepting every nth cloud, then every nth+1 etc after certain buffer lengths each
    // also, this should be configurable with defaults and toggleable
    /// Add a point cloud to the buffer.
    pub fn add_pointcloud(&mut self, cloud: PointCloud2Msg) {
        let cloud_stamp = timestamp_to_secs(&cloud.header.stamp);

        // TODO handle time jumps in the past
        /*let buff_cloud = if cloud_stamp < self.last_cloud_time {
            None
        } else {
            Some(cloud)
        };*/

        /*self.imu_buffer.clear();
        {
            let mut pointcloud_buffer = self.preproc_sender.
            pointcloud_buffer.clear();
        }
        self.last_cloud_time = 0.0;
        self.last_imu_time = 0.0;
        self.undistort.reset();
        to_buffer.clear();*/

        self.last_cloud_time = cloud_stamp;
        let sender = self.preproc_sender.clone();
        tokio::spawn(async move {
            match sender.send(cloud).await {
                Ok(_) => {
                    //dbg!("Pointcloud sent");
                }
                Err(e) => {
                    dbg!("Error sending pointcloud: {}", e);
                }
            }
        });
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
        let sender = self.imu_sender.clone();
        let imu_data = imu.clone();
        tokio::spawn(async move {
            match sender.send(imu_data).await {
                Ok(_) => {}
                Err(e) => {
                    dbg!(e);
                }
            }
        });
        self.runtime_eval.stop("imu_pre").unwrap();
        self.runtime_eval.stop("total").unwrap();

        Some(imu)
    }

    // syncs lidar and imu data by using the timestamps from the headers and timestamps in the lidar message.
    // a lidar message is therefore mapped to multiple imu messages that happened in between the previous end stamp and end stamp of the last point in the lidar message.
    pub async fn sync_lidar_imu_worker(
        prev_lidar_end_time: &mut f64,
        gravity: f64,
        mut lidar_receiver: tokio::sync::mpsc::Receiver<(Vec<PointXYZT>, f64, f64, HeaderMsg)>,
        mut imu_receiver: tokio::sync::mpsc::Receiver<ImuMsg>,
        synced_sender: &tokio::sync::mpsc::Sender<LifisProcessedCloud>,
        imu_intrinsic_calibration_given: bool,
        mut imu_process: ImuProcess,
        undistort_parameters: &config::UndistortParameters,
        imu_pose_prediction_undistort: Option<Box<ImuPredictor>>,
        imu_pose_prediction_init_guess: Option<Box<ImuPredictor>>,
    ) {
        let mut to_next: Option<ImuMsg> = None;
        while let Some((cloud, lidar_begin_time, lidar_end_time, lidar_header)) =
            lidar_receiver.recv().await
        {
            let lidar_time = timestamp_to_secs(&lidar_header.stamp);
            let lidar_begin_time = lidar_begin_time + lidar_time;
            let imu_span_start_time = *prev_lidar_end_time;
            let imu_span_end_time = lidar_end_time + lidar_time;

            let mut imu_data = Vec::with_capacity(100);

            if let Some(last_imu_extra) = to_next.take() {
                let imu_time = timestamp_to_secs(&last_imu_extra.stamp);
                if imu_time > imu_span_start_time && imu_time < imu_span_end_time {
                    imu_data.push(last_imu_extra);
                }
            }

            loop {
                match imu_receiver.recv().await {
                    Some(imu) => {
                        let imu_time = timestamp_to_secs(&imu.stamp);
                        if imu_time < imu_span_start_time {
                            continue;
                        }
                        if imu_time > imu_span_end_time {
                            to_next = Some(imu);
                            break;
                        }
                        imu_data.push(imu);
                    }
                    None => {
                        dbg!("Imu receiver closed"); // TODO logging
                        return;
                    }
                }
            }

            //dbg!(imu_data.len());

            if imu_data.is_empty() {
                dbg!("No imu data in span");
                continue;
            }

            // - data synced -

            let meas = MeasureGroup {
                cloud,
                lidar_start_stamp: lidar_begin_time,
                lidar_end_stamp: lidar_end_time,
                imu_buffer: imu_data.clone(),
            };

            if let Some(imu_cloud) =
                imu_process.process(meas, undistort_parameters, &imu_pose_prediction_undistort)
            {
                let imu_int = match imu_pose_prediction_init_guess.as_ref() {
                    Some(imu_pose_prediction_init_guess) => imu_pose_prediction_init_guess(
                        &imu_data.clone(), // TODO maybe need reversed?
                        lidar_begin_time,
                        imu_span_end_time,
                    ),
                    None => {
                        let estimated_pose = if imu_intrinsic_calibration_given {
                            // given a calibration, we trust the IMU data and use it for the pose
                            #[cfg(feature = "gtsam")]
                            {
                                ImuProcess::integrate_imu_gtsam(
                                    &imu_data,
                                    *prev_lidar_end_time,
                                    Some(&params.undistort),
                                    params.imu.clone(), // TODO maybe not clone but changes inside the function?
                                )
                            }

                            #[cfg(not(feature = "gtsam"))]
                            {
                                naive_preint_eth::integrate_imu(
                                    &imu_data,
                                    *prev_lidar_end_time,
                                    Some(undistort_parameters),
                                    false, // TODO check ahrs orientation via config or if enabled
                                    gravity,
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

                //let mut out_runtime = self.runtime_eval.clone();
                //out_runtime.merge(&imu_cloud.runtime_eval); // TODO avoid clone, do new object with both evaluations
                let res = LifisProcessedCloud {
                    cloud: imu_cloud.cloud,
                    header: lidar_header,
                    end_of_frame_guess: imu_int.pose,
                    end_of_frame_time: imu_data.last().unwrap().stamp.clone(),
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
                    runtime_eval: RuntimeEvaluator::new(), // TODO currently nothing
                };

                synced_sender.send(res).await.unwrap();
            }

            *prev_lidar_end_time = imu_span_end_time;
        }
    }
}

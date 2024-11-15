use nalgebra::*;

#[derive(Debug, Clone, Default)]
pub struct LifisParameters {
    pub undistort: UndistortParameters,
    pub imu: ImuParameters,
    pub downsample: DownsampleParameters,
    pub passthrough_filter: PassthroughFilterParameters,
}

#[derive(Debug, Clone)]
pub struct ImuCalibrationParameters {
    pub imu_intrinsics_calib_file_path: String,
    pub imu_calibrate_gyros: bool,
    pub gyro_calibration_samples: usize,
}

impl Default for ImuCalibrationParameters {
    fn default() -> Self {
        Self {
            imu_intrinsics_calib_file_path: "calib.txt".to_string(),
            imu_calibrate_gyros: true,
            gyro_calibration_samples: 100,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct UndistortParameters {
    pub lidar_to_imu: Isometry3<f64>,
}

#[derive(Debug, Clone)]
pub struct DownsampleParameters {
    pub voxel_size: f32,
}

impl Default for DownsampleParameters {
    fn default() -> Self {
        Self { voxel_size: 0.15 }
    }
}

#[derive(Debug, Clone)]
pub struct ImuParameters {
    pub gravity: f64,
    pub accel_noise: f64,
    pub gyro_noise: f64,
    pub accel_bias_noise: f64,
    pub gyro_bias_noise: f64,
    pub integration_noise: f64,
    pub bias_acc: Vector3<f64>, // TODO if we have a known bias, it is removed while applying the calibration?
    pub bias_gyro: Vector3<f64>,
    pub imu_lidar_tf_override: bool,
}

impl Default for ImuParameters {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            accel_noise: 1e-4,
            gyro_noise: 1e-4,
            accel_bias_noise: 1e-5,
            gyro_bias_noise: 1e-5,
            integration_noise: 1e-8,
            bias_acc: Vector3::new(0.0, 0.0, 0.0),
            bias_gyro: Vector3::new(0.0, 0.0, 0.0),
            imu_lidar_tf_override: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PassthroughFilterParameters {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
    pub invert_bounding_box: bool,
    pub min_dist: f32,
    pub max_dist: f32,
    pub invert_distance: bool,
    pub filter_frame_transform: Isometry3<f32>,
    pub fov_right: f32,
    pub fov_left: f32,
    pub forward: Vector2<f32>,
    pub enable_horizontal_fov: bool,
    pub polygon: Option<Vec<Point2<f32>>>,
    pub invert_polygon: bool,
    pub invert_fov: bool,
    pub min_intensity: f32,
    pub max_intensity: f32,
    pub invert_intensity: bool,
}

impl Default for PassthroughFilterParameters {
    fn default() -> Self {
        Self {
            min: Vector3::new(f32::MIN, f32::MIN, f32::MIN),
            max: Vector3::new(f32::MAX, f32::MAX, f32::MAX),
            invert_bounding_box: false,
            min_dist: 0.0,
            max_dist: f32::MAX / 2.5,
            invert_distance: false,
            filter_frame_transform: Isometry3::identity(),
            fov_right: 0.0,
            fov_left: 0.0,
            forward: Vector2::new(1.0, 0.0),
            enable_horizontal_fov: false,
            polygon: None,
            invert_polygon: false,
            invert_fov: false,
            min_intensity: f32::MIN,
            max_intensity: f32::MAX,
            invert_intensity: false,
        }
    }
}

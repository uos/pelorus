use nalgebra::*;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LifisParameters {
    pub imu: ImuParameters,
    pub downsample: DownsampleParameters,
    pub passthrough_filter: PassthroughFilterParameters,
    pub lidar_vendor: crate::LidarVendor,
    pub imu_lidar_tf_override: bool,
    pub global_zero_frame: String,
    pub map_frame: String,
    pub initial_scan_accumulate: usize,
    pub wheel_odom_cov: Option<f64>,
    pub wheel_imu_tf_override: bool,
    pub wheel_to_imu_rotation: nalgebra::Matrix3<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImuCalibrationParameters {
    pub imu_intrinsics_calib_file_path: Option<String>,
    pub imu_calibrate_gyros: bool,
    pub gyro_calibration_samples: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UndistortParameters {
    pub lidar_to_imu: Isometry3<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DownsampleParameters {
    pub voxel_size: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImuParameters {
    pub gravity: f64,
    pub imu_lidar_tf_override: bool,
    pub madgwick_setting: crate::MadgwickSetting,
    pub calibration: ImuCalibrationParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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

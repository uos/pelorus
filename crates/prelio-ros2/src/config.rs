use std::str::FromStr;

use prelio::config::LifisParameters;
use r2r::RosParams;

#[derive(RosParams, Debug)]
pub struct ConfigParse {
    pub verbose: bool,
    pub aabb_min_x: f32,
    pub aabb_max_x: f32,
    pub aabb_min_y: f32,
    pub aabb_max_y: f32,
    pub aabb_min_z: f32,
    pub aabb_max_z: f32,
    pub invert_aabb: bool,
    pub min_dist: f32,
    pub max_dist: f32,
    pub invert_distance: bool,
    pub global_zero_frame: String,
    pub invert_polygon: bool,
    pub map_frame: String,
    pub imu_lidar_tf_override: bool,
    pub fov_right: f32,
    pub fov_left: f32,
    pub fov_enabled: bool,
    pub invert_fov: bool,
    pub disable_with_first_polygon: bool,
    pub min_intensity: f32,
    pub max_intensity: f32,
    pub invert_intensity: bool,
    pub voxel_size: f32,
    pub imu_frame: String,
    pub lidar_vendor: String, // determine point types with timestamp format
    pub imu_intrinsic_calibration_file: String, // SM and bias
    pub imu_calibrate_gyros_at_startup: bool,
    pub gyro_calibration_samples: u32,
    pub gravity: f64,
    pub lidar_to_imu_pose_x: f64,
    pub lidar_to_imu_pose_y: f64,
    pub lidar_to_imu_pose_z: f64,
    pub lidar_to_imu_pose_roll: f64,
    pub lidar_to_imu_pose_pitch: f64,
    pub lidar_to_imu_pose_yaw: f64,
    pub madgwick_filter: String, // CreateOrientationOnly, FilterAccelerationOnly, FilterAccelerationAndCreateOrientation, Disable
}

#[derive(Debug, Clone, Default)]
pub enum LidarVendor {
    /// microseconds
    Velodyne,

    /// nanoseconds
    #[default]
    Ouster,
    // microseconds, TODO the driver currently does not publish timestamp per point in PointCloud2, only in custom messages
    //Livox,
    Skip,
}

#[derive(Debug, Clone)]
pub enum LidarParseError {
    UnknownVendor,
}

impl std::fmt::Display for LidarParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LidarParseError::UnknownVendor => write!(
                f,
                "Unknown LiDAR vendor, can not assume per point timestamp format"
            ),
        }
    }
}

impl std::error::Error for LidarParseError {}

impl FromStr for LidarVendor {
    type Err = LidarParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "velodyne" => Ok(LidarVendor::Velodyne),
            "ouster" => Ok(LidarVendor::Ouster),
            //"livox" => Ok(LidarVendor::Livox),
            "skip" => Ok(LidarVendor::Skip),
            _ => Err(LidarParseError::UnknownVendor),
        }
    }
}

impl Default for ConfigParse {
    fn default() -> Self {
        ConfigParse {
            verbose: false,
            aabb_min_x: f32::MIN,
            aabb_max_x: f32::MAX,
            aabb_min_y: f32::MIN,
            aabb_max_y: f32::MAX,
            aabb_min_z: f32::MIN,
            aabb_max_z: f32::MAX,
            invert_aabb: false,
            min_dist: 0.0,
            max_dist: f32::MAX / 2.5 as f32,
            invert_distance: false,
            global_zero_frame: "".to_string(),
            invert_polygon: false,
            imu_lidar_tf_override: false,
            map_frame: "".to_string(),
            fov_right: 0.0,
            fov_left: 0.0,
            fov_enabled: false,
            invert_fov: false,
            disable_with_first_polygon: false,
            min_intensity: f32::MIN,
            max_intensity: f32::MAX,
            invert_intensity: false,
            voxel_size: 0.15,
            imu_frame: "".to_string(),
            lidar_vendor: "ouster".to_string(),
            imu_intrinsic_calibration_file: "".to_string(),
            imu_calibrate_gyros_at_startup: true,
            gyro_calibration_samples: 100,
            gravity: 9.80665,
            lidar_to_imu_pose_x: 0.0,
            lidar_to_imu_pose_y: 0.0,
            lidar_to_imu_pose_z: 0.0,
            lidar_to_imu_pose_roll: 0.0,
            lidar_to_imu_pose_pitch: 0.0,
            lidar_to_imu_pose_yaw: 0.0,
            madgwick_filter: "CreateOrientationOnly".to_string(),
        }
    }
}

pub fn parse(config: &ConfigParse) -> anyhow::Result<prelio::config::LifisParameters> {
    let passthrough_filter = prelio::config::PassthroughFilterParameters {
        min: nalgebra::Vector3::new(config.aabb_min_x, config.aabb_min_y, config.aabb_min_z),
        max: nalgebra::Vector3::new(config.aabb_max_x, config.aabb_max_y, config.aabb_max_z),
        invert_bounding_box: config.invert_aabb,
        min_dist: config.min_dist,
        max_dist: config.max_dist,
        invert_distance: config.invert_distance,
        filter_frame_transform: nalgebra::Isometry3::identity(),
        fov_right: config.fov_right,
        fov_left: config.fov_left,
        forward: nalgebra::Vector2::new(1.0, 0.0),
        enable_horizontal_fov: config.fov_enabled,
        polygon: None,
        invert_polygon: config.invert_polygon,
        invert_fov: config.invert_fov,
        min_intensity: config.min_intensity,
        max_intensity: config.max_intensity,
        invert_intensity: config.invert_intensity,
    };

    let downsample = prelio::config::DownsampleParameters {
        voxel_size: config.voxel_size,
    };

    let undistort = prelio::config::UndistortParameters {
        lidar_to_imu: nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(
                config.lidar_to_imu_pose_x,
                config.lidar_to_imu_pose_y,
                config.lidar_to_imu_pose_z,
            ),
            nalgebra::UnitQuaternion::from_euler_angles(
                config.lidar_to_imu_pose_roll,
                config.lidar_to_imu_pose_pitch,
                config.lidar_to_imu_pose_yaw,
            ),
        ),
    };

    let mut imu = prelio::config::ImuParameters::default();
    imu.gravity = config.gravity;
    imu.imu_lidar_tf_override = config.imu_lidar_tf_override;

    Ok(LifisParameters {
        passthrough_filter,
        imu,
        downsample,
        undistort,
    })
}

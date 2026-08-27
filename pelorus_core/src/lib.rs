use bytemuck::{Pod, Zeroable};
use either::Either;
use nalgebra::{Isometry3, Matrix3, Point2, Point3, Vector2, Vector3};
use ros_pointcloud2::prelude::*;
use serde::{Deserialize, Serialize};

pub mod adaptive_execution;
pub mod config;
pub mod lie;
pub mod timing;

pub mod log_debouncer;

pub use lie::{SE3, SO3};

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct RobosensePoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: u8,
    pub timestamp: f64,
}

impl PointXYZITimestamp<5> for RobosensePoint {
    fn intensity(&self) -> f32 {
        self.intensity as f32 / 255.0 // TODO right?
    }

    fn timestamp(&self) -> f64 {
        self.timestamp * 1.0
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct RobosensePointNoIntensity {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub timestamp: f64,
}

impl PointXYZITimestamp<4> for RobosensePointNoIntensity {
    fn intensity(&self) -> f32 {
        0.0
    }

    fn timestamp(&self) -> f64 {
        self.timestamp * 1.0
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct VelodynePoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub time: f32,
}

impl PointXYZITimestamp<5> for VelodynePoint {
    fn intensity(&self) -> f32 {
        self.intensity / 255.0
    }

    fn timestamp(&self) -> f64 {
        self.time as f64
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct VelodynePointNoIntensity {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub time: f32,
}

impl PointXYZITimestamp<4> for VelodynePointNoIntensity {
    fn intensity(&self) -> f32 {
        1.0
    }

    fn timestamp(&self) -> f64 {
        self.time as f64
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct LivoxPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub t: u32,
    pub intensity: u32,
}

impl PointXYZITimestamp<5> for LivoxPoint {
    fn intensity(&self) -> f32 {
        self.intensity as f32 / 255.0
    }

    fn timestamp(&self) -> f64 {
        self.t as f64 * 1e-9
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct LivoxPointNoIntensity {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub t: u32,
}

impl PointXYZITimestamp<4> for LivoxPointNoIntensity {
    fn intensity(&self) -> f32 {
        1.0
    }

    fn timestamp(&self) -> f64 {
        self.t as f64 * 1e-9
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct LivoxPointOffsetTime {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    #[ros(remap("offset_time"))]
    pub t: u32,
    pub intensity: u32,
}

impl PointXYZITimestamp<5> for LivoxPointOffsetTime {
    fn intensity(&self) -> f32 {
        self.intensity as f32 / 255.0
    }

    fn timestamp(&self) -> f64 {
        self.t as f64 * 1e-9
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct LivoxPointOffsetTimeNoIntensity {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    #[ros(remap("offset_time"))]
    pub t: u32,
}

impl PointXYZITimestamp<4> for LivoxPointOffsetTimeNoIntensity {
    fn intensity(&self) -> f32 {
        1.0
    }

    fn timestamp(&self) -> f64 {
        self.t as f64 * 1e-9
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct LivoxMid360Point {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub timestamp: f64,
}

impl PointXYZITimestamp<5> for LivoxMid360Point {
    fn intensity(&self) -> f32 {
        self.intensity / 255.0
    }

    fn timestamp(&self) -> f64 {
        self.timestamp * 1e-9
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct LivoxMid360PointNoIntensity {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub timestamp: f64,
}

impl PointXYZITimestamp<4> for LivoxMid360PointNoIntensity {
    fn intensity(&self) -> f32 {
        1.0
    }

    fn timestamp(&self) -> f64 {
        self.timestamp * 1e-9
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct HesaiPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub timestamp: f64,
}

impl PointXYZITimestamp<5> for HesaiPoint {
    fn intensity(&self) -> f32 {
        self.intensity / 255.0
    }

    fn timestamp(&self) -> f64 {
        self.timestamp * 1.0
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct HesaiPointNoIntensity {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub timestamp: f64,
}

impl PointXYZITimestamp<4> for HesaiPointNoIntensity {
    fn intensity(&self) -> f32 {
        1.0
    }

    fn timestamp(&self) -> f64 {
        self.timestamp * 1.0
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct SickMultiScanPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub i: f32,
    pub ts: f32, // time offset in seconds relative to the header timestamp (4 byte, float32)
}

impl PointXYZITimestamp<5> for SickMultiScanPoint {
    fn intensity(&self) -> f32 {
        self.i as f32 / 65535.0
    }

    fn timestamp(&self) -> f64 {
        self.ts as f64
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct SickMultiScanPointNoIntensity {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub ts: f32, // time offset in seconds relative to the header timestamp (4 byte, float32)
}

impl PointXYZITimestamp<4> for SickMultiScanPointNoIntensity {
    fn intensity(&self) -> f32 {
        1.0
    }

    fn timestamp(&self) -> f64 {
        self.ts as f64
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct BlickfeldPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: u32,
    pub point_time_offset: u32,
}

impl PointXYZITimestamp<5> for BlickfeldPoint {
    fn intensity(&self) -> f32 {
        self.intensity as f32 / 255.0
    }

    fn timestamp(&self) -> f64 {
        self.point_time_offset as f64 * 1e-9
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct BlickfeldPointNoIntensity {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub point_time_offset: u32,
}

impl PointXYZITimestamp<4> for BlickfeldPointNoIntensity {
    fn intensity(&self) -> f32 {
        1.0
    }

    fn timestamp(&self) -> f64 {
        self.point_time_offset as f64 * 1e-9
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct OusterPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub t: u32,
}

impl PointXYZITimestamp<5> for OusterPoint {
    fn intensity(&self) -> f32 {
        self.intensity / 255.0
    }

    fn timestamp(&self) -> f64 {
        self.t as f64 * 1e-9
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct OusterPointNoIntensity {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub t: u32,
}

impl PointXYZITimestamp<4> for OusterPointNoIntensity {
    fn intensity(&self) -> f32 {
        1.0
    }

    fn timestamp(&self) -> f64 {
        self.t as f64 * 1e-9
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct PointXYZI {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
}

impl PointXYZIntensity<4> for PointXYZI {
    fn intensity(&self) -> f32 {
        self.intensity / 255.0
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible, Copy)]
pub struct PointXYZ {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl PointXYZIntensity<3> for PointXYZ {
    fn intensity(&self) -> f32 {
        1.0
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

// const MAX_INI_COUNT: usize = 100; # unused, using time instead
pub const COV_IMU_RAW_DEFAULT: f64 = 0.001;
pub const COV_IMU_BIAS_DEFAULT: f64 = 0.0001;

#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub struct ImuParameters {
    pub cov_acc: Vector3<f64>,
    pub cov_gyr: Vector3<f64>,
    pub cov_gyr_bias: Vector3<f64>,
    pub cov_acc_bias: Vector3<f64>,
    pub initial_translation_lidar_to_imu: Vector3<f64>,
    pub initial_rotation_lidar_to_imu: Matrix3<f64>,
    pub kinematic_constraint_std: f64,
    pub kinematic_model: KinematicModel,
    pub wheel_odom_cov_override: Option<f64>,
    /// Keep refining the LiDAR-to-IMU extrinsic in the filter instead of
    /// holding it at the configured value.
    pub estimate_extrinsic_online: bool,
    /// Ignore the IMU and propagate at a constant twist instead.
    pub lidar_only: bool,
}

impl Default for ImuParameters {
    fn default() -> Self {
        Self {
            cov_acc: Vector3::new(
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
            ),
            cov_gyr: Vector3::new(
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
            ),
            cov_gyr_bias: Vector3::new(
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
            ),
            cov_acc_bias: Vector3::new(
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
            ),
            initial_translation_lidar_to_imu: Vector3::default(),
            initial_rotation_lidar_to_imu: Matrix3::identity(),
            estimate_extrinsic_online: false,
            lidar_only: false,
            kinematic_constraint_std: 0.05,
            kinematic_model: KinematicModel::Differential,
            wheel_odom_cov_override: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Copy, Serialize, Deserialize)]
pub enum KinematicModel {
    /// No kinematic constraints are applied (full 3D motion)
    Full3D,
    /// 2D constraints for a differential drive (on-plane, no side-slip)
    Differential,
    /// 2D holonomic motion (can move in any direction on-plane)
    Holonomic2D,
    /// Ackermann steering model (car-like)
    Ackermann { imu_to_rear_axle: Vector3<f64> },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PrelioParameters {
    pub imu: ImuConfigParameter,
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
    pub warning_debounce: f64,
    pub lidar_as_sync: bool,
    pub parallelism_decision_threshold_stddevs: f64,
    pub parallelism_decision_samples_per_mode: usize,
    pub parallelism_execution_policy: adaptive_execution::ExecutionPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImuCalibrationParameters {
    pub imu_intrinsics_calib_file_path: Option<String>,
    pub imu_calibrate_gyros: bool,
    pub gyro_calibration_samples: usize,
    /// Seconds the IMU clock runs ahead of the LiDAR clock. Subtracted from
    /// every IMU timestamp to put both sensors on one clock.
    pub time_offset: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UndistortParameters {
    pub lidar_to_imu: Isometry3<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DownsampleParameters {
    pub voxel_size: f32,
    pub bypass_output_downsample: bool,
    pub point_skip: usize,
    pub backend: PreprocessingBackend,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Serialize, Deserialize, Default)]
pub enum PreprocessingBackend {
    Cpu,
    #[default]
    Backend,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImuConfigParameter {
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PrelioNodeParameters {
    pub wheel_src_topic: String,
    pub cloud_src_topic: String,
    pub imu_src_topic: String,
    pub logger_prefix: String,
    pub imu_filtered_topic: String,
    pub imu_predictor_init_time_sec: f64,
    pub cloud_qos: Qos,
    pub cloud_subscriber_mode: PointCloudMode,
    pub wheel_qos: Qos,
    pub imu_qos: Qos,
    pub imu_filtered_qos: Qos,
    pub synchronous_mode: bool,
}

#[cfg(feature = "r2r")]
impl From<Qos> for r2r::QosProfile {
    fn from(value: Qos) -> Self {
        match value {
            Qos::Sensor(reliability) => r2r::QosProfile::parameters()
                .durability(r2r::qos::DurabilityPolicy::Volatile)
                .deadline(std::time::Duration::MAX)
                .reliability(match reliability {
                    Reliability::Reliable => r2r::qos::ReliabilityPolicy::Reliable,
                    Reliability::BestEffort => r2r::qos::ReliabilityPolicy::BestEffort,
                })
                .history(r2r::qos::HistoryPolicy::KeepLast)
                .keep_last(5)
                .lifespan(std::time::Duration::MAX),
            _ => r2r::QosProfile::parameters()
                .durability(r2r::qos::DurabilityPolicy::Volatile)
                .deadline(std::time::Duration::MAX)
                .reliability(r2r::qos::ReliabilityPolicy::Reliable)
                .history(r2r::qos::HistoryPolicy::KeepLast)
                .keep_last(1)
                .lifespan(std::time::Duration::MAX),
        }
    }
}

#[derive(Serialize, Deserialize, Copy, Clone, Debug, Default)]
pub enum Reliability {
    Reliable,
    #[default]
    BestEffort,
}

#[derive(Serialize, Deserialize, Copy, Clone, Debug, Default)]
pub enum Qos {
    Sensor(Reliability),
    #[default]
    SystemDefault,
}

/// Point-cloud message type used by the `mt_pubsub` backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PointCloudMode {
    Compressed,
    #[default]
    PointCloud2,
}

/// Controls which estimator updates are published as odometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OdometryRate {
    /// Publish only LiDAR-corrected states (legacy behavior).
    #[default]
    Lidar,
    /// Publish live IMU predictions and LiDAR-corrected states.
    ImuAndLidar,
}

/// Top-level configuration used by both the binary and the FFI layer.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LioConfig {
    pub sensor_drivers: SensorDriverConfig,
    pub show_tui: bool,
    pub debug_fps: u32,
    pub registered_cloud_qos: Qos,
    pub cloud_publisher_mode: PointCloudMode,
    pub odom_qos: Qos,
    pub odometry_rate: OdometryRate,
    pub imu_odom_qos: Qos,
    pub imu_path_qos: Qos,
    pub path_qos: Qos,
    pub map_resolution: f64,
    pub map_capacity: usize,
    pub imu_params: ImuParameters,
    pub lidar_odom_cov: f64,
    pub enable_live_extrinsic_calibration: bool,
    pub prelio_params: PrelioParameters,
    pub prelio_node_params: PrelioNodeParameters,
    pub neighbor_max_range: f64,
    pub ekf_max_iterations: usize,
    pub dist_update_sensitivity: f32,
    pub rerun_undistorted_static_world: bool,
    pub base_link_frame: Option<String>,
    pub base_to_lidar: Isometry3<f64>,
    pub base_to_lidar_tf_override: bool,
    pub cpu_only: bool,
    pub num_threads: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SensorDriverConfig {
    pub lidar: LidarDriverConfig,
    pub imu: ImuDriverConfig,
    pub ouster: OusterConfig,
}

impl Default for SensorDriverConfig {
    fn default() -> Self {
        Self {
            lidar: Default::default(),
            imu: Default::default(),
            ouster: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum LidarDriverType {
    #[default]
    Disabled,
    Velodyne,
    Ouster,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ImuDriverType {
    #[default]
    Disabled,
    Xsens,
    Ouster,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VelodyneModel {
    Vlp16,
    Vlp32c,
    PuckLite,
    PuckHiRes,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VelodyneTimestampMode {
    Sensor,
    Host,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum XsensTimestampMode {
    Utc,
    Host,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OusterTimestampMode {
    Ptp,
    Host,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CoordinateSystem {
    Enu,
    Ned,
    Nwu,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LidarDriverConfig {
    pub driver_type: LidarDriverType,
    pub velodyne: VelodyneConfig,
}
impl Default for LidarDriverConfig {
    fn default() -> Self {
        Self {
            driver_type: Default::default(),
            velodyne: Default::default(),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImuDriverConfig {
    pub driver_type: ImuDriverType,
    pub xsens: XsensConfig,
}
impl Default for ImuDriverConfig {
    fn default() -> Self {
        Self {
            driver_type: Default::default(),
            xsens: Default::default(),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VelodyneConfig {
    pub model: VelodyneModel,
    pub bind_address: String,
    pub data_port: u16,
    pub position_port: u16,
    pub device_ip: Option<String>,
    pub frame_id: String,
    pub cut_angle_rad: f64,
    pub timestamp_mode: VelodyneTimestampMode,
    pub allow_host_hour_anchor: bool,
    pub max_host_skew_secs: f64,
    pub startup_timeout_secs: f64,
}
impl Default for VelodyneConfig {
    fn default() -> Self {
        Self {
            model: VelodyneModel::Vlp32c,
            bind_address: "0.0.0.0".into(),
            data_port: 2368,
            position_port: 8308,
            device_ip: None,
            frame_id: "velodyne".into(),
            cut_angle_rad: 0.0,
            timestamp_mode: VelodyneTimestampMode::Sensor,
            allow_host_hour_anchor: true,
            max_host_skew_secs: 0.020,
            startup_timeout_secs: 15.0,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct XsensConfig {
    pub port: String,
    pub baudrate: u32,
    pub frame_id: String,
    pub output_rate: u32,
    pub coordinate_system: CoordinateSystem,
    pub configure_device: bool,
    pub timestamp_mode: XsensTimestampMode,
    pub max_host_skew_secs: f64,
    pub startup_timeout_secs: f64,
}
impl Default for XsensConfig {
    fn default() -> Self {
        Self {
            port: "/dev/ttyUSB0".into(),
            baudrate: 115200,
            frame_id: "imu".into(),
            output_rate: 400,
            coordinate_system: CoordinateSystem::Enu,
            configure_device: true,
            timestamp_mode: XsensTimestampMode::Utc,
            max_host_skew_secs: 0.020,
            startup_timeout_secs: 15.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OusterConfig {
    pub hostname: String,
    pub metadata_path: Option<String>,
    pub bind_address: String,
    pub lidar_port: u16,
    pub imu_port: u16,
    pub device_ip: Option<String>,
    pub lidar_frame_id: String,
    pub imu_frame_id: String,
    pub timestamp_mode: OusterTimestampMode,
    pub max_host_skew_secs: f64,
    pub startup_timeout_secs: f64,
}
impl Default for OusterConfig {
    fn default() -> Self {
        Self {
            hostname: "os.local".into(),
            metadata_path: None,
            bind_address: "0.0.0.0".into(),
            lidar_port: 7502,
            imu_port: 7503,
            device_ip: None,
            lidar_frame_id: "os_sensor".into(),
            imu_frame_id: "os_imu".into(),
            timestamp_mode: OusterTimestampMode::Ptp,
            max_host_skew_secs: 0.020,
            startup_timeout_secs: 15.0,
        }
    }
}

pub fn timestamp_to_secs(t: &TimeMsg) -> f64 {
    t.sec as f64 + t.nanosec as f64 / 1_000_000_000.0
}

pub fn secs_to_timestamp(secs: f64) -> TimeMsg {
    let sec = secs as i32;
    let nanosec = ((secs - sec as f64) * 1_000_000_000.0) as u32;
    TimeMsg { sec, nanosec }
}

pub trait PointXYZITimestamp<const N: usize>:
    PointConvertible<N> + Send + Sync + Sized + Copy
{
    fn xyz(&self) -> Point3<f32>;

    /// Normalize to real 0..1 scale for filtering.
    /// When setting back the point, it will be converted to 0..255 (f32).
    fn intensity(&self) -> f32;

    /// Normalize vendor specific timestamps to seconds.
    /// The semantics of the per-point timestamp are handled elsewhere.
    fn timestamp(&self) -> f64;
}

pub trait PointXYZIntensity<const N: usize>:
    PointConvertible<N> + Send + Sync + Sized + Copy
{
    fn xyz(&self) -> Point3<f32>;

    /// Normalize to real 0..1 scale for filtering.
    /// When setting back the point, it will be converted to 0..255 (f32).
    fn intensity(&self) -> f32;
}

#[derive(Debug, Clone, Default)]
pub struct MeasureGroup {
    pub cloud: Vec<PointXYZIT>,
    pub last_lidar_end_stamp: f64,
    pub lidar_start_stamp: f64,
    pub lidar_end_stamp: f64,
    pub imu_buffer: Vec<ImuMsg>,
    pub odom_buffer: Vec<OdomMsg>,
}

#[derive(Debug, Clone, Default, Copy, PointConvertible, PartialEq, PartialOrd, Pod, Zeroable)]
#[repr(C)]
pub struct PointXYZIT {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub timestamp: f32,
}

const _: [u8; 20] = [0u8; std::mem::size_of::<PointXYZIT>()];

impl PointXYZIT {
    pub fn from_p(point: Point3<f32>, intensity: f32, timestamp: f32) -> Self {
        Self {
            x: point.x,
            y: point.y,
            z: point.z,
            intensity,
            timestamp,
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
    pub linear_velocity: Vector3<f64>,
    pub angular_velocity: Vector3<f64>,
    // pub lidar_imu_extrinsic_rotation: nalgebra::Matrix3<f64>,
    // pub lidar_imu_extrinsic_translation: nalgebra::Vector3<f64>,
}

#[derive(Debug)]
pub struct PrelioProcessedCloud {
    pub scan_id: u64,
    pub received_at: std::time::Instant,
    pub cloud: Vec<PointXYZIT>,
    pub passthrough: Option<Vec<PointXYZIT>>,
    /// IMU measurements synchronized to this LiDAR frame.
    pub imu_buffer: Vec<ImuMsg>,
    pub imu_frame: String,
    pub header: HeaderMsg,
    pub end_of_frame_guess: Isometry3<f64>,
    pub end_of_frame_time: TimeMsg,
    pub lidar_end_time_abs: TimeMsg, // TODO merge with upper end_frame_time. currently changing the upper leads to strange effects. probably misused
    pub end_of_frame_twist: Twist,
    pub pose_covariance: [f64; 36],
    pub twist_covariance: [f64; 36],
    pub should_map_only: bool,
    pub runtime_eval: timing::RuntimeEvaluator,
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Default,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum MadgwickSetting {
    CreateOrientationOnly,
    FilterAccelerationOnly,
    #[default]
    FilterAccelerationAndCreateOrientation,
    Disable,
}

pub type ImuPredictor = dyn for<'a> Fn(&'a [ImuMsg], f64, f64, f64) -> ImuPosePrediction + Send;

pub type OdomCorrection = dyn Fn(OdomMsg) + Send;

pub type ImuPredictorUndistort = dyn for<'a, 'b> Fn(
        &'a [ImuMsg],
        &'b [OdomMsg],
        f64,
        f64,
        f64,
        &mut Vec<PointXYZIT>,
        Option<&mut Vec<PointXYZIT>>,
    ) -> Either<ImuPosePrediction, Isometry3<f64>>
    + Send;

pub fn imu_msg_to_cov(msg: &ImuMsg) -> [f64; 36] {
    let mut covariances = [0.0; 36];
    covariances[0] =
        (msg.linear_acceleration_covariance[0] + msg.angular_velocity_covariance[0]) / 2.0;
    covariances[7] =
        (msg.linear_acceleration_covariance[4] + msg.angular_velocity_covariance[4]) / 2.0;
    covariances[14] =
        (msg.linear_acceleration_covariance[8] + msg.angular_velocity_covariance[8]) / 2.0;

    covariances[21] =
        (msg.linear_acceleration_covariance[0] + msg.angular_velocity_covariance[0]) / 2.0;
    covariances[28] =
        (msg.linear_acceleration_covariance[4] + msg.angular_velocity_covariance[4]) / 2.0;
    covariances[35] =
        (msg.linear_acceleration_covariance[8] + msg.angular_velocity_covariance[8]) / 2.0;

    covariances
}

pub fn imu_msg_to_twist(msg: &ImuMsg) -> [f64; 6] {
    let mut twist = [0.0; 6];
    twist[0] = msg.linear_acceleration[0];
    twist[1] = msg.linear_acceleration[1];
    twist[2] = msg.linear_acceleration[2];
    twist[3] = msg.angular_velocity[0];
    twist[4] = msg.angular_velocity[1];
    twist[5] = msg.angular_velocity[2];
    twist
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, Copy)]
pub enum LidarVendor {
    Velodyne,
    Livox,
    LivoxMid360,
    #[default]
    Ouster,
    /// microseconds?
    Robosense,
    Hesai,
    Blickfeld,
    SickMultiscan,
    NoTimestamp,
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

#[derive(Debug, Clone, Default)]
pub struct ImuMsg {
    pub orientation: [f64; 4],
    pub orientation_covariance: [f64; 9],
    pub angular_velocity: [f64; 3],
    pub angular_velocity_covariance: [f64; 9],
    pub linear_acceleration: [f64; 3],
    pub linear_acceleration_covariance: [f64; 9],
    pub frame_id: String,
    pub stamp: crate::TimeMsg,
}

pub fn time_msg_from_secs(secs: f64) -> crate::TimeMsg {
    let sec = secs.floor() as i32;
    let nanosec = ((secs - secs.floor()) * 1e9).floor() as u32;
    crate::TimeMsg { sec, nanosec }
}

#[derive(Debug, Clone, Copy)]
pub struct OdomMsg {
    pub twist_linear: nalgebra::Vector3<f64>,
    pub twist_angular: nalgebra::Vector3<f64>,
    pub twist_cov: [f64; 36],

    pub stamp: crate::TimeMsg,
}

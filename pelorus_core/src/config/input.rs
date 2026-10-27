//! Typed representation of the Ratslang configuration file.
//! Constraints stay in default.rl; adapters bridge quantities and runtime types.
use super::*;

// Typed input uses Ratslang's native uom/range adapters. Runtime units remain SI.
fn metres<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    ratslang::serde::length::deserialize(d).map(|v| v.get::<meter>())
}
fn seconds<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    ratslang::serde::time::deserialize(d).map(|v| v.get::<second>())
}
#[derive(Deserialize)]
struct Distance(#[serde(deserialize_with = "metres")] f64);
fn translation<'de, D: Deserializer<'de>>(d: D) -> Result<Vector3<f64>, D::Error> {
    <[Distance; 3]>::deserialize(d).map(|v| Vector3::new(v[0].0, v[1].0, v[2].0))
}
fn matrix<'de, D: Deserializer<'de>>(d: D) -> Result<Matrix3<f64>, D::Error> {
    <[[f64; 3]; 3]>::deserialize(d).map(|m| Matrix3::from_row_slice(&m.concat()))
}
fn forward<'de, D: Deserializer<'de>>(d: D) -> Result<Vector2<f32>, D::Error> {
    <[f32; 2]>::deserialize(d).map(Vector2::from)
}
fn polygon<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Point2<f32>>, D::Error> {
    Vec::<[Distance; 2]>::deserialize(d).map(|v| {
        v.into_iter()
            .map(|p| Point2::new(p[0].0 as f32, p[1].0 as f32))
            .collect()
    })
}
fn optional_string<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    String::deserialize(d).map(|s| (!s.is_empty()).then_some(s))
}
fn ip_address<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let value = String::deserialize(d)?;
    value
        .parse::<std::net::IpAddr>()
        .map_err(serde::de::Error::custom)?;
    Ok(value)
}
fn optional_ip<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    let value = optional_string(d)?;
    if let Some(value) = &value {
        value
            .parse::<std::net::IpAddr>()
            .map_err(serde::de::Error::custom)?;
    }
    Ok(value)
}
fn thread_count<'de, D: Deserializer<'de>>(d: D) -> Result<usize, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Threads {
        Count(usize),
        Name(String),
    }
    match Threads::deserialize(d)? {
        Threads::Count(n) => Ok(n),
        Threads::Name(name) if name.eq_ignore_ascii_case("all") => Ok(0),
        _ => Err(serde::de::Error::custom(
            "parallelism.num_threads must be a non-negative integer or All",
        )),
    }
}
#[derive(Deserialize)]
pub(super) struct LengthRange(
    #[serde(with = "ratslang::serde::range")] ratslang::serde::Range<Distance>,
);
impl LengthRange {
    pub(super) fn bounds(self, from: f64, to: f64) -> (f64, f64) {
        (
            self.0.start.map_or(from, |v| v.0),
            self.0.end.map_or(to, |v| v.0),
        )
    }
}
#[derive(Deserialize)]
pub(super) struct NumberRange(
    #[serde(with = "ratslang::serde::range")] ratslang::serde::Range<f64>,
);
impl NumberRange {
    pub(super) fn bounds(self, from: f64, to: f64) -> (f64, f64) {
        (self.0.start.unwrap_or(from), self.0.end.unwrap_or(to))
    }
}
// Runtime enums have public serialization formats of their own. These local
// derived enums accept the case-insensitive Ratslang spelling and aliases.
macro_rules! named {
    ($name:ident, $ty:ty, {$($(#[$attr:meta])* $variant:ident => $value:expr,)*}) => {
        fn $name<'de, D: Deserializer<'de>>(d: D) -> Result<$ty, D::Error> {
            #[derive(Deserialize)]
            enum Choice {$($(#[$attr])* $variant,)*}
            let text = String::deserialize(d)?.to_ascii_lowercase();
            let choice = Choice::deserialize(serde::de::value::StringDeserializer::<D::Error>::new(text))?;
            Ok(match choice {$(Choice::$variant => $value,)*})
        }
    };
}
#[derive(Deserialize)]
pub(super) struct RawImuAhrsFiltering {
    #[serde(deserialize_with = "name_madgwicksetting")]
    pub(super) method: MadgwickSetting,
}

#[derive(Deserialize)]
pub(super) struct RawImuCalibrationGyroscope {
    #[serde(rename = "$value")]
    pub(super) value: bool,
    pub(super) samples: usize,
}

#[derive(Deserialize)]
pub(super) struct RawImuCalibration {
    pub(super) gyroscope: RawImuCalibrationGyroscope,
    #[serde(deserialize_with = "optional_string")]
    pub(super) parameter_file: Option<String>,
    #[serde(deserialize_with = "seconds")]
    pub(super) time_offset: f64,
}

#[derive(Deserialize)]
pub(super) struct RawImuDriverXsensTimestamp {
    #[serde(deserialize_with = "seconds")]
    pub(super) max_host_skew: f64,
    #[serde(deserialize_with = "name_xsenstimestampmode")]
    pub(super) mode: XsensTimestampMode,
    #[serde(deserialize_with = "seconds")]
    pub(super) startup_timeout: f64,
}

#[derive(Deserialize)]
pub(super) struct RawImuDriverXsens {
    pub(super) baudrate: u32,
    pub(super) configure_device: bool,
    #[serde(deserialize_with = "name_coordinatesystem")]
    pub(super) coordinate_system: CoordinateSystem,
    pub(super) frame_id: String,
    pub(super) output_rate: u32,
    pub(super) port: String,
    pub(super) timestamp: RawImuDriverXsensTimestamp,
}

#[derive(Deserialize)]
pub(super) struct RawImuDriver {
    #[serde(deserialize_with = "name_imudrivertype")]
    pub(super) r#type: ImuDriverType,
    pub(super) xsens: RawImuDriverXsens,
}

#[derive(Deserialize)]
pub(super) struct RawImuFromLidarTransform {
    pub(super) estimate_online: bool,
    #[serde(deserialize_with = "matrix")]
    pub(super) rotation: Matrix3<f64>,
    #[serde(deserialize_with = "translation")]
    pub(super) translation: Vector3<f64>,
    pub(super) use_tf_tree: bool,
}

#[derive(Deserialize)]
pub(super) struct RawImuQos {
    #[serde(deserialize_with = "name_qos")]
    pub(super) reliability: Qos,
}

#[derive(Deserialize)]
pub(super) struct RawImu {
    pub(super) ahrs_filtering: RawImuAhrsFiltering,
    pub(super) calibration: RawImuCalibration,
    pub(super) covariance_acceleration: f64,
    pub(super) covariance_acceleration_bias: f64,
    pub(super) covariance_gyroscope: f64,
    pub(super) covariance_gyroscope_bias: f64,
    pub(super) driver: RawImuDriver,
    pub(super) from_lidar_transform: RawImuFromLidarTransform,
    pub(super) gravity: f64,
    #[serde(deserialize_with = "seconds")]
    pub(super) initial_setup: f64,
    pub(super) lidar_only: bool,
    pub(super) qos: RawImuQos,
    pub(super) topic: String,
}

#[derive(Deserialize)]
pub(super) struct RawLidarDriverRobosenseTimestamp {
    #[serde(deserialize_with = "seconds")]
    pub(super) max_host_skew: f64,
    #[serde(deserialize_with = "name_robosensetimestampmode")]
    pub(super) mode: RobosenseTimestampMode,
    #[serde(deserialize_with = "seconds")]
    pub(super) startup_timeout: f64,
}

#[derive(Deserialize)]
pub(super) struct RawLidarDriverRobosense {
    #[serde(deserialize_with = "ip_address")]
    pub(super) bind_address: String,
    #[serde(deserialize_with = "optional_ip")]
    pub(super) device_ip: Option<String>,
    pub(super) difop_port: u16,
    pub(super) frame_id: String,
    #[serde(deserialize_with = "name_robosensemodel")]
    pub(super) model: RobosenseModel,
    pub(super) msop_port: u16,
    pub(super) timestamp: RawLidarDriverRobosenseTimestamp,
}

#[derive(Deserialize)]
pub(super) struct RawLidarDriverVelodyneTimestamp {
    pub(super) allow_host_hour_anchor: bool,
    #[serde(deserialize_with = "seconds")]
    pub(super) max_host_skew: f64,
    #[serde(deserialize_with = "name_velodynetimestampmode")]
    pub(super) mode: VelodyneTimestampMode,
    #[serde(deserialize_with = "seconds")]
    pub(super) startup_timeout: f64,
}

#[derive(Deserialize)]
pub(super) struct RawLidarDriverVelodyne {
    #[serde(deserialize_with = "ip_address")]
    pub(super) bind_address: String,
    pub(super) cut_angle: f64,
    pub(super) data_port: u16,
    #[serde(deserialize_with = "optional_ip")]
    pub(super) device_ip: Option<String>,
    pub(super) frame_id: String,
    #[serde(deserialize_with = "name_velodynemodel")]
    pub(super) model: VelodyneModel,
    pub(super) position_port: u16,
    pub(super) timestamp: RawLidarDriverVelodyneTimestamp,
}

#[derive(Deserialize)]
pub(super) struct RawLidarDriver {
    pub(super) robosense: RawLidarDriverRobosense,
    #[serde(deserialize_with = "name_lidardrivertype")]
    pub(super) r#type: LidarDriverType,
    pub(super) velodyne: RawLidarDriverVelodyne,
}

#[derive(Deserialize)]
pub(super) struct RawLidar {
    pub(super) as_sync: bool,
    pub(super) covariance: f64,
    pub(super) driver: RawLidarDriver,
    pub(super) qos: RawImuQos,
    #[serde(deserialize_with = "name_pointcloudmode")]
    pub(super) subscriber_mode: PointCloudMode,
    pub(super) topic: String,
    #[serde(deserialize_with = "name_lidarvendor")]
    pub(super) vendor: LidarVendor,
}

#[derive(Deserialize)]
pub(super) struct RawOutputBaseLinkFromLidarTransform {
    #[serde(deserialize_with = "matrix")]
    pub(super) rotation: Matrix3<f64>,
    #[serde(deserialize_with = "translation")]
    pub(super) translation: Vector3<f64>,
    pub(super) use_tf_tree: bool,
}

#[derive(Deserialize)]
pub(super) struct RawOutputBaseLink {
    #[serde(deserialize_with = "optional_string")]
    pub(super) frame_id: Option<String>,
    pub(super) from_lidar_transform: RawOutputBaseLinkFromLidarTransform,
}

#[derive(Deserialize)]
pub(super) struct RawOutputMcap {
    pub(super) cloud_topic: String,
    #[serde(deserialize_with = "optional_string")]
    pub(super) file: Option<String>,
    pub(super) odom_topic: String,
    pub(super) path_topic: String,
    #[serde(deserialize_with = "optional_string")]
    pub(super) rosbag_dir: Option<String>,
    #[serde(deserialize_with = "optional_string")]
    pub(super) rosbag_metadata: Option<String>,
    pub(super) tf_topic: String,
}

#[derive(Deserialize)]
pub(super) struct RawOutputRerunMap {
    #[serde(rename = "$value")]
    pub(super) value: bool,
    #[serde(deserialize_with = "seconds")]
    pub(super) publish_interval: f64,
    pub(super) visual_scale: f32,
}

#[derive(Deserialize)]
pub(super) struct RawOutputRerun {
    #[serde(rename = "$value")]
    pub(super) value: bool,
    pub(super) map: RawOutputRerunMap,
    pub(super) server_addr: String,
}

#[derive(Deserialize)]
pub(super) struct RawOutputTerminal {
    pub(super) debug: bool,
    pub(super) debug_fps: u32,
    #[serde(default)]
    pub(super) log_level: Option<String>,
    pub(super) ui: bool,
    #[serde(deserialize_with = "seconds")]
    pub(super) warning_debounce: f64,
}

#[derive(Deserialize)]
pub(super) struct RawOutput {
    pub(super) base_link: RawOutputBaseLink,
    pub(super) global_frame: String,
    #[serde(deserialize_with = "name_qos")]
    pub(super) imu_ahrs_filtered_msg_qos: Qos,
    pub(super) imu_ahrs_filtered_topic: String,
    #[serde(deserialize_with = "name_qos")]
    pub(super) imu_odometry_msg_qos: Qos,
    #[serde(deserialize_with = "name_qos")]
    pub(super) imu_path_msg_qos: Qos,
    pub(super) latency: String,
    #[serde(deserialize_with = "name_qos")]
    pub(super) lio_odometry_msg_qos: Qos,
    #[serde(deserialize_with = "name_qos")]
    pub(super) lio_path_msg_qos: Qos,
    pub(super) logger_prefix: String,
    pub(super) mcap: RawOutputMcap,
    #[serde(deserialize_with = "name_odometryrate")]
    pub(super) odometry_rate: OdometryRate,
    pub(super) processing_time: String,
    #[serde(deserialize_with = "name_pointcloudmode")]
    pub(super) publisher_mode: PointCloudMode,
    pub(super) rerun: RawOutputRerun,
    pub(super) terminal: RawOutputTerminal,
    pub(super) trajectory: String,
}

#[derive(Deserialize)]
pub(super) struct RawParallelism {
    #[serde(deserialize_with = "name_pipelining")]
    pub(super) pipelining: Pipelining,
    pub(super) cpu_only: bool,
    pub(super) decision_samples_per_mode: usize,
    pub(super) decision_threshold_stddevs: f64,
    #[serde(deserialize_with = "name_executionpolicy")]
    pub(super) execution_mode: ExecutionPolicy,
    #[serde(deserialize_with = "thread_count")]
    pub(super) num_threads: usize,
}

#[derive(Deserialize)]
pub(super) struct RawPreprocessingDownsample {
    #[serde(deserialize_with = "name_preprocessingbackend")]
    pub(super) backend: PreprocessingBackend,
    pub(super) bypass_output_downsample: bool,
    pub(super) point_skip: usize,
    pub(super) query_resolution_factor: f64,
}

#[derive(Deserialize)]
pub(super) struct RawPreprocessingFilterBoundingBox {
    pub(super) invert: bool,
    pub(super) x: LengthRange,
    pub(super) y: LengthRange,
    pub(super) z: LengthRange,
}

#[derive(Deserialize)]
pub(super) struct RawPreprocessingFilterDistance {
    pub(super) invert: bool,
    pub(super) range: LengthRange,
}

#[derive(Deserialize)]
pub(super) struct RawPreprocessingFilterHorizontalFov {
    #[serde(rename = "$value")]
    pub(super) value: bool,
    pub(super) degrees: NumberRange,
    #[serde(deserialize_with = "forward")]
    pub(super) forward: Vector2<f32>,
    pub(super) invert: bool,
}

#[derive(Deserialize)]
pub(super) struct RawPreprocessingFilterIntensity {
    pub(super) invert: bool,
    pub(super) values: NumberRange,
}

#[derive(Deserialize)]
pub(super) struct RawPreprocessingFilterPolygon2d {
    #[serde(deserialize_with = "polygon")]
    pub(super) coords: Vec<Point2<f32>>,
    pub(super) global_zero_frame: String,
    pub(super) invert: bool,
}

#[derive(Deserialize)]
pub(super) struct RawPreprocessingFilter {
    pub(super) bounding_box: RawPreprocessingFilterBoundingBox,
    pub(super) distance: RawPreprocessingFilterDistance,
    pub(super) horizontal_fov: RawPreprocessingFilterHorizontalFov,
    pub(super) intensity: RawPreprocessingFilterIntensity,
    pub(super) polygon_2d: RawPreprocessingFilterPolygon2d,
}

#[derive(Deserialize)]
pub(super) struct RawPreprocessing {
    pub(super) downsample: RawPreprocessingDownsample,
    pub(super) filter: RawPreprocessingFilter,
    pub(super) initial_scan_accumulate: usize,
}

#[derive(Deserialize)]
pub(super) struct RawRegistrationMapResolution {
    #[serde(rename = "$value", deserialize_with = "metres")]
    pub(super) value: f64,
    #[serde(deserialize_with = "name_bool")]
    pub(super) r#type: bool,
}

#[derive(Deserialize)]
pub(super) struct RawRegistrationMap {
    pub(super) capacity: usize,
    pub(super) distribution_update_sensitivity: f32,
    pub(super) resolution: RawRegistrationMapResolution,
}

#[derive(Deserialize)]
pub(super) struct RawRegistration {
    pub(super) map: RawRegistrationMap,
    pub(super) max_iter: usize,
    #[serde(deserialize_with = "metres")]
    pub(super) neighbors_max_range: f64,
}

#[derive(Deserialize)]
pub(super) struct RawSensorDriversOusterTimestamp {
    #[serde(deserialize_with = "seconds")]
    pub(super) max_host_skew: f64,
    #[serde(deserialize_with = "name_oustertimestampmode")]
    pub(super) mode: OusterTimestampMode,
    #[serde(deserialize_with = "seconds")]
    pub(super) startup_timeout: f64,
}

#[derive(Deserialize)]
pub(super) struct RawSensorDriversOuster {
    #[serde(deserialize_with = "ip_address")]
    pub(super) bind_address: String,
    #[serde(deserialize_with = "optional_ip")]
    pub(super) device_ip: Option<String>,
    pub(super) hostname: String,
    pub(super) imu_frame_id: String,
    pub(super) imu_port: u16,
    pub(super) lidar_frame_id: String,
    pub(super) lidar_port: u16,
    #[serde(deserialize_with = "optional_string")]
    pub(super) metadata_path: Option<String>,
    pub(super) timestamp: RawSensorDriversOusterTimestamp,
}

#[derive(Deserialize)]
pub(super) struct RawSensorDrivers {
    pub(super) ouster: RawSensorDriversOuster,
}

#[derive(Deserialize)]
pub(super) struct RawWheelAckermann {
    #[serde(deserialize_with = "translation")]
    pub(super) imu_to_rear_axle: Vector3<f64>,
}

#[derive(Deserialize)]
pub(super) struct RawWheelToImuTransform {
    #[serde(deserialize_with = "matrix")]
    pub(super) rotation: Matrix3<f64>,
    pub(super) use_tf_tree: bool,
}

#[derive(Deserialize)]
pub(super) struct RawWheel {
    pub(super) ackermann: RawWheelAckermann,
    pub(super) kinematic_constraint_std: f64,
    #[serde(deserialize_with = "name_kinematicmodel")]
    pub(super) kinematic_model: KinematicModel,
    pub(super) qos: RawImuQos,
    pub(super) to_imu_transform: RawWheelToImuTransform,
    pub(super) topic: String,
    #[serde(default)]
    pub(super) covariance: Option<f64>,
}

#[derive(Deserialize)]
pub(super) struct RawConfig {
    pub(super) imu: RawImu,
    pub(super) lidar: RawLidar,
    pub(super) offline: crate::OfflineConfig,
    pub(super) output: RawOutput,
    pub(super) parallelism: RawParallelism,
    pub(super) preprocessing: RawPreprocessing,
    pub(super) registration: RawRegistration,
    pub(super) sensor_drivers: RawSensorDrivers,
    pub(super) wheel: RawWheel,
}

named!(name_qos, Qos, {
    #[serde(rename = "besteffort")]
    V0 => Qos::Sensor(Reliability::BestEffort),
    #[serde(rename = "reliable")]
    V1 => Qos::Sensor(Reliability::Reliable),
    #[serde(rename = "systemdefault")]
    V2 => Qos::SystemDefault,
});

named!(name_pointcloudmode, PointCloudMode, {
    #[serde(rename = "compressed")]
    V0 => PointCloudMode::Compressed,
    #[serde(rename = "pointcloud2")]
    V1 => PointCloudMode::PointCloud2,
});

named!(name_lidarvendor, LidarVendor, {
    #[serde(rename = "velodyne")]
    V0 => LidarVendor::Velodyne,
    #[serde(rename = "ouster")]
    V1 => LidarVendor::Ouster,
    #[serde(rename = "livox")]
    V2 => LidarVendor::Livox,
    #[serde(rename = "livoxmid360")]
    V3 => LidarVendor::LivoxMid360,
    #[serde(rename = "robosense")]
    V4 => LidarVendor::Robosense,
    #[serde(rename = "hesai")]
    V5 => LidarVendor::Hesai,
    #[serde(rename = "blickfeld")]
    V6 => LidarVendor::Blickfeld,
    #[serde(rename = "sickmultiscan")]
    V7 => LidarVendor::SickMultiscan,
    #[serde(rename = "notimestamp")]
    #[serde(alias = "no_timestamp")]
    #[serde(alias = "genericnotimestamp")]
    #[serde(alias = "generic_no_timestamp")]
    V8 => LidarVendor::NoTimestamp,
});

named!(name_lidardrivertype, LidarDriverType, {
    #[serde(rename = "disabled")]
    V0 => LidarDriverType::Disabled,
    #[serde(rename = "velodyne")]
    V1 => LidarDriverType::Velodyne,
    #[serde(rename = "ouster")]
    V2 => LidarDriverType::Ouster,
    #[serde(rename = "robosense")]
    V3 => LidarDriverType::Robosense,
});

named!(name_imudrivertype, ImuDriverType, {
    #[serde(rename = "disabled")]
    V0 => ImuDriverType::Disabled,
    #[serde(rename = "xsens")]
    V1 => ImuDriverType::Xsens,
    #[serde(rename = "ouster")]
    V2 => ImuDriverType::Ouster,
});

named!(name_velodynemodel, VelodyneModel, {
    #[serde(rename = "vlp16")]
    V0 => VelodyneModel::Vlp16,
    #[serde(rename = "vlp32c")]
    V1 => VelodyneModel::Vlp32c,
    #[serde(rename = "pucklite")]
    V2 => VelodyneModel::PuckLite,
    #[serde(rename = "puckhires")]
    V3 => VelodyneModel::PuckHiRes,
});

named!(name_robosensemodel, RobosenseModel, {
    #[serde(rename = "m1")]
    V0 => RobosenseModel::M1,
    #[serde(rename = "m2")]
    V1 => RobosenseModel::M2,
    #[serde(rename = "e1")]
    V2 => RobosenseModel::E1,
    #[serde(rename = "mx")]
    V3 => RobosenseModel::Mx,
});

named!(name_velodynetimestampmode, VelodyneTimestampMode, {
    #[serde(rename = "sensor")]
    V0 => VelodyneTimestampMode::Sensor,
    #[serde(rename = "host")]
    V1 => VelodyneTimestampMode::Host,
});

named!(name_robosensetimestampmode, RobosenseTimestampMode, {
    #[serde(rename = "sensor")]
    V0 => RobosenseTimestampMode::Sensor,
    #[serde(rename = "host")]
    V1 => RobosenseTimestampMode::Host,
});

named!(name_xsenstimestampmode, XsensTimestampMode, {
    #[serde(rename = "utc")]
    V0 => XsensTimestampMode::Utc,
    #[serde(rename = "host")]
    V1 => XsensTimestampMode::Host,
});

named!(name_oustertimestampmode, OusterTimestampMode, {
    #[serde(rename = "ptp")]
    V0 => OusterTimestampMode::Ptp,
    #[serde(rename = "host")]
    V1 => OusterTimestampMode::Host,
});

named!(name_coordinatesystem, CoordinateSystem, {
    #[serde(rename = "enu")]
    V0 => CoordinateSystem::Enu,
    #[serde(rename = "ned")]
    V1 => CoordinateSystem::Ned,
    #[serde(rename = "nwu")]
    V2 => CoordinateSystem::Nwu,
});

named!(name_madgwicksetting, MadgwickSetting, {
    #[serde(rename = "createorientationonly")]
    V0 => MadgwickSetting::CreateOrientationOnly,
    #[serde(rename = "filteraccelerationandcreateorientation")]
    V1 => MadgwickSetting::FilterAccelerationAndCreateOrientation,
    #[serde(rename = "filteraccelerationonly")]
    V2 => MadgwickSetting::FilterAccelerationOnly,
    #[serde(rename = "disable")]
    V3 => MadgwickSetting::Disable,
});

named!(name_preprocessingbackend, PreprocessingBackend, {
    #[serde(rename = "cpu")]
    V0 => PreprocessingBackend::Cpu,
    #[serde(rename = "backend")]
    V1 => PreprocessingBackend::Backend,
});

named!(name_executionpolicy, ExecutionPolicy, {
    #[serde(rename = "auto")]
    V0 => ExecutionPolicy::Auto,
    #[serde(rename = "singlecore")]
    #[serde(alias = "single_core")]
    #[serde(alias = "single-core")]
    V1 => ExecutionPolicy::SingleCore,
    #[serde(rename = "rayon")]
    V2 => ExecutionPolicy::Rayon,
});

named!(name_odometryrate, OdometryRate, {
    #[serde(rename = "lidar")]
    V0 => OdometryRate::Lidar,
    #[serde(rename = "imuandlidar")]
    #[serde(alias = "imu_and_lidar")]
    V1 => OdometryRate::ImuAndLidar,
});

named!(name_bool, bool, {
    #[serde(rename = "single")]
    V0 => false,
    #[serde(rename = "multi")]
    V1 => true,
});

named!(name_pipelining, Pipelining, {
    #[serde(rename = "pipelined")]
    V0 => Pipelining::Pipelined,
    #[serde(rename = "sequential")]
    V1 => Pipelining::Sequential,
});

named!(name_kinematicmodel, KinematicModel, {
    #[serde(rename = "full3d")]
    Full3D => KinematicModel::Full3D,
    #[serde(rename = "differential", alias = "differentialdrive", alias = "wheeled2d")]
    Differential => KinematicModel::Differential,
    #[serde(rename = "holonomic2d", alias = "holonomic")]
    Holonomic => KinematicModel::Holonomic2D,
    #[serde(rename = "ackermann")]
    Ackermann => KinematicModel::Ackermann { imu_to_rear_axle: Vector3::zeros() },
});

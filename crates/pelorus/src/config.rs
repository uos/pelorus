use anyhow::anyhow;
use nalgebra::{Matrix3, Point2, Vector2, Vector3};
use prelio_ros2::prelio::config::{
    DownsampleParameters, ImuCalibrationParameters, ImuParameters, LifisParameters,
};
use prelio_ros2::prelio::{config::PassthroughFilterParameters, LidarVendor, MadgwickSetting};
use prelio_ros2::{LifisNodeParameters, Qos, Reliability};
use ratslang::{Evaluated, NumVal, Rhs, Unit, UnitVal, Val, VariableHistory};
use uom::si::angle::{degree, radian};
use uom::si::f32::{Angle, Length, Time};
use uom::si::length::{meter, millimeter};
use uom::si::time::{millisecond, second};

use crate::LioConfig;

//================================================================================
//  Helper Macros
//================================================================================

/// Resolves a variable from the user or default configuration, matching a specific pattern.
macro_rules! resolve_var {
    // Arm for variable names passed as identifiers
    ($asts:expr, $var_name_ident:ident, as $target_type:ty, $($pattern:pat_param)|+ => $extraction_block:block) => {{
        let __var_name_str = stringify!($var_name_ident);
        // Use resolve() which returns a Result<Option<...>>
        let __resolved_opt = match $asts.user.resolve(__var_name_str)? {
            Some(val) => Some(val),
            None => $asts.defaults.resolve(__var_name_str)?,
        };

        match __resolved_opt {
            Some(__resolved_val) => match __resolved_val {
                $($pattern)|+ => {
                    let __extracted_val = $extraction_block;
                    Ok(__extracted_val)
                }
                _ => Err(anyhow!(format!(
                    "Pattern mismatch for '{}'. Expected pattern for type `{}`.",
                    __var_name_str,
                    stringify!($target_type)
                ))),
            }?
            .try_into()
            .map_err(|e| {
                anyhow!(
                    "Failed to convert '{}' to type `{}`: {:?}",
                    __var_name_str,
                    stringify!($target_type),
                    e
                )
            }),
            None => Err(anyhow!(concat!(
                "Required variable '",
                stringify!($var_name_ident),
                "' not found in any configuration."
            ))),
        }
    }};

    // Arm for variable names passed as string expressions
    ($asts:expr, $var_name_expr:expr, as $target_type:ty, $($pattern:pat_param)|+ => $extraction_block:block) => {{
        let __var_name_str = $var_name_expr;
        let __resolved_opt = match $asts.user.resolve(__var_name_str)? {
            Some(val) => Some(val),
            None => $asts.defaults.resolve(__var_name_str)?,
        };

        match __resolved_opt {
            Some(__resolved_val) => match __resolved_val {
                $($pattern)|+ => {
                    let __extracted_val = $extraction_block;
                    Ok(__extracted_val)
                }
                _ => Err(anyhow!(format!(
                    "Pattern mismatch for '{}'. Expected pattern for type `{}`.",
                    __var_name_str,
                    stringify!($target_type)
                ))),
            }?
            .try_into()
            .map_err(|e| {
                anyhow!(
                    "Failed to convert '{}' to type `{}`: {:?}",
                    __var_name_str,
                    stringify!($target_type),
                    e
                )
            }),
            None => Err(anyhow!(format!(
                "Required variable '{}' not found in any configuration.",
                __var_name_str
            ))),
        }
    }};
}

/// Resolves a length range (e.g., for bounding boxes or distance).
/// It handles `..` and `100mm..2000mm` syntax, converting all values to meters (f32).
macro_rules! resolve_length_range {
    ($ns:expr, $field:expr, $default_from:expr, $default_to:expr) => {{
        let convert_mm_to_meters = |val: &Val| -> anyhow::Result<f32> {
            match val {
                Val::UnitedVal(uv) if uv.unit == Unit::WayMillimeter => {
                    Ok(Length::new::<millimeter>(uv.val as f32).get::<meter>())
                },
                _ => Err(anyhow!("Expected a united value with 'mm' unit for length range '{}'", $field))
            }
        };

        // Use the now-correct resolve_var! to fetch the Rhs value
        let rhs_val = resolve_var!($ns, $field, as &Rhs, r => {r})?;
        match rhs_val {
            Rhs::EmptyRange => Ok(($default_from, $default_to)),
            Rhs::Range { from, to } => {
                let from_val = from.as_ref().map_or(Ok($default_from), |v| convert_mm_to_meters(v))?;
                let to_val = to.as_ref().map_or(Ok($default_to), |v| convert_mm_to_meters(v))?;
                Ok((from_val, to_val))
            }
            _ => Err(anyhow!("Expected a range expression for field '{}'", $field))
        }
    }};
}

/// Resolves an angle range (e.g., for FOV).
/// It assumes input is in degrees and converts it to radians (f32).
macro_rules! resolve_angle_range {
    ($ns:expr, $field:expr, $default_from_deg:expr, $default_to_deg:expr) => {{
        let rhs_val = resolve_var!($ns, $field, as &Rhs, r => {r})?;
        match rhs_val {
            Rhs::EmptyRange => {
                Ok((Angle::new::<degree>($default_from_deg).get::<radian>(), Angle::new::<degree>($default_to_deg).get::<radian>()))
            },
            Rhs::Range { from, to } => {
                let from_deg = match from {
                    Some(Val::NumVal(NumVal::Floating(f))) => f as f32,
                    Some(Val::NumVal(NumVal::Integer(i))) => i as f32,
                    None => $default_from_deg,
                    _ => return Err(anyhow!(format!("Expected numeric value for 'from' in angle range '{}'", $field))),
                };
                 let to_deg = match to {
                    Some(Val::NumVal(NumVal::Floating(f))) => f as f32,
                    Some(Val::NumVal(NumVal::Integer(i))) => i as f32,
                    None => $default_to_deg,
                    _ => return Err(anyhow!(format!("Expected numeric value for 'to' in angle range '{}'", $field))),
                };
                Ok((Angle::new::<degree>(from_deg).get::<radian>(), Angle::new::<degree>(to_deg).get::<radian>()))
            },
            _ => Err(anyhow!(format!("Expected a range expression for field '{}'", $field)))
        }
    }};
}

/// Resolves a unitless float range (e.g., for intensity).
macro_rules! resolve_float_range {
    ($ns:expr, $field:expr, $default_from:expr, $default_to:expr) => {{
        let rhs_val = resolve_var!($ns, $field, as &Rhs, r => {r})?;
        match rhs_val {
            Rhs::EmptyRange => Ok(($default_from, $default_to)),
            Rhs::Range { from, to } => {
                let from_val = match from {
                    Some(Val::NumVal(NumVal::Floating(f))) => f as f32,
                    Some(Val::NumVal(NumVal::Integer(i))) => i as f32,
                    None => $default_from,
                    _ => return Err(anyhow!(format!("Expected numeric value for 'from' in float range '{}'", $field))),
                };
                let to_val = match to {
                    Some(Val::NumVal(NumVal::Floating(f))) => f as f32,
                    Some(Val::NumVal(NumVal::Integer(i))) => i as f32,
                    None => $default_to,
                    _ => return Err(anyhow!(format!("Expected numeric value for 'to' in float range '{}'", $field))),
                };
                Ok((from_val, to_val))
            },
            _ => Err(anyhow!(format!("Expected a range expression for field '{}'", $field)))
        }
    }};
}

//================================================================================
//  Configuration Structs and Parsers
//================================================================================

/// A struct that holds the user and default configuration variable maps.
pub struct Asts {
    user: VariableHistory,
    defaults: VariableHistory,
}

impl Asts {
    pub fn new(defaults: VariableHistory, user: VariableHistory) -> Self {
        Self { user, defaults }
    }

    /// Creates a new `Asts` struct filtered by a namespace path.
    fn filter_ns(&self, path: &[&str]) -> Asts {
        Asts::new(self.defaults.filter_ns(path), self.user.filter_ns(path))
    }
}

/// Constructs an `LioConfig` by layering a user configuration over a default one.
#[allow(unreachable_patterns, clippy::unnecessary_fallible_conversions)]
pub fn ast_to_config(defaults: Evaluated, user: Evaluated) -> anyhow::Result<LioConfig> {
    let asts = Asts::new(defaults.vars, user.vars);

    // --- Helper Functions for Enum Parsing ---
    fn reliablility_parse(src: &str) -> anyhow::Result<Qos> {
        Ok(match src.to_lowercase().as_str() {
            "besteffort" => Qos::Sensor(Reliability::BestEffort),
            "reliable" => Qos::Sensor(Reliability::Reliable),
            "systemdefault" => Qos::SystemDefault,
            _ => return Err(anyhow!("Unknown reliability setting.")),
        })
    }

    fn lidar_vendor_parse(src: &str) -> anyhow::Result<LidarVendor> {
        Ok(match src.to_lowercase().as_str() {
            "velodyne" => LidarVendor::Velodyne,
            "ouster" => LidarVendor::Ouster,
            "robosense" => LidarVendor::Robosense,
            "skip" => LidarVendor::Skip,
            _ => return Err(anyhow!("Unknown Lidar vendor.")),
        })
    }

    fn ahrs_method_parse(s: &str) -> anyhow::Result<MadgwickSetting> {
        match s.to_lowercase().as_str() {
            "createorientationonly" => Ok(MadgwickSetting::CreateOrientationOnly),
            "filteraccelerationandcreateorientation" => {
                Ok(MadgwickSetting::FilterAccelerationAndCreateOrientation)
            }
            "filteraccelerationonly" => Ok(MadgwickSetting::FilterAccelerationOnly),
            "disable" => Ok(MadgwickSetting::Disable),
            _ => Err(anyhow!("Unknown AHRS filtering method: {}", s)),
        }
    }

    //================================================================================
    //  1. Parse all values into local variables
    //================================================================================

    // --- Registration Configuration ---
    let (k_neighbors, neighbor_max_range, map_resolution, map_capacity, dist_update_sensitivity) = {
        let registration = asts.filter_ns(&["registration"]);
        let k_neighbors = resolve_var!(registration, "k_neighborhood", as usize, Rhs::Val(Val::NumVal(NumVal::Integer(i))) => {i})?;
        let neighbor_max_range = resolve_var!(registration, "neighbors_max_range", as f32, Rhs::Val(Val::UnitedVal(UnitVal { val: i, unit: Unit::WayMillimeter })) => { Length::new::<millimeter>(i as f32).get::<meter>() })?;

        let map = registration.filter_ns(&["map"]);
        let map_resolution = resolve_var!(map, "resolution", as f32, Rhs::Val(Val::UnitedVal(UnitVal { val: i, unit: Unit::WayMillimeter })) => { Length::new::<millimeter>(i as f32).get::<meter>() })?;
        let map_capacity = resolve_var!(map, "capacity", as usize, Rhs::Val(Val::NumVal(NumVal::Integer(i))) => {i})?;
        let dist_update_sensitivity = resolve_var!(map, "distribution_update_sensitivity", as f32, Rhs::Val(Val::NumVal(NumVal::Floating(i))) => {i as f32})?;

        (
            k_neighbors,
            neighbor_max_range,
            map_resolution,
            map_capacity,
            dist_update_sensitivity,
        )
    };

    // --- Lidar Configuration ---
    let (cloud_src_topic, cloud_qos, lidar_vendor, lidar_odom_cov) = {
        let lidar = asts.filter_ns(&["lidar"]);
        let cloud_src_topic = resolve_var!(lidar, "topic", as String, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {s})?;
        let cloud_qos = resolve_var!(lidar, "qos.reliability", as Qos, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {reliablility_parse(&s)?})?;
        let lidar_vendor = resolve_var!(lidar, "vendor", as LidarVendor, Rhs::Val(Val::StringVal(s)) => {lidar_vendor_parse(&s)?})?;
        let lidar_odom_cov = resolve_var!(lidar, "covariance", as f32, Rhs::Val(Val::NumVal(NumVal::Floating(f))) => {f})?;
        (cloud_src_topic, cloud_qos, lidar_vendor, lidar_odom_cov)
    };

    // --- IMU Configuration ---
    let (imu_params, imu_prelio_params, imu_node_params) = {
        let imu = asts.filter_ns(&["imu"]);
        let imu_src_topic = resolve_var!(imu, "topic", as String, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {s})?;
        let imu_qos = resolve_var!(imu, "qos.reliability", as Qos, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {reliablility_parse(&s)?})?;
        let imu_predictor_init_time_sec = resolve_var!(imu, "initial_setup", as f32, Rhs::Val(Val::UnitedVal(UnitVal { val: i, unit: Unit::TimeMilliseconds })) => { Time::new::<millisecond>(i as f32).get::<second>() })?;

        let transform = imu.filter_ns(&["from_lidar_transform"]);
        let imu_lidar_tf_override =
            resolve_var!(transform, "use_tf_tree", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;
        let initial_translation_lidar_to_imu = resolve_var!(transform, "translation", as Vector3<f64>, Rhs::Array(v) => {
            let (Some(x_rhs), Some(y_rhs), Some(z_rhs)) = (v.get(0), v.get(1), v.get(2)) else { return Err(anyhow!("Expected three vector components.")) };
            let (Rhs::Val(Val::UnitedVal(UnitVal{ val: x, .. })), Rhs::Val(Val::UnitedVal(UnitVal{ val: y, .. })), Rhs::Val(Val::UnitedVal(UnitVal{ val: z, .. }))) = (&**x_rhs, &**y_rhs, &**z_rhs) else { return Err(anyhow!("All components must be values in millimeters.")) };
            let x_m = Length::new::<millimeter>(*x as f32).get::<meter>();
            let y_m = Length::new::<millimeter>(*y as f32).get::<meter>();
            let z_m = Length::new::<millimeter>(*z as f32).get::<meter>();
            Vector3::new(x_m as f64, y_m as f64, z_m as f64)
        })?;
        let initial_rotation_lidar_to_imu = resolve_var!(transform, "rotation", as Matrix3<f64>, Rhs::Array(rows) => {
            if rows.len() != 3 { return Err(anyhow!("Rotation matrix must have 3 rows.")) }
            let mut elements = Vec::with_capacity(9);
            for row_rhs in rows {
                let Rhs::Array(cols) = &*row_rhs else { return Err(anyhow!("Expected a row to be an array.")) };
                if cols.len() != 3 { return Err(anyhow!("Matrix row must have 3 columns.")) }
                for col_rhs in cols {
                    let Rhs::Val(Val::NumVal(NumVal::Floating(val))) = &**col_rhs else { return Err(anyhow!("Matrix element must be a float.")) };
                    elements.push(*val);
                }
            }
            Matrix3::new(elements[0], elements[1], elements[2], elements[3], elements[4], elements[5], elements[6], elements[7], elements[8])
        })?;

        let ahrs_filtering = imu.filter_ns(&["ahrs_filtering"]);
        let madgwick_setting = resolve_var!(ahrs_filtering, "method", as MadgwickSetting, Rhs::Val(Val::StringVal(s)) => {ahrs_method_parse(&s)?})?;
        let gravity = resolve_var!(imu, "gravity", as f64, Rhs::Val(Val::NumVal(NumVal::Floating(f))) => {f})?;

        let cov_gyr_val = resolve_var!(imu, "covariance_gyroscope", as f64, Rhs::Val(Val::NumVal(NumVal::Floating(f))) => {f})?;
        let cov_gyr = Vector3::new(cov_gyr_val, cov_gyr_val, cov_gyr_val);
        let cov_gyr_bias_val = resolve_var!(imu, "covariance_gyroscope_bias", as f64, Rhs::Val(Val::NumVal(NumVal::Floating(f))) => {f})?;
        let cov_gyr_bias = Vector3::new(cov_gyr_bias_val, cov_gyr_bias_val, cov_gyr_bias_val);
        let cov_acc_val = resolve_var!(imu, "covariance_acceleration", as f64, Rhs::Val(Val::NumVal(NumVal::Floating(f))) => {f})?;
        let cov_acc = Vector3::new(cov_acc_val, cov_acc_val, cov_acc_val);
        let cov_acc_bias_val = resolve_var!(imu, "covariance_acceleration_bias", as f64, Rhs::Val(Val::NumVal(NumVal::Floating(f))) => {f})?;
        let cov_acc_bias = Vector3::new(cov_acc_bias_val, cov_acc_bias_val, cov_acc_bias_val);
        let accel_noise = resolve_var!(imu, "acceleration_noise", as f64, Rhs::Val(Val::NumVal(NumVal::Floating(f))) => {f})?;
        let accel_bias_noise = resolve_var!(imu, "acceleration_bias_noise", as f64, Rhs::Val(Val::NumVal(NumVal::Floating(f))) => {f})?;
        let gyro_noise = resolve_var!(imu, "gyroscope_noise", as f64, Rhs::Val(Val::NumVal(NumVal::Floating(f))) => {f})?;
        let gyro_bias_noise = resolve_var!(imu, "gyroscope_bias_noise", as f64, Rhs::Val(Val::NumVal(NumVal::Floating(f))) => {f})?;

        let calib = imu.filter_ns(&["calibration"]);
        let imu_calibrate_gyros =
            resolve_var!(calib, "gyroscope", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;
        let gyro_calibration_samples = resolve_var!(calib, "samples", as usize, Rhs::Val(Val::NumVal(NumVal::Integer(i))) => {i})?;
        let imu_intrinsics_calib_file_path = resolve_var!(calib, "parameter_file", as Option<String>, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => { if s.is_empty() { None } else { Some(s) } })?;

        let imu_params = pelorus_reg::imu::ImuParameters {
            initial_translation_lidar_to_imu,
            initial_rotation_lidar_to_imu,
            cov_gyr,
            cov_gyr_bias,
            cov_acc,
            cov_acc_bias,
        };
        let imu_prelio_params = ImuParameters {
            madgwick_setting,
            gravity,
            accel_noise,
            accel_bias_noise,
            gyro_noise,
            gyro_bias_noise,
            calibration: ImuCalibrationParameters {
                imu_calibrate_gyros,
                gyro_calibration_samples,
                imu_intrinsics_calib_file_path,
                ..Default::default()
            },
            ..Default::default()
        };
        let imu_node_params = (
            imu_src_topic,
            imu_qos,
            imu_predictor_init_time_sec,
            imu_lidar_tf_override,
        );

        (imu_params, imu_prelio_params, imu_node_params)
    };

    // --- Output Configuration ---
    let (
        feedback_fps,
        map_frame,
        lidar_as_map_tf,
        logger_prefix,
        mapping_resolution,
        odom_qos,
        path_qos,
        imu_odom_qos,
        imu_path_qos,
        imu_filtered_qos,
    ) = {
        let output_ns = asts.filter_ns(&["output"]);
        let terminal_ns = output_ns.filter_ns(&["terminal"]);
        let show_ui: bool =
            resolve_var!(terminal_ns, "ui", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;
        let feedback_fps = if show_ui {
            resolve_var!(terminal_ns, "fps", as i32, Rhs::Val(Val::NumVal(NumVal::Integer(i))) => {i})?
        } else {
            0
        };
        let map_frame = resolve_var!(output_ns, "global_frame", as String, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {s})?;
        let lidar_as_map_tf = resolve_var!(output_ns, "lidar_as_map_tf_base", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;
        let logger_prefix = resolve_var!(output_ns, "logger_prefix", as String, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {s})?;
        let mapping_ns = output_ns.filter_ns(&["mapping"]);
        let mapping_resolution = resolve_var!(mapping_ns, "resolution", as f32, Rhs::Val(Val::UnitedVal(UnitVal { val, unit: Unit::WayMillimeter })) => { Length::new::<millimeter>(val as f32).get::<meter>() })?;
        let odom_qos = resolve_var!(output_ns, "lio_odometry_msg_qos", as Qos, Rhs::Val(Val::StringVal(s)) => {reliablility_parse(&s)?})?;
        let path_qos = resolve_var!(output_ns, "lio_path_msg_qos", as Qos, Rhs::Val(Val::StringVal(s)) => {reliablility_parse(&s)?})?;
        let imu_odom_qos = resolve_var!(output_ns, "imu_odometry_msg_qos", as Qos, Rhs::Val(Val::StringVal(s)) => {reliablility_parse(&s)?})?;
        let imu_path_qos = resolve_var!(output_ns, "imu_path_msg_qos", as Qos, Rhs::Val(Val::StringVal(s)) => {reliablility_parse(&s)?})?;
        let imu_filtered_qos = resolve_var!(output_ns, "imu_ahrs_filtered_msg_qos", as Qos, Rhs::Val(Val::StringVal(s)) => {reliablility_parse(&s)?})?;
        (
            feedback_fps,
            map_frame,
            lidar_as_map_tf,
            logger_prefix,
            mapping_resolution,
            odom_qos,
            path_qos,
            imu_odom_qos,
            imu_path_qos,
            imu_filtered_qos,
        )
    };

    // --- Preprocessing Configuration ---
    let (voxel_size, passthrough_filter) = {
        let preproc_ns = asts.filter_ns(&["preprocessing"]);
        let downsample_ns = preproc_ns.filter_ns(&["downsample"]);
        let voxel_size = resolve_var!(downsample_ns, "resolution", as f32, Rhs::Val(Val::UnitedVal(UnitVal { val, unit: Unit::WayMillimeter })) => { Length::new::<millimeter>(val as f32).get::<meter>() })?;

        let filter_ns = preproc_ns.filter_ns(&["filter"]);
        let bbox_ns = filter_ns.filter_ns(&["bounding_box"]);
        let (min_x, max_x) = resolve_length_range!(bbox_ns, "x", f32::MIN, f32::MAX)?;
        let (min_y, max_y) = resolve_length_range!(bbox_ns, "y", f32::MIN, f32::MAX)?;
        let (min_z, max_z) = resolve_length_range!(bbox_ns, "z", f32::MIN, f32::MAX)?;
        let min = Vector3::new(min_x, min_y, min_z);
        let max = Vector3::new(max_x, max_y, max_z);
        let invert_bounding_box =
            resolve_var!(bbox_ns, "invert", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;

        let dist_ns = filter_ns.filter_ns(&["distance"]);
        let (min_dist, max_dist) = resolve_length_range!(dist_ns, "range", 0.0, f32::MAX / 2.5)?;
        let invert_distance =
            resolve_var!(dist_ns, "invert", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;

        let hfov_ns = filter_ns.filter_ns(&["horizontal_fov"]);
        let (fov_left, fov_right) = resolve_angle_range!(hfov_ns, "range", -180.0, 180.0)?;
        let enable_horizontal_fov =
            resolve_var!(hfov_ns, "enable", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;
        let invert_fov =
            resolve_var!(hfov_ns, "invert", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;
        let forward = resolve_var!(hfov_ns, "forward", as Vector2<f32>, Rhs::Array(v) => {
            let (Some(x_rhs), Some(y_rhs)) = (v.get(0), v.get(1)) else { return Err(anyhow!("Expected two components for forward vector.")) };
            let (Rhs::Val(Val::NumVal(NumVal::Floating(x))), Rhs::Val(Val::NumVal(NumVal::Floating(y)))) = (&**x_rhs, &**y_rhs) else { return Err(anyhow!("Forward vector components must be floats.")) };
            Vector2::new(*x as f32, *y as f32)
        })?;

        let poly_ns = filter_ns.filter_ns(&["polygon_2d"]);
        let coords_vec: Vec<Point2<f32>> = resolve_var!(poly_ns, "coords", as Vec<Point2<f32>>, Rhs::Array(points) => {
            let mut coords = Vec::new();
            for point_rhs in points {
                let Rhs::Array(point_coords) = &*point_rhs else { return Err(anyhow!("Expected polygon vertex to be an array [x, y].")) };
                let (Some(x_rhs), Some(y_rhs)) = (point_coords.get(0), point_coords.get(1)) else { return Err(anyhow!("Expected two components [x, y] for polygon vertex.")) };
                let (Rhs::Val(Val::UnitedVal(UnitVal{ val: x, .. })), Rhs::Val(Val::UnitedVal(UnitVal{ val: y, .. }))) = (&**x_rhs, &**y_rhs) else { return Err(anyhow!("Polygon vertex components must be united values (e.g., in meters).")) };
                coords.push(Point2::new(*x as f32, *y as f32));
            }
            coords
        })?;
        let polygon = if !coords_vec.is_empty() {
            Some(coords_vec)
        } else {
            None
        };
        let invert_polygon =
            resolve_var!(poly_ns, "invert", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;

        let intensity_ns = filter_ns.filter_ns(&["intensity"]);
        let (min_intensity, max_intensity) =
            resolve_float_range!(intensity_ns, "range", f32::MIN, f32::MAX)?;
        let invert_intensity =
            resolve_var!(intensity_ns, "invert", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;

        let passthrough_filter = PassthroughFilterParameters {
            min,
            max,
            invert_bounding_box,
            min_dist,
            max_dist,
            invert_distance,
            fov_left,
            fov_right,
            forward,
            enable_horizontal_fov,
            polygon,
            invert_polygon,
            invert_fov,
            min_intensity,
            max_intensity,
            invert_intensity,
            ..Default::default()
        };
        (voxel_size, passthrough_filter)
    };

    //================================================================================
    //  2. Construct the final LioConfig struct
    //================================================================================
    let final_config = LioConfig {
        k_neighbors,
        neighbor_max_range,
        map_resolution,
        map_capacity,
        dist_update_sensitivity,
        lidar_odom_cov,
        feedback_fps,
        lidar_as_map_tf,
        mapping_resolution,
        odom_qos,
        path_qos,
        imu_odom_qos,
        imu_path_qos,
        imu_params,
        prelio_node_params: LifisNodeParameters {
            cloud_src_topic,
            cloud_qos,
            imu_src_topic: imu_node_params.0,
            imu_qos: imu_node_params.1,
            imu_predictor_init_time_sec: imu_node_params.2,
            logger_prefix,
            imu_filtered_qos,
            ..Default::default()
        },
        prelio_params: LifisParameters {
            lidar_vendor,
            imu_lidar_tf_override: imu_node_params.3,
            map_frame,
            imu: imu_prelio_params,
            downsample: DownsampleParameters { voxel_size },
            passthrough_filter,
            ..Default::default()
        },
        // TODO currently hardcoded
        registered_cloud_qos: Qos::Sensor(Reliability::Reliable),
        enable_live_extrinsic_calibration: false,
        ekf_max_iterations: 4,
    };

    Ok(final_config)
}

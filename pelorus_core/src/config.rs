use anyhow::anyhow;
use nalgebra::{Matrix3, Point2, Vector2, Vector3};
use ratslang::uom::si::angle::{degree, radian};
use ratslang::uom::si::f32::Angle;
use ratslang::uom::si::length::meter;
use ratslang::uom::si::time::second;
use ratslang::{
    Evaluated, NumVal, Rhs, Unit, UnitVal, Val, VariableHistory, resolve_float_range,
    resolve_length_range_meters_float, resolve_var,
};

use crate::{
    CoordinateSystem, DownsampleParameters, ImuCalibrationParameters, ImuConfigParameter,
    ImuDriverConfig, ImuDriverType, ImuParameters, KinematicModel, LidarDriverConfig,
    LidarDriverType, LidarVendor, LioConfig, MadgwickSetting, OdometryRate, OusterConfig,
    OusterTimestampMode, PassthroughFilterParameters, PointCloudMode, SyncNodeParameters,
    SyncParameters, PreprocessingBackend, Qos, Reliability, RerunConfig, RerunMapConfig,
    SensorDriverConfig, VelodyneConfig, VelodyneModel, VelodyneTimestampMode, XsensConfig,
    XsensTimestampMode, adaptive_execution::ExecutionPolicy,
};

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

/// A parsed `KEY=VALUE` override.
#[derive(Debug, Clone)]
pub struct ConfigOverride {
    /// Fully qualified variable name, e.g. `output.odometry_rate`.
    pub key: String,
    statement: String,
}

/// Parses command-line `KEY=VALUE` pairs as Ratslang assignments.
///
/// The value is passed through untouched, so it carries its own Ratslang
/// syntax -- units (`0.5m`, `2s`), booleans, enums and quoted strings all work
/// as they do in a configuration file.
pub fn parse_overrides(overrides: &[String]) -> anyhow::Result<Vec<ConfigOverride>> {
    overrides
        .iter()
        .map(|raw| {
            let (key, value) = raw
                .split_once('=')
                .ok_or_else(|| anyhow!("override '{raw}' is not KEY=VALUE"))?;
            let key = key.trim();
            let value = value.trim();
            if key.is_empty() {
                return Err(anyhow!("override '{raw}' has an empty key"));
            }
            if value.is_empty() {
                return Err(anyhow!("override '{raw}' has an empty value"));
            }
            let statement = format!("{key} = {value}");
            // Compile separately so syntax errors identify the override.
            let mut syntax_check = ratslang::compile_code(&statement)
                .map_err(|error| anyhow!("override '{raw}' is not valid Ratslang: {error}"))?;
            // Only checks syntax here. The value is used later, so silence the drop warning.
            syntax_check.vars = syntax_check.vars.with_drop_warning(false);
            Ok(ConfigOverride {
                key: key.to_owned(),
                statement,
            })
        })
        .collect()
}

/// Builds a user configuration from an optional file and command-line overrides.
///
/// Ratslang assigns sequentially and the last assignment wins, so the
/// overrides are appended after the file's own text and compiled as one unit.
/// Ratslang consults prior variable state before evaluating statements, so the
/// overrides must be appended to the source to take precedence.
///
/// Relative includes resolve from the configuration file's directory.
pub fn user_config_with_overrides(
    path: Option<&std::path::Path>,
    overrides: &[ConfigOverride],
) -> anyhow::Result<Evaluated> {
    let source = match path {
        Some(path) => std::fs::read_to_string(path)
            .map_err(|error| anyhow!("failed to read '{}': {error}", path.display()))?,
        None => String::new(),
    };
    let parent = path
        .and_then(|path| path.parent())
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(|parent| parent.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    layer_overrides(&source, &parent, overrides).map_err(|error| match path {
        Some(path) => anyhow!("failed to compile '{}': {error}", path.display()),
        None => anyhow!("failed to compile the command-line overrides: {error}"),
    })
}

/// Builds a user layer from source text and rejects unknown overrides.
pub fn user_config_source_with_checked_overrides(
    defaults_source: &str,
    source: &str,
    parent: &std::path::Path,
    overrides: &[ConfigOverride],
) -> anyhow::Result<Evaluated> {
    let user = layer_overrides(source, parent, overrides)?;
    if overrides.is_empty() {
        return Ok(user);
    }
    let mut defaults = ratslang::compile_code(defaults_source)
        .map_err(|error| anyhow!("failed to compile embedded default config: {error}"))?;
    defaults.vars = defaults.vars.with_drop_warning(false);
    let mut probe = layer_overrides(source, parent, overrides)?;
    probe.vars = probe.vars.with_drop_warning(false);
    let (_, (_, evaluated)) = ast_to_config_with_user_usage(defaults, probe)?;
    reject_unknown_overrides(overrides, &evaluated)?;
    Ok(user)
}

/// Applies overrides to source text. Relative includes resolve from `parent`.
pub fn layer_overrides(
    source: &str,
    parent: &std::path::Path,
    overrides: &[ConfigOverride],
) -> anyhow::Result<Evaluated> {
    let mut source = source.to_owned();
    for entry in overrides {
        // The leading newline handles Ratslang's empty-source token edge case.
        source.push('\n');
        source.push_str(&entry.statement);
    }
    source.push('\n');
    ratslang::compile_code_with_state(&source, parent, None, std::io::stderr(), true)
        .map_err(|error| anyhow!("{error}"))
}

/// Rejects overrides for variables the configuration never reads.
///
/// Unknown variables in configuration files remain warnings for compatibility.
/// Command-line overrides are strict because they apply to a single run.
pub fn reject_unknown_overrides(
    overrides: &[ConfigOverride],
    evaluated_variables: &[String],
) -> anyhow::Result<()> {
    let unknown = overrides
        .iter()
        .filter(|entry| !evaluated_variables.contains(&entry.key))
        .map(|entry| format!("'{}'", entry.key))
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        return Err(anyhow!(
            "override{} {} not recognized; run 'config generate full' to see every setting",
            if unknown.len() == 1 { "" } else { "s" },
            unknown.join(", ")
        ));
    }
    Ok(())
}

/// Builds the user layer and rejects any override that
/// names a setting the configuration never reads.
///
/// Validation requires a second parse and is skipped when no overrides exist.
pub fn user_config_with_checked_overrides(
    defaults_source: &str,
    path: Option<&std::path::Path>,
    overrides: &[ConfigOverride],
) -> anyhow::Result<Evaluated> {
    let source = match path {
        Some(path) => std::fs::read_to_string(path)
            .map_err(|error| anyhow!("failed to read '{}': {error}", path.display()))?,
        None => String::new(),
    };
    let parent = path
        .and_then(std::path::Path::parent)
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    user_config_source_with_checked_overrides(defaults_source, &source, parent, overrides).map_err(
        |error| match path {
            Some(path) => anyhow!("failed to compile '{}': {error}", path.display()),
            None => anyhow!("failed to compile the command-line overrides: {error}"),
        },
    )
}

/// Validates configuration and returns usage statistics for the defaults.
/// This is primarily intended for build-time validation.
///
/// Returns (config, (assigned_vars, evaluated_vars))
#[allow(unreachable_patterns, clippy::unnecessary_fallible_conversions)]
/// The four process-noise variances implied by an IMU calibration file.
struct DerivedNoise {
    gyr: f64,
    acc: f64,
    gyr_bias: f64,
    acc_bias: f64,
}

/// Turn the stochastic half of a Kalibr calibration into filter covariances.
///
/// The filter propagates `P += F_w Q F_w^T` with `F_w = f_w * dt`
/// (`pelorus_reg::eqf::predict`), so every entry of `Q` picks up a `dt^2`. For
/// the covariance of a white-noise-driven state to grow as `sigma^2 * t`, `Q`
/// `Q` therefore holds *discrete-time* variances converted from the
/// continuous-time densities in the file:
///
/// ```text
/// Q = (density * inflation)^2 * update_rate
/// ```
///
/// The inflation is not derivable and has to be chosen. Allan variance is
/// measured on a completely still sensor at constant temperature; a moving
/// robot has vibration, thermal drift and timing jitter on top, and those
/// dominate. Feeding the bench figures in raw makes the filter trust the IMU
/// far more than it deserves.
fn derive_noise_covariances(path: &str, inflation: f64) -> anyhow::Result<DerivedNoise> {
    anyhow::ensure!(
        inflation > 0.0 && inflation.is_finite(),
        "imu.calibration.noise_inflation must be a positive, finite value"
    );

    let intrinsics = imu_calib::ImuIntrinsics::from_kalibr_yaml(path)
        .map_err(|error| anyhow::anyhow!("Failed to read the IMU calibration '{path}': {error}"))?;
    let noise = intrinsics.noise;

    // Zero densities mean the calibration file has no noise run.
    anyhow::ensure!(
        noise.accel_noise_density > 0.0
            && noise.gyro_noise_density > 0.0
            && noise.accel_random_walk > 0.0
            && noise.gyro_random_walk > 0.0,
        "'{path}' carries no noise parameters, so imu.calibration.noise_from_file has \
         nothing to derive from. Record a few hours of stationary data, run \
         `dedrunk noise` on it, and `dedrunk merge` the result with the intrinsics."
    );
    anyhow::ensure!(
        noise.update_rate > 0.0,
        "'{path}' has a non-positive update_rate, which the covariances scale with"
    );

    let variance = |density: f64| (density * inflation).powi(2) * noise.update_rate;

    Ok(DerivedNoise {
        gyr: variance(noise.gyro_noise_density),
        acc: variance(noise.accel_noise_density),
        gyr_bias: variance(noise.gyro_random_walk),
        acc_bias: variance(noise.accel_random_walk),
    })
}

pub fn ast_to_config_with_usage(
    defaults: Evaluated,
    user: Evaluated,
) -> anyhow::Result<(LioConfig, (Vec<String>, Vec<String>))> {
    let config = ast_to_config_inner(defaults, user, true, false)?;
    Ok((config.0, config.1))
}

#[allow(unreachable_patterns, clippy::unnecessary_fallible_conversions)]
pub fn ast_to_config_with_user_usage(
    mut defaults: Evaluated,
    user: Evaluated,
) -> anyhow::Result<(LioConfig, (Vec<String>, Vec<String>))> {
    // Defaults get shadowed by user values on purpose. Silence their drop warning
    // but keep usage diagnostics for the user config, which is what this helper checks.
    defaults.vars = defaults.vars.with_drop_warning(false);
    let config = ast_to_config_inner(defaults, user, false, true)?;
    Ok((config.0, config.2))
}

#[cfg(test)]
mod override_tests {
    use super::*;

    const DEFAULTS: &str = include_str!("../default.rl");

    #[test]
    fn an_override_beats_the_same_setting_in_the_config_file() {
        let directory =
            std::env::temp_dir().join(format!("pelorus-override-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("user.rl");
        std::fs::write(&path, "output.odometry_rate = ImuAndLidar\n").unwrap();

        let overrides = parse_overrides(&["output.odometry_rate=Lidar".to_owned()]).unwrap();
        let user = user_config_with_overrides(Some(&path), &overrides).unwrap();
        let defaults = ratslang::compile_code(DEFAULTS).unwrap();
        let config = ast_to_config(defaults, user).unwrap();

        assert_eq!(config.odometry_rate, crate::OdometryRate::Lidar);
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn rerun_server_address_can_be_overridden() {
        let overrides =
            parse_overrides(&["output.rerun.server_addr=\"172.22.129.240:9876\"".to_owned()])
                .unwrap();
        let user = user_config_with_checked_overrides(DEFAULTS, None, &overrides).unwrap();
        let defaults = ratslang::compile_code(DEFAULTS).unwrap();
        let config = ast_to_config(defaults, user).unwrap();

        assert_eq!(config.rerun.server_addr, "172.22.129.240:9876");
    }

    #[test]
    fn rerun_map_settings_can_be_overridden() {
        let overrides = parse_overrides(&[
            "output.rerun.map.enabled=true".to_owned(),
            "output.rerun.map.publish_interval=2s".to_owned(),
            "output.rerun.map.visual_scale=4.5".to_owned(),
        ])
        .unwrap();
        let user = user_config_with_checked_overrides(DEFAULTS, None, &overrides).unwrap();
        let defaults = ratslang::compile_code(DEFAULTS).unwrap();
        let config = ast_to_config(defaults, user).unwrap();

        assert!(config.rerun.map.enabled);
        assert_eq!(config.rerun.map.publish_interval_sec, 2.0);
        assert_eq!(config.rerun.map.visual_scale, 4.5);
    }

    /// Writes the merged calibration format produced by `dedrunk merge`.
    fn write_calibration(directory: &std::path::Path, with_noise: bool) -> std::path::PathBuf {
        let (accel_density, accel_walk, gyro_density, gyro_walk) = if with_noise {
            (0.002, 0.0004, 0.0003, 2.0e-5)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        let mut contents = String::from("imu0:\n");
        contents.push_str("  model: scale-misalignment\n");
        contents.push_str("  update_rate: 200.0\n");
        contents.push_str(&format!("  accelerometer_noise_density: {accel_density}\n"));
        contents.push_str(&format!("  accelerometer_random_walk: {accel_walk}\n"));
        contents.push_str(&format!("  gyroscope_noise_density: {gyro_density}\n"));
        contents.push_str(&format!("  gyroscope_random_walk: {gyro_walk}\n"));
        contents.push_str("  accelerometers:\n");
        contents.push_str("    M:\n");
        contents.push_str("    - [1.0, 0.0, 0.0]\n");
        contents.push_str("    - [0.0, 1.0, 0.0]\n");
        contents.push_str("    - [0.0, 0.0, 1.0]\n");

        let path = directory.join("imu.yaml");
        std::fs::write(&path, contents).unwrap();
        path
    }

    fn config_with(source: &str) -> crate::LioConfig {
        let user = ratslang::compile_code(source).unwrap();
        let defaults = ratslang::compile_code(DEFAULTS).unwrap();
        ast_to_config(defaults, user).unwrap()
    }

    #[test]
    fn the_imu_time_offset_is_read_in_seconds() {
        assert_eq!(
            config_with("").sync_params.imu.calibration.time_offset,
            0.0
        );

        let shifted = config_with("imu.calibration.time_offset = -5.9ms\n");
        assert!(
            (shifted.sync_params.imu.calibration.time_offset + 0.0059).abs() < 1e-12,
            "{}",
            shifted.sync_params.imu.calibration.time_offset
        );

        let seconds = config_with("imu.calibration.time_offset = 0.25s\n");
        assert!((seconds.sync_params.imu.calibration.time_offset - 0.25).abs() < 1e-12);
    }

    #[test]
    fn lidar_only_forces_the_extrinsic_to_identity() {
        let source = "imu.from_lidar_transform.translation = [1.0m, 2.0m, 3.0m]\n\
                      imu.from_lidar_transform.use_tf_tree = true\n";
        let with_imu = config_with(source);
        assert!(with_imu.sync_params.imu_lidar_tf_override);
        assert_eq!(
            with_imu.imu_params.initial_translation_lidar_to_imu,
            Vector3::new(1.0, 2.0, 3.0)
        );

        let lidar_only = config_with(&format!("{source}imu.lidar_only = true\n"));
        assert!(lidar_only.imu_params.lidar_only);
        assert!(!lidar_only.sync_params.imu_lidar_tf_override);
        assert_eq!(
            lidar_only.imu_params.initial_translation_lidar_to_imu,
            Vector3::zeros()
        );
        assert_eq!(
            lidar_only.imu_params.initial_rotation_lidar_to_imu,
            Matrix3::identity()
        );
    }

    #[test]
    fn the_extrinsic_is_held_fixed_by_default() {
        assert!(!config_with("").imu_params.estimate_extrinsic_online);
        assert!(
            config_with("imu.from_lidar_transform.estimate_online = true\n")
                .imu_params
                .estimate_extrinsic_online
        );
    }

    #[test]
    fn the_tuned_covariances_are_kept_unless_asked_otherwise() {
        let config = config_with("");
        assert_eq!(config.imu_params.cov_gyr[0], 0.1);
        assert_eq!(config.imu_params.cov_acc[0], 0.1);
        assert_eq!(config.imu_params.cov_gyr_bias[0], 0.0001);
        assert_eq!(config.imu_params.cov_acc_bias[0], 0.0001);
    }

    /// `Q` is sandwiched with `dt^2` in the propagation, so it holds discrete
    /// variances: `(density * inflation)^2 * rate`.
    #[test]
    fn covariances_can_be_derived_from_the_calibration_file() {
        let directory = std::env::temp_dir().join(format!("pelorus-noise-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = write_calibration(&directory, true);

        let config = config_with(&format!(
            "imu.calibration.parameter_file = \"{}\"\n\
             imu.calibration.noise_from_file = true\n\
             imu.calibration.noise_inflation = 1.0\n",
            path.display()
        ));

        let close = |actual: f64, expected: f64| {
            assert!(
                (actual / expected - 1.0).abs() < 1e-9,
                "{actual:e} vs expected {expected:e}"
            );
        };
        // density^2 * update_rate, with the rate 200 Hz.
        close(config.imu_params.cov_gyr[0], 0.0003_f64.powi(2) * 200.0);
        close(config.imu_params.cov_acc[0], 0.002_f64.powi(2) * 200.0);
        close(
            config.imu_params.cov_gyr_bias[0],
            2.0e-5_f64.powi(2) * 200.0,
        );
        close(
            config.imu_params.cov_acc_bias[0],
            0.0004_f64.powi(2) * 200.0,
        );

        assert!(config.imu_params.cov_gyr[0] < 0.1);

        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn the_inflation_applies_before_squaring() {
        let directory =
            std::env::temp_dir().join(format!("pelorus-inflation-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = write_calibration(&directory, true);

        let config = config_with(&format!(
            "imu.calibration.parameter_file = \"{}\"\n\
             imu.calibration.noise_from_file = true\n\
             imu.calibration.noise_inflation = 10.0\n",
            path.display()
        ));

        let expected = (0.0003_f64 * 10.0).powi(2) * 200.0;
        assert!(
            (config.imu_params.cov_gyr[0] / expected - 1.0).abs() < 1e-9,
            "{:e} vs expected {expected:e}",
            config.imu_params.cov_gyr[0]
        );
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn a_calibration_without_noise_parameters_is_refused() {
        let directory =
            std::env::temp_dir().join(format!("pelorus-nonoise-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = write_calibration(&directory, false);

        let user = ratslang::compile_code(&format!(
            "imu.calibration.parameter_file = \"{}\"\n\
             imu.calibration.noise_from_file = true\n",
            path.display()
        ))
        .unwrap();
        let defaults = ratslang::compile_code(DEFAULTS).unwrap();
        let error = ast_to_config(defaults, user).unwrap_err().to_string();
        assert!(error.contains("dedrunk noise"), "{error}");

        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn deriving_without_a_calibration_file_is_refused() {
        let user = ratslang::compile_code("imu.calibration.noise_from_file = true\n").unwrap();
        let defaults = ratslang::compile_code(DEFAULTS).unwrap();
        let error = ast_to_config(defaults, user).unwrap_err().to_string();
        assert!(error.contains("parameter_file"), "{error}");
    }

    #[test]
    fn overrides_apply_without_any_config_file() {
        let overrides = parse_overrides(&["output.odometry_rate=ImuAndLidar".to_owned()]).unwrap();
        let user = user_config_with_overrides(None, &overrides).unwrap();
        let defaults = ratslang::compile_code(DEFAULTS).unwrap();
        let config = ast_to_config(defaults, user).unwrap();

        assert_eq!(config.odometry_rate, crate::OdometryRate::ImuAndLidar);
    }

    #[test]
    fn the_last_of_several_overrides_for_one_key_wins() {
        let overrides = parse_overrides(&[
            "output.odometry_rate=ImuAndLidar".to_owned(),
            "output.odometry_rate=Lidar".to_owned(),
        ])
        .unwrap();
        let user = user_config_with_overrides(None, &overrides).unwrap();
        let defaults = ratslang::compile_code(DEFAULTS).unwrap();
        let config = ast_to_config(defaults, user).unwrap();

        assert_eq!(
            config.odometry_rate,
            crate::OdometryRate::Lidar,
            "the rightmost -o must win"
        );
    }

    #[test]
    fn a_syntactically_invalid_override_names_itself_not_the_config_file() {
        let error =
            parse_overrides(&["output.odometry_rate=Lidar,imu.gyro_bias_std=0.1".to_owned()])
                .unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("output.odometry_rate=Lidar,imu.gyro_bias_std=0.1"),
            "the error must quote the offending override: {message}"
        );
        assert!(
            !message.contains("default.rl"),
            "the error must not blame a config file: {message}"
        );
    }

    #[test]
    fn independent_overrides_all_apply_together() {
        let overrides = parse_overrides(&[
            "output.odometry_rate=ImuAndLidar".to_owned(),
            "registration.max_iter=23".to_owned(),
        ])
        .unwrap();
        let user = user_config_with_overrides(None, &overrides).unwrap();
        let defaults = ratslang::compile_code(DEFAULTS).unwrap();
        let config = ast_to_config(defaults, user).unwrap();

        assert_eq!(config.odometry_rate, crate::OdometryRate::ImuAndLidar);
        assert_eq!(config.ekf_max_iterations, 23);
    }

    #[test]
    fn malformed_overrides_are_rejected_before_anything_runs() {
        for bad in ["output.odometry_rate", "=Lidar", "output.odometry_rate="] {
            assert!(
                parse_overrides(&[bad.to_owned()]).is_err(),
                "'{bad}' must be rejected"
            );
        }
    }

    #[test]
    fn an_override_naming_an_unknown_setting_is_an_error() {
        let overrides = parse_overrides(&["output.odometry_rat=Lidar".to_owned()]).unwrap();
        let error = user_config_with_checked_overrides(DEFAULTS, None, &overrides).unwrap_err();
        assert!(
            error.to_string().contains("output.odometry_rat"),
            "the error must name the typo: {error}"
        );

        let good = parse_overrides(&["output.odometry_rate=Lidar".to_owned()]).unwrap();
        assert!(user_config_with_checked_overrides(DEFAULTS, None, &good).is_ok());
    }

    #[test]
    fn no_overrides_leaves_the_ordinary_path_untouched() {
        let user = user_config_with_checked_overrides(DEFAULTS, None, &[]).unwrap();
        let defaults = ratslang::compile_code(DEFAULTS).unwrap();
        assert!(ast_to_config(defaults, user).is_ok());
    }
}

#[cfg(test)]
mod tests {
    use super::ast_to_config;
    use crate::{OdometryRate, OusterTimestampMode, SensorDriverConfig};

    #[test]
    fn default_sensor_driver_config_parses() {
        let defaults = ratslang::compile_code(include_str!("../default.rl")).unwrap();
        let user = ratslang::compile_code("").unwrap();
        let config = ast_to_config(defaults, user).unwrap();
        assert_eq!(
            config.sensor_drivers,
            SensorDriverConfig {
                ouster: crate::OusterConfig {
                    timestamp_mode: OusterTimestampMode::Ptp,
                    ..Default::default()
                },
                ..Default::default()
            }
        );
    }

    #[test]
    fn odometry_rate_defaults_to_lidar() {
        let defaults = ratslang::compile_code(include_str!("../default.rl")).unwrap();
        let user = ratslang::compile_code("").unwrap();
        let config = ast_to_config(defaults, user).unwrap();
        assert_eq!(config.odometry_rate, OdometryRate::Lidar);
    }

    #[test]
    fn imu_and_lidar_odometry_rate_parses() {
        let defaults = ratslang::compile_code(include_str!("../default.rl")).unwrap();
        let user = ratslang::compile_code("output.odometry_rate = ImuAndLidar").unwrap();
        let config = ast_to_config(defaults, user).unwrap();
        assert_eq!(config.odometry_rate, OdometryRate::ImuAndLidar);
    }

    #[test]
    fn rerun_defaults_parse() {
        let defaults = ratslang::compile_code(include_str!("../default.rl")).unwrap();
        let user = ratslang::compile_code("").unwrap();
        let config = ast_to_config(defaults, user).unwrap();

        assert!(!config.rerun.enabled);
        assert_eq!(config.rerun.server_addr, "127.0.0.1:9876");
        assert!(!config.rerun.map.enabled);
        assert_eq!(config.rerun.map.publish_interval_sec, 1.0);
        assert_eq!(config.rerun.map.visual_scale, 3.0);
    }
}

/// Constructs an `LioConfig` by layering a user configuration over a default one.
#[allow(unreachable_patterns, clippy::unnecessary_fallible_conversions)]
pub fn ast_to_config(mut defaults: Evaluated, user: Evaluated) -> anyhow::Result<LioConfig> {
    defaults.vars = defaults.vars.with_drop_warning(false);
    let (config, _default_usage, _user_usage) = ast_to_config_inner(defaults, user, false, true)?;

    Ok(config)
}

fn ast_to_config_inner(
    defaults: Evaluated,
    user: Evaluated,
    capture_default_usage: bool,
    capture_user_usage: bool,
) -> anyhow::Result<(
    LioConfig,
    (Vec<String>, Vec<String>),
    (Vec<String>, Vec<String>),
)> {
    let asts = Asts::new(defaults.vars, user.vars);

    fn reliablility_parse(src: &str) -> anyhow::Result<Qos> {
        Ok(match src.to_lowercase().as_str() {
            "besteffort" => Qos::Sensor(Reliability::BestEffort),
            "reliable" => Qos::Sensor(Reliability::Reliable),
            "systemdefault" => Qos::SystemDefault,
            _ => return Err(anyhow!("Unknown reliability setting.")),
        })
    }

    fn pointcloud_mode_parse(src: &str) -> anyhow::Result<PointCloudMode> {
        match src.to_lowercase().as_str() {
            "compressed" => Ok(PointCloudMode::Compressed),
            "pointcloud2" => Ok(PointCloudMode::PointCloud2),
            _ => Err(anyhow!(
                "Unknown point-cloud mode '{src}'; expected Compressed or PointCloud2."
            )),
        }
    }

    fn lidar_vendor_parse(src: &str) -> anyhow::Result<LidarVendor> {
        Ok(match src.to_lowercase().as_str() {
            "velodyne" => LidarVendor::Velodyne,
            "ouster" => LidarVendor::Ouster,
            "livox" => LidarVendor::Livox,
            "livoxmid360" => LidarVendor::LivoxMid360,
            "robosense" => LidarVendor::Robosense,
            "hesai" => LidarVendor::Hesai,
            "blickfeld" => LidarVendor::Blickfeld,
            "sickmultiscan" => LidarVendor::SickMultiscan,
            "notimestamp" | "no_timestamp" | "genericnotimestamp" | "generic_no_timestamp" => {
                LidarVendor::NoTimestamp
            }
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

    fn preprocessing_backend_parse(s: &str) -> anyhow::Result<PreprocessingBackend> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(PreprocessingBackend::Cpu),
            "backend" => Ok(PreprocessingBackend::Backend),
            _ => Err(anyhow!(
                "Unknown preprocessing backend: {}. Expected 'Backend' or 'Cpu'.",
                s
            )),
        }
    }

    fn execution_policy_parse(s: &str) -> anyhow::Result<ExecutionPolicy> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(ExecutionPolicy::Auto),
            "singlecore" | "single_core" | "single-core" => Ok(ExecutionPolicy::SingleCore),
            "rayon" => Ok(ExecutionPolicy::Rayon),
            _ => Err(anyhow!(
                "Unknown parallelism execution mode: {}. Expected 'Auto', 'SingleCore', or 'Rayon'.",
                s
            )),
        }
    }

    fn kinematic_model_parse(
        s: &str,
        ackermann_ns: Option<&Asts>,
    ) -> anyhow::Result<KinematicModel> {
        match s.to_lowercase().as_str() {
            "full3d" => Ok(KinematicModel::Full3D),
            "differential" | "differentialdrive" | "wheeled2d" => Ok(KinematicModel::Differential),
            "holonomic2d" | "holonomic" => Ok(KinematicModel::Holonomic2D),
            "ackermann" => {
                let ackermann_ns = ackermann_ns.ok_or_else(|| anyhow!("Ackermann kinematic model requires configuration under kinematic_model.ackermann namespace"))?;
                let imu_to_rear_axle = resolve_var!(ackermann_ns, "imu_to_rear_axle", as Vector3<f64>, Rhs::Array(v) => {
                    let (Some(x_rhs), Some(y_rhs), Some(z_rhs)) = (v.first(), v.get(1), v.get(2)) else { return Err(anyhow!("Expected three vector components for imu_to_rear_axle")) };
                    let (Rhs::Val(Val::UnitedVal(UnitVal::LengthVal(x))), Rhs::Val(Val::UnitedVal(UnitVal::LengthVal(y))), Rhs::Val(Val::UnitedVal(UnitVal::LengthVal(z)))) = (&**x_rhs, &**y_rhs, &**z_rhs) else { return Err(anyhow!("All imu_to_rear_axle components must be length values.")) };
                    Vector3::new(x.get::<meter>(), y.get::<meter>(), z.get::<meter>())
                })?;
                Ok(KinematicModel::Ackermann { imu_to_rear_axle })
            }
            _ => {
                if let Some(ack_ns) = ackermann_ns {
                    let _ = ack_ns.defaults.evaluate_stats_only("imu_to_rear_axle");
                    let _ = ack_ns.user.evaluate_stats_only("imu_to_rear_axle");
                }
                Err(anyhow!("Unknown kinematic model: {}", s))
            }
        }
    }

    let (
        neighbor_max_range,
        map_resolution,
        map_capacity,
        dist_update_sensitivity,
        ekf_max_iterations,
    ) = {
        let registration = asts.filter_ns(&["registration"]);
        let neighbor_max_range = resolve_var!(registration, "neighbors_max_range", as f64, Rhs::Val(Val::UnitedVal(UnitVal::LengthVal(v))) => { v.get::<meter>() })?;

        let ekf_max_iterations = resolve_var!(registration, "max_iter", as usize, Rhs::Val(Val::NumVal(NumVal::Integer(i))) => { i })?;

        let map = registration.filter_ns(&["map"]);
        let map_resolution = resolve_var!(map, "resolution", as f64, Rhs::Val(Val::UnitedVal(UnitVal::LengthVal(v))) => { v.get::<meter>() })?;
        let map_capacity = resolve_var!(map, "capacity", as usize, Rhs::Val(Val::NumVal(NumVal::Integer(i))) => {i})?;
        let dist_update_sensitivity = resolve_var!(map, "distribution_update_sensitivity", as f32, Rhs::Val(Val::NumVal(NumVal::Floating(i))) => {i as f32} )?;

        (
            neighbor_max_range,
            map_resolution,
            map_capacity,
            dist_update_sensitivity,
            ekf_max_iterations,
        )
    };

    let sensor_drivers = {
        let lidar = asts.filter_ns(&["lidar", "driver"]);
        let vel = lidar.filter_ns(&["velodyne"]);
        let vel_ts = vel.filter_ns(&["timestamp"]);
        let lidar_type = resolve_var!(lidar, "type", as LidarDriverType, Rhs::Val(Val::StringVal(s)) => { match s.to_lowercase().as_str() { "disabled" => LidarDriverType::Disabled, "velodyne" => LidarDriverType::Velodyne, "ouster" => LidarDriverType::Ouster, _ => return Err(anyhow!("invalid lidar.driver.type '{s}'; expected Disabled, Velodyne, or Ouster")) } })?;
        let model = resolve_var!(vel, "model", as VelodyneModel, Rhs::Val(Val::StringVal(s)) => { match s.to_lowercase().as_str() { "vlp16" => VelodyneModel::Vlp16, "vlp32c" => VelodyneModel::Vlp32c, "pucklite" => VelodyneModel::PuckLite, "puckhires" => VelodyneModel::PuckHiRes, _ => return Err(anyhow!("unsupported Velodyne model '{s}'")) } })?;
        let bind_address: String =
            resolve_var!(vel, "bind_address", as String, Rhs::Val(Val::StringVal(s)) => {s})?;
        bind_address
            .parse::<std::net::IpAddr>()
            .map_err(|_| anyhow!("lidar.driver.velodyne.bind_address must be an IP address"))?;
        let data_port = resolve_var!(vel, "data_port", as u16, Rhs::Val(Val::NumVal(NumVal::Integer(i))) => { u16::try_from(i).map_err(|_| anyhow!("Velodyne data_port must be in 1..=65535"))? })?;
        let position_port = resolve_var!(vel, "position_port", as u16, Rhs::Val(Val::NumVal(NumVal::Integer(i))) => { u16::try_from(i).map_err(|_| anyhow!("Velodyne position_port must be in 1..=65535"))? })?;
        if data_port == 0 || position_port == 0 || data_port == position_port {
            return Err(anyhow!("Velodyne ports must be non-zero and distinct"));
        }
        let device_ip_raw: String =
            resolve_var!(vel, "device_ip", as String, Rhs::Val(Val::StringVal(s)) => {s})?;
        let device_ip = if device_ip_raw.is_empty() {
            None
        } else {
            device_ip_raw
                .parse::<std::net::IpAddr>()
                .map_err(|_| anyhow!("lidar.driver.velodyne.device_ip must be an IP address"))?;
            Some(device_ip_raw)
        };
        let v_max_skew = resolve_var!(vel_ts, "max_host_skew", as f64, Rhs::Val(Val::UnitedVal(UnitVal::TimeVal(v))) => {v.get::<second>()})?;
        let v_timeout = resolve_var!(vel_ts, "startup_timeout", as f64, Rhs::Val(Val::UnitedVal(UnitVal::TimeVal(v))) => {v.get::<second>()})?;
        let v_timestamp_mode = resolve_var!(vel_ts, "mode", as VelodyneTimestampMode, Rhs::Val(Val::StringVal(s)) => { match s.to_lowercase().as_str() { "sensor" => VelodyneTimestampMode::Sensor, "host" => VelodyneTimestampMode::Host, _ => return Err(anyhow!("Velodyne timestamp mode must be Sensor or Host")) } })?;
        let imu = asts.filter_ns(&["imu", "driver"]);
        let xs = imu.filter_ns(&["xsens"]);
        let xs_ts = xs.filter_ns(&["timestamp"]);
        let imu_type = resolve_var!(imu, "type", as ImuDriverType, Rhs::Val(Val::StringVal(s)) => { match s.to_lowercase().as_str() { "disabled" => ImuDriverType::Disabled, "xsens" => ImuDriverType::Xsens, "ouster" => ImuDriverType::Ouster, _ => return Err(anyhow!("invalid imu.driver.type '{s}'; expected Disabled, Xsens, or Ouster")) } })?;
        let baudrate = resolve_var!(xs, "baudrate", as u32, Rhs::Val(Val::NumVal(NumVal::Integer(i))) => {u32::try_from(i)?})?;
        let output_rate = resolve_var!(xs, "output_rate", as u32, Rhs::Val(Val::NumVal(NumVal::Integer(i))) => {u32::try_from(i)?})?;
        if baudrate == 0 || output_rate == 0 || output_rate > 2000 {
            return Err(anyhow!(
                "Xsens baudrate and output_rate must be positive; output_rate must not exceed 2000 Hz"
            ));
        }
        let coordinate_system = resolve_var!(xs, "coordinate_system", as CoordinateSystem, Rhs::Val(Val::StringVal(s)) => {match s.to_lowercase().as_str(){"enu"=>CoordinateSystem::Enu,"ned"=>CoordinateSystem::Ned,"nwu"=>CoordinateSystem::Nwu,_=>return Err(anyhow!("invalid Xsens coordinate_system '{s}'"))}})?;
        let x_max_skew = resolve_var!(xs_ts, "max_host_skew", as f64, Rhs::Val(Val::UnitedVal(UnitVal::TimeVal(v))) => {v.get::<second>()})?;
        let x_timeout = resolve_var!(xs_ts, "startup_timeout", as f64, Rhs::Val(Val::UnitedVal(UnitVal::TimeVal(v))) => {v.get::<second>()})?;
        let x_timestamp_mode = resolve_var!(xs_ts, "mode", as XsensTimestampMode, Rhs::Val(Val::StringVal(s)) => { match s.to_lowercase().as_str() { "utc" => XsensTimestampMode::Utc, "host" => XsensTimestampMode::Host, _ => return Err(anyhow!("Xsens timestamp mode must be Utc or Host")) } })?;
        if v_max_skew <= 0.0 || v_timeout <= 0.0 || x_max_skew <= 0.0 || x_timeout <= 0.0 {
            return Err(anyhow!(
                "driver clock skew and startup timeout values must be positive"
            ));
        }
        let ouster = asts.filter_ns(&["sensor_drivers", "ouster"]);
        let ouster_ts = ouster.filter_ns(&["timestamp"]);
        let ouster_bind_address: String =
            resolve_var!(ouster, "bind_address", as String, Rhs::Val(Val::StringVal(s)) => {s})?;
        ouster_bind_address
            .parse::<std::net::IpAddr>()
            .map_err(|_| anyhow!("sensor_drivers.ouster.bind_address must be an IP address"))?;
        let ouster_lidar_port = resolve_var!(ouster, "lidar_port", as u16, Rhs::Val(Val::NumVal(NumVal::Integer(i))) => { u16::try_from(i).map_err(|_| anyhow!("Ouster lidar_port must be in 1..=65535"))? })?;
        let ouster_imu_port = resolve_var!(ouster, "imu_port", as u16, Rhs::Val(Val::NumVal(NumVal::Integer(i))) => { u16::try_from(i).map_err(|_| anyhow!("Ouster imu_port must be in 1..=65535"))? })?;
        if ouster_lidar_port == 0 || ouster_imu_port == 0 || ouster_lidar_port == ouster_imu_port {
            return Err(anyhow!("Ouster ports must be non-zero and distinct"));
        }
        let ouster_device_ip_raw: String =
            resolve_var!(ouster, "device_ip", as String, Rhs::Val(Val::StringVal(s)) => {s})?;
        let ouster_device_ip = if ouster_device_ip_raw.is_empty() {
            None
        } else {
            ouster_device_ip_raw
                .parse::<std::net::IpAddr>()
                .map_err(|_| anyhow!("sensor_drivers.ouster.device_ip must be an IP address"))?;
            Some(ouster_device_ip_raw)
        };
        let metadata_path: String =
            resolve_var!(ouster, "metadata_path", as String, Rhs::Val(Val::StringVal(s)) => {s})?;
        let metadata_path = (!metadata_path.is_empty()).then_some(metadata_path);
        let ouster_max_skew = resolve_var!(ouster_ts, "max_host_skew", as f64, Rhs::Val(Val::UnitedVal(UnitVal::TimeVal(v))) => {v.get::<second>()})?;
        let ouster_timeout = resolve_var!(ouster_ts, "startup_timeout", as f64, Rhs::Val(Val::UnitedVal(UnitVal::TimeVal(v))) => {v.get::<second>()})?;
        if ouster_max_skew <= 0.0 || ouster_timeout <= 0.0 {
            return Err(anyhow!(
                "Ouster clock skew and startup timeout values must be positive"
            ));
        }
        let ouster_timestamp_mode = resolve_var!(ouster_ts, "mode", as OusterTimestampMode, Rhs::Val(Val::StringVal(s)) => { match s.to_lowercase().as_str() { "ptp" => OusterTimestampMode::Ptp, "host" => OusterTimestampMode::Host, _ => return Err(anyhow!("Ouster timestamp mode must be Ptp or Host")) } })?;
        let ouster_hostname: String =
            resolve_var!(ouster, "hostname", as String, Rhs::Val(Val::StringVal(s)) => {s})?;
        if (lidar_type == LidarDriverType::Ouster || imu_type == ImuDriverType::Ouster)
            && metadata_path.is_none()
            && ouster_hostname.is_empty()
        {
            return Err(anyhow!(
                "Ouster hostname is required when metadata_path is empty"
            ));
        }

        SensorDriverConfig {
            lidar: LidarDriverConfig {
                driver_type: lidar_type,
                velodyne: VelodyneConfig {
                    model,
                    bind_address,
                    data_port,
                    position_port,
                    device_ip,
                    frame_id: resolve_var!(vel,"frame_id",as String,Rhs::Val(Val::StringVal(s))=>{s})?,
                    cut_angle_rad: resolve_var!(vel,"cut_angle",as f64,Rhs::Val(Val::NumVal(NumVal::Floating(v)))=>{v})?,
                    timestamp_mode: v_timestamp_mode,
                    allow_host_hour_anchor: resolve_var!(vel_ts,"allow_host_hour_anchor",as bool,Rhs::Val(Val::BoolVal(b))=>{b})?,
                    max_host_skew_secs: v_max_skew,
                    startup_timeout_secs: v_timeout,
                },
            },
            imu: ImuDriverConfig {
                driver_type: imu_type,
                xsens: XsensConfig {
                    port: resolve_var!(xs,"port",as String,Rhs::Val(Val::StringVal(s))=>{s})?,
                    baudrate,
                    frame_id: resolve_var!(xs,"frame_id",as String,Rhs::Val(Val::StringVal(s))=>{s})?,
                    output_rate,
                    coordinate_system,
                    configure_device: resolve_var!(xs,"configure_device",as bool,Rhs::Val(Val::BoolVal(b))=>{b})?,
                    timestamp_mode: x_timestamp_mode,
                    max_host_skew_secs: x_max_skew,
                    startup_timeout_secs: x_timeout,
                },
            },
            ouster: OusterConfig {
                hostname: ouster_hostname,
                metadata_path,
                bind_address: ouster_bind_address,
                lidar_port: ouster_lidar_port,
                imu_port: ouster_imu_port,
                device_ip: ouster_device_ip,
                lidar_frame_id: resolve_var!(ouster,"lidar_frame_id",as String,Rhs::Val(Val::StringVal(s))=>{s})?,
                imu_frame_id: resolve_var!(ouster,"imu_frame_id",as String,Rhs::Val(Val::StringVal(s))=>{s})?,
                timestamp_mode: ouster_timestamp_mode,
                max_host_skew_secs: ouster_max_skew,
                startup_timeout_secs: ouster_timeout,
            },
        }
    };

    // --- Lidar Configuration ---
    let (
        cloud_src_topic,
        cloud_qos,
        cloud_subscriber_mode,
        lidar_vendor,
        lidar_odom_cov,
        lidar_as_sync,
    ) = {
        let lidar = asts.filter_ns(&["lidar"]);
        let cloud_src_topic = resolve_var!(lidar, "topic", as String, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {s})?;
        let cloud_qos = resolve_var!(lidar, "qos.reliability", as Qos, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {reliablility_parse(&s)?})?;
        let cloud_subscriber_mode = resolve_var!(lidar, "subscriber_mode", as PointCloudMode, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {pointcloud_mode_parse(&s)?})?;
        let lidar_vendor = resolve_var!(lidar, "vendor", as LidarVendor, Rhs::Val(Val::StringVal(s)) => {lidar_vendor_parse(&s)?})?;
        let lidar_odom_cov = resolve_var!(lidar, "covariance", as f64, Rhs::Val(Val::NumVal(NumVal::Floating(f))) => {f})?;
        let lidar_as_sync =
            resolve_var!(lidar, "as_sync", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;
        (
            cloud_src_topic,
            cloud_qos,
            cloud_subscriber_mode,
            lidar_vendor,
            lidar_odom_cov,
            lidar_as_sync,
        )
    };

    let (
        wheel_src_topic,
        wheel_qos,
        wheel_odom_cov,
        wheel_kinematic_constraint_std,
        wheel_kinematic_model,
        wheel_to_imu_tf_override,
        wheel_to_imu_rotation,
    ) = {
        let wheel = asts.filter_ns(&["wheel"]);
        let wheel_src_topic = resolve_var!(wheel, "topic", as String, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {s})?;
        let wheel_qos = resolve_var!(wheel, "qos.reliability", as Qos, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {reliablility_parse(&s)?})?;
        let wheel_odom_cov = match resolve_var!(wheel, "covariance", as f64, Rhs::Val(Val::NumVal(NumVal::Floating(f))) => {f})
        {
            Ok(v) => Some(v),
            Err(_) => None,
        };
        let wheel_kinematic_constraint_std = resolve_var!(wheel, "kinematic_constraint_std", as f64, Rhs::Val(Val::NumVal(NumVal::Floating(f))) => {f})?;

        let kinematic_model_str: String =
            resolve_var!(wheel, "kinematic_model", as String, Rhs::Val(Val::StringVal(s)) => {s})?;
        let ackermann_ns = if kinematic_model_str.to_lowercase() == "ackermann" {
            Some(wheel.filter_ns(&["ackermann"]))
        } else {
            let ack_ns = wheel.filter_ns(&["ackermann"]);
            let _ = ack_ns.defaults.evaluate_stats_only("imu_to_rear_axle");
            let _ = ack_ns.user.evaluate_stats_only("imu_to_rear_axle");
            None
        };
        let wheel_kinematic_model =
            kinematic_model_parse(&kinematic_model_str, ackermann_ns.as_ref())?;
        let transform = wheel.filter_ns(&["to_imu_transform"]);
        let wheel_imu_tf_override =
            resolve_var!(transform, "use_tf_tree", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;
        let initial_rotation_odom_to_imu = resolve_var!(transform, "rotation", as Matrix3<f64>, Rhs::Array(rows) => {
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
        (
            wheel_src_topic,
            wheel_qos,
            wheel_odom_cov,
            wheel_kinematic_constraint_std,
            wheel_kinematic_model,
            wheel_imu_tf_override,
            initial_rotation_odom_to_imu,
        )
    };

    let (imu_params, imu_sync_params, imu_node_params) = {
        let imu = asts.filter_ns(&["imu"]);
        let imu_src_topic = resolve_var!(imu, "topic", as String, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {s})?;
        let imu_qos = resolve_var!(imu, "qos.reliability", as Qos, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {reliablility_parse(&s)?})?;
        let imu_predictor_init_time_sec = resolve_var!(imu, "initial_setup", as f64, Rhs::Val(Val::UnitedVal(UnitVal::TimeVal(v))) => { v.get::<second>() })?;

        let transform = imu.filter_ns(&["from_lidar_transform"]);
        let imu_lidar_tf_override =
            resolve_var!(transform, "use_tf_tree", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;
        let initial_translation_lidar_to_imu = resolve_var!(transform, "translation", as Vector3<f64>, Rhs::Array(v) => {
            let (Some(x_rhs), Some(y_rhs), Some(z_rhs)) = (v.first(), v.get(1), v.get(2)) else { return Err(anyhow!("Expected three vector components.")) };
            let (Rhs::Val(Val::UnitedVal(UnitVal::LengthVal(x))), Rhs::Val(Val::UnitedVal(UnitVal::LengthVal(y))), Rhs::Val(Val::UnitedVal(UnitVal::LengthVal(z)))) = (&**x_rhs, &**y_rhs, &**z_rhs) else { return Err(anyhow!("All components must be length values.")) };
            let x_m = x.get::<meter>();
            let y_m = y.get::<meter>();
            let z_m = z.get::<meter>();
            Vector3::new(x_m, y_m, z_m)
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

        let from_lidar = imu.filter_ns(&["from_lidar_transform"]);
        let estimate_extrinsic_online =
            resolve_var!(from_lidar, "estimate_online", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;
        let lidar_only =
            resolve_var!(imu, "lidar_only", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;

        let calib = imu.filter_ns(&["calibration"]);
        let imu_calibrate_gyros =
            resolve_var!(calib, "gyroscope", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;
        let gyro_calibration_samples = resolve_var!(calib, "samples", as usize, Rhs::Val(Val::NumVal(NumVal::Integer(i))) => {i})?;
        let imu_intrinsics_calib_file_path: Option<String> = resolve_var!(calib, "parameter_file", as Option<String>, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => { if s.is_empty() { None } else { Some(s) } })?;
        let imu_time_offset = resolve_var!(calib, "time_offset", as f64, Rhs::Val(Val::UnitedVal(UnitVal::TimeVal(v))) => { v.get::<second>() })?;

        let noise_from_file =
            resolve_var!(calib, "noise_from_file", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;
        let noise_inflation = resolve_var!(calib, "noise_inflation", as f64, Rhs::Val(Val::NumVal(nv)) => {
            match nv {
                NumVal::Floating(f) => f,
                NumVal::Integer(i) => i as f64,
            }
        })?;

        // Replacing the tuned constants is opt-in, so this only runs when it
        // was asked for, and says which numbers it swapped.
        let (cov_gyr, cov_acc, cov_gyr_bias, cov_acc_bias) = if noise_from_file {
            let path = imu_intrinsics_calib_file_path.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "imu.calibration.noise_from_file is on but imu.calibration.parameter_file \
                     is empty, so there is no file to take the noise parameters from"
                )
            })?;
            let derived = derive_noise_covariances(path, noise_inflation)?;
            log::info!(
                "IMU covariances from {path} (inflation {noise_inflation}x): \
                 gyroscope {:.3e} (was {:.3e}), acceleration {:.3e} (was {:.3e}), \
                 gyroscope bias {:.3e} (was {:.3e}), acceleration bias {:.3e} (was {:.3e})",
                derived.gyr,
                cov_gyr_val,
                derived.acc,
                cov_acc_val,
                derived.gyr_bias,
                cov_gyr_bias_val,
                derived.acc_bias,
                cov_acc_bias_val,
            );
            (
                Vector3::new(derived.gyr, derived.gyr, derived.gyr),
                Vector3::new(derived.acc, derived.acc, derived.acc),
                Vector3::new(derived.gyr_bias, derived.gyr_bias, derived.gyr_bias),
                Vector3::new(derived.acc_bias, derived.acc_bias, derived.acc_bias),
            )
        } else {
            (cov_gyr, cov_acc, cov_gyr_bias, cov_acc_bias)
        };

        // LiDAR-only trajectories use the LiDAR frame directly.
        let (initial_translation_lidar_to_imu, initial_rotation_lidar_to_imu) = if lidar_only {
            if initial_translation_lidar_to_imu != Vector3::zeros()
                || initial_rotation_lidar_to_imu != Matrix3::identity()
            {
                log::info!(
                    "imu.lidar_only is on, so imu.from_lidar_transform is ignored and the \
                     trajectory is the LiDAR's own."
                );
            }
            (Vector3::zeros(), Matrix3::identity())
        } else {
            (
                initial_translation_lidar_to_imu,
                initial_rotation_lidar_to_imu,
            )
        };

        let imu_params = ImuParameters {
            initial_translation_lidar_to_imu,
            initial_rotation_lidar_to_imu,
            estimate_extrinsic_online,
            lidar_only,
            cov_gyr,
            cov_gyr_bias,
            cov_acc,
            cov_acc_bias,
            kinematic_constraint_std: wheel_kinematic_constraint_std,
            kinematic_model: wheel_kinematic_model,
            wheel_odom_cov_override: wheel_odom_cov,
        };
        let imu_sync_params = ImuConfigParameter {
            madgwick_setting,
            gravity,
            calibration: ImuCalibrationParameters {
                imu_calibrate_gyros,
                gyro_calibration_samples,
                imu_intrinsics_calib_file_path,
                time_offset: imu_time_offset,
                ..Default::default()
            },
            ..Default::default()
        };
        let imu_node_params = (
            imu_src_topic,
            imu_qos,
            imu_predictor_init_time_sec,
            imu_lidar_tf_override && !lidar_only,
        );

        (imu_params, imu_sync_params, imu_node_params)
    };

    let (
        imu_filtered_topic,
        show_tui,
        debug_fps,
        map_frame,
        logger_prefix,
        odom_qos,
        odometry_rate,
        path_qos,
        imu_odom_qos,
        imu_path_qos,
        imu_filtered_qos,
        warning_debounce,
        rerun,
        base_link_frame,
        base_to_lidar,
        base_to_lidar_tf_override,
        cloud_publisher_mode,
    ): (
        _,
        _,
        _,
        _,
        _,
        _,
        OdometryRate,
        _,
        _,
        _,
        _,
        _,
        _,
        Option<String>,
        nalgebra::Isometry3<f64>,
        bool,
        PointCloudMode,
    ) = {
        let output_ns = asts.filter_ns(&["output"]);
        let terminal_ns = output_ns.filter_ns(&["terminal"]);
        let show_tui: bool =
            resolve_var!(terminal_ns, "ui", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;
        let show_debug: bool =
            resolve_var!(terminal_ns, "debug", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;
        let debug_fps = if show_debug {
            resolve_var!(terminal_ns, "debug_fps", as i32, Rhs::Val(Val::NumVal(NumVal::Integer(i))) => {i})?
        } else {
            // Mark debug_fps as "used" even when debug is disabled
            let _ = terminal_ns.defaults.evaluate_stats_only("debug_fps");
            let _ = terminal_ns.user.evaluate_stats_only("debug_fps");
            0
        };
        let warning_debounce = resolve_var!(terminal_ns, "warning_debounce", as f64, Rhs::Val(Val::UnitedVal(UnitVal::TimeVal(v))) => { v.get::<second>() as f64 })?;
        // optional log level for terminal/logger
        let log_level: Option<String> = match resolve_var!(terminal_ns, "log_level", as String, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {s})
        {
            Ok(s) => Some(s),
            Err(_) => None,
        };
        if let Some(ref lvl) = log_level {
            unsafe {
                std::env::set_var("RUST_LOG", lvl.as_str());
            }
        }
        let rerun_ns = output_ns.filter_ns(&["rerun"]);
        let rerun_server_addr = resolve_var!(rerun_ns, "server_addr", as String, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {s})?;
        let rerun_map_ns = rerun_ns.filter_ns(&["map"]);
        let rerun_map_enabled =
            resolve_var!(rerun_map_ns, "enabled", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;
        let rerun_map_publish_interval_sec: f64 = resolve_var!(rerun_map_ns, "publish_interval", as f64, Rhs::Val(Val::UnitedVal(UnitVal::TimeVal(v))) => {v.get::<second>()})?;
        let rerun_map_visual_scale: f32 = resolve_var!(rerun_map_ns, "visual_scale", as f32, Rhs::Val(Val::NumVal(NumVal::Floating(v))) => {v as f32})?;
        if !rerun_map_publish_interval_sec.is_finite() || rerun_map_publish_interval_sec < 0.0 {
            return Err(anyhow!(
                "output.rerun.map.publish_interval must be finite and non-negative"
            ));
        }
        if !rerun_map_visual_scale.is_finite() || rerun_map_visual_scale <= 0.0 {
            return Err(anyhow!(
                "output.rerun.map.visual_scale must be finite and positive"
            ));
        }
        let rerun = RerunConfig {
            enabled: false,
            server_addr: rerun_server_addr,
            map: RerunMapConfig {
                enabled: rerun_map_enabled,
                publish_interval_sec: rerun_map_publish_interval_sec,
                visual_scale: rerun_map_visual_scale,
            },
        };

        let map_frame = resolve_var!(output_ns, "global_frame", as String, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {s})?;
        let logger_prefix = resolve_var!(output_ns, "logger_prefix", as String, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {s})?;
        let odom_qos = resolve_var!(output_ns, "lio_odometry_msg_qos", as Qos, Rhs::Val(Val::StringVal(s)) => {reliablility_parse(&s)?})?;
        let odometry_rate = resolve_var!(output_ns, "odometry_rate", as OdometryRate, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {
            match s.to_lowercase().as_str() {
                "lidar" => OdometryRate::Lidar,
                "imuandlidar" | "imu_and_lidar" => OdometryRate::ImuAndLidar,
                _ => return Err(anyhow!("invalid output.odometry_rate '{s}'; expected Lidar or ImuAndLidar")),
            }
        })?;
        let imu_filtered_topic = resolve_var!(output_ns, "imu_ahrs_filtered_topic", as String, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {s})?;
        let path_qos = resolve_var!(output_ns, "lio_path_msg_qos", as Qos, Rhs::Val(Val::StringVal(s)) => {reliablility_parse(&s)?})?;
        let imu_odom_qos = resolve_var!(output_ns, "imu_odometry_msg_qos", as Qos, Rhs::Val(Val::StringVal(s)) => {reliablility_parse(&s)?})?;
        let imu_path_qos = resolve_var!(output_ns, "imu_path_msg_qos", as Qos, Rhs::Val(Val::StringVal(s)) => {reliablility_parse(&s)?})?;
        let imu_filtered_qos = resolve_var!(output_ns, "imu_ahrs_filtered_msg_qos", as Qos, Rhs::Val(Val::StringVal(s)) => {reliablility_parse(&s)?})?;
        let cloud_publisher_mode = resolve_var!(output_ns, "publisher_mode", as PointCloudMode, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {pointcloud_mode_parse(&s)?})?;

        let base_link_ns = output_ns.filter_ns(&["base_link"]);
        let base_link_frame = match resolve_var!(base_link_ns, "frame_id", as String, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {s})
        {
            Ok(s) => {
                let s: String = s;
                if s.is_empty() { None } else { Some(s) }
            }
            Err(_) => None,
        };

        let base_transform_ns = base_link_ns.filter_ns(&["from_lidar_transform"]);
        let base_to_lidar_tf_override = resolve_var!(base_transform_ns, "use_tf_tree", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;
        let base_translation: Vector3<f64> = resolve_var!(base_transform_ns, "translation", as Vector3<f64>, Rhs::Array(v) => {
            let (Some(x_rhs), Some(y_rhs), Some(z_rhs)) = (v.first(), v.get(1), v.get(2)) else { return Err(anyhow!("Expected three vector components.")) };
            let (Rhs::Val(Val::UnitedVal(UnitVal::LengthVal(x))), Rhs::Val(Val::UnitedVal(UnitVal::LengthVal(y))), Rhs::Val(Val::UnitedVal(UnitVal::LengthVal(z)))) = (&**x_rhs, &**y_rhs, &**z_rhs) else { return Err(anyhow!("All components must be length values.")) };
            Vector3::new(x.get::<meter>(), y.get::<meter>(), z.get::<meter>())
        })?;
        let base_rotation: Matrix3<f64> = resolve_var!(base_transform_ns, "rotation", as Matrix3<f64>, Rhs::Array(rows) => {
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
        let base_to_lidar = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::from(base_translation),
            nalgebra::UnitQuaternion::from_matrix(&base_rotation),
        );

        (
            imu_filtered_topic,
            show_tui,
            debug_fps,
            map_frame,
            logger_prefix,
            odom_qos,
            odometry_rate,
            path_qos,
            imu_odom_qos,
            imu_path_qos,
            imu_filtered_qos,
            warning_debounce,
            rerun,
            base_link_frame,
            base_to_lidar,
            base_to_lidar_tf_override,
            cloud_publisher_mode,
        )
    };

    let (
        initial_scan_accumulate,
        point_skip,
        preprocessing_backend,
        query_resolution_factor,
        bypass_output_downsample,
        passthrough_filter,
        global_zero_frame,
    ) = {
        let preproc_ns = asts.filter_ns(&["preprocessing"]);
        let initial_scan_accumulate = resolve_var!(preproc_ns, "initial_scan_accumulate", as usize, Rhs::Val(Val::NumVal(NumVal::Integer(i))) => { i })?;
        let downsample_ns = preproc_ns.filter_ns(&["downsample"]);
        let point_skip = resolve_var!(downsample_ns, "point_skip", as usize, Rhs::Val(Val::NumVal(NumVal::Integer(i))) => { i })?;
        let preprocessing_backend = resolve_var!(downsample_ns, "backend", as PreprocessingBackend, Rhs::Val(Val::StringVal(s)) => { preprocessing_backend_parse(&s)? })?;
        let bypass_output_downsample = resolve_var!(downsample_ns, "bypass_output_downsample", as bool, Rhs::Val(Val::BoolVal(b)) => { b })?;
        let query_resolution_factor: f64 = resolve_var!(downsample_ns, "query_resolution_factor", as f64, Rhs::Val(Val::NumVal(NumVal::Floating(v))) => { v })?;

        let filter_ns = preproc_ns.filter_ns(&["filter"]);
        let bbox_ns = filter_ns.filter_ns(&["bounding_box"]);
        let (min_x, max_x) = resolve_length_range_meters_float!(bbox_ns, "x", f64::MIN, f64::MAX)?;
        let (min_y, max_y) = resolve_length_range_meters_float!(bbox_ns, "y", f64::MIN, f64::MAX)?;
        let (min_z, max_z) = resolve_length_range_meters_float!(bbox_ns, "z", f64::MIN, f64::MAX)?;
        let min = Vector3::new(min_x, min_y, min_z);
        let max = Vector3::new(max_x, max_y, max_z);
        let invert_bounding_box =
            resolve_var!(bbox_ns, "invert", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;

        let dist_ns = filter_ns.filter_ns(&["distance"]);
        let (min_dist, max_dist) =
            resolve_length_range_meters_float!(dist_ns, "range", 0.0, f64::MAX / 2.5)?;
        let invert_distance =
            resolve_var!(dist_ns, "invert", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;

        let hfov_ns = filter_ns.filter_ns(&["horizontal_fov"]);
        let (fov_left, fov_right) = resolve_angle_range!(hfov_ns, "degrees", -180.0, 180.0)?;
        let enable_horizontal_fov =
            resolve_var!(hfov_ns, "enable", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;
        let invert_fov =
            resolve_var!(hfov_ns, "invert", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;
        let forward = resolve_var!(hfov_ns, "forward", as Vector2<f32>, Rhs::Array(v) => {
            let (Some(x_rhs), Some(y_rhs)) = (v.first(), v.get(1)) else { return Err(anyhow!("Expected two components for forward vector.")) };
            let (Rhs::Val(Val::NumVal(NumVal::Floating(x))), Rhs::Val(Val::NumVal(NumVal::Floating(y)))) = (&**x_rhs, &**y_rhs) else { return Err(anyhow!("Forward vector components must be floats.")) };
            Vector2::new(*x as f32, *y as f32)
        })?;

        let poly_ns = filter_ns.filter_ns(&["polygon_2d"]);
        let coords_vec: Vec<Point2<f32>> = resolve_var!(poly_ns, "coords", as Vec<Point2<f32>>, Rhs::Array(points) => {
            let mut coords = Vec::new();
            for point_rhs in points {
                let Rhs::Array(point_coords) = &*point_rhs else { return Err(anyhow!("Expected polygon vertex to be an array [x, y].")) };
                let (Some(x_rhs), Some(y_rhs)) = (point_coords.first(), point_coords.get(1)) else { return Err(anyhow!("Expected two components [x, y] for polygon vertex.")) };
                let (Rhs::Val(Val::UnitedVal(UnitVal::LengthVal(x))), Rhs::Val(Val::UnitedVal(UnitVal::LengthVal(y)))) = (&**x_rhs, &**y_rhs) else { return Err(anyhow!("Polygon vertex components must be united values (e.g., in meters).")) };
                coords.push(Point2::new(x.get::<meter>() as f32, y.get::<meter>() as f32));
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
        let global_zero_frame = resolve_var!(poly_ns, "global_zero_frame", as String, Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => {s})?;

        let intensity_ns = filter_ns.filter_ns(&["intensity"]);
        let (min_intensity, max_intensity) =
            resolve_float_range!(intensity_ns, "values", f64::MIN, f64::MAX)?;
        let invert_intensity =
            resolve_var!(intensity_ns, "invert", as bool, Rhs::Val(Val::BoolVal(b)) => {b})?;

        let passthrough_filter = PassthroughFilterParameters {
            min: min.cast::<f32>(),
            max: max.cast::<f32>(),
            invert_bounding_box,
            min_dist: min_dist as f32,
            max_dist: max_dist as f32,
            invert_distance,
            fov_left,
            fov_right,
            forward,
            enable_horizontal_fov,
            polygon,
            invert_polygon,
            invert_fov,
            min_intensity: min_intensity as f32,
            max_intensity: max_intensity as f32,
            invert_intensity,
            filter_frame_transform: nalgebra::Isometry3::identity(), // TODO what is this param again? the override for global_zero_frame in tf?
        };
        (
            initial_scan_accumulate,
            point_skip,
            preprocessing_backend,
            query_resolution_factor,
            bypass_output_downsample,
            passthrough_filter,
            global_zero_frame,
        )
    };

    let (
        num_threads,
        cpu_only,
        parallelism_decision_threshold_stddevs,
        parallelism_decision_samples_per_mode,
        parallelism_execution_policy,
    ) = {
        let parallelism_ns = asts.filter_ns(&["parallelism"]);
        let num_threads = {
            let rhs = resolve_var!(parallelism_ns, "num_threads", as &Rhs, r => { r })?;
            match rhs {
                Rhs::Val(Val::NumVal(NumVal::Integer(i))) => i as usize,
                Rhs::Val(Val::StringVal(s)) if s.eq_ignore_ascii_case("all") => 0usize,
                _ => {
                    return Err(anyhow!(
                        "parallelism.num_threads must be an integer or 'all'"
                    ));
                }
            }
        };
        let cpu_only =
            resolve_var!(parallelism_ns, "cpu_only", as bool, Rhs::Val(Val::BoolVal(b)) => { b })?;
        let parallelism_execution_policy = resolve_var!(
            parallelism_ns,
            "execution_mode",
            as ExecutionPolicy,
            Rhs::Val(Val::StringVal(s)) | Rhs::Path(s) => { execution_policy_parse(&s)? }
        )?;
        let parallelism_decision_threshold_stddevs: f64 = resolve_var!(
            parallelism_ns,
            "decision_threshold_stddevs",
            as f64,
            Rhs::Val(Val::NumVal(NumVal::Floating(v))) => { v }
        )?;
        if !parallelism_decision_threshold_stddevs.is_finite()
            || parallelism_decision_threshold_stddevs < 0.0
        {
            return Err(anyhow!(
                "parallelism.decision_threshold_stddevs must be finite and non-negative"
            ));
        }
        let parallelism_decision_samples_per_mode = resolve_var!(
            parallelism_ns,
            "decision_samples_per_mode",
            as usize,
            Rhs::Val(Val::NumVal(NumVal::Integer(v))) => { v }
        )?;
        if parallelism_decision_samples_per_mode == 0 {
            return Err(anyhow!(
                "parallelism.decision_samples_per_mode must be greater than zero"
            ));
        }
        (
            num_threads,
            cpu_only,
            parallelism_decision_threshold_stddevs,
            parallelism_decision_samples_per_mode,
            parallelism_execution_policy,
        )
    };

    let final_config = LioConfig {
        sensor_drivers,
        rerun,
        neighbor_max_range,
        map_resolution,
        map_capacity,
        dist_update_sensitivity,
        ekf_max_iterations,
        lidar_odom_cov,
        show_tui,
        debug_fps,
        odom_qos,
        odometry_rate,
        path_qos,
        imu_odom_qos,
        imu_path_qos,
        imu_params,
        base_link_frame,
        base_to_lidar,
        base_to_lidar_tf_override,
        cpu_only,
        sync_node_params: SyncNodeParameters {
            wheel_qos,
            wheel_src_topic,
            cloud_src_topic,
            cloud_qos,
            cloud_subscriber_mode,
            imu_src_topic: imu_node_params.0,
            imu_qos: imu_node_params.1,
            imu_predictor_init_time_sec: imu_node_params.2,
            logger_prefix,
            imu_filtered_qos,
            imu_filtered_topic,
            synchronous_mode: false,
        },
        sync_params: SyncParameters {
            lidar_vendor,
            imu_lidar_tf_override: imu_node_params.3,
            map_frame,
            imu: imu_sync_params,
            downsample: DownsampleParameters {
                voxel_size: (query_resolution_factor * map_resolution) as f32,
                bypass_output_downsample,
                point_skip,
                backend: preprocessing_backend,
            },
            passthrough_filter,
            global_zero_frame,
            initial_scan_accumulate,
            wheel_imu_tf_override: wheel_to_imu_tf_override,
            wheel_to_imu_rotation,
            wheel_odom_cov,
            lidar_as_sync,
            warning_debounce,
            parallelism_decision_threshold_stddevs,
            parallelism_decision_samples_per_mode,
            parallelism_execution_policy,
        },
        // TODO currently hardcoded
        registered_cloud_qos: Qos::Sensor(Reliability::Reliable),
        cloud_publisher_mode,
        enable_live_extrinsic_calibration: false,
        num_threads,
    };

    let default_usage_info = if capture_default_usage {
        asts.defaults.usage_vars()
    } else {
        (vec![], vec![])
    };

    let user_usage_info = if capture_user_usage {
        asts.user.usage_vars()
    } else {
        (vec![], vec![])
    };

    Ok((final_config, default_usage_info, user_usage_info))
}

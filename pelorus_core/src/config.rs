use anyhow::anyhow;
use nalgebra::{Matrix3, Point2, Vector2, Vector3};
use ratslang::uom::si::angle::{degree, radian};
use ratslang::uom::si::f32::Angle;
use ratslang::uom::si::length::meter;
use ratslang::uom::si::time::second;
use ratslang::{Evaluated, VariableHistory};
use serde::{Deserialize, Deserializer};

use crate::real;
use crate::{
    CoordinateSystem, DownsampleParameters, ImuCalibrationParameters, ImuConfigParameter,
    ImuDriverConfig, ImuDriverType, ImuParameters, KinematicModel, LidarDriverConfig,
    LidarDriverType, LidarVendor, LioConfig, MadgwickSetting, McapConfig, OdometryRate,
    OusterConfig, OusterTimestampMode, PassthroughFilterParameters, Pipelining, PointCloudMode,
    PreprocessingBackend, Qos, Reliability, RerunConfig, RerunMapConfig, RobosenseConfig,
    RobosenseModel, RobosenseTimestampMode, SensorDriverConfig, SyncNodeParameters, SyncParameters,
    VelodyneConfig, VelodyneModel, VelodyneTimestampMode, XsensConfig, XsensTimestampMode,
    adaptive_execution::ExecutionPolicy,
};

/// A struct that holds the user and default configuration variable maps.
pub struct Asts {
    user: VariableHistory,
    defaults: VariableHistory,
}

impl Asts {
    pub fn new(defaults: VariableHistory, user: VariableHistory) -> Self {
        Self { user, defaults }
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
            "override{} {} not recognized. Run 'config generate full' to see every setting",
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
/// Config variable names as `(assigned, evaluated)`, used to detect unused variables.
pub type ConfigUsage = (Vec<String>, Vec<String>);

pub fn ast_to_config_with_usage(
    defaults: Evaluated,
    user: Evaluated,
) -> anyhow::Result<(LioConfig, ConfigUsage)> {
    let config = ast_to_config_inner(defaults, user, true, false)?;
    Ok((config.0, config.1))
}

#[allow(unreachable_patterns, clippy::unnecessary_fallible_conversions)]
pub fn ast_to_config_with_user_usage(
    mut defaults: Evaluated,
    user: Evaluated,
) -> anyhow::Result<(LioConfig, ConfigUsage)> {
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
    fn rerun_is_enabled_via_config_instead_of_a_cli_flag() {
        let overrides = parse_overrides(&["output.rerun=true".to_owned()]).unwrap();
        let user = user_config_with_checked_overrides(DEFAULTS, None, &overrides).unwrap();
        let defaults = ratslang::compile_code(DEFAULTS).unwrap();
        let config = ast_to_config(defaults, user).unwrap();

        assert!(config.rerun.enabled);
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
            "output.rerun.map=true".to_owned(),
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

    fn config_with(source: &str) -> crate::LioConfig {
        let user = ratslang::compile_code(source).unwrap();
        let defaults = ratslang::compile_code(DEFAULTS).unwrap();
        ast_to_config(defaults, user).unwrap()
    }

    #[test]
    fn renamed_export_settings_name_their_new_key() {
        let defaults = ratslang::compile_code(DEFAULTS).unwrap();
        let user = ratslang::compile_code("output.export.latency = \"run.txt\"").unwrap();
        let message = format!("{:#}", ast_to_config(defaults, user).err().unwrap());
        assert!(
            message.contains("'output.export.latency' is now 'output.latency'"),
            "{message}"
        );
    }

    #[test]
    fn exports_and_offline_settings_have_the_names_polarstern_uses() {
        let config = config_with("");
        assert!(config.export.trajectory.is_none());
        assert!(config.export.latency.is_none() && config.export.processing_time.is_none());
        assert!(!config.offline.keep_open && !config.offline.open_router);
        assert!(config.offline.cache_stream);

        let config = config_with(
            r#"
output.trajectory = "run.tum"
output.processing_time = "auto"
offline.keep_open = true
offline.open_router = true
offline.cache_stream = false
"#,
        );
        assert_eq!(config.export.trajectory.as_deref(), Some("run.tum"));
        // "auto" is the empty path the exporter names after the current time.
        assert_eq!(config.export.processing_time.as_deref(), Some(""));
        assert!(config.export.latency.is_none());
        assert!(config.offline.keep_open && config.offline.open_router);
        assert!(!config.offline.cache_stream);
    }

    #[test]
    fn multiresolution_is_the_default_and_can_be_opted_out_without_changing_downsampling() {
        let adaptive = config_with("");
        let fixed = config_with("registration.map.resolution.type = Single\n");
        assert!(adaptive.map_multiresolution);
        assert!(!fixed.map_multiresolution);
        // Enum-like settings are case-insensitive, same as every other one.
        assert!(config_with("registration.map.resolution.type = multi\n").map_multiresolution);
        assert!(!config_with("registration.map.resolution.type = SINGLE\n").map_multiresolution);
        assert_eq!(
            fixed.sync_params.downsample.voxel_size,
            adaptive.sync_params.downsample.voxel_size
        );
        for source in [
            "registration.map.resolution.type = Multi\nregistration.map.capacity = 3\n",
            "registration.map.resolution.type = Multi\nregistration.map.resolution = 0m\n",
        ] {
            let user = ratslang::compile_code(source).unwrap();
            let defaults = ratslang::compile_code(DEFAULTS).unwrap();
            assert!(ast_to_config(defaults, user).is_err());
        }
    }

    #[test]
    fn the_imu_time_offset_is_read_in_seconds() {
        assert_eq!(config_with("").sync_params.imu.calibration.time_offset, 0.0);

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
    fn the_tuned_covariances_are_the_defaults() {
        let config = config_with("");
        assert_eq!(config.imu_params.cov_gyr[0], 0.1);
        assert_eq!(config.imu_params.cov_acc[0], 0.1);
        assert_eq!(config.imu_params.cov_gyr_bias[0], 0.0001);
        assert_eq!(config.imu_params.cov_acc_bias[0], 0.0001);
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

    /// `LioConfig::defaults` must equal the configuration `default.rl` resolves to.
    /// Run with `PELORUS_UPDATE_DEFAULTS=1` to regenerate `default_config.rs`,
    /// and with the `f32` feature as well to regenerate `default_config_f32.rs`.
    #[test]
    fn default_config_matches_default_rl() {
        let defaults = ratslang::compile_code(include_str!("../default.rl")).unwrap();
        let user = ratslang::compile_code("").unwrap();
        let resolved = postcard::to_allocvec(&ast_to_config(defaults, user).unwrap()).unwrap();
        let file = if crate::REAL_IS_F32 {
            "default_config_f32.rs"
        } else {
            "default_config.rs"
        };
        if std::env::var_os("PELORUS_UPDATE_DEFAULTS").is_some() {
            let path = format!("{}/src/{file}", env!("CARGO_MANIFEST_DIR"));
            std::fs::write(
                path,
                format!(
                    "// Generated from default.rl by `default_config_matches_default_rl` in config.rs.\n\
                     // The postcard encoding of the resolved LioConfig with `Real` as `{}`.\n\
                     pub(crate) const DEFAULT_CONFIG: &[u8] = &{resolved:?};\n",
                    if crate::REAL_IS_F32 { "f32" } else { "f64" }
                ),
            )
            .unwrap();
            return;
        }
        let embedded = postcard::to_allocvec(&crate::LioConfig::defaults()).unwrap();
        assert_eq!(
            embedded, resolved,
            "{file} is stale, regenerate it with PELORUS_UPDATE_DEFAULTS=1"
        );
    }

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

mod input;
use input::RawConfig;

fn ast_to_config_inner(
    defaults: Evaluated,
    user: Evaluated,
    capture_default_usage: bool,
    capture_user_usage: bool,
) -> anyhow::Result<(LioConfig, ConfigUsage, ConfigUsage)> {
    let asts = Asts::new(defaults.vars, user.vars);
    let raw: RawConfig = ratslang::serde::from_variables_with_defaults(&asts.defaults, &asts.user)?;
    let RawConfig {
        registration,
        preprocessing,
        sensor_drivers,
        lidar,
        imu,
        wheel,
        output,
        parallelism,
        offline,
    } = raw;
    // Serde supplies None without visiting absent optional fields. Preserve the
    // recognized-key probes used by strict CLI override checks.
    for (key, absent) in [
        ("wheel.covariance", wheel.covariance.is_none()),
        (
            "output.terminal.log_level",
            output.terminal.log_level.is_none(),
        ),
    ] {
        if absent {
            let _ = asts.defaults.evaluate_stats_only(key);
            let _ = asts.user.evaluate_stats_only(key);
        }
    }
    let map = registration.map;
    anyhow::ensure!(
        !map.resolution.r#type
            || (map.resolution.value.is_finite()
                && map.resolution.value > 0.0
                && map.capacity >= 4),
        "multiresolution requires a finite positive map resolution and capacity >= 4"
    );

    let vel = lidar.driver.velodyne;
    let rs = lidar.driver.robosense;
    let xs = imu.driver.xsens;
    let ouster = sensor_drivers.ouster;
    for (name, first_port, second_port) in [
        ("Velodyne", vel.data_port, vel.position_port),
        ("Robosense", rs.msop_port, rs.difop_port),
        ("Ouster", ouster.lidar_port, ouster.imu_port),
    ] {
        anyhow::ensure!(first_port != second_port, "{name} ports must be distinct");
    }
    anyhow::ensure!(
        !(lidar.driver.r#type == LidarDriverType::Ouster
            || imu.driver.r#type == ImuDriverType::Ouster)
            || ouster.metadata_path.is_some()
            || !ouster.hostname.is_empty(),
        "Ouster hostname is required when metadata_path is empty"
    );
    let sensor_drivers = SensorDriverConfig {
        lidar: LidarDriverConfig {
            driver_type: lidar.driver.r#type,
            velodyne: VelodyneConfig {
                model: vel.model,
                bind_address: vel.bind_address,
                data_port: vel.data_port,
                position_port: vel.position_port,
                device_ip: vel.device_ip,
                frame_id: vel.frame_id,
                cut_angle_rad: vel.cut_angle,
                timestamp_mode: vel.timestamp.mode,
                allow_host_hour_anchor: vel.timestamp.allow_host_hour_anchor,
                max_host_skew_secs: vel.timestamp.max_host_skew,
                startup_timeout_secs: vel.timestamp.startup_timeout,
            },
            robosense: RobosenseConfig {
                model: rs.model,
                bind_address: rs.bind_address,
                msop_port: rs.msop_port,
                difop_port: rs.difop_port,
                device_ip: rs.device_ip,
                frame_id: rs.frame_id,
                timestamp_mode: rs.timestamp.mode,
                max_host_skew_secs: rs.timestamp.max_host_skew,
                startup_timeout_secs: rs.timestamp.startup_timeout,
            },
        },
        imu: ImuDriverConfig {
            driver_type: imu.driver.r#type,
            xsens: XsensConfig {
                port: xs.port,
                baudrate: xs.baudrate,
                output_rate: xs.output_rate,
                frame_id: xs.frame_id,
                coordinate_system: xs.coordinate_system,
                configure_device: xs.configure_device,
                timestamp_mode: xs.timestamp.mode,
                max_host_skew_secs: xs.timestamp.max_host_skew,
                startup_timeout_secs: xs.timestamp.startup_timeout,
            },
        },
        ouster: OusterConfig {
            hostname: ouster.hostname,
            metadata_path: ouster.metadata_path,
            bind_address: ouster.bind_address,
            lidar_port: ouster.lidar_port,
            imu_port: ouster.imu_port,
            device_ip: ouster.device_ip,
            lidar_frame_id: ouster.lidar_frame_id,
            imu_frame_id: ouster.imu_frame_id,
            timestamp_mode: ouster.timestamp.mode,
            max_host_skew_secs: ouster.timestamp.max_host_skew,
            startup_timeout_secs: ouster.timestamp.startup_timeout,
        },
    };
    let kinematic_model = match wheel.kinematic_model {
        KinematicModel::Ackermann { .. } => KinematicModel::Ackermann {
            imu_to_rear_axle: wheel.ackermann.imu_to_rear_axle.cast(),
        },
        model => model,
    };
    if !matches!(kinematic_model, KinematicModel::Ackermann { .. }) {
        // Keep inactive known settings recognized by strict CLI override checks.
        let _ = asts
            .defaults
            .evaluate_stats_only("wheel.ackermann.imu_to_rear_axle");
        let _ = asts
            .user
            .evaluate_stats_only("wheel.ackermann.imu_to_rear_axle");
    }
    let (translation, rotation) = if imu.lidar_only {
        if imu.from_lidar_transform.translation != Vector3::zeros()
            || imu.from_lidar_transform.rotation != Matrix3::identity()
        {
            log::info!(
                "imu.lidar_only is on, so imu.from_lidar_transform is ignored and the trajectory is the lidar's own."
            );
        }
        (Vector3::zeros(), Matrix3::identity())
    } else {
        (
            imu.from_lidar_transform.translation,
            imu.from_lidar_transform.rotation,
        )
    };
    let imu_params = ImuParameters {
        initial_translation_lidar_to_imu: translation.cast(),
        initial_rotation_lidar_to_imu: rotation.cast(),
        estimate_extrinsic_online: imu.from_lidar_transform.estimate_online,
        lidar_only: imu.lidar_only,
        cov_gyr: Vector3::repeat(real(imu.covariance_gyroscope)),
        cov_acc: Vector3::repeat(real(imu.covariance_acceleration)),
        cov_gyr_bias: Vector3::repeat(real(imu.covariance_gyroscope_bias)),
        cov_acc_bias: Vector3::repeat(real(imu.covariance_acceleration_bias)),
        kinematic_constraint_std: real(wheel.kinematic_constraint_std),
        kinematic_model,
        wheel_odom_cov_override: wheel.covariance.map(real),
    };
    let imu_sync_params = ImuConfigParameter {
        madgwick_setting: imu.ahrs_filtering.method,
        gravity: real(imu.gravity),
        calibration: ImuCalibrationParameters {
            imu_calibrate_gyros: imu.calibration.gyroscope.value,
            gyro_calibration_samples: imu.calibration.gyroscope.samples,
            imu_intrinsics_calib_file_path: imu.calibration.parameter_file,
            time_offset: real(imu.calibration.time_offset),
        },
        ..Default::default()
    };
    let terminal = output.terminal;
    let debug_fps = if terminal.debug {
        terminal.debug_fps
    } else {
        let _ = asts
            .defaults
            .evaluate_stats_only("output.terminal.debug_fps");
        let _ = asts.user.evaluate_stats_only("output.terminal.debug_fps");
        0
    };
    // Preserve the existing logger integration. The browser has no process environment.
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    if let Some(level) = terminal.log_level {
        unsafe {
            std::env::set_var("RUST_LOG", level);
        }
    }
    let mcap = output.mcap;
    anyhow::ensure!(
        mcap.file.is_none() || mcap.rosbag_dir.is_none(),
        "output.mcap.file and output.mcap.rosbag_dir are mutually exclusive"
    );
    anyhow::ensure!(
        mcap.rosbag_metadata.is_none() || mcap.rosbag_dir.is_none(),
        "output.mcap.rosbag_metadata cannot be combined with output.mcap.rosbag_dir; rosbag_dir always writes its own metadata.yaml"
    );
    anyhow::ensure!(
        mcap.rosbag_metadata.is_none() || mcap.file.is_some(),
        "output.mcap.rosbag_metadata requires output.mcap.file"
    );
    let mcap = McapConfig {
        file: mcap.file,
        rosbag_dir: mcap.rosbag_dir,
        rosbag_metadata: mcap.rosbag_metadata,
        cloud_topic: mcap.cloud_topic,
        odom_topic: mcap.odom_topic,
        path_topic: mcap.path_topic,
        tf_topic: mcap.tf_topic,
    };
    let filter = preprocessing.filter;
    let (min_x, max_x) = filter.bounding_box.x.bounds(f64::MIN, f64::MAX);
    let (min_y, max_y) = filter.bounding_box.y.bounds(f64::MIN, f64::MAX);
    let (min_z, max_z) = filter.bounding_box.z.bounds(f64::MIN, f64::MAX);
    let (min_dist, max_dist) = filter.distance.range.bounds(0.0, f64::MAX / 2.5);
    let (left, right) = filter.horizontal_fov.degrees.bounds(-180.0, 180.0);
    let (min_intensity, max_intensity) = filter.intensity.values.bounds(f64::MIN, f64::MAX);
    let passthrough_filter = PassthroughFilterParameters {
        min: Vector3::new(min_x, min_y, min_z).cast(),
        max: Vector3::new(max_x, max_y, max_z).cast(),
        invert_bounding_box: filter.bounding_box.invert,
        min_dist: min_dist as f32,
        max_dist: max_dist as f32,
        invert_distance: filter.distance.invert,
        fov_left: Angle::new::<degree>(left as f32).get::<radian>(),
        fov_right: Angle::new::<degree>(right as f32).get::<radian>(),
        forward: filter.horizontal_fov.forward,
        enable_horizontal_fov: filter.horizontal_fov.value,
        invert_fov: filter.horizontal_fov.invert,
        polygon: (!filter.polygon_2d.coords.is_empty()).then_some(filter.polygon_2d.coords),
        invert_polygon: filter.polygon_2d.invert,
        min_intensity: min_intensity as f32,
        max_intensity: max_intensity as f32,
        invert_intensity: filter.intensity.invert,
        filter_frame_transform: nalgebra::Isometry3::identity(),
    };
    let export_path = |value: String| match value.as_str() {
        "" => None,
        "auto" => Some(String::new()),
        _ => Some(value),
    };
    let final_config = LioConfig {
        sensor_drivers,
        rerun: RerunConfig {
            enabled: output.rerun.value,
            server_addr: output.rerun.server_addr,
            map: RerunMapConfig {
                enabled: output.rerun.map.value,
                publish_interval_sec: output.rerun.map.publish_interval,
                visual_scale: output.rerun.map.visual_scale,
            },
        },
        mcap,
        export: crate::ExportConfig {
            trajectory: export_path(output.trajectory),
            latency: export_path(output.latency),
            processing_time: export_path(output.processing_time),
        },
        offline,
        neighbor_max_range: real(registration.neighbors_max_range),
        map_resolution: real(map.resolution.value),
        map_capacity: map.capacity,
        map_multiresolution: map.resolution.r#type,
        dist_update_sensitivity: map.distribution_update_sensitivity,
        ekf_max_iterations: registration.max_iter,
        lidar_odom_cov: real(lidar.covariance),
        show_tui: terminal.ui,
        debug_fps,
        odom_qos: output.lio_odometry_msg_qos,
        odometry_rate: output.odometry_rate,
        path_qos: output.lio_path_msg_qos,
        imu_odom_qos: output.imu_odometry_msg_qos,
        imu_path_qos: output.imu_path_msg_qos,
        imu_params,
        base_link_frame: output.base_link.frame_id,
        base_to_lidar: nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::from(output.base_link.from_lidar_transform.translation),
            nalgebra::UnitQuaternion::from_matrix(&output.base_link.from_lidar_transform.rotation),
        )
        .cast(),
        base_to_lidar_tf_override: output.base_link.from_lidar_transform.use_tf_tree,
        cpu_only: parallelism.cpu_only,
        sync_node_params: SyncNodeParameters {
            wheel_qos: wheel.qos.reliability,
            wheel_src_topic: wheel.topic,
            cloud_src_topic: lidar.topic,
            cloud_qos: lidar.qos.reliability,
            cloud_subscriber_mode: lidar.subscriber_mode,
            imu_src_topic: imu.topic,
            imu_qos: imu.qos.reliability,
            imu_predictor_init_time_sec: real(imu.initial_setup),
            logger_prefix: output.logger_prefix,
            imu_filtered_qos: output.imu_ahrs_filtered_msg_qos,
            imu_filtered_topic: output.imu_ahrs_filtered_topic,
            synchronous_mode: false,
        },
        sync_params: SyncParameters {
            lidar_vendor: lidar.vendor,
            imu_lidar_tf_override: imu.from_lidar_transform.use_tf_tree && !imu.lidar_only,
            map_frame: output.global_frame,
            imu: imu_sync_params,
            downsample: DownsampleParameters {
                voxel_size: (preprocessing.downsample.query_resolution_factor
                    * map.resolution.value) as f32,
                bypass_output_downsample: preprocessing.downsample.bypass_output_downsample,
                point_skip: preprocessing.downsample.point_skip,
                backend: if parallelism.cpu_only {
                    PreprocessingBackend::Cpu
                } else {
                    preprocessing.downsample.backend
                },
            },
            passthrough_filter,
            global_zero_frame: filter.polygon_2d.global_zero_frame,
            initial_scan_accumulate: preprocessing.initial_scan_accumulate,
            wheel_imu_tf_override: wheel.to_imu_transform.use_tf_tree,
            wheel_to_imu_rotation: wheel.to_imu_transform.rotation.cast(),
            wheel_odom_cov: wheel.covariance.map(real),
            lidar_as_sync: lidar.as_sync,
            warning_debounce: real(terminal.warning_debounce),
            parallelism_decision_threshold_stddevs: real(parallelism.decision_threshold_stddevs),
            parallelism_decision_samples_per_mode: parallelism.decision_samples_per_mode,
            parallelism_execution_policy: parallelism.execution_mode,
        },
        registered_cloud_qos: Qos::Sensor(Reliability::Reliable),
        cloud_publisher_mode: output.publisher_mode,
        num_threads: parallelism.num_threads,
        pipelining: parallelism.pipelining,
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
    reject_renamed_keys(&asts.user.usage_vars().0)?;
    Ok((final_config, default_usage_info, user_usage_info))
}

/// Settings that moved, as `(old, new)`. An old name is an error rather than
/// the usual unknown-variable warning: a config still using one would quietly
/// stop writing its files.
const RENAMED_KEYS: &[(&str, &str)] = &[
    ("output.export.trajectory", "output.trajectory"),
    ("output.export.latency", "output.latency"),
    ("output.export.processing_time", "output.processing_time"),
];

fn reject_renamed_keys(assigned: &[String]) -> anyhow::Result<()> {
    let moved = RENAMED_KEYS
        .iter()
        .filter(|(old, _)| assigned.iter().any(|key| key == old))
        .map(|(old, new)| format!("'{old}' is now '{new}'"))
        .collect::<Vec<_>>();
    if !moved.is_empty() {
        return Err(anyhow!(
            "renamed setting{}: {}",
            if moved.len() == 1 { "" } else { "s" },
            moved.join(", ")
        ));
    }
    Ok(())
}

#[cfg(test)]
mod serde_integration_tests {
    use super::*;
    #[test]
    fn defaults_constrain_unannotated_overrides_and_preserve_unknown_key_tracking() {
        for source in [
            "lidar.driver.velodyne.data_port = 0\n",
            "lidar.driver.robosense.msop_port = 65536\n",
            "imu.driver.xsens.output_rate = 2001\n",
            "sensor_drivers.ouster.timestamp.startup_timeout = 0s\n",
            "output.rerun.map.visual_scale = 0.0\n",
            "parallelism.decision_samples_per_mode = 0\n",
            "parallelism.num_threads = -1\n",
            "imu.from_lidar_transform.translation = [1m, 2m]\n",
        ] {
            let defaults = ratslang::compile_code(include_str!("../default.rl")).unwrap();
            let user = ratslang::compile_code(source).unwrap();
            assert!(ast_to_config(defaults, user).is_err(), "{source}");
        }
        let defaults = ratslang::compile_code(include_str!("../default.rl")).unwrap();
        let user = ratslang::compile_code("output.odometry_rate = imu_and_lidar\nparallelism.num_threads = All\nunknown.typo = true\n").unwrap();
        let (config, (assigned, read)) = ast_to_config_with_user_usage(defaults, user).unwrap();
        assert_eq!(config.odometry_rate, OdometryRate::ImuAndLidar);
        assert_eq!(config.num_threads, 0);
        assert!(assigned.contains(&"unknown.typo".into()));
        assert!(!read.contains(&"unknown.typo".into()));
    }
}

//! `pelorus online raw` — run the native sensor drivers as a plain driver for
//! the Minot pub/sub network.
//!
//! Nothing from the odometry pipeline is involved. Measurements go from the
//! driver thread onto the very topics `pelorus online` subscribes to, so a raw
//! process on the sensor host and a full Pelorus elsewhere compose without any
//! extra configuration. In particular the IMU is published exactly as the
//! device reported it: `imu.ahrs_filtering.method` describes what the consumer
//! wants to do with the samples, and applying it here would hand every consumer
//! pre-filtered data it cannot undo.

use std::path::PathBuf;

use anyhow::{Result, bail};

use pelorus_core::{ImuDriverType, LidarDriverType, LioConfig, PointCloudMode, SensorDriverConfig};

#[cfg(all(
    feature = "mt_pubsub",
    any(
        feature = "velodyne-driver",
        feature = "xsens-driver",
        feature = "ouster-driver"
    )
))]
mod runner;

#[cfg(all(
    feature = "mt_pubsub",
    any(
        feature = "velodyne-driver",
        feature = "xsens-driver",
        feature = "ouster-driver"
    )
))]
pub use runner::run;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum RawSensor {
    /// Every native driver enabled in the configuration.
    All,
    /// Only the lidar driver.
    Lidar,
    /// Only the IMU driver.
    Imu,
}

impl RawSensor {
    fn wants_lidar(self) -> bool {
        matches!(self, Self::All | Self::Lidar)
    }
    fn wants_imu(self) -> bool {
        matches!(self, Self::All | Self::Imu)
    }
    #[allow(dead_code)]
    fn label(self) -> &'static str {
        match self {
            Self::All => "lidar and IMU",
            Self::Lidar => "lidar",
            Self::Imu => "IMU",
        }
    }
}

/// Command-line inputs for a raw run, separated from configuration.
#[derive(Debug, Clone)]
pub struct RawOptions {
    pub sensor: RawSensor,
    pub output_mcap: Option<PathBuf>,
    pub output_rosbag: Option<PathBuf>,
    pub output_rosbag_metadata: Option<PathBuf>,
    pub mcap_cloud_topic: Option<String>,
    pub mcap_imu_topic: Option<String>,
    pub local_only: bool,
    pub yes: bool,
}

/// What a raw run publishes, once configuration and the selected sensor have
/// been reconciled.
#[derive(Debug, Clone, PartialEq)]
pub struct RawPlan {
    pub drivers: SensorDriverConfig,
    pub lidar: bool,
    pub imu: bool,
    pub cloud_topic: String,
    pub imu_topic: String,
    pub cloud_mode: PointCloudMode,
}

/// Narrows the configured drivers down to the selected sensor.
///
/// The unselected driver is switched off rather than merely ignored, because
/// `DriverManager` decides what to launch from these fields alone: leaving one
/// set would open that sensor's socket and then publish nothing from it. On a
/// combined device such as the Ouster this is the difference between reading
/// one UDP stream and both of them.
pub fn plan_raw_run(config: &LioConfig, sensor: RawSensor) -> Result<RawPlan> {
    let mut drivers = config.sensor_drivers.clone();

    let lidar = sensor.wants_lidar() && drivers.lidar.driver_type != LidarDriverType::Disabled;
    let imu = sensor.wants_imu() && drivers.imu.driver_type != ImuDriverType::Disabled;

    if sensor == RawSensor::Lidar && !lidar {
        bail!(
            "`pelorus online raw lidar` needs a native lidar driver, but `lidar.driver.type` is Disabled."
        );
    }
    if sensor == RawSensor::Imu && !imu {
        bail!(
            "`pelorus online raw imu` needs a native IMU driver, but `imu.driver.type` is Disabled."
        );
    }
    if !lidar && !imu {
        bail!(
            "`pelorus online raw` needs at least one native driver, but both `lidar.driver.type` and `imu.driver.type` are Disabled."
        );
    }

    if !lidar {
        drivers.lidar.driver_type = LidarDriverType::Disabled;
    }
    if !imu {
        drivers.imu.driver_type = ImuDriverType::Disabled;
    }

    // The published point layout follows the driver type, so a consumer that
    // deskews by `lidar.vendor` only gets usable per-point timestamps when the
    // two agree. Catch the mismatch here rather than after a recording.
    #[cfg(any(
        feature = "velodyne-driver",
        feature = "xsens-driver",
        feature = "ouster-driver"
    ))]
    if lidar {
        prelio_io::validate_native_lidar_vendor(
            &drivers.lidar.driver_type,
            config.prelio_params.lidar_vendor,
        )?;
    }

    let cloud_topic = config.prelio_node_params.cloud_src_topic.clone();
    let imu_topic = config.prelio_node_params.imu_src_topic.clone();
    if lidar && cloud_topic.is_empty() {
        bail!("`lidar.topic` is empty, so there is no topic to publish raw scans on.");
    }
    if imu && imu_topic.is_empty() {
        bail!("`imu.topic` is empty, so there is no topic to publish raw IMU data on.");
    }

    Ok(RawPlan {
        drivers,
        lidar,
        imu,
        cloud_topic,
        imu_topic,
        cloud_mode: config.prelio_node_params.cloud_subscriber_mode,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pelorus_core::LidarVendor;

    fn config_with(lidar: LidarDriverType, imu: ImuDriverType) -> LioConfig {
        let mut config = LioConfig::default();
        config.sensor_drivers.lidar.driver_type = lidar;
        config.sensor_drivers.imu.driver_type = imu;
        config.prelio_node_params.cloud_src_topic = "/ouster/points".into();
        config.prelio_node_params.imu_src_topic = "/ouster/imu".into();
        config.prelio_params.lidar_vendor = LidarVendor::Ouster;
        config
    }

    #[test]
    fn imu_only_run_disables_the_lidar_driver() {
        let config = config_with(LidarDriverType::Ouster, ImuDriverType::Ouster);
        let plan = plan_raw_run(&config, RawSensor::Imu).expect("plan");
        assert!(plan.imu);
        assert!(!plan.lidar);
        assert_eq!(plan.drivers.lidar.driver_type, LidarDriverType::Disabled);
        assert_eq!(plan.drivers.imu.driver_type, ImuDriverType::Ouster);
        assert_eq!(plan.imu_topic, "/ouster/imu");
    }

    #[test]
    fn lidar_only_run_disables_the_imu_driver() {
        let config = config_with(LidarDriverType::Ouster, ImuDriverType::Xsens);
        let plan = plan_raw_run(&config, RawSensor::Lidar).expect("plan");
        assert!(plan.lidar);
        assert!(!plan.imu);
        assert_eq!(plan.drivers.imu.driver_type, ImuDriverType::Disabled);
        assert_eq!(plan.cloud_topic, "/ouster/points");
    }

    #[test]
    fn all_keeps_every_configured_driver() {
        let config = config_with(LidarDriverType::Ouster, ImuDriverType::Xsens);
        let plan = plan_raw_run(&config, RawSensor::All).expect("plan");
        assert!(plan.lidar && plan.imu);
        assert_eq!(plan.drivers.lidar.driver_type, LidarDriverType::Ouster);
        assert_eq!(plan.drivers.imu.driver_type, ImuDriverType::Xsens);
    }

    #[test]
    fn all_tolerates_a_single_configured_driver() {
        let config = config_with(LidarDriverType::Disabled, ImuDriverType::Xsens);
        let plan = plan_raw_run(&config, RawSensor::All).expect("plan");
        assert!(plan.imu);
        assert!(!plan.lidar);
    }

    #[test]
    fn selecting_a_disabled_driver_is_an_error() {
        let config = config_with(LidarDriverType::Disabled, ImuDriverType::Xsens);
        assert!(plan_raw_run(&config, RawSensor::Lidar).is_err());

        let config = config_with(LidarDriverType::Velodyne, ImuDriverType::Disabled);
        assert!(plan_raw_run(&config, RawSensor::Imu).is_err());
    }

    #[test]
    fn no_configured_driver_at_all_is_an_error() {
        let config = config_with(LidarDriverType::Disabled, ImuDriverType::Disabled);
        assert!(plan_raw_run(&config, RawSensor::All).is_err());
    }

    #[cfg(any(
        feature = "velodyne-driver",
        feature = "xsens-driver",
        feature = "ouster-driver"
    ))]
    #[test]
    fn lidar_vendor_must_match_the_driver() {
        let mut config = config_with(LidarDriverType::Ouster, ImuDriverType::Disabled);
        config.prelio_params.lidar_vendor = LidarVendor::Velodyne;
        assert!(plan_raw_run(&config, RawSensor::Lidar).is_err());
    }

    #[cfg(any(
        feature = "velodyne-driver",
        feature = "xsens-driver",
        feature = "ouster-driver"
    ))]
    #[test]
    fn imu_only_run_ignores_a_mismatched_lidar_vendor() {
        let mut config = config_with(LidarDriverType::Ouster, ImuDriverType::Ouster);
        config.prelio_params.lidar_vendor = LidarVendor::Velodyne;
        assert!(plan_raw_run(&config, RawSensor::Imu).is_ok());
    }

    #[test]
    fn an_empty_input_topic_is_rejected() {
        let mut config = config_with(LidarDriverType::Disabled, ImuDriverType::Xsens);
        config.prelio_node_params.imu_src_topic = String::new();
        assert!(plan_raw_run(&config, RawSensor::Imu).is_err());
    }
}

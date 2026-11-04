pub use pelorus_core::Stamp;
use pelorus_core::{ImuDriverType, ImuMsg, LidarDriverType, SensorDriverConfig};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[cfg(feature = "ouster-driver")]
mod ouster;
#[cfg(feature = "robosense-driver")]
mod robosense;
#[cfg(feature = "velodyne-driver")]
mod velodyne;
#[cfg(feature = "xsens-driver")]
mod xsens;

#[derive(Debug, Clone, PartialEq)]
pub struct RawLidarPoint {
    pub xyz: [f32; 3],
    pub intensity: f32,
    pub time_offset_secs: f64,
}
#[derive(Debug, Clone, PartialEq)]
pub struct ReceiveMetadata {
    pub received_at: std::time::SystemTime,
    pub sender: Option<std::net::IpAddr>,
    pub packets_received: u64,
    pub packets_dropped: u64,
}
#[derive(Debug, Clone, PartialEq)]
pub struct RawLidarScan {
    pub frame_id: String,
    /// Validated Unix UTC scan timestamp.
    pub stamp: Stamp,
    pub points: Vec<RawLidarPoint>,
    pub receive: ReceiveMetadata,
}

pub trait SensorIngress: Send + Sync + 'static {
    fn push_lidar(&self, scan: RawLidarScan) -> Result<(), DriverError>;
    fn push_imu(&self, imu: ImuMsg) -> Result<(), DriverError>;

    /// Called when a native driver terminates with an error. Implementations
    /// should stop their consumer pipeline.
    fn driver_failed(&self, error: &DriverError);
}
pub trait SensorOutputSink: Send + Sync + 'static {
    fn mirror_lidar(&self, scan: &RawLidarScan) -> Result<(), DriverError>;
    fn mirror_imu(&self, imu: &ImuMsg) -> Result<(), DriverError>;
}

#[cfg(feature = "native-drivers")]
struct DriverStartGate {
    ready: std::sync::atomic::AtomicUsize,
    participants: usize,
}

#[cfg(feature = "native-drivers")]
impl DriverStartGate {
    fn wait(&self, stop: &AtomicBool) -> Result<(), DriverError> {
        self.ready.fetch_add(1, Ordering::AcqRel);
        while self.ready.load(Ordering::Acquire) < self.participants {
            if stop.load(Ordering::Acquire) {
                return Err(DriverError::Driver("driver startup cancelled".into()));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }
}

type DriverThread = JoinHandle<Result<(), DriverError>>;
type DriverLauncher = Box<
    dyn FnOnce(
        Arc<dyn SensorIngress>,
        Option<Arc<dyn SensorOutputSink>>,
        Arc<AtomicBool>,
    ) -> Result<DriverThread, DriverError>,
>;

fn launch_drivers(
    launchers: Vec<DriverLauncher>,
    ingress: Arc<dyn SensorIngress>,
    sink: Option<Arc<dyn SensorOutputSink>>,
    stop: Arc<AtomicBool>,
) -> Result<Vec<DriverThread>, DriverError> {
    let mut threads = Vec::with_capacity(launchers.len());
    for launch in launchers {
        threads.push(launch(
            Arc::clone(&ingress),
            sink.clone(),
            Arc::clone(&stop),
        )?);
    }
    Ok(threads)
}

#[derive(Debug, thiserror::Error)]
pub enum DriverError {
    #[error("{0} driver is enabled but Pelorus was built without feature `{1}")]
    FeatureUnavailable(&'static str, &'static str),
    #[error("sensor timestamp differs from host clock by {skew:?}, exceeding {maximum:?}")]
    ClockSkew { skew: Duration, maximum: Duration },
    #[error("driver startup timed out waiting for a valid sensor timestamp")]
    StartupTimeout,
    #[error("driver failed: {0}")]
    Driver(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Reconstruct time-of-hour against the nearest host UTC hour, including rollover.
pub fn velodyne_host_hour_timestamp(
    host: Stamp,
    micros_since_hour: u32,
    max_skew: Duration,
) -> Result<Stamp, DriverError> {
    const HOUR_NANOS: i64 = 3_600_000_000_000;
    let within_hour = i64::from(micros_since_hour) * 1_000;
    if within_hour >= HOUR_NANOS {
        return Err(DriverError::Driver("invalid Velodyne time-of-hour".into()));
    }
    let hour = host.0.div_euclid(HOUR_NANOS) * HOUR_NANOS;
    let timestamp = [-HOUR_NANOS, 0, HOUR_NANOS]
        .into_iter()
        .map(|d| Stamp(hour + d + within_hour))
        .min_by_key(|candidate| (candidate.0 - host.0).abs())
        .unwrap();
    validate_timestamp(timestamp, host, max_skew)?;
    Ok(timestamp)
}

/// The host clock as a stamp.
pub fn host_stamp(time: SystemTime) -> Result<Stamp, DriverError> {
    let since_epoch = time
        .duration_since(UNIX_EPOCH)
        .map_err(|error| DriverError::Driver(error.to_string()))?;
    i64::try_from(since_epoch.as_nanos())
        .map(Stamp)
        .map_err(|error| DriverError::Driver(error.to_string()))
}

pub fn validate_timestamp(
    sensor: Stamp,
    host: Stamp,
    maximum: Duration,
) -> Result<(), DriverError> {
    if sensor.0 < 0 {
        return Err(DriverError::Driver("invalid UTC timestamp".into()));
    }
    let skew = Duration::from_nanos((sensor.0 - host.0).unsigned_abs());
    if skew > maximum {
        Err(DriverError::ClockSkew { skew, maximum })
    } else {
        Ok(())
    }
}

pub struct DriverManager {
    stop: Arc<AtomicBool>,
    threads: Vec<DriverThread>,
}
impl DriverManager {
    pub fn start(
        config: &SensorDriverConfig,
        ingress: Arc<dyn SensorIngress>,
        sink: Option<Arc<dyn SensorOutputSink>>,
    ) -> Result<Self, DriverError> {
        if config.lidar.driver_type == LidarDriverType::Velodyne
            && !cfg!(feature = "velodyne-driver")
        {
            return Err(DriverError::FeatureUnavailable(
                "Velodyne",
                "velodyne-driver",
            ));
        }
        if config.lidar.driver_type == LidarDriverType::Robosense
            && !cfg!(feature = "robosense-driver")
        {
            return Err(DriverError::FeatureUnavailable(
                "RoboSense",
                "robosense-driver",
            ));
        }
        if config.imu.driver_type == ImuDriverType::Xsens && !cfg!(feature = "xsens-driver") {
            return Err(DriverError::FeatureUnavailable("Xsens", "xsens-driver"));
        }
        if (config.lidar.driver_type == LidarDriverType::Ouster
            || config.imu.driver_type == ImuDriverType::Ouster)
            && !cfg!(feature = "ouster-driver")
        {
            return Err(DriverError::FeatureUnavailable("Ouster", "ouster-driver"));
        }
        let stop = Arc::new(AtomicBool::new(false));
        #[cfg(feature = "native-drivers")]
        let enabled_driver_count = usize::from(
            cfg!(feature = "velodyne-driver")
                && config.lidar.driver_type == LidarDriverType::Velodyne,
        ) + usize::from(
            cfg!(feature = "xsens-driver") && config.imu.driver_type == ImuDriverType::Xsens,
        ) + usize::from(
            cfg!(feature = "ouster-driver")
                && (config.lidar.driver_type == LidarDriverType::Ouster
                    || config.imu.driver_type == ImuDriverType::Ouster),
        ) + usize::from(
            cfg!(feature = "robosense-driver")
                && config.lidar.driver_type == LidarDriverType::Robosense,
        );
        #[cfg(feature = "native-drivers")]
        let startup_barrier = (enabled_driver_count > 1).then(|| {
            Arc::new(DriverStartGate {
                ready: std::sync::atomic::AtomicUsize::new(0),
                participants: enabled_driver_count,
            })
        });
        // `unused_mut` when the crate is built without any driver feature.
        #[allow(unused_mut)]
        let mut launchers: Vec<DriverLauncher> = Vec::new();
        #[cfg(feature = "velodyne-driver")]
        if config.lidar.driver_type == LidarDriverType::Velodyne {
            let cfg = config.lidar.clone();
            let startup_barrier = startup_barrier.clone();
            launchers.push(Box::new(move |ingress, sink, stop| {
                std::thread::Builder::new()
                    .name("pelorus-velodyne".into())
                    .spawn(move || {
                        let result = velodyne::run(
                            cfg,
                            Arc::clone(&ingress),
                            sink,
                            Arc::clone(&stop),
                            startup_barrier,
                        );
                        if let Err(error) = &result {
                            stop.store(true, Ordering::Release);
                            ingress.driver_failed(error);
                        }
                        result
                    })
                    .map_err(DriverError::Io)
            }));
        }
        #[cfg(feature = "xsens-driver")]
        if config.imu.driver_type == ImuDriverType::Xsens {
            let cfg = config.imu.clone();
            let startup_barrier = startup_barrier.clone();
            launchers.push(Box::new(move |ingress, sink, stop| {
                std::thread::Builder::new()
                    .name("pelorus-xsens".into())
                    .spawn(move || {
                        let result = xsens::run(
                            cfg,
                            Arc::clone(&ingress),
                            sink,
                            Arc::clone(&stop),
                            startup_barrier,
                        );
                        if let Err(error) = &result {
                            stop.store(true, Ordering::Release);
                            ingress.driver_failed(error);
                        }
                        result
                    })
                    .map_err(DriverError::Io)
            }));
        }
        #[cfg(feature = "ouster-driver")]
        if config.lidar.driver_type == LidarDriverType::Ouster
            || config.imu.driver_type == ImuDriverType::Ouster
        {
            let cfg = config.ouster.clone();
            let receive_lidar = config.lidar.driver_type == LidarDriverType::Ouster;
            let receive_imu = config.imu.driver_type == ImuDriverType::Ouster;
            let startup_barrier = startup_barrier.clone();
            launchers.push(Box::new(move |ingress, sink, stop| {
                std::thread::Builder::new()
                    .name("pelorus-ouster".into())
                    .spawn(move || {
                        let result = ouster::run(
                            cfg,
                            receive_lidar,
                            receive_imu,
                            Arc::clone(&ingress),
                            sink,
                            Arc::clone(&stop),
                            startup_barrier,
                        );
                        if let Err(error) = &result {
                            stop.store(true, Ordering::Release);
                            ingress.driver_failed(error);
                        }
                        result
                    })
                    .map_err(DriverError::Io)
            }));
        }
        #[cfg(feature = "robosense-driver")]
        if config.lidar.driver_type == LidarDriverType::Robosense {
            let cfg = config.lidar.robosense.clone();
            let startup_barrier = startup_barrier.clone();
            launchers.push(Box::new(move |ingress, sink, stop| {
                std::thread::Builder::new()
                    .name("pelorus-robosense".into())
                    .spawn(move || {
                        let result = robosense::run(
                            cfg,
                            Arc::clone(&ingress),
                            sink,
                            Arc::clone(&stop),
                            startup_barrier,
                        );
                        if let Err(error) = &result {
                            stop.store(true, Ordering::Release);
                            ingress.driver_failed(error);
                        }
                        result
                    })
                    .map_err(DriverError::Io)
            }));
        }
        let threads = launch_drivers(launchers, ingress, sink, Arc::clone(&stop))?;
        Ok(Self { stop, threads })
    }
    pub fn shutdown(mut self) -> Result<(), DriverError> {
        self.stop.store(true, Ordering::Release);
        for thread in self.threads.drain(..) {
            thread
                .join()
                .map_err(|_| DriverError::Driver("driver panicked".into()))??;
        }
        Ok(())
    }
}
impl Drop for DriverManager {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        while let Some(t) = self.threads.pop() {
            drop(t.join());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn velodyne_time_of_hour_takes_the_hour_closest_to_the_host() {
        const HOUR: i64 = 3_600_000_000_000;
        let host = Stamp(470_000 * HOUR + 1_000_000_000);
        // 59 min 59 s into an hour, read one second after the hour turned.
        let stamp =
            velodyne_host_hour_timestamp(host, 3_599_000_000, Duration::from_secs(5)).unwrap();
        assert_eq!(stamp, Stamp(470_000 * HOUR - 1_000_000_000));
        let stamp = velodyne_host_hour_timestamp(host, 500, Duration::from_secs(5)).unwrap();
        assert_eq!(stamp, Stamp(470_000 * HOUR + 500_000));
    }

    #[test]
    fn host_skew_is_measured_in_nanoseconds() {
        let host = Stamp(1_700_000_000_000_000_000);
        let maximum = Duration::from_millis(10);
        assert!(validate_timestamp(Stamp(host.0 + 10_000_000), host, maximum).is_ok());
        assert!(matches!(
            validate_timestamp(Stamp(host.0 - 10_000_001), host, maximum),
            Err(DriverError::ClockSkew { .. })
        ));
    }
}

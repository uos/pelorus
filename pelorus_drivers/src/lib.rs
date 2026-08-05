//! Native sensor driver boundaries, independent of UDP, serial, ROS, or Zenoh.
use pelorus_core::{ImuDriverType, ImuMsg, LidarDriverType, SensorDriverConfig};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::JoinHandle,
    time::Duration,
};

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
    /// Validated Unix UTC scan timestamp in seconds.
    pub stamp: f64,
    pub points: Vec<RawLidarPoint>,
    pub receive: ReceiveMetadata,
}

pub trait SensorIngress: Send + Sync + 'static {
    fn push_lidar(&self, scan: RawLidarScan) -> Result<(), DriverError>;
    fn push_imu(&self, imu: ImuMsg) -> Result<(), DriverError>;

    /// Called when a native driver terminates with an error. Implementations
    /// should stop their consumer pipeline; continuing with a missing or
    /// invalid sensor stream is not supported.
    fn driver_failed(&self, _error: &DriverError) {}
}
pub trait SensorOutputSink: Send + Sync + 'static {
    fn mirror_lidar(&self, scan: &RawLidarScan) -> Result<(), DriverError>;
    fn mirror_imu(&self, imu: &ImuMsg) -> Result<(), DriverError>;
}

struct DriverStartGate {
    ready: AtomicUsize,
    participants: usize,
}

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
    host_unix_secs: f64,
    micros_since_hour: u32,
    max_skew: Duration,
) -> Result<f64, DriverError> {
    let within_hour = micros_since_hour as f64 / 1_000_000.0;
    if within_hour >= 3600.0 {
        return Err(DriverError::Driver("invalid Velodyne time-of-hour".into()));
    }
    let hour = (host_unix_secs / 3600.0).floor() * 3600.0;
    let timestamp = [-3600.0, 0.0, 3600.0]
        .into_iter()
        .map(|d| hour + d + within_hour)
        .min_by(|a, b| {
            (a - host_unix_secs)
                .abs()
                .total_cmp(&(b - host_unix_secs).abs())
        })
        .unwrap();
    validate_timestamp(timestamp, host_unix_secs, max_skew)?;
    Ok(timestamp)
}
pub fn validate_timestamp(
    sensor_unix_secs: f64,
    host_unix_secs: f64,
    maximum: Duration,
) -> Result<(), DriverError> {
    if !sensor_unix_secs.is_finite() || sensor_unix_secs < 0.0 {
        return Err(DriverError::Driver("invalid UTC timestamp".into()));
    }
    let skew = Duration::from_secs_f64((sensor_unix_secs - host_unix_secs).abs());
    if skew > maximum {
        Err(DriverError::ClockSkew { skew, maximum })
    } else {
        Ok(())
    }
}

pub struct DriverManager {
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<Result<(), DriverError>>>,
}
impl DriverManager {
    pub fn start(
        config: &SensorDriverConfig,
        _ingress: Arc<dyn SensorIngress>,
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
        if config.imu.driver_type == ImuDriverType::Xsens && !cfg!(feature = "xsens-driver") {
            return Err(DriverError::FeatureUnavailable("Xsens", "xsens-driver"));
        }
        let stop = Arc::new(AtomicBool::new(false));
        let enabled_driver_count =
            usize::from(config.lidar.driver_type == LidarDriverType::Velodyne)
                + usize::from(config.imu.driver_type == ImuDriverType::Xsens);
        let startup_barrier = (enabled_driver_count > 1).then(|| {
            Arc::new(DriverStartGate {
                ready: AtomicUsize::new(0),
                participants: enabled_driver_count,
            })
        });
        #[allow(unused_mut)]
        let mut threads = Vec::new();
        #[cfg(feature = "velodyne-driver")]
        if config.lidar.driver_type == LidarDriverType::Velodyne {
            let (cfg, ingress, sink, stop, startup_barrier) = (
                config.lidar.clone(),
                Arc::clone(&_ingress),
                sink.clone(),
                Arc::clone(&stop),
                startup_barrier.clone(),
            );
            threads.push(
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
                    .map_err(DriverError::Io)?,
            );
        }
        #[cfg(feature = "xsens-driver")]
        if config.imu.driver_type == ImuDriverType::Xsens {
            let (cfg, ingress, sink, stop, startup_barrier) = (
                config.imu.clone(),
                Arc::clone(&_ingress),
                sink.clone(),
                Arc::clone(&stop),
                startup_barrier.clone(),
            );
            threads.push(
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
                    .map_err(DriverError::Io)?,
            );
        }
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
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nearest_hour_rolls_forward() {
        assert!(
            (velodyne_host_hour_timestamp(3599.9, 100_000, Duration::from_secs(2)).unwrap()
                - 3600.1)
                .abs()
                < 1e-9
        );
    }
    #[test]
    fn nearest_hour_rolls_backward() {
        assert!(
            (velodyne_host_hour_timestamp(3600.1, 3_599_900_000, Duration::from_secs(2)).unwrap()
                - 3599.9)
                .abs()
                < 1e-9
        );
    }
    #[test]
    fn rejects_skew() {
        assert!(matches!(
            validate_timestamp(10.021, 10.0, Duration::from_millis(20)),
            Err(DriverError::ClockSkew { .. })
        ));
    }

    #[test]
    fn accepts_skew_at_hardware_sync_limit() {
        validate_timestamp(10.020, 10.0, Duration::from_millis(20)).unwrap();
    }
}

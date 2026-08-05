use pelorus_core::{ImuDriverType, LidarDriverType, SensorDriverConfig};
use pelorus_drivers::{DriverError, DriverManager, RawLidarScan, SensorIngress};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

#[derive(Default)]
struct Probe {
    scans: AtomicU64,
    points: AtomicU64,
    imus: AtomicU64,
}
impl SensorIngress for Probe {
    fn push_lidar(&self, scan: RawLidarScan) -> Result<(), DriverError> {
        let scan_index = self.scans.fetch_add(1, Ordering::Relaxed);
        if scan_index < 3 {
            let first = scan.points.first().map(|p| p.time_offset_secs);
            let last = scan.points.last().map(|p| p.time_offset_secs);
            eprintln!(
                "lidar[{scan_index}] stamp={:.6} offsets={first:?}..{last:?}",
                scan.stamp
            );
        }
        self.points
            .fetch_add(scan.points.len() as u64, Ordering::Relaxed);
        Ok(())
    }
    fn push_imu(&self, imu: pelorus_core::ImuMsg) -> Result<(), DriverError> {
        let imu_index = self.imus.fetch_add(1, Ordering::Relaxed);
        if imu_index < 3 {
            let stamp = imu.stamp.sec as f64 + imu.stamp.nanosec as f64 / 1e9;
            eprintln!(
                "imu[{imu_index}] stamp={stamp:.6} acc={:?} gyro={:?}",
                imu.linear_acceleration, imu.angular_velocity
            );
        }
        Ok(())
    }
}
fn main() -> Result<(), DriverError> {
    let probe = Arc::new(Probe::default());
    let mut cfg = SensorDriverConfig::default();
    cfg.lidar.driver_type = LidarDriverType::Velodyne;
    cfg.lidar.velodyne.device_ip = Some("192.168.0.50".into());
    cfg.lidar.velodyne.timestamp_mode = pelorus_core::VelodyneTimestampMode::Host;
    cfg.imu.driver_type = ImuDriverType::Xsens;
    cfg.imu.xsens.port = "/dev/serial/by-id/usb-Xsens_MTi-G-700_GPS_INS_0778263D-if01-port0".into();
    cfg.imu.xsens.timestamp_mode = pelorus_core::XsensTimestampMode::Host;
    if std::env::var_os("PROBE_XSENS_ONLY").is_some() {
        cfg.lidar.driver_type = LidarDriverType::Disabled;
    }
    let manager = DriverManager::start(&cfg, probe.clone(), None)?;
    std::thread::sleep(std::time::Duration::from_secs(8));
    manager.shutdown()?;
    println!(
        "scans={} points={} imus={}",
        probe.scans.load(Ordering::Relaxed),
        probe.points.load(Ordering::Relaxed),
        probe.imus.load(Ordering::Relaxed)
    );
    Ok(())
}

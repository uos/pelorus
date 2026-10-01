use crate::*;
use chrono::{NaiveDate, TimeZone, Utc};
use pelorus_core::{LidarDriverConfig, VelodyneModel, VelodyneTimestampMode};
use std::{
    net::{IpAddr, UdpSocket},
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use velodyne_lidar::{
    Config, DataPacket, ReturnMode,
    types::{format::FormatKind, point::Point},
};

fn parse_rmc(nmea: &[u8]) -> Option<f64> {
    let s = std::str::from_utf8(nmea).ok()?;
    let start = s.find("$GPRMC").or_else(|| s.find("$GNRMC"))?;
    let fields: Vec<_> = s[start..].split(',').collect();
    if fields.len() < 10 || fields[2] != "A" {
        return None;
    }
    let t = fields[1];
    let d = fields[9];
    if t.len() < 6 || d.len() != 6 {
        return None;
    }
    let hour = t[0..2].parse().ok()?;
    let minute = t[2..4].parse().ok()?;
    let sec: f64 = t[4..].parse().ok()?;
    let day = d[0..2].parse().ok()?;
    let month = d[2..4].parse().ok()?;
    let yy: i32 = d[4..6].parse().ok()?;
    let date = NaiveDate::from_ymd_opt(if yy >= 70 { 1900 + yy } else { 2000 + yy }, month, day)?;
    let whole = sec.floor() as u32;
    let dt = date.and_hms_nano_opt(hour, minute, whole, ((sec - whole as f64) * 1e9) as u32)?;
    Some(Utc.from_utc_datetime(&dt).timestamp_nanos_opt()? as f64 / 1e9)
}

/// Convert a return's packet-local firing time into the scan-relative seconds
/// expected by Pelorus' Velodyne point type.  `packet_stamp` and `scan_stamp`
/// use the selected (sensor or host) clock. `firing_offset` is always relative
/// to the packet timestamp supplied by the Velodyne packet decoder.
fn scan_relative_time(packet_stamp: f64, scan_stamp: f64, firing_offset: f64) -> f64 {
    (packet_stamp - scan_stamp) + firing_offset
}

/// `velodyne-lidar` uses +Y at azimuth zero and +X to the right.  Match the
/// ROS Velodyne/REP-103 sensor frame used by existing Pelorus extrinsics:
/// +X forward, +Y left, +Z up.
fn decoder_xyz_to_ros(xyz: [f32; 3]) -> [f32; 3] {
    [xyz[1], -xyz[0], xyz[2]]
}

pub fn run(
    cfg: LidarDriverConfig,
    ingress: Arc<dyn SensorIngress>,
    sink: Option<Arc<dyn SensorOutputSink>>,
    stop: Arc<AtomicBool>,
    startup_barrier: Option<Arc<DriverStartGate>>,
) -> Result<(), DriverError> {
    let c = &cfg.velodyne;
    let socket = UdpSocket::bind((c.bind_address.as_str(), c.data_port))?;
    socket.set_read_timeout(Some(Duration::from_millis(200)))?;
    let position = UdpSocket::bind((c.bind_address.as_str(), c.position_port))?;
    position.set_nonblocking(true)?;
    let mut pos_bytes = [0u8; 512];
    let mut utc_anchor: Option<(f64, u32)> = None;
    let expected: Option<IpAddr> = c
        .device_ip
        .as_deref()
        .map(str::parse)
        .transpose()
        .map_err(|e| DriverError::Driver(format!("invalid device IP: {e}")))?;
    let started = Instant::now();
    let timeout = Duration::from_secs_f64(c.startup_timeout_secs);
    let max_skew = Duration::from_secs_f64(c.max_host_skew_secs);
    let mut bytes = [0u8; 1206];
    let mut points = Vec::with_capacity(100_000);
    let mut last_az = None;
    let mut packets = 0u64;
    let mut dropped = 0u64;
    let mut scan_stamp = 0.0;
    if let Some(barrier) = startup_barrier {
        barrier.wait(&stop)?;
    }
    while !stop.load(Ordering::Acquire) {
        while let Ok((512, from)) = position.recv_from(&mut pos_bytes) {
            if expected.is_none_or(|ip| ip == from.ip())
                && let Ok(p) = velodyne_lidar::PositionPacket::from_slice(&pos_bytes)
                && let Some(epoch) = parse_rmc(&p.nmea)
            {
                utc_anchor = Some((epoch, p.toh));
            }
        }
        let (n, from) = match socket.recv_from(&mut bytes) {
            Ok(v) => v,
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                if packets == 0 && started.elapsed() > timeout {
                    return Err(DriverError::StartupTimeout);
                }
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        if n != 1206 || expected.is_some_and(|ip| ip != from.ip()) {
            dropped += 1;
            continue;
        }
        let packet = DataPacket::from_slice(&bytes)
            .map_err(|e| DriverError::Driver(format!("Velodyne packet: {e}")))?;
        let model_ok = matches!(
            (&c.model, packet.product_id),
            (VelodyneModel::Vlp32c, velodyne_lidar::ProductID::VLP32C)
                | (
                    VelodyneModel::Vlp16 | VelodyneModel::PuckLite | VelodyneModel::PuckHiRes,
                    velodyne_lidar::ProductID::VLP16
                )
        );
        if !model_ok {
            return Err(DriverError::Driver(format!(
                "configured model {:?} does not match packet product {:?}",
                c.model, packet.product_id
            )));
        }
        let host = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| DriverError::Driver(e.to_string()))?
            .as_secs_f64();
        let stamp = if c.timestamp_mode == VelodyneTimestampMode::Host {
            host
        } else if let Some((epoch, toh)) = utc_anchor {
            let mut delta = (packet.toh as f64 - toh as f64) / 1e6;
            if delta > 1800.0 {
                delta -= 3600.0
            } else if delta < -1800.0 {
                delta += 3600.0
            }
            let value = epoch + delta;
            validate_timestamp(value, host, max_skew)?;
            value
        } else if c.allow_host_hour_anchor {
            velodyne_host_hour_timestamp(host, packet.toh, max_skew)?
        } else {
            if started.elapsed() > timeout {
                return Err(DriverError::StartupTimeout);
            }
            continue;
        };
        if points.is_empty() {
            scan_stamp = stamp;
        }
        let config = match (&c.model, packet.return_mode) {
            (VelodyneModel::Vlp32c, ReturnMode::Strongest) => Config::new_vlp_32c_strongest(),
            (VelodyneModel::Vlp32c, ReturnMode::Last) => Config::new_vlp_32c_last(),
            (VelodyneModel::Vlp32c, ReturnMode::Dual) => Config::new_vlp_32c_dual(),
            (_, ReturnMode::Strongest) => Config::new_vlp_16_strongest(),
            (_, ReturnMode::Last) => Config::new_vlp_16_last(),
            (_, ReturnMode::Dual) => Config::new_vlp_16_dual(),
        };
        let kind = config
            .try_into_kind()
            .map_err(|_| DriverError::Driver("invalid Velodyne calibration".into()))?;
        let packet_azimuth = packet.blocks[0].azimuth_radians();
        let shifted_azimuth = (packet_azimuth - c.cut_angle_rad).rem_euclid(std::f64::consts::TAU);
        let crossed = last_az.is_some_and(|last: f64| last > 5.5 && shifted_azimuth < 0.8);
        if crossed && !points.is_empty() {
            let scan = RawLidarScan {
                frame_id: c.frame_id.clone(),
                stamp: scan_stamp,
                points: std::mem::take(&mut points),
                receive: ReceiveMetadata {
                    received_at: SystemTime::now(),
                    sender: Some(from.ip()),
                    packets_received: packets,
                    packets_dropped: dropped,
                },
            };
            if let Some(sink) = sink.as_ref() {
                sink.mirror_lidar(&scan)?
            }
            ingress.push_lidar(scan)?;
            scan_stamp = stamp;
        }
        last_az = Some(shifted_azimuth);
        let packet_points: Vec<Point> = match kind {
            FormatKind::Single16(x) => packet
                .firing_xyz_iter_s16(&x)
                .flat_map(|f| f.points.map(Point::from))
                .collect(),
            FormatKind::Single32(x) => packet
                .firing_xyz_iter_s32(&x)
                .flat_map(|f| f.points.map(Point::from))
                .collect(),
            FormatKind::Dual16(x) => packet
                .firing_xyz_iter_d16(&x)
                .flat_map(|f| f.points.map(Point::from))
                .collect(),
            FormatKind::Dual32(x) => packet
                .firing_xyz_iter_d32(&x)
                .flat_map(|f| f.points.map(Point::from))
                .collect(),
        };
        for p in packet_points {
            let add = |m: &velodyne_lidar::types::measurements::Measurement,
                       pts: &mut Vec<RawLidarPoint>| {
                if m.distance.as_meters() > 0.0 {
                    pts.push(RawLidarPoint {
                        xyz: decoder_xyz_to_ros(m.xyz.map(|v| v.as_meters() as f32)),
                        intensity: m.intensity as f32 / 255.0,
                        time_offset_secs: scan_relative_time(
                            stamp,
                            scan_stamp,
                            p.time().as_secs_f64() - packet.toh().as_secs_f64(),
                        ),
                    })
                }
            };
            match &p {
                Point::Single(v) => add(&v.measurement, &mut points),
                Point::Dual(v) => {
                    add(v.measurement_strongest(), &mut points);
                    if v.measurement_last() != v.measurement_strongest() {
                        add(v.measurement_last(), &mut points)
                    }
                }
            }
        }
        packets += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{decoder_xyz_to_ros, scan_relative_time};

    #[test]
    fn decoder_axes_are_converted_to_ros_rep_103() {
        assert_eq!(decoder_xyz_to_ros([0.0, 2.0, 3.0]), [2.0, -0.0, 3.0]);
        assert_eq!(decoder_xyz_to_ros([2.0, 0.0, 3.0]), [0.0, -2.0, 3.0]);
    }

    #[test]
    fn point_time_accumulates_across_packets_in_a_scan() {
        let scan_stamp = 1_700_000_000.0;
        let packet_stamp = scan_stamp + 0.037;
        let point_time = scan_relative_time(packet_stamp, scan_stamp, 0.000_421);
        assert!((point_time - 0.037_421).abs() < 1e-7);
    }
}

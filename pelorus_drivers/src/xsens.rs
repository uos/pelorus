use crate::*;
use chrono::{NaiveDate, TimeZone, Utc};
use pelorus_core::{
    CoordinateSystem, ImuDriverConfig, ImuMsg, XsensTimestampMode, time_msg_from_secs,
};
use std::{
    io::Read,
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use xsens_mti::{
    decoder::Decoder,
    message::{MessageDecode, MessageExt},
    messages::MTData2,
    mtdata2::UtcTime,
    wire::{DataType, Precision},
};

fn vec3(payload: &[u8], precision: Precision) -> Result<[f64; 3], DriverError> {
    match precision {
        Precision::Float32 if payload.len() == 12 => {
            Ok([0, 4, 8].map(|i| f32::from_be_bytes(payload[i..i + 4].try_into().unwrap()) as f64))
        }
        Precision::Float64 if payload.len() == 24 => {
            Ok([0, 8, 16].map(|i| f64::from_be_bytes(payload[i..i + 8].try_into().unwrap())))
        }
        _ => Err(DriverError::Driver(
            "unsupported Xsens vector precision".into(),
        )),
    }
}
fn quat(payload: &[u8], precision: Precision) -> Result<[f64; 4], DriverError> {
    match precision {
        Precision::Float32 if payload.len() == 16 => {
            Ok([0, 4, 8, 12]
                .map(|i| f32::from_be_bytes(payload[i..i + 4].try_into().unwrap()) as f64))
        }
        Precision::Float64 if payload.len() == 32 => {
            Ok([0, 8, 16, 24].map(|i| f64::from_be_bytes(payload[i..i + 8].try_into().unwrap())))
        }
        _ => Err(DriverError::Driver(
            "unsupported Xsens quaternion precision".into(),
        )),
    }
}
fn utc_secs(v: UtcTime) -> Result<f64, DriverError> {
    if !v.flags.utc_valid() {
        return Err(DriverError::Driver("Xsens UTC field is not valid".into()));
    }
    let date = NaiveDate::from_ymd_opt(v.year as i32, v.month as u32, v.day as u32)
        .and_then(|d| d.and_hms_nano_opt(v.hour as u32, v.minute as u32, v.second as u32, v.ns))
        .ok_or_else(|| DriverError::Driver("invalid Xsens UTC date".into()))?;
    Ok(Utc
        .from_utc_datetime(&date)
        .timestamp_nanos_opt()
        .ok_or_else(|| DriverError::Driver("Xsens UTC out of range".into()))? as f64
        / 1e9)
}

fn command(mid: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = vec![0xfa, 0xff, mid, payload.len() as u8];
    out.extend_from_slice(payload);
    let checksum = (out[1..].iter().fold(0u8, |sum, v| sum.wrapping_add(*v))).wrapping_neg();
    out.push(checksum);
    out
}

fn coordinate_bits(coordinate_system: &CoordinateSystem) -> u16 {
    match coordinate_system {
        CoordinateSystem::Enu => 0x0000,
        CoordinateSystem::Ned => 0x0004,
        CoordinateSystem::Nwu => 0x0008,
    }
}

fn configure(
    port: &mut dyn serialport::SerialPort,
    rate: u32,
    coordinate_system: &CoordinateSystem,
) -> Result<(), DriverError> {
    port.write_all(&command(0x30, &[]))?;
    std::thread::sleep(Duration::from_millis(150));
    let hz = u16::try_from(rate)
        .map_err(|_| DriverError::Driver("Xsens output rate is too high".into()))?;
    let mut payload = Vec::with_capacity(16);
    for id in [0x1010u16] {
        payload.extend_from_slice(&id.to_be_bytes());
        payload.extend_from_slice(&hz.to_be_bytes());
    }
    for base_id in [0x2010u16, 0x4020, 0x8020] {
        let id = base_id | coordinate_bits(coordinate_system);
        payload.extend_from_slice(&id.to_be_bytes());
        payload.extend_from_slice(&hz.to_be_bytes());
    }
    port.write_all(&command(0xc0, &payload))?;
    std::thread::sleep(Duration::from_millis(150));
    port.write_all(&command(0x10, &[]))?;
    std::thread::sleep(Duration::from_millis(150));
    Ok(())
}

pub fn run(
    cfg: ImuDriverConfig,
    ingress: Arc<dyn SensorIngress>,
    sink: Option<Arc<dyn SensorOutputSink>>,
    stop: Arc<AtomicBool>,
    startup_barrier: Option<Arc<DriverStartGate>>,
) -> Result<(), DriverError> {
    let c = &cfg.xsens;
    let mut port = serialport::new(&c.port, c.baudrate)
        .timeout(Duration::from_millis(100))
        .open()
        .map_err(|e| DriverError::Driver(format!("open {}: {e}", c.port)))?;
    if c.configure_device {
        configure(&mut *port, c.output_rate, &c.coordinate_system)?;
    }
    if let Some(barrier) = startup_barrier {
        barrier.wait(&stop)?;
    }
    let mut decoder =
        Decoder::new(vec![0u8; 4096]).map_err(|e| DriverError::Driver(e.to_string()))?;
    let begun = Instant::now();
    let timeout = Duration::from_secs_f64(c.startup_timeout_secs);
    let skew = Duration::from_secs_f64(c.max_host_skew_secs);
    let mut byte = [0u8; 1];
    while !stop.load(Ordering::Acquire) {
        match port.read(&mut byte) {
            Ok(1) => {}
            Ok(_) => continue,
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                ) =>
            {
                if begun.elapsed() > timeout {
                    return Err(DriverError::StartupTimeout);
                }
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        let frame = match decoder.decode(byte[0]) {
            Ok(Some(f)) => f,
            Ok(None) => continue,
            Err(_) => continue,
        };
        if frame.message_id() != MTData2::MSG_ID {
            continue;
        }
        let data = MTData2::decode_frame(&frame).map_err(|e| DriverError::Driver(e.to_string()))?;
        let (mut utc, mut acc, mut rot, mut q) = (None, None, None, None);
        for packet in data {
            let p = packet.map_err(|e| DriverError::Driver(e.to_string()))?;
            let id = p.data_id();
            match id.data_type() {
                DataType::UtcTime => {
                    utc = Some(
                        UtcTime::from_be_slice(p.payload())
                            .map_err(|e| DriverError::Driver(e.to_string()))?,
                    )
                }
                DataType::Acceleration => acc = Some(vec3(p.payload(), id.precision())?),
                DataType::RateOfTurn => rot = Some(vec3(p.payload(), id.precision())?),
                DataType::Quaternion => q = Some(quat(p.payload(), id.precision())?),
                _ => {}
            }
        }
        let Some((acc, rot, q)) = acc.zip(rot).zip(q).map(|((a, b), c)| (a, b, c)) else {
            continue;
        };
        let host = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| DriverError::Driver(e.to_string()))?
            .as_secs_f64();
        let stamp = match c.timestamp_mode {
            XsensTimestampMode::Host => host,
            XsensTimestampMode::Utc => {
                let value = utc
                    .ok_or_else(|| {
                        DriverError::Driver("Xsens MTData2 message is missing UTC".into())
                    })
                    .and_then(utc_secs)?;
                validate_timestamp(value, host, skew)?;
                value
            }
        };
        let imu = ImuMsg {
            orientation: [q[1], q[2], q[3], q[0]],
            angular_velocity: rot,
            linear_acceleration: acc,
            frame_id: c.frame_id.clone(),
            stamp: time_msg_from_secs(stamp),
            ..Default::default()
        };
        if let Some(sink) = sink.as_ref() {
            sink.mirror_imu(&imu)?
        }
        ingress.push_imu(imu)?;
    }
    Ok(())
}

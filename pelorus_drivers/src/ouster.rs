use crate::{
    DriverError, DriverStartGate, RawLidarPoint, RawLidarScan, ReceiveMetadata, SensorIngress,
    SensorOutputSink, validate_timestamp,
};
use pelorus_core::{ImuMsg, OusterConfig, OusterTimestampMode, time_msg_from_secs};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::{
    io::{Read, Write},
    net::{IpAddr, TcpStream, UdpSocket},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const LIDAR_PACKET_HEADER_SIZE: usize = 32;
const LIDAR_COLUMN_HEADER_SIZE: usize = 12;
const LIDAR_PACKET_FOOTER_SIZE: usize = 32;
const IMU_PACKET_SIZE: usize = 48;
const STANDARD_GRAVITY: f64 = 9.80665;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LidarProfile {
    Single,
    LowDataRate,
}

impl LidarProfile {
    fn parse(value: &str) -> Result<Self, DriverError> {
        match value {
            "RNG19_RFL8_SIG16_NIR16" => Ok(Self::Single),
            "RNG15_RFL8_NIR8" => Ok(Self::LowDataRate),
            value => Err(DriverError::Driver(format!(
                "unsupported Ouster lidar UDP profile `{value}`; supported profiles are RNG19_RFL8_SIG16_NIR16 and RNG15_RFL8_NIR8"
            ))),
        }
    }

    fn channel_size(self) -> usize {
        match self {
            Self::Single => 12,
            Self::LowDataRate => 4,
        }
    }

    fn range_and_reflectivity(self, pixel: &[u8]) -> (u32, u8) {
        let word = u32::from_le_bytes(pixel[0..4].try_into().expect("pixel word is present"));
        match self {
            Self::Single => (word & 0x7ffff, pixel[4]),
            Self::LowDataRate => ((word & 0x7fff) * 8, pixel[2]),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct LidarDataFormat {
    columns_per_frame: usize,
    columns_per_packet: usize,
    pixels_per_column: usize,
    column_window: [usize; 2],
    udp_profile_lidar: String,
    #[serde(default)]
    header_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct BeamIntrinsics {
    beam_altitude_angles: Vec<f64>,
    beam_azimuth_angles: Vec<f64>,
    beam_to_lidar_transform: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct LidarIntrinsics {
    lidar_to_sensor_transform: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ImuIntrinsics {
    imu_to_sensor_transform: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ConfigParams {
    timestamp_mode: String,
    #[serde(default)]
    udp_profile_imu: Option<String>,
}

#[derive(Debug, Clone)]
struct SensorMetadata {
    lidar_data_format: Option<LidarDataFormat>,
    beam_intrinsics: Option<BeamIntrinsics>,
    lidar_intrinsics: Option<LidarIntrinsics>,
    imu_intrinsics: Option<ImuIntrinsics>,
}

fn decode_section<T: DeserializeOwned>(root: &Value, name: &str) -> Result<T, DriverError> {
    let value = root.get(name).unwrap_or(root).clone();
    serde_json::from_value(value)
        .map_err(|error| DriverError::Driver(format!("invalid Ouster {name} metadata: {error}")))
}

fn http_get_json(hostname: &str, path: &str) -> Result<Value, DriverError> {
    let mut stream = TcpStream::connect((hostname, 80))?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {hostname}\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
    )?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    let header_end = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| DriverError::Driver("invalid HTTP response from Ouster sensor".into()))?;
    let header = std::str::from_utf8(&response[..header_end])
        .map_err(|error| DriverError::Driver(format!("invalid Ouster HTTP header: {error}")))?;
    let status = header
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| DriverError::Driver("invalid HTTP status from Ouster sensor".into()))?;
    if !(200..300).contains(&status) {
        return Err(DriverError::Driver(format!(
            "Ouster HTTP request {path} returned status {status}"
        )));
    }
    let body = &response[header_end + 4..];
    serde_json::from_slice(body)
        .map_err(|error| DriverError::Driver(format!("invalid Ouster JSON from {path}: {error}")))
}

fn load_metadata(
    cfg: &OusterConfig,
    receive_lidar: bool,
    receive_imu: bool,
) -> Result<SensorMetadata, DriverError> {
    let root = if let Some(path) = cfg.metadata_path.as_ref() {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|error| {
            DriverError::Driver(format!("invalid Ouster metadata file {path}: {error}"))
        })?
    } else {
        http_get_json(&cfg.hostname, "/api/v1/sensor/metadata")?
    };

    let config_params: ConfigParams = match decode_section(&root, "config_params") {
        Ok(config) => config,
        Err(_) if cfg.metadata_path.is_none() => {
            let value = http_get_json(&cfg.hostname, "/api/v1/sensor/config")?;
            decode_section(&value, "config")?
        }
        Err(error) => return Err(error),
    };
    if cfg.timestamp_mode == OusterTimestampMode::Ptp
        && config_params.timestamp_mode != "TIME_FROM_PTP_1588"
    {
        return Err(DriverError::Driver(format!(
            "Ouster Ptp timestamp mode requires sensor timestamp_mode TIME_FROM_PTP_1588, found {}",
            config_params.timestamp_mode
        )));
    }
    if receive_imu
        && config_params
            .udp_profile_imu
            .as_deref()
            .is_some_and(|profile| profile != "LEGACY")
    {
        return Err(DriverError::Driver(format!(
            "unsupported Ouster IMU UDP profile `{}`; only LEGACY is supported",
            config_params
                .udp_profile_imu
                .as_deref()
                .expect("profile is present")
        )));
    }

    let lidar_data_format = receive_lidar
        .then(|| decode_section(&root, "lidar_data_format"))
        .transpose()?;
    let beam_intrinsics = receive_lidar
        .then(|| decode_section(&root, "beam_intrinsics"))
        .transpose()?;
    let lidar_intrinsics = receive_lidar
        .then(|| decode_section(&root, "lidar_intrinsics"))
        .transpose()?;
    let imu_intrinsics = receive_imu
        .then(|| decode_section(&root, "imu_intrinsics"))
        .transpose()?;
    Ok(SensorMetadata {
        lidar_data_format,
        beam_intrinsics,
        lidar_intrinsics,
        imu_intrinsics,
    })
}

fn matrix(value: Vec<f64>, name: &str) -> Result<[f64; 16], DriverError> {
    value.try_into().map_err(|value: Vec<f64>| {
        DriverError::Driver(format!(
            "Ouster {name} must contain 16 values, found {}",
            value.len()
        ))
    })
}

struct LidarDecoder {
    format: LidarDataFormat,
    profile: LidarProfile,
    altitude: Vec<f64>,
    azimuth: Vec<f64>,
    beam_to_lidar: [f64; 16],
    lidar_to_sensor: [f64; 16],
}

impl LidarDecoder {
    fn new(
        format: LidarDataFormat,
        beam: BeamIntrinsics,
        lidar: LidarIntrinsics,
    ) -> Result<Self, DriverError> {
        if format.columns_per_frame == 0
            || format.columns_per_packet == 0
            || format.pixels_per_column == 0
        {
            return Err(DriverError::Driver(
                "Ouster lidar dimensions must be non-zero".into(),
            ));
        }
        if format
            .header_type
            .as_deref()
            .is_some_and(|header| header != "STANDARD")
        {
            return Err(DriverError::Driver(format!(
                "unsupported Ouster lidar header type `{}`; only STANDARD is supported",
                format.header_type.as_deref().expect("header is present")
            )));
        }
        if beam.beam_altitude_angles.len() != format.pixels_per_column
            || beam.beam_azimuth_angles.len() != format.pixels_per_column
        {
            return Err(DriverError::Driver(format!(
                "Ouster beam-angle count does not match pixels_per_column {}",
                format.pixels_per_column
            )));
        }
        let profile = LidarProfile::parse(&format.udp_profile_lidar)?;
        Ok(Self {
            format,
            profile,
            altitude: beam
                .beam_altitude_angles
                .into_iter()
                .map(f64::to_radians)
                .collect(),
            azimuth: beam
                .beam_azimuth_angles
                .into_iter()
                .map(f64::to_radians)
                .collect(),
            beam_to_lidar: matrix(beam.beam_to_lidar_transform, "beam_to_lidar_transform")?,
            lidar_to_sensor: matrix(lidar.lidar_to_sensor_transform, "lidar_to_sensor_transform")?,
        })
    }

    fn packet_size(&self) -> usize {
        LIDAR_PACKET_HEADER_SIZE
            + self.format.columns_per_packet
                * (LIDAR_COLUMN_HEADER_SIZE
                    + self.format.pixels_per_column * self.profile.channel_size())
            + LIDAR_PACKET_FOOTER_SIZE
    }

    fn active_columns(&self) -> usize {
        let [start, end] = self.format.column_window;
        if end >= start {
            end - start + 1
        } else {
            self.format.columns_per_frame - start + end + 1
        }
    }

    fn xyz(&self, measurement_id: usize, row: usize, range_mm: u32) -> [f32; 3] {
        let encoder = std::f64::consts::TAU
            * (1.0 - measurement_id as f64 / self.format.columns_per_frame as f64);
        let beam_angle = encoder - self.azimuth[row];
        let altitude = self.altitude[row];
        let beam_x = self.beam_to_lidar[3];
        let beam_z = self.beam_to_lidar[11];
        let beam_offset = beam_x.hypot(beam_z);
        let range = range_mm as f64;
        let radial = range - beam_offset;
        let lidar = [
            radial * beam_angle.cos() * altitude.cos() + beam_x * encoder.cos(),
            radial * beam_angle.sin() * altitude.cos() + beam_x * encoder.sin(),
            radial * altitude.sin() + beam_z,
        ];
        let transform = &self.lidar_to_sensor;
        [
            ((transform[0] * lidar[0]
                + transform[1] * lidar[1]
                + transform[2] * lidar[2]
                + transform[3])
                / 1000.0) as f32,
            ((transform[4] * lidar[0]
                + transform[5] * lidar[1]
                + transform[6] * lidar[2]
                + transform[7])
                / 1000.0) as f32,
            ((transform[8] * lidar[0]
                + transform[9] * lidar[1]
                + transform[10] * lidar[2]
                + transform[11])
                / 1000.0) as f32,
        ]
    }
}

struct ScanAssembler {
    decoder: LidarDecoder,
    timestamp_mode: OusterTimestampMode,
    maximum_skew: Duration,
    frame_id: String,
    current_frame: Option<u16>,
    first_sensor_ns: u64,
    stamp: f64,
    received_at: SystemTime,
    sender: Option<IpAddr>,
    packets_received: u64,
    invalid_packets: u64,
    points: Vec<RawLidarPoint>,
    seen_columns: Vec<bool>,
}

impl ScanAssembler {
    fn new(decoder: LidarDecoder, cfg: &OusterConfig) -> Self {
        let seen_columns = vec![false; decoder.format.columns_per_frame];
        Self {
            decoder,
            timestamp_mode: cfg.timestamp_mode.clone(),
            maximum_skew: Duration::from_secs_f64(cfg.max_host_skew_secs),
            frame_id: cfg.lidar_frame_id.clone(),
            current_frame: None,
            first_sensor_ns: 0,
            stamp: 0.0,
            received_at: UNIX_EPOCH,
            sender: None,
            packets_received: 0,
            invalid_packets: 0,
            points: Vec::with_capacity(150_000),
            seen_columns,
        }
    }

    fn finish_scan(&mut self) -> Option<RawLidarScan> {
        if self.current_frame.is_none() || self.points.is_empty() {
            return None;
        }
        let missing_columns = self
            .decoder
            .active_columns()
            .saturating_sub(self.seen_columns.iter().filter(|seen| **seen).count());
        let packets_dropped = missing_columns.div_ceil(self.decoder.format.columns_per_packet)
            as u64
            + self.invalid_packets;
        Some(RawLidarScan {
            frame_id: self.frame_id.clone(),
            stamp: self.stamp,
            points: std::mem::take(&mut self.points),
            receive: ReceiveMetadata {
                received_at: self.received_at,
                sender: self.sender,
                packets_received: self.packets_received,
                packets_dropped,
            },
        })
    }

    fn begin_frame(
        &mut self,
        frame: u16,
        first_sensor_ns: u64,
        host_stamp: f64,
        received_at: SystemTime,
        sender: IpAddr,
    ) -> Result<(), DriverError> {
        let stamp = match self.timestamp_mode {
            OusterTimestampMode::Ptp => {
                let stamp = first_sensor_ns as f64 / 1e9;
                validate_timestamp(stamp, host_stamp, self.maximum_skew)?;
                stamp
            }
            OusterTimestampMode::Host => host_stamp,
        };
        self.current_frame = Some(frame);
        self.first_sensor_ns = first_sensor_ns;
        self.stamp = stamp;
        self.received_at = received_at;
        self.sender = Some(sender);
        self.packets_received = 0;
        self.invalid_packets = 0;
        self.seen_columns.fill(false);
        Ok(())
    }

    fn note_invalid_packet(&mut self) {
        self.invalid_packets += 1;
    }

    fn push_packet(
        &mut self,
        packet: &[u8],
        sender: IpAddr,
        received_at: SystemTime,
    ) -> Result<Option<RawLidarScan>, DriverError> {
        if packet.len() != self.decoder.packet_size() {
            return Err(DriverError::Driver(format!(
                "Ouster lidar packet has size {}, expected {}",
                packet.len(),
                self.decoder.packet_size()
            )));
        }
        if u16::from_le_bytes(packet[0..2].try_into().expect("packet header is present")) != 1 {
            return Err(DriverError::Driver(
                "invalid Ouster lidar packet type".into(),
            ));
        }
        let encoded_crc = u64::from_le_bytes(
            packet[packet.len() - 8..]
                .try_into()
                .expect("packet footer is present"),
        );
        if encoded_crc != 0 && encoded_crc != crc64(&packet[..packet.len() - 8]) {
            return Err(DriverError::Driver(
                "Ouster lidar packet CRC mismatch".into(),
            ));
        }
        let frame = u16::from_le_bytes(packet[2..4].try_into().expect("frame id is present"));
        let first_column = LIDAR_PACKET_HEADER_SIZE;
        let first_sensor_ns = u64::from_le_bytes(
            packet[first_column..first_column + 8]
                .try_into()
                .expect("column timestamp is present"),
        );
        let host_stamp = system_time_secs(received_at)?;
        let completed = if self.current_frame != Some(frame) {
            let completed = self.finish_scan();
            self.begin_frame(frame, first_sensor_ns, host_stamp, received_at, sender)?;
            completed
        } else {
            None
        };
        self.packets_received += 1;

        let channel_size = self.decoder.profile.channel_size();
        let column_size =
            LIDAR_COLUMN_HEADER_SIZE + self.decoder.format.pixels_per_column * channel_size;
        for column_index in 0..self.decoder.format.columns_per_packet {
            let column_offset = LIDAR_PACKET_HEADER_SIZE + column_index * column_size;
            let timestamp = u64::from_le_bytes(
                packet[column_offset..column_offset + 8]
                    .try_into()
                    .expect("column timestamp is present"),
            );
            let measurement_id = u16::from_le_bytes(
                packet[column_offset + 8..column_offset + 10]
                    .try_into()
                    .expect("measurement id is present"),
            ) as usize;
            let status = u16::from_le_bytes(
                packet[column_offset + 10..column_offset + 12]
                    .try_into()
                    .expect("column status is present"),
            );
            if status & 1 == 0 {
                continue;
            }
            if measurement_id >= self.decoder.format.columns_per_frame {
                return Err(DriverError::Driver(format!(
                    "Ouster measurement ID {measurement_id} exceeds columns_per_frame {}",
                    self.decoder.format.columns_per_frame
                )));
            }
            self.seen_columns[measurement_id] = true;
            let time_offset_secs = signed_nanoseconds(timestamp, self.first_sensor_ns) / 1e9;
            let pixel_start = column_offset + LIDAR_COLUMN_HEADER_SIZE;
            for row in 0..self.decoder.format.pixels_per_column {
                let offset = pixel_start + row * channel_size;
                let pixel = &packet[offset..offset + channel_size];
                let (range_mm, reflectivity) = self.decoder.profile.range_and_reflectivity(pixel);
                if range_mm == 0 {
                    continue;
                }
                self.points.push(RawLidarPoint {
                    xyz: self.decoder.xyz(measurement_id, row, range_mm),
                    intensity: reflectivity as f32 / 255.0,
                    time_offset_secs,
                });
            }
        }
        Ok(completed)
    }
}

fn signed_nanoseconds(value: u64, origin: u64) -> f64 {
    if value >= origin {
        (value - origin) as f64
    } else {
        -((origin - value) as f64)
    }
}

fn system_time_secs(value: SystemTime) -> Result<f64, DriverError> {
    value
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .map_err(|error| DriverError::Driver(error.to_string()))
}

fn crc64(bytes: &[u8]) -> u64 {
    let mut crc = u64::MAX;
    for byte in bytes {
        crc ^= u64::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xc96c_5795_d787_0f42
            } else {
                crc >> 1
            };
        }
    }
    crc ^ u64::MAX
}

fn rotate(matrix: &[f64; 16], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[4] * vector[0] + matrix[5] * vector[1] + matrix[6] * vector[2],
        matrix[8] * vector[0] + matrix[9] * vector[1] + matrix[10] * vector[2],
    ]
}

fn decode_imu(
    packet: &[u8],
    received_at: SystemTime,
    cfg: &OusterConfig,
    imu_to_sensor: &[f64; 16],
) -> Result<ImuMsg, DriverError> {
    if packet.len() != IMU_PACKET_SIZE {
        return Err(DriverError::Driver(format!(
            "Ouster IMU packet has size {}, expected {IMU_PACKET_SIZE}",
            packet.len()
        )));
    }
    let accel_timestamp = u64::from_le_bytes(packet[8..16].try_into().expect("timestamp exists"));
    let gyro_timestamp = u64::from_le_bytes(packet[16..24].try_into().expect("timestamp exists"));
    let sensor_timestamp =
        accel_timestamp / 2 + gyro_timestamp / 2 + (accel_timestamp % 2 + gyro_timestamp % 2) / 2;
    let host_stamp = system_time_secs(received_at)?;
    let stamp = match cfg.timestamp_mode {
        OusterTimestampMode::Ptp => {
            let stamp = sensor_timestamp as f64 / 1e9;
            validate_timestamp(
                stamp,
                host_stamp,
                Duration::from_secs_f64(cfg.max_host_skew_secs),
            )?;
            stamp
        }
        OusterTimestampMode::Host => host_stamp,
    };
    let float = |offset: usize| {
        f32::from_le_bytes(
            packet[offset..offset + 4]
                .try_into()
                .expect("IMU component exists"),
        ) as f64
    };
    let acceleration = rotate(
        imu_to_sensor,
        [float(24), float(28), float(32)].map(|value| value * STANDARD_GRAVITY),
    );
    let angular_velocity = rotate(
        imu_to_sensor,
        [float(36), float(40), float(44)].map(f64::to_radians),
    );
    if acceleration
        .into_iter()
        .chain(angular_velocity)
        .any(|value| !value.is_finite())
    {
        return Err(DriverError::Driver(
            "Ouster IMU packet contains a non-finite measurement".into(),
        ));
    }
    Ok(ImuMsg {
        orientation_covariance: [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        angular_velocity,
        linear_acceleration: acceleration,
        frame_id: cfg.imu_frame_id.clone(),
        stamp: time_msg_from_secs(stamp),
        ..Default::default()
    })
}

fn send_scan(
    scan: RawLidarScan,
    ingress: &dyn SensorIngress,
    sink: Option<&Arc<dyn SensorOutputSink>>,
) -> Result<(), DriverError> {
    if let Some(sink) = sink {
        sink.mirror_lidar(&scan)?;
    }
    ingress.push_lidar(scan)
}

fn send_imu(
    imu: ImuMsg,
    ingress: &dyn SensorIngress,
    sink: Option<&Arc<dyn SensorOutputSink>>,
) -> Result<(), DriverError> {
    if let Some(sink) = sink {
        sink.mirror_imu(&imu)?;
    }
    ingress.push_imu(imu)
}

pub fn run(
    cfg: OusterConfig,
    receive_lidar: bool,
    receive_imu: bool,
    ingress: Arc<dyn SensorIngress>,
    sink: Option<Arc<dyn SensorOutputSink>>,
    stop: Arc<AtomicBool>,
    startup_barrier: Option<Arc<DriverStartGate>>,
) -> Result<(), DriverError> {
    let metadata = load_metadata(&cfg, receive_lidar, receive_imu)?;
    let mut lidar = match (
        metadata.lidar_data_format,
        metadata.beam_intrinsics,
        metadata.lidar_intrinsics,
    ) {
        (Some(format), Some(beam), Some(intrinsics)) => Some(ScanAssembler::new(
            LidarDecoder::new(format, beam, intrinsics)?,
            &cfg,
        )),
        (None, None, None) if !receive_lidar => None,
        _ => {
            return Err(DriverError::Driver(
                "incomplete Ouster lidar metadata".into(),
            ));
        }
    };
    let imu_transform = match metadata.imu_intrinsics {
        Some(intrinsics) => Some(matrix(
            intrinsics.imu_to_sensor_transform,
            "imu_to_sensor_transform",
        )?),
        None if !receive_imu => None,
        None => return Err(DriverError::Driver("missing Ouster IMU metadata".into())),
    };
    let expected: Option<IpAddr> = cfg
        .device_ip
        .as_deref()
        .map(str::parse)
        .transpose()
        .map_err(|error| DriverError::Driver(format!("invalid Ouster device IP: {error}")))?;
    let lidar_socket = receive_lidar
        .then(|| UdpSocket::bind((cfg.bind_address.as_str(), cfg.lidar_port)))
        .transpose()?;
    let imu_socket = receive_imu
        .then(|| UdpSocket::bind((cfg.bind_address.as_str(), cfg.imu_port)))
        .transpose()?;
    if let Some(socket) = lidar_socket.as_ref() {
        socket.set_nonblocking(true)?;
    }
    if let Some(socket) = imu_socket.as_ref() {
        socket.set_nonblocking(true)?;
    }
    if let Some(barrier) = startup_barrier {
        barrier.wait(&stop)?;
    }

    let begun = Instant::now();
    let timeout = Duration::from_secs_f64(cfg.startup_timeout_secs);
    let mut lidar_started = !receive_lidar;
    let mut imu_started = !receive_imu;
    let mut bytes = [0u8; 65_535];
    while !stop.load(Ordering::Acquire) {
        let mut received = false;
        if let Some(socket) = lidar_socket.as_ref() {
            loop {
                let (size, from) = match socket.recv_from(&mut bytes) {
                    Ok(value) => value,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                    Err(error) => return Err(error.into()),
                };
                received = true;
                if expected.is_some_and(|address| address != from.ip()) {
                    continue;
                }
                let received_at = SystemTime::now();
                let assembler = lidar
                    .as_mut()
                    .expect("lidar decoder exists when socket exists");
                match assembler.push_packet(&bytes[..size], from.ip(), received_at) {
                    Ok(Some(scan)) => send_scan(scan, ingress.as_ref(), sink.as_ref())?,
                    Ok(None) => {}
                    Err(error @ DriverError::ClockSkew { .. }) => return Err(error),
                    Err(_) => {
                        assembler.note_invalid_packet();
                        continue;
                    }
                }
                lidar_started = true;
            }
        }
        if let Some(socket) = imu_socket.as_ref() {
            loop {
                let (size, from) = match socket.recv_from(&mut bytes) {
                    Ok(value) => value,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                    Err(error) => return Err(error.into()),
                };
                received = true;
                if expected.is_some_and(|address| address != from.ip()) {
                    continue;
                }
                let imu = match decode_imu(
                    &bytes[..size],
                    SystemTime::now(),
                    &cfg,
                    imu_transform
                        .as_ref()
                        .expect("IMU transform exists when socket exists"),
                ) {
                    Ok(imu) => imu,
                    Err(error @ DriverError::ClockSkew { .. }) => return Err(error),
                    Err(_) => continue,
                };
                send_imu(imu, ingress.as_ref(), sink.as_ref())?;
                imu_started = true;
            }
        }
        if (!lidar_started || !imu_started) && begun.elapsed() > timeout {
            return Err(DriverError::StartupTimeout);
        }
        if !received {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Vec<f64> {
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]
    }

    fn decoder(profile: &str) -> LidarDecoder {
        LidarDecoder::new(
            LidarDataFormat {
                columns_per_frame: 2,
                columns_per_packet: 1,
                pixels_per_column: 1,
                column_window: [0, 1],
                udp_profile_lidar: profile.into(),
                header_type: Some("STANDARD".into()),
            },
            BeamIntrinsics {
                beam_altitude_angles: vec![0.0],
                beam_azimuth_angles: vec![0.0],
                beam_to_lidar_transform: identity(),
            },
            LidarIntrinsics {
                lidar_to_sensor_transform: identity(),
            },
        )
        .unwrap()
    }

    fn packet(decoder: &LidarDecoder, frame: u16, range_word: u32, reflectivity: u8) -> Vec<u8> {
        let mut packet = vec![0; decoder.packet_size()];
        packet[0..2].copy_from_slice(&1u16.to_le_bytes());
        packet[2..4].copy_from_slice(&frame.to_le_bytes());
        packet[32..40].copy_from_slice(&1_000_000_000u64.to_le_bytes());
        packet[40..42].copy_from_slice(&0u16.to_le_bytes());
        packet[42..44].copy_from_slice(&1u16.to_le_bytes());
        packet[44..48].copy_from_slice(&range_word.to_le_bytes());
        match decoder.profile {
            LidarProfile::Single => packet[48] = reflectivity,
            LidarProfile::LowDataRate => {
                packet[46] = reflectivity;
            }
        }
        let crc = crc64(&packet[..packet.len() - 8]);
        let footer = packet.len() - 8;
        packet[footer..].copy_from_slice(&crc.to_le_bytes());
        packet
    }

    fn host_config() -> OusterConfig {
        OusterConfig {
            timestamp_mode: OusterTimestampMode::Host,
            ..Default::default()
        }
    }

    #[test]
    fn single_return_packet_decodes_range_and_reflectivity() {
        let decoder = decoder("RNG19_RFL8_SIG16_NIR16");
        let first = packet(&decoder, 4, 1_000, 128);
        let next = packet(&decoder, 5, 2_000, 255);
        let mut assembler = ScanAssembler::new(decoder, &host_config());
        assembler
            .push_packet(&first, "192.0.2.1".parse().unwrap(), SystemTime::now())
            .unwrap();
        let scan = assembler
            .push_packet(&next, "192.0.2.1".parse().unwrap(), SystemTime::now())
            .unwrap()
            .unwrap();
        assert_eq!(scan.points.len(), 1);
        assert!((scan.points[0].xyz[0] - 1.0).abs() < 1e-6);
        assert!((scan.points[0].intensity - 128.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn low_data_rate_range_uses_eight_millimeter_units() {
        let decoder = decoder("RNG15_RFL8_NIR8");
        let first = packet(&decoder, 4, 125, 64);
        let next = packet(&decoder, 5, 250, 255);
        let mut assembler = ScanAssembler::new(decoder, &host_config());
        assembler
            .push_packet(&first, "192.0.2.1".parse().unwrap(), SystemTime::now())
            .unwrap();
        let scan = assembler
            .push_packet(&next, "192.0.2.1".parse().unwrap(), SystemTime::now())
            .unwrap()
            .unwrap();
        assert!((scan.points[0].xyz[0] - 1.0).abs() < 1e-6);
        assert!((scan.points[0].intensity - 64.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn imu_values_are_converted_to_si_units() {
        let mut packet = [0u8; IMU_PACKET_SIZE];
        packet[8..16].copy_from_slice(&1_000_000_000u64.to_le_bytes());
        packet[16..24].copy_from_slice(&1_000_000_000u64.to_le_bytes());
        packet[24..28].copy_from_slice(&1.0f32.to_le_bytes());
        packet[40..44].copy_from_slice(&180.0f32.to_le_bytes());
        let transform: [f64; 16] = identity().try_into().unwrap();
        let imu = decode_imu(&packet, SystemTime::now(), &host_config(), &transform).unwrap();
        assert!((imu.linear_acceleration[0] - STANDARD_GRAVITY).abs() < 1e-9);
        assert!((imu.angular_velocity[1] - std::f64::consts::PI).abs() < 1e-9);
        assert_eq!(imu.orientation_covariance[0], -1.0);
    }

    #[test]
    fn crc_detects_corruption() {
        let decoder = decoder("RNG19_RFL8_SIG16_NIR16");
        let mut packet = packet(&decoder, 1, 1_000, 1);
        packet[44] ^= 1;
        let mut assembler = ScanAssembler::new(decoder, &host_config());
        assert!(
            assembler
                .push_packet(&packet, "192.0.2.1".parse().unwrap(), SystemTime::now())
                .is_err()
        );
    }

    #[test]
    fn unsupported_dual_return_profile_is_rejected() {
        assert!(LidarProfile::parse("RNG19_RFL8_SIG16_NIR16_DUAL").is_err());
    }

    #[test]
    fn ptp_imu_timestamp_is_checked_against_host_time() {
        let mut packet = [0u8; IMU_PACKET_SIZE];
        packet[8..16].copy_from_slice(&1_000_000_000u64.to_le_bytes());
        packet[16..24].copy_from_slice(&1_000_000_000u64.to_le_bytes());
        let transform: [f64; 16] = identity().try_into().unwrap();
        assert!(matches!(
            decode_imu(
                &packet,
                SystemTime::now(),
                &OusterConfig::default(),
                &transform
            ),
            Err(DriverError::ClockSkew { .. })
        ));
    }
}

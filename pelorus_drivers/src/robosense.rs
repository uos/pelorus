//! RoboSense solid-state (M-series) MSOP/DIFOP driver.

use crate::{
    DriverError, DriverStartGate, RawLidarPoint, RawLidarScan, ReceiveMetadata, SensorIngress,
    SensorOutputSink, validate_timestamp,
};
use pelorus_core::{RobosenseConfig, RobosenseModel, RobosenseTimestampMode};
use std::{
    net::{IpAddr, UdpSocket},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const MSOP_ID: [u8; 4] = [0x55, 0xAA, 0x5A, 0xA5];
const DIFOP_ID: [u8; 8] = [0xA5, 0xFF, 0x00, 0x5A, 0x11, 0x11, 0x55, 0x55];
const DIFOP_PACKET_SIZE: usize = 256;
const HEADER_SIZE: usize = 32;
/// Range LSB, metres. Identical across the M-series.
const DISTANCE_RESOLUTION: f64 = 0.005;
/// Unit-vector components are signed 16-bit fixed point over 2^15.
const VECTOR_BASE: f64 = 32768.0;
/// M1 encodes pitch and yaw biased by 2^15, in hundredths of a degree.
const ANGLE_OFFSET: i32 = 32768;
const ANGLE_RESOLUTION_DEG: f64 = 0.01;
/// Nominal frame period; only used to back-date host timestamps by one packet.
const FRAME_DURATION_SECS: f64 = 0.1;

/// A packet sequence this far below the previous one is a frame restart rather
/// than reordering. Mirrors `SplitStrategyBySeq::RANGE` in RoboSense rs_driver.
const SEQUENCE_REWIND_SLACK: u16 = 10;

/// How long the startup clock report samples before it prints.
const SYNC_REPORT_WINDOW: Duration = Duration::from_secs(3);
/// Drift measured at startup is projected this far ahead to decide whether it
/// will exhaust the configured skew budget.
const DRIFT_HORIZON: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EchoMode {
    Single,
    Dual,
}

impl EchoMode {
    /// 0x00 is dual; 0x04/0x05/0x06 select strongest/last/first.
    fn parse(value: u8) -> Self {
        match value {
            0x00 => Self::Dual,
            _ => Self::Single,
        }
    }
}

/// Per-point payload layout. M1 carries spherical angles, every later model
/// carries a Cartesian unit vector that is scaled by the measured range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChannelLayout {
    /// `distance:u16, pitch:u16, yaw:u16, intensity:u8, attribute:u8, elongation:u8`
    SphericalM1,
    /// `distance:u16, x:i16, y:i16, z:i16, intensity:u8, attribute:u8`
    Vector,
    /// `range_first:u16, range_second:u16, x:i16, y:i16, z:i16,
    ///  intensity_first:u8, intensity_second:u8, attribute:u8`
    VectorDualMx,
}

impl ChannelLayout {
    const fn size(self) -> usize {
        match self {
            Self::SphericalM1 => 9,
            Self::Vector => 10,
            Self::VectorDualMx => 13,
        }
    }
}

/// Everything about a model that the decoder varies on.
#[derive(Debug, Clone, Copy)]
struct ModelSpec {
    packet_size: usize,
    blocks_per_packet: usize,
    channels_per_block: usize,
    layout: ChannelLayout,
    /// Bytes of per-block header ahead of the channels (time offset, and on M1
    /// and M2 a return-sequence byte).
    block_header_size: usize,
    /// E1 stores its block time offset as a big-endian u16; the others use u8.
    wide_time_offset: bool,
    /// Packets expected per frame, used to pace host-clock timestamps.
    packets_per_frame: u32,
    minimum_range_m: f64,
    maximum_range_m: f64,
}

impl ModelSpec {
    fn for_model(model: RobosenseModel) -> Self {
        match model {
            // 32 + 25 * (2 + 5 * 9) + 3 = 1210
            RobosenseModel::M1 => Self {
                packet_size: 1210,
                blocks_per_packet: 25,
                channels_per_block: 5,
                layout: ChannelLayout::SphericalM1,
                block_header_size: 2,
                wide_time_offset: false,
                packets_per_frame: 630,
                minimum_range_m: 0.2,
                maximum_range_m: 200.0,
            },
            // 32 + 25 * (2 + 5 * 10) + 4 + 4 + 2 = 1342
            RobosenseModel::M2 => Self {
                packet_size: 1342,
                blocks_per_packet: 25,
                channels_per_block: 5,
                layout: ChannelLayout::Vector,
                block_header_size: 2,
                wide_time_offset: false,
                packets_per_frame: 1260,
                minimum_range_m: 0.2,
                maximum_range_m: 250.0,
            },
            // 32 + 96 * (2 + 1 * 10) + 16 = 1200
            RobosenseModel::E1 => Self {
                packet_size: 1200,
                blocks_per_packet: 96,
                channels_per_block: 1,
                layout: ChannelLayout::Vector,
                block_header_size: 2,
                wide_time_offset: true,
                packets_per_frame: 288,
                minimum_range_m: 0.0,
                maximum_range_m: 200.0,
            },
            // 32 + 50 * (1 + 2 * 13) + 16 + 4 + 2 = 1404
            RobosenseModel::Mx => Self {
                packet_size: 1404,
                blocks_per_packet: 50,
                channels_per_block: 2,
                layout: ChannelLayout::VectorDualMx,
                block_header_size: 1,
                wide_time_offset: false,
                packets_per_frame: 630,
                minimum_range_m: 0.2,
                maximum_range_m: 200.0,
            },
        }
    }

    const fn block_size(&self) -> usize {
        self.block_header_size + self.channels_per_block * self.layout.size()
    }

    fn packet_duration_secs(&self) -> f64 {
        FRAME_DURATION_SECS / f64::from(self.packets_per_frame)
    }

    /// A zero range means the channel saw nothing. E1 declares a minimum range
    /// of zero, so without the explicit check every empty return would decode
    /// to a point at the sensor origin and pile up a phantom cluster there.
    fn range_is_usable(&self, range_m: f64) -> bool {
        range_m > 0.0 && range_m >= self.minimum_range_m && range_m <= self.maximum_range_m
    }
}

fn be_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_be_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .expect("slice is two bytes"),
    )
}

fn be_i16(bytes: &[u8], offset: usize) -> i16 {
    i16::from_be_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .expect("slice is two bytes"),
    )
}

/// RSTimestampUTC: six big-endian seconds bytes then four big-endian
/// microsecond bytes.
fn parse_sensor_timestamp(header: &[u8]) -> f64 {
    let seconds = header[10..16]
        .iter()
        .fold(0u64, |value, byte| (value << 8) | u64::from(*byte));
    let micros = header[16..20]
        .iter()
        .fold(0u64, |value, byte| (value << 8) | u64::from(*byte));
    seconds as f64 + micros as f64 / 1e6
}

fn system_time_secs(value: SystemTime) -> Result<f64, DriverError> {
    value
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .map_err(|error| DriverError::Driver(error.to_string()))
}

/// What the sensor clock did against the host clock during startup.
#[derive(Debug, Clone, Copy, PartialEq)]
struct SyncReport {
    samples: u64,
    /// Mean of `sensor - host`. Biased positive-to-negative by transport
    /// latency, since the host time is taken on arrival rather than at
    /// measurement, so this is sync error plus latency, not sync error alone.
    mean_offset_secs: f64,
    /// Largest minus smallest offset seen, so jitter and drift together.
    spread_secs: f64,
    /// Offset at the end of the window minus the offset at its start.
    drift_secs: f64,
    window_secs: f64,
}

impl SyncReport {
    fn drift_rate_per_sec(&self) -> f64 {
        if self.window_secs > 0.0 {
            self.drift_secs / self.window_secs
        } else {
            0.0
        }
    }

    /// Whether drift at the observed rate would exhaust `budget` within
    /// `horizon`. A locked PTP or GPS clock holds a roughly constant offset, so
    /// a rate that eats the budget this quickly means the sensor is running on
    /// its own oscillator and will fail the skew check later in the run.
    fn exhausts_budget_within(&self, budget: Duration, horizon: Duration) -> bool {
        self.drift_rate_per_sec().abs() * horizon.as_secs_f64() > budget.as_secs_f64()
    }
}

/// Samples the sensor-versus-host clock offset over the first few seconds.
///
/// The per-packet skew check rejects a sensor that is not synchronized at all,
/// but it cannot separate a locked clock from one that sits a few milliseconds
/// out and creeps: both stay inside `max_host_skew` at startup and only one of
/// them still does an hour later. This reports what the offset actually looked
/// like, so a marginal clock is visible in the log instead of showing up as
/// odometry that slowly degrades.
#[derive(Default)]
struct SyncMonitor {
    begun: Option<Instant>,
    samples: u64,
    sum: f64,
    minimum: f64,
    maximum: f64,
    first: f64,
    last: f64,
    reported: bool,
}

impl SyncMonitor {
    /// Records one packet. Returns the report exactly once, on the first packet
    /// after the sampling window closes.
    fn observe(&mut self, sensor_stamp: f64, host_stamp: f64, now: Instant) -> Option<SyncReport> {
        if self.reported {
            return None;
        }
        let offset = sensor_stamp - host_stamp;
        let begun = *self.begun.get_or_insert(now);
        if self.samples == 0 {
            self.first = offset;
            self.minimum = offset;
            self.maximum = offset;
        }
        self.samples += 1;
        self.sum += offset;
        self.minimum = self.minimum.min(offset);
        self.maximum = self.maximum.max(offset);
        self.last = offset;

        let elapsed = now.duration_since(begun);
        if elapsed < SYNC_REPORT_WINDOW {
            return None;
        }
        self.reported = true;
        Some(SyncReport {
            samples: self.samples,
            mean_offset_secs: self.sum / self.samples as f64,
            spread_secs: self.maximum - self.minimum,
            drift_secs: self.last - self.first,
            window_secs: elapsed.as_secs_f64(),
        })
    }
}

fn log_sync_report(
    report: &SyncReport,
    model: RobosenseModel,
    mode: &RobosenseTimestampMode,
    budget: Duration,
) {
    let ms = |seconds: f64| seconds * 1e3;
    match mode {
        RobosenseTimestampMode::Sensor => {
            log::info!(
                "RoboSense {model:?} clock: sensor runs {:+.3} ms from host \
                 (including transport latency) over {:.1} s and {} packets, \
                 spread {:.3} ms, drift {:+.3} ms",
                ms(report.mean_offset_secs),
                report.window_secs,
                report.samples,
                ms(report.spread_secs),
                ms(report.drift_secs),
            );
            if report.exhausts_budget_within(budget, DRIFT_HORIZON) {
                log::warn!(
                    "RoboSense {model:?} clock is drifting {:+.3} ms/s against the host, \
                     which passes the {:.0} ms max_host_skew within {:.0} s. The sensor is \
                     probably free-running: check that PTP or GPS sync is locked on the device.",
                    ms(report.drift_rate_per_sec()),
                    ms(budget.as_secs_f64()),
                    DRIFT_HORIZON.as_secs_f64(),
                );
            }
        }
        // The sensor clock is not used for stamping here, so this is only
        // useful for deciding whether Sensor mode is worth switching to.
        RobosenseTimestampMode::Host => {
            log::debug!(
                "RoboSense {model:?} clock (unused in Host mode): sensor runs {:+.3} ms \
                 from host over {:.1} s and {} packets, drift {:+.3} ms",
                ms(report.mean_offset_secs),
                report.window_secs,
                report.samples,
                ms(report.drift_secs),
            );
        }
    }
}

/// Tracks the MSOP sequence counter to find frame boundaries.
///
/// The sequence restarts at the top of every frame. Packets can arrive slightly
/// out of order, so only a drop of more than [`SEQUENCE_REWIND_SLACK`] below the
/// last in-order sequence counts as a restart.
#[derive(Default)]
struct SequenceSplitter {
    previous: Option<u16>,
}

impl SequenceSplitter {
    fn is_frame_start(&mut self, sequence: u16) -> bool {
        let Some(previous) = self.previous else {
            self.previous = Some(sequence);
            // The first packet opens a frame without completing one.
            return false;
        };
        let floor = previous.saturating_sub(SEQUENCE_REWIND_SLACK);
        if sequence < floor {
            self.previous = Some(sequence);
            return true;
        }
        if sequence >= previous {
            self.previous = Some(sequence);
        }
        // Between `floor` and `previous` is reordering, not a restart.
        false
    }
}

struct ScanAssembler {
    spec: ModelSpec,
    model: RobosenseModel,
    timestamp_mode: RobosenseTimestampMode,
    maximum_skew: Duration,
    frame_id: String,
    echo_mode: EchoMode,
    splitter: SequenceSplitter,
    sync_monitor: SyncMonitor,
    started: bool,
    /// Timestamp of the packet that opened the current frame.
    stamp: f64,
    received_at: SystemTime,
    sender: Option<IpAddr>,
    packets_received: u64,
    invalid_packets: u64,
    /// Highest sequence seen in the frame, to estimate how many were lost.
    highest_sequence: u16,
    points: Vec<RawLidarPoint>,
}

impl ScanAssembler {
    fn new(cfg: &RobosenseConfig) -> Self {
        let spec = ModelSpec::for_model(cfg.model);
        Self {
            spec,
            model: cfg.model,
            timestamp_mode: cfg.timestamp_mode.clone(),
            maximum_skew: Duration::from_secs_f64(cfg.max_host_skew_secs),
            frame_id: cfg.frame_id.clone(),
            // Until a DIFOP packet says otherwise, assume the common case.
            echo_mode: EchoMode::Single,
            splitter: SequenceSplitter::default(),
            sync_monitor: SyncMonitor::default(),
            started: false,
            stamp: 0.0,
            received_at: UNIX_EPOCH,
            sender: None,
            packets_received: 0,
            invalid_packets: 0,
            highest_sequence: 0,
            points: Vec::with_capacity(150_000),
        }
    }

    fn set_echo_mode(&mut self, echo_mode: EchoMode) {
        self.echo_mode = echo_mode;
    }

    fn note_invalid_packet(&mut self) {
        self.invalid_packets += 1;
    }

    fn finish_scan(&mut self) -> Option<RawLidarScan> {
        if !self.started || self.points.is_empty() {
            return None;
        }
        // Sequences are 0-based, so the highest one seen implies the count.
        let expected = u64::from(self.highest_sequence) + 1;
        let packets_dropped = expected.saturating_sub(self.packets_received) + self.invalid_packets;
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

    fn begin_frame(&mut self, stamp: f64, received_at: SystemTime, sender: IpAddr, sequence: u16) {
        self.started = true;
        self.stamp = stamp;
        self.received_at = received_at;
        self.sender = Some(sender);
        self.packets_received = 0;
        self.invalid_packets = 0;
        self.highest_sequence = sequence;
    }

    fn push_packet(
        &mut self,
        packet: &[u8],
        sender: IpAddr,
        received_at: SystemTime,
    ) -> Result<Option<RawLidarScan>, DriverError> {
        if packet.len() != self.spec.packet_size {
            return Err(DriverError::Driver(format!(
                "RoboSense {:?} MSOP packet has size {}, expected {}",
                self.model,
                packet.len(),
                self.spec.packet_size
            )));
        }
        if packet[0..4] != MSOP_ID {
            return Err(DriverError::Driver(
                "invalid RoboSense MSOP packet identifier".into(),
            ));
        }
        let sequence = be_u16(packet, 4);
        let host_stamp = system_time_secs(received_at)?;
        // Sampled in both modes: in Host mode the sensor clock is not used for
        // stamping, but knowing what it was doing is how you decide whether
        // Sensor mode is available.
        let sensor_stamp = parse_sensor_timestamp(packet);
        if let Some(report) = self
            .sync_monitor
            .observe(sensor_stamp, host_stamp, Instant::now())
        {
            log_sync_report(&report, self.model, &self.timestamp_mode, self.maximum_skew);
        }
        let packet_stamp = match self.timestamp_mode {
            RobosenseTimestampMode::Sensor => {
                validate_timestamp(sensor_stamp, host_stamp, self.maximum_skew)?;
                sensor_stamp
            }
            // Roll back one packet so the stamp approximates when the first
            // block in this packet was measured rather than when it landed.
            RobosenseTimestampMode::Host => host_stamp - self.spec.packet_duration_secs(),
        };

        let completed = if self.splitter.is_frame_start(sequence) {
            let completed = self.finish_scan();
            self.begin_frame(packet_stamp, received_at, sender, sequence);
            completed
        } else {
            if !self.started {
                self.begin_frame(packet_stamp, received_at, sender, sequence);
            }
            None
        };
        self.packets_received += 1;
        self.highest_sequence = self.highest_sequence.max(sequence);

        // Per-point offsets are relative to the scan stamp, not this packet.
        let packet_offset = packet_stamp - self.stamp;
        self.decode_blocks(packet, packet_offset);
        Ok(completed)
    }

    fn decode_blocks(&mut self, packet: &[u8], packet_offset_secs: f64) {
        let block_size = self.spec.block_size();
        let channel_size = self.spec.layout.size();
        for block_index in 0..self.spec.blocks_per_packet {
            let block = HEADER_SIZE + block_index * block_size;
            let block_micros = if self.spec.wide_time_offset {
                f64::from(be_u16(packet, block))
            } else {
                f64::from(packet[block])
            };
            let time_offset_secs = packet_offset_secs + block_micros / 1e6;
            for channel_index in 0..self.spec.channels_per_block {
                let offset = block + self.spec.block_header_size + channel_index * channel_size;
                let channel = &packet[offset..offset + channel_size];
                self.decode_channel(channel, time_offset_secs);
            }
        }
    }

    fn decode_channel(&mut self, channel: &[u8], time_offset_secs: f64) {
        match self.spec.layout {
            ChannelLayout::SphericalM1 => {
                let range = f64::from(be_u16(channel, 0)) * DISTANCE_RESOLUTION;
                if !self.spec.range_is_usable(range) {
                    return;
                }
                let pitch =
                    (i32::from(be_u16(channel, 2)) - ANGLE_OFFSET) as f64 * ANGLE_RESOLUTION_DEG;
                let yaw =
                    (i32::from(be_u16(channel, 4)) - ANGLE_OFFSET) as f64 * ANGLE_RESOLUTION_DEG;
                let (pitch, yaw) = (pitch.to_radians(), yaw.to_radians());
                let xyz = [
                    range * pitch.cos() * yaw.cos(),
                    range * pitch.cos() * yaw.sin(),
                    range * pitch.sin(),
                ];
                self.push_point(xyz, channel[6], time_offset_secs);
            }
            ChannelLayout::Vector => {
                let range = f64::from(be_u16(channel, 0)) * DISTANCE_RESOLUTION;
                if !self.spec.range_is_usable(range) {
                    return;
                }
                let xyz = scale_unit_vector(channel, 2, range);
                self.push_point(xyz, channel[8], time_offset_secs);
            }
            ChannelLayout::VectorDualMx => {
                // Both returns share one direction and differ only in range.
                let returns: &[(usize, usize)] = match self.echo_mode {
                    EchoMode::Single => &[(0, 10)],
                    EchoMode::Dual => &[(0, 10), (2, 11)],
                };
                for (range_offset, intensity_offset) in returns {
                    let range = f64::from(be_u16(channel, *range_offset)) * DISTANCE_RESOLUTION;
                    if !self.spec.range_is_usable(range) {
                        continue;
                    }
                    let xyz = scale_unit_vector(channel, 4, range);
                    self.push_point(xyz, channel[*intensity_offset], time_offset_secs);
                }
            }
        }
    }

    fn push_point(&mut self, xyz: [f64; 3], intensity: u8, time_offset_secs: f64) {
        self.points.push(RawLidarPoint {
            xyz: [xyz[0] as f32, xyz[1] as f32, xyz[2] as f32],
            intensity: f32::from(intensity) / 255.0,
            time_offset_secs,
        });
    }
}

/// Decode the signed 16-bit direction triple at `offset` and scale it by range.
fn scale_unit_vector(channel: &[u8], offset: usize, range_m: f64) -> [f64; 3] {
    [
        f64::from(be_i16(channel, offset)) * range_m / VECTOR_BASE,
        f64::from(be_i16(channel, offset + 2)) * range_m / VECTOR_BASE,
        f64::from(be_i16(channel, offset + 4)) * range_m / VECTOR_BASE,
    ]
}

/// DIFOP carries the echo mode, which decides how many returns MX reports.
fn decode_difop_echo_mode(packet: &[u8]) -> Option<EchoMode> {
    if packet.len() != DIFOP_PACKET_SIZE || packet[0..8] != DIFOP_ID {
        return None;
    }
    // id[8] + reserved[1] + frame_rate[1] + eth[18] + version[10] + sn[6].
    const RETURN_MODE_OFFSET: usize = 44;
    Some(EchoMode::parse(packet[RETURN_MODE_OFFSET]))
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

pub fn run(
    cfg: RobosenseConfig,
    ingress: Arc<dyn SensorIngress>,
    sink: Option<Arc<dyn SensorOutputSink>>,
    stop: Arc<AtomicBool>,
    startup_barrier: Option<Arc<DriverStartGate>>,
) -> Result<(), DriverError> {
    let expected: Option<IpAddr> = cfg
        .device_ip
        .as_deref()
        .map(str::parse)
        .transpose()
        .map_err(|error| DriverError::Driver(format!("invalid RoboSense device IP: {error}")))?;
    let msop_socket = UdpSocket::bind((cfg.bind_address.as_str(), cfg.msop_port))?;
    let difop_socket = UdpSocket::bind((cfg.bind_address.as_str(), cfg.difop_port))?;
    msop_socket.set_nonblocking(true)?;
    difop_socket.set_nonblocking(true)?;

    let mut assembler = ScanAssembler::new(&cfg);
    if let Some(barrier) = startup_barrier {
        barrier.wait(&stop)?;
    }

    let begun = Instant::now();
    let timeout = Duration::from_secs_f64(cfg.startup_timeout_secs);
    let mut started = false;
    let mut bytes = [0u8; 65_535];
    while !stop.load(Ordering::Acquire) {
        let mut received = false;
        loop {
            let (size, from) = match msop_socket.recv_from(&mut bytes) {
                Ok(value) => value,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) => return Err(error.into()),
            };
            received = true;
            if expected.is_some_and(|address| address != from.ip()) {
                continue;
            }
            match assembler.push_packet(&bytes[..size], from.ip(), SystemTime::now()) {
                Ok(Some(scan)) => send_scan(scan, ingress.as_ref(), sink.as_ref())?,
                Ok(None) => {}
                Err(error @ DriverError::ClockSkew { .. }) => return Err(error),
                Err(_) => {
                    assembler.note_invalid_packet();
                    continue;
                }
            }
            started = true;
        }
        loop {
            let (size, from) = match difop_socket.recv_from(&mut bytes) {
                Ok(value) => value,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) => return Err(error.into()),
            };
            received = true;
            if expected.is_some_and(|address| address != from.ip()) {
                continue;
            }
            if let Some(echo_mode) = decode_difop_echo_mode(&bytes[..size]) {
                assembler.set_echo_mode(echo_mode);
            }
        }
        if !started && begun.elapsed() > timeout {
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

    fn config(model: RobosenseModel) -> RobosenseConfig {
        RobosenseConfig {
            model,
            timestamp_mode: RobosenseTimestampMode::Host,
            ..Default::default()
        }
    }

    /// Build an MSOP packet whose first channel of the first block is populated.
    fn packet(spec: &ModelSpec, sequence: u16) -> Vec<u8> {
        let mut packet = vec![0u8; spec.packet_size];
        packet[0..4].copy_from_slice(&MSOP_ID);
        packet[4..6].copy_from_slice(&sequence.to_be_bytes());
        packet
    }

    fn sender() -> IpAddr {
        "192.0.2.1".parse().unwrap()
    }

    #[test]
    fn model_specs_match_the_documented_packet_sizes() {
        for (model, size) in [
            (RobosenseModel::M1, 1210),
            (RobosenseModel::M2, 1342),
            (RobosenseModel::E1, 1200),
            (RobosenseModel::Mx, 1404),
        ] {
            let spec = ModelSpec::for_model(model);
            // Header plus blocks must fit inside the packet, with the model's
            // trailer (reserved bytes, CRC, rolling counter) left over.
            assert_eq!(spec.packet_size, size, "{model:?}");
            assert!(
                HEADER_SIZE + spec.blocks_per_packet * spec.block_size() <= spec.packet_size,
                "{model:?} blocks overrun the packet"
            );
        }
    }

    #[test]
    fn m1_decodes_spherical_angles_to_cartesian() {
        let spec = ModelSpec::for_model(RobosenseModel::M1);
        let mut first = packet(&spec, 100);
        // 10 m straight ahead: pitch 0, yaw 0, both biased by 2^15.
        let channel = HEADER_SIZE + spec.block_header_size;
        first[channel..channel + 2].copy_from_slice(&2000u16.to_be_bytes());
        first[channel + 2..channel + 4].copy_from_slice(&(ANGLE_OFFSET as u16).to_be_bytes());
        first[channel + 4..channel + 6].copy_from_slice(&(ANGLE_OFFSET as u16).to_be_bytes());
        first[channel + 6] = 128;

        let mut assembler = ScanAssembler::new(&config(RobosenseModel::M1));
        assembler
            .push_packet(&first, sender(), SystemTime::now())
            .unwrap();
        let scan = assembler
            .push_packet(&packet(&spec, 0), sender(), SystemTime::now())
            .unwrap()
            .expect("sequence rewind completes the frame");
        assert_eq!(scan.points.len(), 1);
        assert!((scan.points[0].xyz[0] - 10.0).abs() < 1e-4);
        assert!(scan.points[0].xyz[1].abs() < 1e-4);
        assert!(scan.points[0].xyz[2].abs() < 1e-4);
        assert!((scan.points[0].intensity - 128.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn m1_yaw_of_ninety_degrees_points_left() {
        let spec = ModelSpec::for_model(RobosenseModel::M1);
        let mut first = packet(&spec, 100);
        let channel = HEADER_SIZE + spec.block_header_size;
        let yaw = (ANGLE_OFFSET + 9000) as u16;
        first[channel..channel + 2].copy_from_slice(&2000u16.to_be_bytes());
        first[channel + 2..channel + 4].copy_from_slice(&(ANGLE_OFFSET as u16).to_be_bytes());
        first[channel + 4..channel + 6].copy_from_slice(&yaw.to_be_bytes());

        let mut assembler = ScanAssembler::new(&config(RobosenseModel::M1));
        assembler
            .push_packet(&first, sender(), SystemTime::now())
            .unwrap();
        let scan = assembler
            .push_packet(&packet(&spec, 0), sender(), SystemTime::now())
            .unwrap()
            .unwrap();
        assert!(scan.points[0].xyz[0].abs() < 1e-3);
        assert!((scan.points[0].xyz[1] - 10.0).abs() < 1e-3);
    }

    #[test]
    fn e1_scales_the_unit_vector_by_range() {
        let spec = ModelSpec::for_model(RobosenseModel::E1);
        let mut first = packet(&spec, 100);
        let channel = HEADER_SIZE + spec.block_header_size;
        // 10 m along -Y: the direction triple is full-scale negative on y.
        first[channel..channel + 2].copy_from_slice(&2000u16.to_be_bytes());
        first[channel + 2..channel + 4].copy_from_slice(&0i16.to_be_bytes());
        first[channel + 4..channel + 6].copy_from_slice(&(-32768i16).to_be_bytes());
        first[channel + 6..channel + 8].copy_from_slice(&0i16.to_be_bytes());
        first[channel + 8] = 200;

        let mut assembler = ScanAssembler::new(&config(RobosenseModel::E1));
        assembler
            .push_packet(&first, sender(), SystemTime::now())
            .unwrap();
        let scan = assembler
            .push_packet(&packet(&spec, 0), sender(), SystemTime::now())
            .unwrap()
            .unwrap();
        assert_eq!(scan.points.len(), 1);
        assert!(scan.points[0].xyz[0].abs() < 1e-6);
        assert!((scan.points[0].xyz[1] + 10.0).abs() < 1e-4);
        assert!((scan.points[0].intensity - 200.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn e1_reads_its_block_time_offset_as_sixteen_bits() {
        let spec = ModelSpec::for_model(RobosenseModel::E1);
        let mut first = packet(&spec, 100);
        // 1000 µs does not fit in a byte, so a u8 read would report 0.
        first[HEADER_SIZE..HEADER_SIZE + 2].copy_from_slice(&1000u16.to_be_bytes());
        let channel = HEADER_SIZE + spec.block_header_size;
        first[channel..channel + 2].copy_from_slice(&2000u16.to_be_bytes());
        first[channel + 2..channel + 4].copy_from_slice(&32767i16.to_be_bytes());

        let mut assembler = ScanAssembler::new(&config(RobosenseModel::E1));
        assembler
            .push_packet(&first, sender(), SystemTime::now())
            .unwrap();
        let scan = assembler
            .push_packet(&packet(&spec, 0), sender(), SystemTime::now())
            .unwrap()
            .unwrap();
        assert!((scan.points[0].time_offset_secs - 0.001).abs() < 1e-9);
    }

    #[test]
    fn mx_emits_the_second_return_only_in_dual_echo_mode() {
        let spec = ModelSpec::for_model(RobosenseModel::Mx);
        let mut first = packet(&spec, 100);
        let channel = HEADER_SIZE + spec.block_header_size;
        first[channel..channel + 2].copy_from_slice(&2000u16.to_be_bytes());
        first[channel + 2..channel + 4].copy_from_slice(&4000u16.to_be_bytes());
        first[channel + 4..channel + 6].copy_from_slice(&32767i16.to_be_bytes());
        first[channel + 10] = 10;
        first[channel + 11] = 20;

        let mut single = ScanAssembler::new(&config(RobosenseModel::Mx));
        single
            .push_packet(&first, sender(), SystemTime::now())
            .unwrap();
        let scan = single
            .push_packet(&packet(&spec, 0), sender(), SystemTime::now())
            .unwrap()
            .unwrap();
        assert_eq!(scan.points.len(), 1);
        assert!((scan.points[0].xyz[0] - 10.0).abs() < 1e-3);

        let mut dual = ScanAssembler::new(&config(RobosenseModel::Mx));
        dual.set_echo_mode(EchoMode::Dual);
        dual.push_packet(&first, sender(), SystemTime::now())
            .unwrap();
        let scan = dual
            .push_packet(&packet(&spec, 0), sender(), SystemTime::now())
            .unwrap()
            .unwrap();
        assert_eq!(scan.points.len(), 2);
        assert!((scan.points[1].xyz[0] - 20.0).abs() < 1e-3);
    }

    #[test]
    fn out_of_range_returns_are_dropped() {
        let spec = ModelSpec::for_model(RobosenseModel::M1);
        let mut first = packet(&spec, 100);
        let channel = HEADER_SIZE + spec.block_header_size;
        // 0.05 m is below the 0.2 m minimum range.
        first[channel..channel + 2].copy_from_slice(&10u16.to_be_bytes());
        first[channel + 2..channel + 4].copy_from_slice(&(ANGLE_OFFSET as u16).to_be_bytes());
        first[channel + 4..channel + 6].copy_from_slice(&(ANGLE_OFFSET as u16).to_be_bytes());

        let mut assembler = ScanAssembler::new(&config(RobosenseModel::M1));
        assembler
            .push_packet(&first, sender(), SystemTime::now())
            .unwrap();
        assert!(assembler.points.is_empty());
    }

    #[test]
    fn empty_returns_never_become_points_at_the_origin() {
        // E1 declares a zero minimum range, so an all-zero packet is the case
        // most likely to leak a phantom cluster at the sensor.
        let spec = ModelSpec::for_model(RobosenseModel::E1);
        let mut assembler = ScanAssembler::new(&config(RobosenseModel::E1));
        assembler
            .push_packet(&packet(&spec, 100), sender(), SystemTime::now())
            .unwrap();
        assert!(assembler.points.is_empty());
    }

    #[test]
    fn wrong_packet_size_is_rejected() {
        let mut assembler = ScanAssembler::new(&config(RobosenseModel::M1));
        let short = vec![0u8; 100];
        assert!(
            assembler
                .push_packet(&short, sender(), SystemTime::now())
                .is_err()
        );
    }

    #[test]
    fn foreign_packet_identifier_is_rejected() {
        let spec = ModelSpec::for_model(RobosenseModel::M1);
        let mut bad = packet(&spec, 100);
        bad[0] = 0x00;
        let mut assembler = ScanAssembler::new(&config(RobosenseModel::M1));
        assert!(
            assembler
                .push_packet(&bad, sender(), SystemTime::now())
                .is_err()
        );
    }

    #[test]
    fn sequence_rewind_splits_a_frame_but_reordering_does_not() {
        let mut splitter = SequenceSplitter::default();
        assert!(!splitter.is_frame_start(0));
        assert!(!splitter.is_frame_start(1));
        assert!(!splitter.is_frame_start(40));
        // Within the slack window: a late packet, not a new frame.
        assert!(!splitter.is_frame_start(35));
        assert!(!splitter.is_frame_start(41));
        // Far below: the counter restarted.
        assert!(splitter.is_frame_start(0));
        assert!(!splitter.is_frame_start(1));
    }

    #[test]
    fn sensor_timestamp_is_parsed_as_seconds_and_microseconds() {
        let mut header = [0u8; HEADER_SIZE];
        header[10..16].copy_from_slice(&1_700_000_000u64.to_be_bytes()[2..]);
        header[16..20].copy_from_slice(&500_000u32.to_be_bytes());
        assert!((parse_sensor_timestamp(&header) - 1_700_000_000.5).abs() < 1e-6);
    }

    #[test]
    fn sensor_mode_rejects_a_packet_that_disagrees_with_the_host_clock() {
        let spec = ModelSpec::for_model(RobosenseModel::M1);
        // An all-zero timestamp field is epoch, far from any real host clock.
        let stale = packet(&spec, 100);
        let mut assembler = ScanAssembler::new(&RobosenseConfig {
            model: RobosenseModel::M1,
            ..Default::default()
        });
        assert!(matches!(
            assembler.push_packet(&stale, sender(), SystemTime::now()),
            Err(DriverError::ClockSkew { .. })
        ));
    }

    #[test]
    fn sync_monitor_reports_once_after_the_window_closes() {
        let mut monitor = SyncMonitor::default();
        let start = Instant::now();
        // Inside the window: sampling, no report yet.
        assert!(monitor.observe(100.0, 100.0, start).is_none());
        assert!(
            monitor
                .observe(101.0, 101.0, start + Duration::from_secs(1))
                .is_none()
        );
        let report = monitor
            .observe(104.0, 104.0, start + SYNC_REPORT_WINDOW)
            .expect("window closed");
        assert_eq!(report.samples, 3);
        // Further packets never produce a second report.
        assert!(
            monitor
                .observe(110.0, 110.0, start + Duration::from_secs(30))
                .is_none()
        );
    }

    #[test]
    fn sync_monitor_measures_offset_spread_and_drift() {
        let mut monitor = SyncMonitor::default();
        let start = Instant::now();
        // Offsets of +10 ms, +20 ms, +30 ms: mean 20, spread 20, drift 20.
        monitor.observe(100.010, 100.0, start);
        monitor.observe(101.020, 101.0, start + Duration::from_secs(1));
        let report = monitor
            .observe(103.030, 103.0, start + SYNC_REPORT_WINDOW)
            .expect("window closed");
        assert!((report.mean_offset_secs - 0.020).abs() < 1e-9);
        assert!((report.spread_secs - 0.020).abs() < 1e-9);
        assert!((report.drift_secs - 0.020).abs() < 1e-9);
    }

    #[test]
    fn a_locked_clock_does_not_trip_the_drift_warning() {
        // Constant offset with a little jitter: the hallmark of PTP or GPS lock.
        let steady = SyncReport {
            samples: 600,
            mean_offset_secs: -0.0004,
            spread_secs: 0.0002,
            drift_secs: 0.000_01,
            window_secs: 3.0,
        };
        assert!(!steady.exhausts_budget_within(Duration::from_millis(20), DRIFT_HORIZON));
    }

    #[test]
    fn a_free_running_clock_trips_the_drift_warning() {
        // 2 ms of drift over 3 s is 0.67 ms/s, which eats a 20 ms budget in 30 s.
        let creeping = SyncReport {
            samples: 600,
            mean_offset_secs: 0.001,
            spread_secs: 0.002,
            drift_secs: 0.002,
            window_secs: 3.0,
        };
        assert!(creeping.exhausts_budget_within(Duration::from_millis(20), DRIFT_HORIZON));
        // The direction of the drift must not matter.
        let backwards = SyncReport {
            drift_secs: -0.002,
            ..creeping
        };
        assert!(backwards.exhausts_budget_within(Duration::from_millis(20), DRIFT_HORIZON));
    }

    #[test]
    fn difop_echo_mode_is_read_from_the_return_mode_byte() {
        let mut difop = vec![0u8; DIFOP_PACKET_SIZE];
        difop[0..8].copy_from_slice(&DIFOP_ID);
        difop[44] = 0x00;
        assert_eq!(decode_difop_echo_mode(&difop), Some(EchoMode::Dual));
        difop[44] = 0x04;
        assert_eq!(decode_difop_echo_mode(&difop), Some(EchoMode::Single));
        difop[0] = 0x11;
        assert_eq!(decode_difop_echo_mode(&difop), None);
    }
}

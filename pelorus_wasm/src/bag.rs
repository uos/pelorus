//! Replay of an MCAP bag into an [`Odometer`], reading the file through
//! JavaScript.
//!
//! The reader never holds the whole file. Whenever it needs bytes it reports
//! the range in [`readOffset`](McapBag::read_offset) and
//! [`readLength`](McapBag::read_length), and the caller answers with
//! [`supply`](McapBag::supply), for example from `File.slice()`. Messages come
//! out in log-time order, and only the chunks of the selected topics are read.

use std::collections::{BTreeMap, BTreeSet};
use std::io::SeekFrom;

use mcap::sans_io::{
    IndexedReadEvent, IndexedReader, IndexedReaderOptions, SummaryReadEvent, SummaryReader,
    SummaryReaderOptions,
};
use pelorus::ImuMeasurement;
use ros2_interfaces_jazzy_serde::sensor_msgs::msg::{Imu, PointCloud2};
use wasm_bindgen::prelude::*;

use crate::{Odometer, js_error};

ros_pointcloud2::impl_pointcloud2_for_ros2_interfaces_jazzy_serde!();

const POINTCLOUD2: &str = "sensor_msgs/msg/PointCloud2";
const IMU: &str = "sensor_msgs/msg/Imu";
/// Channel metadata that marks clouds marina stored compressed. The schema
/// stays PointCloud2 and the payload is a cloudini `CompressedPointCloud2`.
const CODEC_KEY: &str = "marina.pointcloud.codec";
const CLOUDINI_CODEC: &str = "cloudini";

/// How a cloud channel stores its clouds.
#[derive(Clone, Copy)]
enum CloudCodec {
    Plain,
    Cloudini,
}

fn decode_cloud(data: &[u8], codec: CloudCodec) -> Result<pelorus::PointCloud2Msg, JsError> {
    match codec {
        CloudCodec::Plain => {
            let cloud: PointCloud2 = cdr::deserialize(data)
                .map_err(|e| js_error(format!("could not decode a cloud: {e}")))?;
            Ok(impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(cloud))
        }
        CloudCodec::Cloudini => {
            let cloud: cloudini::ros::CompressedPointCloud2 = cdr::deserialize(data)
                .map_err(|e| js_error(format!("could not decode a compressed cloud: {e}")))?;
            cloud
                .decompress()
                .map_err(|e| js_error(format!("could not decompress a cloud: {e}")))
        }
    }
}

/// What [`McapBag::next`] stopped at.
#[wasm_bindgen]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BagStep {
    /// Supply the range in `readOffset` and `readLength`, then call again.
    NeedData = 0,
    /// A cloud went into the odometer. Collect its scans with `nextScan`.
    Cloud = 1,
    /// Every message of the selected topics is replayed.
    End = 2,
}

/// A topic of the bag.
#[wasm_bindgen]
pub struct TopicInfo {
    topic: String,
    message_type: String,
    messages: u64,
}

#[wasm_bindgen]
impl TopicInfo {
    #[wasm_bindgen(getter)]
    pub fn topic(&self) -> String {
        self.topic.clone()
    }

    /// The ROS message type, for example `sensor_msgs/msg/PointCloud2`.
    #[wasm_bindgen(getter, js_name = messageType)]
    pub fn message_type(&self) -> String {
        self.message_type.clone()
    }

    /// Number of messages, from the bag's statistics.
    #[wasm_bindgen(getter)]
    pub fn messages(&self) -> f64 {
        self.messages as f64
    }
}

enum State {
    Summary {
        reader: Box<SummaryReader>,
        pos: u64,
        /// Bytes the summary reader asked for at `pos`.
        pending: Option<usize>,
    },
    Loaded,
    Playing {
        reader: Box<IndexedReader>,
        cloud_channels: BTreeMap<u16, CloudCodec>,
        imu_channels: BTreeSet<u16>,
        pending: Option<(u64, usize)>,
    },
}

/// An MCAP bag read through JavaScript and replayed into an [`Odometer`].
///
/// Load the summary by answering reads until [`loadSummary`](Self::load_summary)
/// returns `true`, choose the topics with [`play`](Self::play), then call
/// [`next`](Self::next) until it returns [`BagStep::End`].
#[wasm_bindgen]
pub struct McapBag {
    size: u64,
    summary: Option<mcap::Summary>,
    state: State,
    last_log_time: u64,
}

#[wasm_bindgen]
impl McapBag {
    /// A bag of `size` bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(size: f64) -> McapBag {
        console_error_panic_hook::set_once();
        let size = size as u64;
        McapBag {
            size,
            summary: None,
            state: State::Summary {
                reader: Box::new(SummaryReader::new_with_options(SummaryReaderOptions {
                    file_size: Some(size),
                    ..Default::default()
                })),
                pos: 0,
                pending: None,
            },
            last_log_time: 0,
        }
    }

    /// Offset of the bytes to supply next, or `undefined` when none are needed.
    #[wasm_bindgen(js_name = readOffset)]
    pub fn read_offset(&self) -> Option<f64> {
        match &self.state {
            State::Summary {
                pos,
                pending: Some(_),
                ..
            } => Some(*pos as f64),
            State::Playing {
                pending: Some((offset, _)),
                ..
            } => Some(*offset as f64),
            _ => None,
        }
    }

    /// Number of bytes to supply next. A read at the end of the file may
    /// supply fewer.
    #[wasm_bindgen(js_name = readLength)]
    pub fn read_length(&self) -> usize {
        match &self.state {
            State::Summary {
                pos,
                pending: Some(length),
                ..
            } => (*length).min(self.size.saturating_sub(*pos) as usize),
            State::Playing {
                pending: Some((_, length)),
                ..
            } => *length,
            _ => 0,
        }
    }

    /// Answer the read reported by `readOffset` and `readLength`.
    pub fn supply(&mut self, bytes: &[u8]) -> Result<(), JsError> {
        match &mut self.state {
            State::Summary {
                reader,
                pos,
                pending,
            } => {
                let length = pending
                    .take()
                    .ok_or_else(|| js_error("no read is pending"))?;
                let n = bytes.len().min(length);
                reader.insert(length)[..n].copy_from_slice(&bytes[..n]);
                reader.notify_read(n);
                *pos += n as u64;
            }
            State::Playing {
                reader, pending, ..
            } => {
                let (offset, _) = pending
                    .take()
                    .ok_or_else(|| js_error("no read is pending"))?;
                reader
                    .insert_chunk_record_data(offset, bytes)
                    .map_err(js_error)?;
            }
            State::Loaded => return Err(js_error("no read is pending")),
        }
        Ok(())
    }

    /// Read the summary. Returns `false` while it needs bytes, then `true`.
    #[wasm_bindgen(js_name = loadSummary)]
    pub fn load_summary(&mut self) -> Result<bool, JsError> {
        let State::Summary {
            reader,
            pos,
            pending,
        } = &mut self.state
        else {
            return Ok(true);
        };
        if pending.is_some() {
            return Ok(false);
        }
        while let Some(event) = reader.next_event() {
            match event.map_err(js_error)? {
                SummaryReadEvent::ReadRequest(n) => {
                    *pending = Some(n);
                    return Ok(false);
                }
                SummaryReadEvent::SeekRequest(to) => {
                    *pos = match to {
                        SeekFrom::Start(offset) => offset,
                        SeekFrom::End(delta) => self.size.saturating_add_signed(delta),
                        SeekFrom::Current(delta) => pos.saturating_add_signed(delta),
                    };
                    reader.notify_seeked(*pos);
                }
            }
        }
        let State::Summary { reader, .. } = std::mem::replace(&mut self.state, State::Loaded)
        else {
            unreachable!("the state was checked above");
        };
        let summary = reader
            .finish()
            .ok_or_else(|| js_error("MCAP summary missing. Check that the file is complete"))?;
        self.summary = Some(summary);
        Ok(true)
    }

    /// The topics of the bag, once the summary is loaded.
    pub fn topics(&self) -> Result<Vec<TopicInfo>, JsError> {
        let summary = self.loaded_summary()?;
        let counts = summary.stats.as_ref().map(|s| &s.channel_message_counts);
        let mut topics: Vec<TopicInfo> = summary
            .channels
            .values()
            .map(|channel| TopicInfo {
                topic: channel.topic.clone(),
                message_type: channel
                    .schema
                    .as_ref()
                    .map(|schema| schema.name.clone())
                    .unwrap_or_default(),
                messages: counts
                    .and_then(|counts| counts.get(&channel.id))
                    .copied()
                    .unwrap_or(0),
            })
            .collect();
        topics.sort_by(|a, b| a.topic.cmp(&b.topic));
        Ok(topics)
    }

    /// Start the replay of `cloudTopic`, and of `imuTopic` when given.
    pub fn play(&mut self, cloud_topic: &str, imu_topic: Option<String>) -> Result<(), JsError> {
        let summary = self.loaded_summary()?;
        let channels = |topic: &str,
                        message_type: &str|
         -> Result<BTreeMap<u16, CloudCodec>, JsError> {
            let ids: BTreeMap<u16, CloudCodec> = summary
                .channels
                .values()
                .filter(|channel| channel.topic == topic)
                .map(|channel| {
                    let found = channel.schema.as_ref().map(|s| s.name.as_str());
                    if found != Some(message_type) {
                        return Err(js_error(format!(
                            "{topic} carries {}, not {message_type}",
                            found.unwrap_or("no schema")
                        )));
                    }
                    if channel.message_encoding != "cdr" {
                        return Err(js_error(format!(
                            "{topic} is encoded as {}, only cdr is supported",
                            channel.message_encoding
                        )));
                    }
                    let codec = match channel.metadata.get(CODEC_KEY) {
                        Some(codec) if codec.starts_with(CLOUDINI_CODEC) => CloudCodec::Cloudini,
                        Some(codec) => {
                            return Err(js_error(format!(
                                "{topic} is stored with the unsupported codec {codec}"
                            )));
                        }
                        None => CloudCodec::Plain,
                    };
                    Ok((channel.id, codec))
                })
                .collect::<Result<_, _>>()?;
            if ids.is_empty() {
                return Err(js_error(format!("the bag has no topic {topic}")));
            }
            Ok(ids)
        };
        let cloud_channels = channels(cloud_topic, POINTCLOUD2)?;
        let imu_channels: BTreeSet<u16> = match &imu_topic {
            Some(topic) => channels(topic, IMU)?.into_keys().collect(),
            None => BTreeSet::new(),
        };
        let topics = std::iter::once(cloud_topic.to_owned()).chain(imu_topic);
        let reader = Box::new(
            IndexedReader::new_with_options(
                summary,
                IndexedReaderOptions::new().include_topics(topics),
            )
            .map_err(js_error)?,
        );
        self.state = State::Playing {
            reader,
            cloud_channels,
            imu_channels,
            pending: None,
        };
        Ok(())
    }

    /// Replay messages into `odometer` until a cloud went in, bytes are
    /// needed or the bag ends.
    pub fn next(&mut self, odometer: &mut Odometer) -> Result<BagStep, JsError> {
        let State::Playing {
            reader,
            cloud_channels,
            imu_channels,
            pending,
        } = &mut self.state
        else {
            return Err(js_error("call play first"));
        };
        if pending.is_some() {
            return Ok(BagStep::NeedData);
        }
        // Clouds wait for the IMU initialization, which needs IMU messages.
        if imu_channels.is_empty() && !odometer.inner.imu_initialized() {
            return Err(js_error(
                "no IMU topic is replayed, but the configuration expects an IMU. \
                 Set imu.lidar_only = true to run on the lidar alone",
            ));
        }
        while let Some(event) = reader.next_event() {
            match event.map_err(js_error)? {
                IndexedReadEvent::ReadChunkRequest { offset, length } => {
                    *pending = Some((offset, length));
                    return Ok(BagStep::NeedData);
                }
                IndexedReadEvent::Message { header, data } => {
                    self.last_log_time = header.log_time;
                    if let Some(codec) = cloud_channels.get(&header.channel_id) {
                        // Clouds start once the IMU is initialized, as in an offline
                        // replay through Minot.
                        if !odometer.inner.imu_initialized() {
                            continue;
                        }
                        let cloud = decode_cloud(data, *codec)?;
                        odometer.inner.add_pointcloud2(cloud);
                        return Ok(BagStep::Cloud);
                    }
                    if imu_channels.contains(&header.channel_id) {
                        let imu: Imu = cdr::deserialize(data).map_err(|e| {
                            js_error(format!("could not decode an IMU message: {e}"))
                        })?;
                        odometer.inner.add_imu(imu_measurement(&imu));
                    }
                }
            }
        }
        Ok(BagStep::End)
    }

    /// Share of the bag's time span replayed so far, from 0 to 1.
    pub fn progress(&self) -> f64 {
        let Some(stats) = self.summary.as_ref().and_then(|s| s.stats.as_ref()) else {
            return 0.0;
        };
        let span = stats
            .message_end_time
            .saturating_sub(stats.message_start_time);
        if span == 0 {
            return 0.0;
        }
        let done = self.last_log_time.saturating_sub(stats.message_start_time);
        (done as f64 / span as f64).clamp(0.0, 1.0)
    }

    fn loaded_summary(&self) -> Result<&mcap::Summary, JsError> {
        self.summary
            .as_ref()
            .ok_or_else(|| js_error("load the summary first"))
    }
}

fn imu_measurement(imu: &Imu) -> ImuMeasurement {
    let q = &imu.orientation;
    let w = &imu.angular_velocity;
    let a = &imu.linear_acceleration;
    ImuMeasurement {
        timestamp_sec: imu.header.stamp.sec,
        timestamp_nanosec: imu.header.stamp.nanosec,
        orientation: [q.w, q.x, q.y, q.z],
        orientation_covariance: imu.orientation_covariance,
        angular_velocity: [w.x, w.y, w.z],
        angular_velocity_covariance: imu.angular_velocity_covariance,
        linear_acceleration: [a.x, a.y, a.z],
        linear_acceleration_covariance: imu.linear_acceleration_covariance,
    }
}

//! Cut a demo bag for the browser from a ROS 2 MCAP bag.
//!
//! Keeps one lidar and one IMU topic over a time window and stores the clouds
//! compressed with cloudini, the way marina stores them, so [`McapBag`] plays
//! the result like any other bag.
//!
//! The cut starts at the bag start unless `--start` says otherwise. The IMU
//! initialization needs the platform to stand still for its first seconds, so
//! only move the start to another phase where it stands still.
//!
//! ```sh
//! cargo run -p pelorus_wasm --release --example demo_bag -- \
//!     input.mcap demo.mcap --cloud /hesai/pandar --imu /alphasense_driver_ros/imu \
//!     --duration 60 --drop-fields ring --precision 0.01
//! ```
//!
//! [`McapBag`]: pelorus_wasm::McapBag

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom};

use anyhow::{Context, Result, anyhow, bail};
use cloudini::ros::{CompressedPointCloud2, CompressionConfig};
use mcap::records::MessageHeader;
use mcap::sans_io::{IndexedReadEvent, IndexedReader, IndexedReaderOptions, SummaryReadEvent};
use mcap::{Compression, Summary, WriteOptions};

ros_pointcloud2::impl_pointcloud2_for_ros2_interfaces_jazzy_serde!();

const USAGE: &str = "usage: demo_bag <input.mcap> <output.mcap> --cloud <topic> [--imu <topic>] \
[--start <s>] [--duration <s>] [--precision <m>] [--drop-fields <a,b>] [--keep-every <n>]";
const OPTIONS: [&str; 7] = [
    "cloud",
    "imu",
    "start",
    "duration",
    "precision",
    "drop-fields",
    "keep-every",
];
const CODEC_KEY: &str = "marina.pointcloud.codec";

struct Args {
    input: String,
    output: String,
    cloud: String,
    imu: Option<String>,
    start_s: f64,
    duration_s: Option<f64>,
    precision_m: f64,
    /// Point fields left out of the demo clouds.
    drop_fields: Vec<String>,
    /// Keep every n-th point of each cloud.
    keep_every: usize,
}

fn parse_args() -> Result<Args> {
    let mut positional = Vec::new();
    let mut flags = BTreeMap::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(name) = arg.strip_prefix("--") {
            let value = args
                .next()
                .ok_or_else(|| anyhow!("--{name} needs a value"))?;
            flags.insert(name.to_owned(), value);
        } else {
            positional.push(arg);
        }
    }
    let [input, output] = <[String; 2]>::try_from(positional).map_err(|_| anyhow!(USAGE))?;
    let number = |name: &str| -> Result<Option<f64>> {
        flags
            .get(name)
            .map(|v| {
                v.parse()
                    .with_context(|| format!("--{name} is not a number"))
            })
            .transpose()
    };
    let args = Args {
        input,
        output,
        cloud: flags.get("cloud").cloned().ok_or_else(|| anyhow!(USAGE))?,
        imu: flags.get("imu").cloned(),
        start_s: number("start")?.unwrap_or(0.0),
        duration_s: number("duration")?,
        precision_m: number("precision")?.unwrap_or(0.01),
        drop_fields: flags
            .get("drop-fields")
            .map(|v| v.split(',').map(str::to_owned).collect())
            .unwrap_or_default(),
        keep_every: flags
            .get("keep-every")
            .map(|v| v.parse().context("--keep-every is not a whole number"))
            .transpose()?
            .unwrap_or(1),
    };
    if args.keep_every == 0 {
        bail!("--keep-every must be at least 1");
    }
    for name in flags.keys() {
        if !OPTIONS.contains(&name.as_str()) {
            bail!("unknown option --{name}. {USAGE}");
        }
    }
    Ok(args)
}

/// Bytes of one value of a PointCloud2 field datatype.
fn datatype_size(datatype: u8) -> Result<usize> {
    Ok(match datatype {
        1 | 2 => 1,
        3 | 4 => 2,
        5..=7 => 4,
        8 => 8,
        other => bail!("unknown PointCloud2 datatype {other}"),
    })
}

/// `cloud` without the fields in `drop` and with every `keep_every`-th point,
/// packed without padding.
fn thin_cloud(
    mut cloud: ros2_interfaces_jazzy_serde::sensor_msgs::msg::PointCloud2,
    drop: &[String],
    keep_every: usize,
) -> Result<ros2_interfaces_jazzy_serde::sensor_msgs::msg::PointCloud2> {
    if drop.is_empty() && keep_every == 1 {
        return Ok(cloud);
    }
    // (source offset, size) of each kept field, in output order.
    let mut kept = Vec::new();
    let mut fields = Vec::new();
    let mut offset = 0;
    for field in &cloud.fields {
        if drop.contains(&field.name) {
            continue;
        }
        let size = datatype_size(field.datatype)? * field.count as usize;
        kept.push((field.offset as usize, size));
        let mut field = field.clone();
        field.offset = offset as u32;
        fields.push(field);
        offset += size;
    }
    let point_step = offset;
    let source_step = cloud.point_step as usize;
    let points = (cloud.width * cloud.height) as usize;
    let mut data = Vec::with_capacity(points.div_ceil(keep_every) * point_step);
    for point in cloud.data.chunks_exact(source_step).step_by(keep_every) {
        for &(at, size) in &kept {
            data.extend_from_slice(&point[at..at + size]);
        }
    }
    cloud.width = (data.len() / point_step.max(1)) as u32;
    cloud.height = 1;
    cloud.fields = fields;
    cloud.point_step = point_step as u32;
    cloud.row_step = data.len() as u32;
    cloud.data = data;
    Ok(cloud)
}

fn read_summary(file: &mut File) -> Result<Summary> {
    let mut reader = mcap::sans_io::SummaryReader::new();
    while let Some(event) = reader.next_event() {
        match event? {
            SummaryReadEvent::ReadRequest(n) => {
                let read = file.read(reader.insert(n))?;
                reader.notify_read(read);
            }
            SummaryReadEvent::SeekRequest(to) => {
                reader.notify_seeked(file.seek(to)?);
            }
        }
    }
    reader
        .finish()
        .ok_or_else(|| anyhow!("the MCAP summary is missing. Check that the file is complete"))
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let mut input =
        File::open(&args.input).with_context(|| format!("could not open {}", args.input))?;
    let summary = read_summary(&mut input)?;
    let bag_start = summary
        .stats
        .as_ref()
        .map(|s| s.message_start_time)
        .ok_or_else(|| anyhow!("the bag has no statistics"))?;
    let start = bag_start + (args.start_s * 1e9) as u64;
    if args.start_s > 0.0 {
        eprintln!(
            "the cut starts {:.1} s into the bag. The IMU initialization needs the platform to stand still there.",
            args.start_s
        );
    }

    let mut options = IndexedReaderOptions::new()
        .include_topics(std::iter::once(args.cloud.clone()).chain(args.imu.clone()))
        .starting_at(start);
    if let Some(duration) = args.duration_s {
        options = options.ending_before(start + (duration * 1e9) as u64);
    }
    let mut reader = IndexedReader::new_with_options(&summary, options)?;

    let mut writer = WriteOptions::new()
        .compression(Some(Compression::Zstd))
        .profile("ros2")
        .create(BufWriter::new(
            File::create(&args.output)
                .with_context(|| format!("could not create {}", args.output))?,
        ))?;

    // Output channel per input channel, and whether its clouds get compressed.
    let mut channels: BTreeMap<u16, (u16, bool)> = BTreeMap::new();
    let mut chunk = Vec::new();
    let (mut clouds, mut imus, mut raw_bytes, mut cloud_bytes) = (0u64, 0u64, 0u64, 0u64);
    let compression = CompressionConfig::lossy_zstd(args.precision_m as f32);

    while let Some(event) = reader.next_event() {
        match event? {
            IndexedReadEvent::ReadChunkRequest { offset, length } => {
                input.seek(SeekFrom::Start(offset))?;
                chunk.resize(length, 0);
                input.read_exact(&mut chunk)?;
                reader.insert_chunk_record_data(offset, &chunk)?;
            }
            IndexedReadEvent::Message { header, data } => {
                let channel = summary
                    .channels
                    .get(&header.channel_id)
                    .ok_or_else(|| anyhow!("unknown channel {}", header.channel_id))?;
                let (out_channel, is_cloud) = match channels.get(&header.channel_id) {
                    Some(known) => *known,
                    None => {
                        let schema = channel
                            .schema
                            .as_ref()
                            .ok_or_else(|| anyhow!("{} has no schema", channel.topic))?;
                        let schema_id =
                            writer.add_schema(&schema.name, &schema.encoding, &schema.data)?;
                        let is_cloud = channel.topic == args.cloud;
                        let mut metadata = channel.metadata.clone();
                        if is_cloud {
                            metadata.insert(
                                CODEC_KEY.to_owned(),
                                format!("cloudini/lossy-zstd/{:.9}m", args.precision_m),
                            );
                        }
                        let id = writer.add_channel(
                            schema_id,
                            &channel.topic,
                            &channel.message_encoding,
                            &metadata,
                        )?;
                        channels.insert(header.channel_id, (id, is_cloud));
                        (id, is_cloud)
                    }
                };
                let out_header = MessageHeader {
                    channel_id: out_channel,
                    ..header
                };
                if is_cloud {
                    let cloud: ros2_interfaces_jazzy_serde::sensor_msgs::msg::PointCloud2 =
                        cdr::deserialize(data).context("could not decode a cloud")?;
                    if clouds == 0 {
                        let fields: Vec<String> = cloud
                            .fields
                            .iter()
                            .map(|f| format!("{}@{}:{}", f.name, f.offset, f.datatype))
                            .collect();
                        println!(
                            "first cloud: {} x {} points, {} bytes per point, fields {}",
                            cloud.width,
                            cloud.height,
                            cloud.point_step,
                            fields.join(" ")
                        );
                    }
                    let cloud = thin_cloud(cloud, &args.drop_fields, args.keep_every)?;
                    let compressed = CompressedPointCloud2::compress(
                        impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(cloud),
                        compression.clone(),
                    )
                    .map_err(|e| anyhow!("could not compress a cloud: {e}"))?;
                    let payload = cdr::serialize::<_, _, cdr::CdrLe>(&compressed, cdr::Infinite)?;
                    clouds += 1;
                    raw_bytes += data.len() as u64;
                    cloud_bytes += payload.len() as u64;
                    writer.write_to_known_channel(&out_header, &payload)?;
                } else {
                    imus += 1;
                    writer.write_to_known_channel(&out_header, data)?;
                }
            }
        }
    }
    writer.finish()?;
    let size = std::fs::metadata(&args.output)?.len();
    println!(
        "{clouds} clouds ({:.1} MB compressed to {:.1} MB) and {imus} IMU messages, {:.1} MB in {}",
        raw_bytes as f64 / 1e6,
        cloud_bytes as f64 / 1e6,
        size as f64 / 1e6,
        args.output
    );
    Ok(())
}

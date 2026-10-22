//! Native CPU/wgpu replay comparison using the same SDK and message order as web/.
//! Run a release build; no renderer, one CPU thread, fresh odometers each replay.
//! Usage: benchmark_wgpu bag.mcap config.rl cloud_topic imu_topic [repeats=3] [Multi|Single]

use std::{collections::BTreeMap, time::Instant};

use anyhow::{Context, Result, anyhow, ensure};
use mcap::sans_io::{IndexedReadEvent, IndexedReader, IndexedReaderOptions};
use pelorus::{ImuMeasurement, Odometer};
use ros2_interfaces_jazzy_serde::sensor_msgs::msg::{Imu, PointCloud2};
use serde_json::{Value, json};

ros_pointcloud2::impl_pointcloud2_for_ros2_interfaces_jazzy_serde!();

struct Pose {
    stamp: (i32, u32),
    position: [f64; 3],
    orientation: [f64; 4],
}

struct Run {
    adapter: String,
    initialization_s: f64,
    replay_s: f64,
    cloud_ms: Vec<f64>,
    poses: Vec<Pose>,
}

fn percentile(values: &[f64], fraction: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]
}

impl Run {
    fn metrics(&self) -> Value {
        json!({
            "adapter": self.adapter,
            "initialization_s": self.initialization_s,
            "replay_s": self.replay_s,
            "clouds": self.cloud_ms.len(), "poses": self.poses.len(),
            "odometry_s": self.cloud_ms.iter().sum::<f64>() / 1000.0,
            "cloud_mean_ms": self.cloud_ms.iter().sum::<f64>() / self.cloud_ms.len() as f64,
            "cloud_p50_ms": percentile(&self.cloud_ms, 0.5),
            "cloud_p95_ms": percentile(&self.cloud_ms, 0.95),
        })
    }
}

fn imu_measurement(imu: Imu) -> ImuMeasurement {
    let q = imu.orientation;
    let w = imu.angular_velocity;
    let a = imu.linear_acceleration;
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

async fn replay(
    bytes: &[u8],
    summary: &mcap::Summary,
    config: &str,
    cloud: &str,
    imu: &str,
    gpu: bool,
) -> Result<Run> {
    let setup = Instant::now();
    let mut odometer = Odometer::from_config_source(config, true)?;
    let adapter = if gpu {
        odometer.enable_webgpu().await?;
        odometer.webgpu_status().map_err(|e| anyhow!(e))?.to_owned()
    } else {
        "CPU (single core)".to_owned()
    };
    let initialization_s = setup.elapsed().as_secs_f64();
    let started = Instant::now();
    let options = IndexedReaderOptions::new().include_topics([cloud.to_owned(), imu.to_owned()]);
    let mut reader = IndexedReader::new_with_options(summary, options)?;
    let mut cloud_ms = Vec::new();
    let mut poses = Vec::new();
    while let Some(event) = reader.next_event() {
        match event? {
            IndexedReadEvent::ReadChunkRequest { offset, length } => {
                let at = usize::try_from(offset)?;
                let chunk = bytes
                    .get(at..at.checked_add(length).context("chunk length overflow")?)
                    .context("chunk outside file")?;
                reader.insert_chunk_record_data(offset, chunk)?;
            }
            IndexedReadEvent::Message { header, data } => {
                let channel = &summary.channels[&header.channel_id];
                ensure!(channel.message_encoding == "cdr", "expected CDR messages");
                if channel.topic == imu {
                    odometer.add_imu(imu_measurement(cdr::deserialize(data)?));
                } else if channel.topic == cloud && odometer.imu_initialized() {
                    let cloud = match channel.metadata.get("marina.pointcloud.codec") {
                        Some(codec) if codec.starts_with("cloudini") => {
                            let compressed: cloudini::ros::CompressedPointCloud2 =
                                cdr::deserialize(data)?;
                            compressed
                                .decompress()
                                .map_err(|e| anyhow!("cloud decode: {e}"))?
                        }
                        Some(codec) => anyhow::bail!("unsupported cloud codec {codec}"),
                        None => {
                            let cloud: PointCloud2 = cdr::deserialize(data)?;
                            impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(cloud)
                        }
                    };
                    let processing = Instant::now();
                    odometer.add_pointcloud2(cloud);
                    while let Some(scan) = odometer.next_scan_async().await {
                        poses.push(Pose {
                            stamp: scan.timestamp(),
                            position: scan.position(),
                            orientation: scan.orientation(),
                        });
                    }
                    cloud_ms.push(processing.elapsed().as_secs_f64() * 1000.0);
                    if gpu {
                        odometer
                            .webgpu_status()
                            .map_err(|e| anyhow!("GPU fallback invalidates comparison: {e}"))?;
                    }
                }
            }
        }
    }
    let replay_s = started.elapsed().as_secs_f64();
    ensure!(
        !cloud_ms.is_empty() && !poses.is_empty(),
        "no registered scans; check topic names and initialization"
    );
    Ok(Run {
        adapter,
        initialization_s,
        replay_s,
        cloud_ms,
        poses,
    })
}

fn agreement(cpu: &Run, gpu: &Run) -> Result<Value> {
    ensure!(
        cpu.cloud_ms.len() == gpu.cloud_ms.len(),
        "different cloud counts"
    );
    ensure!(cpu.poses.len() == gpu.poses.len(), "different pose counts");
    let reference: BTreeMap<_, _> = cpu.poses.iter().map(|p| (p.stamp, p)).collect();
    let (mut maximum, mut squared, mut rotation) = (0.0_f64, 0.0_f64, 0.0_f64);
    for pose in &gpu.poses {
        let expected = reference
            .get(&pose.stamp)
            .context("missing matching CPU timestamp")?;
        let distance2: f64 = pose
            .position
            .iter()
            .zip(expected.position)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        maximum = maximum.max(distance2.sqrt());
        squared += distance2;
        let dot: f64 = pose
            .orientation
            .iter()
            .zip(expected.orientation)
            .map(|(a, b)| a * b)
            .sum();
        rotation = rotation.max(2.0 * dot.abs().clamp(0.0, 1.0).acos().to_degrees());
    }
    Ok(
        json!({"matched_poses": gpu.poses.len(), "max_position_difference_m": maximum,
        "rms_position_difference_m": (squared/gpu.poses.len() as f64).sqrt(), "max_rotation_difference_deg": rotation}),
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        (4..=6).contains(&args.len()),
        "usage: benchmark_wgpu bag.mcap config.rl cloud_topic imu_topic [repeats=3] [Multi|Single]"
    );
    let repeats = args
        .get(4)
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(3);
    ensure!(repeats > 0, "at least one measured repeat required");
    let mode = args.get(5).map(String::as_str).unwrap_or("Multi");
    ensure!(
        matches!(mode, "Multi" | "Single"),
        "map must be Multi or Single"
    );
    let bytes = std::fs::read(&args[0])?;
    let summary = mcap::Summary::read(&bytes)?.context("MCAP summary missing")?;
    for topic in [&args[2], &args[3]] {
        ensure!(
            summary.channels.values().any(|c| &c.topic == topic),
            "missing topic {topic}"
        );
    }
    let config = format!(
        "{}\nparallelism.execution_mode = SingleCore\nregistration.map.resolution.type = {mode}\n",
        std::fs::read_to_string(&args[1])?
    );
    futures_lite::future::block_on(async {
        for gpu in [false, true] {
            eprintln!("warmup: {}", if gpu { "wgpu" } else { "CPU" });
            let run = replay(&bytes, &summary, &config, &args[2], &args[3], gpu).await?;
            eprintln!("{}: {:.3} s (warmup excluded)", run.adapter, run.replay_s);
        }
        let mut cpu_runs = Vec::new();
        let mut gpu_runs = Vec::new();
        let mut comparisons = Vec::new();
        for repeat in 0..repeats {
            let mut pair = [None, None];
            for gpu in if repeat % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                eprintln!(
                    "repeat {}: {}",
                    repeat + 1,
                    if gpu { "wgpu" } else { "CPU" }
                );
                let run = replay(&bytes, &summary, &config, &args[2], &args[3], gpu).await?;
                eprintln!(
                    "{}: {:.3} s, {} clouds, {} poses",
                    run.adapter,
                    run.replay_s,
                    run.cloud_ms.len(),
                    run.poses.len()
                );
                pair[usize::from(gpu)] = Some(run);
            }
            let [Some(cpu), Some(gpu)] = pair else {
                unreachable!()
            };
            comparisons.push(agreement(&cpu, &gpu)?);
            cpu_runs.push(cpu.metrics());
            gpu_runs.push(gpu.metrics());
        }
        let median = |runs: &[Value]| {
            percentile(
                &runs
                    .iter()
                    .map(|v| v["replay_s"].as_f64().unwrap())
                    .collect::<Vec<_>>(),
                0.5,
            )
        };
        let cpu_median = median(&cpu_runs);
        let gpu_median = median(&gpu_runs);
        let report = json!({
            "bag": args[0], "config": args[1], "map": mode, "repeats": repeats,
            "method": "release; SingleCore; in-memory MCAP; one excluded warmup per backend; alternating order; setup and rendering excluded; replay includes decompression/IMU/preprocessing/map/filter/readback",
            "cpu": cpu_runs, "wgpu": gpu_runs, "trajectory_comparisons": comparisons,
            "cpu_median_s": cpu_median, "wgpu_median_s": gpu_median, "speedup": cpu_median/gpu_median,
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(())
    })
}

//! Checks that the no_std build of the Pelorus library computes what the std
//! build computes.
//!
//! The driver does the IO, so it always links std. Built without the `std`
//! feature it runs the library's no_std code: libm instead of the system's float
//! maths, spin locks, and a clock that never advances. `mask nostd-check` builds
//! it twice and compares the two runs.
//!
//! ```text
//! pelorus-nostd-check convert <config.rl> <bag.mcap> <lidar topic> <imu topic> <out.stream>
//! pelorus-nostd-check config <config.rl> <out.config>
//! pelorus-nostd-check run <in.stream> <out.scans> [config]
//! pelorus-nostd-check compare <a.scans> <b.scans>
//! ```
//!
//! `convert` and `config` need the `convert` feature. `-` as the stream of
//! `convert` and `run` is the standard output and input, so a bag can stream
//! from one build into the other without a file. The stream carries the
//! configuration in the precision of the build that converted it. `config`
//! writes it in the precision of its own build, for `run` of an `f32` build. `run` writes one line per registered
//! scan: id, stamp, position, orientation, and hashes of the covariances, the
//! twist, the biases and the deskewed points. Floats are written so that they
//! read back to the same bits.

#[cfg(feature = "convert")]
mod convert;
mod stream;

use std::error::Error;
use std::io::{BufRead, BufWriter, Write};

use pelorus::{LioConfig, Odometer, Real, Scan};
use stream::{Reader, Record};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match args.as_slice() {
        #[cfg(feature = "convert")]
        ["convert", config, bag, lidar_topic, imu_topic, out] => {
            convert::convert(config, bag, lidar_topic, imu_topic, out)
        }
        #[cfg(feature = "convert")]
        ["config", config, out] => convert::config(config, out),
        ["run", input, out] => run(input, out, None),
        ["run", input, out, config] => run(input, out, Some(config)),
        ["compare", a, b] => compare(a, b),
        _ => Err(concat!(
            "usage:\n",
            "  pelorus-nostd-check convert <config.rl> <bag.mcap> <lidar topic> <imu topic> <out.stream>\n",
            "  pelorus-nostd-check config <config.rl> <out.config>\n",
            "  pelorus-nostd-check run <in.stream> <out.scans> [config]\n",
            "  pelorus-nostd-check compare <a.scans> <b.scans>"
        )
        .into()),
    }
}

fn run(input: &str, out: &str, config: Option<&str>) -> Result<(), Box<dyn Error>> {
    let file: Box<dyn std::io::Read> = if input == "-" {
        Box::new(std::io::stdin().lock())
    } else {
        Box::new(std::fs::File::open(input)?)
    };
    let (stream_config, mut reader) =
        Reader::open(std::io::BufReader::with_capacity(1 << 20, file))?;
    let config = match config {
        Some(path) => std::fs::read(path)?,
        None => stream_config,
    };
    let config: LioConfig = postcard::from_bytes(&config)
        .map_err(|e| format!("the configuration does not decode in this precision: {e}"))?;
    let mut odometer = Odometer::from_config(config, true)?;
    let mut out = BufWriter::new(std::fs::File::create(out)?);
    let mut scans = 0_usize;
    while let Some(record) = reader.next_record()? {
        match record {
            Record::Imu(imu) => odometer.add_imu(imu),
            Record::Cloud(cloud) => {
                odometer.add_pointcloud2(cloud);
            }
        }
        while let Some(scan) = odometer.next_scan() {
            writeln!(out, "{}", scan_line(&scan))?;
            scans += 1;
        }
    }
    out.flush()?;
    let build = match (cfg!(feature = "std"), pelorus_core::REAL_IS_F32) {
        (true, false) => "std",
        (true, true) => "std f32",
        (false, false) => "no_std",
        (false, true) => "no_std f32",
    };
    eprintln!("{build}: {scans} scans registered");
    Ok(())
}

fn scan_line(scan: &Scan<'_>) -> String {
    let (sec, nanosec) = scan.timestamp();
    let [x, y, z] = scan.position();
    let [qw, qx, qy, qz] = scan.orientation();
    let (linear, angular) = scan.twist();
    let state = fnv1a(
        scan.pose_covariance()
            .iter()
            .chain(scan.twist_covariance())
            .chain(&linear)
            .chain(&angular)
            .chain(&scan.accel_bias())
            .chain(&scan.gyro_bias())
            .map(|value| bits(*value)),
    );
    let points = fnv1a(scan.points().iter().flat_map(|p| {
        [p.x, p.y, p.z, p.intensity, p.timestamp].map(|value| u64::from(value.to_bits()))
    }));
    // `{:?}` prints the shortest decimal that reads back to the same f64.
    format!(
        "{} {sec}.{nanosec:09} {x:?} {y:?} {z:?} {qw:?} {qx:?} {qy:?} {qz:?} {state:016x} {points:016x} {}",
        scan.scan_id(),
        scan.points().len(),
    )
}

/// The bits of a [`Real`], widened to 64.
fn bits(value: Real) -> u64 {
    if pelorus_core::REAL_IS_F32 {
        u64::from(pelorus_core::single(value).to_bits())
    } else {
        pelorus_core::double(value).to_bits()
    }
}

fn fnv1a(values: impl Iterator<Item = u64>) -> u64 {
    values.fold(0xcbf2_9ce4_8422_2325, |hash, value| {
        value.to_le_bytes().iter().fold(hash, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
        })
    })
}

/// One line of a `run` output.
struct ScanRecord {
    line: String,
    position: [f64; 3],
    orientation: [f64; 4],
}

fn read_scans(path: &str) -> Result<Vec<ScanRecord>, Box<dyn Error>> {
    let file = std::io::BufReader::new(std::fs::File::open(path)?);
    file.lines()
        .map(|line| {
            let line = line?;
            let values: Vec<f64> = line
                .split(' ')
                .skip(2)
                .take(7)
                .map(str::parse)
                .collect::<Result<_, _>>()
                .map_err(|e| format!("{path}: {e} in '{line}'"))?;
            let [x, y, z, qw, qx, qy, qz] = values[..] else {
                return Err(format!("{path}: malformed line '{line}'").into());
            };
            Ok(ScanRecord {
                line,
                position: [x, y, z],
                orientation: [qw, qx, qy, qz],
            })
        })
        .collect()
}

fn compare(a: &str, b: &str) -> Result<(), Box<dyn Error>> {
    let (a_scans, b_scans) = (read_scans(a)?, read_scans(b)?);
    let identical = a_scans
        .iter()
        .zip(&b_scans)
        .filter(|(a, b)| a.line == b.line)
        .count();
    let mut max_position = 0.0_f64;
    let mut max_angle = 0.0_f64;
    for (a, b) in a_scans.iter().zip(&b_scans) {
        let distance = (0..3)
            .map(|i| (a.position[i] - b.position[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        max_position = max_position.max(distance);
        // Angle between the two orientations, from the quaternion dot product.
        let dot = (0..4)
            .map(|i| a.orientation[i] * b.orientation[i])
            .sum::<f64>()
            .abs()
            .min(1.0);
        max_angle = max_angle.max(2.0 * dot.acos());
    }
    println!(
        "{a}: {} scans, {b}: {} scans, {identical} identical lines",
        a_scans.len(),
        b_scans.len()
    );
    if a_scans.len() == b_scans.len() && identical == a_scans.len() {
        println!("identical: poses, covariances, twists, biases and points match bit for bit");
        return Ok(());
    }
    if let Some((index, (a, b))) = a_scans
        .iter()
        .zip(&b_scans)
        .enumerate()
        .find(|(_, (a, b))| a.line != b.line)
    {
        println!(
            "first difference at line {}:\n  {}\n  {}",
            index + 1,
            a.line,
            b.line
        );
    }
    println!(
        "largest position difference {max_position:e} m, largest rotation difference {max_angle:e} rad"
    );
    Err("the runs differ".into())
}

pub mod imu;
mod naive_preint;
mod naive_preint_eth;

use log::{error, info, warn};

use nalgebra::*;
use pelorus_core::log_debouncer::{set_default_debounce_secs, should_log_once_default};
use pelorus_core::{
    ImuCalibrationParameters, ImuMsg, ImuPosePrediction, ImuPredictor, ImuPredictorUndistort,
    LidarVendor, LifisParameters, MadgwickSetting, OdomMsg, PointXYZIT, PointXYZITimestamp,
    PrelioProcessedCloud, Twist, time_msg_from_secs, timing::RuntimeEvaluator,
};
#[cfg(feature = "cuda")]
use pelorus_cuda::CudaBuffer;
use std::thread;

use anyhow::Result;
use ros_pointcloud2::prelude::*;
pub use ros_pointcloud2::ros::TimeMsg;

#[derive(Clone)]
pub struct LabeledPointCloud2Msg {
    pub scan_id: u64,
    pub cloud: PointCloud2Msg,
}

use imu::*;

#[inline]
fn safe_angle(angle: f64) -> f64 {
    let factor = (angle / (2.0 * std::f64::consts::PI)).floor() as i32;
    angle - factor as f64 * 2.0 * std::f64::consts::PI
}

#[inline]
fn angle2d(v1: Vector2<f32>, v2: Vector2<f32>) -> f32 {
    v1.y.atan2(v1.x) - v2.y.atan2(v2.x)
}

#[inline]
fn inside_box(p: Point3<f32>, min: Vector3<f32>, max: Vector3<f32>) -> bool {
    p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y && p.z >= min.z && p.z <= max.z
}

#[inline]
fn inside_horizontal_fov(
    p: Point3<f32>,
    fov_right: f32,
    fov_left: f32,
    forward: Vector2<f32>,
) -> bool {
    let angle = angle2d(forward, Vector2::new(p.x, p.y));
    let fov_angle = safe_angle((fov_right - fov_left).into());
    let local_angle = safe_angle((angle - fov_left).into());
    local_angle <= fov_angle
}

#[inline]
fn point_inside_polygon_winding_number(
    p: Point3<f32>,
    polygon: &[Point2<f32>],
    polygon_size: usize,
) -> bool {
    let mut winding_number = 0;
    for i in 0..polygon_size {
        let v1 = polygon[i];
        let v2 = polygon[(i + 1) % polygon_size];

        let cont =
            p.y < v1.y.min(v2.y) || p.y > v1.y.max(v2.y) || p.x > v1.x.max(v2.x) || v1.y == v2.y;

        if cont {
            continue;
        }

        let y_slope = (v2.x - v1.x) / (v2.y - v1.y);
        let x_intercept = (p.y - v1.y) * y_slope + v1.y;
        if v1.x == v2.x || p.x <= x_intercept {
            winding_number += if v2.y > v1.y { 1 } else { -1 };
        }
    }

    winding_number != 0
}

pub fn timestamp_to_secs(t: &TimeMsg) -> f64 {
    t.sec as f64 + t.nanosec as f64 / 1_000_000_000.0
}

// TODO check correctness
pub fn secs_to_timestamp(secs: f64) -> TimeMsg {
    let sec = secs as i32;
    let nanosec = ((secs - sec as f64) * 1_000_000_000.0) as u32;
    TimeMsg { sec, nanosec }
}

#[derive(Debug, Clone, Default)]
pub struct MeasureGroup {
    cloud: Vec<PointXYZIT>,
    last_lidar_end_stamp: f64,
    lidar_start_stamp: f64,
    lidar_end_stamp: f64,
    imu_buffer: Vec<ImuMsg>,
    odom_buffer: Vec<OdomMsg>,
}

pub enum LocatedBufferView<T> {
    Ram(Vec<T>),
    #[cfg(feature = "cuda")]
    Cuda {
        buffer: pelorus_cuda::CudaBuffer,
    },
}

impl<T> LocatedBufferView<T> {
    #[cfg(feature = "cuda")]
    pub fn alloc_points_cuda(stream: &pelorus_cuda::Stream, len: usize) -> Self {
        Self::Cuda {
            buffer: CudaBuffer::new(stream, size_of::<PointXYZI>(), len),
        }
    }
    #[cfg(feature = "cuda")]
    pub fn alloc_neighbors_cuda(stream: &pelorus_cuda::Stream, len: usize) -> Self {
        Self::Cuda {
            buffer: CudaBuffer::new(stream, size_of::<usize>(), len),
        }
    }
    pub fn from_vec_cpu(src: Vec<T>) -> Self {
        Self::Ram(src)
    }
    #[cfg(feature = "cuda")]
    pub fn copy_from_slice_cuda(stream: &pelorus_cuda::Stream, src: &[PointXYZI]) -> Self {
        Self::Cuda {
            buffer: CudaBuffer::from_cloud(&stream, src),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            LocatedBufferView::Ram(point_xyzs) => point_xyzs.len(),
            #[cfg(feature = "cuda")]
            LocatedBufferView::Cuda { buffer: cloud } => cloud.n,
        }
    }
}

#[cfg(feature = "downsample")]
fn hash(point: &Point3<f32>, voxel_size: f32) -> i32 {
    const PRIME_1: i32 = 73_856_093;
    const PRIME_2: i32 = 19_349_663;
    const PRIME_3: i32 = 83_492_791;
    const PRIME_4: i32 = 116_881;

    let xyz = point * (1.0 / voxel_size);

    let x = xyz.x.round() as i32;
    let y = xyz.y.round() as i32;
    let z = xyz.z.round() as i32;

    let mut hash = x.wrapping_mul(PRIME_1);
    hash = hash.wrapping_add(y.wrapping_mul(PRIME_2));
    hash = hash.wrapping_add(z.wrapping_mul(PRIME_3));
    hash = hash.wrapping_rem(PRIME_4);

    hash
}

/// Helper function to find the medoid within a slice of points.
#[cfg(feature = "downsample")]
fn find_medoid_in_slice(points: &[(i32, PointXYZIT)]) -> PointXYZIT {
    if points.len() == 1 {
        return points[0].1;
    }

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_z = 0.0;
    for (_, p) in points {
        sum_x += p.x;
        sum_y += p.y;
        sum_z += p.z;
    }
    let centroid = Point3::new(
        sum_x / points.len() as f32,
        sum_y / points.len() as f32,
        sum_z / points.len() as f32,
    );

    *points
        .iter()
        .map(|(_, p)| p)
        .min_by(|a, b| {
            let dist_a = (a.x - centroid.x).powi(2)
                + (a.y - centroid.y).powi(2)
                + (a.z - centroid.z).powi(2);
            let dist_b = (b.x - centroid.x).powi(2)
                + (b.y - centroid.y).powi(2)
                + (b.z - centroid.z).powi(2);
            dist_a.partial_cmp(&dist_b).unwrap()
        })
        .unwrap()
}

fn filter_downsample<const N: usize, P: PointXYZITimestamp<N>>(
    cloud: PointCloud2Msg,
    lifis_params: &LifisParameters,
) -> Result<Option<(Vec<PointXYZIT>, f64, f64, HeaderMsg)>> {
    let header = cloud.header.clone();
    #[cfg(feature = "downsample")]
    let voxel_size = lifis_params.downsample.voxel_size;
    #[cfg(feature = "parallel")]
    const PAR_SORT_BOUNDARY: usize = 1; // When tested on an M3, par sort was always faster, no matter the size. TODO test on other platforms
    let iter = {
        #[cfg(feature = "parallel")]
        {
            cloud.try_into_par_iter()?
        }
        #[cfg(not(feature = "parallel"))]
        {
            cloud.try_into_iter()?
        }
    };

    #[cfg(feature = "downsample")]
    type PointsFiltered = (i32, PointXYZIT);
    #[cfg(not(feature = "downsample"))]
    type PointsFiltered = PointXYZIT;

    // NOTE: prepared start for when other lidar vendors get ideas of doing absolute timestamps...
    let (_, end_t) = match lifis_params.lidar_vendor {
        LidarVendor::Velodyne | LidarVendor::LivoxMid360 => {
            (None, Some(timestamp_to_secs(&header.stamp)))
        }
        _ => (Some(timestamp_to_secs(&header.stamp)), None),
    };

    #[allow(unused_mut)]
    let mut keyed_points: Vec<PointsFiltered> = iter
        .filter_map(|point: P| {
            let params = &lifis_params.passthrough_filter;
            let p_t = params.filter_frame_transform * point.xyz();

            let dist = p_t.coords.norm_squared();
            let is_within_range = if params.invert_distance {
                dist < params.min_dist.powi(2) || dist > params.max_dist.powi(2)
            } else {
                dist >= params.min_dist.powi(2) && dist <= params.max_dist.powi(2)
            };
            if !is_within_range {
                return None;
            }

            let is_within_intensity = if params.invert_intensity {
                point.intensity() < params.min_intensity || point.intensity() > params.max_intensity
            } else {
                point.intensity() >= params.min_intensity
                    && point.intensity() <= params.max_intensity
            };
            if !is_within_intensity {
                return None;
            }

            let is_within_box = if params.invert_bounding_box {
                !inside_box(p_t, params.min, params.max)
            } else {
                inside_box(p_t, params.min, params.max)
            };
            if !is_within_box {
                return None;
            }

            let is_within_fov = if params.enable_horizontal_fov {
                if params.invert_fov {
                    !inside_horizontal_fov(p_t, params.fov_right, params.fov_left, params.forward)
                } else {
                    inside_horizontal_fov(p_t, params.fov_right, params.fov_left, params.forward)
                }
            } else {
                true
            };
            if !is_within_fov {
                return None;
            }

            let is_within_polygon = match &params.polygon {
                None => true,
                Some(polygon) => {
                    let is_within_polygon = if params.invert_polygon {
                        !point_inside_polygon_winding_number(p_t, polygon, polygon.len())
                    } else {
                        point_inside_polygon_winding_number(p_t, polygon, polygon.len())
                    };
                    !polygon.is_empty() && is_within_polygon
                }
            };
            if !is_within_polygon {
                return None;
            }

            let ts = point.timestamp();
            if ts.is_nan() {
                return None;
            }

            // absolute to relative ts
            let ts = if matches!(lifis_params.lidar_vendor, LidarVendor::LivoxMid360) {
                let scan_end_t = end_t.expect("vendor match out of date");
                (ts - scan_end_t).max(0.0) // handle driver inaccuracies
            } else {
                ts
            };

            let point_raw = point.xyz();
            #[cfg(feature = "downsample")]
            {
                let key = hash(&point_raw, voxel_size);
                Some((
                    key,
                    PointXYZIT::from_p(point_raw, point.intensity() * 255.0, ts as f32),
                ))
            }
            #[cfg(not(feature = "downsample"))]
            return Some(PointXYZIT::from_p(point_raw, point.intensity() * 255.0, ts));
        })
        .collect::<Vec<_>>();

    #[cfg(feature = "downsample")]
    let mut downsampled_cloud = {
        // After this, all points in the same voxel are adjacent
        #[cfg(feature = "parallel")]
        {
            if keyed_points.len() > PAR_SORT_BOUNDARY {
                keyed_points.par_sort_unstable_by_key(|(key, _)| *key);
            } else {
                keyed_points.sort_unstable_by_key(|(key, _)| *key);
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            keyed_points.sort_unstable_by_key(|(key, _)| *key);
        }

        if keyed_points.is_empty() {
            return Ok(None);
        }

        let downsampled_cloud: Vec<PointXYZIT> = {
            // TODO chunk by seems extremely fast on M3 Pro
            // #[cfg(feature = "parallel")]
            // {
            //     if keyed_points.len() > PAR_SORT_BOUNDARY {
            //         keyed_points
            //             .par_chunk_by(|a, b| a.0 == b.0)
            //             .map(|group_slice| find_medoid_in_slice(group_slice))
            //             .collect()
            //     } else {
            //         keyed_points
            //             .chunk_by(|a, b| a.0 == b.0)
            //             .map(|group_slice| find_medoid_in_slice(group_slice))
            //             .collect()
            //     }
            // }
            // #[cfg(not(feature = "parallel"))]
            // {
            keyed_points
                .chunk_by(|a, b| a.0 == b.0)
                .map(find_medoid_in_slice)
                .collect()
            // }
        };
        downsampled_cloud
    };
    #[cfg(not(feature = "downsample"))]
    let mut downsampled_cloud = keyed_points;

    // needed for undistort
    let cmp = |a: &PointXYZIT, b: &PointXYZIT| {
        // if (a.timestamp - b.timestamp).abs() < 1e-6 {
        //     a.z.partial_cmp(&b.z).unwrap()
        // } else {
        a.timestamp.partial_cmp(&b.timestamp).unwrap()
        // }
    };
    #[cfg(feature = "parallel")]
    {
        if downsampled_cloud.len() > PAR_SORT_BOUNDARY {
            downsampled_cloud.par_sort_unstable_by(cmp);
        } else {
            downsampled_cloud.sort_unstable_by(cmp);
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        downsampled_cloud.sort_unstable_by(cmp);
    }

    let (first, last) = match (&downsampled_cloud.first(), &downsampled_cloud.last()) {
        (Some(first), Some(last)) if (last.timestamp > 0.0) => (first.timestamp, last.timestamp),
        _ => return Ok(None),
    };

    // TODO back to keeping the timestamp f64, only testing right now because of artefacts
    Ok(Some((downsampled_cloud, first as f64, last as f64, header)))
}

pub struct Prelio {
    preproc_sender: std::sync::mpsc::Sender<LabeledPointCloud2Msg>,
    imu_sender: std::sync::mpsc::Sender<ImuMsg>,
    odom_sender: std::sync::mpsc::Sender<OdomMsg>,
    synced_receiver: Option<std::sync::mpsc::Receiver<PrelioProcessedCloud>>,
    last_cloud_time: f64,
    imu_filter_fusion: ImuMadgwick,
    last_imu_time: f64,
    runtime_eval: RuntimeEvaluator,
    imu_calibration_apply: Option<imu_calib::CalibrationApplication>,
    madgwick_filter: MadgwickSetting,
}

// fn find_sort_boundaries() {
//     let sizes = [
//         10_000, 20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 150_000, 170_000, 200_000,
//         500_000,
//     ];
//     let max_size = *sizes.iter().max().unwrap_or(&1_000_000);

//     println!(
//         "{:<12} | {:<20} | {:<20} | {}",
//         "Vec Size", "Sequential Sort", "Parallel Sort", "Winner"
//     );
//     println!("{:-<13}|{:-<22}|{:-<22}|{:-<10}", "", "", "", "");

//     let mut rng = rand::rng();
//     let master_points: Vec<nalgebra::Point3<f32>> = (0..max_size)
//         .map(|_| {
//             nalgebra::Point3::new(
//                 rng.random_range(-1.0..1.0),
//                 rng.random_range(-1.0..1.0),
//                 rng.random_range(-1.0..1.0),
//             )
//         })
//         .collect();

//     for &size in &sizes {
//         let points = master_points[..size].to_vec();

//         let mut seq_points = points.clone();
//         let start_seq = std::time::Instant::now();
//         seq_points.sort_unstable_by(|a, b| {
//             a.coords
//                 .norm_squared()
//                 .partial_cmp(&b.coords.norm_squared())
//                 .unwrap()
//         });
//         let seq_duration = start_seq.elapsed();

//         let mut par_points = points.clone();
//         let start_par = std::time::Instant::now();
//         par_points.par_sort_unstable_by(|a, b| {
//             a.coords
//                 .norm_squared()
//                 .partial_cmp(&b.coords.norm_squared())
//                 .unwrap()
//         });
//         let par_duration = start_par.elapsed();

//         // --- 4. Report Results ---
//         let winner = if par_duration < seq_duration {
//             "Parallel"
//         } else {
//             "Sequential"
//         };
//         println!(
//             "{:<12} | {:<20.2?} | {:<20.2?} | {}",
//             size, seq_duration, par_duration, winner
//         );
//     }
// }

// /// Runs the benchmark to compare sequential vs parallel chunk iteration.
// fn find_chunking_boundaries() {
//     // The total vector sizes we want to test.
//     let sizes = [
//         1_000, 10_000, 20_000, 40_000, 60_000, 80_000, 100_000, 200_000, 500_000, 1_000_000,
//         5_000_000,
//     ];
//     let max_size = *sizes.iter().max().unwrap_or(&1_000_000);

//     // The size of each chunk we will process.
//     let chunk_size = 24;

//     println!(
//         "This benchmark compares iterating over data with `chunks_exact` vs. `par_chunks_exact`."
//     );
//     println!("We will process the data by summing the squared distances of all points.");
//     println!("Chunk Size for all tests: {}", chunk_size);
//     println!();
//     println!(
//         "{:<12} | {:<25} | {:<25} | {}",
//         "Vec Size", "Sequential Chunks (sum)", "Parallel Chunks (sum)", "Winner"
//     );
//     println!("{:-<13}|{:-<27}|{:-<27}|{:-<10}", "", "", "", "");

//     // --- 1. Data Generation ---
//     println!(
//         "Generating {} master points for consistent benchmarking...",
//         max_size
//     );
//     let mut rng = rand::rng();
//     let master_points: Vec<nalgebra::Point3<f32>> = (0..max_size)
//         .map(|_| {
//             nalgebra::Point3::new(
//                 rng.random_range(-1.0..1.0),
//                 rng.random_range(-1.0..1.0),
//                 rng.random_range(-1.0..1.0),
//             )
//         })
//         .collect();
//     println!("Generation complete. Starting benchmarks.\n");

//     for &size in &sizes {
//         if size < chunk_size {
//             continue; // Skip sizes smaller than one chunk.
//         }
//         let points = &master_points[..size];

//         // --- 2. Sequential Chunking Benchmark ---
//         let start_seq = std::time::Instant::now();
//         // Iterate over chunks sequentially, map each chunk to its sum, and sum the results.
//         let mut seq_total: f32 = points
//             .chunks_exact(chunk_size)
//             .map(|chunk| chunk.iter().map(|p| p.coords.norm_squared()).sum::<f32>())
//             .sum();
//         // Don't forget to process the remainder!
//         seq_total += points
//             .chunks_exact(chunk_size)
//             .remainder()
//             .iter()
//             .map(|p| p.coords.norm_squared())
//             .sum::<f32>();
//         let seq_duration = start_seq.elapsed();

//         // --- 3. Parallel Chunking Benchmark ---
//         let start_par = std::time::Instant::now();
//         // Iterate over chunks in parallel, map each chunk to its sum in parallel, and sum the results.
//         let mut par_total: f32 = points
//             .par_chunks_exact(chunk_size)
//             .map(|chunk| chunk.iter().map(|p| p.coords.norm_squared()).sum::<f32>())
//             .sum();
//         // Process the remainder in parallel as well.
//         par_total += points
//             .par_chunks_exact(chunk_size)
//             .remainder()
//             .par_iter()
//             .map(|p| p.coords.norm_squared())
//             .sum::<f32>();
//         let par_duration = start_par.elapsed();

//         // --- 4. Report Results ---
//         // We can assert that the results are the same (within a small float tolerance)
//         // to ensure both methods are doing the same work.
//         // assert!((seq_total - par_total).abs() < 1e-6);

//         let winner = if par_duration < seq_duration {
//             "Parallel"
//         } else {
//             "Sequential"
//         };
//         println!(
//             "{:<12} | {:<25.2?} | {:<25.2?} | {}",
//             size, seq_duration, par_duration, winner
//         );
//     }
// }

impl Prelio {
    pub fn create<const N: usize, P: PointXYZITimestamp<N>>(
        gravity: f64,
        madgwick_filter: MadgwickSetting,
        imu_intrinsic_calibration_params: &ImuCalibrationParameters,
        params: &LifisParameters,
        imu_pose_prediction_undistort: Option<Box<ImuPredictorUndistort>>,
        imu_pose_prediction_init_guess: Option<Box<ImuPredictor>>,
        lidar_vendor: Option<LidarVendor>,
        ready_for_next: Option<std::sync::mpsc::Receiver<()>>,
    ) -> anyhow::Result<Self> {
        let imu_intrinsic_calibration = match imu_intrinsic_calibration_params
            .imu_intrinsics_calib_file_path
            .as_ref()
        {
            Some(imu_intrinsic_calibration_path) => {
                let imu_calib = imu_calib::CalibrationApplication::create(
                    imu_intrinsic_calibration_params.imu_calibrate_gyros,
                    imu_intrinsic_calibration_params.gyro_calibration_samples,
                    imu_intrinsic_calibration_path.clone(),
                    gravity,
                )?;
                Some(imu_calib)
            }
            None => None,
        };

        let (preproc_tx, preproc_rx) = std::sync::mpsc::channel::<LabeledPointCloud2Msg>();
        let (imu_tx, imu_rx) = std::sync::mpsc::channel();
        let (odom_tx, odom_rx): (
            std::sync::mpsc::Sender<OdomMsg>,
            std::sync::mpsc::Receiver<OdomMsg>,
        ) = std::sync::mpsc::channel();
        // filtered_cloud carries: (cloud_meta, scan_id, runtime_eval)
        let (filtered_cloud_tx, filtered_cloud_rx) = std::sync::mpsc::channel::<(
            (Vec<PointXYZIT>, f64, f64, HeaderMsg),
            u64,
            RuntimeEvaluator,
        )>();
        let (synced_cloud_imu_tx, synced_cloud_imu_rx) = std::sync::mpsc::channel();

        // test parallel boundaries
        // #[cfg(feature = "parallel")]
        // {
        // find_sort_boundaries();
        // find_chunking_boundaries();
        // }

        // sync and imu integration worker
        // TODO move to main processing, since else predict and correct is async
        let intrinsic_given = imu_intrinsic_calibration.is_some();
        let initial_scan_accumulate = params.initial_scan_accumulate;
        let lidar_as_sync = params.lidar_as_sync;

        // configure global default for the log debouncer from parameters
        set_default_debounce_secs(params.warning_debounce);

        thread::spawn(move || {
            let mut last_lidar_frame_time = 0.0;
            let imu_process = ImuProcess::new(gravity, intrinsic_given);

            #[cfg(feature = "no-imu-sync")]
            let imu_rx = None;

            #[cfg(not(feature = "no-imu-sync"))]
            let imu_rx = Some(imu_rx);

            Self::sync_lidar_imu_worker(
                initial_scan_accumulate,
                &mut last_lidar_frame_time,
                gravity,
                filtered_cloud_rx,
                imu_rx,
                odom_rx,
                &synced_cloud_imu_tx,
                intrinsic_given,
                imu_process,
                imu_pose_prediction_undistort,
                imu_pose_prediction_init_guess,
                lidar_vendor,
                lidar_as_sync,
                ready_for_next,
            )
        });

        // preprocessing worker
        let params_clone = params.clone();
        thread::spawn(move || {
            let mut eval = RuntimeEvaluator::new();
            while let Ok(labeled) = preproc_rx.recv() {
                let eval_name = if cfg!(feature = "downsample") {
                    "filter_downsample"
                } else {
                    "filter"
                };
                eval.start(eval_name).unwrap();
                match filter_downsample::<N, P>(labeled.cloud, &params_clone) {
                    Ok(Some(cloud_meta)) => {
                        eval.stop(eval_name).unwrap();
                        if let Err(e) =
                            filtered_cloud_tx.send((cloud_meta, labeled.scan_id, eval.clone()))
                        {
                            error!("Error sending pointcloud: {}", e);
                        }
                    }
                    Ok(None) => {
                        eval.stop(eval_name).unwrap();
                        info!("Pointcloud was filtered out");
                    }
                    Err(e) => {
                        eval.stop(eval_name).unwrap();
                        error!("Error processing pointcloud: {}", e);
                    }
                }
            }
        });

        Ok(Prelio {
            imu_sender: imu_tx,
            odom_sender: odom_tx,
            synced_receiver: Some(synced_cloud_imu_rx),
            preproc_sender: preproc_tx,
            last_cloud_time: 0.0,
            imu_filter_fusion: ImuMadgwick::new(gravity),
            last_imu_time: 0.0,
            runtime_eval: RuntimeEvaluator::new(),
            imu_calibration_apply: imu_intrinsic_calibration,
            madgwick_filter,
        })
    }

    pub fn get_chan(&mut self) -> Option<std::sync::mpsc::Receiver<PrelioProcessedCloud>> {
        self.synced_receiver.take()
    }

    // TODO: debouncing - make sure we don't get too many clouds by buffer length and only accepting every nth cloud, then every nth+1 etc after certain buffer lengths each
    // also, this should be configurable with defaults and toggleable
    /// Add a point cloud to the buffer. Returns a `scan_id` assigned to this cloud.
    pub fn add_pointcloud(&mut self, cloud: PointCloud2Msg) -> u64 {
        let cloud_stamp = timestamp_to_secs(&cloud.header.stamp);

        // TODO handle time jumps in the past?
        // let buff_cloud = if cloud_stamp < self.last_cloud_time {
        //    None
        // } else {
        //     Some(cloud)
        // };

        // self.imu_buffer.clear();
        // {
        //     let mut pointcloud_buffer = self.preproc_sender.
        //     pointcloud_buffer.clear();
        // }
        // self.last_cloud_time = 0.0;
        // self.last_imu_time = 0.0;
        // self.undistort.reset();
        // to_buffer.clear();
        self.last_cloud_time = cloud_stamp;
        // Generate a unique scan id and send a labeled cloud into preprocessing.
        static SCAN_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let scan_id = SCAN_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let labeled = LabeledPointCloud2Msg { scan_id, cloud };
        match self.preproc_sender.send(labeled) {
            Ok(_) => {}
            Err(e) => {
                error!("Error sending pointcloud: {}", e);
            }
        };
        scan_id
    }

    /// Filter raw IMU message and add to the buffer for synchronization and integration.
    /// Returns the filtered IMU message for republishing.
    pub fn add_imu(&mut self, mut imu: ImuMsg) -> Option<ImuMsg> {
        let imu_time = timestamp_to_secs(&imu.stamp);
        if imu_time < self.last_imu_time {
            return None; // wrapping time (mostly bag file restarts), wait for lidar wrap to reset
        }

        self.runtime_eval.start("imu_pre").unwrap();

        let dt = imu_time - self.last_imu_time;
        self.last_imu_time = imu_time;

        let mut gyro = [
            imu.angular_velocity[0],
            imu.angular_velocity[1],
            imu.angular_velocity[2],
        ];

        let mut accel = [
            imu.linear_acceleration[0],
            imu.linear_acceleration[1],
            imu.linear_acceleration[2],
        ];

        // apply calibration if available
        if let Some(imu_calib) = &mut self.imu_calibration_apply {
            let calibrated = imu_calib.imu_callback(imu_calib::ImuMsg::new(&accel, &gyro))?;
            accel = calibrated.linear_acceleration;
            gyro = calibrated.angular_velocity;
        }

        // Madgwick filter for global orientation
        match self.madgwick_filter {
            MadgwickSetting::CreateOrientationOnly => {
                self.imu_filter_fusion
                    .update_no_magnetometer_ros(dt, gyro, accel);
                imu.orientation = self.imu_filter_fusion.get_quaternion();
            }
            MadgwickSetting::FilterAccelerationOnly => {
                self.imu_filter_fusion
                    .update_no_magnetometer_ros(dt, gyro, accel);
                let filtered_acc = self.imu_filter_fusion.get_linear_acceleration_ros();
                imu.linear_acceleration[0] = filtered_acc[0];
                imu.linear_acceleration[1] = filtered_acc[1];
                imu.linear_acceleration[2] = filtered_acc[2];
            }
            MadgwickSetting::FilterAccelerationAndCreateOrientation => {
                self.imu_filter_fusion
                    .update_no_magnetometer_ros(dt, gyro, accel);
                let filtered_acc = self.imu_filter_fusion.get_linear_acceleration_ros();
                imu.orientation = self.imu_filter_fusion.get_quaternion();

                // Update imu message
                imu.linear_acceleration[0] = filtered_acc[0];
                imu.linear_acceleration[1] = filtered_acc[1];
                imu.linear_acceleration[2] = filtered_acc[2];
            }
            MadgwickSetting::Disable => {}
        }

        // Add to buffer
        let imu_data = imu.clone();
        match self.imu_sender.send(imu_data) {
            Ok(_) => {}
            Err(e) => {
                log::error!("Error while sending imu message to sync worker: {e}");
            }
        }
        self.runtime_eval.stop("imu_pre").unwrap();

        Some(imu)
    }

    /// Add an odometry message to the buffer for synchronization with lidar.
    pub fn add_odom(&mut self, odom: OdomMsg) {
        match self.odom_sender.send(odom) {
            Ok(_) => {}
            Err(e) => {
                log::error!("Error while sending odom message to sync worker: {e}");
            }
        }
    }

    /// Syncs lidar, imu, and odometry data by using the timestamps from the headers and timestamps in the lidar message.
    /// A lidar message is therefore mapped to multiple imu/odometry messages that happened in between the previous end stamp and end stamp of the last point in the lidar message.
    pub fn sync_lidar_imu_worker(
        initial_scan_accumulate: usize,
        prev_lidar_end_time: &mut f64,
        gravity: f64,
        lidar_receiver: std::sync::mpsc::Receiver<(
            (Vec<PointXYZIT>, f64, f64, HeaderMsg),
            u64,
            RuntimeEvaluator,
        )>,
        imu_receiver: Option<std::sync::mpsc::Receiver<ImuMsg>>,
        odom_receiver: std::sync::mpsc::Receiver<OdomMsg>,
        synced_sender: &std::sync::mpsc::Sender<PrelioProcessedCloud>,
        imu_intrinsic_calibration_given: bool,
        mut imu_process: ImuProcess,
        imu_pose_prediction_undistort: Option<Box<ImuPredictorUndistort>>,
        imu_pose_prediction_init_guess: Option<Box<ImuPredictor>>,
        lidar_vendor: Option<LidarVendor>,
        lidar_as_sync: bool,
        ready_for_next: Option<std::sync::mpsc::Receiver<()>>,
    ) {
        let mut eval = RuntimeEvaluator::new();
        let mut to_next: Option<ImuMsg> = None;
        let mut cloud_count: usize = 0;
        while let Ok((
            (cloud, lidar_begin_time, lidar_end_time, lidar_header),
            scan_id,
            prelio_eval,
        )) = lidar_receiver.recv()
        {
            eval.merge(&prelio_eval);
            let lidar_time = timestamp_to_secs(&lidar_header.stamp);
            let abs_lidar_end =
                secs_to_timestamp(timestamp_to_secs(&lidar_header.stamp) + lidar_end_time);

            if cloud_count < initial_scan_accumulate {
                cloud_count += 1;
                let res = PrelioProcessedCloud {
                    scan_id: scan_id,
                    cloud,
                    imu_frame: "".to_string(),
                    header: lidar_header,
                    end_of_frame_guess: Isometry3::identity(),
                    end_of_frame_time: time_msg_from_secs(0.0),
                    pose_covariance: [0.0; 36],
                    twist_covariance: [0.0; 36],
                    end_of_frame_twist: Twist {
                        linear: Vector3::zeros(),
                        angular: Vector3::zeros(),
                    },
                    runtime_eval: eval.clone(),
                    should_map_only: true,
                    lidar_end_time_abs: abs_lidar_end,
                };

                synced_sender.send(res).unwrap();
                continue;
            }

            // Handle relative vs absolute timestamps
            let (lidar_begin_time, imu_span_end_time) = match lidar_vendor {
                Some(LidarVendor::Robosense) => (lidar_begin_time, lidar_end_time),
                Some(LidarVendor::LivoxMid360) => (lidar_time - lidar_end_time, lidar_time),
                None | Some(_) => (lidar_begin_time + lidar_time, lidar_end_time + lidar_time),
            };

            // Use 0 as start if there was no last lidar frame which we can use as start for our time range
            let imu_span_start_time = if *prev_lidar_end_time > 0.0 {
                *prev_lidar_end_time
            } else {
                lidar_begin_time
            };

            let mut imu_data = Vec::with_capacity(100);
            let mut odom_data = Vec::with_capacity(100);

            // Get last IMU message that was destructively taken
            if let Some(last_imu_extra) = to_next.take() {
                let imu_time = timestamp_to_secs(&last_imu_extra.stamp);
                if imu_time > imu_span_start_time && imu_time < imu_span_end_time {
                    imu_data.push(last_imu_extra);
                }
            }

            // dbg!(
            //     imu_span_start_time,
            //     imu_span_end_time,
            //     lidar_begin_time,
            //     lidar_end_time
            // );
            // Populate IMU with buffered
            if let Some(imu_receiver) = imu_receiver.as_ref() {
                if lidar_as_sync {
                    loop {
                        match imu_receiver.try_recv() {
                            Ok(imu) => {
                                let imu_time = timestamp_to_secs(&imu.stamp);

                                if imu_time < imu_span_start_time {
                                    // skip
                                    continue;
                                }

                                // Too late: stash for next cloud and stop draining
                                if imu_time > imu_span_end_time {
                                    to_next = Some(imu);
                                    break;
                                }

                                imu_data.push(imu);
                            }
                            Err(std::sync::mpsc::TryRecvError::Empty) => break,
                            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                                error!("Imu receiver closed");
                                return;
                            }
                        }
                    }
                } else {
                    loop {
                        match imu_receiver.recv() {
                            Ok(imu) => {
                                let imu_time = timestamp_to_secs(&imu.stamp);
                                if imu_time < imu_span_start_time {
                                    warn!("IMU msg too early, skipped.");
                                    continue;
                                }

                                // Too late
                                if imu_time > imu_span_end_time {
                                    to_next = Some(imu);
                                    break;
                                }

                                imu_data.push(imu);
                            }
                            Err(e) => {
                                error!("Imu receiver closed: {e}");
                                return;
                            }
                        }
                    }
                }

                if imu_data.is_empty() {
                    warn!("No imu data in span");
                    continue;
                }
            }

            // Populate Odometry with buffered (non-blocking)
            loop {
                match odom_receiver.try_recv() {
                    Ok(odom) => {
                        let odom_time = timestamp_to_secs(&odom.stamp);
                        if odom_time < imu_span_start_time {
                            if should_log_once_default("odom_msg_too_early") {
                                warn!("Odom msg too early, skipped.");
                            }
                            continue;
                        }
                        if odom_time > imu_span_end_time {
                            if should_log_once_default("odom_msg_too_late") {
                                warn!("Odom msg too late, will be picked up in next frame.");
                            }
                            continue;
                        }
                        odom_data.push(odom);
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        error!("Odom receiver closed");
                        return;
                    }
                }
            }

            // Sort buffers with timestamp to ensure correct order
            imu_data.sort_by(|a, b| {
                timestamp_to_secs(&a.stamp)
                    .partial_cmp(&timestamp_to_secs(&b.stamp))
                    .unwrap()
            });
            odom_data.sort_by(|a, b| {
                timestamp_to_secs(&a.stamp)
                    .partial_cmp(&timestamp_to_secs(&b.stamp))
                    .unwrap()
            });

            // - data synced -
            // debug!("data synced");

            let meas = MeasureGroup {
                cloud,
                last_lidar_end_stamp: imu_span_start_time,
                lidar_start_stamp: lidar_begin_time,
                lidar_end_stamp: lidar_end_time,
                imu_buffer: imu_data.clone(),
                odom_buffer: odom_data.clone(),
            };

            if meas.imu_buffer.is_empty() {
                // Passthrough when lidar only
                let cloud = meas.cloud;

                let res = PrelioProcessedCloud {
                    scan_id: scan_id,
                    cloud,
                    imu_frame: "".to_string(),
                    header: lidar_header,
                    end_of_frame_guess: Isometry3::identity(),
                    end_of_frame_time: secs_to_timestamp(meas.lidar_end_stamp),
                    pose_covariance: [0.0; 36],
                    twist_covariance: [0.0; 36],
                    end_of_frame_twist: Twist {
                        linear: Vector3::new(0., 0., 0.),
                        angular: Vector3::new(0., 0., 0.),
                    },
                    runtime_eval: eval.clone(),
                    should_map_only: false,
                    lidar_end_time_abs: abs_lidar_end,
                };
                synced_sender.send(res).unwrap();
            } else {
                // Waiting for sync signal. Needed for cases where undistortion uses an EKF state that predicts forward and the correction must be made first before predicting further.
                if let Some(wait_for_next) = ready_for_next.as_ref() {
                    wait_for_next.recv().unwrap();
                }

                let imu_frame = meas
                    .imu_buffer
                    .first()
                    .expect("what did we just merge?")
                    .frame_id
                    .clone();

                // Deskewing cloud with synced data
                let deskew = imu_process.process(meas, &imu_pose_prediction_undistort);

                if let Some(imu_cloud) = deskew {
                    eval.merge(imu_cloud.runtime_eval);
                    let imu_int = match imu_pose_prediction_init_guess.as_ref() {
                        Some(imu_pose_prediction_init_guess) => imu_pose_prediction_init_guess(
                            &imu_data.clone(),
                            imu_span_start_time,
                            lidar_begin_time,
                            imu_span_end_time,
                        ),
                        None => {
                            let estimated_pose = if imu_intrinsic_calibration_given {
                                eval.start("preint imu").unwrap();
                                // given a calibration, we trust the IMU data and use it for the pose
                                naive_preint_eth::integrate_imu(
                                    &imu_data,
                                    *prev_lidar_end_time,
                                    false, // TODO check ahrs orientation via config or if enabled
                                    gravity,
                                )
                            } else {
                                // no calibration, gyro is used for rotation but nothing else
                                Isometry3::from_parts(
                                    Translation::identity(),
                                    imu_cloud.end_of_frame_guess.rotation,
                                )
                            };

                            if imu_intrinsic_calibration_given {
                                eval.stop("preint imu").unwrap();
                            }

                            ImuPosePrediction {
                                pose: estimated_pose,
                                linear_velocity: Vector3::zeros(),
                                angular_velocity: Vector3::new(
                                    imu_cloud.last_twist[3],
                                    imu_cloud.last_twist[4],
                                    imu_cloud.last_twist[5],
                                ),
                            }
                        }
                    };

                    let res = PrelioProcessedCloud {
                        scan_id: scan_id,
                        cloud: imu_cloud.cloud,
                        imu_frame,
                        header: lidar_header,
                        end_of_frame_guess: imu_int.pose,
                        end_of_frame_time: imu_data.last().unwrap().stamp,
                        pose_covariance: imu_cloud.integrated_covariance,
                        twist_covariance: imu_cloud.last_covariance,
                        end_of_frame_twist: Twist {
                            linear: Vector3::new(
                                imu_cloud.last_twist[0],
                                imu_cloud.last_twist[1],
                                imu_cloud.last_twist[2],
                            ),
                            angular: Vector3::new(
                                imu_cloud.last_twist[3],
                                imu_cloud.last_twist[4],
                                imu_cloud.last_twist[5],
                            ),
                        },
                        runtime_eval: eval.clone(),
                        should_map_only: false,
                        lidar_end_time_abs: abs_lidar_end,
                    };

                    synced_sender.send(res).unwrap();
                }
            }

            *prev_lidar_end_time = imu_span_end_time;
        }
    }
}

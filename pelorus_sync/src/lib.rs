pub mod imu;
mod naive_preint;
mod naive_preint_eth;
#[cfg(not(feature = "rayon"))]
use pelorus_core::adaptive_execution::ExecutionMode;
#[cfg(feature = "rayon")]
use pelorus_core::adaptive_execution::{AdaptiveExecutionPlanner, ExecutionMode, HeavyPhase};
use pelorus_core::{
    BlickfeldPoint, BlickfeldPointNoIntensity, HesaiPoint, HesaiPointNoIntensity,
    SickMultiScanPoint, SickMultiScanPointNoIntensity,
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "threads")]
use log::debug;
use log::{error, info, warn};

use nalgebra::*;
use pelorus_core::log_debouncer::{set_default_debounce_secs, should_log_once_default};
use pelorus_core::{
    ImuCalibrationParameters, ImuMsg, ImuPosePrediction, ImuPredictor, ImuPredictorUndistort,
    LidarVendor, LivoxMid360Point, LivoxMid360PointNoIntensity, LivoxPoint, LivoxPointNoIntensity,
    LivoxPointOffsetTime, LivoxPointOffsetTimeNoIntensity, MadgwickSetting, OdomMsg, OusterPoint,
    OusterPointNoIntensity, PointXYZ, PointXYZI, PointXYZIT, PointXYZITimestamp, PointXYZIntensity,
    RobosensePoint, RobosensePointNoIntensity, SyncParameters, SyncProcessedCloud, Twist,
    VelodynePoint, VelodynePointNoIntensity, time_msg_from_secs, timing::RuntimeEvaluator,
};

#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
use pelorus_core::PreprocessingBackend;
#[cfg(feature = "cuda")]
use pelorus_cuda::CudaBuffer;
#[cfg(feature = "vulkan")]
use pelorus_vulkan::VulkanBuffer;
use std::collections::VecDeque;
use std::sync::Arc;
#[cfg(feature = "threads")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "threads")]
use std::thread;
use pelorus_core::Instant;

use anyhow::Result;
use either::Either::{Left, Right};
use ros_pointcloud2::prelude::*;
pub use ros_pointcloud2::ros::TimeMsg;

/// Preprocessed scan handed to the sync worker, as
/// `((downsampled, passthrough, first_ts, last_ts, header), scan_id, received_at, runtime_eval)`.
pub type FilteredCloudMsg = (
    (
        Vec<PointXYZIT>,
        Option<Vec<PointXYZIT>>,
        f64,
        f64,
        HeaderMsg,
    ),
    u64,
    Instant,
    RuntimeEvaluator,
);

/// Hook that receives the IMU batch used for initialization.
pub type ImuInitCallback = Arc<std::sync::Mutex<dyn FnMut(&[ImuMsg]) + Send + Sync>>;

/// Output-only hook invoked after calibration and AHRS filtering.
pub type RealtimeImuCallback = Arc<dyn Fn(ImuMsg) + Send + Sync>;
/// Output-only hook invoked as soon as a Lidar scan enters preprocessing.
pub type RealtimeLidarBarrierCallback = Arc<dyn Fn() + Send + Sync>;

#[cfg(all(feature = "threads", not(feature = "large-processing-queue")))]
const PREPROC_QUEUE_CAP: usize = 8;
#[cfg(all(feature = "threads", feature = "large-processing-queue"))]
const PREPROC_QUEUE_CAP: usize = 1_024;

#[cfg(all(feature = "threads", not(feature = "large-processing-queue")))]
const IMU_QUEUE_CAP: usize = 16_384;
#[cfg(all(feature = "threads", feature = "large-processing-queue"))]
const IMU_QUEUE_CAP: usize = 1_048_576;

#[cfg(all(feature = "threads", not(feature = "large-processing-queue")))]
const ODOM_QUEUE_CAP: usize = 8_192;
#[cfg(all(feature = "threads", feature = "large-processing-queue"))]
const ODOM_QUEUE_CAP: usize = 1_048_576;

#[cfg(all(feature = "threads", not(feature = "large-processing-queue")))]
const FILTERED_CLOUD_QUEUE_CAP: usize = 8;
#[cfg(all(feature = "threads", feature = "large-processing-queue"))]
const FILTERED_CLOUD_QUEUE_CAP: usize = 1_024;

#[cfg(all(feature = "threads", not(feature = "large-processing-queue")))]
const SYNCED_CLOUD_QUEUE_CAP: usize = 8;
#[cfg(all(feature = "threads", feature = "large-processing-queue"))]
const SYNCED_CLOUD_QUEUE_CAP: usize = 1_024;

#[cfg(feature = "rayon")]
fn sort_unstable_by<T, F>(slice: &mut [T], cmp: F, mode: ExecutionMode)
where
    T: Send,
    F: Fn(&T, &T) -> std::cmp::Ordering + Sync,
{
    match mode {
        ExecutionMode::Sequential => slice.sort_unstable_by(cmp),
        ExecutionMode::Rayon => slice.par_sort_unstable_by(cmp),
    }
}

#[cfg(not(feature = "rayon"))]
fn sort_unstable_by<T, F>(slice: &mut [T], cmp: F, _mode: ExecutionMode)
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    slice.sort_unstable_by(cmp);
}

#[cfg(feature = "rayon")]
fn sort_unstable_by_key<T, K, F>(slice: &mut [T], f: F, mode: ExecutionMode)
where
    T: Send,
    K: Ord + Send,
    F: Fn(&T) -> K + Sync,
{
    match mode {
        ExecutionMode::Sequential => slice.sort_unstable_by_key(f),
        ExecutionMode::Rayon => slice.par_sort_unstable_by_key(f),
    }
}

#[cfg(not(feature = "rayon"))]
fn sort_unstable_by_key<T, K, F>(slice: &mut [T], f: F, _mode: ExecutionMode)
where
    K: Ord,
    F: Fn(&T) -> K,
{
    slice.sort_unstable_by_key(f);
}

pub struct LabeledPointCloud2Msg {
    pub scan_id: u64,
    pub received_at: Instant,
    pub cloud: Box<dyn PreprocessingPointCloud>,
}

/// A queue-owned point cloud that exposes borrowed storage to preprocessing.
pub trait PreprocessingPointCloud: Send {
    fn header(&self) -> HeaderMsg;
    fn source(&self) -> &dyn ros_pointcloud2::PointCloud2Source;
}

impl PreprocessingPointCloud for PointCloud2Msg {
    fn header(&self) -> HeaderMsg {
        self.header.clone()
    }

    fn source(&self) -> &dyn ros_pointcloud2::PointCloud2Source {
        self
    }
}

use imu::*;

pub trait BufferStream {}
#[cfg(feature = "cuda")]
impl BufferStream for pelorus_cuda::Stream {}
#[cfg(feature = "metal")]
impl BufferStream for pelorus_metal::Stream {}
#[cfg(feature = "vulkan")]
impl BufferStream for pelorus_vulkan::Stream {}

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

/// Selects how the synchronizer obtains IMU messages for each Lidar interval.
pub enum ImuSyncMode {
    /// Process Lidar frames without consuming or waiting for IMU messages.
    Disabled {
        /// Apply producer backpressure instead of dropping scans. Enable this
        /// for managed offline replay, not live sensor/topic input.
        lossless: bool,
    },
    /// Wait for timestamp-bracketing IMU messages from a continuous source.
    Continuous,
    /// Drain only the IMU messages already available when a Lidar scan arrives.
    LidarDriven,
    /// During bounded replay, advance one scan when timestamp-bracketing IMU data
    /// has not arrived yet.
    DemandDrivenReplay {
        request_next_scan: Box<dyn FnMut() + Send>,
    },
}

#[derive(Debug, Clone, Default)]
pub struct MeasureGroup {
    cloud: Vec<PointXYZIT>,
    cloud_passthrough: Option<Vec<PointXYZIT>>,
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
    #[cfg(feature = "metal")]
    Metal {
        buffer: pelorus_metal::MetalBuffer,
    },
    #[cfg(feature = "vulkan")]
    Vulkan {
        buffer: pelorus_vulkan::VulkanBuffer,
    },
}

impl<T> LocatedBufferView<T> {
    #[cfg(feature = "cuda")]
    pub fn alloc_points_cuda(stream: &pelorus_cuda::Stream, len: usize) -> Self {
        Self::Cuda {
            buffer: CudaBuffer::new(stream, size_of::<T>(), len),
        }
    }
    #[cfg(feature = "cuda")]
    pub fn alloc_neighbors_cuda(stream: &pelorus_cuda::Stream, len: usize) -> Self {
        Self::Cuda {
            buffer: CudaBuffer::new(stream, size_of::<usize>(), len),
        }
    }
    #[cfg(feature = "metal")]
    pub fn alloc_points_metal(stream: &pelorus_metal::Stream, len: usize) -> Self {
        Self::Metal {
            buffer: pelorus_metal::MetalBuffer::new(stream, size_of::<T>(), len),
        }
    }
    #[cfg(feature = "metal")]
    pub fn alloc_neighbors_metal(stream: &pelorus_metal::Stream, len: usize) -> Self {
        Self::Metal {
            buffer: pelorus_metal::MetalBuffer::new(stream, size_of::<usize>(), len),
        }
    }
    #[cfg(feature = "vulkan")]
    pub fn alloc_points_vulkan(stream: &pelorus_vulkan::Stream, len: usize) -> Self {
        Self::Vulkan {
            buffer: VulkanBuffer::new(stream, size_of::<T>(), len),
        }
    }
    #[cfg(feature = "vulkan")]
    pub fn alloc_neighbors_vulkan(stream: &pelorus_vulkan::Stream, len: usize) -> Self {
        Self::Vulkan {
            buffer: VulkanBuffer::new(stream, size_of::<usize>(), len),
        }
    }
    pub fn from_vec_cpu(src: Vec<T>) -> Self {
        Self::Ram(src)
    }
    #[cfg(feature = "cuda")]
    pub fn copy_from_slice_cuda(stream: &pelorus_cuda::Stream, src: &[PointXYZIT]) -> Self {
        Self::Cuda {
            buffer: CudaBuffer::from_cloud(stream, src),
        }
    }
    #[cfg(feature = "metal")]
    pub fn copy_from_slice_metal(stream: &pelorus_metal::Stream, src: &[PointXYZIT]) -> Self {
        Self::Metal {
            buffer: pelorus_metal::MetalBuffer::from_cloud(stream, src),
        }
    }
    #[cfg(feature = "vulkan")]
    pub fn copy_from_slice_vulkan(stream: &pelorus_vulkan::Stream, src: &[PointXYZIT]) -> Self {
        Self::Vulkan {
            buffer: VulkanBuffer::from_cloud(stream, src),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn realloc_if_needed_cuda(&mut self, stream: &pelorus_cuda::Stream, len: usize) {
        if let LocatedBufferView::Cuda { buffer } = self {
            if buffer.n < len {
                *buffer = pelorus_cuda::CudaBuffer::new(
                    stream,
                    size_of::<T>(),
                    (len as f32 * 1.2) as usize,
                );
            } else {
                buffer.n = len;
            }
        }
    }

    #[cfg(feature = "metal")]
    pub fn realloc_if_needed_metal(&mut self, stream: &pelorus_metal::Stream, len: usize) {
        match self {
            LocatedBufferView::Metal { buffer } => {
                if buffer.capacity() < len {
                    *buffer = pelorus_metal::MetalBuffer::new(
                        stream,
                        size_of::<T>(),
                        (len as f32 * 1.2) as usize,
                    );
                }
                buffer.n = len;
            }
            _ => {}
        }
    }
    #[cfg(feature = "vulkan")]
    pub fn realloc_if_needed_vulkan(&mut self, stream: &pelorus_vulkan::Stream, len: usize) {
        if let LocatedBufferView::Vulkan { buffer } = self {
            if buffer.capacity() < len {
                *buffer = VulkanBuffer::new(stream, size_of::<T>(), (len as f32 * 1.2) as usize);
            }
            buffer.n = len;
        }
    }

    #[cfg(feature = "cuda")]
    pub fn upload_from_slice_cuda(&mut self, stream: &pelorus_cuda::Stream, src: &[PointXYZIT]) {
        if let LocatedBufferView::Cuda { buffer } = self {
            buffer.upload(stream, src);
        }
    }

    #[cfg(feature = "metal")]
    pub fn upload_from_slice_metal(&mut self, stream: &pelorus_metal::Stream, src: &[PointXYZIT]) {
        if let LocatedBufferView::Metal { buffer } = self {
            buffer.upload(stream, src);
        }
    }
    #[cfg(feature = "vulkan")]
    pub fn upload_from_slice_vulkan(
        &mut self,
        stream: &pelorus_vulkan::Stream,
        src: &[PointXYZIT],
    ) {
        if let LocatedBufferView::Vulkan { buffer } = self {
            buffer.upload(stream, src);
        }
    }

    pub fn len(&self) -> usize {
        match self {
            LocatedBufferView::Ram(point_xyzs) => point_xyzs.len(),
            #[cfg(feature = "cuda")]
            LocatedBufferView::Cuda { buffer: cloud } => cloud.n,
            #[cfg(feature = "metal")]
            LocatedBufferView::Metal { buffer } => buffer.n,
            #[cfg(feature = "vulkan")]
            LocatedBufferView::Vulkan { buffer } => buffer.n,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ros_pointcloud2::ros::{HeaderMsg, PointFieldMsg, TimeMsg, make_field_name};
    use ros_pointcloud2::{Denseness, Endian, FieldDatatype, PointCloud2MsgBuilder};
    #[cfg(feature = "threads")]
    use std::sync::atomic::AtomicBool;

    fn field(name: &'static str, offset: u32) -> PointFieldMsg {
        PointFieldMsg {
            name: make_field_name(name),
            offset,
            datatype: FieldDatatype::F32.into(),
            count: 1,
        }
    }

    fn no_timestamp_params() -> SyncParameters {
        let mut params = SyncParameters {
            lidar_vendor: LidarVendor::NoTimestamp,
            ..Default::default()
        };
        params.downsample.voxel_size = 0.01;
        params.downsample.point_skip = 1;
        params.downsample.backend = pelorus_core::PreprocessingBackend::Cpu;
        params.passthrough_filter.min_dist = 0.0;
        params.passthrough_filter.max_dist = 100.0;
        params.passthrough_filter.min_intensity = 0.0;
        params.passthrough_filter.max_intensity = 255.0;
        params.passthrough_filter.min = Vector3::new(-100.0, -100.0, -100.0);
        params.passthrough_filter.max = Vector3::new(100.0, 100.0, 100.0);
        params.passthrough_filter.filter_frame_transform = Isometry3::identity();
        params
    }

    fn cloud(points: &[(f32, f32, f32, Option<f32>)]) -> PointCloud2Msg {
        let has_intensity = points.iter().any(|p| p.3.is_some());
        let point_step = if has_intensity { 16 } else { 12 };
        let mut data = Vec::with_capacity(points.len() * point_step);
        for (x, y, z, intensity) in points {
            data.extend_from_slice(&x.to_le_bytes());
            data.extend_from_slice(&y.to_le_bytes());
            data.extend_from_slice(&z.to_le_bytes());
            if has_intensity {
                data.extend_from_slice(&intensity.unwrap_or(1.0).to_le_bytes());
            }
        }

        let mut fields = vec![field("x", 0), field("y", 4), field("z", 8)];
        if has_intensity {
            fields.push(field("intensity", 12));
        }

        PointCloud2MsgBuilder::new()
            .with_header(HeaderMsg {
                seq: 0,
                stamp: TimeMsg {
                    sec: 123,
                    nanosec: 456,
                },
                frame_id: "lidar".to_string(),
            })
            .with_width(points.len() as u32)
            .with_fields(fields)
            .with_endian(Endian::Little)
            .with_point_step(point_step as u32)
            .with_row_step((points.len() * point_step) as u32)
            .with_data(data)
            .with_dense(Denseness::Dense)
            .build()
            .unwrap()
    }

    fn adversarial_downsample_cloud() -> Vec<PointXYZIT> {
        vec![
            PointXYZIT {
                x: 0.25,
                y: 0.5,
                z: 0.5,
                intensity: 30.0,
                timestamp: 3.0,
            },
            PointXYZIT {
                x: 1.75,
                y: 0.5,
                z: 0.5,
                intensity: 20.0,
                timestamp: 2.0,
            },
            PointXYZIT {
                x: 0.75,
                y: 0.5,
                z: 0.5,
                intensity: 10.0,
                timestamp: 1.0,
            },
            PointXYZIT {
                x: 1.25,
                y: 0.5,
                z: 0.5,
                intensity: 40.0,
                timestamp: 2.0,
            },
            PointXYZIT {
                x: 2.25,
                y: 0.5,
                z: 0.5,
                intensity: 50.0,
                timestamp: 0.0,
            },
        ]
    }

    fn cpu_downsample_reference(points: &[PointXYZIT]) -> Vec<PointXYZIT> {
        let mut eval = RuntimeEvaluator::new();
        let mut buffers = DownsampleBuffers {
            filtered_cloud: points.to_vec(),
            ..Default::default()
        };
        let mut downsampled = Vec::new();
        cpu_downsample(
            &mut buffers,
            1,
            1.0,
            &mut downsampled,
            true,
            &mut eval,
            ExecutionMode::Sequential,
        )
        .unwrap();
        downsampled
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    fn dense_downsample_cloud() -> Vec<PointXYZIT> {
        let voxel_count = 4096usize;
        let points_per_voxel = 16usize;
        let mut points = Vec::with_capacity(voxel_count * points_per_voxel);
        let mut state = 0x9e37_79b9u32;

        for sample in 0..points_per_voxel {
            for voxel in 0..voxel_count {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let jitter_x = ((state & 0xff) as f32 / 255.0 - 0.5) * 0.4;
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let jitter_y = ((state & 0xff) as f32 / 255.0 - 0.5) * 0.4;
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let jitter_z = ((state & 0xff) as f32 / 255.0 - 0.5) * 0.4;

                let vx = (voxel % 32) as i32 - 16;
                let vy = ((voxel / 32) % 32) as i32 - 16;
                let vz = (voxel / (32 * 32)) as i32 - 2;
                points.push(PointXYZIT {
                    x: vx as f32 + 0.5 + jitter_x,
                    y: vy as f32 + 0.5 + jitter_y,
                    z: vz as f32 + 0.5 + jitter_z,
                    intensity: (state & 0xffff) as f32 / 65535.0,
                    timestamp: (sample * voxel_count + voxel) as f32
                        / (voxel_count * points_per_voxel) as f32,
                });
            }
        }

        points
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    fn rounding_sensitive_medoid_cloud() -> Vec<PointXYZIT> {
        vec![
            PointXYZIT {
                x: -46.769,
                y: 0.041,
                z: -0.66800004,
                intensity: 27.000002,
                timestamp: 0.00058619,
            },
            PointXYZIT {
                x: -46.989002,
                y: 0.32900003,
                z: -0.67200005,
                intensity: 41.000004,
                timestamp: 0.00068444,
            },
        ]
    }

    #[test]
    fn timestamp_less_xyzi_cloud_uses_zero_scan_bounds() {
        let params = no_timestamp_params();
        let mut eval = RuntimeEvaluator::new();
        let mut buffers = DownsampleBuffers::default();
        let cloud = cloud(&[(1.0, 0.0, 0.0, Some(42.0)), (2.0, 0.0, 0.0, Some(84.0))]);

        let (_, _, begin, end, _) = filter_downsample_without_timestamps::<4, PointXYZI>(
            &cloud,
            &params,
            &mut eval,
            &mut buffers,
            ExecutionMode::Sequential,
        )
        .unwrap()
        .unwrap()
        .0;

        assert_eq!(begin, 0.0);
        assert_eq!(end, 0.0);
    }

    #[cfg(feature = "threads")]
    #[test]
    fn disabled_imu_sync_processes_cloud_without_imu() {
        let mut params = no_timestamp_params();
        params.initial_scan_accumulate = 0;
        let synthetic_undistort_called = Arc::new(AtomicBool::new(false));
        let called = Arc::clone(&synthetic_undistort_called);
        let undistort: Box<ImuPredictorUndistort> = Box::new(
            move |imu, _odom, _last_end, _cloud_begin, _cloud_end, _cloud, _passthrough| {
                assert!(imu.is_empty());
                called.store(true, Ordering::Release);
                Right(Isometry3::identity())
            },
        );
        let mut sync = Synchronizer::create(
            params.imu.gravity,
            params.imu.madgwick_setting,
            &params.imu.calibration,
            &params,
            params.lidar_vendor,
            ImuSyncMode::Disabled { lossless: true },
            SynchronizerHooks {
                imu_pose_prediction_undistort: Some(undistort),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(sync.lossless_replay);
        let receiver = sync.get_chan().unwrap();

        sync.add_pointcloud(cloud(&[(1.0, 0.0, 0.0, Some(42.0))]));

        let processed = receiver
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("Lidar-only cloud should not wait for IMU data");
        assert!(processed.imu_buffer.is_empty());
        assert_eq!(processed.imu_frame, "");
        assert!(synthetic_undistort_called.load(Ordering::Acquire));
    }

    #[cfg(feature = "threads")]
    #[test]
    fn robosense_scan_span_is_not_collapsed_by_f32_stamps() {
        // Robosense publishes absolute epoch stamps per point. Stored raw in
        // the f32 of `PointXYZIT` they all round to the same value (the ULP up
        // there is 128 s), the scan span comes out empty and every scan is
        // silently dropped. Preprocessing makes them header-relative, so the
        // synchronizer receives small offsets.
        let scan_stamp = 1_788_966_795.8;
        let (lidar_tx, lidar_rx) = std::sync::mpsc::channel();
        let (_odom_tx, odom_rx) = std::sync::mpsc::channel();
        let (synced_tx, synced_rx) = std::sync::mpsc::sync_channel(8);

        lidar_tx
            .send((
                (
                    vec![PointXYZIT::default()],
                    None,
                    0.0,
                    0.1,
                    HeaderMsg {
                        seq: 0,
                        stamp: secs_to_timestamp(scan_stamp),
                        frame_id: "rslidar".to_string(),
                    },
                ),
                1,
                Instant::now(),
                RuntimeEvaluator::new(),
            ))
            .unwrap();
        drop(lidar_tx);

        let mut previous_lidar_end = 0.0;
        Synchronizer::sync_lidar_imu_worker(
            SyncWorkerSettings {
                initial_scan_accumulate: 0,
                gravity: 9.81,
                imu_intrinsic_calibration_given: false,
                lidar_vendor: LidarVendor::Robosense,
            },
            &mut previous_lidar_end,
            SyncWorkerChannels {
                lidar: lidar_rx,
                imu: None,
                odom: odom_rx,
                synced: &synced_tx,
                ready_for_next: None,
            },
            ImuProcess::new(9.81, false),
            None,
            None,
            ImuSyncMode::Disabled { lossless: false },
        );

        let processed = synced_rx
            .try_recv()
            .expect("scan should be forwarded, not dropped as an empty span");
        let end = timestamp_to_secs(&processed.lidar_end_time_abs);
        assert!(
            (end - (scan_stamp + 0.1)).abs() < 1e-3,
            "frame end {end} should sit at the header stamp plus the scan span"
        );
        assert!((previous_lidar_end - (scan_stamp + 0.1)).abs() < 1e-3);
    }

    #[test]
    fn point_stamps_stay_representable_as_f32() {
        let header = HeaderMsg {
            seq: 0,
            stamp: secs_to_timestamp(1_788_966_795.8),
            frame_id: "rslidar".to_string(),
        };
        for vendor in [
            LidarVendor::Robosense,
            LidarVendor::LivoxMid360,
            LidarVendor::Hesai,
        ] {
            let reference = lidar_scan_reference_time(vendor, &header);
            let absolute = 1_788_966_795.85;
            let internal = internal_point_timestamp(absolute, vendor, reference).unwrap();
            let round_tripped = internal as f32 as f64;
            assert!(
                (round_tripped - internal).abs() < 1e-3,
                "{vendor:?} stamp {internal} does not survive the f32 in PointXYZIT"
            );
        }
    }

    #[cfg(feature = "threads")]
    #[test]
    fn lidar_only_advances_the_interval_across_scans() {
        let (lidar_tx, lidar_rx) = std::sync::mpsc::channel();
        let (_odom_tx, odom_rx) = std::sync::mpsc::channel();
        let (synced_tx, synced_rx) = std::sync::mpsc::sync_channel(8);

        for (scan_id, stamp, begin, end) in [(1u64, 100.0, 0.0, 0.1), (2, 100.1, 0.0, 0.1)] {
            lidar_tx
                .send((
                    (
                        vec![PointXYZIT::default()],
                        None,
                        begin,
                        end,
                        HeaderMsg {
                            seq: 0,
                            stamp: secs_to_timestamp(stamp),
                            frame_id: "lidar".to_string(),
                        },
                    ),
                    scan_id,
                    Instant::now(),
                    RuntimeEvaluator::new(),
                ))
                .unwrap();
        }
        drop(lidar_tx);

        let mut previous_lidar_end = 0.0;
        Synchronizer::sync_lidar_imu_worker(
            SyncWorkerSettings {
                initial_scan_accumulate: 0,
                gravity: 9.81,
                imu_intrinsic_calibration_given: false,
                lidar_vendor: LidarVendor::Robosense,
            },
            &mut previous_lidar_end,
            SyncWorkerChannels {
                lidar: lidar_rx,
                imu: None,
                odom: odom_rx,
                synced: &synced_tx,
                ready_for_next: None,
            },
            ImuProcess::new(9.81, false),
            None,
            None,
            ImuSyncMode::Disabled { lossless: false },
        );

        // Both scans go through, and the interval start tracks the last end
        // instead of being re-derived from each scan's own begin stamp.
        let scan_ids: Vec<_> = synced_rx.try_iter().map(|cloud| cloud.scan_id).collect();
        assert_eq!(scan_ids, vec![1, 2]);
        assert!(
            (previous_lidar_end - 100.2).abs() < 1e-6,
            "interval end was {previous_lidar_end}"
        );
    }

    #[cfg(feature = "threads")]
    #[test]
    fn preprocessed_scan_gap_does_not_stop_sync_worker() {
        let (lidar_tx, lidar_rx) = std::sync::mpsc::channel();
        let (_odom_tx, odom_rx) = std::sync::mpsc::channel();
        let (synced_tx, synced_rx) = std::sync::mpsc::sync_channel(8);

        for scan_id in [1, 3] {
            lidar_tx
                .send((
                    (
                        vec![PointXYZIT::default()],
                        None,
                        0.0,
                        0.0,
                        HeaderMsg {
                            seq: 0,
                            stamp: TimeMsg {
                                sec: scan_id as i32,
                                nanosec: 0,
                            },
                            frame_id: "lidar".to_string(),
                        },
                    ),
                    scan_id,
                    Instant::now(),
                    RuntimeEvaluator::new(),
                ))
                .unwrap();
        }
        drop(lidar_tx);

        let mut previous_lidar_end = 0.0;
        Synchronizer::sync_lidar_imu_worker(
            SyncWorkerSettings {
                initial_scan_accumulate: 2,
                gravity: 9.81,
                imu_intrinsic_calibration_given: false,
                lidar_vendor: LidarVendor::NoTimestamp,
            },
            &mut previous_lidar_end,
            SyncWorkerChannels {
                lidar: lidar_rx,
                imu: None,
                odom: odom_rx,
                synced: &synced_tx,
                ready_for_next: None,
            },
            ImuProcess::new(9.81, false),
            None,
            None,
            ImuSyncMode::Disabled { lossless: false },
        );

        let scan_ids: Vec<_> = synced_rx.try_iter().map(|cloud| cloud.scan_id).collect();
        assert_eq!(scan_ids, vec![1, 3]);
    }

    fn test_imu(stamp: f64) -> ImuMsg {
        ImuMsg {
            orientation: [1.0, 0.0, 0.0, 0.0],
            linear_acceleration: [0.0, 0.0, 9.81],
            frame_id: "imu".to_string(),
            stamp: secs_to_timestamp(stamp),
            ..Default::default()
        }
    }

    #[cfg(feature = "threads")]
    fn test_scan(scan_id: u64, stamp: f64) -> FilteredCloudMsg {
        let points = (0..8)
            .map(|i| PointXYZIT {
                x: 1.0 + i as f32,
                y: 0.5,
                z: 0.0,
                intensity: 1.0,
                timestamp: 0.0125 * i as f32,
            })
            .collect();
        (
            (
                points,
                None,
                0.0,
                0.1,
                HeaderMsg {
                    seq: 0,
                    stamp: secs_to_timestamp(stamp),
                    frame_id: "lidar".to_string(),
                },
            ),
            scan_id,
            Instant::now(),
            RuntimeEvaluator::new(),
        )
    }

    #[cfg(feature = "threads")]
    fn test_settings() -> SyncWorkerSettings {
        SyncWorkerSettings {
            initial_scan_accumulate: 1,
            gravity: 9.81,
            imu_intrinsic_calibration_given: false,
            lidar_vendor: LidarVendor::Robosense,
        }
    }

    #[cfg(feature = "threads")]
    fn summary(cloud: &SyncProcessedCloud) -> String {
        format!(
            "{} {} {:?} {:?} {:?} {:?} {}",
            cloud.scan_id,
            cloud.imu_buffer.len(),
            cloud.end_of_frame_time,
            cloud.end_of_frame_guess,
            cloud.cloud,
            cloud.pose_covariance,
            cloud.should_map_only,
        )
    }

    #[cfg(feature = "threads")]
    #[test]
    fn sync_core_matches_threaded_worker() {
        let stamps = [100.0, 100.1, 100.2, 100.3];
        let imus: Vec<_> = (0..45).map(|i| test_imu(99.95 + 0.01 * i as f64)).collect();

        let (lidar_tx, lidar_rx) = std::sync::mpsc::channel();
        let (imu_tx, imu_rx) = std::sync::mpsc::channel();
        let (_odom_tx, odom_rx) = std::sync::mpsc::channel();
        let (synced_tx, synced_rx) = std::sync::mpsc::sync_channel(8);
        for (i, stamp) in stamps.iter().enumerate() {
            lidar_tx.send(test_scan(i as u64 + 1, *stamp)).unwrap();
        }
        for imu in &imus {
            imu_tx.send(imu.clone()).unwrap();
        }
        drop(lidar_tx);
        drop(imu_tx);
        let mut previous_lidar_end = 0.0;
        Synchronizer::sync_lidar_imu_worker(
            test_settings(),
            &mut previous_lidar_end,
            SyncWorkerChannels {
                lidar: lidar_rx,
                imu: Some(imu_rx),
                odom: odom_rx,
                synced: &synced_tx,
                ready_for_next: None,
            },
            ImuProcess::new(9.81, false),
            None,
            None,
            ImuSyncMode::Continuous,
        );
        let threaded: Vec<_> = synced_rx.try_iter().map(|cloud| summary(&cloud)).collect();

        let mut core = SyncCore::new(
            test_settings(),
            true,
            0.0,
            ImuProcess::new(9.81, false),
            None,
            None,
        );
        let mut imu_iter = imus.into_iter();
        let mut stepped = Vec::new();
        for (i, stamp) in stamps.iter().enumerate() {
            let synced = match core.begin_scan(test_scan(i as u64 + 1, *stamp)) {
                ScanStart::Ready(synced) => Some(*synced),
                ScanStart::Skipped => None,
                ScanStart::CollectingImu => {
                    // Like the threaded worker, a scan without an IMU message
                    // past its end never completes.
                    imu_iter
                        .by_ref()
                        .any(|imu| core.offer_imu(imu))
                        .then(|| core.finish_scan(std::iter::empty()))
                        .flatten()
                }
            };
            stepped.extend(synced.as_ref().map(summary));
        }

        // The last scan ends after the last IMU message.
        assert_eq!(threaded.len(), stamps.len() - 1);
        assert_eq!(threaded, stepped);
        assert_eq!(previous_lidar_end, core.prev_lidar_end_time());
    }

    fn counting_init() -> (ImuInitCallback, Arc<std::sync::Mutex<Vec<usize>>>) {
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = Arc::clone(&calls);
        let callback: ImuInitCallback =
            Arc::new(std::sync::Mutex::new(move |msgs: &[ImuMsg]| {
                seen.lock().unwrap().push(msgs.len());
            }));
        (callback, calls)
    }

    #[test]
    fn imu_init_window_hands_over_the_window() {
        let (callback, calls) = counting_init();
        let mut window = ImuInitWindow::new(Some(callback), 0.1);
        for stamp in [10.0, 10.05, 10.09, 10.15, 10.2] {
            window.observe(&test_imu(stamp));
        }
        assert_eq!(*calls.lock().unwrap(), vec![3]);
        assert!(window.is_complete());
    }

    #[test]
    fn imu_init_window_without_duration_uses_the_first_message() {
        let (callback, calls) = counting_init();
        let mut window = ImuInitWindow::new(Some(callback), 0.0);
        window.observe(&test_imu(10.0));
        window.observe(&test_imu(10.1));
        assert_eq!(*calls.lock().unwrap(), vec![1]);
    }

    #[test]
    fn imu_init_window_restarts_when_time_runs_backwards() {
        let (callback, calls) = counting_init();
        let mut window = ImuInitWindow::new(Some(callback), 0.1);
        for stamp in [10.0, 10.05, 5.0, 5.02, 5.15] {
            window.observe(&test_imu(stamp));
        }
        assert_eq!(*calls.lock().unwrap(), vec![2]);
    }

    #[test]
    fn stepped_synchronizer_waits_for_the_imu_interval() {
        let mut params = no_timestamp_params();
        params.initial_scan_accumulate = 0;
        let mut sync = SteppedSynchronizer::create(
            &params,
            params.lidar_vendor,
            ImuSyncMode::Continuous,
            SteppedSynchronizerHooks::default(),
        )
        .unwrap();

        let scan_id = sync.add_pointcloud(cloud(&[(1.0, 0.0, 0.0, Some(42.0))]));
        assert_eq!(scan_id, 1);
        assert!(sync.add_imu(test_imu(122.9)).is_some());
        assert!(sync.next_synced().is_none());
        assert_eq!(sync.pending_scans(), 1);

        sync.add_imu(test_imu(123.1));
        let processed = sync
            .next_synced()
            .expect("scan completes once an IMU message lies past its end");
        assert_eq!(processed.scan_id, 1);
        assert_eq!(processed.imu_buffer.len(), 1);
        assert_eq!(sync.pending_scans(), 0);
        assert!(sync.next_synced().is_none());
    }

    #[test]
    fn stepped_synchronizer_without_imu_completes_immediately() {
        let mut params = no_timestamp_params();
        params.initial_scan_accumulate = 0;
        let mut sync = SteppedSynchronizer::create(
            &params,
            params.lidar_vendor,
            ImuSyncMode::Disabled { lossless: true },
            SteppedSynchronizerHooks::default(),
        )
        .unwrap();

        sync.add_pointcloud(cloud(&[(1.0, 0.0, 0.0, Some(42.0))]));
        assert!(sync.add_imu(test_imu(123.1)).is_none());
        let processed = sync.next_synced().expect("Lidar-only scans need no IMU");
        assert!(processed.imu_buffer.is_empty());
    }

    #[test]
    fn timestamp_less_xyz_cloud_falls_back_without_intensity() {
        let params = no_timestamp_params();
        let mut eval = RuntimeEvaluator::new();
        let mut buffers = DownsampleBuffers::default();
        let cloud = cloud(&[(1.0, 0.0, 0.0, None), (2.0, 0.0, 0.0, None)]);

        let (points, _, begin, end, _) =
            filter_downsample_without_timestamps_with_fallback::<4, 3, PointXYZI, PointXYZ>(
                &cloud,
                &params,
                &mut eval,
                &mut buffers,
                ExecutionMode::Sequential,
            )
            .unwrap()
            .unwrap()
            .0;

        assert_eq!(begin, 0.0);
        assert_eq!(end, 0.0);
        assert!(points.iter().all(|p| p.timestamp == 0.0));
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn sequential_and_rayon_preprocessing_match() {
        let params = no_timestamp_params();
        let cloud = cloud(&[
            (1.0, 0.0, 0.0, Some(42.0)),
            (2.0, 0.0, 0.0, Some(84.0)),
            (1.001, 0.0, 0.0, Some(21.0)),
        ]);

        let run = |mode| {
            let mut eval = RuntimeEvaluator::new();
            let mut buffers = DownsampleBuffers::default();
            filter_downsample_without_timestamps::<4, PointXYZI>(
                &cloud,
                &params,
                &mut eval,
                &mut buffers,
                mode,
            )
            .unwrap()
            .unwrap()
            .0
            .0
        };

        assert_eq!(run(ExecutionMode::Sequential), run(ExecutionMode::Rayon));
    }

    #[test]
    fn cpu_downsample_resolves_medoid_ties_deterministically() {
        let points = adversarial_downsample_cloud();
        let actual = cpu_downsample_reference(&points);

        assert_eq!(actual, vec![points[4], points[2], points[3]]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_downsample_matches_cpu_ties_and_ordering() {
        let points = adversarial_downsample_cloud();
        let expected = cpu_downsample_reference(&points);
        let stream = pelorus_cuda::Stream::new();
        let (actual, _) = pelorus_cuda::downsample_points_cuda(&stream, &points, 1, 1.0);

        assert_eq!(actual, expected);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_downsample_matches_cpu_at_realistic_density() {
        let points = dense_downsample_cloud();
        let expected = cpu_downsample_reference(&points);
        let stream = pelorus_cuda::Stream::new();
        let (actual, _) = pelorus_cuda::downsample_points_cuda(&stream, &points, 1, 1.0);

        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual, expected,
                "CUDA downsample differs from CPU at output point {index}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_downsample_matches_cpu_for_rounding_sensitive_medoid() {
        let points = rounding_sensitive_medoid_cloud();
        let expected = cpu_downsample_reference(&points);
        let stream = pelorus_cuda::Stream::new();
        let (actual, _) = pelorus_cuda::downsample_points_cuda(&stream, &points, 1, 1.0);

        assert_eq!(expected, vec![points[0]]);
        assert_eq!(actual, expected);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_downsample_matches_cpu_ties_and_ordering() {
        let points = adversarial_downsample_cloud();
        let expected = cpu_downsample_reference(&points);
        let stream = pelorus_vulkan::Stream::new();
        let (actual, _) = pelorus_vulkan::downsample_points_vulkan(&stream, &points, 1, 1.0);

        assert_eq!(actual, expected);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_downsample_matches_cpu_ties_and_ordering() {
        let points = adversarial_downsample_cloud();
        let expected = cpu_downsample_reference(&points);
        let stream = pelorus_metal::Stream::new();
        let (actual, _) = pelorus_metal::downsample_points_metal(&stream, &points, 1, 1.0);

        assert_eq!(actual, expected);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_downsample_matches_cpu_for_rounding_sensitive_medoid() {
        let points = rounding_sensitive_medoid_cloud();
        let expected = cpu_downsample_reference(&points);
        let stream = pelorus_metal::Stream::new();
        let (actual, _) = pelorus_metal::downsample_points_metal(&stream, &points, 1, 1.0);

        assert_eq!(expected, vec![points[0]]);
        assert_eq!(actual, expected);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_downsample_matches_cpu_at_realistic_density() {
        let points = dense_downsample_cloud();
        let expected = cpu_downsample_reference(&points);
        let stream = pelorus_metal::Stream::new();
        let (actual, _) = pelorus_metal::downsample_points_metal(&stream, &points, 1, 1.0);

        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual, expected,
                "Metal downsample differs from CPU at output point {index}"
            );
        }
    }

    /// `downsample_rle` picks each run's output slot with an atomic
    /// fetch-add, so the medoids come back in thread-race order. The output
    /// ordering must not depend on that: sorting it is what pins the result
    /// down, and it has to be a total order to do so. Without it the whole
    /// Metal backend drifts run to run (~0.017 m over a 139 s bag), because
    /// per-voxel mean/covariance sums are accumulated in array order.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_downsample_is_deterministic_across_repeated_runs() {
        let points = dense_downsample_cloud();
        let stream = pelorus_metal::Stream::new();
        let (first, _) = pelorus_metal::downsample_points_metal(&stream, &points, 1, 1.0);
        assert!(first.len() > 1000, "expected a substantial cloud");
        for attempt in 1..8 {
            let (again, _) = pelorus_metal::downsample_points_metal(&stream, &points, 1, 1.0);
            assert_eq!(
                again, first,
                "Metal downsample differed on attempt {attempt}"
            );
        }
    }
}

fn hash(point: &Point3<f32>, voxel_size: f32) -> i32 {
    const PRIME_1: i32 = 73_856_093;
    const PRIME_2: i32 = 471_943;
    const PRIME_3: i32 = 83_492_791;

    let xyz = point * (1.0 / voxel_size);

    let x = xyz.x.floor() as i32;
    let y = xyz.y.floor() as i32;
    let z = xyz.z.floor() as i32;

    let mut hash = x.wrapping_mul(PRIME_1);
    hash = hash.wrapping_add(y.wrapping_mul(PRIME_2));
    hash = hash.wrapping_add(z.wrapping_mul(PRIME_3));

    hash
}

/// Helper function to find the medoid within a slice of points.
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
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    a.timestamp
                        .partial_cmp(&b.timestamp)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.z.partial_cmp(&b.z).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| {
                    a.intensity
                        .partial_cmp(&b.intensity)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        })
        .unwrap()
}

type DownsampleCloudMeta = (
    Vec<PointXYZIT>,
    Option<Vec<PointXYZIT>>,
    f64,
    f64,
    HeaderMsg,
);
type DownsampleResult = (DownsampleCloudMeta, bool);

#[derive(Debug, Clone, Copy, Default)]
struct DownsamplePhaseTimings {
    gpu_downsample_us: u64,
    gpu_upload_us: u64,
    gpu_key_build_us: u64,
    gpu_keyed_sort_us: u64,
    gpu_run_length_encode_us: u64,
    gpu_run_starts_scan_us: u64,
    gpu_unique_count_sync_us: u64,
    gpu_medoid_reduce_us: u64,
    gpu_timestamp_key_build_us: u64,
    gpu_timestamp_sort_us: u64,
    gpu_download_us: u64,
}

type KeyedPoint = (i32, PointXYZIT);

#[derive(Default)]
struct DownsampleBuffers {
    filtered_cloud: Vec<PointXYZIT>,
    keyed_points: Vec<KeyedPoint>,
}

impl DownsampleBuffers {
    fn prepare(&mut self, capacity: usize) {
        self.filtered_cloud.clear();
        self.keyed_points.clear();

        if self.filtered_cloud.capacity() < capacity {
            self.filtered_cloud
                .reserve(capacity - self.filtered_cloud.capacity());
        }
        if self.keyed_points.capacity() < capacity {
            self.keyed_points
                .reserve(capacity - self.keyed_points.capacity());
        }
    }
}

#[inline]
fn compare_f32(a: f32, b: f32) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
}

#[inline]
fn compare_point_timestamp(a: &PointXYZIT, b: &PointXYZIT) -> std::cmp::Ordering {
    compare_f32(a.timestamp, b.timestamp)
        .then_with(|| compare_f32(a.x, b.x))
        .then_with(|| compare_f32(a.y, b.y))
        .then_with(|| compare_f32(a.z, b.z))
        .then_with(|| compare_f32(a.intensity, b.intensity))
}

#[inline]
fn keyed_point_sort_key((key, point): &KeyedPoint) -> (i32, u32, u32, u32, u32, u32) {
    (
        *key,
        point.timestamp.to_bits(),
        point.x.to_bits(),
        point.y.to_bits(),
        point.z.to_bits(),
        point.intensity.to_bits(),
    )
}

#[inline]
fn key_downsample_point(point: &PointXYZIT, voxel_size: f32) -> KeyedPoint {
    let key = hash(
        &nalgebra::Point3::new(point.x, point.y, point.z),
        voxel_size,
    );
    (key, *point)
}

#[inline]
fn lidar_scan_reference_time(lidar_vendor: LidarVendor, header: &HeaderMsg) -> Option<f64> {
    match lidar_vendor {
        LidarVendor::Velodyne
        | LidarVendor::LivoxMid360
        | LidarVendor::Hesai
        | LidarVendor::Robosense => Some(timestamp_to_secs(&header.stamp)),
        _ => None,
    }
}

#[inline]
fn internal_point_timestamp(
    timestamp: f64,
    lidar_vendor: LidarVendor,
    scan_reference_time: Option<f64>,
) -> Option<f64> {
    if timestamp.is_nan() {
        return None;
    }

    Some(match lidar_vendor {
        // Absolute per-point epochs do not survive the f32 in `PointXYZIT`:
        // near 1.8e9 the spacing is 128 s, so every point in a scan collapses
        // onto one value and the scan span comes out empty. Store them
        // relative to the header instead.
        LidarVendor::LivoxMid360 | LidarVendor::Hesai | LidarVendor::Robosense => {
            let scan_reference_time = scan_reference_time.expect("vendor match out of date");
            (timestamp - scan_reference_time).max(0.0)
        }
        _ => timestamp,
    })
}

#[inline]
fn has_point_timestamps(lidar_vendor: LidarVendor) -> bool {
    !matches!(lidar_vendor, LidarVendor::NoTimestamp)
}

#[inline]
fn passes_passthrough_filter(point: &PointXYZIT, sync_params: &SyncParameters) -> bool {
    if !(point.x.is_finite() && point.y.is_finite() && point.z.is_finite()) {
        return false;
    }

    let params = &sync_params.passthrough_filter;
    let p_t = params.filter_frame_transform * nalgebra::Point3::new(point.x, point.y, point.z);

    let dist = p_t.coords.norm_squared();
    let is_within_range = if params.invert_distance {
        dist < params.min_dist.powi(2) || dist > params.max_dist.powi(2)
    } else {
        dist >= params.min_dist.powi(2) && dist <= params.max_dist.powi(2)
    };
    if !is_within_range {
        return false;
    }

    let is_within_intensity = if params.invert_intensity {
        point.intensity < params.min_intensity || point.intensity > params.max_intensity
    } else {
        point.intensity >= params.min_intensity && point.intensity <= params.max_intensity
    };
    if !is_within_intensity {
        return false;
    }

    let is_within_box = if params.invert_bounding_box {
        !inside_box(p_t, params.min, params.max)
    } else {
        inside_box(p_t, params.min, params.max)
    };
    if !is_within_box {
        return false;
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
        return false;
    }

    match &params.polygon {
        None => true,
        Some(polygon) => {
            let is_within_polygon = if params.invert_polygon {
                !point_inside_polygon_winding_number(p_t, polygon, polygon.len())
            } else {
                point_inside_polygon_winding_number(p_t, polygon, polygon.len())
            };
            !polygon.is_empty() && is_within_polygon
        }
    }
}

#[inline]
fn filter_point(mut point: PointXYZIT, sync_params: &SyncParameters) -> Option<PointXYZIT> {
    if !passes_passthrough_filter(&point, sync_params) {
        return None;
    }

    if point.timestamp.is_nan() {
        return None;
    }

    point.intensity *= 255.0;
    Some(point)
}

#[inline]
fn cpu_key_downsample(
    filtered_cloud: &[PointXYZIT],
    point_skip: usize,
    voxel_size: f32,
    keyed_points: &mut Vec<KeyedPoint>,
    eval: &mut RuntimeEvaluator,
    mode: ExecutionMode,
) -> Result<()> {
    #[cfg(not(feature = "rayon"))]
    let _ = mode;
    eval.start("filter_downsample/filter_map_down")?;
    if point_skip == 1 {
        #[cfg(feature = "rayon")]
        match mode {
            ExecutionMode::Sequential => keyed_points.extend(
                filtered_cloud
                    .iter()
                    .map(|point| key_downsample_point(point, voxel_size)),
            ),
            ExecutionMode::Rayon => keyed_points.par_extend(
                filtered_cloud
                    .par_iter()
                    .map(|point| key_downsample_point(point, voxel_size)),
            ),
        }
        #[cfg(not(feature = "rayon"))]
        keyed_points.extend(
            filtered_cloud
                .iter()
                .map(|point| key_downsample_point(point, voxel_size)),
        );
    } else {
        #[cfg(feature = "rayon")]
        match mode {
            ExecutionMode::Sequential => {
                keyed_points.extend(
                    filtered_cloud
                        .iter()
                        .enumerate()
                        .filter(|&(idx, _point)| idx % point_skip == 0)
                        .map(|(_idx, point)| key_downsample_point(point, voxel_size)),
                );
            }
            ExecutionMode::Rayon => {
                keyed_points.par_extend(filtered_cloud.par_iter().enumerate().filter_map(
                    |(idx, point)| {
                        (idx % point_skip == 0).then(|| key_downsample_point(point, voxel_size))
                    },
                ));
            }
        }
        #[cfg(not(feature = "rayon"))]
        keyed_points.extend(
            filtered_cloud
                .iter()
                .enumerate()
                .filter(|&(idx, _)| idx % point_skip == 0)
                .map(|(_, point)| key_downsample_point(point, voxel_size)),
        );
    }
    eval.stop("filter_downsample/filter_map_down")?;

    Ok(())
}

#[inline]
fn cpu_reduce_downsample(
    keyed_points: &mut [KeyedPoint],
    downsampled: &mut Vec<PointXYZIT>,
    sort_by_timestamp: bool,
    eval: &mut RuntimeEvaluator,
    mode: ExecutionMode,
) -> Result<()> {
    eval.start("filter_downsample/keyed_sort")?;
    sort_unstable_by_key(keyed_points, keyed_point_sort_key, mode);
    eval.stop("filter_downsample/keyed_sort")?;

    eval.start("filter_downsample/medoid_reduce")?;
    downsampled.extend(
        keyed_points
            .chunk_by(|a, b| a.0 == b.0)
            .map(find_medoid_in_slice),
    );
    eval.stop("filter_downsample/medoid_reduce")?;

    if sort_by_timestamp {
        eval.start("filter_downsample/timestamp_sort")?;
        sort_unstable_by(downsampled, compare_point_timestamp, mode);
        eval.stop("filter_downsample/timestamp_sort")?;
    }

    Ok(())
}

#[inline]
fn cpu_downsample(
    buffers: &mut DownsampleBuffers,
    point_skip: usize,
    voxel_size: f32,
    downsampled: &mut Vec<PointXYZIT>,
    sort_by_timestamp: bool,
    eval: &mut RuntimeEvaluator,
    mode: ExecutionMode,
) -> Result<()> {
    cpu_key_downsample(
        &buffers.filtered_cloud,
        point_skip,
        voxel_size,
        &mut buffers.keyed_points,
        eval,
        mode,
    )?;
    cpu_reduce_downsample(
        &mut buffers.keyed_points,
        downsampled,
        sort_by_timestamp,
        eval,
        mode,
    )
}

#[inline]
fn sort_bypass_cloud(
    filtered_cloud: &mut [PointXYZIT],
    eval: &mut RuntimeEvaluator,
    mode: ExecutionMode,
) -> Result<()> {
    eval.start("filter_downsample/bypass_sort")?;
    sort_unstable_by(filtered_cloud, compare_point_timestamp, mode);
    eval.stop("filter_downsample/bypass_sort")?;
    Ok(())
}

#[inline]
fn scan_time_bounds(
    downsampled: &[PointXYZIT],
    raw_out: Option<&Vec<PointXYZIT>>,
    sort_by_timestamp: bool,
) -> Option<(f32, f32)> {
    if !sort_by_timestamp && !downsampled.is_empty() {
        return Some((0.0, 0.0));
    }

    match (downsampled.first(), downsampled.last()) {
        (Some(first), Some(last)) if last.timestamp > 0.0 => {
            Some((first.timestamp, last.timestamp))
        }
        _ => raw_out.and_then(|bypass| match (bypass.first(), bypass.last()) {
            (Some(first), Some(last)) if last.timestamp > 0.0 => {
                Some((first.timestamp, last.timestamp))
            }
            _ => None,
        }),
    }
}

fn add_gpu_downsample_timings(eval: &mut RuntimeEvaluator, timings: DownsamplePhaseTimings) {
    eval.add_external_measurement(
        "filter_downsample/gpu_downsample",
        timings.gpu_downsample_us as f64 / 1000.0,
    );

    let mut add_timing = |name: &str, us: u64| {
        if us > 0 {
            eval.add_external_measurement(name, us as f64 / 1000.0);
        }
    };
    add_timing("filter_downsample/gpu_upload", timings.gpu_upload_us);
    add_timing("filter_downsample/gpu_key_build", timings.gpu_key_build_us);
    add_timing(
        "filter_downsample/gpu_keyed_sort",
        timings.gpu_keyed_sort_us,
    );
    add_timing(
        "filter_downsample/gpu_run_length_encode",
        timings.gpu_run_length_encode_us,
    );
    add_timing(
        "filter_downsample/gpu_run_starts_scan",
        timings.gpu_run_starts_scan_us,
    );
    add_timing(
        "filter_downsample/gpu_unique_count_sync",
        timings.gpu_unique_count_sync_us,
    );
    add_timing(
        "filter_downsample/gpu_medoid_reduce",
        timings.gpu_medoid_reduce_us,
    );
    add_timing(
        "filter_downsample/gpu_timestamp_key_build",
        timings.gpu_timestamp_key_build_us,
    );
    add_timing(
        "filter_downsample/gpu_timestamp_sort",
        timings.gpu_timestamp_sort_us,
    );
    add_timing("filter_downsample/gpu_download", timings.gpu_download_us);
}

#[cfg(feature = "cuda")]
thread_local! {
    static CUDA_DOWNSAMPLE_STREAM: pelorus_cuda::Stream = pelorus_cuda::Stream::new();
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
#[inline]
fn ms_to_us(ms: f64) -> u64 {
    (ms * 1000.0) as u64
}

#[cfg(feature = "cuda")]
fn try_cuda_downsample_from_filtered(
    filtered_cloud: &[PointXYZIT],
    point_skip: usize,
    voxel_size: f32,
) -> Option<(Vec<PointXYZIT>, DownsamplePhaseTimings)> {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*AVAILABLE.get_or_init(pelorus_cuda::is_available) {
        return None;
    }

    Some(CUDA_DOWNSAMPLE_STREAM.with(|stream| {
        let (downsampled, timings) =
            pelorus_cuda::downsample_points_cuda(stream, filtered_cloud, point_skip, voxel_size);
        let timings = DownsamplePhaseTimings {
            gpu_downsample_us: ms_to_us(timings.total_ms),
            gpu_upload_us: ms_to_us(timings.upload_ms),
            gpu_key_build_us: ms_to_us(timings.key_build_ms),
            gpu_keyed_sort_us: ms_to_us(timings.keyed_sort_ms),
            gpu_run_length_encode_us: ms_to_us(timings.run_length_encode_ms),
            gpu_run_starts_scan_us: ms_to_us(timings.run_starts_scan_ms),
            gpu_unique_count_sync_us: ms_to_us(timings.unique_count_sync_ms),
            gpu_medoid_reduce_us: ms_to_us(timings.medoid_reduce_ms),
            gpu_timestamp_key_build_us: ms_to_us(timings.timestamp_key_build_ms),
            gpu_timestamp_sort_us: ms_to_us(timings.timestamp_sort_ms),
            gpu_download_us: ms_to_us(timings.download_ms),
        };
        (downsampled, timings)
    }))
}

#[cfg(feature = "metal")]
thread_local! {
    static METAL_DOWNSAMPLE_STREAM: pelorus_metal::Stream = pelorus_metal::Stream::new();
}

#[cfg(feature = "metal")]
fn try_metal_downsample_from_filtered(
    filtered_cloud: &[PointXYZIT],
    point_skip: usize,
    voxel_size: f32,
) -> Option<(Vec<PointXYZIT>, DownsamplePhaseTimings)> {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*AVAILABLE.get_or_init(pelorus_metal::is_available) {
        return None;
    }

    Some(METAL_DOWNSAMPLE_STREAM.with(|stream| {
        let (downsampled, timings) =
            pelorus_metal::downsample_points_metal(stream, filtered_cloud, point_skip, voxel_size);
        let timings = DownsamplePhaseTimings {
            gpu_downsample_us: ms_to_us(timings.total_ms),
            gpu_upload_us: ms_to_us(timings.upload_ms),
            gpu_key_build_us: ms_to_us(timings.key_build_ms),
            gpu_keyed_sort_us: ms_to_us(timings.keyed_sort_ms),
            gpu_run_length_encode_us: ms_to_us(timings.run_length_encode_ms),
            gpu_run_starts_scan_us: ms_to_us(timings.run_starts_scan_ms),
            gpu_unique_count_sync_us: ms_to_us(timings.unique_count_sync_ms),
            gpu_medoid_reduce_us: ms_to_us(timings.medoid_reduce_ms),
            gpu_timestamp_key_build_us: ms_to_us(timings.timestamp_key_build_ms),
            gpu_timestamp_sort_us: ms_to_us(timings.timestamp_sort_ms),
            gpu_download_us: ms_to_us(timings.download_ms),
        };
        (downsampled, timings)
    }))
}

#[cfg(feature = "vulkan")]
thread_local! {
    static VULKAN_DOWNSAMPLE_STREAM: pelorus_vulkan::Stream = pelorus_vulkan::Stream::new();
}

#[cfg(feature = "vulkan")]
fn try_vulkan_downsample_from_filtered(
    filtered_cloud: &[PointXYZIT],
    point_skip: usize,
    voxel_size: f32,
) -> Option<(Vec<PointXYZIT>, DownsamplePhaseTimings)> {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*AVAILABLE.get_or_init(pelorus_vulkan::is_available) {
        return None;
    }

    Some(VULKAN_DOWNSAMPLE_STREAM.with(|stream| {
        let (downsampled, timings) = pelorus_vulkan::downsample_points_vulkan(
            stream,
            filtered_cloud,
            point_skip,
            voxel_size,
        );
        let timings = DownsamplePhaseTimings {
            gpu_downsample_us: ms_to_us(timings.total_ms),
            gpu_upload_us: ms_to_us(timings.upload_ms),
            gpu_key_build_us: ms_to_us(timings.key_build_ms),
            gpu_keyed_sort_us: ms_to_us(timings.keyed_sort_ms),
            gpu_run_length_encode_us: ms_to_us(timings.run_length_encode_ms),
            gpu_run_starts_scan_us: ms_to_us(timings.run_starts_scan_ms),
            gpu_unique_count_sync_us: ms_to_us(timings.unique_count_sync_ms),
            gpu_medoid_reduce_us: ms_to_us(timings.medoid_reduce_ms),
            gpu_timestamp_key_build_us: ms_to_us(timings.timestamp_key_build_ms),
            gpu_timestamp_sort_us: ms_to_us(timings.timestamp_sort_ms),
            gpu_download_us: ms_to_us(timings.download_ms),
        };
        (downsampled, timings)
    }))
}

fn filter_downsample_with_fallback<
    const N: usize,
    const N_FALLBACK: usize,
    P: PointXYZITimestamp<N>,
    PFallback: PointXYZITimestamp<N_FALLBACK>,
>(
    cloud: &dyn PreprocessingPointCloud,
    sync_params: &SyncParameters,
    eval: &mut RuntimeEvaluator,
    buffers: &mut DownsampleBuffers,
    mode: ExecutionMode,
) -> Result<Option<DownsampleResult>> {
    let result = filter_downsample::<N, P>(cloud, sync_params, eval, buffers, mode);

    match result {
        Ok(data) => Ok(data),
        Err(e) => {
            let error_str = format!("{:?}", e);
            if error_str.contains("Some fields") && (error_str.contains("intensity")) {
                filter_downsample::<N_FALLBACK, PFallback>(cloud, sync_params, eval, buffers, mode)
            } else {
                Err(e)
            }
        }
    }
}

fn filter_downsample_livox(
    cloud: &dyn PreprocessingPointCloud,
    sync_params: &SyncParameters,
    eval: &mut RuntimeEvaluator,
    buffers: &mut DownsampleBuffers,
    mode: ExecutionMode,
) -> Result<Option<DownsampleResult>> {
    let result = filter_downsample::<5, LivoxPoint>(cloud, sync_params, eval, buffers, mode);
    match result {
        Ok(data) => Ok(data),
        Err(t_error) => {
            let offset_time_result = filter_downsample::<5, LivoxPointOffsetTime>(
                cloud,
                sync_params,
                eval,
                buffers,
                mode,
            );
            match offset_time_result {
                Ok(data) => Ok(data),
                Err(offset_time_error) => {
                    let error_str = format!("{offset_time_error:?}");
                    if error_str.contains("Some fields") && error_str.contains("intensity") {
                        filter_downsample::<4, LivoxPointOffsetTimeNoIntensity>(
                            cloud,
                            sync_params,
                            eval,
                            buffers,
                            mode,
                        )
                    } else {
                        let t_error_str = format!("{t_error:?}");
                        if t_error_str.contains("Some fields") && t_error_str.contains("intensity")
                        {
                            filter_downsample::<4, LivoxPointNoIntensity>(
                                cloud,
                                sync_params,
                                eval,
                                buffers,
                                mode,
                            )
                        } else {
                            Err(offset_time_error)
                        }
                    }
                }
            }
        }
    }
}

fn filter_downsample<const N: usize, P: PointXYZITimestamp<N>>(
    cloud: &dyn PreprocessingPointCloud,
    sync_params: &SyncParameters,
    eval: &mut RuntimeEvaluator,
    buffers: &mut DownsampleBuffers,
    mode: ExecutionMode,
) -> Result<Option<DownsampleResult>> {
    let header = cloud.header();
    let scan_reference_time = lidar_scan_reference_time(sync_params.lidar_vendor, &header);
    filter_downsample_inner::<N, P>(
        cloud,
        sync_params,
        eval,
        buffers,
        true,
        |p| {
            internal_point_timestamp(p.timestamp(), sync_params.lidar_vendor, scan_reference_time)
                .map(|timestamp| PointXYZIT::from_p(p.xyz(), p.intensity(), timestamp as f32))
        },
        mode,
    )
}

fn filter_downsample_without_timestamps<const N: usize, P: PointXYZIntensity<N>>(
    cloud: &dyn PreprocessingPointCloud,
    sync_params: &SyncParameters,
    eval: &mut RuntimeEvaluator,
    buffers: &mut DownsampleBuffers,
    mode: ExecutionMode,
) -> Result<Option<DownsampleResult>> {
    filter_downsample_inner::<N, P>(
        cloud,
        sync_params,
        eval,
        buffers,
        false,
        |p| Some(PointXYZIT::from_p(p.xyz(), p.intensity(), 0.0)),
        mode,
    )
}

fn filter_downsample_without_timestamps_with_fallback<
    const N: usize,
    const N_FALLBACK: usize,
    P: PointXYZIntensity<N>,
    PFallback: PointXYZIntensity<N_FALLBACK>,
>(
    cloud: &dyn PreprocessingPointCloud,
    sync_params: &SyncParameters,
    eval: &mut RuntimeEvaluator,
    buffers: &mut DownsampleBuffers,
    mode: ExecutionMode,
) -> Result<Option<DownsampleResult>> {
    let result =
        filter_downsample_without_timestamps::<N, P>(cloud, sync_params, eval, buffers, mode);

    match result {
        Ok(data) => Ok(data),
        Err(e) => {
            let error_str = format!("{:?}", e);
            if error_str.contains("Some fields") && error_str.contains("intensity") {
                filter_downsample_without_timestamps::<N_FALLBACK, PFallback>(
                    cloud,
                    sync_params,
                    eval,
                    buffers,
                    mode,
                )
            } else {
                Err(e)
            }
        }
    }
}

fn filter_downsample_inner<const N: usize, P: PointConvertible<N> + Send + Sync>(
    cloud: &dyn PreprocessingPointCloud,
    sync_params: &SyncParameters,
    eval: &mut RuntimeEvaluator,
    buffers: &mut DownsampleBuffers,
    sort_by_timestamp: bool,
    map_to_non_trait: impl Fn(P) -> Option<PointXYZIT> + Send + Sync,
    mode: ExecutionMode,
) -> Result<Option<DownsampleResult>> {
    let bypass_output_downsample = sync_params.downsample.bypass_output_downsample;
    let header = cloud.header();
    let source = cloud.source();
    let voxel_size = sync_params.downsample.voxel_size;
    let filter_map_point = |point| filter_point(point, sync_params);

    buffers.prepare(source.point_count());
    let mut downsampled: Vec<PointXYZIT> = Vec::with_capacity(source.point_count());

    let mut raw_out = None;
    let used_cpu_downsample;

    {
        let point_skip = sync_params.downsample.point_skip.max(1);

        eval.start("filter_downsample/bypass_filter_map")?;
        #[cfg(feature = "rayon")]
        match mode {
            ExecutionMode::Sequential => buffers.filtered_cloud.extend(
                ros_pointcloud2::try_into_iter(source)?
                    .filter_map(map_to_non_trait)
                    .filter_map(filter_map_point),
            ),
            ExecutionMode::Rayon => buffers.filtered_cloud.par_extend(
                ros_pointcloud2::try_into_par_iter(source)?
                    .filter_map(map_to_non_trait)
                    .filter_map(filter_map_point),
            ),
        }
        #[cfg(not(feature = "rayon"))]
        buffers.filtered_cloud.extend(
            ros_pointcloud2::try_into_iter(source)?
                .filter_map(map_to_non_trait)
                .filter_map(filter_map_point),
        );
        eval.stop("filter_downsample/bypass_filter_map")?;

        #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
        let use_backend_preprocessing =
            sync_params.downsample.backend == PreprocessingBackend::Backend;

        #[cfg(feature = "cuda")]
        let gpu_downsample = use_backend_preprocessing
            .then(|| {
                try_cuda_downsample_from_filtered(&buffers.filtered_cloud, point_skip, voxel_size)
            })
            .flatten();

        #[cfg(not(feature = "cuda"))]
        let gpu_downsample: Option<(Vec<PointXYZIT>, DownsamplePhaseTimings)> = None;

        #[cfg(feature = "metal")]
        let gpu_downsample = if use_backend_preprocessing {
            gpu_downsample.or_else(|| {
                try_metal_downsample_from_filtered(&buffers.filtered_cloud, point_skip, voxel_size)
            })
        } else {
            gpu_downsample
        };

        #[cfg(feature = "vulkan")]
        let gpu_downsample = if use_backend_preprocessing {
            gpu_downsample.or_else(|| {
                try_vulkan_downsample_from_filtered(&buffers.filtered_cloud, point_skip, voxel_size)
            })
        } else {
            gpu_downsample
        };

        used_cpu_downsample = gpu_downsample.is_none();
        if let Some((gpu_downsampled, gpu_timings)) = gpu_downsample {
            downsampled = gpu_downsampled;
            add_gpu_downsample_timings(eval, gpu_timings);
            if bypass_output_downsample {
                if sort_by_timestamp {
                    sort_bypass_cloud(&mut buffers.filtered_cloud, eval, mode)?;
                }
                raw_out = Some(std::mem::take(&mut buffers.filtered_cloud));
            }
        } else if bypass_output_downsample {
            cpu_key_downsample(
                &buffers.filtered_cloud,
                point_skip,
                voxel_size,
                &mut buffers.keyed_points,
                eval,
                mode,
            )?;

            #[cfg(feature = "rayon")]
            if mode == ExecutionMode::Rayon {
                let mut downsample_eval = RuntimeEvaluator::new();
                let mut bypass_eval = RuntimeEvaluator::new();
                let (downsample_result, bypass_result) = rayon::join(
                    || {
                        cpu_reduce_downsample(
                            &mut buffers.keyed_points,
                            &mut downsampled,
                            sort_by_timestamp,
                            &mut downsample_eval,
                            mode,
                        )
                    },
                    || {
                        if sort_by_timestamp {
                            sort_bypass_cloud(&mut buffers.filtered_cloud, &mut bypass_eval, mode)
                        } else {
                            Ok(())
                        }
                    },
                );
                downsample_result?;
                bypass_result?;
                eval.merge(&downsample_eval);
                eval.merge(&bypass_eval);
            } else {
                cpu_reduce_downsample(
                    &mut buffers.keyed_points,
                    &mut downsampled,
                    sort_by_timestamp,
                    eval,
                    mode,
                )?;
                if sort_by_timestamp {
                    sort_bypass_cloud(&mut buffers.filtered_cloud, eval, mode)?;
                }
            }

            #[cfg(not(feature = "rayon"))]
            {
                cpu_reduce_downsample(
                    &mut buffers.keyed_points,
                    &mut downsampled,
                    sort_by_timestamp,
                    eval,
                    mode,
                )?;
                if sort_by_timestamp {
                    sort_bypass_cloud(&mut buffers.filtered_cloud, eval, mode)?;
                }
            }

            raw_out = Some(std::mem::take(&mut buffers.filtered_cloud));
        } else {
            cpu_downsample(
                buffers,
                point_skip,
                voxel_size,
                &mut downsampled,
                sort_by_timestamp,
                eval,
                mode,
            )?;
        }
    }
    let Some((first, last)) = scan_time_bounds(&downsampled, raw_out.as_ref(), sort_by_timestamp)
    else {
        return Ok(None);
    };

    // TODO back to keeping the timestamp f64, only testing right now because of artefacts
    Ok(Some((
        (downsampled, raw_out, first as f64, last as f64, header),
        used_cpu_downsample,
    )))
}

/// Applies the configured IMU calibration to incoming samples.
///
/// The parameters are Kalibr's, so `imu_calib::ImuCorrector` holds the inverted
/// matrices and correcting one sample is two matrix-vector products.
///
/// With `imu.calibration.gyroscope` enabled, startup samples are used to update
/// the gyroscope bias and are withheld until calibration completes.
struct ImuIntrinsicCorrection {
    corrector: imu_calib::ImuCorrector,
    intrinsics: imu_calib::ImuIntrinsics,
    warmup: Option<GyroBiasWarmup>,
    announced: bool,
}

/// Running mean of the first stationary samples.
struct GyroBiasWarmup {
    remaining: usize,
    accel_sum: [f64; 3],
    gyro_sum: [f64; 3],
    count: usize,
}

impl ImuIntrinsicCorrection {
    fn load(path: &str, recalibrate_gyro: bool, gyro_samples: usize) -> anyhow::Result<Self> {
        let intrinsics = imu_calib::ImuIntrinsics::from_kalibr_yaml(path)?;
        let corrector = intrinsics.corrector()?;
        Ok(Self {
            corrector,
            intrinsics,
            warmup: (recalibrate_gyro && gyro_samples > 0).then_some(GyroBiasWarmup {
                remaining: gyro_samples,
                accel_sum: [0.0; 3],
                gyro_sum: [0.0; 3],
                count: 0,
            }),
            announced: false,
        })
    }

    /// Correct one sample, or `None` while the gyroscope bias is being measured.
    fn correct(&mut self, accel: [f64; 3], gyro: [f64; 3]) -> Option<([f64; 3], [f64; 3])> {
        if let Some(warmup) = self.warmup.as_mut() {
            if !self.announced {
                log::info!("Calibrating gyros - Do not move the IMU");
                self.announced = true;
            }

            for (sum, value) in warmup.accel_sum.iter_mut().zip(accel.iter()) {
                *sum += value;
            }
            for (sum, value) in warmup.gyro_sum.iter_mut().zip(gyro.iter()) {
                *sum += value;
            }
            warmup.count += 1;
            warmup.remaining -= 1;

            if warmup.remaining > 0 {
                return None;
            }

            let n = warmup.count as f64;
            let mean_accel = warmup.accel_sum.map(|sum| sum / n);
            let mean_gyro = warmup.gyro_sum.map(|sum| sum / n);
            // Include g-sensitivity when estimating the gyroscope bias.
            let bias = self.corrector.gyro_bias_from_static(mean_accel, mean_gyro);
            log::info!(
                "Gyro calibration complete! (bias = [{}, {}, {}])",
                bias[0],
                bias[1],
                bias[2]
            );

            self.intrinsics.gyro_bias = bias;
            match self.intrinsics.corrector() {
                Ok(corrector) => self.corrector = corrector,
                Err(error) => log::error!("Keeping the stored gyroscope bias: {error}"),
            }
            self.warmup = None;
            return None;
        }

        Some(self.corrector.correct(accel, gyro))
    }
}

#[cfg(feature = "threads")]
/// Pose predictors, pacing channel and output hooks that connect a
/// [`Synchronizer`] to the registration stage.
#[derive(Default)]
pub struct SynchronizerHooks {
    pub imu_pose_prediction_undistort: Option<Box<ImuPredictorUndistort>>,
    pub imu_pose_prediction_init_guess: Option<Box<ImuPredictor>>,
    /// Receives the id of each scan once registration finished it.
    pub ready_for_next: Option<std::sync::mpsc::Receiver<u64>>,
    pub realtime_imu_callback: Option<RealtimeImuCallback>,
    pub realtime_lidar_barrier_callback: Option<RealtimeLidarBarrierCallback>,
}

/// Fixed settings of the sync worker.
#[derive(Debug, Clone, Copy)]
pub struct SyncWorkerSettings {
    pub initial_scan_accumulate: usize,
    pub gravity: f64,
    pub imu_intrinsic_calibration_given: bool,
    pub lidar_vendor: LidarVendor,
}

/// Input and output channels of the sync worker.
#[cfg(feature = "threads")]
pub struct SyncWorkerChannels<'a> {
    pub lidar: std::sync::mpsc::Receiver<FilteredCloudMsg>,
    pub imu: Option<std::sync::mpsc::Receiver<ImuMsg>>,
    pub odom: std::sync::mpsc::Receiver<OdomMsg>,
    pub synced: &'a std::sync::mpsc::SyncSender<SyncProcessedCloud>,
    /// Receives the id of each scan once registration finished it.
    pub ready_for_next: Option<std::sync::mpsc::Receiver<u64>>,
}

#[cfg(feature = "threads")]
pub struct Synchronizer {
    preproc_sender: Option<std::sync::mpsc::SyncSender<LabeledPointCloud2Msg>>,
    preproc_queue_depth: Arc<AtomicUsize>,
    last_preproc_warning_message: Option<String>,
    imu_sender: Option<std::sync::mpsc::SyncSender<ImuMsg>>,
    odom_sender: Option<std::sync::mpsc::SyncSender<OdomMsg>>,
    synced_receiver: Option<std::sync::mpsc::Receiver<SyncProcessedCloud>>,
    last_cloud_time: f64,
    imu_filter: ImuFilter,
    runtime_eval: RuntimeEvaluator,
    realtime_imu_callback: Option<RealtimeImuCallback>,
    realtime_lidar_barrier_callback: Option<RealtimeLidarBarrierCallback>,
    lossless_replay: bool,
}

#[cfg(feature = "threads")]
impl Synchronizer {
    pub fn create(
        gravity: f64,
        madgwick_filter: MadgwickSetting,
        imu_intrinsic_calibration_params: &ImuCalibrationParameters,
        params: &SyncParameters,
        lidar_vendor: LidarVendor,
        imu_sync_mode: ImuSyncMode,
        hooks: SynchronizerHooks,
    ) -> anyhow::Result<Self> {
        let SynchronizerHooks {
            imu_pose_prediction_undistort,
            imu_pose_prediction_init_guess,
            ready_for_next,
            realtime_imu_callback,
            realtime_lidar_barrier_callback,
        } = hooks;
        let imu_disabled = matches!(&imu_sync_mode, ImuSyncMode::Disabled { .. });
        // Lidar-only input has no IMU stream to pace it. Apply backpressure at
        // the preprocessing queue instead of silently dropping scans when a
        // dataset or driver produces them faster than registration consumes them.
        let lossless_replay = matches!(
            &imu_sync_mode,
            ImuSyncMode::Disabled { lossless: true } | ImuSyncMode::DemandDrivenReplay { .. }
        );
        #[cfg(not(feature = "rayon"))]
        if params.parallelism_execution_policy
            == pelorus_core::adaptive_execution::ExecutionPolicy::Rayon
        {
            anyhow::bail!("parallelism.execution_mode=Rayon requires the rayon feature");
        }
        let imu_intrinsic_calibration =
            load_imu_intrinsics(imu_disabled, imu_intrinsic_calibration_params)?;

        let (preproc_tx, preproc_rx) =
            std::sync::mpsc::sync_channel::<LabeledPointCloud2Msg>(PREPROC_QUEUE_CAP);
        let preproc_queue_depth = Arc::new(AtomicUsize::new(0));
        let preproc_queue_depth_worker = Arc::clone(&preproc_queue_depth);
        let (imu_tx, imu_rx) = std::sync::mpsc::sync_channel(IMU_QUEUE_CAP);
        let (odom_tx, odom_rx): (
            std::sync::mpsc::SyncSender<OdomMsg>,
            std::sync::mpsc::Receiver<OdomMsg>,
        ) = std::sync::mpsc::sync_channel(ODOM_QUEUE_CAP);
        let (filtered_cloud_tx, filtered_cloud_rx) =
            std::sync::mpsc::sync_channel::<FilteredCloudMsg>(FILTERED_CLOUD_QUEUE_CAP);
        let (synced_cloud_imu_tx, synced_cloud_imu_rx) =
            std::sync::mpsc::sync_channel(SYNCED_CLOUD_QUEUE_CAP);

        // sync and imu integration worker
        // TODO move to main processing, since else predict and correct is async
        let intrinsic_given = imu_intrinsic_calibration.is_some();
        let initial_scan_accumulate = params.initial_scan_accumulate;
        // configure global default for the log debouncer from parameters
        set_default_debounce_secs(params.warning_debounce);

        thread::spawn(move || {
            let mut last_lidar_frame_time = 0.0;
            let imu_process = ImuProcess::new(gravity, intrinsic_given);

            let imu_rx = if cfg!(feature = "no-imu-sync") || imu_disabled {
                None
            } else {
                Some(imu_rx)
            };

            Self::sync_lidar_imu_worker(
                SyncWorkerSettings {
                    initial_scan_accumulate,
                    gravity,
                    imu_intrinsic_calibration_given: intrinsic_given,
                    lidar_vendor,
                },
                &mut last_lidar_frame_time,
                SyncWorkerChannels {
                    lidar: filtered_cloud_rx,
                    imu: imu_rx,
                    odom: odom_rx,
                    synced: &synced_cloud_imu_tx,
                    ready_for_next,
                },
                imu_process,
                imu_pose_prediction_undistort,
                imu_pose_prediction_init_guess,
                imu_sync_mode,
            )
        });

        // preprocessing worker
        let mut preprocessor = Preprocessor::new(params, lidar_vendor);
        thread::spawn(move || {
            while let Ok(labeled) = preproc_rx.recv() {
                preproc_queue_depth_worker
                    .try_update(Ordering::AcqRel, Ordering::Acquire, |depth| {
                        Some(depth.saturating_sub(1))
                    })
                    .ok();
                if let Some(filtered) = preprocessor.process(labeled)
                    && let Err(e) = filtered_cloud_tx.send(filtered)
                {
                    info!("Closing filtering Thread: {}", e);
                    return;
                }
            }
        });

        Ok(Synchronizer {
            imu_sender: (!imu_disabled).then_some(imu_tx),
            odom_sender: Some(odom_tx),
            synced_receiver: Some(synced_cloud_imu_rx),
            preproc_sender: Some(preproc_tx),
            preproc_queue_depth,
            last_preproc_warning_message: None,
            last_cloud_time: 0.0,
            imu_filter: ImuFilter::new(
                gravity,
                madgwick_filter,
                imu_intrinsic_calibration,
                imu_intrinsic_calibration_params.time_offset,
            ),
            runtime_eval: RuntimeEvaluator::new(),
            realtime_imu_callback,
            realtime_lidar_barrier_callback,
            lossless_replay,
        })
    }

    pub fn shutdown(&mut self) {
        self.preproc_sender.take();
        self.imu_sender.take();
        self.odom_sender.take();
    }

    pub fn get_chan(&mut self) -> Option<std::sync::mpsc::Receiver<SyncProcessedCloud>> {
        self.synced_receiver.take()
    }

    // TODO: debouncing - make sure we don't get too many clouds by buffer length and only accepting every nth cloud, then every nth+1 etc after certain buffer lengths each
    // also, this should be configurable with defaults and toggleable
    /// Add a point cloud to the buffer. Returns a `scan_id` assigned to this cloud.
    pub fn add_pointcloud(&mut self, cloud: PointCloud2Msg) -> u64 {
        self.add_pointcloud_with_received_at(cloud, Instant::now())
    }

    /// Add a point cloud with an externally captured callback-entry timestamp.
    pub fn add_pointcloud_with_received_at(
        &mut self,
        cloud: PointCloud2Msg,
        received_at: Instant,
    ) -> u64 {
        self.add_pointcloud_source(Box::new(cloud), received_at)
    }

    /// Enqueue an owned point-cloud source while borrowing its payload in preprocessing.
    pub fn add_pointcloud_source(
        &mut self,
        cloud: Box<dyn PreprocessingPointCloud>,
        received_at: Instant,
    ) -> u64 {
        if let Some(callback) = self.realtime_lidar_barrier_callback.as_ref() {
            callback();
        }
        let cloud_stamp = timestamp_to_secs(&cloud.header().stamp);

        self.last_cloud_time = cloud_stamp;
        // Generate a unique scan id and send a labeled cloud into preprocessing.
        static SCAN_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let scan_id = SCAN_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let labeled = LabeledPointCloud2Msg {
            scan_id,
            received_at,
            cloud,
        };
        match self.preproc_sender.as_ref() {
            Some(sender) => {
                let enqueue_result = if self.lossless_replay {
                    sender.send(labeled).map_err(|error| error.0)
                } else {
                    sender.try_send(labeled).map_err(|error| match error {
                        std::sync::mpsc::TrySendError::Full(value)
                        | std::sync::mpsc::TrySendError::Disconnected(value) => value,
                    })
                };
                match enqueue_result {
                    Ok(_) => {
                        let queue_depth =
                            self.preproc_queue_depth.fetch_add(1, Ordering::Relaxed) + 1;
                        if queue_depth * 4 > PREPROC_QUEUE_CAP * 3 {
                            let warning_message = format!(
                                "Pointcloud preprocessing queue is more than 75% full ({}/{})",
                                queue_depth, PREPROC_QUEUE_CAP
                            );
                            if self.last_preproc_warning_message.as_deref()
                                != Some(warning_message.as_str())
                                && should_log_once_default("pointcloud_preproc_queue_state_change")
                            {
                                warn!("{warning_message}");
                                self.last_preproc_warning_message = Some(warning_message);
                            }
                        } else {
                            self.last_preproc_warning_message = None;
                        }
                    }
                    Err(_) if !self.lossless_replay => {
                        let warning_message = format!(
                            "Dropping pointcloud: preprocessing queue is full ({}/{})",
                            PREPROC_QUEUE_CAP, PREPROC_QUEUE_CAP
                        );
                        if self.last_preproc_warning_message.as_deref()
                            != Some(warning_message.as_str())
                            && should_log_once_default("pointcloud_preproc_queue_state_change")
                        {
                            warn!("{warning_message}");
                            self.last_preproc_warning_message = Some(warning_message);
                        }
                    }
                    Err(_) => {
                        error!("Error sending pointcloud: preprocessing pipeline is disconnected");
                    }
                }
            }
            None => {
                debug!("Pointcloud ignored: pre-processing pipeline is shutting down");
            }
        }
        scan_id
    }

    /// Filter raw IMU message and add to the buffer for synchronization and integration.
    /// Returns the filtered IMU message for republishing.
    pub fn add_imu(&mut self, imu: ImuMsg) -> Option<ImuMsg> {
        self.runtime_eval.start("imu_pre").unwrap();
        let imu = self.imu_filter.filter(imu)?;

        // Add to buffer
        let imu_data = imu.clone();
        match self.imu_sender.as_ref() {
            Some(sender) if self.lossless_replay => match sender.send(imu_data) {
                Ok(_) => {}
                Err(_) => {
                    self.runtime_eval.stop("imu_pre").unwrap();
                    return None;
                }
            },
            Some(sender) => match sender.try_send(imu_data) {
                Ok(_) => {}
                Err(std::sync::mpsc::TrySendError::Full(_)) => {
                    if should_log_once_default("drop_imu_queue_full") {
                        warn!("Dropping IMU message: IMU queue is full");
                    }
                    self.runtime_eval.stop("imu_pre").unwrap();
                    return None;
                }
                Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                    self.runtime_eval.stop("imu_pre").unwrap();
                    return None;
                }
            },
            None => {
                self.runtime_eval.stop("imu_pre").unwrap();
                return None;
            }
        };
        self.runtime_eval.stop("imu_pre").unwrap();

        if let Some(callback) = self.realtime_imu_callback.as_ref() {
            callback(imu.clone());
        }

        Some(imu)
    }

    /// Add an odometry message to the buffer for synchronization with lidar.
    pub fn add_odom(&mut self, odom: OdomMsg) {
        match self.odom_sender.as_ref() {
            Some(sender) if self.lossless_replay => {
                if sender.send(odom).is_err() {
                    error!("Error while sending odom message to sync worker: disconnected");
                }
            }
            Some(sender) => match sender.try_send(odom) {
                Ok(_) => {}
                Err(std::sync::mpsc::TrySendError::Full(_)) => {
                    if should_log_once_default("drop_odom_queue_full") {
                        warn!("Dropping odometry message: odometry queue is full");
                    }
                }
                Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                    log::error!("Error while sending odom message to sync worker: disconnected");
                }
            },
            None => {
                debug!("Odom ignored: sync worker is shutting down");
            }
        }
    }

    /// Syncs lidar, imu, and odometry data by using the timestamps from the headers and timestamps in the lidar message.
    /// A lidar message is therefore mapped to multiple imu/odometry messages that happened in between the previous end stamp and end stamp of the last point in the lidar message.
    ///
    /// This is the threaded driver around [`SyncCore`]. It pulls messages from the
    /// channels according to `imu_sync_mode` and waits for registration through
    /// `ready_for_next` before releasing the next scan.
    pub fn sync_lidar_imu_worker(
        settings: SyncWorkerSettings,
        prev_lidar_end_time: &mut f64,
        channels: SyncWorkerChannels<'_>,
        imu_process: ImuProcess,
        imu_pose_prediction_undistort: Option<Box<ImuPredictorUndistort>>,
        imu_pose_prediction_init_guess: Option<Box<ImuPredictor>>,
        mut imu_sync_mode: ImuSyncMode,
    ) {
        let SyncWorkerChannels {
            lidar: lidar_receiver,
            imu: imu_receiver,
            odom: odom_receiver,
            synced: synced_sender,
            ready_for_next,
        } = channels;
        let mut core = SyncCore::new(
            settings,
            imu_receiver.is_some(),
            *prev_lidar_end_time,
            imu_process,
            imu_pose_prediction_undistort,
            imu_pose_prediction_init_guess,
        );
        let mut awaiting_correction: Option<u64> = None;
        while let Ok(filtered) = lidar_receiver.recv() {
            if !core.is_initializing()
                && let Some(expected_scan_id) = awaiting_correction.take()
                && let Some(completed) = ready_for_next.as_ref()
            {
                loop {
                    match completed.recv() {
                        Ok(scan_id) if scan_id == expected_scan_id => break,
                        Ok(scan_id) if scan_id < expected_scan_id => continue,
                        Ok(scan_id) => {
                            // The barrier is already past the scan it was
                            // waiting for, so carry on instead of tearing the
                            // synchronizer down and stalling the whole pipeline.
                            warn!(
                                "Expected correction for scan {expected_scan_id}, received scan {scan_id}"
                            );
                            break;
                        }
                        Err(_) => {
                            warn!("Could not wait for next sync");
                            return;
                        }
                    }
                }
            }

            let scan_id = filtered.1;
            let synced = match core.begin_scan(filtered) {
                ScanStart::Ready(synced) => Some(*synced),
                ScanStart::Skipped => None,
                ScanStart::CollectingImu => {
                    if let Some(imu_receiver) = imu_receiver.as_ref() {
                        let mut requested_next_replay_scan = false;
                        loop {
                            let imu = match &mut imu_sync_mode {
                                ImuSyncMode::Disabled { .. } => break,
                                ImuSyncMode::Continuous => imu_receiver.recv(),
                                ImuSyncMode::LidarDriven => match imu_receiver.try_recv() {
                                    Ok(imu) => Ok(imu),
                                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                                        error!("Imu receiver closed");
                                        return;
                                    }
                                },
                                ImuSyncMode::DemandDrivenReplay { request_next_scan } => {
                                    match imu_receiver.try_recv() {
                                        Ok(imu) => Ok(imu),
                                        Err(std::sync::mpsc::TryRecvError::Empty) => {
                                            if !requested_next_replay_scan {
                                                request_next_scan();
                                                requested_next_replay_scan = true;
                                            }
                                            imu_receiver.recv()
                                        }
                                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                                            info!(
                                                "Replay input closed: scan {scan_id} has no IMU after its end, stopping"
                                            );
                                            return;
                                        }
                                    }
                                }
                            };

                            let imu = match imu {
                                Ok(imu) => imu,
                                Err(_)
                                    if matches!(
                                        imu_sync_mode,
                                        ImuSyncMode::DemandDrivenReplay { .. }
                                    ) =>
                                {
                                    // The replay closed the inputs once everything it sent
                                    // had arrived, so this scan's IMU can never come.
                                    info!(
                                        "Replay input closed: scan {scan_id} has no IMU after its end, stopping"
                                    );
                                    return;
                                }
                                Err(e) => {
                                    error!("Imu receiver closed: {e}");
                                    return;
                                }
                            };
                            if core.offer_imu(imu) {
                                break;
                            }
                        }
                    }
                    // Closed inputs end the odometry, not the scans already queued.
                    // Those still finish with the IMU they have.
                    core.finish_scan(odom_receiver.try_iter())
                }
            };
            *prev_lidar_end_time = core.prev_lidar_end_time();

            if let Some(synced) = synced {
                synced_sender.send(synced).unwrap();
                awaiting_correction = Some(scan_id);
            }
        }
    }
}

/// Calibration, clock offset and Madgwick filtering applied to every IMU
/// message before synchronization.
struct ImuFilter {
    imu_filter_fusion: ImuMadgwick,
    last_imu_time: f64,
    imu_calibration_apply: Option<ImuIntrinsicCorrection>,
    /// Seconds the IMU clock runs ahead of the Lidar clock.
    imu_time_offset: f64,
    madgwick_filter: MadgwickSetting,
}

impl ImuFilter {
    fn new(
        gravity: f64,
        madgwick_filter: MadgwickSetting,
        imu_calibration_apply: Option<ImuIntrinsicCorrection>,
        imu_time_offset: f64,
    ) -> Self {
        Self {
            imu_filter_fusion: ImuMadgwick::new(gravity),
            last_imu_time: 0.0,
            imu_calibration_apply,
            imu_time_offset,
            madgwick_filter,
        }
    }

    /// Returns the filtered message, or `None` while the intrinsic calibration
    /// still collects gyro samples.
    fn filter(&mut self, mut imu: ImuMsg) -> Option<ImuMsg> {
        // Integration, undistortion, and pairing all use the corrected clock.
        if self.imu_time_offset != 0.0 {
            imu.stamp = pelorus_core::time_msg_from_secs(
                timestamp_to_secs(&imu.stamp) - self.imu_time_offset,
            );
        }
        let imu_time = timestamp_to_secs(&imu.stamp);

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
            let (corrected_accel, corrected_gyro) = imu_calib.correct(accel, gyro)?;
            accel = corrected_accel;
            gyro = corrected_gyro;
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
        Some(imu)
    }
}

fn load_imu_intrinsics(
    imu_disabled: bool,
    params: &ImuCalibrationParameters,
) -> Result<Option<ImuIntrinsicCorrection>> {
    match (imu_disabled, params.imu_intrinsics_calib_file_path.as_ref()) {
        (true, _) | (false, None) => Ok(None),
        (false, Some(path)) => Ok(Some(ImuIntrinsicCorrection::load(
            path,
            params.imu_calibrate_gyros,
            params.gyro_calibration_samples,
        )?)),
    }
}

/// Prediction and output hooks of a [`SteppedSynchronizer`].
#[derive(Default)]
pub struct SteppedSynchronizerHooks {
    pub imu_pose_prediction_undistort: Option<Box<ImuPredictorUndistort>>,
    pub imu_pose_prediction_init_guess: Option<Box<ImuPredictor>>,
    pub realtime_imu_callback: Option<RealtimeImuCallback>,
    pub realtime_lidar_barrier_callback: Option<RealtimeLidarBarrierCallback>,
}

/// Synchronizer that preprocesses, synchronizes and deskews on the calling
/// thread.
///
/// Inputs are buffered by the `add_*` calls. [`next_synced`](Self::next_synced)
/// returns the next scan whose IMU interval is complete. Register each returned
/// scan before calling it again, because the prediction hooks read the
/// registration state.
pub struct SteppedSynchronizer {
    preprocessor: Preprocessor,
    imu_filter: ImuFilter,
    core: SyncCore,
    lidar_driven: bool,
    scans: VecDeque<FilteredCloudMsg>,
    imu: VecDeque<ImuMsg>,
    odom: Vec<OdomMsg>,
    next_scan_id: u64,
    realtime_imu_callback: Option<RealtimeImuCallback>,
    realtime_lidar_barrier_callback: Option<RealtimeLidarBarrierCallback>,
}

impl SteppedSynchronizer {
    /// `ImuSyncMode::DemandDrivenReplay` behaves like `ImuSyncMode::Continuous`
    /// because the caller controls when inputs arrive.
    pub fn create(
        params: &SyncParameters,
        lidar_vendor: LidarVendor,
        imu_sync_mode: ImuSyncMode,
        hooks: SteppedSynchronizerHooks,
    ) -> anyhow::Result<Self> {
        let imu_disabled = matches!(&imu_sync_mode, ImuSyncMode::Disabled { .. });
        #[cfg(not(feature = "rayon"))]
        if params.parallelism_execution_policy
            == pelorus_core::adaptive_execution::ExecutionPolicy::Rayon
        {
            anyhow::bail!("parallelism.execution_mode=Rayon requires the rayon feature");
        }
        let imu_intrinsic_calibration = load_imu_intrinsics(imu_disabled, &params.imu.calibration)?;
        let imu_intrinsic_calibration_given = imu_intrinsic_calibration.is_some();
        set_default_debounce_secs(params.warning_debounce);
        let core = SyncCore::new(
            SyncWorkerSettings {
                initial_scan_accumulate: params.initial_scan_accumulate,
                gravity: params.imu.gravity,
                imu_intrinsic_calibration_given,
                lidar_vendor,
            },
            !(cfg!(feature = "no-imu-sync") || imu_disabled),
            0.0,
            ImuProcess::new(params.imu.gravity, imu_intrinsic_calibration_given),
            hooks.imu_pose_prediction_undistort,
            hooks.imu_pose_prediction_init_guess,
        );
        Ok(Self {
            preprocessor: Preprocessor::new(params, lidar_vendor),
            imu_filter: ImuFilter::new(
                params.imu.gravity,
                params.imu.madgwick_setting,
                imu_intrinsic_calibration,
                params.imu.calibration.time_offset,
            ),
            core,
            lidar_driven: matches!(imu_sync_mode, ImuSyncMode::LidarDriven),
            scans: VecDeque::new(),
            imu: VecDeque::new(),
            odom: Vec::new(),
            next_scan_id: 1,
            realtime_imu_callback: hooks.realtime_imu_callback,
            realtime_lidar_barrier_callback: hooks.realtime_lidar_barrier_callback,
        })
    }

    /// Preprocess a point cloud and queue it. Returns the `scan_id` assigned to it.
    pub fn add_pointcloud(&mut self, cloud: PointCloud2Msg) -> u64 {
        self.add_pointcloud_source(Box::new(cloud), Instant::now())
    }

    /// Preprocess an owned point-cloud source and queue it.
    pub fn add_pointcloud_source(
        &mut self,
        cloud: Box<dyn PreprocessingPointCloud>,
        received_at: Instant,
    ) -> u64 {
        if let Some(callback) = self.realtime_lidar_barrier_callback.as_ref() {
            callback();
        }
        let scan_id = self.next_scan_id;
        self.next_scan_id += 1;
        if let Some(filtered) = self.preprocessor.process(LabeledPointCloud2Msg {
            scan_id,
            received_at,
            cloud,
        }) {
            self.scans.push_back(filtered);
        }
        scan_id
    }

    /// Filter an IMU message and queue it. Returns the filtered message, or
    /// `None` when IMU synchronization is disabled or the intrinsic calibration
    /// still collects samples.
    pub fn add_imu(&mut self, imu: ImuMsg) -> Option<ImuMsg> {
        let imu = self.imu_filter.filter(imu)?;
        if !self.core.imu_enabled() {
            return None;
        }
        self.imu.push_back(imu.clone());
        if let Some(callback) = self.realtime_imu_callback.as_ref() {
            callback(imu.clone());
        }
        Some(imu)
    }

    /// Queue an odometry message for the next synchronized scan.
    pub fn add_odom(&mut self, odom: OdomMsg) {
        self.odom.push(odom);
    }

    /// Number of preprocessed scans waiting for synchronization, including
    /// the one collecting IMU messages.
    pub fn pending_scans(&self) -> usize {
        self.scans.len() + usize::from(self.core.is_collecting())
    }

    /// Synchronize and deskew the next scan whose IMU interval is complete.
    ///
    /// Returns `None` once no queued scan can be completed with the inputs
    /// received so far.
    pub fn next_synced(&mut self) -> Option<SyncProcessedCloud> {
        loop {
            if !self.core.is_collecting() {
                match self.core.begin_scan(self.scans.pop_front()?) {
                    ScanStart::Ready(synced) => return Some(*synced),
                    ScanStart::Skipped => continue,
                    ScanStart::CollectingImu => {}
                }
            }
            if self.core.imu_enabled() {
                let mut interval_complete = false;
                while let Some(imu) = self.imu.pop_front() {
                    if self.core.offer_imu(imu) {
                        interval_complete = true;
                        break;
                    }
                }
                if !interval_complete && !self.lidar_driven {
                    return None;
                }
            }
            if let Some(synced) = self.core.finish_scan(std::mem::take(&mut self.odom)) {
                return Some(synced);
            }
        }
    }
}

/// Collects the IMU messages of the first seconds of a run and hands them to
/// the EKF initialization.
///
/// The messages still go on to synchronization, so scan synchronization does
/// not depend on when the initialization runs.
pub struct ImuInitWindow {
    callback: Option<ImuInitCallback>,
    init_time_sec: f64,
    start: Option<f64>,
    buffer: Option<Vec<ImuMsg>>,
}

impl ImuInitWindow {
    /// Without a callback the window is complete from the start.
    pub fn new(callback: Option<ImuInitCallback>, init_time_sec: f64) -> Self {
        Self {
            callback,
            init_time_sec,
            start: None,
            buffer: Some(Vec::new()),
        }
    }

    /// Record an IMU message. The first message `init_time_sec` after the
    /// first one runs the initialization with the messages before it. With a
    /// window of zero or less the first message initializes on its own.
    ///
    /// A message older than the window start, as when a bag loops, starts the
    /// window over.
    pub fn observe(&mut self, imu: &ImuMsg) {
        let Some(callback) = self.callback.as_ref() else {
            return;
        };
        let Some(buffer) = self.buffer.as_mut() else {
            return;
        };
        let stamp = timestamp_to_secs(&imu.stamp);
        let start = *self.start.get_or_insert(stamp);
        if stamp < start {
            self.start = Some(stamp);
            buffer.clear();
        }
        let start = self.start.unwrap_or(stamp);
        if self.init_time_sec <= 0.0 {
            buffer.push(imu.clone());
        } else if stamp - start < self.init_time_sec {
            buffer.push(imu.clone());
            return;
        }
        let buffer = self.buffer.take().unwrap_or_default();
        // Initialization needs at least one sample.
        if !buffer.is_empty() {
            (callback.lock().unwrap())(&buffer);
            info!("EKF: IMU initialization samples collected");
        }
    }

    /// Whether the initialization ran, or none is needed.
    pub fn is_complete(&self) -> bool {
        self.callback.is_none() || self.buffer.is_none()
    }
}

/// Sensor input in front of a [`SteppedSynchronizer`], with the rules of the
/// Pelorus node: IMU initialization, and dropping empty and out-of-order clouds.
pub struct SteppedIngress {
    sync: SteppedSynchronizer,
    imu_init: ImuInitWindow,
    last_lidar_time: Option<f64>,
}

impl SteppedIngress {
    /// `imu_init` receives the IMU messages of the first `imu_init_time_sec`
    /// seconds. Those messages also go on to synchronization.
    pub fn new(
        sync: SteppedSynchronizer,
        imu_init: Option<ImuInitCallback>,
        imu_init_time_sec: f64,
    ) -> Self {
        Self {
            sync,
            imu_init: ImuInitWindow::new(imu_init, imu_init_time_sec),
            last_lidar_time: None,
        }
    }

    /// Queue a point cloud. Returns its scan id, or `None` for an empty cloud
    /// or one older than the previous cloud.
    pub fn push_cloud(
        &mut self,
        cloud: Box<dyn PreprocessingPointCloud>,
        received_at: Instant,
    ) -> Option<u64> {
        if cloud.source().point_count() == 0 {
            return None;
        }
        let time = timestamp_to_secs(&cloud.header().stamp);
        if self.last_lidar_time.is_some_and(|last| time < last) {
            return None;
        }
        self.last_lidar_time = Some(time);
        Some(self.sync.add_pointcloud_source(cloud, received_at))
    }

    /// Queue an IMU message. Returns the filtered message for republishing.
    pub fn push_imu(&mut self, imu: ImuMsg) -> Option<ImuMsg> {
        self.imu_init.observe(&imu);
        self.sync.add_imu(imu)
    }

    /// Whether the IMU initialization is complete, or not needed.
    pub fn imu_initialized(&self) -> bool {
        self.imu_init.is_complete()
    }

    /// Queue a wheel-odometry message.
    pub fn push_odom(&mut self, odom: OdomMsg) {
        self.sync.add_odom(odom);
    }

    /// The next synchronized and deskewed scan. See
    /// [`SteppedSynchronizer::next_synced`].
    pub fn next_synced(&mut self) -> Option<SyncProcessedCloud> {
        self.sync.next_synced()
    }

    /// Number of scans waiting for IMU data.
    pub fn pending_scans(&self) -> usize {
        self.sync.pending_scans()
    }
}

/// Point-cloud filtering and downsampling ahead of [`SyncCore`].
pub struct Preprocessor {
    params: SyncParameters,
    vendor: LidarVendor,
    eval: RuntimeEvaluator,
    downsample_buffers: DownsampleBuffers,
    #[cfg(feature = "rayon")]
    execution_planner: AdaptiveExecutionPlanner,
}

impl Preprocessor {
    pub fn new(params: &SyncParameters, vendor: LidarVendor) -> Self {
        Self {
            params: params.clone(),
            vendor,
            eval: RuntimeEvaluator::new(),
            downsample_buffers: DownsampleBuffers::default(),
            #[cfg(feature = "rayon")]
            execution_planner: AdaptiveExecutionPlanner::new(
                params.parallelism_decision_threshold_stddevs,
                params.parallelism_decision_samples_per_mode,
                params.parallelism_execution_policy,
            ),
        }
    }

    /// Filter and downsample one scan. Returns `None` when the scan is filtered
    /// out or fails to parse.
    pub fn process(&mut self, labeled: LabeledPointCloud2Msg) -> Option<FilteredCloudMsg> {
        self.eval.start("filter_downsample").unwrap();
        #[cfg(feature = "rayon")]
        let mode = self.execution_planner.mode_for(HeavyPhase::Preprocessing);
        #[cfg(not(feature = "rayon"))]
        let mode = ExecutionMode::Sequential;
        #[cfg(feature = "rayon")]
        let started = Instant::now();
        #[cfg(feature = "rayon")]
        let input_points = labeled.cloud.source().point_count().max(1);

        let result = match self.vendor {
            LidarVendor::Ouster => {
                filter_downsample_with_fallback::<5, 4, OusterPoint, OusterPointNoIntensity>(
                    labeled.cloud.as_ref(),
                    &self.params,
                    &mut self.eval,
                    &mut self.downsample_buffers,
                    mode,
                )
            }
            LidarVendor::Velodyne => {
                filter_downsample_with_fallback::<5, 4, VelodynePoint, VelodynePointNoIntensity>(
                    labeled.cloud.as_ref(),
                    &self.params,
                    &mut self.eval,
                    &mut self.downsample_buffers,
                    mode,
                )
            }
            LidarVendor::Livox => filter_downsample_livox(
                labeled.cloud.as_ref(),
                &self.params,
                &mut self.eval,
                &mut self.downsample_buffers,
                mode,
            ),
            LidarVendor::LivoxMid360 => {
                filter_downsample_with_fallback::<5, 4, LivoxMid360Point, LivoxMid360PointNoIntensity>(
                    labeled.cloud.as_ref(),
                    &self.params,
                    &mut self.eval,
                    &mut self.downsample_buffers,
                    mode,
                )
            }
            LidarVendor::Hesai => {
                filter_downsample_with_fallback::<5, 4, HesaiPoint, HesaiPointNoIntensity>(
                    labeled.cloud.as_ref(),
                    &self.params,
                    &mut self.eval,
                    &mut self.downsample_buffers,
                    mode,
                )
            }
            LidarVendor::Blickfeld => {
                filter_downsample_with_fallback::<5, 4, BlickfeldPoint, BlickfeldPointNoIntensity>(
                    labeled.cloud.as_ref(),
                    &self.params,
                    &mut self.eval,
                    &mut self.downsample_buffers,
                    mode,
                )
            }
            LidarVendor::SickMultiscan => filter_downsample_with_fallback::<
                5,
                4,
                SickMultiScanPoint,
                SickMultiScanPointNoIntensity,
            >(
                labeled.cloud.as_ref(),
                &self.params,
                &mut self.eval,
                &mut self.downsample_buffers,
                mode,
            ),
            LidarVendor::Robosense => {
                filter_downsample_with_fallback::<5, 4, RobosensePoint, RobosensePointNoIntensity>(
                    labeled.cloud.as_ref(),
                    &self.params,
                    &mut self.eval,
                    &mut self.downsample_buffers,
                    mode,
                )
            }
            LidarVendor::NoTimestamp => {
                filter_downsample_without_timestamps_with_fallback::<4, 3, PointXYZI, PointXYZ>(
                    labeled.cloud.as_ref(),
                    &self.params,
                    &mut self.eval,
                    &mut self.downsample_buffers,
                    mode,
                )
            }
        };

        match result {
            Ok(Some((cloud_meta, used_cpu_downsample))) => {
                self.eval.stop("filter_downsample").unwrap();
                #[cfg(feature = "rayon")]
                if used_cpu_downsample {
                    let nanos_per_point = started.elapsed().as_nanos() as f64 / input_points as f64;
                    if let Some(decision) = self.execution_planner.record(
                        HeavyPhase::Preprocessing,
                        mode,
                        nanos_per_point,
                    ) {
                        info!(
                            "{}: using {} ({:.2}x faster)",
                            HeavyPhase::Preprocessing.name(),
                            decision.mode.name(),
                            decision.speedup,
                        );
                    }
                }
                #[cfg(not(feature = "rayon"))]
                let _ = used_cpu_downsample;
                Some((
                    cloud_meta,
                    labeled.scan_id,
                    labeled.received_at,
                    self.eval.clone(),
                ))
            }
            Ok(None) => {
                self.eval.stop("filter_downsample").unwrap();
                info!("Pointcloud was filtered out");
                None
            }
            Err(e) => {
                self.eval.stop("filter_downsample").unwrap();
                error!("Error processing pointcloud: {}", e);
                None
            }
        }
    }
}

/// Outcome of [`SyncCore::begin_scan`].
pub enum ScanStart {
    /// The scan is complete and needs no IMU data. Initialization scans end here.
    Ready(Box<SyncProcessedCloud>),
    /// The scan covers an empty interval and is dropped.
    Skipped,
    /// The scan collects IMU messages through [`SyncCore::offer_imu`] and is
    /// completed by [`SyncCore::finish_scan`].
    CollectingImu,
}

/// Scan held by [`SyncCore`] while it collects the IMU messages of its interval.
struct PendingScan {
    cloud_downsampled: Vec<PointXYZIT>,
    cloud_output: Option<Vec<PointXYZIT>>,
    lidar_header: HeaderMsg,
    scan_id: u64,
    received_at: Instant,
    /// Raw header-relative end stamp of the last point.
    lidar_end_time: f64,
    /// Absolute begin of the scan used for deskewing.
    lidar_begin_time: f64,
    imu_span_start_time: f64,
    imu_span_end_time: f64,
    no_previous_lidar_end_time: bool,
    use_point_timestamps_for_frame: bool,
    abs_lidar_end: TimeMsg,
    imu_data: Vec<ImuMsg>,
}

/// Lidar/IMU synchronization as a state machine without threads or channels.
///
/// A scan enters with [`begin_scan`](Self::begin_scan). While it is collecting,
/// [`offer_imu`](Self::offer_imu) takes IMU messages in time order until one lies
/// past the end of the scan interval. [`finish_scan`](Self::finish_scan) then
/// deskews the scan and returns it for registration.
///
/// Prediction hooks read the registration state, so each returned scan has to
/// be registered before the next scan begins.
pub struct SyncCore {
    settings: SyncWorkerSettings,
    imu_enabled: bool,
    prev_lidar_end_time: f64,
    imu_process: ImuProcess,
    imu_pose_prediction_undistort: Option<Box<ImuPredictorUndistort>>,
    imu_pose_prediction_init_guess: Option<Box<ImuPredictor>>,
    eval: RuntimeEvaluator,
    /// First IMU message past the previous interval, carried into the next one.
    to_next: Option<ImuMsg>,
    last_received_scan_id: Option<u64>,
    pending_skipped_scans: u64,
    cloud_count: usize,
    pending: Option<PendingScan>,
}

impl SyncCore {
    pub fn new(
        settings: SyncWorkerSettings,
        imu_enabled: bool,
        prev_lidar_end_time: f64,
        imu_process: ImuProcess,
        imu_pose_prediction_undistort: Option<Box<ImuPredictorUndistort>>,
        imu_pose_prediction_init_guess: Option<Box<ImuPredictor>>,
    ) -> Self {
        Self {
            settings,
            imu_enabled,
            prev_lidar_end_time,
            imu_process,
            imu_pose_prediction_undistort,
            imu_pose_prediction_init_guess,
            eval: RuntimeEvaluator::new(),
            to_next: None,
            last_received_scan_id: None,
            pending_skipped_scans: 0,
            cloud_count: 0,
            pending: None,
        }
    }

    /// Whether the next scan is still part of the initial map accumulation.
    pub fn is_initializing(&self) -> bool {
        self.cloud_count < self.settings.initial_scan_accumulate
    }

    /// Whether a scan is waiting for IMU messages.
    pub fn is_collecting(&self) -> bool {
        self.pending.is_some()
    }

    /// Whether scans wait for IMU messages at all.
    pub fn imu_enabled(&self) -> bool {
        self.imu_enabled
    }

    /// End of the last synchronized interval, in seconds.
    pub fn prev_lidar_end_time(&self) -> f64 {
        self.prev_lidar_end_time
    }

    /// Start synchronizing a preprocessed scan.
    ///
    /// A scan that is still collecting is replaced.
    pub fn begin_scan(&mut self, filtered: FilteredCloudMsg) -> ScanStart {
        let (
            (cloud_downsampled, cloud_output, lidar_begin_time, lidar_end_time, lidar_header),
            scan_id,
            received_at,
            sync_eval,
        ) = filtered;
        let SyncWorkerSettings { lidar_vendor, .. } = self.settings;
        let initializing = self.is_initializing();
        if let Some(previous_scan_id) = self.last_received_scan_id {
            self.pending_skipped_scans += scan_id.wrapping_sub(previous_scan_id).saturating_sub(1);
        }
        self.last_received_scan_id = Some(scan_id);
        // Bursts while the input is still spinning up (a bag player warming up,
        // a sensor driver flushing its first batch) are expected, so drops are
        // only reported once the pipeline runs steadily, and then aggregated
        // instead of one warning per gap.
        if initializing {
            self.pending_skipped_scans = 0;
        } else if self.pending_skipped_scans > 0
            && should_log_once_default("pointcloud_preproc_skipped_scans")
        {
            warn!(
                "Pointcloud preprocessing dropped {} scan(s) to keep up with the input rate",
                self.pending_skipped_scans
            );
            self.pending_skipped_scans = 0;
        }
        self.eval.merge(&sync_eval);
        let lidar_time = timestamp_to_secs(&lidar_header.stamp);
        let abs_lidar_end = secs_to_timestamp(lidar_time + lidar_end_time);

        if initializing {
            self.cloud_count += 1;
            if !has_point_timestamps(lidar_vendor) {
                self.prev_lidar_end_time = lidar_time;
            }
            return ScanStart::Ready(Box::new(SyncProcessedCloud {
                scan_id,
                received_at,
                cloud: cloud_downsampled,
                passthrough: cloud_output,
                imu_buffer: Vec::new(),
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
                runtime_eval: self.eval.clone(),
                should_map_only: true,
                lidar_end_time_abs: abs_lidar_end,
            }));
        }

        // Handle relative vs absolute timestamps
        let lidar_point_time_span = lidar_end_time - lidar_begin_time;
        let livox_use_header_span = matches!(lidar_vendor, LidarVendor::Livox)
            && (lidar_point_time_span <= f64::EPSILON || lidar_end_time > 0.2);
        let use_point_timestamps_for_frame =
            has_point_timestamps(lidar_vendor) && !livox_use_header_span;
        let (lidar_begin_time, imu_span_end_time) = match lidar_vendor {
            LidarVendor::Livox if livox_use_header_span => (lidar_time, lidar_time),
            LidarVendor::LivoxMid360 => (lidar_time - lidar_end_time, lidar_time),
            LidarVendor::NoTimestamp => (lidar_time, lidar_time),
            _ => (lidar_begin_time + lidar_time, lidar_end_time + lidar_time),
        };

        // Use 0 as start if there was no last lidar frame which we can use as start for our time range
        let no_previous_lidar_end_time = self.prev_lidar_end_time <= 0.0;
        let imu_span_start_time = if self.prev_lidar_end_time > 0.0 {
            self.prev_lidar_end_time
        } else if !has_point_timestamps(lidar_vendor) {
            0.0
        } else {
            lidar_begin_time
        };
        if imu_span_end_time <= imu_span_start_time {
            return ScanStart::Skipped;
        }

        let mut imu_data = Vec::with_capacity(100);

        // Get last IMU message that was destructively taken
        if let Some(last_imu_extra) = self.to_next.take() {
            let imu_time = timestamp_to_secs(&last_imu_extra.stamp);
            if imu_time > imu_span_start_time && imu_time < imu_span_end_time {
                imu_data.push(last_imu_extra);
            }
        }

        self.pending = Some(PendingScan {
            cloud_downsampled,
            cloud_output,
            lidar_header,
            scan_id,
            received_at,
            lidar_end_time,
            lidar_begin_time,
            imu_span_start_time,
            imu_span_end_time,
            no_previous_lidar_end_time,
            use_point_timestamps_for_frame,
            abs_lidar_end,
            imu_data,
        });
        ScanStart::CollectingImu
    }

    /// Offer the next IMU message in time order to the collecting scan.
    ///
    /// Returns `true` once the message lies past the end of the scan interval.
    /// That message is kept for the next scan and the current scan is ready for
    /// [`finish_scan`](Self::finish_scan). Without a collecting scan the message
    /// is dropped.
    pub fn offer_imu(&mut self, imu: ImuMsg) -> bool {
        let Some(pending) = self.pending.as_mut() else {
            return false;
        };
        let imu_time = timestamp_to_secs(&imu.stamp);
        if imu_time < pending.imu_span_start_time {
            return false;
        }
        if imu_time > pending.imu_span_end_time {
            self.to_next = Some(imu);
            return true;
        }
        pending.imu_data.push(imu);
        false
    }

    /// Deskew the collecting scan with the IMU messages it received and the
    /// given odometry messages.
    ///
    /// Odometry outside the scan interval is dropped. Returns `None` when no
    /// scan is collecting, when an IMU-synchronized scan received no IMU
    /// message, or when deskewing fails.
    pub fn finish_scan(
        &mut self,
        odom: impl IntoIterator<Item = OdomMsg>,
    ) -> Option<SyncProcessedCloud> {
        let PendingScan {
            mut cloud_downsampled,
            mut cloud_output,
            lidar_header,
            scan_id,
            received_at,
            lidar_end_time,
            lidar_begin_time,
            mut imu_span_start_time,
            imu_span_end_time,
            no_previous_lidar_end_time,
            use_point_timestamps_for_frame,
            abs_lidar_end,
            mut imu_data,
        } = self.pending.take()?;
        let SyncWorkerSettings {
            gravity,
            imu_intrinsic_calibration_given,
            lidar_vendor,
            ..
        } = self.settings;

        if self.imu_enabled && imu_data.is_empty() {
            warn!("No imu data in span");
            return None;
        }

        let mut odom_data = Vec::with_capacity(100);
        for odom in odom {
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

        // Sort buffers with timestamp to ensure correct order
        imu_data.sort_by(|a, b| {
            timestamp_to_secs(&a.stamp)
                .partial_cmp(&timestamp_to_secs(&b.stamp))
                .unwrap()
        });
        if !has_point_timestamps(lidar_vendor)
            && no_previous_lidar_end_time
            && let Some(first_imu) = imu_data.first()
        {
            imu_span_start_time = timestamp_to_secs(&first_imu.stamp);
        }
        odom_data.sort_by(|a, b| {
            timestamp_to_secs(&a.stamp)
                .partial_cmp(&timestamp_to_secs(&b.stamp))
                .unwrap()
        });

        // `undistort` takes the frame end in the same form the IMU path
        // hands it over: header-relative when the points carry stamps.
        let deskew_end_time = if use_point_timestamps_for_frame {
            lidar_end_time
        } else {
            imu_span_end_time
        };

        let synced = if imu_data.is_empty() {
            // A runtime Lidar-only pipeline still uses this callback to
            // synthesize constant-twist IMU samples, propagate the EKF,
            // and deskew the cloud. Only the generic no-callback case is
            // a true passthrough.
            let end_of_frame_guess = match self.imu_pose_prediction_undistort.as_ref() {
                Some(undistort) => match undistort(
                    &[],
                    &odom_data,
                    imu_span_start_time,
                    lidar_begin_time,
                    deskew_end_time,
                    &mut cloud_downsampled,
                    cloud_output.as_mut(),
                ) {
                    Left(prediction) => prediction.pose,
                    Right(pose) => pose,
                },
                None => Isometry3::identity(),
            };

            Some(SyncProcessedCloud {
                scan_id,
                received_at,
                cloud: cloud_downsampled,
                passthrough: cloud_output,
                imu_buffer: Vec::new(),
                imu_frame: "".to_string(),
                header: lidar_header,
                end_of_frame_guess,
                end_of_frame_time: if use_point_timestamps_for_frame {
                    secs_to_timestamp(lidar_end_time)
                } else {
                    secs_to_timestamp(lidar_begin_time)
                },
                pose_covariance: [0.0; 36],
                twist_covariance: [0.0; 36],
                end_of_frame_twist: Twist {
                    linear: Vector3::new(0., 0., 0.),
                    angular: Vector3::new(0., 0., 0.),
                },
                runtime_eval: self.eval.clone(),
                should_map_only: false,
                lidar_end_time_abs: abs_lidar_end,
            })
        } else {
            let meas = MeasureGroup {
                cloud: cloud_downsampled,
                cloud_passthrough: cloud_output,
                last_lidar_end_stamp: imu_span_start_time,
                lidar_start_stamp: lidar_begin_time,
                lidar_end_stamp: deskew_end_time,
                imu_buffer: imu_data.clone(),
                odom_buffer: odom_data.clone(),
            };

            let imu_frame = meas
                .imu_buffer
                .first()
                .expect("what did we just merge?")
                .frame_id
                .clone();

            // Deskewing cloud with synced data
            let deskew = self
                .imu_process
                .process(meas, &self.imu_pose_prediction_undistort);

            deskew.map(|imu_cloud| {
                self.eval.merge(imu_cloud.runtime_eval);
                let imu_int = match self.imu_pose_prediction_init_guess.as_ref() {
                    Some(imu_pose_prediction_init_guess) => imu_pose_prediction_init_guess(
                        &imu_data.clone(),
                        imu_span_start_time,
                        lidar_begin_time,
                        imu_span_end_time,
                    ),
                    None => {
                        let estimated_pose = if imu_intrinsic_calibration_given {
                            self.eval.start("preint imu").unwrap();
                            // given a calibration, we trust the IMU data and use it for the pose
                            naive_preint_eth::integrate_imu(
                                &imu_data,
                                self.prev_lidar_end_time,
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
                            self.eval.stop("preint imu").unwrap();
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

                SyncProcessedCloud {
                    scan_id,
                    received_at,
                    cloud: imu_cloud.cloud,
                    passthrough: imu_cloud.cloud_output,
                    imu_frame,
                    header: lidar_header,
                    end_of_frame_guess: imu_int.pose,
                    end_of_frame_time: if use_point_timestamps_for_frame {
                        imu_data.last().unwrap().stamp
                    } else {
                        secs_to_timestamp(lidar_begin_time)
                    },
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
                    runtime_eval: self.eval.clone(),
                    should_map_only: false,
                    lidar_end_time_abs: abs_lidar_end,
                    imu_buffer: imu_data,
                }
            })
        };

        self.prev_lidar_end_time = imu_span_end_time;
        synced
    }
}

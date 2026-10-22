//! The per-scan lidar-inertial odometry step shared by every Pelorus front end.
//!
//! [`LioCore::step`] takes one synchronized and deskewed scan, predicts its pose
//! from the IMU, registers it against the local map and returns the corrected
//! pose with its covariances. Every driver of the CLI runs this step, so they
//! produce the same trajectory for the same input.

use pelorus_core::Instant;
use std::sync::{Arc, Mutex};

use log::info;
use nalgebra::{Isometry3, Translation3, UnitQuaternion};
use pelorus_core::timing::RuntimeEvaluator;
use pelorus_core::{ImuMsg, LioConfig, PointXYZIT, SyncProcessedCloud, timestamp_to_secs};
use pelorus_sync::LocatedBufferView;
use ros_pointcloud2::ros::{HeaderMsg, TimeMsg};

use crate::backend::ActiveBackend;
use crate::pipeline::Pipeline;
use crate::registration::RegistrationState;
use crate::{CloudRegisterResult, ImuEkfPrediction, Registration, RegistrationScan};

/// A registered scan and everything the outputs of a front end need from it.
#[derive(Debug)]
pub struct LioScan {
    pub scan_id: u64,
    /// When the scan entered the synchronizer.
    pub received_at: Instant,
    /// Header of the incoming cloud.
    pub cloud_header: HeaderMsg,
    /// Stamp of the corrected pose: the absolute end of the scan.
    pub odom_time: TimeMsg,
    /// `odom_time` in seconds.
    pub current_cloud_time: f64,
    /// IMU messages of the scan interval.
    pub imu_buffer: Vec<ImuMsg>,
    /// The full-resolution deskewed cloud, when preprocessing kept one.
    pub passthrough: Option<Vec<PointXYZIT>>,
    pub register_pose: CloudRegisterResult,
    pub ekf_state: ImuEkfPrediction,
    /// Reported pose: the base link when one is configured, the Lidar otherwise.
    pub t_world: Isometry3<f64>,
    /// IMU-predicted Lidar pose the registration started from.
    pub init_guess_lidar: Isometry3<f64>,
    /// Row-major 6x6 pose covariance of `t_world`.
    pub pose_covariance: [f64; 36],
    /// Row-major 6x6 twist covariance of `t_world`.
    pub twist_covariance: [f64; 36],
}

/// Outcome of [`LioCore::step`].
#[derive(Debug)]
pub enum LioStep {
    Registered(Box<LioScan>),
    /// The scan only extended the map, for example during initialization.
    MapOnly {
        scan_id: u64,
    },
    Failed {
        scan_id: u64,
        error: anyhow::Error,
    },
}

/// Registration state, map and pose bookkeeping of one odometry run.
///
/// Construct it on the thread that runs the steps, because GPU backends bind
/// their streams there. The GPU backends keep one process-wide map, which the
/// first core initializes.
pub struct LioCore {
    config: LioConfig,
    pipeline: Pipeline,
    map: Arc<Mutex<RegistrationState>>,
    active_backend: ActiveBackend,
    #[cfg(feature = "webgpu")]
    webgpu: Option<pelorus_wgpu::Correspondence>,
    t_world: Isometry3<f64>,
    t_last_current: Isometry3<f64>,
    last_scan_time: Option<f64>,
    eval: RuntimeEvaluator,
    #[cfg(feature = "cuda")]
    persistent_cuda_cloud_bufs: Option<[LocatedBufferView<PointXYZIT>; 2]>,
    #[cfg(feature = "cuda")]
    persistent_cuda_cloud_buf_idx: usize,
    /// Reusable buffer for the point cloud on the GPU.
    persistent_cloud_buf: LocatedBufferView<PointXYZIT>,
    /// The deskewed cloud of the last step on the CPU. The CPU backend registers it.
    ram_cloud: LocatedBufferView<PointXYZIT>,
}

impl LioCore {
    pub fn new(config: LioConfig, pipeline: Pipeline) -> Self {
        let map = pipeline.map();
        let active_backend = ActiveBackend::detect(&config);
        // TODO avoid casting f32, maybe res needs f64?
        match &active_backend {
            ActiveBackend::Cpu => {}
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(_) if pelorus_cuda::get_map_size() > 0 => {}
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(stream) => pelorus_cuda::init_map_with_mode(
                stream,
                config.map_capacity,
                config.map_resolution as f32,
                config.dist_update_sensitivity,
                config.neighbor_max_range as f32,
                pelorus_cuda::NeighborType::Neighbor26,
                config.map_multiresolution,
            ),
            #[cfg(feature = "metal")]
            ActiveBackend::Metal(_) if pelorus_metal::get_map_size() > 0 => {}
            #[cfg(feature = "metal")]
            ActiveBackend::Metal(stream) => pelorus_metal::init_map_with_mode(
                stream,
                config.map_capacity,
                config.map_resolution as f32,
                config.dist_update_sensitivity,
                config.neighbor_max_range as f32,
                pelorus_metal::NeighborType::Neighbor26,
                config.map_multiresolution,
            ),
            #[cfg(feature = "vulkan")]
            ActiveBackend::Vulkan(_) if pelorus_vulkan::get_map_size() > 0 => {}
            #[cfg(feature = "vulkan")]
            ActiveBackend::Vulkan(stream) => pelorus_vulkan::init_map_with_mode(
                stream,
                config.map_capacity,
                config.map_resolution as f32,
                config.dist_update_sensitivity,
                config.neighbor_max_range as f32,
                pelorus_vulkan::NeighborType::Neighbor26,
                config.map_multiresolution,
            ),
        }

        #[cfg(feature = "cuda")]
        let persistent_cuda_cloud_bufs = if let ActiveBackend::Cuda(stream) = &active_backend {
            Some([
                LocatedBufferView::alloc_points_cuda(stream, 100_000),
                LocatedBufferView::alloc_points_cuda(stream, 100_000),
            ])
        } else {
            None
        };

        let persistent_cloud_buf = match &active_backend {
            ActiveBackend::Cpu => LocatedBufferView::Ram(Vec::new()),
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(_) => LocatedBufferView::Ram(Vec::new()),
            #[cfg(feature = "metal")]
            ActiveBackend::Metal(stream) => LocatedBufferView::Metal {
                buffer: pelorus_metal::MetalBuffer::new(stream, size_of::<PointXYZIT>(), 100_000),
            },
            #[cfg(feature = "vulkan")]
            ActiveBackend::Vulkan(stream) => LocatedBufferView::Vulkan {
                buffer: pelorus_vulkan::VulkanBuffer::new(stream, size_of::<PointXYZIT>(), 100_000),
            },
        };

        Self {
            config,
            pipeline,
            map,
            active_backend,
            #[cfg(feature = "webgpu")]
            webgpu: None,
            t_world: Isometry3::identity(),
            t_last_current: Isometry3::identity(),
            last_scan_time: None,
            eval: RuntimeEvaluator::new(),
            #[cfg(feature = "cuda")]
            persistent_cuda_cloud_bufs,
            #[cfg(feature = "cuda")]
            persistent_cuda_cloud_buf_idx: 0,
            persistent_cloud_buf,
            ram_cloud: LocatedBufferView::Ram(Vec::new()),
        }
    }

    /// The configuration, with the base-to-Lidar extrinsic the registration uses.
    pub fn config(&self) -> &LioConfig {
        &self.config
    }

    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    pub fn map(&self) -> &Arc<Mutex<RegistrationState>> {
        &self.map
    }

    pub fn active_backend(&self) -> &ActiveBackend {
        &self.active_backend
    }

    /// Timings of the scans so far, including synchronization and preprocessing.
    pub fn eval(&self) -> &RuntimeEvaluator {
        &self.eval
    }

    pub fn eval_mut(&mut self) -> &mut RuntimeEvaluator {
        &mut self.eval
    }

    /// The deskewed cloud of the last step, on the CPU.
    pub fn deskewed_cloud(&self) -> &[PointXYZIT] {
        match &self.ram_cloud {
            LocatedBufferView::Ram(points) => points,
            #[allow(unreachable_patterns)]
            _ => &[],
        }
    }

    /// The cloud the last step registered, on the backend that registered it.
    pub fn registered_cloud(&self) -> &LocatedBufferView<PointXYZIT> {
        match &self.active_backend {
            ActiveBackend::Cpu => &self.ram_cloud,
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(_) => {
                let cuda_cloud_bufs = self
                    .persistent_cuda_cloud_bufs
                    .as_ref()
                    .expect("missing persistent CUDA cloud buffers");
                let last_slot = (self.persistent_cuda_cloud_buf_idx + cuda_cloud_bufs.len() - 1)
                    % cuda_cloud_bufs.len();
                &cuda_cloud_bufs[last_slot]
            }
            #[allow(unreachable_patterns)]
            _ => &self.persistent_cloud_buf,
        }
    }

    /// Predict, register and map one synchronized scan.
    ///
    /// `ready_for_next` receives the scan id once the map is updated, which
    /// releases a threaded synchronizer to the next scan.
    pub fn step(
        &mut self,
        processed: SyncProcessedCloud,
        ready_for_next: Option<&std::sync::mpsc::Sender<u64>>,
    ) -> LioStep {
        pelorus_core::complete_synchronously(self.step_inner(processed, ready_for_next, false))
    }

    /// Async scan processing; enables browser GPU readbacks when configured.
    pub async fn step_async(
        &mut self,
        processed: SyncProcessedCloud,
        ready_for_next: Option<&std::sync::mpsc::Sender<u64>>,
    ) -> LioStep {
        self.step_inner(processed, ready_for_next, true).await
    }

    #[cfg(feature = "webgpu")]
    pub async fn enable_webgpu(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.config.cpu_only, "CPU-only mode is configured");
        anyhow::ensure!(
            matches!(self.active_backend, ActiveBackend::Cpu),
            "a native GPU backend is already active"
        );
        {
            let state = self.map.lock().unwrap();
            anyhow::ensure!(
                state.map.multiresolution_snapshot().is_some()
                    || state.map.single_resolution_snapshot().is_some(),
                "WebGPU requires a multiresolution or single-resolution Nearby26 map"
            );
        }
        self.webgpu = Some(pelorus_wgpu::Correspondence::new().await?);
        Ok(())
    }

    #[cfg(feature = "webgpu")]
    pub fn webgpu_status(&self) -> Result<&str, String> {
        let gpu = self
            .webgpu
            .as_ref()
            .ok_or_else(|| "WebGPU is not enabled".to_owned())?;
        match gpu.failure() {
            Some(error) => Err(error),
            None => Ok(gpu.adapter_name()),
        }
    }

    // A scan updates the map atomically for the predictor thread, as in the
    // synchronous path. The only await is the WebGPU readback, and no other task
    // contends for the map in the meantime.
    #[expect(clippy::await_holding_lock)]
    async fn step_inner(
        &mut self,
        processed: SyncProcessedCloud,
        ready_for_next: Option<&std::sync::mpsc::Sender<u64>>,
        _use_webgpu: bool,
    ) -> LioStep {
        let SyncProcessedCloud {
            scan_id,
            received_at,
            cloud,
            passthrough,
            imu_buffer,
            header: cloud_header,
            end_of_frame_guess,
            runtime_eval,
            should_map_only,
            lidar_end_time_abs: odom_time,
            ..
        } = processed;

        if self.config.base_to_lidar_tf_override {
            self.config.base_to_lidar = self.map.lock().unwrap().base_to_lidar;
        }
        self.eval.merge(&runtime_eval);

        match &self.active_backend {
            ActiveBackend::Cpu => {}
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(stream) => {
                let cuda_cloud_bufs = self
                    .persistent_cuda_cloud_bufs
                    .as_mut()
                    .expect("missing persistent CUDA cloud buffers");
                let current_slot = self.persistent_cuda_cloud_buf_idx;
                self.persistent_cuda_cloud_buf_idx =
                    (self.persistent_cuda_cloud_buf_idx + 1) % cuda_cloud_bufs.len();

                // Uploads and kernels are enqueued on the same stream, so ping-pong slots
                // can be reused without forcing a host sync here.
                let cuda_cloud = &mut cuda_cloud_bufs[current_slot];
                cuda_cloud.realloc_if_needed_cuda(stream, cloud.len());
                cuda_cloud.upload_from_slice_cuda(stream, cloud.as_slice());
            }
            #[cfg(feature = "metal")]
            ActiveBackend::Metal(stream) => {
                // Metal uses shared memory. Sync before overwriting to avoid races.
                stream.sync();
                self.persistent_cloud_buf
                    .realloc_if_needed_metal(stream, cloud.len());
                self.persistent_cloud_buf
                    .upload_from_slice_metal(stream, cloud.as_slice());
            }
            #[cfg(feature = "vulkan")]
            ActiveBackend::Vulkan(stream) => {
                stream.sync();
                self.persistent_cloud_buf
                    .realloc_if_needed_vulkan(stream, cloud.len());
                self.persistent_cloud_buf
                    .upload_from_slice_vulkan(stream, cloud.as_slice());
            }
        }
        self.ram_cloud = LocatedBufferView::Ram(cloud);

        let orig_time = cloud_header.stamp;
        let current_cloud_time = timestamp_to_secs(&odom_time);
        if let Some(existing_last_scan_time) = self.last_scan_time
            && current_cloud_time < existing_last_scan_time
        {
            self.reset_map();
        }

        self.last_scan_time = Some(timestamp_to_secs(&orig_time));

        let init_guess_lidar = self.init_guess_lidar(&end_of_frame_guess);

        let registration_result = {
            let mut map = self.map.lock().unwrap();
            let cloud_for_reg = match &self.active_backend {
                ActiveBackend::Cpu => &self.ram_cloud,
                #[cfg(feature = "cuda")]
                ActiveBackend::Cuda(_) => {
                    let cuda_cloud_bufs = self
                        .persistent_cuda_cloud_bufs
                        .as_ref()
                        .expect("missing persistent CUDA cloud buffers");
                    let last_slot = (self.persistent_cuda_cloud_buf_idx + cuda_cloud_bufs.len()
                        - 1)
                        % cuda_cloud_bufs.len();
                    &cuda_cloud_bufs[last_slot]
                }
                #[allow(unreachable_patterns)]
                _ => &self.persistent_cloud_buf,
            };
            match map
                .register_and_map_async(
                    RegistrationScan {
                        cloud: cloud_for_reg,
                        imu_transform: init_guess_lidar,
                        lidar_timestamp: current_cloud_time,
                        scan_id: scan_id as usize,
                        map_only: should_map_only,
                    },
                    ready_for_next,
                    &mut self.eval,
                    &self.active_backend,
                    #[cfg(feature = "webgpu")]
                    if _use_webgpu {
                        self.webgpu.as_mut().filter(|gpu| gpu.failure().is_none())
                    } else {
                        None
                    },
                )
                .await
            {
                Ok(Some(pose)) => Ok(Some((pose, map.get_state()))),
                Ok(None) => Ok(None),
                Err(err) => Err(err),
            }
        };

        let (register_pose, ekf_state) = match registration_result {
            Ok(Some(result)) => result,
            Ok(None) => return LioStep::MapOnly { scan_id },
            Err(error) => return LioStep::Failed { scan_id, error },
        };

        let old_t_world = self.t_world;
        self.t_world = if self.config.base_link_frame.is_some() {
            register_pose.base_pose
        } else {
            register_pose.pose
        };
        // transformation from the last pose to the current pose in the local frame
        self.t_last_current = old_t_world.inverse() * self.t_world;

        let (pose_covariance, twist_covariance) = {
            let guard = self.map.lock().unwrap();
            let lidar_to_child = if self.config.base_link_frame.is_some() {
                self.config.base_to_lidar.inverse()
            } else {
                Isometry3::identity()
            };
            (
                guard.get_output_pose_covariance(&lidar_to_child),
                guard.get_output_twist_covariance(&self.t_world),
            )
        };

        LioStep::Registered(Box::new(LioScan {
            scan_id,
            received_at,
            cloud_header,
            odom_time,
            current_cloud_time,
            imu_buffer,
            passthrough,
            register_pose,
            ekf_state,
            t_world: self.t_world,
            init_guess_lidar,
            pose_covariance,
            twist_covariance,
        }))
    }

    /// The Lidar pose to start the registration from.
    ///
    /// The synchronizer's prediction is the IMU pose T_W_I. Without a
    /// translation the last motion is reused, keeping the predicted rotation
    /// when there is one.
    fn init_guess_lidar(&self, end_of_frame_guess: &Isometry3<f64>) -> Isometry3<f64> {
        let position = end_of_frame_guess.translation.vector;
        let orientation = end_of_frame_guess.rotation.quaternion();
        let rotation = || {
            UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                orientation.w,
                orientation.i,
                orientation.j,
                orientation.k,
            ))
        };
        if position.x == 0.0 && position.y == 0.0 && position.z == 0.0 {
            // sync did not return a orientation (for example when not using IMU at all), use the last difference as guess
            if orientation.w == 1.0
                && orientation.i == 0.0
                && orientation.j == 0.0
                && orientation.k == 0.0
            {
                self.t_world * self.t_last_current
            } else {
                // there is an orientation calculated by sync for undeskewing the cloud, so we can use that
                self.t_world * Isometry3::from_parts(self.t_last_current.translation, rotation())
            }
        } else {
            let translation = Translation3::new(position.x, position.y, position.z);
            // init_guess from sync is IMU pose T_W_I. Convert to Lidar pose T_W_L.
            Isometry3::from_parts(translation, rotation()) * self.pipeline.ext()
        }
    }

    /// Scan time ran backwards: start over with an empty map.
    fn reset_map(&mut self) {
        let mut map = self.map.lock().unwrap();
        map.clear_localmap(&self.active_backend);

        info!("Resetting map");
        #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
        let config = &self.config;
        match &self.active_backend {
            ActiveBackend::Cpu => {}
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(stream) => pelorus_cuda::init_map_with_mode(
                stream,
                config.map_capacity,
                config.map_resolution as f32,
                config.dist_update_sensitivity,
                config.neighbor_max_range as f32,
                pelorus_cuda::NeighborType::Neighbor26,
                config.map_multiresolution,
            ),
            #[cfg(feature = "metal")]
            ActiveBackend::Metal(stream) => pelorus_metal::init_map_with_mode(
                stream,
                config.map_capacity,
                config.map_resolution as f32,
                config.dist_update_sensitivity,
                config.neighbor_max_range as f32,
                pelorus_metal::NeighborType::Neighbor26,
                config.map_multiresolution,
            ),
            #[cfg(feature = "vulkan")]
            ActiveBackend::Vulkan(stream) => pelorus_vulkan::init_map_with_mode(
                stream,
                config.map_capacity,
                config.map_resolution as f32,
                config.dist_update_sensitivity,
                config.neighbor_max_range as f32,
                pelorus_vulkan::NeighborType::Neighbor26,
                config.map_multiresolution,
            ),
        };
        drop(map);
        self.t_world = Isometry3::identity();
        self.t_last_current = Isometry3::identity();
    }

    /// Quiesce the backend and release this core's GPU buffers.
    ///
    /// The process-wide GPU map stays for other cores. The library front ends
    /// release it through the backend session count, and a process that runs a
    /// single core releases it with [`shutdown_process`](Self::shutdown_process).
    pub fn shutdown(self) {
        self.release();
    }

    /// [`shutdown`](Self::shutdown), then release the process-wide GPU map and
    /// backend state as well.
    pub fn shutdown_process(self) {
        let active_backend = self.release();
        match &active_backend {
            ActiveBackend::Cpu => {}
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(_) => pelorus_cuda::cleanup_map(),
            #[cfg(feature = "metal")]
            ActiveBackend::Metal(_) => pelorus_metal::cleanup_map(),
            #[cfg(feature = "vulkan")]
            // Releases the map *and* the downsample state, so the device and
            // instance are destroyed when `active_backend` drops rather than
            // being left alive for the NVIDIA driver's atexit handler to race
            // against its own worker threads.
            ActiveBackend::Vulkan(_) => pelorus_vulkan::shutdown(),
        }
    }

    fn release(mut self) -> ActiveBackend {
        // Ensure backend work is quiesced and map state is released before GPU
        // buffers/contexts in this thread are dropped.
        self.active_backend.stream_sync();
        // Release GPU buffers before the backend tears the device down. Every
        // child object has to be gone before the device is destroyed.
        self.pipeline.release_gpu_buffers();
        #[cfg(feature = "vulkan")]
        if matches!(self.active_backend, ActiveBackend::Vulkan(_)) {
            drop(core::mem::replace(
                &mut self.persistent_cloud_buf,
                LocatedBufferView::Ram(Vec::new()),
            ));
        }
        #[cfg(feature = "cuda")]
        drop(self.persistent_cuda_cloud_bufs.take());
        let Self { active_backend, .. } = self;
        active_backend
    }
}

use crate::ActiveBackend;
use crate::ImuEkfPrediction;
use crate::eqf::{Eqf, SO3, skew_symmetric_na};
use crate::imu::ImuProcessing;
use nalgebra::*;
#[cfg(not(feature = "rayon"))]
use pelorus_core::adaptive_execution::ExecutionMode;
#[cfg(feature = "rayon")]
use pelorus_core::adaptive_execution::{AdaptiveExecutionPlanner, ExecutionMode, HeavyPhase};
use pelorus_core::timing::RuntimeEvaluator;
use pelorus_core::{ImuParameters, KinematicModel, PointXYZIT};
use pelorus_localmap::*;
use pelorus_sync::LocatedBufferView;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use ros_pointcloud2::prelude::*;
#[cfg(feature = "rayon")]
use std::sync::Mutex;
use std::sync::{Arc, RwLock};

#[cfg(feature = "cuda")]
use pelorus_cuda::CudaBuffer;
#[cfg(feature = "metal")]
use pelorus_metal::MetalBuffer;
#[cfg(feature = "vulkan")]
use pelorus_vulkan::VulkanBuffer;

pub enum GpuPlanesBuffer {
    #[cfg(feature = "cuda")]
    Cuda(CudaBuffer),
    #[cfg(feature = "metal")]
    Metal(MetalBuffer),
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanBuffer),
}

/// Filter and execution settings of a [`RegistrationState`].
#[derive(Debug, Clone, Copy)]
pub struct RegistrationSettings {
    pub gravity: f64,
    pub laser_point_noise_cov: f64,
    pub ekf_max_iterations: usize,
    pub parallelism_decision_threshold_stddevs: f64,
    pub parallelism_decision_samples_per_mode: usize,
    pub parallelism_execution_policy: pelorus_core::adaptive_execution::ExecutionPolicy,
}

impl RegistrationSettings {
    /// Settings with the default adaptive execution policy.
    pub fn new(gravity: f64, laser_point_noise_cov: f64, ekf_max_iterations: usize) -> Self {
        Self {
            gravity,
            laser_point_noise_cov,
            ekf_max_iterations,
            parallelism_decision_threshold_stddevs: 2.0,
            parallelism_decision_samples_per_mode: 20,
            parallelism_execution_policy: pelorus_core::adaptive_execution::ExecutionPolicy::Auto,
        }
    }
}

pub struct RegistrationState {
    pub ekf: Eqf,
    pub imu_processing: ImuProcessing,
    pub map: IVoxHashMap,
    pub laser_point_noise_cov: f64,
    pub ekf_max_iterations: usize,
    pub base_to_lidar: Isometry3<f64>,
    pub planes_cpu_buffer: Vec<PointXYZINormal>,
    pub planes_gpu_buffer: Option<GpuPlanesBuffer>,
    #[cfg(feature = "rayon")]
    execution_planner: Arc<Mutex<AdaptiveExecutionPlanner>>,
    #[cfg(feature = "vulkan")]
    pub planes_vulkan_download_buffer: Vec<pelorus_vulkan::PointNormal>,
    #[cfg(feature = "vulkan")]
    pub planes_vulkan_counter_buffer: [u32; 3],
}

impl RegistrationState {
    pub fn set_lidar_to_imu(&mut self, transform: Isometry3<f64>) {
        assert!(
            !self.ekf.initialized,
            "Cannot change startup extrinsics after initialization"
        );
        self.imu_processing.set_lidar_to_imu(&transform);
        self.ekf.state.offset_t_l_i = transform.translation.vector;
        self.ekf.state.offset_r_l_i =
            SO3::try_from_mat(transform.rotation.to_rotation_matrix().matrix()).unwrap();
    }

    pub(crate) fn set_linear_velocity(&mut self, velocity: Vector3<f64>) {
        self.ekf.state.vel = velocity;
    }

    #[cfg(not(feature = "nn_kdt"))]
    pub fn get_distributions(&self) -> Vec<VoxelDistribution> {
        self.map.get_distributions()
    }

    #[cfg(not(feature = "nn_kdt"))]
    pub fn get_distributions_with_lru_scores(
        &self,
    ) -> Vec<pelorus_localmap::localmap::VoxelDistributionLru> {
        self.map.get_distributions_with_lru_scores()
    }

    #[cfg(not(feature = "nn_kdt"))]
    pub fn get_distributions_with_levels(
        &self,
    ) -> Vec<(pelorus_localmap::localmap::VoxelDistributionLru, u8)> {
        self.map.get_distributions_with_levels()
    }

    fn ensure_cloud_matches_backend(
        backend: &ActiveBackend,
        cloud: &LocatedBufferView<PointXYZIT>,
    ) -> anyhow::Result<()> {
        match backend {
            ActiveBackend::Cpu => {
                if !matches!(cloud, LocatedBufferView::Ram(_)) {
                    anyhow::bail!("CPU backend requires RAM buffer");
                }
            }
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(_) => {
                if !matches!(cloud, LocatedBufferView::Cuda { .. }) {
                    anyhow::bail!("CUDA backend requires CUDA buffer");
                }
            }
            #[cfg(feature = "metal")]
            ActiveBackend::Metal(_) => {
                if !matches!(cloud, LocatedBufferView::Metal { .. }) {
                    anyhow::bail!("Metal backend requires Metal buffer");
                }
            }
            #[cfg(feature = "vulkan")]
            ActiveBackend::Vulkan(_) => {
                if !matches!(cloud, LocatedBufferView::Vulkan { .. }) {
                    anyhow::bail!("Vulkan backend requires Vulkan buffer");
                }
            }
        }
        Ok(())
    }

    fn ensure_gpu_planes_buffer_matches_backend(
        backend: &ActiveBackend,
        #[allow(unused)] planes_gpu_buffer: &Option<GpuPlanesBuffer>,
    ) -> anyhow::Result<()> {
        match backend {
            ActiveBackend::Cpu => {}
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(_) => {
                if !matches!(planes_gpu_buffer, Some(GpuPlanesBuffer::Cuda(_))) {
                    anyhow::bail!("CUDA backend: GPU planes buffer not allocated");
                }
            }
            #[cfg(feature = "metal")]
            ActiveBackend::Metal(_) => {
                if !matches!(planes_gpu_buffer, Some(GpuPlanesBuffer::Metal(_))) {
                    anyhow::bail!("Metal backend: GPU planes buffer not allocated");
                }
            }
            #[cfg(feature = "vulkan")]
            ActiveBackend::Vulkan(_) => {
                if !matches!(planes_gpu_buffer, Some(GpuPlanesBuffer::Vulkan(_))) {
                    anyhow::bail!("Vulkan backend: GPU planes buffer not allocated");
                }
            }
        }
        Ok(())
    }

    pub fn new(
        map: IVoxHashMap,
        imu_params: &ImuParameters,
        lidar_to_imu: &Isometry3<f64>,
        base_to_lidar: &Isometry3<f64>,
        settings: RegistrationSettings,
        initial_guess: Option<ImuEkfPrediction>,
    ) -> Self {
        let RegistrationSettings {
            gravity,
            laser_point_noise_cov,
            ekf_max_iterations,
            parallelism_decision_threshold_stddevs,
            parallelism_decision_samples_per_mode,
            parallelism_execution_policy,
        } = settings;
        #[cfg(not(feature = "rayon"))]
        let _ = (
            parallelism_decision_threshold_stddevs,
            parallelism_decision_samples_per_mode,
            parallelism_execution_policy,
        );
        let mut ekf = Eqf::new(gravity, *base_to_lidar);
        // The configured extrinsic is the starting point either way. This only
        // decides whether the filter is allowed to move it afterwards.
        ekf.extrinsic_est = imu_params.estimate_extrinsic_online;
        if let Some(guess) = initial_guess {
            ekf.state.pos = guess.pose.translation.vector;
            ekf.state.rot =
                SO3::try_from_mat(guess.pose.rotation.to_rotation_matrix().matrix()).unwrap();
            ekf.state.vel = guess.linear_velocity;
            ekf.state.ba = guess.accel_bias;
            ekf.state.bg = guess.gyro_bias;
            ekf.state.offset_r_l_i = SO3::try_from_mat(
                guess
                    .lidar_imu_extrinsic
                    .rotation
                    .to_rotation_matrix()
                    .matrix(),
            )
            .unwrap();
            ekf.state.offset_t_l_i = guess.lidar_imu_extrinsic.translation.vector;
        }

        Self {
            ekf,
            imu_processing: ImuProcessing::new(imu_params, lidar_to_imu, base_to_lidar),
            map,
            laser_point_noise_cov,
            ekf_max_iterations,
            base_to_lidar: *base_to_lidar,
            planes_cpu_buffer: Vec::new(),
            planes_gpu_buffer: None,
            #[cfg(feature = "rayon")]
            execution_planner: Arc::new(Mutex::new(AdaptiveExecutionPlanner::new(
                parallelism_decision_threshold_stddevs,
                parallelism_decision_samples_per_mode,
                parallelism_execution_policy,
            ))),
            #[cfg(feature = "vulkan")]
            planes_vulkan_download_buffer: Vec::new(),
            #[cfg(feature = "vulkan")]
            planes_vulkan_counter_buffer: [0; 3],
        }
    }

    fn execution_mode(&self, phase: pelorus_core::adaptive_execution::HeavyPhase) -> ExecutionMode {
        #[cfg(feature = "rayon")]
        {
            self.execution_planner.lock().unwrap().mode_for(phase)
        }
        #[cfg(not(feature = "rayon"))]
        {
            let _ = phase;
            ExecutionMode::Sequential
        }
    }

    fn record_execution(
        &self,
        phase: pelorus_core::adaptive_execution::HeavyPhase,
        mode: ExecutionMode,
        normalized_duration: f64,
    ) {
        #[cfg(feature = "rayon")]
        if let Some(decision) =
            self.execution_planner
                .lock()
                .unwrap()
                .record(phase, mode, normalized_duration)
        {
            log::info!(
                "{}: using {} ({:.2}x faster)",
                phase.name(),
                decision.mode.name(),
                decision.speedup,
            );
        }
        #[cfg(not(feature = "rayon"))]
        let _ = (phase, mode, normalized_duration);
    }

    fn add_to_map(
        &mut self,
        cloud: &LocatedBufferView<PointXYZIT>,
        pose: Isometry3<f64>,
        scan_id: usize,
    ) -> anyhow::Result<()> {
        #[allow(irrefutable_let_patterns)]
        let LocatedBufferView::Ram(cloud) = cloud else {
            anyhow::bail!("CPU add_to_map called with non-RAM buffer");
        };

        let mode = self.execution_mode(pelorus_core::adaptive_execution::HeavyPhase::MapUpdate);
        if let Some(normalized_duration) = self
            .map
            .add_with_execution_mode(cloud, &pose, scan_id, mode)
        {
            self.record_execution(
                pelorus_core::adaptive_execution::HeavyPhase::MapUpdate,
                mode,
                normalized_duration,
            );
        }
        Ok(())
    }

    pub fn get_entire_map(&self) -> Vec<nalgebra::Point3<f32>> {
        self.map.get_entire_map()
    }

    pub fn clear_localmap(&mut self, backend: &ActiveBackend) {
        self.map.clear();
        match backend {
            ActiveBackend::Cpu => {}
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(_) => pelorus_cuda::cleanup_map(),
            #[cfg(feature = "metal")]
            ActiveBackend::Metal(_) => pelorus_metal::cleanup_map(),
            #[cfg(feature = "vulkan")]
            ActiveBackend::Vulkan(_) => pelorus_vulkan::cleanup_map(),
        }
    }

    pub fn get_odometry_covariances(&self) -> ([f64; 36], [f64; 9]) {
        let pose_cov = self.get_output_pose_covariance(&Isometry3::identity());
        let mut vel_cov = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                vel_cov[i * 3 + j] = self.ekf.cov[(i + 12, j + 12)];
            }
        }
        (pose_cov, vel_cov)
    }

    pub fn get_output_pose_covariance(&self, lidar_to_output: &Isometry3<f64>) -> [f64; 36] {
        output_pose_covariance(&self.ekf, lidar_to_output)
    }

    /// Twist covariance of the IMU frame itself.
    pub fn get_twist_covariance_6x6(&self) -> [f64; 36] {
        self.get_output_twist_covariance(&crate::Registration::get_state(self).pose)
    }

    pub fn get_output_twist_covariance(&self, output_pose: &Isometry3<f64>) -> [f64; 36] {
        output_twist_covariance(&self.ekf, output_pose, &self.imu_processing.ekf_cov_gyr)
    }
}

pub(crate) fn output_pose_covariance(ekf: &Eqf, lidar_to_output: &Isometry3<f64>) -> [f64; 36] {
    let r_wi = ekf.state.rot.matrix();
    let r_il = ekf.state.offset_r_l_i.matrix();
    let r_lo = lidar_to_output.rotation.to_rotation_matrix();
    let t_lo = lidar_to_output.translation.vector;
    let t_io = ekf.state.offset_t_l_i + r_il * t_lo;
    let r_io = r_il * r_lo.matrix();

    // Output pose in the world frame, needed as the moment arm of the global
    // rotation error: p_out <- dp + exp(dtheta) p_out.
    let p_out = ekf.state.pos + r_wi * t_io;
    let r_out = r_wi * r_io;

    let mut jacobian = SMatrix::<f64, 6, 24>::zeros();
    jacobian
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&Matrix3::identity());
    jacobian
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-skew_symmetric_na(&(p_out - ekf.anchor))));
    jacobian
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&r_out.transpose());

    if ekf.extrinsic_est {
        // The extrinsic error is global too, so it enters the output pose exactly
        // like the navigation error: R_out <- exp(dtheta_ext) R_out.
        jacobian
            .fixed_view_mut::<3, 3>(0, 6)
            .copy_from(&(-skew_symmetric_na(&(p_out - ekf.anchor))));
        jacobian
            .fixed_view_mut::<3, 3>(0, 9)
            .copy_from(&Matrix3::identity());
        jacobian
            .fixed_view_mut::<3, 3>(3, 6)
            .copy_from(&r_out.transpose());
    }
    // Fixed extrinsics are exact. Their covariance block is only a placeholder
    // that keeps the filter covariance nonsingular, so it does not enter the
    // output pose.

    let cov = SMatrix::<f64, 24, 24>::from_fn(|row, col| ekf.cov[(row, col)]);
    let output_cov = jacobian * cov * jacobian.transpose();
    matrix6_to_array(output_cov)
}

/// Covariance of the twist [`crate::ImuEkfPrediction::body_twist`] reports for
/// `output_pose`. The angular velocity is one gyro sample minus the estimated
/// bias, so it carries the gyro noise `gyro_noise` (IMU frame) and the bias
/// uncertainty. The linear velocity carries the same through its lever arm.
pub(crate) fn output_twist_covariance(
    ekf: &Eqf,
    output_pose: &Isometry3<f64>,
    gyro_noise: &Matrix3<f64>,
) -> [f64; 36] {
    let r_ow = *output_pose.rotation.inverse().to_rotation_matrix().matrix();
    let r_wi = ekf.state.rot.matrix();
    let lever_arm = output_pose.translation.vector - ekf.state.pos;
    let lever_arm_hat = skew_symmetric_na(&lever_arm);

    // World-frame twist of the output frame, v_out = v - hat(r) w and
    // w = R_WI (gyro - bg). The equivariant velocity error is
    // v <- dv + exp(dtheta) v, so it picks up the rotation block through
    // -hat(v). The bias error is d_bg = R_WI (bg - bg_hat), so w moves by -d_bg.
    let mut jacobian = SMatrix::<f64, 6, 24>::zeros();
    jacobian
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-skew_symmetric_na(&ekf.state.vel)));
    jacobian
        .fixed_view_mut::<3, 3>(0, 12)
        .copy_from(&Matrix3::identity());
    jacobian
        .fixed_view_mut::<3, 3>(0, 15)
        .copy_from(&lever_arm_hat);
    jacobian
        .fixed_view_mut::<3, 3>(3, 15)
        .copy_from(&(-Matrix3::identity()));

    // The gyro noise enters w directly and v through the lever arm.
    let mut noise_jacobian = SMatrix::<f64, 6, 3>::zeros();
    noise_jacobian
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(-lever_arm_hat));
    noise_jacobian
        .fixed_view_mut::<3, 3>(3, 0)
        .copy_from(&Matrix3::identity());
    let gyro_noise_world = r_wi * gyro_noise * r_wi.transpose();

    let cov = SMatrix::<f64, 24, 24>::from_fn(|row, col| ekf.cov[(row, col)]);
    let twist_cov_world = jacobian * cov * jacobian.transpose()
        + noise_jacobian * gyro_noise_world * noise_jacobian.transpose();

    let mut world_to_output = SMatrix::<f64, 6, 6>::zeros();
    world_to_output
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&r_ow);
    world_to_output
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&r_ow);
    matrix6_to_array(world_to_output * twist_cov_world * world_to_output.transpose())
}

fn matrix6_to_array(matrix: SMatrix<f64, 6, 6>) -> [f64; 36] {
    let mut out = [0.0; 36];
    for row in 0..6 {
        for col in 0..6 {
            out[row * 6 + col] = matrix[(row, col)];
        }
    }
    out
}

impl crate::Registration for RegistrationState {
    fn register_and_map(
        &mut self,
        scan: crate::RegistrationScan<'_>,
        ready_for_next: Option<&std::sync::mpsc::Sender<u64>>,
        eval: &mut RuntimeEvaluator,
        backend: &ActiveBackend,
    ) -> anyhow::Result<Option<crate::CloudRegisterResult>> {
        let crate::RegistrationScan {
            cloud,
            imu_transform,
            lidar_timestamp,
            scan_id,
            map_only,
        } = scan;
        let eval_cb = Arc::new(RwLock::new(&mut *eval));

        let map_empty = match backend {
            ActiveBackend::Cpu => self.map.is_empty(),
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(_) => pelorus_cuda::get_map_size() == 0,
            #[cfg(feature = "metal")]
            ActiveBackend::Metal(_) => pelorus_metal::get_map_size() == 0,
            #[cfg(feature = "vulkan")]
            ActiveBackend::Vulkan(_) => pelorus_vulkan::get_map_size() == 0,
        };

        if map_only {
            Self::ensure_cloud_matches_backend(backend, cloud)?;
            match backend {
                ActiveBackend::Cpu => self.add_to_map(cloud, imu_transform, scan_id)?,
                #[cfg(feature = "cuda")]
                ActiveBackend::Cuda(stream) => {
                    let LocatedBufferView::Cuda { buffer } = cloud else {
                        anyhow::bail!("CUDA backend requires CUDA buffer");
                    };
                    let transl = &imu_transform.translation.vector.into();
                    let rot_quat: pelorus_cuda::Quaternion = imu_transform.rotation.into();
                    pelorus_cuda::add_to_map(stream, buffer, transl, &rot_quat, scan_id);
                    // add_to_map is asynchronous. The next registration starts by
                    // reading the device-side map size on the host, so make the
                    // initial/map-only update visible before returning.
                    stream.sync();
                }
                #[cfg(feature = "metal")]
                ActiveBackend::Metal(stream) => {
                    let LocatedBufferView::Metal { buffer } = cloud else {
                        anyhow::bail!("Metal backend requires Metal buffer");
                    };
                    let transl = &imu_transform.translation.vector.into();
                    let rot_quat: pelorus_metal::Quaternion = imu_transform.rotation.into();
                    pelorus_metal::add_to_map(stream, buffer, transl, &rot_quat, scan_id);
                }
                #[cfg(feature = "vulkan")]
                ActiveBackend::Vulkan(stream) => {
                    let LocatedBufferView::Vulkan { buffer } = cloud else {
                        anyhow::bail!("Vulkan backend requires Vulkan buffer");
                    };
                    let transl = &imu_transform.translation.vector.into();
                    let rot_quat: pelorus_vulkan::Quaternion = imu_transform.rotation.into();
                    pelorus_vulkan::add_to_map(stream, buffer, transl, &rot_quat, scan_id);
                }
            }
            return Ok(None);
        }

        if !map_empty {
            #[allow(unreachable_patterns)]
            let extra: f32 = match backend {
                ActiveBackend::Cpu => 1.5,
                _ => 1.7,
            };
            let cloud_len_alloc = (extra * cloud.len() as f32).ceil() as usize;
            {
                eval_cb.write().unwrap().start("ekf_correct").unwrap();
            }

            // Allocate / grow buffers based on the active backend
            match backend {
                ActiveBackend::Cpu => {
                    if self.planes_cpu_buffer.capacity() < cloud.len() {
                        // `reserve` counts from `len`, not `capacity`. Passing the
                        // capacity-relative amount can leave the buffer with less than
                        // `cloud.len()` capacity, which the `set_len(cloud.len())` in the
                        // GPU branches below would then overrun.
                        let additional =
                            cloud_len_alloc.saturating_sub(self.planes_cpu_buffer.len());
                        self.planes_cpu_buffer.reserve(additional);
                    }
                }
                #[cfg(feature = "cuda")]
                ActiveBackend::Cuda(stream) => {
                    if self.planes_cpu_buffer.capacity() < cloud.len() {
                        // `reserve` counts from `len`, not `capacity`. Passing the
                        // capacity-relative amount can leave the buffer with less than
                        // `cloud.len()` capacity, which the `set_len(cloud.len())` in the
                        // GPU branches below would then overrun.
                        let additional =
                            cloud_len_alloc.saturating_sub(self.planes_cpu_buffer.len());
                        self.planes_cpu_buffer.reserve(additional);
                    }
                    let needs_alloc = match &self.planes_gpu_buffer {
                        Some(GpuPlanesBuffer::Cuda(buf)) => buf.n < cloud.len(),
                        _ => true,
                    };
                    if needs_alloc {
                        self.planes_gpu_buffer = Some(GpuPlanesBuffer::Cuda(CudaBuffer::new(
                            stream,
                            std::mem::size_of::<PointXYZINormal>(),
                            cloud_len_alloc,
                        )));
                    }
                }
                #[cfg(feature = "metal")]
                ActiveBackend::Metal(stream) => {
                    if self.planes_cpu_buffer.capacity() < cloud.len() {
                        // `reserve` counts from `len`, not `capacity`. Passing the
                        // capacity-relative amount can leave the buffer with less than
                        // `cloud.len()` capacity, which the `set_len(cloud.len())` in the
                        // GPU branches below would then overrun.
                        let additional =
                            cloud_len_alloc.saturating_sub(self.planes_cpu_buffer.len());
                        self.planes_cpu_buffer.reserve(additional);
                    }
                    let needs_alloc = match &self.planes_gpu_buffer {
                        Some(GpuPlanesBuffer::Metal(buf)) => buf.n < cloud.len(),
                        _ => true,
                    };
                    if needs_alloc {
                        self.planes_gpu_buffer = Some(GpuPlanesBuffer::Metal(MetalBuffer::new(
                            stream,
                            std::mem::size_of::<pelorus_metal::PointNormal>(),
                            cloud_len_alloc,
                        )));
                    }
                }
                #[cfg(feature = "vulkan")]
                ActiveBackend::Vulkan(stream) => {
                    if self.planes_cpu_buffer.capacity() < cloud.len() {
                        // `reserve` counts from `len`, not `capacity`. Passing the
                        // capacity-relative amount can leave the buffer with less than
                        // `cloud.len()` capacity, which the `set_len(cloud.len())` in the
                        // GPU branches below would then overrun.
                        let additional =
                            cloud_len_alloc.saturating_sub(self.planes_cpu_buffer.len());
                        self.planes_cpu_buffer.reserve(additional);
                    }
                    if self.planes_vulkan_download_buffer.capacity() < cloud.len() {
                        let additional = cloud_len_alloc
                            .saturating_sub(self.planes_vulkan_download_buffer.len());
                        self.planes_vulkan_download_buffer.reserve(additional);
                    }
                    if self.planes_vulkan_download_buffer.len() < cloud.len() {
                        self.planes_vulkan_download_buffer
                            .resize(cloud_len_alloc, pelorus_vulkan::PointNormal::default());
                    }
                    let needs_alloc = match &self.planes_gpu_buffer {
                        Some(GpuPlanesBuffer::Vulkan(buf)) => buf.capacity() < cloud.len(),
                        _ => true,
                    };
                    if needs_alloc {
                        self.planes_gpu_buffer =
                            Some(GpuPlanesBuffer::Vulkan(pelorus_vulkan::VulkanBuffer::new(
                                stream,
                                std::mem::size_of::<pelorus_vulkan::PointNormal>(),
                                cloud_len_alloc,
                            )));
                    }
                }
            }
            Self::ensure_cloud_matches_backend(backend, cloud)?;
            Self::ensure_gpu_planes_buffer_matches_backend(backend, &self.planes_gpu_buffer)?;

            let planes_cpu_buffer_ptr = std::ptr::addr_of_mut!(self.planes_cpu_buffer);
            let planes_gpu_buffer_ptr = std::ptr::addr_of_mut!(self.planes_gpu_buffer);
            #[cfg(feature = "vulkan")]
            let planes_vulkan_download_buffer_ptr =
                std::ptr::addr_of_mut!(self.planes_vulkan_download_buffer);
            #[cfg(feature = "vulkan")]
            let planes_vulkan_counter_buffer_ptr =
                std::ptr::addr_of_mut!(self.planes_vulkan_counter_buffer);
            let map_ptr = std::ptr::addr_of!(self.map);
            #[cfg(feature = "rayon")]
            let execution_planner = Arc::clone(&self.execution_planner);

            #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
            let _ = planes_gpu_buffer_ptr;

            let eval_cb_inner = eval_cb.clone();
            let extrinsic_est = self.ekf.extrinsic_est;
            // Re-anchor before capturing: `correct` would otherwise move the anchor
            // after this closure had already captured the old value, and the
            // measurement Jacobian would be built about a different origin than the
            // one the filter is using.
            let anchor_pos = self.ekf.state.pos;
            self.ekf.set_anchor(anchor_pos);
            let anchor = self.ekf.anchor;

            self.ekf.correct(
                self.laser_point_noise_cov,
                lidar_timestamp,
                self.ekf_max_iterations,
                eval_cb.clone(),
                move |ekf_state: &crate::eqf::StateIKFom,
                      valid: &mut bool,
                      _converged: bool,
                      h: &mut faer::Col<f64>,
                      h_x: &mut faer::Mat<f64>,
                      num_measurements: &mut usize| {
                    let mut eval_guard = eval_cb_inner.write().unwrap();
                    let eval_actual = &mut **eval_guard;
                    eval_actual.start("ekf_correct/model/planes").unwrap();

                    // Try to retrieve previous GPU timing measurements (non-blocking)
                    let gpu_timings = match backend {
                        ActiveBackend::Cpu => None,
                        #[cfg(feature = "cuda")]
                        ActiveBackend::Cuda(_) => pelorus_cuda::try_get_gpu_timings(),
                        #[cfg(feature = "metal")]
                        ActiveBackend::Metal(_) => pelorus_metal::try_get_gpu_timings(),
                        #[cfg(feature = "vulkan")]
                        ActiveBackend::Vulkan(_) => pelorus_vulkan::try_get_gpu_timings(),
                    };
                    if let Some((planes_ms, add_to_map_ms)) = gpu_timings {
                        if planes_ms > 0.0 {
                            eval_actual.add_external_measurement(
                                "ekf_correct/model/planes/gpu",
                                planes_ms,
                            );
                        }
                        if add_to_map_ms > 0.0 {
                            eval_actual.add_external_measurement("map_update/gpu", add_to_map_ms);
                        }
                    }

                    // Recompute planes on every EqF iteration - correspondence must be
                    // updated as the state estimate changes.
                    let planes: &mut Vec<PointXYZINormal> = match backend {
                        ActiveBackend::Cpu => {
                            #[allow(irrefutable_let_patterns)]
                            let LocatedBufferView::Ram(cloud_pts) = cloud else {
                                *valid = false;
                                *num_measurements = 0;
                                return;
                            };
                            let planes_buf = unsafe { &mut *planes_cpu_buffer_ptr };
                            planes_buf.clear();
                            let map = unsafe { &*map_ptr };
                            #[cfg(feature = "rayon")]
                            let mode = execution_planner
                                .lock()
                                .unwrap()
                                .mode_for(HeavyPhase::PlaneCorrespondence);
                            #[cfg(feature = "rayon")]
                            let started = std::time::Instant::now();

                            // Match the GPU backends' f32 transform precision so CPU↔GPU
                            // plane selection stays numerically equivalent. GPU shaders do
                            // the lidar→imu→world transform in f32, via UnitQuaternion cast
                            // from the f64 pose rotation matrix. Doing the same here keeps
                            // the query point, pd2 and score bit-close to the GPU path.
                            let offset_q_f32: nalgebra::UnitQuaternion<f32> =
                                nalgebra::UnitQuaternion::from_matrix(
                                    &ekf_state.offset_r_l_i.matrix().clone(),
                                )
                                .cast();
                            let offset_t_f32: nalgebra::Vector3<f32> =
                                ekf_state.offset_t_l_i.cast();
                            let pose_q_f32: nalgebra::UnitQuaternion<f32> =
                                nalgebra::UnitQuaternion::from_matrix(
                                    &ekf_state.pose().rotation().matrix().clone(),
                                )
                                .cast();
                            let pose_t_f32: nalgebra::Vector3<f32> = ekf_state
                                .pose()
                                .matrix()
                                .column(3)
                                .fixed_rows::<3>(0)
                                .into_owned()
                                .cast();

                            let point_to_plane = |point_local: &PointXYZIT| {
                                let p_l = nalgebra::Point3::new(
                                    point_local.x,
                                    point_local.y,
                                    point_local.z,
                                );
                                let p_imu = offset_q_f32 * p_l + offset_t_f32;
                                let p_world = pose_q_f32 * p_imu + pose_t_f32;
                                let query_point = PointXYZ::new(p_world.x, p_world.y, p_world.z);
                                let plane = map.plane_fit_from_distributions(&query_point)?;
                                let pd2 = plane[0] * p_world.x
                                    + plane[1] * p_world.y
                                    + plane[2] * p_world.z
                                    + plane[3];
                                let l2_sq = point_local.x * point_local.x
                                    + point_local.y * point_local.y
                                    + point_local.z * point_local.z;
                                let s = 1.0_f32
                                    - 0.9_f32 * pd2.abs() / l2_sq.max(1e-9_f32).sqrt().sqrt();
                                if s <= 0.9_f32 {
                                    None
                                } else {
                                    Some(PointXYZINormal::new(
                                        point_local.x,
                                        point_local.y,
                                        point_local.z,
                                        plane[3],
                                        plane[0],
                                        plane[1],
                                        plane[2],
                                    ))
                                }
                            };
                            #[cfg(feature = "rayon")]
                            match mode {
                                ExecutionMode::Sequential => {
                                    planes_buf.extend(cloud_pts.iter().filter_map(point_to_plane));
                                }
                                ExecutionMode::Rayon => {
                                    planes_buf.par_extend(
                                        cloud_pts.par_iter().filter_map(point_to_plane),
                                    );
                                }
                            }
                            #[cfg(not(feature = "rayon"))]
                            planes_buf.extend(cloud_pts.iter().filter_map(point_to_plane));
                            #[cfg(feature = "rayon")]
                            if let Some(decision) = execution_planner.lock().unwrap().record(
                                HeavyPhase::PlaneCorrespondence,
                                mode,
                                started.elapsed().as_nanos() as f64 / cloud_pts.len().max(1) as f64,
                            ) {
                                log::info!(
                                    "{}: using {} ({:.2}x faster)",
                                    HeavyPhase::PlaneCorrespondence.name(),
                                    decision.mode.name(),
                                    decision.speedup,
                                );
                            }
                            planes_buf
                        }
                        #[cfg(feature = "cuda")]
                        ActiveBackend::Cuda(stream) => {
                            let planes_buf = unsafe { &mut *planes_cpu_buffer_ptr };
                            planes_buf.clear();
                            unsafe { planes_buf.set_len(cloud.len()) };
                            let planes_gpu = match unsafe { &mut *planes_gpu_buffer_ptr } {
                                Some(GpuPlanesBuffer::Cuda(buf)) => buf,
                                _ => {
                                    *valid = false;
                                    *num_measurements = 0;
                                    return;
                                }
                            };
                            let LocatedBufferView::Cuda { buffer } = cloud else {
                                *valid = false;
                                *num_measurements = 0;
                                return;
                            };
                            let pose_q: pelorus_cuda::Quaternion = UnitQuaternion::from_matrix(
                                &ekf_state.pose().rotation().matrix().clone(),
                            )
                            .into();
                            let offset_q: pelorus_cuda::Quaternion = UnitQuaternion::from_matrix(
                                &ekf_state.offset_r_l_i.matrix().clone(),
                            )
                            .into();
                            let pose_t = ekf_state
                                .pose()
                                .matrix()
                                .column(3)
                                .fixed_rows::<3>(0)
                                .into_owned();
                            let timings = pelorus_cuda::planes_cuda(
                                stream,
                                buffer,
                                &pelorus_cuda::Pose {
                                    position: pelorus_cuda::Vector3::new(
                                        pose_t.x, pose_t.y, pose_t.z,
                                    ),
                                    orientation: pose_q,
                                },
                                &pelorus_cuda::Pose {
                                    position: pelorus_cuda::Vector3::new(
                                        ekf_state.offset_t_l_i.x,
                                        ekf_state.offset_t_l_i.y,
                                        ekf_state.offset_t_l_i.z,
                                    ),
                                    orientation: offset_q,
                                },
                                planes_buf,
                                planes_gpu,
                            );
                            eval_actual.add_external_measurement(
                                "ekf_correct/model/planes/kernel_compact",
                                timings.kernel_compact_ms,
                            );
                            eval_actual.add_external_measurement(
                                "ekf_correct/model/planes/download",
                                timings.download_ms,
                            );
                            planes_buf
                        }
                        #[cfg(feature = "metal")]
                        ActiveBackend::Metal(stream) => {
                            let planes_buf = unsafe { &mut *planes_cpu_buffer_ptr };
                            planes_buf.clear();
                            unsafe { planes_buf.set_len(cloud.len()) };
                            let planes_gpu = match unsafe { &mut *planes_gpu_buffer_ptr } {
                                Some(GpuPlanesBuffer::Metal(buf)) => buf,
                                _ => {
                                    *valid = false;
                                    *num_measurements = 0;
                                    return;
                                }
                            };
                            let LocatedBufferView::Metal { buffer } = cloud else {
                                *valid = false;
                                *num_measurements = 0;
                                return;
                            };
                            let pose_q: pelorus_metal::Quaternion = {
                                let mat: Matrix3<f64> =
                                    ekf_state.pose().rotation().matrix().clone();
                                UnitQuaternion::from_matrix(&mat).into()
                            };
                            let offset_q: pelorus_metal::Quaternion = {
                                let mat: Matrix3<f64> = ekf_state.offset_r_l_i.matrix().clone();
                                UnitQuaternion::from_matrix(&mat).into()
                            };
                            let pose_t = ekf_state
                                .pose()
                                .matrix()
                                .column(3)
                                .fixed_rows::<3>(0)
                                .into_owned();
                            pelorus_metal::planes_metal(
                                stream,
                                buffer,
                                &pelorus_metal::Vector3::new(pose_t.x, pose_t.y, pose_t.z),
                                &pose_q,
                                &pelorus_metal::Vector3::new(
                                    ekf_state.offset_t_l_i.x,
                                    ekf_state.offset_t_l_i.y,
                                    ekf_state.offset_t_l_i.z,
                                ),
                                &offset_q,
                                planes_buf,
                                planes_gpu,
                            );
                            planes_buf
                        }
                        #[cfg(feature = "vulkan")]
                        ActiveBackend::Vulkan(stream) => {
                            let planes_buf = unsafe { &mut *planes_cpu_buffer_ptr };
                            planes_buf.clear();
                            unsafe { planes_buf.set_len(cloud.len()) };
                            let planes_gpu = match unsafe { &mut *planes_gpu_buffer_ptr } {
                                Some(GpuPlanesBuffer::Vulkan(buf)) => buf,
                                _ => {
                                    *valid = false;
                                    *num_measurements = 0;
                                    return;
                                }
                            };
                            let LocatedBufferView::Vulkan { buffer } = cloud else {
                                *valid = false;
                                *num_measurements = 0;
                                return;
                            };
                            let pose_q: pelorus_vulkan::Quaternion = UnitQuaternion::from_matrix(
                                &ekf_state.pose().rotation().matrix().clone(),
                            )
                            .into();
                            let offset_q: pelorus_vulkan::Quaternion = UnitQuaternion::from_matrix(
                                &ekf_state.offset_r_l_i.matrix().clone(),
                            )
                            .into();
                            let pose_t = ekf_state
                                .pose()
                                .matrix()
                                .column(3)
                                .fixed_rows::<3>(0)
                                .into_owned();
                            let timings = pelorus_vulkan::planes_vulkan(
                                stream,
                                buffer,
                                &pelorus_vulkan::Pose {
                                    position: pelorus_vulkan::Vector3::new(
                                        pose_t.x, pose_t.y, pose_t.z,
                                    ),
                                    orientation: pose_q,
                                },
                                &pelorus_vulkan::Pose {
                                    position: pelorus_vulkan::Vector3::new(
                                        ekf_state.offset_t_l_i.x,
                                        ekf_state.offset_t_l_i.y,
                                        ekf_state.offset_t_l_i.z,
                                    ),
                                    orientation: offset_q,
                                },
                                planes_buf,
                                planes_gpu,
                                pelorus_vulkan::VulkanPlaneReadback {
                                    points: unsafe { &mut *planes_vulkan_download_buffer_ptr },
                                    counter: unsafe { &mut *planes_vulkan_counter_buffer_ptr },
                                },
                            );
                            eval_actual.add_external_measurement(
                                "ekf_correct/model/planes/kernel_compact",
                                timings.kernel_compact_ms,
                            );
                            eval_actual.add_external_measurement(
                                "ekf_correct/model/planes/download",
                                timings.download_ms,
                            );
                            planes_buf
                        }
                    };

                    eval_actual.stop("ekf_correct/model/planes").unwrap();

                    if planes.is_empty() {
                        *valid = false;
                        *num_measurements = 0;
                        return;
                    }

                    *num_measurements = planes.len();

                    eval_actual.start("ekf_correct/model/jacobian").unwrap();

                    if planes.len() > h.nrows() {
                        *h = faer::Col::<f64>::zeros(cloud_len_alloc);
                    }
                    if planes.len() > h_x.nrows() {
                        *h_x = faer::Mat::<f64>::zeros(cloud_len_alloc, 12);
                    }

                    for (i, normal_point) in planes.iter().enumerate() {
                        let point_local = Vector3::new(
                            normal_point.x as f64,
                            normal_point.y as f64,
                            normal_point.z as f64,
                        );
                        let norm = Vector3::new(
                            normal_point.normal_x as f64,
                            normal_point.normal_y as f64,
                            normal_point.normal_z as f64,
                        );
                        let plane_d = normal_point.intensity as f64;

                        let p_imu =
                            ekf_state.offset_r_l_i.matrix() * point_local + ekf_state.offset_t_l_i;

                        let p_world = ekf_state.pose().transform(p_imu);
                        let pd2 = norm.dot(&p_world) + plane_d;

                        // Anchor the global rotation error near the robot to keep
                        // the moment arm bounded by sensor range.
                        let a_vec = skew_symmetric_na(&(p_world - anchor)) * norm;

                        h_x[(i, 0)] = norm.x;
                        h_x[(i, 1)] = norm.y;
                        h_x[(i, 2)] = norm.z;

                        h_x[(i, 3)] = a_vec.x;
                        h_x[(i, 4)] = a_vec.y;
                        h_x[(i, 5)] = a_vec.z;

                        if extrinsic_est {
                            h_x[(i, 6)] = a_vec.x;
                            h_x[(i, 7)] = a_vec.y;
                            h_x[(i, 8)] = a_vec.z;

                            h_x[(i, 9)] = norm.x;
                            h_x[(i, 10)] = norm.y;
                            h_x[(i, 11)] = norm.z;
                        } else {
                            for col in 6..12 {
                                h_x[(i, col)] = 0.0;
                            }
                        }

                        h[i] = -pd2;
                    }
                    eval_actual.stop("ekf_correct/model/jacobian").unwrap();
                },
            );
            eval_cb.write().unwrap().stop("ekf_correct").unwrap();
        };

        let t_i_l = Isometry3::from_parts(
            Translation3::from(self.ekf.state.offset_t_l_i),
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                &self.ekf.state.offset_r_l_i.matrix(),
            )),
        );

        let t_w_i = Isometry3::from_parts(
            Translation3::from(self.ekf.state.pos),
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                &self.ekf.state.rot.matrix(),
            )),
        );

        let t_w_l = t_w_i * t_i_l;
        let t_w_b = t_w_l * self.base_to_lidar.inverse();

        {
            let mut eval_guard = eval_cb.write().unwrap();
            let eval_actual = &mut **eval_guard;
            eval_actual.start("map_update").unwrap();
        }

        match backend {
            ActiveBackend::Cpu => self.add_to_map(cloud, t_w_l, scan_id)?,
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(stream) => {
                let LocatedBufferView::Cuda { buffer } = cloud else {
                    anyhow::bail!("CUDA backend requires CUDA buffer");
                };
                let rot_quat: pelorus_cuda::Quaternion = t_w_l.rotation.into();
                pelorus_cuda::add_to_map(
                    stream,
                    buffer,
                    &t_w_l.translation.vector.into(),
                    &rot_quat,
                    scan_id,
                );
                stream.sync();
            }
            #[cfg(feature = "metal")]
            ActiveBackend::Metal(stream) => {
                let LocatedBufferView::Metal { buffer } = cloud else {
                    anyhow::bail!("Metal backend requires Metal buffer");
                };
                let transl = &t_w_l.translation.vector.into();
                let rot_quat: pelorus_metal::Quaternion = {
                    let mat: Matrix3<f64> = t_w_l.rotation.to_rotation_matrix().matrix().clone();
                    UnitQuaternion::from_matrix(&mat).into()
                };
                // No `stream.sync()` here, matching the Vulkan arm. The map
                // build enqueues its kernels and returns; nothing on the CPU
                // reads their results, and the next `planes_metal` is enqueued
                // on the same queue, which Metal executes in submission order.
                // Waiting here instead put the whole GPU map rebuild on the
                // critical path — it was ~2.9 ms of the 4.2 ms `map_update`
                // stage, the most expensive stage in the pipeline.
                pelorus_metal::add_to_map(stream, buffer, transl, &rot_quat, scan_id);
            }
            #[cfg(feature = "vulkan")]
            ActiveBackend::Vulkan(stream) => {
                let LocatedBufferView::Vulkan { buffer } = cloud else {
                    anyhow::bail!("Vulkan backend requires Vulkan buffer");
                };
                let transl = &t_w_l.translation.vector.into();
                let rot_quat: pelorus_vulkan::Quaternion = {
                    let mat: Matrix3<f64> = *t_w_l.rotation.to_rotation_matrix().matrix();
                    UnitQuaternion::from_matrix(&mat).into()
                };
                pelorus_vulkan::add_to_map(stream, buffer, transl, &rot_quat, scan_id);
            }
        }
        {
            let mut eval_guard = eval_cb.write().unwrap();
            let eval_actual = &mut **eval_guard;
            eval_actual.stop("map_update").unwrap();
        }

        if let Some(ready) = ready_for_next {
            let _ = ready.send(scan_id as u64);
        }

        Ok(Some(crate::CloudRegisterResult {
            pose: t_w_l,
            base_pose: t_w_b,
            avg_iterations: None,
        }))
    }

    fn requested_register_and_map_cloud_type(&self) -> LocatedBufferView<PointXYZ> {
        LocatedBufferView::Ram(Vec::new())
    }

    fn get_state(&self) -> crate::ImuEkfPrediction {
        crate::ImuEkfPrediction {
            pose: Isometry3::from_parts(
                Translation3::from(self.ekf.state.pos),
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                    &self.ekf.state.rot.matrix(),
                )),
            ),
            linear_velocity: self.ekf.state.vel,
            angular_velocity: self.imu_processing.angvel_last,
            lidar_imu_extrinsic: Isometry3::from_parts(
                Translation3::from(self.ekf.state.offset_t_l_i),
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                    &self.ekf.state.offset_r_l_i.matrix(),
                )),
            ),
            accel_bias: self.ekf.state.ba,
            gyro_bias: self.ekf.state.bg,
        }
    }

    fn undistort(
        &mut self,
        scan: crate::UndistortScan<'_>,
        kinematic: KinematicModel,
        constraint_std: f64,
    ) {
        let mode = self.execution_mode(pelorus_core::adaptive_execution::HeavyPhase::PointDeskew);
        let measurement =
            self.imu_processing
                .undistort(&mut self.ekf, scan, kinematic, constraint_std, mode);
        if let Some(normalized_duration) = measurement {
            self.record_execution(
                pelorus_core::adaptive_execution::HeavyPhase::PointDeskew,
                mode,
                normalized_duration,
            );
        }
    }

    fn imu_ekf_predict(
        &mut self,
        acc: [f64; 3],
        gyro: [f64; 3],
        dt: f64,
    ) -> crate::ImuEkfPrediction {
        self.imu_processing
            .predict(&mut self.ekf, acc, gyro, dt)
            .unwrap()
    }

    fn init_imu_ekf(&mut self, message_buffer: &[crate::ImuMsg]) -> Vector3<f64> {
        self.imu_processing
            .initialize(message_buffer, &mut self.ekf)
    }

    fn set_base_to_lidar(&mut self, base_to_lidar: Isometry3<f64>) {
        self.base_to_lidar = base_to_lidar;
        self.ekf.base_to_lidar = base_to_lidar;
        self.imu_processing.set_base_to_lidar(&base_to_lidar);
    }
}

#[cfg(test)]
mod covariance_tests {
    use super::*;

    #[test]
    fn enabled_extrinsic_pose_covariance_uses_the_eqf_anchor() {
        let mut ekf = Eqf::new(9.81, Isometry3::identity());
        ekf.extrinsic_est = true;
        ekf.anchor = Vector3::new(90.0, -40.0, 7.0);
        ekf.state.pos = Vector3::new(100.0, -35.0, 8.0);
        ekf.state.offset_t_l_i = Vector3::new(0.2, -0.1, 0.05);
        ekf.cov = faer::Mat::<f64>::zeros(24, 24);
        ekf.cov[(6, 6)] = 1.0;

        let covariance = output_pose_covariance(&ekf, &Isometry3::identity());
        let p_out = ekf.state.pos + ekf.state.rot.matrix() * ekf.state.offset_t_l_i;
        let jacobian_column = -skew_symmetric_na(&(p_out - ekf.anchor)).column(0);
        let expected = jacobian_column * jacobian_column.transpose();

        for row in 0..3 {
            for col in 0..3 {
                assert!(
                    (covariance[row * 6 + col] - expected[(row, col)]).abs() < 1e-12,
                    "anchored extrinsic covariance mismatch at ({row}, {col})"
                );
            }
        }
    }

    #[test]
    fn fixed_extrinsic_placeholder_does_not_enter_the_pose_covariance() {
        let mut ekf = Eqf::new(9.81, Isometry3::identity());
        ekf.extrinsic_est = false;
        ekf.state.offset_t_l_i = Vector3::new(0.2, -0.1, 0.05);
        ekf.cov = faer::Mat::<f64>::zeros(24, 24);
        for index in 6..12 {
            ekf.cov[(index, index)] = 1.0;
        }
        let lidar_to_output = Isometry3::from_parts(
            Translation3::new(-0.38, 0.0, -0.29),
            UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
        );

        let covariance = output_pose_covariance(&ekf, &lidar_to_output);

        assert!(covariance.iter().all(|value| value.abs() < 1e-12));
    }

    #[test]
    fn twist_covariance_carries_gyro_noise_and_bias_into_a_yawed_lever_arm_frame() {
        let mut ekf = Eqf::new(9.81, Isometry3::identity());
        ekf.cov = faer::Mat::<f64>::zeros(24, 24);
        for index in 15..18 {
            ekf.cov[(index, index)] = 0.5;
        }
        let gyro_noise = Matrix3::from_diagonal(&Vector3::new(0.25, 0.25, 1.0));
        // Output 1 m ahead of the IMU along world x, yawed 90 degrees.
        let output_pose = Isometry3::from_parts(
            Translation3::new(1.0, 0.0, 0.0),
            UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
        );

        let covariance = output_twist_covariance(&ekf, &output_pose, &gyro_noise);
        let at = |row: usize, col: usize| covariance[row * 6 + col];

        // World angular variances are (0.75, 0.75, 1.5). Yawing swaps x and y.
        assert!((at(3, 3) - 0.75).abs() < 1e-12);
        assert!((at(4, 4) - 0.75).abs() < 1e-12);
        assert!((at(5, 5) - 1.5).abs() < 1e-12);
        // v_out = -hat(r) w with r = x: only w_y and w_z move it, along world
        // y and z. The yawed frame's x axis is world y.
        assert!((at(0, 0) - 1.5).abs() < 1e-12);
        assert!(at(1, 1).abs() < 1e-12);
        assert!((at(2, 2) - 0.75).abs() < 1e-12);
        for row in 0..6 {
            for col in 0..6 {
                assert!((at(row, col) - at(col, row)).abs() < 1e-12);
            }
        }
    }
}

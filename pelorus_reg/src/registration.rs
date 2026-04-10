use crate::ActiveBackend;
use crate::iekf::{Iekf, skew_symmetric_na, SO3};
use crate::imu::ImuProcessing;
use crate::ImuEkfPrediction;
use nalgebra::*;
use pelorus_core::timing::RuntimeEvaluator;
use pelorus_core::{KinematicModel, PointXYZIT, LidarVendor, ImuParameters, ImuMsg, OdomMsg};
use prelio::LocatedBufferView;
use rayon::prelude::*;
use ros_pointcloud2::prelude::*;
use spole::*;
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

pub struct RegistrationState {
    pub ekf: Iekf,
    pub imu_processing: ImuProcessing,
    pub map: IVoxHashMap,
    pub laser_point_noise_cov: f64,
    pub ekf_max_iterations: usize,
    pub base_to_lidar: Isometry3<f64>,
    pub planes_cpu_buffer: Vec<PointXYZINormal>,
    pub planes_gpu_buffer: Option<GpuPlanesBuffer>,
}

impl RegistrationState {
    pub fn new(
        map: IVoxHashMap,
        imu_params: &ImuParameters,
        lidar_to_imu: &Isometry3<f64>,
        base_to_lidar: &Isometry3<f64>,
        gravity: f64,
        laser_point_noise_cov: f64,
        ekf_max_iterations: usize,
        initial_guess: Option<ImuEkfPrediction>,
    ) -> Self {
        let mut ekf = Iekf::new(gravity, base_to_lidar.clone());
        if let Some(guess) = initial_guess {
            ekf.state.pos = guess.pose.translation.vector;
            ekf.state.rot = SO3::try_from_mat(guess.pose.rotation.to_rotation_matrix().matrix()).unwrap();
            ekf.state.vel = guess.linear_velocity;
            ekf.state.ba = guess.accel_bias;
            ekf.state.bg = guess.gyro_bias;
            ekf.state.offset_r_l_i = SO3::try_from_mat(guess.lidar_imu_extrinsic.rotation.to_rotation_matrix().matrix()).unwrap();
            ekf.state.offset_t_l_i = guess.lidar_imu_extrinsic.translation.vector;
        }

        Self {
            ekf,
            imu_processing: ImuProcessing::new(
                imu_params,
                lidar_to_imu,
                base_to_lidar,
            ),
            map,
            laser_point_noise_cov,
            ekf_max_iterations,
            base_to_lidar: base_to_lidar.clone(),
            planes_cpu_buffer: Vec::new(),
            planes_gpu_buffer: None,
        }
    }

    fn add_to_map(
        &mut self,
        cloud: &LocatedBufferView<PointXYZIT>,
        pose: Isometry3<f64>,
        scan_id: usize,
    ) {
        #[allow(irrefutable_let_patterns)]
        let LocatedBufferView::Ram(cloud) = cloud else {
            panic!("CPU add_to_map called with non-RAM buffer");
        };

        self.map.add(cloud, &pose, scan_id);
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
        let mut pose_cov = [0.0; 36];
        for i in 0..3 {
            for j in 0..3 {
                pose_cov[i * 6 + j] = self.ekf.cov[(i, j)];
                pose_cov[(i + 3) * 6 + (j + 3)] = self.ekf.cov[(i + 3, j + 3)];
            }
        }
        let mut vel_cov = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                vel_cov[i * 3 + j] = self.ekf.cov[(i + 12, j + 12)];
            }
        }
        (pose_cov, vel_cov)
    }

    pub fn get_twist_covariance_6x6(&self) -> [f64; 36] {
        let mut twist_cov = [0.0; 36];
        for i in 0..3 {
            for j in 0..3 {
                twist_cov[i * 6 + j] = self.ekf.cov[(i + 12, j + 12)];
                twist_cov[(i + 3) * 6 + (j + 3)] = 0.0001; // TODO
            }
        }
        twist_cov
    }
}

impl crate::Registration for RegistrationState {
    fn register_and_map(
        &mut self,
        map_only: bool,
        imu_transform: Isometry3<f64>,
        cloud: &LocatedBufferView<PointXYZIT>,
        lidar_timestamp: f64,
        scan_id: usize,
        ready_for_next: Option<&std::sync::mpsc::Sender<()>>,
        eval: &mut RuntimeEvaluator,
        backend: &ActiveBackend,
    ) -> anyhow::Result<Option<crate::CloudRegisterResult>> {
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
            match backend {
                ActiveBackend::Cpu => self.add_to_map(cloud, imu_transform, scan_id),
                #[cfg(feature = "cuda")]
                ActiveBackend::Cuda(stream) => {
                    let LocatedBufferView::Cuda { buffer } = cloud else {
                        panic!("CUDA backend requires CUDA buffer");
                    };
                    let transl = &imu_transform.translation.vector.into();
                    let rot_quat: pelorus_cuda::Quaternion = imu_transform.rotation.into();
                    pelorus_cuda::add_to_map(stream, buffer, transl, &rot_quat, scan_id);
                }
                #[cfg(feature = "metal")]
                ActiveBackend::Metal(stream) => {
                    let LocatedBufferView::Metal { buffer } = cloud else {
                        panic!("Metal backend requires Metal buffer");
                    };
                    let transl = &imu_transform.translation.vector.into();
                    let rot_quat: pelorus_metal::Quaternion = imu_transform.rotation.into();
                    pelorus_metal::add_to_map(stream, buffer, transl, &rot_quat, scan_id);
                }
                #[cfg(feature = "vulkan")]
                ActiveBackend::Vulkan(stream) => {
                    let LocatedBufferView::Vulkan { buffer } = cloud else {
                        panic!("Vulkan backend requires Vulkan buffer");
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
                        self.planes_cpu_buffer
                            .reserve(cloud_len_alloc - self.planes_cpu_buffer.capacity());
                    }
                }
                #[cfg(feature = "cuda")]
                ActiveBackend::Cuda(stream) => {
                    if self.planes_cpu_buffer.capacity() < cloud.len() {
                        self.planes_cpu_buffer
                            .reserve(cloud_len_alloc - self.planes_cpu_buffer.capacity());
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
                        self.planes_cpu_buffer
                            .reserve(cloud_len_alloc - self.planes_cpu_buffer.capacity());
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
                        self.planes_cpu_buffer
                            .reserve(cloud_len_alloc - self.planes_cpu_buffer.capacity());
                    }
                    let needs_alloc = match &self.planes_gpu_buffer {
                        Some(GpuPlanesBuffer::Vulkan(buf)) => buf.n < cloud.len(),
                        _ => true,
                    };
                    if needs_alloc {
                        self.planes_gpu_buffer = Some(GpuPlanesBuffer::Vulkan(pelorus_vulkan::VulkanBuffer::new(
                            stream,
                            std::mem::size_of::<pelorus_vulkan::PointNormal>(),
                            cloud_len_alloc,
                        )));
                    }
                }
            }

            let _planes_cpu_buffer_ptr = std::ptr::addr_of_mut!(self.planes_cpu_buffer);
            let _planes_gpu_buffer_ptr = std::ptr::addr_of_mut!(self.planes_gpu_buffer);
            let map_ptr = std::ptr::addr_of!(self.map);

            let mut cached_planes: Option<Vec<PointXYZINormal>> = None;
            let eval_cb_inner = eval_cb.clone();

            self.ekf.correct(
                self.laser_point_noise_cov,
                lidar_timestamp,
                self.ekf_max_iterations,
                eval_cb.clone(),
                move |ekf_state: &crate::iekf::StateIKFom,
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

                    let planes: &Vec<PointXYZINormal> = if let Some(ref p) = cached_planes {
                        p
                    } else {
                        let retrieved_planes = match backend {
                            ActiveBackend::Cpu => {
                                #[allow(irrefutable_let_patterns)]
                                let LocatedBufferView::Ram(cloud_pts) = cloud else {
                                    panic!("CPU backend requires RAM buffer");
                                };
                                let map = unsafe { &*map_ptr };
                                cloud_pts
                                    .par_iter()
                                    .flat_map(|point_local| {
                                        let p_l = nalgebra::Point3::new(
                                            point_local.x as f64,
                                            point_local.y as f64,
                                            point_local.z as f64,
                                        );
                                        let p_imu = ekf_state.offset_r_l_i.matrix() * p_l.coords
                                            + ekf_state.offset_t_l_i;
                                        let p_world = ekf_state.pose().transform(p_imu);
                                        let query_point = PointXYZ::new(
                                            p_world.x as f32,
                                            p_world.y as f32,
                                            p_world.z as f32,
                                        );
                                        let plane = map.plane_fit_from_distributions(&query_point)?;
                                        let pd2 = plane[0] * p_world.x as f32
                                            + plane[1] * p_world.y as f32
                                            + plane[2] * p_world.z as f32
                                            + plane[3];
                                        let l2_sq = point_local.x * point_local.x
                                            + point_local.y * point_local.y
                                            + point_local.z * point_local.z;
                                        let s = 1.0_f32
                                            - 0.9_f32 * pd2.abs()
                                                / l2_sq.max(1e-9_f32).sqrt().sqrt();
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
                                    })
                                    .collect()
                            }
                            #[cfg(feature = "cuda")]
                            ActiveBackend::Cuda(stream) => {
                                let planes_buf = unsafe { &mut *_planes_cpu_buffer_ptr };
                                planes_buf.clear();
                                unsafe { planes_buf.set_len(cloud.len()) };
                                let planes_gpu = match unsafe { &mut *_planes_gpu_buffer_ptr } {
                                    Some(GpuPlanesBuffer::Cuda(buf)) => buf,
                                    _ => panic!("CUDA backend: GPU planes buffer not allocated"),
                                };
                                let LocatedBufferView::Cuda { buffer } = cloud else {
                                    panic!("CUDA backend requires CUDA buffer");
                                };
                                let pose_q: pelorus_cuda::Quaternion =
                                    UnitQuaternion::from_matrix(
                                        &ekf_state.pose().rotation().matrix().clone(),
                                    )
                                    .into();
                                let offset_q: pelorus_cuda::Quaternion =
                                    UnitQuaternion::from_matrix(
                                        &ekf_state.offset_r_l_i.matrix().clone(),
                                    )
                                    .into();
                                let pose_t = ekf_state.pose().matrix().column(3).fixed_rows::<3>(0).into_owned();
                                pelorus_cuda::planes_cuda(
                                    stream,
                                    buffer,
                                    &pelorus_cuda::Vector3::new(
                                        pose_t.x,
                                        pose_t.y,
                                        pose_t.z,
                                    ),
                                    &pose_q,
                                    &pelorus_cuda::Vector3::new(
                                        ekf_state.offset_t_l_i.x,
                                        ekf_state.offset_t_l_i.y,
                                        ekf_state.offset_t_l_i.z,
                                    ),
                                    &offset_q,
                                    planes_buf,
                                    planes_gpu,
                                );
                                planes_buf.clone()
                            }
                            #[cfg(feature = "metal")]
                            ActiveBackend::Metal(stream) => {
                                let planes_buf = unsafe { &mut *_planes_cpu_buffer_ptr };
                                planes_buf.clear();
                                unsafe { planes_buf.set_len(cloud.len()) };
                                let planes_gpu = match unsafe { &mut *_planes_gpu_buffer_ptr } {
                                    Some(GpuPlanesBuffer::Metal(buf)) => buf,
                                    _ => panic!("Metal backend: GPU planes buffer not allocated"),
                                };
                                let LocatedBufferView::Metal { buffer } = cloud else {
                                    panic!("Metal backend requires Metal buffer");
                                };
                                let pose_q: pelorus_metal::Quaternion = {
                                    let mat: Matrix3<f64> =
                                        ekf_state.pose().rotation().matrix().clone();
                                    UnitQuaternion::from_matrix(&mat).into()
                                };
                                let offset_q: pelorus_metal::Quaternion = {
                                    let mat: Matrix3<f64> =
                                        ekf_state.offset_r_l_i.matrix().clone();
                                    UnitQuaternion::from_matrix(&mat).into()
                                };
                                let pose_t = ekf_state.pose().matrix().column(3).fixed_rows::<3>(0).into_owned();
                                pelorus_metal::planes_metal(
                                    stream,
                                    buffer,
                                    &pelorus_metal::Vector3::new(
                                        pose_t.x,
                                        pose_t.y,
                                        pose_t.z,
                                    ),
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
                                planes_buf.clone()
                            }
                            #[cfg(feature = "vulkan")]
                            ActiveBackend::Vulkan(stream) => {
                                let planes_buf = unsafe { &mut *_planes_cpu_buffer_ptr };
                                planes_buf.clear();
                                unsafe { planes_buf.set_len(cloud.len()) };
                                let planes_gpu = match unsafe { &mut *_planes_gpu_buffer_ptr } {
                                    Some(GpuPlanesBuffer::Vulkan(buf)) => buf,
                                    _ => panic!("Vulkan backend: GPU planes buffer not allocated"),
                                };
                                let LocatedBufferView::Vulkan { buffer } = cloud else {
                                    panic!("Vulkan backend requires Vulkan buffer");
                                };
                                let pose_q: pelorus_vulkan::Quaternion =
                                    UnitQuaternion::from_matrix(
                                        &ekf_state.pose().rotation().matrix().clone(),
                                    )
                                    .into();
                                let offset_q: pelorus_vulkan::Quaternion =
                                    UnitQuaternion::from_matrix(
                                        &ekf_state.offset_r_l_i.matrix().clone(),
                                    )
                                    .into();
                                let pose_t = ekf_state.pose().matrix().column(3).fixed_rows::<3>(0).into_owned();
                                pelorus_vulkan::planes_vulkan(
                                    stream,
                                    buffer,
                                    &pelorus_vulkan::Vector3::new(
                                        pose_t.x,
                                        pose_t.y,
                                        pose_t.z,
                                    ),
                                    &pose_q,
                                    &pelorus_vulkan::Vector3::new(
                                        ekf_state.offset_t_l_i.x,
                                        ekf_state.offset_t_l_i.y,
                                        ekf_state.offset_t_l_i.z,
                                    ),
                                    &offset_q,
                                    planes_buf,
                                    planes_gpu,
                                );
                                planes_buf.clone()
                            }
                        };
                        cached_planes = Some(retrieved_planes);
                        cached_planes.as_ref().unwrap()
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
                        let point_local = Vector3::new(normal_point.x as f64, normal_point.y as f64, normal_point.z as f64);
                        let norm = Vector3::new(
                            normal_point.normal_x as f64,
                            normal_point.normal_y as f64,
                            normal_point.normal_z as f64,
                        );
                        let plane_d = normal_point.intensity as f64;

                        let p_imu = ekf_state.offset_r_l_i.matrix() * point_local
                            + ekf_state.offset_t_l_i;

                        let p_world = ekf_state.pose().transform(p_imu);
                        let pd2 = norm.dot(&p_world) + plane_d;

                        let point_imu_skew = skew_symmetric_na(&p_imu);
                        let c = ekf_state.pose().rotation().matrix().transpose() * norm;
                        let a_vec = point_imu_skew * c;

                        h_x[(i, 0)] = norm.x;
                        h_x[(i, 1)] = norm.y;
                        h_x[(i, 2)] = norm.z;

                        h_x[(i, 3)] = a_vec.x;
                        h_x[(i, 4)] = a_vec.y;
                        h_x[(i, 5)] = a_vec.z;

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

        if let Some(ready) = ready_for_next {
            let _ = ready.send(());
        }

        {
            let mut eval_guard = eval_cb.write().unwrap();
            let eval_actual = &mut **eval_guard;
            eval_actual.start("map_update").unwrap();
        }

        match backend {
            ActiveBackend::Cpu => self.add_to_map(cloud, t_w_l, scan_id),
            #[cfg(feature = "cuda")]
            ActiveBackend::Cuda(stream) => {
                let LocatedBufferView::Cuda { buffer } = cloud else {
                    panic!("CUDA backend requires CUDA buffer");
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
                    panic!("Metal backend requires Metal buffer");
                };
                let transl = &t_w_l.translation.vector.into();
                let rot_quat: pelorus_metal::Quaternion = {
                    let mat: Matrix3<f64> =
                        t_w_l.rotation.to_rotation_matrix().matrix().clone();
                    UnitQuaternion::from_matrix(&mat).into()
                };
                pelorus_metal::add_to_map(stream, buffer, transl, &rot_quat, scan_id);
                stream.sync();
            }
            #[cfg(feature = "vulkan")]
            ActiveBackend::Vulkan(stream) => {
                let LocatedBufferView::Vulkan { buffer } = cloud else {
                    panic!("Vulkan backend requires Vulkan buffer");
                };
                let transl = &t_w_l.translation.vector.into();
                let rot_quat: pelorus_vulkan::Quaternion = {
                    let mat: Matrix3<f64> =
                        t_w_l.rotation.to_rotation_matrix().matrix().clone();
                    UnitQuaternion::from_matrix(&mat).into()
                };
                pelorus_vulkan::add_to_map(
                    stream,
                    buffer,
                    transl,
                    &rot_quat,
                    scan_id,
                );
                stream.sync();
            }
        }
        {
            let mut eval_guard = eval_cb.write().unwrap();
            let eval_actual = &mut **eval_guard;
            eval_actual.stop("map_update").unwrap();
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
        imu_msgs: &[ImuMsg],
        odom_msgs: &[OdomMsg],
        kinematic: KinematicModel,
        constraint_std: f64,
        last_cloud_end_time: f64,
        cloud_begin_time: f64,
        cloud_end_time: f64,
        cloud: &mut Vec<PointXYZIT>,
        cloud_passthrough: Option<&mut Vec<PointXYZIT>>,
        lidar_vendor: Option<LidarVendor>,
    ) {
        self.imu_processing.undistort(
            &mut self.ekf,
            imu_msgs,
            odom_msgs,
            kinematic,
            constraint_std,
            last_cloud_end_time,
            cloud_begin_time,
            cloud_end_time,
            cloud,
            cloud_passthrough,
            lidar_vendor
        );
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

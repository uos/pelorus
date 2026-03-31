use faer::Mat as FaerMat;
use nalgebra::*;
use pelorus_core::{
    ImuParameters, KinematicModel, LidarVendor, OdomMsg, PointXYZIT, timing::RuntimeEvaluator,
};
#[cfg(feature = "cuda")]
use pelorus_cuda::CudaBuffer;
#[cfg(feature = "metal")]
use pelorus_metal::MetalBuffer;
#[cfg(feature = "vulkan")]
use pelorus_vulkan::VulkanBuffer;
use prelio::LocatedBufferView;
#[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
use rayon::prelude::*;
use ros_pointcloud2::{
    impl_pointxyz_for_nalgebra,
    points::{PointXYZ, PointXYZINormal},
};
impl_pointxyz_for_nalgebra!();

use sophus_lie::prelude::*;
pub use spole::IVoxNearbyType;

use crate::{iekf::skew_symmetric_na, registration::impl_nalgebra::AsNalgebra};

pub struct RegistrationState {
    ekf: crate::iekf::Iekf,
    imu_processing: crate::imu::ImuProcessing,
    pub map: spole::IVoxHashMap,
    laser_point_noise_cov: f64,
    ekf_max_iterations: usize,
    base_to_lidar: Isometry3<f64>,
    #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
    planes_cpu_buffer: Vec<PointXYZINormal>,
    #[cfg(feature = "cuda")]
    planes_gpu_buffer: Option<CudaBuffer>,
    #[cfg(feature = "metal")]
    planes_gpu_buffer: Option<MetalBuffer>,
    #[cfg(feature = "vulkan")]
    planes_gpu_buffer: Option<VulkanBuffer>,
    #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
    planes_buffer: Vec<PointXYZINormal>,
}

fn extract_3x3_block(mat: &FaerMat<f64>, row_offset: usize, col_offset: usize) -> Matrix3<f64> {
    let mut result = Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            result[(i, j)] = mat[(row_offset + i, col_offset + j)];
        }
    }
    result
}

fn rotate_covariance(cov: &Matrix3<f64>, rot: &Matrix3<f64>) -> Matrix3<f64> {
    rot * cov * rot.transpose()
}

impl RegistrationState {
    pub fn new(
        map_resolution: f64,
        map_capacity: usize,
        g: f64,
        imu_params: &ImuParameters,
        laser_point_noise_cov: f64,
        _neighbor_max_range: f64,
        lidar_to_imu: Isometry3<f64>,
        base_to_lidar: Isometry3<f64>,
        knn_nearby_voxel_search_range: IVoxNearbyType,
        update_cov_sensitivity: f32,
        ekf_max_iterations: usize,
    ) -> Self {
        let map = spole::IVoxHashMap::new(
            map_resolution as f32,
            map_capacity,
            knn_nearby_voxel_search_range,
            update_cov_sensitivity,
        );

        let ekf = crate::iekf::Iekf::new(g, base_to_lidar);

        Self {
            ekf,
            imu_processing: crate::imu::ImuProcessing::new(
                imu_params,
                &lidar_to_imu,
                &base_to_lidar,
            ),
            map,
            laser_point_noise_cov,
            ekf_max_iterations,
            base_to_lidar,
            #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
            planes_cpu_buffer: Vec::new(),
            #[cfg(feature = "cuda")]
            planes_gpu_buffer: None,
            #[cfg(feature = "metal")]
            planes_gpu_buffer: None,
            #[cfg(feature = "vulkan")]
            planes_gpu_buffer: None,
            #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
            planes_buffer: Vec::new(),
        }
    }

    #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
    fn add_to_map(&mut self, cloud: &LocatedBufferView<PointXYZIT>, pose: Isometry3<f64>) {
        let LocatedBufferView::Ram(cloud) = cloud;

        #[cfg(feature = "nn_kdt")]
        {
            self.map.add(cloud, &pose);
        }

        #[cfg(not(feature = "nn_kdt"))]
        {
            self.map.add(cloud, &pose);
        }
    }

    // Just for debugging! Really slow.
    pub fn get_entire_map(&self) -> Vec<nalgebra::Point3<f32>> {
        self.map.get_entire_map()
    }

    pub fn clear_localmap(&mut self) {
        self.map.clear();
        #[cfg(feature = "cuda")]
        pelorus_cuda::cleanup_map();
        #[cfg(feature = "metal")]
        pelorus_metal::cleanup_map();
        #[cfg(feature = "vulkan")]
        pelorus_vulkan::cleanup_map();
    }

    pub fn get_odometry_covariances(&self) -> ([f64; 36], [f64; 9]) {
        // 1. Get current rotation matrix (World to Body)
        let r_wb = self.ekf.state.pose().rotation().matrix();
        let r_bw = r_wb.transpose(); // Body to World (and vice versa for rotation)

        // --- PREPARE POSE COVARIANCE (6x6) in WORLD FRAME ---
        // 24-dim state: pos(0-2), rot(3-5), offset_R(6-8), offset_T(9-11), vel(12-14), bg(15-17), ba(18-20), grav(21-23)
        // Pos (0..3) is World Frame -> OK
        // Rot (3..6) is Body Frame  -> Needs Rotation to World

        // Extract raw blocks from faer Mat
        let cov_pp = extract_3x3_block(&self.ekf.cov, 0, 0); // Pos-Pos (World)
        let cov_rr_body = extract_3x3_block(&self.ekf.cov, 3, 3); // Rot-Rot (Body)
        let cov_pr_mixed = extract_3x3_block(&self.ekf.cov, 0, 3); // Pos-Rot (Row:World, Col:Body)

        // Rotate Orientation block to World: R * Cov_body * R^T
        let cov_rr_world = rotate_covariance(&cov_rr_body, &r_wb);

        // Rotate Cross-Correlations (Pos-Rot):
        // Current: E[dPos_world * dRot_body^T]
        // Target:  E[dPos_world * (R * dRot_body)^T] = E[...] * R^T
        let cov_pr_world = cov_pr_mixed * r_wb.transpose();
        let cov_rp_world = cov_pr_world.transpose();

        let mut pose_cov6 = [0.0; 36];
        // Populate ROS Pose Covariance (All World Frame now)
        for i in 0..3 {
            for j in 0..3 {
                // Top-Left: Position (World)
                pose_cov6[i * 6 + j] = cov_pp[(i, j)];
                // Top-Right: Pos-Rot (World)
                pose_cov6[i * 6 + (3 + j)] = cov_pr_world[(i, j)];
                // Bottom-Left: Rot-Pos (World)
                pose_cov6[(3 + i) * 6 + j] = cov_rp_world[(i, j)];
                // Bottom-Right: Rotation (World)
                pose_cov6[(3 + i) * 6 + (3 + j)] = cov_rr_world[(i, j)];
            }
        }

        // --- PREPARE TWIST COVARIANCE (3x3 linear part) in BODY FRAME ---
        // Current state: Vel (12..15) is World Frame.
        // Target: ROS Twist Linear is Body Frame.
        // Transform: Cov_body = R^T * Cov_world * R

        let cov_vv_world = extract_3x3_block(&self.ekf.cov, 12, 12);
        // Use r_bw (which is R^T) to rotate World->Body
        let cov_vv_body = rotate_covariance(&cov_vv_world, &r_bw);

        let mut lv_cov3 = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                lv_cov3[i * 3 + j] = cov_vv_body[(i, j)];
            }
        }

        (pose_cov6, lv_cov3)
    }

    pub fn get_twist_covariance_6x6(&self) -> [f64; 36] {
        let mut cov6 = [0.0; 36];

        let r_wb = self.ekf.state.pose().rotation().matrix();
        let r_bw = r_wb.transpose();

        // 1. Linear Velocity (Indices 12..15 in 24-dim state) - Currently World Frame
        // Needs rotation to Body Frame for ROS Twist
        let cov_vv_world = extract_3x3_block(&self.ekf.cov, 12, 12);
        let cov_vv_body = rotate_covariance(&cov_vv_world, &r_bw);

        // 2. Angular Velocity
        // The IMU measurement noise/bias is usually already in Body Frame.
        // So this part is likely fine without rotation.

        for i in 0..3 {
            for j in 0..3 {
                // Linear Velocity (Body)
                cov6[i * 6 + j] = cov_vv_body[(i, j)];

                // Angular Velocity (Body)
                // Assuming ekf_cov_gyr is already in sensor frame (Body)
                cov6[(3 + i) * 6 + (3 + j)] = self.imu_processing.ekf_cov_gyr[(i, j)]
                    + self.imu_processing.ekf_cov_gyr_bias[(i, j)];
            }
        }
        cov6
    }

    #[allow(dead_code)]
    fn symmetric_eigen_3x3(a: &Matrix3<f32>) -> Vector3<f32> {
        // The characteristic equation is λ³ - tr(A)λ² + c₂λ - det(A) = 0
        // where c₂ = 0.5 * (tr(A)² - tr(A²))
        let trace_a = a.trace();

        let c2 = (a[(0, 0)] * a[(1, 1)] + a[(0, 0)] * a[(2, 2)] + a[(1, 1)] * a[(2, 2)])
            - (a[(0, 1)] * a[(0, 1)] + a[(0, 2)] * a[(0, 2)] + a[(1, 2)] * a[(1, 2)]);

        let det_a = a.determinant();

        // Solve the cubic equation: x³ + p*x² + q*x + r = 0
        let p = -trace_a;
        let q = c2;
        let r = -det_a;

        // Convert to depressed cubic: t³ + a*t + b = 0
        let a_cub = (3.0 * q - p * p) / 3.0;
        let b_cub = (2.0 * p * p * p - 9.0 * p * q + 27.0 * r) / 27.0;

        // For symmetric matrices, eigenvalues are real, so the discriminant is <= 0.
        // We use the trigonometric solution for three real roots.
        let angle = if a_cub.abs() < 1e-6 {
            // Handle case where a is zero
            0.0
        } else {
            let acos_arg = (-b_cub / 2.0) / (-a_cub * a_cub * a_cub / 27.0).sqrt();
            // Clamp the argument to acos to handle floating point inaccuracies
            acos_arg.clamp(-1.0, 1.0).acos()
        };

        let two_sqrt_neg_a_div_3 = 2.0 * (-a_cub / 3.0).sqrt();
        let p_div_3 = p / 3.0;

        // Root 1
        let e1 = two_sqrt_neg_a_div_3 * (angle / 3.0).cos() - p_div_3;
        // Root 2
        let e2 =
            two_sqrt_neg_a_div_3 * ((angle + 2.0 * std::f32::consts::PI) / 3.0).cos() - p_div_3;
        // Root 3
        let e3 =
            two_sqrt_neg_a_div_3 * ((angle + 4.0 * std::f32::consts::PI) / 3.0).cos() - p_div_3;

        Vector3::new(e1, e2, e3)
    }

    #[allow(dead_code)]
    pub fn plane_fit_stable<I>(points_iter: I, n: usize, error_threshold: f32) -> Option<[f32; 4]>
    where
        I: Iterator<Item = Vector3<f32>> + Clone,
    {
        if n < 3 {
            return None;
        }
        let n_f32 = n as f32;

        let centroid = points_iter.clone().sum::<Vector3<f32>>() / n_f32;
        let mut covariance = points_iter.clone().fold(Matrix3::zeros(), |cov, p| {
            let r = p - centroid;
            cov + r * r.transpose()
        });
        covariance /= n_f32;

        let eigenvalues = Self::symmetric_eigen_3x3(&covariance);

        let min_eigenvalue = eigenvalues.x.min(eigenvalues.y).min(eigenvalues.z);

        // the eigenvector for the smallest eigenvalue
        let mut m = covariance;
        m[(0, 0)] -= min_eigenvalue;
        m[(1, 1)] -= min_eigenvalue;
        m[(2, 2)] -= min_eigenvalue;

        let row0 = m.row(0).transpose();
        let row1 = m.row(1).transpose();
        let row2 = m.row(2).transpose();

        // compute all three cross products and pick the one with the largest magnitude for stability
        let v1 = row0.cross(&row1);
        let v2 = row1.cross(&row2);
        let v3 = row2.cross(&row0);

        let mag1_sq = v1.norm_squared();
        let mag2_sq = v2.norm_squared();
        let mag3_sq = v3.norm_squared();

        let normal_vec = if mag1_sq > mag2_sq && mag1_sq > mag3_sq {
            v1
        } else if mag2_sq > mag3_sq {
            v2
        } else {
            v3
        };

        let normal = normal_vec.try_normalize(1e-12)?;

        let d = -normal.dot(&centroid);

        for p in points_iter {
            let distance = (normal.dot(&p) + d).abs();
            if distance > error_threshold {
                return None;
            }
        }

        Some([normal.x, normal.y, normal.z, d])
    }

    // https://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html
    #[allow(dead_code)]
    fn plane_from_points(points: &[PointXYZ], error_threshold: f64) -> Option<[f64; 4]> {
        let points = points
            .iter()
            .map(|p| nalgebra::Point3::new(p.x, p.y, p.z).cast())
            .collect::<Vec<_>>();
        let n = points.len();
        if n < 3 {
            return None;
        }

        let mut sum = Vector3::new(0.0, 0.0, 0.0);
        for p in points.iter() {
            sum += Vector3::new(p.x, p.y, p.z);
        }
        let centroid = sum * (1.0 / (n as f64));

        // Calculate full 3x3 covariance matrix, excluding symmetries:
        let mut xx = 0.0;
        let mut xy = 0.0;
        let mut xz = 0.0;
        let mut yy = 0.0;
        let mut yz = 0.0;
        let mut zz = 0.0;

        for p in points.iter() {
            let r = p - centroid;
            xx += r.x * r.x;
            xy += r.x * r.y;
            xz += r.x * r.z;
            yy += r.y * r.y;
            yz += r.y * r.z;
            zz += r.z * r.z;
        }

        xx /= n as f64;
        xy /= n as f64;
        xz /= n as f64;
        yy /= n as f64;
        yz /= n as f64;
        zz /= n as f64;

        let mut weighted_dir = Vector3::new(0.0, 0.0, 0.0);

        {
            let det_x = yy * zz - yz * yz;
            let axis_dir = Vector3::new(det_x, xz * yz - xy * zz, xy * yz - xz * yy);
            let mut weight = det_x * det_x;
            if weighted_dir.dot(&axis_dir) < 0.0 {
                weight = -weight;
            }
            weighted_dir += axis_dir * weight;
        }

        {
            let det_y = xx * zz - xz * xz;
            let axis_dir = Vector3::new(xz * yz - xy * zz, det_y, xy * xz - yz * xx);
            let mut weight = det_y * det_y;
            if weighted_dir.dot(&axis_dir) < 0.0 {
                weight = -weight;
            }
            weighted_dir += axis_dir * weight;
        }

        {
            let det_z = xx * yy - xy * xy;
            let axis_dir = Vector3::new(xy * yz - xz * yy, xy * xz - yz * xx, det_z);
            let mut weight = det_z * det_z;
            if weighted_dir.dot(&axis_dir) < 0.0 {
                weight = -weight;
            }
            weighted_dir += axis_dir * weight;
        }

        let normal = &weighted_dir.normalize();
        if normal.iter().all(|x| x.is_finite()) {
            let d = -normal.dot(&centroid);
            for p in points {
                let error = normal.x * p.x + normal.y * p.y + normal.z * p.z + d;
                if error.abs() > error_threshold {
                    return None;
                }
            }
            Some([normal.x, normal.y, normal.z, d])
        } else {
            None
        }
    }
}

// TODO add ekf init function to here, not in pelorus
impl crate::Registration for RegistrationState {
    fn undistort(
        &mut self,
        imu_msgs: &[crate::ImuMsg],
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
        // cloud.iter().map(|p| PointXYZ::new(p.x, p.y, p.z)).collect() // bypass
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
            lidar_vendor,
        )
    }

    fn requested_register_and_map_cloud_type(&self) -> LocatedBufferView<PointXYZ> {
        crate::LocatedBufferView::Ram(Vec::new())
    }

    fn get_state(&self) -> crate::ImuEkfPrediction {
        crate::ImuEkfPrediction {
            pose: Isometry3::from_parts(
                Translation3::from(self.ekf.state.pose().translation()),
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                    &self.ekf.state.pose().rotation().matrix(),
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

    fn register_and_map(
        &mut self,
        map_only: bool,
        imu_transform: Isometry3<f64>,
        cloud: &LocatedBufferView<PointXYZIT>,
        lidar_timestamp: f64,
        ready_for_next: Option<&std::sync::mpsc::Sender<()>>,
        eval: &mut RuntimeEvaluator,
        #[cfg(feature = "cuda")] stream: &pelorus_cuda::Stream,
        #[cfg(feature = "metal")] stream: &pelorus_metal::Stream,
        #[cfg(feature = "vulkan")] stream: &pelorus_vulkan::Stream,
    ) -> anyhow::Result<Option<crate::CloudRegisterResult>> {
        let eval_outer = std::sync::Arc::new(std::sync::RwLock::new(eval));
        #[cfg(feature = "cuda")]
        let map_empty = pelorus_cuda::get_map_size() == 0;
        #[cfg(feature = "metal")]
        let map_empty = pelorus_metal::get_map_size() == 0;
        #[cfg(feature = "vulkan")]
        let map_empty = pelorus_vulkan::get_map_size() == 0;

        if map_only {
            #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
            self.add_to_map(cloud, imu_transform);
            #[cfg(feature = "cuda")]
            {
                let cloud = match cloud {
                    LocatedBufferView::Ram(_) => panic!("cuda buffer in cpu task"),
                    LocatedBufferView::Cuda { buffer } => buffer,
                    #[cfg(any(feature = "metal", feature = "vulkan"))]
                    _ => unreachable!(),
                };

                let transl = &imu_transform.translation.vector.into();
                let rot_quat: pelorus_cuda::Quaternion = imu_transform.rotation.into();
                pelorus_cuda::add_to_map(&stream, cloud, &transl, &rot_quat);
            }
            #[cfg(feature = "metal")]
            {
                let cloud = match cloud {
                    LocatedBufferView::Ram(_) => panic!("metal buffer in cpu task"),
                    LocatedBufferView::Metal { buffer } => buffer,
                    #[cfg(any(feature = "cuda", feature = "vulkan"))]
                    _ => unreachable!(),
                };

                let transl = &imu_transform.translation.vector.into();
                let rot_quat: pelorus_metal::Quaternion = imu_transform.rotation.into();
                pelorus_metal::add_to_map(&stream, cloud, &transl, &rot_quat);
            }
            #[cfg(feature = "vulkan")]
            {
                let cloud = match cloud {
                    LocatedBufferView::Ram(_) => panic!("vulkan buffer in cpu task"),
                    LocatedBufferView::Vulkan { buffer } => buffer,
                    #[cfg(any(feature = "cuda", feature = "metal"))]
                    _ => unreachable!(),
                };

                let transl = &imu_transform.translation.vector.into();
                let rot_quat: pelorus_vulkan::Quaternion = imu_transform.rotation.into();
                pelorus_vulkan::add_to_map(&stream, cloud, &transl, &rot_quat);
            }

            return Ok(None);
        }

        #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
        let map_empty = self.map.is_empty();
        if !map_empty {
            #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
            let map = &mut self.map;
            const EXTRA: f32 = if cfg!(any(feature = "cuda", feature = "metal", feature = "vulkan"))
            {
                1.7
            } else {
                1.5
            };
            let cloud_len_alloc = (EXTRA * cloud.len() as f32).ceil() as usize;
            {
                eval_outer.write().unwrap().start("ekf_correct").unwrap();
            }

            let eval_cb = eval_outer.clone();

            #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
            {
                if self.planes_cpu_buffer.capacity() < cloud.len() {
                    self.planes_cpu_buffer
                        .reserve(cloud_len_alloc - self.planes_cpu_buffer.capacity());
                }

                #[cfg(feature = "cuda")]
                if self.planes_gpu_buffer.is_none()
                    || self.planes_gpu_buffer.as_ref().unwrap().n < cloud.len()
                {
                    self.planes_gpu_buffer = Some(CudaBuffer::new(
                        stream,
                        size_of::<PointXYZINormal>(),
                        cloud_len_alloc,
                    ));
                }

                #[cfg(feature = "metal")]
                if self.planes_gpu_buffer.is_none()
                    || self.planes_gpu_buffer.as_ref().unwrap().n < cloud.len()
                {
                    self.planes_gpu_buffer = Some(MetalBuffer::new(
                        stream,
                        size_of::<pelorus_metal::PointNormal>(),
                        cloud_len_alloc,
                    ));
                }
                #[cfg(feature = "vulkan")]
                if self.planes_gpu_buffer.is_none()
                    || self.planes_gpu_buffer.as_ref().unwrap().n < cloud.len()
                {
                    self.planes_gpu_buffer = Some(VulkanBuffer::new_host_visible(
                        stream,
                        size_of::<pelorus_vulkan::PointNormal>(),
                        cloud_len_alloc,
                    ));
                }
            }

            #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
            {
                if self.planes_buffer.capacity() < cloud.len() {
                    self.planes_buffer
                        .reserve(cloud_len_alloc - self.planes_buffer.capacity());
                }
            }

            #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
            let planes_cpu_buffer = std::ptr::addr_of_mut!(self.planes_cpu_buffer);
            #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
            let planes_gpu_buffer = std::ptr::addr_of_mut!(self.planes_gpu_buffer);

            #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
            let planes_buffer = std::ptr::addr_of_mut!(self.planes_buffer);

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
                    let mut eval = eval_cb.write().unwrap();
                    eval.start("ekf_correct/model/planes").unwrap();

                    #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
                    let planes = unsafe { &mut *planes_cpu_buffer };
                    #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
                    {
                        planes.clear();
                        unsafe { planes.set_len(cloud.len()) };
                    }

                    #[cfg(feature = "cuda")]
                    let planes = {
                        let planes_gpu = unsafe { (*planes_gpu_buffer).as_mut().unwrap() };
                        let cloud = match cloud {
                            LocatedBufferView::Ram(_) => {
                                panic!("cloud not allocated on GPU")
                            }
                            LocatedBufferView::Cuda { buffer } => buffer,
                            #[cfg(any(feature = "metal", feature = "vulkan"))]
                            _ => unreachable!(),
                        };

                        // call to cuda kernel
                        let pose_q: pelorus_cuda::Quaternion =
                            nalgebra::UnitQuaternion::from_matrix(
                                &ekf_state.pose().rotation().matrix().clone(),
                            )
                            .into();
                        let offset_q: pelorus_cuda::Quaternion =
                            nalgebra::UnitQuaternion::from_matrix(
                                &ekf_state.offset_r_l_i.matrix().clone(),
                            )
                            .into();
                        pelorus_cuda::planes_cuda(
                            &stream,
                            cloud,
                            &pelorus_cuda::Vector3::new(
                                ekf_state.pose().translation().x,
                                ekf_state.pose().translation().y,
                                ekf_state.pose().translation().z,
                            ),
                            &pose_q,
                            &pelorus_cuda::Vector3::new(
                                ekf_state.offset_t_l_i.x,
                                ekf_state.offset_t_l_i.y,
                                ekf_state.offset_t_l_i.z,
                            ),
                            &offset_q,
                            planes,
                            planes_gpu,
                        );

                        planes
                    };

                    #[cfg(feature = "metal")]
                    let planes = {
                        let planes_gpu = unsafe { (*planes_gpu_buffer).as_mut().unwrap() };
                        let cloud = match cloud {
                            LocatedBufferView::Ram(_) => {
                                panic!("cloud not allocated on GPU")
                            }
                            LocatedBufferView::Metal { buffer } => buffer,
                            #[cfg(any(feature = "cuda", feature = "vulkan"))]
                            _ => unreachable!(),
                        };

                        // call to metal kernel
                        // prepare quaternion values up front to avoid borrowing
                        let pose_q: pelorus_metal::Quaternion = {
                            let mat: nalgebra::Matrix3<f64> =
                                ekf_state.pose().rotation().matrix().clone();
                            nalgebra::UnitQuaternion::from_matrix(&mat).into()
                        };
                        let offset_q: pelorus_metal::Quaternion = {
                            let mat: nalgebra::Matrix3<f64> =
                                ekf_state.offset_r_l_i.matrix().clone();
                            nalgebra::UnitQuaternion::from_matrix(&mat).into()
                        };

                        pelorus_metal::planes_metal(
                            &stream,
                            cloud,
                            &pelorus_metal::Vector3::new(
                                ekf_state.pose().translation().x,
                                ekf_state.pose().translation().y,
                                ekf_state.pose().translation().z,
                            ),
                            &pose_q,
                            &pelorus_metal::Vector3::new(
                                ekf_state.offset_t_l_i.x,
                                ekf_state.offset_t_l_i.y,
                                ekf_state.offset_t_l_i.z,
                            ),
                            &offset_q,
                            planes,
                            planes_gpu,
                        );

                        planes
                    };
                    #[cfg(feature = "vulkan")]
                    let planes = {
                        let planes_gpu = unsafe { (*planes_gpu_buffer).as_mut().unwrap() };
                        let cloud = match cloud {
                            LocatedBufferView::Ram(_) => {
                                panic!("cloud not allocated on GPU")
                            }
                            LocatedBufferView::Vulkan { buffer } => buffer,
                            #[cfg(any(feature = "cuda", feature = "metal"))]
                            _ => unreachable!(),
                        };

                        let pose_q: pelorus_vulkan::Quaternion =
                            nalgebra::UnitQuaternion::from_matrix(
                                &ekf_state.pose().rotation().matrix().clone(),
                            )
                            .into();
                        let offset_q: pelorus_vulkan::Quaternion =
                            nalgebra::UnitQuaternion::from_matrix(
                                &ekf_state.offset_r_l_i.matrix().clone(),
                            )
                            .into();
                        pelorus_vulkan::planes_vulkan(
                            &stream,
                            cloud,
                            &pelorus_vulkan::Vector3::new(
                                ekf_state.pose().translation().x,
                                ekf_state.pose().translation().y,
                                ekf_state.pose().translation().z,
                            ),
                            &pose_q,
                            &pelorus_vulkan::Vector3::new(
                                ekf_state.offset_t_l_i.x,
                                ekf_state.offset_t_l_i.y,
                                ekf_state.offset_t_l_i.z,
                            ),
                            &offset_q,
                            planes,
                            planes_gpu,
                        );
                        planes
                    };

                    #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
                    let planes = {
                        let LocatedBufferView::Ram(cloud) = cloud;

                        let planes_buf: &mut Vec<PointXYZINormal> = unsafe { &mut *planes_buffer };
                        planes_buf.clear();

                        let pm = cloud.par_iter().flat_map(|point_local| {
                            let p_l =
                                nalgebra::Point3::new(point_local.x, point_local.y, point_local.z);
                            let p_imu = ekf_state.offset_r_l_i.matrix() * p_l.cast()
                                + ekf_state.offset_t_l_i;
                            let p_world = ekf_state.pose().transform(p_imu.coords);

                            let query_point =
                                PointXYZ::new(p_world.x as f32, p_world.y as f32, p_world.z as f32);

                            {
                                let plane = map.plane_fit_from_distributions(&query_point)?;
                                let pd2 = plane[0] * p_world.x as f32
                                    + plane[1] * p_world.y as f32
                                    + plane[2] * p_world.z as f32
                                    + plane[3];
                                let l2_sq = point_local.x * point_local.x
                                    + point_local.y * point_local.y
                                    + point_local.z * point_local.z;
                                let s =
                                    1.0_f32 - 0.9 * pd2.abs() / l2_sq.max(1e-9_f32).sqrt().sqrt();

                                if s <= 0.9 {
                                    None
                                } else {
                                    Some(PointXYZINormal::new(
                                        point_local.x,
                                        point_local.y,
                                        point_local.z,
                                        pd2,
                                        plane[0],
                                        plane[1],
                                        plane[2],
                                    ))
                                }
                            }
                        });

                        planes_buf.par_extend(pm);

                        planes_buf
                    };
                    eval.stop("ekf_correct/model/planes").unwrap();

                    if planes.is_empty() {
                        *valid = false;
                        *num_measurements = 0;
                        return;
                    }

                    // Set actual number of measurements
                    *num_measurements = planes.len();

                    eval.start("ekf_correct/model/jacobian").unwrap();

                    // Resize h and h_x if needed using faer API
                    if planes.len() > h.nrows() {
                        *h = faer::Col::<f64>::zeros(cloud_len_alloc);
                    }

                    // S-FAST_LIO uses sparse H matrix with only 12 columns:
                    // pos(0-2), rot(3-5), offset_R_L_I(6-8), offset_T_L_I(9-11)
                    // vel, bg, ba, grav don't affect point-to-plane measurement
                    if planes.len() > h_x.nrows() {
                        *h_x = faer::Mat::<f64>::zeros(cloud_len_alloc, 12);
                    }

                    // Process each plane
                    for (i, normal_point) in planes.iter().enumerate() {
                        let point_local = normal_point.xyz().cast::<f64>();
                        let nearest_plane = [
                            normal_point.normal_x as f64,
                            normal_point.normal_y as f64,
                            normal_point.normal_z as f64,
                            normal_point.intensity as f64,
                        ];
                        // point in IMU frame (p_I)
                        let point_imu =
                            ekf_state.offset_r_l_i.matrix() * point_local + ekf_state.offset_t_l_i;

                        let norm =
                            Vector3::new(nearest_plane[0], nearest_plane[1], nearest_plane[2]);

                        // Rotation perturbation - following S-FAST_LIO:
                        // C = R^T * n
                        // A = [p_I]× * C = [p_I]× * R^T * n
                        // Jacobian row is A^T = n^T * R * (-[p_I]×) = -n^T * R * [p_I]×
                        let point_imu_vec = point_imu.coords;
                        let point_imu_skew = skew_symmetric_na(&point_imu_vec);
                        let c = ekf_state.pose().rotation().matrix().transpose() * norm;
                        let a_vec = point_imu_skew * c; // This is A (3x1 column vector)

                        // Position Jacobian (columns 0-2): just the normal vector
                        h_x[(i, 0)] = norm.x;
                        h_x[(i, 1)] = norm.y;
                        h_x[(i, 2)] = norm.z;

                        // Rotation Jacobian (columns 3-5): A vector (transposed in the row)
                        h_x[(i, 3)] = a_vec.x;
                        h_x[(i, 4)] = a_vec.y;
                        h_x[(i, 5)] = a_vec.z;

                        // Extrinsic offset_R_L_I and offset_T_L_I (columns 6-11): zero for now
                        // TODO: implement extrinsic estimation if needed
                        for j in 6..12 {
                            h_x[(i, j)] = 0.0;
                        }

                        h[i] = -nearest_plane[3];
                    }
                    eval.stop("ekf_correct/model/jacobian").unwrap();
                },
            );
            eval_outer.write().unwrap().stop("ekf_correct").unwrap();
        };

        // Prepare transform from LiDAR to IMU frame (T_I_L).
        let t_i_l = Isometry3::from_parts(
            Translation3::from(self.ekf.state.offset_t_l_i),
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                &self.ekf.state.offset_r_l_i.matrix(),
            )),
        );

        // Transform from IMU frame to World frame (T_W_I).
        let t_w_i = Isometry3::from_parts(
            Translation3::from(self.ekf.state.pos),
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                &self.ekf.state.rot.matrix(),
            )),
        );

        // Calculate LiDAR pose in World frame (T_W_L = T_W_I * T_I_L).
        let t_w_l = t_w_i * t_i_l;

        // Calculate Base pose in World frame (T_W_B = T_W_L * T_L_B).
        // self.base_to_lidar is T_B_L, so T_L_B is base_to_lidar.inverse().
        let t_w_b = t_w_l * self.base_to_lidar.inverse();

        // Signal prelio it can use the ekf again
        if let Some(ready) = ready_for_next {
            let _ = ready.send(());
        }

        {
            eval_outer.write().unwrap().start("map_update").unwrap();
        }

        #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
        self.add_to_map(cloud, t_w_l);
        #[cfg(feature = "cuda")]
        {
            let cloud = match cloud {
                LocatedBufferView::Ram(_) => panic!("cuda buffer in cpu task"),
                LocatedBufferView::Cuda { buffer } => buffer,
                #[cfg(any(feature = "metal", feature = "vulkan"))]
                _ => unreachable!(),
            };

            let rot_quat: pelorus_cuda::Quaternion = t_w_l.rotation.into();
            pelorus_cuda::add_to_map(&stream, cloud, &t_w_l.translation.vector.into(), &rot_quat);
            if std::env::var_os("PELORUS_CUDA_SYNC_MAP").is_some() {
                stream.sync();
            }
        }
        #[cfg(feature = "metal")]
        {
            let cloud = match cloud {
                LocatedBufferView::Ram(_) => panic!("metal buffer in cpu task"),
                LocatedBufferView::Metal { buffer } => buffer,
                #[cfg(any(feature = "cuda", feature = "vulkan"))]
                _ => unreachable!(),
            };

            let transl = &t_w_l.translation.vector.into();
            let rot_quat: pelorus_metal::Quaternion = {
                let mat: nalgebra::Matrix3<f64> =
                    t_w_l.rotation.to_rotation_matrix().matrix().clone();
                nalgebra::UnitQuaternion::from_matrix(&mat).into()
            };
            pelorus_metal::add_to_map(&stream, cloud, transl, &rot_quat);
        }
        #[cfg(feature = "vulkan")]
        {
            let cloud = match cloud {
                LocatedBufferView::Ram(_) => panic!("vulkan buffer in cpu task"),
                LocatedBufferView::Vulkan { buffer } => buffer,
                #[cfg(any(feature = "cuda", feature = "metal"))]
                _ => unreachable!(),
            };

            let transl = &t_w_l.translation.vector.into();
            let rot_quat: pelorus_vulkan::Quaternion = {
                let mat: nalgebra::Matrix3<f64> =
                    t_w_l.rotation.to_rotation_matrix().matrix().clone();
                nalgebra::UnitQuaternion::from_matrix(&mat).into()
            };
            pelorus_vulkan::add_to_map(&stream, cloud, transl, &rot_quat);
        }
        {
            eval_outer.write().unwrap().stop("map_update").unwrap();
        }

        Ok(Some(crate::CloudRegisterResult {
            pose: t_w_l,
            base_pose: t_w_b,
            avg_iterations: None,
        }))
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

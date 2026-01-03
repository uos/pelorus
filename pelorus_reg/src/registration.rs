use std::sync::{Arc, Mutex};

use nalgebra::*;
use pelorus_core::{
    ImuParameters, KinematicModel, LidarVendor, OdomMsg, PointXYZIT, timing::RuntimeEvaluator,
};
#[cfg(feature = "cuda")]
use pelorus_cuda::CudaBuffer;
use prelio::LocatedBufferView;
#[cfg(all(feature = "parallel", not(feature = "cuda")))]
use rayon::prelude::*;
use ros_pointcloud2::points::{PointXYZ, PointXYZI, PointXYZINormal};

use sophus_lie::prelude::*;
pub use spole::IVoxNearbyType;

use crate::iekf::skew_symmetric;

pub struct RegistrationState {
    ekf: crate::iekf::Iekf,
    imu_processing: crate::imu::ImuProcessing,
    pub map: spole::IVoxHashMap,
    laser_point_noise_cov: f64,
    ekf_max_iterations: usize,
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
        initial_extrinsic_calibration: Isometry3<f64>,
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

        let ekf = crate::iekf::Iekf::new(g);

        Self {
            ekf,
            imu_processing: crate::imu::ImuProcessing::new(
                imu_params,
                &initial_extrinsic_calibration,
            ),
            map,
            laser_point_noise_cov,
            ekf_max_iterations,
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn add_to_map(&mut self, cloud: &LocatedBufferView<PointXYZI>, pose: Isometry3<f64>) {
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
    }

    pub fn get_odometry_covariances(&self) -> ([f64; 36], [f64; 9]) {
        // 1. Get current rotation matrix (World to Body)
        let r_wb = self.ekf.state.pose.rotation().matrix();
        let r_bw = r_wb.transpose(); // Body to World (and vice versa for rotation)

        // --- PREPARE POSE COVARIANCE (6x6) in WORLD FRAME ---
        // Current state:
        // Pos (3..6) is World Frame -> OK
        // Rot (0..3) is Body Frame  -> Needs Rotation to World

        // Extract raw blocks
        let cov_pp = self.ekf.cov.fixed_view::<3, 3>(3, 3); // Pos-Pos (World)
        let cov_rr_body = self.ekf.cov.fixed_view::<3, 3>(0, 0); // Rot-Rot (Body)
        let cov_pr_mixed = self.ekf.cov.fixed_view::<3, 3>(3, 0); // Pos-Rot (Row:World, Col:Body)

        // Rotate Orientation block to World: R * Cov_body * R^T
        let cov_rr_world = rotate_covariance(&cov_rr_body.into(), &r_wb);

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
        // Current state: Vel (6..9) is World Frame.
        // Target: ROS Twist Linear is Body Frame.
        // Transform: Cov_body = R^T * Cov_world * R

        let cov_vv_world = self.ekf.cov.fixed_view::<3, 3>(6, 6);
        // Use r_bw (which is R^T) to rotate World->Body
        let cov_vv_body = rotate_covariance(&cov_vv_world.into(), &r_bw);

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

        let r_wb = self.ekf.state.pose.rotation().matrix();
        let r_bw = r_wb.transpose();

        // 1. Linear Velocity (Indices 6..9) - Currently World Frame
        // Needs rotation to Body Frame for ROS Twist
        let cov_vv_world = self.ekf.cov.fixed_view::<3, 3>(6, 6);
        let cov_vv_body = rotate_covariance(&cov_vv_world.into(), &r_bw);

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
        cloud: &[PointXYZIT],
        lidar_vendor: Option<LidarVendor>,
    ) -> Vec<PointXYZI> {
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
            lidar_vendor,
        )
    }

    fn requested_register_and_map_cloud_type(&self) -> LocatedBufferView<PointXYZ> {
        crate::LocatedBufferView::Ram(Vec::new())
    }

    fn get_state(&self) -> crate::ImuEkfPrediction {
        crate::ImuEkfPrediction {
            pose: Isometry3::from_parts(
                Translation3::from(self.ekf.state.pose.translation()),
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                    &self.ekf.state.pose.rotation().matrix(),
                )),
            ),
            linear_velocity: self.ekf.state.vel,
            lidar_imu_extrinsic: Isometry3::from_parts(
                Translation3::from(self.ekf.state.offset_t_l_i),
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                    &self.ekf.state.offset_r_l_i.matrix(),
                )),
            ),
        }
    }

    fn register_and_map(
        &mut self,
        map_only: bool,
        imu_transform: Isometry3<f64>,
        cloud: &LocatedBufferView<PointXYZI>,
        lidar_timestamp: f64,
        ready_for_next: Option<&std::sync::mpsc::Sender<()>>,
        eval: &mut RuntimeEvaluator,
        #[cfg(feature = "cuda")] stream: &pelorus_cuda::Stream,
    ) -> anyhow::Result<Option<crate::CloudRegisterResult>> {
        let first_it = Arc::new(Mutex::new(true));
        let eval_outer = std::sync::Arc::new(std::sync::RwLock::new(eval));
        #[cfg(feature = "cuda")]
        let map_empty = pelorus_cuda::get_map_size() == 0;

        if map_only {
            #[cfg(not(feature = "cuda"))]
            self.add_to_map(cloud, imu_transform);
            #[cfg(feature = "cuda")]
            {
                let cloud = match cloud {
                    LocatedBufferView::Ram(_) => panic!("cuda buffer in cpu task"),
                    LocatedBufferView::Cuda { buffer } => buffer,
                };

                let transl = &imu_transform.translation.vector.into();

                let rotmat = imu_transform.rotation.to_rotation_matrix().matrix().clone();
                pelorus_cuda::add_to_map(&stream, cloud, &transl, &rotmat.into());
            }

            return Ok(None);
        }

        #[cfg(not(feature = "cuda"))]
        let map_empty = self.map.is_empty();
        if !map_empty {
            #[cfg(not(feature = "cuda"))]
            let map = &mut self.map;
            const EXTRA: f32 = if cfg!(feature = "cuda") { 1.3 } else { 1.2 };
            let cloud_len_alloc = (EXTRA * cloud.len() as f32).ceil() as usize;
            {
                eval_outer.write().unwrap().start("ekf correct").unwrap();
            }

            let eval_cb = eval_outer.clone();

            #[cfg(feature = "cuda")]
            let planes_cpu_buffer = Arc::new(Mutex::new(Vec::new()));
            #[cfg(feature = "cuda")]
            let planes_gpu_buffer = Arc::new(Mutex::new(CudaBuffer::new(
                stream,
                size_of::<PointXYZINormal>(),
                cloud_len_alloc,
            )));

            self.ekf.correct(
                self.laser_point_noise_cov,
                lidar_timestamp,
                self.ekf_max_iterations,
                move |ekf_state: &crate::iekf::StateIKFom,
                      valid: &mut bool,
                      _converged: bool,
                      h: &mut DVector<f64>,
                      h_x: &mut DMatrix<f64>| {
                    #[cfg(feature = "cuda")]
                    let mut planes = planes_cpu_buffer.lock().unwrap();
                    #[cfg(feature = "cuda")]
                    planes.resize_with(cloud_len_alloc, || PointXYZINormal::default());
                    #[cfg(feature = "cuda")]
                    let mut planes_gpu = planes_gpu_buffer.lock().unwrap();

                    let mut eval = eval_cb.write().unwrap();

                    eval.start("planes").unwrap();
                    #[cfg(feature = "cuda")]
                    let planes = {
                        let cloud = match cloud {
                            LocatedBufferView::Ram(_) => {
                                panic!("cloud not allocated on GPU")
                            }
                            LocatedBufferView::Cuda { buffer } => buffer,
                        };

                        // call to cuda kernel
                        pelorus_cuda::planes_cuda(
                            &stream,
                            cloud,
                            &pelorus_cuda::Vector3::new(
                                ekf_state.pose.translation().x,
                                ekf_state.pose.translation().y,
                                ekf_state.pose.translation().z,
                            ),
                            &ekf_state.pose.rotation().matrix().into(),
                            &pelorus_cuda::Vector3::new(
                                ekf_state.offset_t_l_i.x,
                                ekf_state.offset_t_l_i.y,
                                ekf_state.offset_t_l_i.z,
                            ),
                            &ekf_state.offset_r_l_i.matrix().into(),
                            5, // TODO remove
                            &mut planes,
                            &mut planes_gpu,
                        );

                        planes
                    };

                    #[cfg(not(feature = "cuda"))]
                    let planes = {
                        let LocatedBufferView::Ram(cloud) = cloud;
                        let iter = {
                            #[cfg(feature = "parallel")]
                            {
                                cloud.par_iter()
                            }
                            #[cfg(not(feature = "parallel"))]
                            {
                                cloud.iter()
                            }
                        };

                        let planes: Vec<PointXYZINormal> = iter
                            .filter_map(|point_local| {
                                // Global in IMU frame
                                let point_imu = ekf_state.offset_r_l_i.matrix()
                                    * nalgebra::Point3::new(
                                        point_local.x,
                                        point_local.y,
                                        point_local.z,
                                    )
                                    .cast()
                                    + ekf_state.offset_t_l_i;
                                let point_global = ekf_state.pose.transform(point_imu.coords);

                                let query_point = PointXYZ::new(
                                    point_global.x as f32,
                                    point_global.y as f32,
                                    point_global.z as f32,
                                );

                                {
                                    let plane = map.plane_fit_from_distributions(&query_point)?;
                                    let pd2 = plane[0] * point_global.x as f32
                                        + plane[1] * point_global.y as f32
                                        + plane[2] * point_global.z as f32
                                        + plane[3];
                                    let s = 1.0
                                        - 0.9 * pd2.abs()
                                            / nalgebra::Point3::new(
                                                point_local.x,
                                                point_local.y,
                                                point_local.z,
                                            )
                                            .cast::<f64>()
                                            .coords
                                            .norm()
                                            .sqrt()
                                                as f32;

                                    if s <= 0.9 {
                                        None
                                    } else {
                                        // winning_factor = Some((pd2, 0));

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
                            })
                            .collect::<Vec<_>>();

                        planes
                    };
                    eval.stop("planes").unwrap();

                    *first_it.lock().unwrap() = false;

                    if planes.is_empty() {
                        *valid = false;
                        return;
                    }

                    if planes.len() > h.nrows() {
                        *h = DVector::<f64>::zeros(cloud_len_alloc);
                    }

                    if planes.len() > h_x.nrows() {
                        *h_x = DMatrix::<f64>::zeros(cloud_len_alloc, 18);
                    }

                    eval.start("jacobian").unwrap();
                    planes
                        .iter()
                        .zip(h_x.row_iter_mut())
                        .zip(h.row_iter_mut())
                        .for_each(|((normal_point, mut h_x_row), mut h_row)| {
                            let point_local = nalgebra::Point3::new(
                                normal_point.x,
                                normal_point.y,
                                normal_point.z,
                            )
                            .cast::<f64>();
                            let nearest_plane = [
                                normal_point.normal_x as f64,
                                normal_point.normal_y as f64,
                                normal_point.normal_z as f64,
                                normal_point.intensity as f64,
                            ];
                            let v = ekf_state.offset_r_l_i.matrix() * point_local
                                + ekf_state.offset_t_l_i;

                            let norm =
                                Vector3::new(nearest_plane[0], nearest_plane[1], nearest_plane[2]);

                            // Rotation perturbation
                            let p_imu_global = ekf_state.pose.rotation().matrix() * v.coords;
                            let a = -1.0 * norm.transpose() * skew_symmetric(&p_imu_global);

                            // Position perturbation
                            // derivative of plane equation wrt global position
                            let pos_jacobian = norm;

                            // Rotation Jacobian
                            h_x_row[0] = a.x;
                            h_x_row[1] = a.y;
                            h_x_row[2] = a.z;

                            // Position Jacobian
                            h_x_row[3] = pos_jacobian.x;
                            h_x_row[4] = pos_jacobian.y;
                            h_x_row[5] = pos_jacobian.z;

                            // Velocity, gyro bias, accel bias, gravity: no influence on measurement
                            h_x_row[6] = 0.0;
                            h_x_row[7] = 0.0;
                            h_x_row[8] = 0.0;
                            h_x_row[9] = 0.0;
                            h_x_row[10] = 0.0;
                            h_x_row[11] = 0.0;
                            h_x_row[12] = 0.0;
                            h_x_row[13] = 0.0;
                            h_x_row[14] = 0.0;
                            h_x_row[15] = 0.0;
                            h_x_row[16] = 0.0;
                            h_x_row[17] = 0.0;
                            h_row[0] = -nearest_plane[3];
                        });
                    eval.stop("jacobian").unwrap();
                },
            );
            eval_outer.write().unwrap().stop("ekf correct").unwrap();
        };

        // TODO imu_t now static
        let imu_t = Isometry3::from_parts(
            Translation3::from(self.ekf.state.offset_t_l_i),
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                &self.ekf.state.offset_r_l_i.matrix(),
            )),
        );
        let global_pose = Isometry3::from_parts(
            Translation3::from(self.ekf.state.pose.translation()),
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                &self.ekf.state.pose.rotation().matrix(),
            )),
        );

        let global_pose = global_pose * imu_t;

        // Signal Prelio it can use the ekf again
        ready_for_next
            .expect("we need ready_for_next!")
            .send(())
            .unwrap();

        {
            eval_outer.write().unwrap().start("map update").unwrap();
        }

        #[cfg(not(feature = "cuda"))]
        self.add_to_map(cloud, global_pose);
        #[cfg(feature = "cuda")]
        {
            let cloud = match cloud {
                LocatedBufferView::Ram(_) => panic!("cuda buffer in cpu task"),
                LocatedBufferView::Cuda { buffer } => buffer,
            };

            let rotmat = global_pose.rotation.to_rotation_matrix().matrix().clone();
            pelorus_cuda::add_to_map(
                &stream,
                cloud,
                &global_pose.translation.vector.into(),
                &rotmat.into(),
            );
        }
        {
            eval_outer.write().unwrap().stop("map update").unwrap();
        }

        Ok(Some(crate::CloudRegisterResult {
            pose: global_pose,
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
}

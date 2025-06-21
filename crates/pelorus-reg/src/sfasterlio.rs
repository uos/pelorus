//! An implementation of point cloud registering, tightly coupled with the IMU.
//! The used method is a combination of the EKF from S-FAST_LIO and a hashgrid-based nearest neighbor search similar to faster-lio.

use std::sync::{Arc, Mutex};

use nalgebra::*;
use ros_pointcloud2::points::{PointXYZ, PointXYZINormal};

pub use spole::IVoxNearbyType;

const PLANE_POINT_ERROR_THRESHOLD: f64 = 0.1;

pub struct SFasterLio {
    ekf: crate::sophus_ekf::Esekf,
    imu_processing: crate::imu::ImuProcessing,
    map: spole::IVoxHashMap,
    laser_point_noise_cov: f64,
    ekf_max_iterations: i8,
    k_neighbors: usize,
    neighbor_max_range: f64,
    imu_lidar_extrinsic_calibration_optimization: bool,
}

impl Default for SFasterLio {
    fn default() -> Self {
        Self::new(
            0.2,
            9.81,
            &crate::imu::ImuParameters::default(),
            0.001,
            4,
            5,
            5.0,
            false,
            Isometry3::identity(),
            IVoxNearbyType::default(),
        )
    }
}

impl SFasterLio {
    pub fn new(
        map_resolution: f64,
        g: f64,
        imu_params: &crate::imu::ImuParameters,
        laser_point_noise_cov: f64,
        ekf_max_iterations: i8,
        k_neighbors: usize,
        neighbor_max_range: f64,
        imu_lidar_extrinsic_calibration_optimization: bool,
        initial_extrinsic_calibration: Isometry3<f64>,
        knn_nearby_voxel_search_range: IVoxNearbyType,
    ) -> Self {
        let map = spole::IVoxHashMap::new(
            map_resolution as f32,
            100_000,
            knn_nearby_voxel_search_range,
        );
        let mut cov_gyr = Matrix3::identity();
        cov_gyr[(0, 0)] = imu_params.cov_gyr.x;
        cov_gyr[(1, 1)] = imu_params.cov_gyr.y;
        cov_gyr[(2, 2)] = imu_params.cov_gyr.z;

        let mut cov_acc = Matrix3::identity();
        cov_acc[(0, 0)] = imu_params.cov_acc.x;
        cov_acc[(1, 1)] = imu_params.cov_acc.y;
        cov_acc[(2, 2)] = imu_params.cov_acc.z;

        let mut cov_acc_bias = Matrix3::identity();
        cov_acc_bias[(0, 0)] = imu_params.cov_acc_bias.x;
        cov_acc_bias[(1, 1)] = imu_params.cov_acc_bias.y;
        cov_acc_bias[(2, 2)] = imu_params.cov_acc_bias.z;

        let mut cov_gyr_bias = Matrix3::identity();
        cov_gyr_bias[(0, 0)] = imu_params.cov_gyr_bias.x;
        cov_gyr_bias[(1, 1)] = imu_params.cov_gyr_bias.y;
        cov_gyr_bias[(2, 2)] = imu_params.cov_gyr_bias.z;

        Self {
            ekf: crate::sophus_ekf::Esekf::new(g),
            imu_processing: crate::imu::ImuProcessing::new(
                imu_params,
                &initial_extrinsic_calibration,
            ),
            map,
            laser_point_noise_cov,
            ekf_max_iterations,
            k_neighbors,
            neighbor_max_range,
            imu_lidar_extrinsic_calibration_optimization,
        }
    }

    fn add_to_map(&mut self, cloud: &[PointXYZ], pose: Isometry3<f64>) {
        self.map.add(cloud.iter().map(|point| {
            let p = nalgebra::Point3::new(point.x as f64, point.y as f64, point.z as f64);
            let global = pose.transform_point(&p);
            PointXYZ {
                x: global.x as f32,
                y: global.y as f32,
                z: global.z as f32,
            }
        }));
    }

    // Just for debugging! Really slow.
    pub fn get_entire_map(&self) -> Vec<PointXYZ> {
        self.map.get_entire_map()
    }

    pub fn clear_localmap(&mut self) {
        self.map.clear();
    }

    // https://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html
    fn plane_from_points(points: &[PointXYZ], error_threshold: f64) -> Option<[f64; 4]> {
        let points = points.iter().map(|p| p.xyz_f64()).collect::<Vec<_>>();
        let n = points.len();
        if n < 3 {
            return None;
        }

        let mut sum = Vector3::new(0.0, 0.0, 0.0);
        for p in points.iter() {
            sum = sum + Vector3::new(p.x, p.y, p.z);
        }
        let centroid = &sum * (1.0 / (n as f64));

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
            weighted_dir += &axis_dir * weight;
        }

        {
            let det_y = xx * zz - xz * xz;
            let axis_dir = Vector3::new(xz * yz - xy * zz, det_y, xy * xz - yz * xx);
            let mut weight = det_y * det_y;
            if weighted_dir.dot(&axis_dir) < 0.0 {
                weight = -weight;
            }
            weighted_dir += &axis_dir * weight;
        }

        {
            let det_z = xx * yy - xy * xy;
            let axis_dir = Vector3::new(xy * yz - xz * yy, xy * xz - yz * xx, det_z);
            let mut weight = det_z * det_z;
            if weighted_dir.dot(&axis_dir) < 0.0 {
                weight = -weight;
            }
            weighted_dir += &axis_dir * weight;
        }

        let normal = &weighted_dir.normalize();
        if normal.iter().all(|x| x.is_finite()) {
            let d = -normal.dot(&centroid);
            for p in points {
                let error =
                    normal.x * p.x as f64 + normal.y * p.y as f64 + normal.z * p.z as f64 + d;
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
impl crate::Registration for SFasterLio {
    fn undistort(
        &mut self,
        imu_msgs: &[crate::ImuMsg],
        last_cloud_end_time: f64,
        cloud_begin_time: f64,
        cloud_end_time: f64,
        cloud: &[prelio::PointXYZT],
        lidar_vendor: Option<prelio::LidarVendor>,
    ) -> Vec<PointXYZ> {
        // cloud.iter().map(|p| PointXYZ::new(p.x, p.y, p.z)).collect() // bypass
        self.imu_processing.undistort(
            &mut self.ekf,
            imu_msgs,
            last_cloud_end_time,
            cloud_begin_time,
            cloud_end_time,
            cloud,
            lidar_vendor,
        )
    }

    fn requested_register_and_map_cloud_type(&self) -> crate::PointXYZLocatedBufferView {
        return crate::LocatedBufferView::Ram(Vec::new());
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
        data: crate::Mode,
        ready_for_next: Option<&std::sync::mpsc::Sender<()>>,
    ) -> anyhow::Result<crate::CloudRegisterResult> {
        #[allow(unused_variables)]
        let cloud = match data {
            crate::Mode::ActiveRegistration {
                cloud,
                initial_guess: _,
            } => match cloud {
                crate::LocatedBufferView::Ram(cloud) => cloud,
                crate::LocatedBufferView::Cuda(_) => {
                    return Err(anyhow::anyhow!("Gave GPU buffer to CPU-only function"));
                }
            },
            _ => return Err(anyhow::anyhow!("Invalid mode")),
        };

        let cloud = cloud.as_slice(); // TODO use views, not copies after accuracy is done

        let normal_cloud = Arc::new(Mutex::new(None));
        let neighbors = Arc::new(Mutex::new(None));
        let first_it = Arc::new(Mutex::new(true));
        if !self.map.is_empty() {
            let map = &mut self.map;
            let k = self.k_neighbors;
            let neighbor_max_range = self.neighbor_max_range;
            let extrinsic_estimation = self.imu_lidar_extrinsic_calibration_optimization;

            let normal_cloud_in_ekf = normal_cloud.clone();
            let neighbors_cloud_in_ekf = neighbors.clone();
            self.ekf.correct(
                self.laser_point_noise_cov,
                self.ekf_max_iterations,
                extrinsic_estimation,
                move |ekf_state: &crate::sophus_ekf::StateIKFom,
                      valid: &mut bool,
                      converged: bool,
                      h: &mut DVector<f64>,
                      h_x: &mut DMatrix<f64>,
                      extrinsic_estimation| {
                    // Initial guess from ekf state from last prediciton in undistort
                    let rot = Rotation3::from_matrix(&ekf_state.rot.matrix());
                    let quat = UnitQuaternion::from_rotation_matrix(&rot);
                    let initial_guess = Isometry3::from_parts(ekf_state.pos.into(), quat);

                    let first_it_local = { first_it.lock().unwrap().clone() };

                    let mut neighbors = neighbors_cloud_in_ekf
                        .lock()
                        .unwrap()
                        .take()
                        .unwrap_or(vec![PointXYZ::default(); cloud.len() * k]);

                    // Perf: should never be larger than k because k never changes
                    if neighbors.len() < cloud.len() * k {
                        neighbors = vec![PointXYZ::default(); cloud.len() * k];
                    }

                    let planes: Vec<PointXYZINormal> = cloud
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, point_local)| {
                            let point_global =
                                initial_guess.transform_point(&point_local.xyz_f64());
                            let query_point = PointXYZ::new(
                                point_global.x as f32,
                                point_global.y as f32,
                                point_global.z as f32,
                            );

                            let curr_point_neighbors = &mut neighbors[idx * k..(idx + 1) * k];
                            debug_assert!(curr_point_neighbors.len() == k);

                            // Safety: converged is always true in the first iteration so the neighbors have been populated
                            let out_size = &mut 0;
                            if first_it_local || converged {
                                if !map.k_nearest_neighbor_query(
                                    &query_point,
                                    k,
                                    neighbor_max_range,
                                    curr_point_neighbors,
                                    out_size,
                                ) {
                                    return None;
                                }
                            }

                            let plane = Self::plane_from_points(
                                &curr_point_neighbors,
                                PLANE_POINT_ERROR_THRESHOLD,
                            )?;

                            let pd2 = plane[0] * point_global.x
                                + plane[1] * point_global.y
                                + plane[2] * point_global.z
                                + plane[3];
                            let s =
                                1.0 - 0.9 * pd2.abs() / point_local.xyz_f64().coords.norm().sqrt();

                            if s <= 0.9 {
                                return None;
                            }

                            Some(PointXYZINormal::new(
                                point_local.x,
                                point_local.y,
                                point_local.z,
                                pd2 as f32,
                                plane[0] as f32,
                                plane[1] as f32,
                                plane[2] as f32,
                            ))
                        })
                        .collect::<Vec<_>>();

                    // put neighbors back
                    neighbors_cloud_in_ekf.lock().unwrap().replace(neighbors);

                    *first_it.lock().unwrap() = false;

                    if planes.is_empty() {
                        *valid = false;
                        return;
                    }

                    if planes.len() > h.nrows() {
                        *h = DVector::<f64>::zeros(planes.len());
                    }

                    if planes.len() > h_x.nrows() {
                        *h_x = DMatrix::<f64>::zeros(planes.len(), 12);
                    }

                    // TODO (AS REF) if converge, do use old planes/neighbors since it is likely to be the same

                    planes
                        .iter()
                        .zip(h_x.row_iter_mut())
                        .zip(h.row_iter_mut())
                        .for_each(|((normal_point, mut h_x_row), mut h_row)| {
                            let point_local = normal_point.xyz_f64();
                            let nearest_plane = [
                                normal_point.normal_x as f64,
                                normal_point.normal_y as f64,
                                normal_point.normal_z as f64,
                                normal_point.intensity as f64,
                            ];
                            let v = ekf_state.offset_r_l_i.matrix() * point_local
                                + ekf_state.offset_t_l_i;

                            let point_crossmat =
                                Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0);

                            let norm =
                                Vector3::new(nearest_plane[0], nearest_plane[1], nearest_plane[2]);

                            // Jacobian w.r.t extrinsic translation
                            let c = ekf_state.rot.matrix().transpose() * norm;

                            // Jacobian w.r.t IMU rotation
                            let a = point_crossmat * c;

                            if extrinsic_estimation {
                                // Jacobian w.r.t extrinsic rotation
                                let b = point_crossmat * (ekf_state.rot.matrix().transpose() * c);

                                // Position Jacobian
                                h_x_row[0] = norm.x;
                                h_x_row[1] = norm.y;
                                h_x_row[2] = norm.z;

                                h_x_row[3] = a.x;
                                h_x_row[4] = a.y;
                                h_x_row[5] = a.z;

                                h_x_row[6] = b.x;
                                h_x_row[7] = b.y;
                                h_x_row[8] = b.z;

                                h_x_row[9] = c.x;
                                h_x_row[10] = c.y;
                                h_x_row[11] = c.z;
                            } else {
                                // Position Jacobian
                                h_x_row[0] = norm.x;
                                h_x_row[1] = norm.y;
                                h_x_row[2] = norm.z;

                                h_x_row[3] = a.x;
                                h_x_row[4] = a.y;
                                h_x_row[5] = a.z;

                                h_x_row[6] = 0.0;
                                h_x_row[7] = 0.0;
                                h_x_row[8] = 0.0;
                                h_x_row[9] = 0.0;
                                h_x_row[10] = 0.0;
                                h_x_row[11] = 0.0;
                            }

                            h_row[0] = -nearest_plane[3];
                        });

                    let mut glob = normal_cloud_in_ekf.lock().unwrap();
                    *glob = Some(planes);
                },
            );
        };

        let pos_lidar =
            self.ekf.state.pos + self.ekf.state.rot.matrix() * self.ekf.state.offset_t_l_i;

        let global_pose = Isometry3::from_parts(
            Translation3::from(pos_lidar),
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                &self.ekf.state.rot.matrix(),
            )),
        );

        // Signal Prelio it can use the ekf again
        ready_for_next
            .expect("we need ready_for_next!")
            .send(())
            .unwrap();

        //let start_stamp = std::time::Instant::now();
        self.add_to_map(&cloud, global_pose);
        //dbg!(start_stamp.elapsed());

        let out_cloud = normal_cloud.lock().unwrap().take();

        Ok(crate::CloudRegisterResult {
            pose: global_pose,
            avg_iterations: None,
            registered_cloud: out_cloud,
        })
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

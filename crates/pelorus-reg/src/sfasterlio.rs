//! An implementation of point cloud registering, tightly coupled with the IMU.
//! The used method is a combination of the EKF from S-FAST_LIO and nearest neighbor search from FASTER-LIO using hashmaps.

use nalgebra::*;
use ros_pointcloud2::points::PointXYZ;

pub use spole::IVoxNearbyType;

pub struct SFasterLio {
    ekf: crate::sophus_ekf::Esekf,
    imu_processing: crate::imu::ImuProcessing,
    map: spole::IVoxHashMap,
    laser_point_cov: f64,
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
            &IVoxNearbyType::default(),
        )
    }
}

impl SFasterLio {
    pub fn new(
        map_resolution: f64,
        g: f64,
        imu_params: &crate::imu::ImuParameters,
        laser_point_cov: f64,
        ekf_max_iterations: i8,
        k_neighbors: usize,
        neighbor_max_range: f64,
        imu_lidar_extrinsic_calibration_optimization: bool,
        initial_extrinsic_calibration: Isometry3<f64>,
        knn_nearby_voxel_search_range: &IVoxNearbyType,
    ) -> Self {
        let map = spole::IVoxHashMap::new(
            map_resolution as f32,
            100_000,
            &knn_nearby_voxel_search_range,
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
            laser_point_cov,
            ekf_max_iterations,
            k_neighbors,
            neighbor_max_range,
            imu_lidar_extrinsic_calibration_optimization,
        }
    }

    fn add_to_map(&mut self, cloud: &[PointXYZ], pose: Isometry3<f64>) {
        self.map.add(cloud.iter().map(|point| {
            let transformed = nalgebra::Point3::new(point.x as f64, point.y as f64, point.z as f64);
            let global = pose * transformed;
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
    fn plane_from_points(points: &[PointXYZ]) -> Option<[f64; 4]> {
        let n = points.len();
        if n < 3 {
            return None;
        }

        let points_vec = points
            .iter()
            .map(|p| Vector3::new(p.x as f64, p.y as f64, p.z as f64));

        let mut sum = Vector3::new(0.0, 0.0, 0.0);
        for p in points_vec.clone() {
            sum = &sum + p;
        }
        let centroid = &sum * (1.0 / (n as f64));

        // Calculate full 3x3 covariance matrix, excluding symmetries:
        let mut xx = 0.0;
        let mut xy = 0.0;
        let mut xz = 0.0;
        let mut yy = 0.0;
        let mut yz = 0.0;
        let mut zz = 0.0;

        for p in points_vec {
            let r = &p - centroid;
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
            Some([normal.x, normal.y, normal.z, d])
        } else {
            None
        }
    }
}

impl crate::Registration for SFasterLio {
    fn register_and_map(
        &mut self,
        cloud: &[PointXYZ],
        initial_guess: &Isometry3<f64>,
    ) -> anyhow::Result<crate::CloudRegisterResult> {
        if !self.map.is_empty() {
            let map = &mut self.map;
            let k = self.k_neighbors;
            let neighbor_max_range = self.neighbor_max_range;
            let extrinsic_estimation = self.imu_lidar_extrinsic_calibration_optimization;
            self.ekf.update(
                self.laser_point_cov,
                self.ekf_max_iterations,
                extrinsic_estimation,
                move |ekf_state: &crate::sophus_ekf::StateIKFom,
                      valid: &mut bool,
                      converged: bool,
                      h: &mut DVector<f64>,
                      h_x: &mut DMatrix<f64>,
                      extrinsic_estimation| {
                    if !converged {
                        dbg!("not converged");
                        return;
                    }

                    let initial_guess = Isometry3::identity(); // TODO rm

                    let planes: Vec<(Vector3<f64>, [f64; 4])> = cloud
                        .into_iter()
                        .filter_map(|point| {
                            let p_t = initial_guess.transform_point(&Point3::new(
                                point.x as f64,
                                point.y as f64,
                                point.z as f64,
                            ));
                            let query_point =
                                PointXYZ::new(p_t.x as f32, p_t.y as f32, p_t.z as f32);
                            Self::plane_from_points(&map.k_nearest_neighbor_query(
                                &query_point,
                                k,
                                neighbor_max_range,
                            ))
                            .map(|plane| {
                                (
                                    Vector3::new(point.x as f64, point.y as f64, point.z as f64),
                                    plane,
                                )
                            })
                        })
                        .collect();

                    dbg!(planes.len());

                    if planes.is_empty() {
                        *valid = false;
                        return;
                    }

                    if planes.len() > h.nrows() || planes.len() > h_x.nrows() {
                        *h = DVector::<f64>::zeros(planes.len());
                        *h_x = DMatrix::<f64>::zeros(planes.len(), 12);
                    }

                    planes
                        .into_iter()
                        .zip(h_x.row_iter_mut())
                        .zip(h.row_iter_mut())
                        .for_each(|(((point, nearest_plane), mut h_x_row), mut h_row)| {
                            let v =
                                ekf_state.offset_r_l_i.matrix() * point + ekf_state.offset_t_l_i;

                            let point_crossmat =
                                Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0);

                            let norm =
                                Vector3::new(nearest_plane[0], nearest_plane[1], nearest_plane[2]);

                            let c = ekf_state.rot.matrix().conjugate() * norm;
                            let a = point_crossmat * c;

                            if extrinsic_estimation {
                                let b = point_crossmat * (ekf_state.rot.matrix().conjugate() * c);

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

        dbg!(global_pose);

        // localmap cache
        // TODO WAAAAAAY to sloow > 10-50ms
        self.map.remove_cold_voxels(600);

        //let start_stamp = std::time::Instant::now();
        self.add_to_map(cloud, global_pose);
        //dbg!(start_stamp.elapsed());

        dbg!(self.map.get_entire_map().len());

        Ok(crate::CloudRegisterResult {
            pose: global_pose,
            avg_iterations: None,
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

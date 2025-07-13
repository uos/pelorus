//! An implementation of point cloud registering, tightly coupled with the IMU.
//! The used method is a combination of the EKF from S-FAST_LIO and a hashgrid-based nearest neighbor search similar to faster-lio.

use std::sync::{Arc, Mutex};

use nalgebra::*;
use prelio::{LocatedBufferView, timing::RuntimeEvaluator};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use ros_pointcloud2::points::{PointXYZ, PointXYZINormal};

use sophus_lie::traits::IsTranslationProductGroup;
pub use spole::IVoxNearbyType;

const PLANE_POINT_ERROR_THRESHOLD: f64 = 0.1;

pub struct SFasterLio {
    ekf: crate::sophus_ekf::Esekf,
    imu_processing: crate::imu::ImuProcessing,
    pub map: spole::IVoxHashMap,
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
            10_000,
            9.81,
            &crate::imu::ImuParameters::default(),
            0.001,
            4,
            5,
            5.0,
            false,
            Isometry3::identity(),
            IVoxNearbyType::default(),
            3.0,
        )
    }
}

impl SFasterLio {
    pub fn new(
        map_resolution: f64,
        map_capacity: usize,
        g: f64,
        imu_params: &crate::imu::ImuParameters,
        laser_point_noise_cov: f64,
        ekf_max_iterations: i8,
        k_neighbors: usize,
        neighbor_max_range: f64,
        imu_lidar_extrinsic_calibration_optimization: bool,
        initial_extrinsic_calibration: Isometry3<f64>,
        knn_nearby_voxel_search_range: IVoxNearbyType,
        update_cov_sensitivity: f32,
    ) -> Self {
        let map = spole::IVoxHashMap::new(
            map_resolution as f32,
            map_capacity,
            knn_nearby_voxel_search_range,
            update_cov_sensitivity,
        );

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

    fn requested_register_and_map_cloud_type(&self) -> LocatedBufferView<PointXYZ> {
        return crate::LocatedBufferView::Ram(Vec::new());
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
        cloud: &LocatedBufferView<PointXYZ>,
        ready_for_next: Option<&std::sync::mpsc::Sender<()>>,
        eval: &mut RuntimeEvaluator,
    ) -> anyhow::Result<crate::CloudRegisterResult> {
        let cloud = match cloud {
            LocatedBufferView::Ram(items) => items,
            LocatedBufferView::Cuda(_) => todo!(),
        };
        let neighbors = Arc::new(Mutex::new(None));
        let neighbor_sizes = Arc::new(Mutex::new(None));
        let first_it = Arc::new(Mutex::new(true));
        let eval_outer = std::sync::Arc::new(std::sync::RwLock::new(eval));
        if !self.map.is_empty() {
            let map = &mut self.map;
            let k = self.k_neighbors;
            let neighbor_max_range = self.neighbor_max_range as f32;
            let extrinsic_estimation = self.imu_lidar_extrinsic_calibration_optimization;

            let neighbors_cloud_in_ekf = neighbors.clone();
            let neighbor_sizes_in_ekf = neighbor_sizes.clone();
            let cloud_len_alloc = (1.2 * cloud.len() as f32).ceil() as usize;
            {
                eval_outer.write().unwrap().start("ekf correct").unwrap();
            }
            let eval_cb = eval_outer.clone();
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
                    // let initial_guess = Isometry3::from_parts(
                    //     ekf_state.pos.into(),
                    //     UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                    //         &ekf_state.rot.matrix(),
                    //     )),
                    // );

                    let first_it_local = { first_it.lock().unwrap().clone() };

                    let mut neighbors = neighbors_cloud_in_ekf
                        .lock()
                        .unwrap()
                        .take()
                        .unwrap_or(vec![PointXYZ::default(); cloud_len_alloc * k]);
                    let mut neighbor_sizes = neighbor_sizes_in_ekf
                        .lock()
                        .unwrap()
                        .take()
                        .unwrap_or(vec![0usize; cloud_len_alloc]);

                    // Perf: should never be larger than k because k never changes. add 20% extra room to avoid triggering with small upward changes.
                    if neighbors.len() < cloud.len() * k {
                        neighbors = vec![PointXYZ::default(); cloud_len_alloc * k];
                    }

                    let mut eval = eval_cb.write().unwrap();

                    let iter = {
                        #[cfg(feature = "parallel")]
                        {
                            cloud
                                .par_iter()
                                .zip(neighbor_sizes.par_iter_mut())
                                .zip(neighbors.par_chunks_exact_mut(k))
                        }
                        #[cfg(not(feature = "parallel"))]
                        {
                            cloud
                                .iter()
                                .zip(neighbor_sizes.iter_mut())
                                .zip(neighbors.chunks_exact_mut(k))
                        }
                    };

                    let planes = {
                        eval.start("planes").unwrap();
                        let planes: Vec<PointXYZINormal> = iter
                            .filter_map(|((point_local, neighbor_size), curr_point_neighbors)| {
                                // Global in IMU frame
                                let point_imu = ekf_state.offset_r_l_i.matrix()
                                    * point_local.xyz_f64()
                                    + ekf_state.offset_t_l_i;
                                let point_global = ekf_state.pose.transform(&point_imu.coords);

                                let query_point = PointXYZ::new(
                                    point_global.x as f32,
                                    point_global.y as f32,
                                    point_global.z as f32,
                                );

                                if first_it_local || converged {
                                    if !map.k_nearest_neighbor_query(
                                        &query_point,
                                        k,
                                        neighbor_max_range,
                                        curr_point_neighbors,
                                        neighbor_size,
                                    ) {
                                        return None;
                                    }
                                }

                                let plane = Self::plane_from_points(
                                    &curr_point_neighbors[..*neighbor_size],
                                    PLANE_POINT_ERROR_THRESHOLD,
                                )?;

                                let pd2 = plane[0] * point_global.x
                                    + plane[1] * point_global.y
                                    + plane[2] * point_global.z
                                    + plane[3];
                                let s = 1.0
                                    - 0.9 * pd2.abs() / point_local.xyz_f64().coords.norm().sqrt();

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
                        eval.stop("planes").unwrap();
                        planes
                    };

                    // put neighbors back
                    {
                        neighbors_cloud_in_ekf.lock().unwrap().replace(neighbors);
                        neighbor_sizes_in_ekf
                            .lock()
                            .unwrap()
                            .replace(neighbor_sizes);
                    }

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

                    eval.start("jacobian").unwrap();
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
                            let c = ekf_state.pose.rotation().matrix().transpose() * norm;

                            // Jacobian w.r.t IMU rotation
                            let a = point_crossmat * c;

                            if extrinsic_estimation {
                                // Jacobian w.r.t extrinsic rotation
                                let b = point_crossmat
                                    * (ekf_state.pose.rotation().matrix().transpose() * c);

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
                    eval.stop("jacobian").unwrap();
                },
            );
            eval_outer.write().unwrap().stop("ekf correct").unwrap();
        };

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
        self.add_to_map(&cloud, global_pose);
        {
            eval_outer.write().unwrap().stop("map update").unwrap();
        }

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

use std::vec;

use log::warn;
use nalgebra::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use ros_pointcloud2::points::PointXYZ;
use serde::{Deserialize, Serialize};
use sophus_lie::traits::IsTranslationProductGroup;

use crate::sophus_ekf::Esekf;

pub type SE3 = sophus_lie::Isometry3<f64, 1>;

// const MAX_INI_COUNT: usize = 100; # unused, using time instead
const COV_IMU_RAW_DEFAULT: f64 = 0.001;
const COV_IMU_BIAS_DEFAULT: f64 = 0.0001;

// Initial state covariance. Changes with prediction (increase) and correction (decrease)
const COV1: f64 = 0.00001;
const COV2: f64 = 0.0001;
const COV3: f64 = 0.001;

pub struct ImuProcessing {
    ekf_inited: bool,
    last_imu: crate::ImuMsg,
    cov_acc: Vector3<f64>,
    cov_acc_bias: Vector3<f64>,
    cov_gyr: Vector3<f64>,
    cov_gyr_bias: Vector3<f64>,
    mean_acc: Vector3<f64>,
    mean_gyr: Vector3<f64>,
    lidar_t_wrt_imu: Vector3<f64>,
    lidar_r_wrt_imu: Matrix3<f64>,
    angvel_last: Vector3<f64>,
    acc_s_last: Vector3<f64>,
    imu_state: crate::sophus_ekf::StateIKFom,
    pub ekf_cov_gyr_bias: Matrix3<f64>,
    pub ekf_cov_acc_bias: Matrix3<f64>,
    pub ekf_cov_acc: Matrix3<f64>,
    pub ekf_cov_gyr: Matrix3<f64>,
}

impl Default for ImuProcessing {
    fn default() -> Self {
        Self {
            ekf_inited: false,
            cov_acc: Vector3::new(
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
            ),
            cov_gyr: Vector3::new(
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
            ),
            last_imu: crate::ImuMsg::default(),
            mean_acc: Vector3::zeros(),
            mean_gyr: Vector3::zeros(),
            lidar_t_wrt_imu: Vector3::zeros(),
            lidar_r_wrt_imu: Matrix3::identity(),
            ekf_cov_gyr_bias: Matrix3::zeros(),
            ekf_cov_acc_bias: Matrix3::zeros(),
            ekf_cov_acc: Matrix3::zeros(),
            ekf_cov_gyr: Matrix3::zeros(),
            cov_acc_bias: Vector3::new(
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
            ),
            cov_gyr_bias: Vector3::new(
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
            ),
            angvel_last: Vector3::zeros(),
            acc_s_last: Vector3::zeros(),
            imu_state: crate::sophus_ekf::StateIKFom::default(),
        }
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub struct ImuParameters {
    pub cov_acc: Vector3<f64>,
    pub cov_gyr: Vector3<f64>,
    pub cov_gyr_bias: Vector3<f64>,
    pub cov_acc_bias: Vector3<f64>,
    pub initial_translation_lidar_to_imu: Vector3<f64>,
    pub initial_rotation_lidar_to_imu: Matrix3<f64>,
}

impl Default for ImuParameters {
    fn default() -> Self {
        Self {
            cov_acc: Vector3::new(
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
            ),
            cov_gyr: Vector3::new(
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
            ),
            cov_gyr_bias: Vector3::new(
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
            ),
            cov_acc_bias: Vector3::new(
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
            ),
            initial_translation_lidar_to_imu: Vector3::default(),
            initial_rotation_lidar_to_imu: Matrix3::identity(),
        }
    }
}

#[derive(Clone, Debug, Copy)]
struct Pose6D {
    offset_time: f64,
    acc: nalgebra::Vector3<f64>,
    gyr: nalgebra::Vector3<f64>,
    vel: nalgebra::Vector3<f64>,
    pose: SE3,
}

impl Pose6D {
    pub fn new(
        offset_time: f64,
        acc: nalgebra::Vector3<f64>,
        gyr: nalgebra::Vector3<f64>,
        vel: nalgebra::Vector3<f64>,
        pose: SE3,
    ) -> Self {
        Self {
            offset_time,
            acc,
            gyr,
            vel,
            pose,
        }
    }
}

impl ImuProcessing {
    pub fn new(params: &ImuParameters, lidar_imu_extr: &Isometry3<f64>) -> Self {
        let mut s = Self::default();

        s.ekf_cov_acc[(0, 0)] = params.cov_acc[0];
        s.ekf_cov_acc[(1, 1)] = params.cov_acc[1];
        s.ekf_cov_acc[(2, 2)] = params.cov_acc[2];

        s.ekf_cov_gyr[(0, 0)] = params.cov_gyr[0];
        s.ekf_cov_gyr[(1, 1)] = params.cov_gyr[1];
        s.ekf_cov_gyr[(2, 2)] = params.cov_gyr[2];

        s.ekf_cov_gyr_bias[(0, 0)] = params.cov_gyr_bias[0];
        s.ekf_cov_gyr_bias[(1, 1)] = params.cov_gyr_bias[1];
        s.ekf_cov_gyr_bias[(2, 2)] = params.cov_gyr_bias[2];

        s.ekf_cov_acc_bias[(0, 0)] = params.cov_acc_bias[0];
        s.ekf_cov_acc_bias[(1, 1)] = params.cov_acc_bias[1];
        s.ekf_cov_acc_bias[(2, 2)] = params.cov_acc_bias[2];

        s.cov_acc = params.cov_acc;
        s.cov_gyr = params.cov_gyr;
        s.cov_gyr_bias = params.cov_gyr_bias;
        s.cov_acc_bias = params.cov_acc_bias;

        s.lidar_r_wrt_imu = *lidar_imu_extr.rotation.to_rotation_matrix().matrix();
        s.lidar_t_wrt_imu = lidar_imu_extr.translation.vector;

        s
    }

    /*pub fn set_extrinsic(&mut self, t: &SMatrix<f64, 4, 4>) {
        self.lidar_t_wrt_imu = Vector3::new(t[(0, 3)], t[(1, 3)], t[(2, 3)]);
        self.lidar_r_wrt_imu = Matrix3::new(
            t[(0, 0)],
            t[(0, 1)],
            t[(0, 2)],
            t[(1, 0)],
            t[(1, 1)],
            t[(1, 2)],
            t[(2, 0)],
            t[(2, 1)],
            t[(2, 2)],
        );
    }*/

    pub fn initialize(
        &mut self,
        msgs: &[crate::ImuMsg],
        ekf: &mut crate::sophus_ekf::Esekf,
    ) -> Vector3<f64> {
        let mut n = 1.0;
        self.last_imu = crate::ImuMsg::default();
        self.mean_acc = msgs[0].linear_acceleration;
        self.mean_gyr = msgs[0].angular_velocity;

        for msg in msgs {
            let cur_acc = msg.linear_acceleration;
            let cur_gyr = msg.angular_velocity;

            self.mean_acc += (cur_acc - self.mean_acc) / (n as f64);
            self.mean_gyr += (cur_gyr - self.mean_gyr) / (n as f64);

            self.cov_acc = self.cov_acc * (n - 1.0) / n
                + (cur_acc - self.mean_acc).component_mul(&(cur_acc - self.mean_acc)) * (n - 1.0)
                    / (n * n);
            self.cov_gyr = self.cov_gyr * (n - 1.0) / n
                + (cur_gyr - self.mean_gyr).component_mul(&(cur_gyr - self.mean_gyr)) * (n - 1.0)
                    / (n * n);

            n += 1.0;
        }

        ekf.state.grav = -self.mean_acc / self.mean_acc.norm() * 9.81; // TODO PARAM
        ekf.state.bg = self.mean_gyr;
        ekf.state.offset_t_l_i = self.lidar_t_wrt_imu;
        ekf.state.offset_r_l_i =
            sophus_lie::Rotation3::try_from_mat(&self.lidar_r_wrt_imu).unwrap();

        ekf.cov[(0, 0)] = 1.0; // pos
        ekf.cov[(1, 1)] = 1.0;
        ekf.cov[(2, 2)] = 1.0;

        ekf.cov[(3, 3)] = 1.0; // rot
        ekf.cov[(4, 4)] = 1.0;
        ekf.cov[(5, 5)] = 1.0;

        ekf.cov[(6, 6)] = COV1; // offset_r_l_i
        ekf.cov[(7, 7)] = COV1;
        ekf.cov[(8, 8)] = COV1;

        ekf.cov[(9, 9)] = COV1; // offset_t_l_i
        ekf.cov[(10, 10)] = COV1;
        ekf.cov[(11, 11)] = COV1;

        ekf.cov[(12, 12)] = 1.0; // vel
        ekf.cov[(13, 13)] = 1.0;
        ekf.cov[(14, 14)] = 1.0;

        ekf.cov[(15, 15)] = COV2; // bg
        ekf.cov[(16, 16)] = COV2;
        ekf.cov[(17, 17)] = COV2;

        ekf.cov[(18, 18)] = COV3; // ba
        ekf.cov[(19, 19)] = COV3;
        ekf.cov[(20, 20)] = COV3;

        ekf.cov[(21, 21)] = COV1; // grav
        ekf.cov[(22, 22)] = COV1;
        ekf.cov[(23, 23)] = COV1;

        self.last_imu = msgs[msgs.len() - 1].clone();

        self.ekf_inited = true;

        self.mean_acc
    }

    pub fn undistort(
        &mut self,
        ekf: &mut Esekf,
        imu_msgs: &[crate::ImuMsg],
        last_cloud_end_time: f64,
        cloud_begin_time: f64,
        mut cloud_end_time: f64,
        cloud: &[prelio::PointXYZT],
        lidar_vendor: Option<prelio::LidarVendor>,
    ) -> Vec<PointXYZ> {
        assert!(!imu_msgs.is_empty());
        assert!(last_cloud_end_time > 0.0);
        assert!(cloud_begin_time > 0.0);
        assert!(cloud_end_time > 0.0);

        if !matches!(lidar_vendor, Some(prelio::LidarVendor::Robosense)) {
            cloud_end_time += cloud_begin_time;
        }

        // dbg!(self.lidar_t_wrt_imu, self.lidar_r_wrt_imu);
        const GRAVITY: f64 = 9.81; // TODO PARAM

        let mean_acc = self.mean_acc.norm();

        self.imu_state = ekf.state.clone();

        let mut imu_poses = vec![Pose6D::new(
            0.0,
            self.acc_s_last,
            self.angvel_last,
            ekf.state.vel,
            ekf.state.pose.clone(),
        )];

        let mut angvel_avr_x = 0.0;
        let mut angvel_avr_y = 0.0;
        let mut angvel_avr_z = 0.0;

        let mut acc_avr_x = 0.0;
        let mut acc_avr_y = 0.0;
        let mut acc_avr_z = 0.0;

        let mut all_imus = std::iter::once(&self.last_imu).chain(imu_msgs.iter());

        // Handle the (unlikely) case of an empty iterator gracefully.
        let Some(mut head) = all_imus.next() else {
            // This will not be reached due to the assert! above, but is robust code.
            return vec![];
        };

        while let Some(tail) = all_imus.next() {
            let head_stamp = head.timestamp_sec;
            let tail_stamp = tail.timestamp_sec;

            if head_stamp == tail_stamp {
                warn!("Found the same stamps from IMU, skipping a msg");
                head = tail;
                continue;
            }

            assert!(head_stamp < tail_stamp);

            angvel_avr_x = (head.angular_velocity[0] + tail.angular_velocity[0]) * 0.5;
            angvel_avr_y = (head.angular_velocity[1] + tail.angular_velocity[1]) * 0.5;
            angvel_avr_z = (head.angular_velocity[2] + tail.angular_velocity[2]) * 0.5;

            acc_avr_x = (head.linear_acceleration[0] + tail.linear_acceleration[0]) * 0.5 * GRAVITY
                / mean_acc;
            acc_avr_y = (head.linear_acceleration[1] + tail.linear_acceleration[1]) * 0.5 * GRAVITY
                / mean_acc;
            acc_avr_z = (head.linear_acceleration[2] + tail.linear_acceleration[2]) * 0.5 * GRAVITY
                / mean_acc;

            let dt = if head_stamp < last_cloud_end_time {
                tail_stamp - last_cloud_end_time
            } else {
                tail_stamp - head_stamp
            };

            assert!(dt > 0.0); // TODO does this need > only? predict 0 ok? first ever dt is 0

            let avr_x = [acc_avr_x, acc_avr_y, acc_avr_z];
            let angvel_avr = [angvel_avr_x, angvel_avr_y, angvel_avr_z];
            _ = self.predict(ekf, avr_x, angvel_avr, dt);
            self.imu_state = ekf.state.clone();

            self.angvel_last = tail.angular_velocity.xyz() - self.imu_state.bg;
            self.acc_s_last = self.imu_state.pose.rotation().matrix()
                * ((tail.linear_acceleration.xyz() * GRAVITY / mean_acc) - self.imu_state.ba)
                + self.imu_state.grav;
            let offs_t = tail_stamp - cloud_begin_time;
            imu_poses.push(Pose6D::new(
                offs_t,
                self.acc_s_last,
                self.angvel_last,
                self.imu_state.vel,
                self.imu_state.pose.clone(),
            ));

            head = tail;
        }

        // Integrate to end of cloud
        self.last_imu = imu_msgs[imu_msgs.len() - 1].clone();
        let dt = cloud_end_time - self.last_imu.timestamp_sec;
        assert!(dt > 0.0);

        let avr_x = [acc_avr_x, acc_avr_y, acc_avr_z];
        let angvel_avr = [angvel_avr_x, angvel_avr_y, angvel_avr_z];
        let _ = self.predict(ekf, avr_x, angvel_avr, dt);
        self.imu_state = ekf.state.clone();

        // TODO perf this alloc could be optimized out by changing the undistort closure return to be a mut& Vec. then it could be resized here to avoid new allocs for the cloud
        let mut out_cloud = vec![PointXYZ::default(); cloud.len()];

        #[cfg(not(feature = "parallel"))]
        {
            let mut it_pcl = cloud.len();
            for i in (1..imu_poses.len()).rev() {
                let head = &imu_poses[i - 1];
                let tail = &imu_poses[i];

                while it_pcl > 0 && cloud[it_pcl - 1].timestamp > head.offset_time {
                    it_pcl -= 1;
                    let pcl_point = &cloud[it_pcl];

                    let mut dt = pcl_point.timestamp - head.offset_time;
                    if dt < 0.0 {
                        // This can happen due to floating point inaccuracies if the point is very close
                        // to the head time. We can treat it as having zero offset.
                        warn!(
                            "Point timestamp slightly before IMU head offset_time, clamping dt to 0."
                        );
                        dt = 0.0;
                    }

                    let r_i = head
                        .pose
                        .rotation()
                        .group_mul(&crate::sophus_ekf::SO3::exp(&(tail.gyr * dt)))
                        .matrix();
                    let t_ei = head.pose.translation() + head.vel * dt + 0.5 * tail.acc * dt * dt
                        - self.imu_state.pose.translation();

                    let p_imu_frame = self.imu_state.offset_r_l_i.matrix()
                        * Vector3::new(pcl_point.x as f64, pcl_point.y as f64, pcl_point.z as f64)
                        + self.imu_state.offset_t_l_i;
                    let p_compensate = self.imu_state.offset_r_l_i.matrix().transpose()
                        * (self.imu_state.pose.rotation().matrix().transpose()
                            * (r_i * p_imu_frame + t_ei)
                            - self.imu_state.offset_t_l_i);

                    out_cloud[it_pcl].x = p_compensate.x as f32;
                    out_cloud[it_pcl].y = p_compensate.y as f32;
                    out_cloud[it_pcl].z = p_compensate.z as f32;
                }
            }
        }

        #[cfg(feature = "parallel")]
        out_cloud
            .par_iter_mut()
            .zip(cloud.par_iter())
            .for_each(|(out_point, pcl_point)| {
                // For each point, find the correct IMU pose interval.
                // binary_search_by finds the index of the first pose with a timestamp >= the point's timestamp.
                let tail_idx = imu_poses
                    .binary_search_by(|p| {
                        p.offset_time
                            .partial_cmp(&pcl_point.timestamp)
                            .unwrap_or(std::cmp::Ordering::Less)
                    })
                    .unwrap_or_else(|e| e);

                if tail_idx == 0 || tail_idx >= imu_poses.len() {
                    // point is before the first pose or after the last.
                    return;
                }
                let head_idx = tail_idx - 1;

                let head = &imu_poses[head_idx];
                let tail = &imu_poses[tail_idx];

                let mut dt = pcl_point.timestamp - head.offset_time;
                if dt < 0.0 {
                    dt = 0.0;
                }

                let r_i = head
                    .pose
                    .rotation()
                    .group_mul(&crate::sophus_ekf::SO3::exp(&(tail.gyr * dt)))
                    .matrix();

                let t_ei = head.pose.translation() + head.vel * dt + 0.5 * tail.acc * dt * dt
                    - self.imu_state.pose.translation();

                let p_imu_frame = self.imu_state.offset_r_l_i.matrix()
                    * Vector3::new(pcl_point.x as f64, pcl_point.y as f64, pcl_point.z as f64)
                    + self.imu_state.offset_t_l_i;

                let p_compensate = self.imu_state.offset_r_l_i.matrix().transpose()
                    * (self.imu_state.pose.rotation().matrix().transpose()
                        * (r_i * p_imu_frame + t_ei)
                        - self.imu_state.offset_t_l_i);

                out_point.x = p_compensate.x as f32;
                out_point.y = p_compensate.y as f32;
                out_point.z = p_compensate.z as f32;
            });

        out_cloud
    }

    pub fn predict(
        &self,
        ekf: &mut Esekf,
        acc: [f64; 3],
        gyro: [f64; 3],
        dt: f64,
    ) -> anyhow::Result<crate::ImuEkfPrediction> {
        let inp = crate::sophus_ekf::InputIKFom {
            acc: Vector3::new(acc[0], acc[1], acc[2]),
            gyro: Vector3::new(gyro[0], gyro[1], gyro[2]),
        };

        let q = {
            let mut q = SMatrix::<f64, 12, 12>::zeros();
            q.fixed_view_mut::<3, 3>(0, 0).copy_from(&self.ekf_cov_gyr);
            q.fixed_view_mut::<3, 3>(3, 3).copy_from(&self.ekf_cov_acc);
            q.fixed_view_mut::<3, 3>(6, 6)
                .copy_from(&self.ekf_cov_gyr_bias);
            q.fixed_view_mut::<3, 3>(9, 9)
                .copy_from(&self.ekf_cov_acc_bias);

            q
        };

        if !self.ekf_inited {
            anyhow::bail!("EKF not initialized");
        }

        ekf.predict(dt, &q, &inp);

        Ok(crate::ImuEkfPrediction {
            pose: Isometry3::from_parts(
                Translation3::from(ekf.state.pose.translation()),
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                    &ekf.state.pose.rotation().matrix(),
                )),
            ),
            linear_velocity: ekf.state.vel,
            lidar_imu_extrinsic: Isometry3::from_parts(
                Translation3::from(ekf.state.offset_t_l_i),
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                    &ekf.state.offset_r_l_i.matrix(),
                )),
            ),
        })
    }
}

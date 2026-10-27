use alloc::vec;

use log::warn;
use nalgebra::*;
use pelorus_core::log_debouncer::should_log_once_default;
use pelorus_core::{
    COV_IMU_BIAS_DEFAULT, COV_IMU_RAW_DEFAULT, ImuMsg, ImuParameters, KinematicModel, LidarVendor,
    PointXYZIT, adaptive_execution::ExecutionMode, timestamp_to_secs,
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::eqf::Eqf;
pub type SE3 = crate::eqf::SE3;
pub type SO3 = crate::eqf::SO3;

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
    lidar_to_imu_t: Vector3<f64>,
    lidar_to_imu_r: Matrix3<f64>,
    lidar_to_base_t: Vector3<f64>,
    lidar_to_base_r: Matrix3<f64>,
    pub angvel_last: Vector3<f64>,
    acc_s_last: Vector3<f64>,
    imu_state: crate::eqf::StateIKFom,
    pub ekf_cov_gyr_bias: Matrix3<f64>,
    pub ekf_cov_acc_bias: Matrix3<f64>,
    pub ekf_cov_acc: Matrix3<f64>,
    pub ekf_cov_gyr: Matrix3<f64>,
    pub wheel_odom_cov_override: Option<f64>,
}

#[cfg(feature = "threads")]
#[derive(Clone)]
pub(crate) struct ImuPredictionContext {
    pub last_imu: crate::ImuMsg,
    pub mean_acc: Vector3<f64>,
    pub angvel_last: Vector3<f64>,
    pub ekf_inited: bool,
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
            lidar_to_imu_t: Vector3::zeros(),
            lidar_to_imu_r: Matrix3::identity(),
            lidar_to_base_t: Vector3::zeros(),
            lidar_to_base_r: Matrix3::identity(),
            ekf_cov_gyr_bias: Matrix3::zeros(),
            ekf_cov_acc_bias: Matrix3::zeros(),
            ekf_cov_acc: Matrix3::zeros(),
            ekf_cov_gyr: Matrix3::zeros(),
            wheel_odom_cov_override: None,
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
            imu_state: crate::eqf::StateIKFom::default(),
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
    #[cfg(feature = "threads")]
    pub(crate) fn prediction_context(&self) -> ImuPredictionContext {
        ImuPredictionContext {
            last_imu: self.last_imu.clone(),
            mean_acc: self.mean_acc,
            angvel_last: self.angvel_last,
            ekf_inited: self.ekf_inited,
        }
    }

    #[cfg(feature = "threads")]
    pub(crate) fn restore_prediction_context(&mut self, context: &ImuPredictionContext) {
        self.last_imu = context.last_imu.clone();
        self.mean_acc = context.mean_acc;
        self.angvel_last = context.angvel_last;
        self.ekf_inited = context.ekf_inited;
    }

    #[cfg(feature = "threads")]
    pub(crate) fn update_realtime_tail(
        &mut self,
        tail: &crate::ImuMsg,
        state: crate::eqf::StateIKFom,
    ) {
        self.last_imu = tail.clone();
        self.imu_state = state;
        self.angvel_last = Vector3::from(tail.angular_velocity) - state.bg;
    }

    #[cfg(feature = "threads")]
    pub(crate) fn mean_acc_norm(&self) -> f64 {
        self.mean_acc.norm()
    }

    pub fn new(
        params: &ImuParameters,
        lidar_to_imu: &Isometry3<f64>,
        base_to_lidar: &Isometry3<f64>,
    ) -> Self {
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
        s.wheel_odom_cov_override = params.wheel_odom_cov_override;
        s.wheel_odom_cov_override = params.wheel_odom_cov_override;

        s.lidar_to_imu_r = *lidar_to_imu.rotation.to_rotation_matrix().matrix();
        s.lidar_to_imu_t = lidar_to_imu.translation.vector;

        let lidar_to_base = base_to_lidar.inverse();
        s.lidar_to_base_r = *lidar_to_base.rotation.to_rotation_matrix().matrix();
        s.lidar_to_base_t = lidar_to_base.translation.vector;

        s
    }

    pub fn set_lidar_to_imu(&mut self, lidar_to_imu: &Isometry3<f64>) {
        self.lidar_to_imu_r = *lidar_to_imu.rotation.to_rotation_matrix().matrix();
        self.lidar_to_imu_t = lidar_to_imu.translation.vector;
    }

    pub fn set_base_to_lidar(&mut self, base_to_lidar: &Isometry3<f64>) {
        let lidar_to_base = base_to_lidar.inverse();
        self.lidar_to_base_r = *lidar_to_base.rotation.to_rotation_matrix().matrix();
        self.lidar_to_base_t = lidar_to_base.translation.vector;
    }

    pub fn initialize(&mut self, msgs: &[ImuMsg], ekf: &mut crate::eqf::Eqf) -> Vector3<f64> {
        let mut n = 1.0;
        self.last_imu = crate::ImuMsg::default();
        self.mean_acc = msgs[0].linear_acceleration.into();
        self.mean_gyr = msgs[0].angular_velocity.into();

        for msg in msgs {
            let cur_acc: Vector3<f64> = msg.linear_acceleration.into();
            let cur_gyr: Vector3<f64> = msg.angular_velocity.into();

            self.mean_acc += (cur_acc - self.mean_acc) / n;
            self.mean_gyr += (cur_gyr - self.mean_gyr) / n;

            self.cov_acc = self.cov_acc * (n - 1.0) / n
                + (cur_acc - self.mean_acc).component_mul(&(cur_acc - self.mean_acc)) * (n - 1.0)
                    / (n * n);
            self.cov_gyr = self.cov_gyr * (n - 1.0) / n
                + (cur_gyr - self.mean_gyr).component_mul(&(cur_gyr - self.mean_gyr)) * (n - 1.0)
                    / (n * n);

            n += 1.0;
        }

        ekf.state.bg = self.mean_gyr;
        ekf.state.offset_t_l_i = self.lidar_to_imu_t;
        ekf.state.offset_r_l_i = SO3::try_from_mat(&self.lidar_to_imu_r).unwrap();

        // Initialize R_W_I so the world frame aligns with the base frame (not the IMU or Lidar frame).
        // R_W_B = R_W_I * R_I_L * R_L_B = I  =>  R_W_I = (R_I_L * R_L_B)^{-1} = R_B_L * R_L_I
        // Note: self.lidar_to_base_r is R_B_L.
        let r_b_l = self.lidar_to_base_r;
        let r_l_i = self.lidar_to_imu_r.transpose();
        let r_w_i_mat = r_b_l * r_l_i;
        ekf.state.rot = SO3::try_from_mat(&r_w_i_mat).unwrap();

        // Initialize position such that the Base starts at the World origin.
        // p_W_B = R_W_I * (R_I_L * p_L_B + t_I_L) + p_W_I = 0
        // p_W_I = - R_W_I * (R_I_L * p_L_B + t_I_L)
        let p_l_b = self.lidar_to_base_t;
        let p_i_b = self.lidar_to_imu_r * p_l_b + self.lidar_to_imu_t;
        ekf.state.pos = -(ekf.state.rot.matrix() * p_i_b);

        // Gravity in IMU frame, then rotated to the new world frame: g_W = R_W_I * g_IMU
        let g_imu = -self.mean_acc / self.mean_acc.norm() * 9.81; // TODO PARAM
        ekf.state.grav = ekf.state.rot.matrix() * g_imu;

        // This covariance is being written fresh, so it is expressed about whatever
        // origin we declare here. Anchor on the robot so it means what the diagonal
        // below says it means.
        ekf.anchor = ekf.state.pos;
        ekf.cov = faer::Mat::<f64>::identity(24, 24);

        // 24-dim state: pos(0-2), rot(3-5), offset_R(6-8), offset_T(9-11), vel(12-14), bg(15-17), ba(18-20), grav(21-23)
        // bg bias covariance (15-17)
        ekf.cov[(15, 15)] = COV2;
        ekf.cov[(16, 16)] = COV2;
        ekf.cov[(17, 17)] = COV2;

        // ba bias covariance (18-20)
        ekf.cov[(18, 18)] = COV3;
        ekf.cov[(19, 19)] = COV3;
        ekf.cov[(20, 20)] = COV3;

        // grav covariance (21-23)
        ekf.cov[(21, 21)] = COV1;
        ekf.cov[(22, 22)] = COV1;
        ekf.cov[(23, 23)] = COV1;

        self.last_imu = msgs[msgs.len() - 1].clone();

        self.ekf_inited = true;
        ekf.initialized = true;

        self.mean_acc
    }

    pub fn undistort(
        &mut self,
        ekf: &mut Eqf,
        scan: crate::UndistortScan<'_>,
        kinematic: KinematicModel,
        constraint_std: f64,
        execution_mode: ExecutionMode,
    ) -> Option<f64> {
        let crate::UndistortScan {
            imu_msgs,
            odom_msgs,
            last_cloud_end_time,
            cloud_begin_time,
            mut cloud_end_time,
            cloud,
            cloud_passthrough,
            lidar_vendor,
        } = scan;
        debug_assert!(!imu_msgs.is_empty());
        debug_assert!(last_cloud_end_time > 0.0);
        debug_assert!(cloud_begin_time > 0.0);
        debug_assert!(cloud_end_time > 0.0);

        if !matches!(lidar_vendor, Some(LidarVendor::NoTimestamp)) {
            cloud_end_time += cloud_begin_time;
        }

        const GRAVITY: f64 = 9.81; // TODO PARAM

        let mean_acc = self.mean_acc.norm();

        self.imu_state = ekf.state;

        let mut imu_poses = vec![Pose6D::new(
            0.0,
            self.acc_s_last,
            self.angvel_last,
            ekf.state.vel,
            ekf.state.pose(),
        )];

        let mut angvel_avr_x = 0.0;
        let mut angvel_avr_y = 0.0;
        let mut angvel_avr_z = 0.0;

        let mut acc_avr_x = 0.0;
        let mut acc_avr_y = 0.0;
        let mut acc_avr_z = 0.0;

        let mut all_imus = ::core::iter::once(&self.last_imu).chain(imu_msgs.iter());

        let mut head = all_imus.next()?;

        // Keep odometry corrections in lockstep with IMU prediction.
        ekf.last_time = last_cloud_end_time;
        let mut odom_iter = odom_msgs.iter().peekable();

        for tail in all_imus {
            let head_stamp = timestamp_to_secs(&head.stamp);
            let tail_stamp = timestamp_to_secs(&tail.stamp);

            if head_stamp == tail_stamp {
                if should_log_once_default("imu_same_stamp_skip") {
                    warn!("Found the same stamps from IMU, skipping a msg");
                }
                head = tail;
                continue;
            }

            debug_assert!(head_stamp < tail_stamp);

            angvel_avr_x = (head.angular_velocity[0] + tail.angular_velocity[0]) * 0.5;
            angvel_avr_y = (head.angular_velocity[1] + tail.angular_velocity[1]) * 0.5;
            angvel_avr_z = (head.angular_velocity[2] + tail.angular_velocity[2]) * 0.5;

            acc_avr_x = (head.linear_acceleration[0] + tail.linear_acceleration[0]) * 0.5 * GRAVITY
                / mean_acc;
            acc_avr_y = (head.linear_acceleration[1] + tail.linear_acceleration[1]) * 0.5 * GRAVITY
                / mean_acc;
            acc_avr_z = (head.linear_acceleration[2] + tail.linear_acceleration[2]) * 0.5 * GRAVITY
                / mean_acc;

            let avr_x = [acc_avr_x, acc_avr_y, acc_avr_z];
            let angvel_avr = [angvel_avr_x, angvel_avr_y, angvel_avr_z];

            // Process events in timestamp order: advance to each odometry sample,
            // correct there, and only then finish the interval at the IMU sample.
            while let Some(odom) = odom_iter.peek() {
                let odom_time = timestamp_to_secs(&odom.stamp);
                if odom_time > tail_stamp {
                    break;
                }

                let dt_pred = odom_time - ekf.last_time;
                if dt_pred > 0.0 {
                    _ = self.predict(ekf, avr_x, angvel_avr, dt_pred);
                    // Enforce kinematic constraints after prediction segment
                    ekf.correct_kinematics(kinematic, tail.angular_velocity.into(), constraint_std);
                } else if dt_pred < 0.0 {
                    if should_log_once_default("skip_odom_non_positive_dt") {
                        warn!("Skipping stale odom sample with dt: {dt_pred:.6}");
                    }
                    odom_iter.next();
                    continue;
                }
                ekf.correct_with_body_velocities(
                    *(*odom),
                    kinematic,
                    self.wheel_odom_cov_override,
                    Vector3::new(angvel_avr_x, angvel_avr_y, angvel_avr_z),
                );
                odom_iter.next();
            }

            // Finish prediction to the tail timestamp.
            let dt_to_tail = tail_stamp - ekf.last_time;
            if dt_to_tail > 0.0 {
                _ = self.predict(ekf, avr_x, angvel_avr, dt_to_tail);
                // Enforce kinematic constraints after final prediction
                ekf.correct_kinematics(kinematic, tail.angular_velocity.into(), constraint_std);
            }

            self.imu_state = ekf.state;

            let tail_ang_vel: Vector3<f64> = tail.angular_velocity.into();
            let tail_lin_acc: Vector3<f64> = tail.linear_acceleration.into();
            self.angvel_last = tail_ang_vel - self.imu_state.bg;
            self.acc_s_last = self.imu_state.pose().rotation().matrix()
                * ((tail_lin_acc * GRAVITY / mean_acc) - self.imu_state.ba)
                + self.imu_state.grav;

            // A zero acceleration mean indicates missing initialization data.
            debug_assert!(!self.acc_s_last.iter().any(|a| a.is_nan()));

            let offs_t = tail_stamp - cloud_begin_time;
            imu_poses.push(Pose6D::new(
                offs_t,
                self.acc_s_last,
                self.angvel_last,
                self.imu_state.vel,
                self.imu_state.pose(),
            ));

            head = tail;
        }

        // Integrate remaining odometry and advance to end of cloud
        self.last_imu = imu_msgs[imu_msgs.len() - 1].clone();

        let avr_x = [acc_avr_x, acc_avr_y, acc_avr_z];
        let angvel_avr = [angvel_avr_x, angvel_avr_y, angvel_avr_z];

        while let Some(odom) = odom_iter.peek() {
            let odom_time = timestamp_to_secs(&odom.stamp);
            if odom_time > cloud_end_time {
                break;
            }

            let dt_pred = odom_time - ekf.last_time;
            if dt_pred > 0.0 {
                _ = self.predict(ekf, avr_x, angvel_avr, dt_pred);
                // Enforce kinematic constraints after prediction segment
                let gyro = Vector3::new(angvel_avr_x, angvel_avr_y, angvel_avr_z);
                ekf.correct_kinematics(kinematic, gyro, constraint_std);
            } else if dt_pred < 0.0 {
                if should_log_once_default("skip_odom_non_positive_dt") {
                    warn!("Skipping stale odom sample with dt: {dt_pred:.6}");
                }
                odom_iter.next();
                continue;
            }

            ekf.correct_with_body_velocities(
                *(*odom),
                kinematic,
                self.wheel_odom_cov_override,
                Vector3::new(angvel_avr_x, angvel_avr_y, angvel_avr_z),
            );
            odom_iter.next();
        }

        let dt = cloud_end_time - ekf.last_time;
        assert!(dt >= 0.0);
        let _ = self.predict(ekf, avr_x, angvel_avr, dt);
        // Enforce kinematic constraints after final prediction to cloud end
        let gyro = Vector3::new(angvel_avr_x, angvel_avr_y, angvel_avr_z);
        ekf.correct_kinematics(kinematic, gyro, constraint_std);
        self.imu_state = ekf.state;

        if !matches!(lidar_vendor, Some(LidarVendor::NoTimestamp)) {
            let imu_state_pose_translation = self.imu_state.pose().translation();
            let imu_state_pose_rotation_t = self.imu_state.pose().rotation().matrix().transpose();
            let imu_state_lidar_to_imu_t = self.imu_state.offset_t_l_i;
            let imu_state_lidar_to_imu_r = self.imu_state.offset_r_l_i.matrix();
            let imu_state_lidar_to_imu_r_t = imu_state_lidar_to_imu_r.transpose();
            let imu_poses_for_downsampled = imu_poses.clone();
            let point_count =
                cloud.len() + cloud_passthrough.as_ref().map_or(0, |points| points.len());

            let deskew = |point: &mut PointXYZIT| {
                // For each point, find the correct IMU pose interval.
                // binary_search_by finds the index of the first pose with a timestamp >= the point's timestamp.

                let tail_idx = imu_poses_for_downsampled
                    .binary_search_by(|p| {
                        p.offset_time
                            .partial_cmp(&(point.timestamp as f64))
                            .unwrap_or(::core::cmp::Ordering::Less)
                    })
                    .unwrap_or_else(|e| e);

                if tail_idx == 0 || tail_idx >= imu_poses_for_downsampled.len() {
                    // point is before the first pose or after the last.
                    return;
                }
                let head_idx = tail_idx - 1;

                let head = &imu_poses_for_downsampled[head_idx];
                let tail = &imu_poses_for_downsampled[tail_idx];

                let mut dt = point.timestamp as f64 - head.offset_time;
                if dt < 0.0 {
                    dt = 0.0;
                }

                let r_i = head
                    .pose
                    .rotation()
                    .group_mul(crate::eqf::SO3::exp(tail.gyr * dt))
                    .matrix();

                let t_ei = head.pose.translation() + head.vel * dt + 0.5 * tail.acc * dt * dt
                    - imu_state_pose_translation;

                let p_imu_frame = imu_state_lidar_to_imu_r
                    * Vector3::new(point.x as f64, point.y as f64, point.z as f64)
                    + imu_state_lidar_to_imu_t;

                let p_compensate = imu_state_lidar_to_imu_r_t
                    * (imu_state_pose_rotation_t * (r_i * p_imu_frame + t_ei)
                        - imu_state_lidar_to_imu_t);

                point.x = p_compensate.x as f32;
                point.y = p_compensate.y as f32;
                point.z = p_compensate.z as f32;
            };

            let started = pelorus_core::Instant::now();
            #[cfg(feature = "rayon")]
            match execution_mode {
                ExecutionMode::Sequential => {
                    cloud.iter_mut().for_each(deskew);
                    if let Some(passthrough) = cloud_passthrough {
                        passthrough.iter_mut().for_each(deskew);
                    }
                }
                ExecutionMode::Rayon => rayon::scope(move |scope| {
                    scope.spawn(move |_| {
                        cloud.par_iter_mut().for_each(deskew);
                    });
                    if let Some(passthrough) = cloud_passthrough {
                        scope.spawn(move |_| {
                            passthrough.par_iter_mut().for_each(deskew);
                        });
                    }
                }),
            }
            #[cfg(not(feature = "rayon"))]
            {
                let _ = execution_mode;
                cloud.iter_mut().for_each(deskew);
                if let Some(passthrough) = cloud_passthrough {
                    passthrough.iter_mut().for_each(deskew);
                }
            }
            return Some(started.elapsed().as_nanos() as f64 / point_count.max(1) as f64);
        }
        None
    }

    pub fn predict(
        &self,
        ekf: &mut Eqf,
        acc: [f64; 3],
        gyro: [f64; 3],
        dt: f64,
    ) -> anyhow::Result<crate::ImuEkfPrediction> {
        let inp = crate::eqf::InputIKFom {
            acc: Vector3::new(acc[0], acc[1], acc[2]),
            gyro: Vector3::new(gyro[0], gyro[1], gyro[2]),
        };

        let q = {
            let mut q = faer::Mat::<f64>::zeros(12, 12);
            // Copy ekf_cov_gyr to block (0, 0)
            for i in 0..3 {
                for j in 0..3 {
                    q[(i, j)] = self.ekf_cov_gyr[(i, j)];
                }
            }
            // Copy ekf_cov_acc to block (3, 3)
            for i in 0..3 {
                for j in 0..3 {
                    q[(3 + i, 3 + j)] = self.ekf_cov_acc[(i, j)];
                }
            }
            // Copy ekf_cov_gyr_bias to block (6, 6)
            for i in 0..3 {
                for j in 0..3 {
                    q[(6 + i, 6 + j)] = self.ekf_cov_gyr_bias[(i, j)];
                }
            }
            // Copy ekf_cov_acc_bias to block (9, 9)
            for i in 0..3 {
                for j in 0..3 {
                    q[(9 + i, 9 + j)] = self.ekf_cov_acc_bias[(i, j)];
                }
            }

            q
        };

        if !self.ekf_inited {
            anyhow::bail!("EKF not initialized");
        }

        ekf.predict(dt, &q, &inp);

        Ok(crate::ImuEkfPrediction {
            pose: Isometry3::from_parts(
                Translation3::from(ekf.state.pose().translation()),
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                    &ekf.state.pose().rotation().matrix(),
                )),
            ),
            linear_velocity: ekf.state.vel,
            angular_velocity: self.angvel_last,
            lidar_imu_extrinsic: Isometry3::from_parts(
                Translation3::from(ekf.state.offset_t_l_i),
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                    &ekf.state.offset_r_l_i.matrix(),
                )),
            ),
            accel_bias: ekf.state.ba,
            gyro_bias: ekf.state.bg,
        })
    }
}

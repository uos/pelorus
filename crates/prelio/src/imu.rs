use rayon::iter::{IntoParallelIterator, ParallelIterator};
use sophus_lie::traits::IsTranslationProductGroup;

use nalgebra::*;

use crate::{config, timestamp_to_secs, MeasureGroup, PointXYZT};

use ros_pointcloud2::ros::TimeMsg;

#[derive(Debug, Clone, Default)]
pub struct ImuMsg {
    pub orientation: [f64; 4],
    pub orientation_covariance: [f64; 9],
    pub angular_velocity: [f64; 3],
    pub angular_velocity_covariance: [f64; 9],
    pub linear_acceleration: [f64; 3],
    pub linear_acceleration_covariance: [f64; 9],
    pub frame_id: String,
    pub stamp: crate::TimeMsg,
}

pub fn time_msg_from_secs(secs: f64) -> crate::TimeMsg {
    let sec = secs.floor() as i32;
    let nanosec = ((secs - secs.floor()) * 1e9).floor() as u32;
    crate::TimeMsg { sec, nanosec }
}

type SO3 = sophus_lie::Rotation3<f64, 1>;
type SE3 = sophus_lie::Isometry3<f64, 1>;

#[derive(Debug)]
pub struct GyrInt {
    start_stamp: f64,
    last_imu: Option<ImuMsg>,
    v_imu: Vec<ImuMsg>,
    v_rot: Vec<SO3>,
    v_se3: Vec<SE3>,
    imu_kum: ImuMsg,
}

fn imu_msg_to_cov(msg: &ImuMsg) -> [f64; 36] {
    let mut covariances = [0.0; 36];
    covariances[0] =
        (msg.linear_acceleration_covariance[0] + msg.angular_velocity_covariance[0]) / 2.0;
    covariances[7] =
        (msg.linear_acceleration_covariance[4] + msg.angular_velocity_covariance[4]) / 2.0;
    covariances[14] =
        (msg.linear_acceleration_covariance[8] + msg.angular_velocity_covariance[8]) / 2.0;

    covariances[21] =
        (msg.linear_acceleration_covariance[0] + msg.angular_velocity_covariance[0]) / 2.0;
    covariances[28] =
        (msg.linear_acceleration_covariance[4] + msg.angular_velocity_covariance[4]) / 2.0;
    covariances[35] =
        (msg.linear_acceleration_covariance[8] + msg.angular_velocity_covariance[8]) / 2.0;

    covariances
}

fn imu_msg_to_twist(msg: &ImuMsg) -> [f64; 6] {
    let mut twist = [0.0; 6];
    twist[0] = msg.linear_acceleration[0];
    twist[1] = msg.linear_acceleration[1];
    twist[2] = msg.linear_acceleration[2];
    twist[3] = msg.angular_velocity[0];
    twist[4] = msg.angular_velocity[1];
    twist[5] = msg.angular_velocity[2];
    twist
}

impl GyrInt {
    pub fn new() -> Self {
        Self {
            start_stamp: -1.0,
            last_imu: None,
            v_imu: Vec::new(),
            v_rot: Vec::new(),
            v_se3: Vec::new(),
            imu_kum: ImuMsg::default(),
        }
    }

    pub fn reset(&mut self, start_time: f64, imu: Option<ImuMsg>) {
        self.start_stamp = start_time;
        self.last_imu = imu;
        self.v_imu.clear();
        self.v_rot.clear();
        self.v_se3.clear();
        self.imu_kum = ImuMsg::default();
    }

    pub fn get_pose(&self) -> SE3 {
        match self.v_se3.last() {
            Some(pose) => pose.clone(),
            None => SE3::identity(),
        }
    }

    pub fn get_rot(&self) -> SO3 {
        match self.v_rot.last() {
            Some(rot) => rot.clone(),
            None => SO3::identity(),
        }
    }

    pub fn get_kum_covariances(&self) -> [f64; 36] {
        imu_msg_to_cov(&self.imu_kum)
    }

    // TODO remove acceleration usage
    pub fn integrate(&mut self, imu: &ImuMsg, init_imu: &ImuMsg, gravity: f64) {
        let gravity = Vector3::new(0.0, 0.0, -gravity);

        if self.v_imu.is_empty() {
            debug_assert!(self.start_stamp > 0.0);
            let last_imu = self.last_imu.as_ref().unwrap_or(init_imu);

            self.v_rot.push(SO3::identity());
            self.v_se3.push(SE3::identity());

            let dt1 = self.start_stamp - timestamp_to_secs(&last_imu.stamp).max(0.0);
            let dt2 = timestamp_to_secs(&imu.stamp) - self.start_stamp;
            debug_assert!(dt2 >= 0.0);

            let w1 = dt2 / (dt1 + dt2 + 1e-9);
            let w2 = dt1 / (dt1 + dt2 + 1e-9);

            let gyr1 = last_imu.angular_velocity;
            let acc1 = last_imu.linear_acceleration;
            let gyr2 = imu.angular_velocity;
            let acc2 = imu.linear_acceleration;

            let mut imu_interpolated = ImuMsg::default();
            imu_interpolated.stamp = time_msg_from_secs(self.start_stamp);
            imu_interpolated.angular_velocity[0] = w1 * gyr1[0] + w2 * gyr2[0];
            imu_interpolated.angular_velocity[1] = w1 * gyr1[1] + w2 * gyr2[1];
            imu_interpolated.angular_velocity[2] = w1 * gyr1[2] + w2 * gyr2[2];
            imu_interpolated.linear_acceleration[0] = w1 * acc1[0] + w2 * acc2[0];
            imu_interpolated.linear_acceleration[1] = w1 * acc1[1] + w2 * acc2[1];
            imu_interpolated.linear_acceleration[2] = w1 * acc1[2] + w2 * acc2[2];

            for i in 0..9 {
                self.imu_kum.orientation_covariance[i] +=
                    w1 * last_imu.orientation_covariance[i] + w2 * imu.orientation_covariance[i];
                self.imu_kum.angular_velocity_covariance[i] += w1
                    * last_imu.angular_velocity_covariance[i]
                    + w2 * imu.angular_velocity_covariance[i];
                self.imu_kum.linear_acceleration_covariance[i] += w1
                    * last_imu.linear_acceleration_covariance[i]
                    + w2 * imu.linear_acceleration_covariance[i];
            }

            self.v_imu.push(imu_interpolated);
        }

        let rot_last = self.v_rot.last().unwrap().clone();
        let pose_last = self.v_se3.last().unwrap().clone();
        let imu_msg_last = self.v_imu.last().unwrap().clone();
        let time_last = timestamp_to_secs(&imu_msg_last.stamp);
        let gyr_last = Vector3::new(
            imu_msg_last.angular_velocity[0],
            imu_msg_last.angular_velocity[1],
            imu_msg_last.angular_velocity[2],
        );
        let acc = Vector3::new(
            imu.linear_acceleration[0],
            imu.linear_acceleration[1],
            imu.linear_acceleration[2],
        );
        let time = timestamp_to_secs(&imu.stamp);
        debug_assert!(time >= 0.0);
        let gyr = Vector3::new(
            imu.angular_velocity[0],
            imu.angular_velocity[1],
            imu.angular_velocity[2],
        );
        let dt = time - time_last;
        let delta_angle = dt * 0.5 * (gyr + gyr_last);

        // Transform the acceleration to the world frame using the last pose's rotation
        let acc_world = pose_last.rotation().matrix() * (acc - gravity);

        let delta_translation = dt * dt * 0.5 * acc_world;
        let xi = Vector6::new(
            delta_translation.x,
            delta_translation.y,
            delta_translation.z,
            delta_angle.x,
            delta_angle.y,
            delta_angle.z,
        );

        let delta_pose = SE3::exp(&xi);

        let delta_r = SO3::exp(&delta_angle);

        let rot = rot_last.group_mul(&delta_r);

        let pose_with_accel = pose_last.group_mul(&delta_pose);

        self.v_rot.push(rot);
        self.v_se3.push(pose_with_accel);
        self.v_imu.push(imu.clone());
    }
}

#[derive(Debug)]
pub struct ImuProcess {
    // lidar to imu transform
    //transform_imu_lidar: SE3,
    last_lidar_stamp: Option<f64>,
    last_imu: Option<ImuMsg>,
    pub gyr_int: GyrInt,
    integrated_guess: (Isometry3<f64>, TimeMsg),
    runtime_eval: crate::RuntimeEvaluator,

    pub intrinsic_calibration_given: bool,
    pub gravity: f64,
}

fn quaternion_to_rotation_vector(q: &UnitQuaternion<f64>) -> Vector3<f64> {
    let angle = 2.0 * q.w.acos();
    let sin_half_angle = (1.0 - q.w * q.w).sqrt();

    // If sin_half_angle is too small, treat as zero for numerical stability
    if sin_half_angle.abs() < 1e-8 {
        Vector3::new(q.i, q.j, q.k)
    } else {
        Vector3::new(q.i, q.j, q.k) * (angle / sin_half_angle)
    }
}

fn quaternion_to_so3_exp(q: &UnitQuaternion<f64>) -> SO3 {
    SO3::exp(&quaternion_to_rotation_vector(q))
}

#[derive(Debug)]
pub struct ImuProcessedCloud<'a> {
    pub cloud: Vec<ros_pointcloud2::points::PointXYZ>,
    pub last_covariance: [f64; 36],
    pub last_twist: [f64; 6],
    pub integrated_covariance: [f64; 36],
    pub end_of_frame_guess: Isometry3<f64>,
    pub end_of_frame_guess_time: TimeMsg,
    pub runtime_eval: &'a crate::RuntimeEvaluator,
}

impl ImuProcess {
    pub fn new(gravity: f64, intrinsic_calibration_given: bool) -> Self {
        Self {
            gravity,
            last_lidar_stamp: None,
            last_imu: None,
            gyr_int: GyrInt::new(),
            integrated_guess: (Isometry3::identity(), TimeMsg::default()),
            runtime_eval: crate::RuntimeEvaluator::new(),
            intrinsic_calibration_given,
        }
    }

    pub fn integrate_imu_for_undistortion(
        &mut self,
        imu_data: &Vec<ImuMsg>,
        start_time: f64,
        config: Option<&config::UndistortParameters>,
    ) -> (Isometry3<f64>, SE3) {
        self.integrate_gyr(&imu_data, start_time);

        // only use translation in integration if we have calibration
        /*let mut t_l_be = if self.intrinsic_calibration_given {
            self.gyr_int.get_pose()
        } else {*/

        let mut t_l_be = SE3::from_translation_and_rotation(
            &Vector3::new(0.0, 0.0, 0.0), // too unstable for accurate undistortion
            &self.gyr_int.get_rot(),
        );

        if let Some(config) = config {
            let transform_imu_lidar = SE3::from_translation_and_rotation(
                &Vector3::new(
                    config.lidar_to_imu.translation.x,
                    config.lidar_to_imu.translation.y,
                    config.lidar_to_imu.translation.z,
                ),
                &quaternion_to_so3_exp(&config.lidar_to_imu.rotation),
            );

            t_l_be = transform_imu_lidar
                .inverse()
                .group_mul(&t_l_be)
                .group_mul(&transform_imu_lidar)
        }

        // quat from rot mat
        let rot3 = Rotation3::from_matrix(&t_l_be.rotation().matrix());
        let quat = UnitQuaternion::from_rotation_matrix(&rot3);
        let trans = Translation3::from(t_l_be.translation());
        (Isometry3::from_parts(trans, quat), t_l_be)
    }

    /// Integrates IMU data using Lie algebra for smoothing.
    /// Returns the estimated pose in the lidar frame.
    #[cfg(feature = "gtsam")]
    pub fn integrate_imu_gtsam(
        imu_data: &Vec<ImuMsg>,
        start_time: f64,
        undistort_params: Option<&config::UndistortParameters>,
        mut imu_params: config::ImuParameters,
    ) -> Isometry3<f64> {
        // update noise parameters with imu covariance, assuming constant noise. could be improved by averaging over all imu data
        if let Some(first_imu_entry) = imu_data.first() {
            imu_params.gyro_noise = first_imu_entry.angular_velocity_covariance[0];
            imu_params.accel_noise = first_imu_entry.linear_acceleration_covariance[0];
        } else {
            return Isometry3::identity(); // no imu data here
        }

        let mut gtsam_imu = gtsam_sys::ImuIntegration::new(&imu_params);
        gtsam_imu.reset_state(Isometry3::identity(), Vector3::zeros(), Vector3::zeros());

        /*self.gtsam_imu
        .reset_bias(Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.0));*/
        // TODO Find real bias

        for imu in imu_data {
            let dt = timestamp_to_secs(&imu.stamp) - start_time;
            if dt < 0.0 {
                continue;
            }
            gtsam_imu.add_imu(
                dt,
                Vector3::new(
                    imu.linear_acceleration[0],
                    imu.linear_acceleration[1],
                    imu.linear_acceleration[2],
                ),
                Vector3::new(
                    imu.angular_velocity[0],
                    imu.angular_velocity[1],
                    imu.angular_velocity[2],
                ),
            )
        }

        let integrated = gtsam_imu.get_state();
        let integrated_pose = Isometry3::from_parts(
            Translation3::from(integrated.translation),
            integrated.rotation,
        );

        // to lidar frame
        if let Some(config) = undistort_params {
            let lidar_frame_guess =
                config.lidar_to_imu.inverse() * integrated_pose * config.lidar_to_imu;
            lidar_frame_guess
        } else {
            integrated_pose
        }
    }

    pub fn process(
        &mut self,
        meas: MeasureGroup,
        config: &config::UndistortParameters,
        imu_pose_prediction: Option<impl FnMut(&[ImuMsg], f64, f64) -> crate::ImuPosePrediction>,
    ) -> Option<ImuProcessedCloud> {
        debug_assert!(!meas.imu_buffer.is_empty());
        debug_assert!(!meas.cloud.is_empty());

        self.runtime_eval.start("deskew").unwrap();

        let dt = meas.lidar_end_stamp - meas.lidar_start_stamp;

        let estimated_pose = match imu_pose_prediction
            .map(|mut f| f(&meas.imu_buffer, meas.lidar_start_stamp, dt))
        {
            Some(prediction) => {
                let rot =
                    Rotation3::from_matrix(&prediction.pose.rotation.to_rotation_matrix().matrix());
                let quat = UnitQuaternion::from_rotation_matrix(&rot);
                let iso = Isometry3::from_parts(prediction.pose.translation.into(), quat);
                let sophus = SE3::from_translation_and_rotation(
                    &prediction.pose.translation.vector,
                    &quaternion_to_so3_exp(&quat),
                );
                (iso, sophus)
            }
            None => self.integrate_imu_for_undistortion(
                &meas.imu_buffer,
                meas.lidar_start_stamp,
                Some(&config),
            ),
        };

        let n_last_imu = meas.imu_buffer.last().unwrap().clone();

        /*let my_integration =
        naive_preint::integrate_imu_up_to(sensor_data, &config.imu_preint_options);*/

        //dbg!(my_integration.pose.translation);

        let guess = (estimated_pose.0, time_msg_from_secs(meas.lidar_end_stamp));

        let deskewed_cloud = self.undistort_cloud_livox(meas.cloud, dt, estimated_pose.1);
        self.runtime_eval.stop("deskew").unwrap();

        self.last_lidar_stamp = Some(meas.lidar_end_stamp);
        self.last_imu = Some(n_last_imu.clone());

        Some(ImuProcessedCloud {
            cloud: deskewed_cloud,
            last_covariance: imu_msg_to_cov(&n_last_imu),
            last_twist: imu_msg_to_twist(&n_last_imu),
            integrated_covariance: self.gyr_int.get_kum_covariances(),
            end_of_frame_guess: guess.0,
            end_of_frame_guess_time: guess.1.clone(), // TODO impl copy for time?
            runtime_eval: &self.runtime_eval,
        })
    }

    pub fn reset(&mut self) {
        self.last_lidar_stamp = None;
        self.last_imu = None;
        self.gyr_int.reset(-1.0, None);
        self.integrated_guess = (Isometry3::identity(), TimeMsg::default());
    }

    pub fn integrate_gyr(&mut self, v_imu: &Vec<ImuMsg>, init_time: f64) {
        let init_imu = match v_imu.first() {
            Some(imu) => imu.clone(),
            None => return,
        };
        self.gyr_int.reset(
            self.last_lidar_stamp.unwrap_or(init_time),
            self.last_imu.clone(),
        );
        for imu in v_imu {
            self.gyr_int.integrate(imu, &init_imu, self.gravity);
        }
    }

    pub fn undistort_cloud_livox(
        &self,
        cloud_in_out: Vec<PointXYZT>,
        delta_time: f64,
        pose_end_of_frame: SE3,
    ) -> Vec<ros_pointcloud2::points::PointXYZ> {
        let tbe = pose_end_of_frame.translation();
        let rso3_be = pose_end_of_frame.rotation().log();
        cloud_in_out
            .into_par_iter()
            .map(|p| {
                let ring = p.timestamp.floor() as i64;
                let dt_bi = p.timestamp - ring as f64;
                let ratio_bi = dt_bi / delta_time;
                let ratio_ie = 1.0 - ratio_bi;

                let rso3_ie = ratio_ie * rso3_be;
                let rie = SO3::exp(&rso3_ie);

                let tie = ratio_ie * tbe;
                let v_pt_i = Vector3::new(p.x as f64, p.y as f64, p.z as f64);
                let v_pt_comp_e = rie.inverse().matrix() * (v_pt_i - tie);
                ros_pointcloud2::points::PointXYZ::new(
                    v_pt_comp_e.x as f32,
                    v_pt_comp_e.y as f32,
                    v_pt_comp_e.z as f32,
                )
            })
            .collect()
    }
}

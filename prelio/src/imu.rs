use either::Either::{Left, Right};
use imu_fusion::FusionAhrs;
use pelorus_core::{
    ImuMsg, ImuPredictorUndistort, PointXYZIT, UndistortParameters, imu_msg_to_cov,
    imu_msg_to_twist, time_msg_from_secs,
};
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use sophus_lie::prelude::*;

use nalgebra::*;

use crate::{MeasureGroup, timestamp_to_secs};

use ros_pointcloud2::ros::TimeMsg;

type SO3 = sophus_lie::Rotation3F64;
type SE3 = sophus_lie::Isometry3F64;

pub struct ImuMadgwick {
    fusion: imu_fusion::FusionAhrs,
    gravity: f64,
}

impl ImuMadgwick {
    pub fn new(gravity: f64) -> Self {
        ImuMadgwick {
            fusion: FusionAhrs::new(),
            gravity,
        }
    }
    /// Update the AHRS with new sensor data.
    /// gyroscope units are radians/s
    /// accelerometer units are m/s^2
    pub fn update_no_magnetometer_ros(&mut self, dt: f64, gyro: [f64; 3], accel: [f64; 3]) {
        let gyro = gyro.map(|v| v.to_degrees());
        let gyro = imu_fusion::FusionVector::new(gyro[0] as f32, gyro[1] as f32, gyro[2] as f32);

        let accel = accel.map(|a| a / self.gravity);
        let accel =
            imu_fusion::FusionVector::new(accel[0] as f32, accel[1] as f32, accel[2] as f32);

        self.fusion.update_no_mag(gyro, accel, dt as f32)
    }

    /// Get orientation in world frame as a quaternion.
    /// The quaternion is in the form [w, x, y, z].
    pub fn get_quaternion(&self) -> [f64; 4] {
        [
            self.fusion.quaternion.w as f64,
            self.fusion.quaternion.x as f64,
            self.fusion.quaternion.y as f64,
            self.fusion.quaternion.z as f64,
        ]
    }

    /// Get linear acceleration in sensor frame.
    /// The acceleration is in the form [x, y, z].
    /// Units are g.
    pub fn get_linear_acceleration(&self) -> [f64; 3] {
        let acc = self.fusion.linear_acc();
        [acc.x as f64, acc.y as f64, acc.z as f64]
    }

    pub fn get_linear_acceleration_ros(&self) -> [f64; 3] {
        self.get_linear_acceleration().map(|a| a * self.gravity)
    }
}

#[derive(Debug)]
pub struct GyrInt {
    start_stamp: f64,
    last_imu: Option<ImuMsg>,
    v_imu: Vec<ImuMsg>,
    v_rot: Vec<SO3>,
    v_se3: Vec<SE3>,
    imu_kum: ImuMsg,
}

impl Default for GyrInt {
    fn default() -> Self {
        Self::new()
    }
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
            Some(pose) => *pose,
            None => SE3::identity(),
        }
    }

    pub fn get_rot(&self) -> SO3 {
        match self.v_rot.last() {
            Some(rot) => *rot,
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

        let rot_last = *self.v_rot.last().unwrap();
        let pose_last = *self.v_se3.last().unwrap();
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

        let delta_pose = SE3::exp(xi);
        let delta_r = SO3::exp(delta_angle);
        let rot = rot_last.group_mul(delta_r);
        let pose_with_accel = pose_last.group_mul(delta_pose);

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
    SO3::exp(quaternion_to_rotation_vector(q))
}

#[derive(Debug)]
pub struct ImuProcessedCloud<'a> {
    pub cloud: Vec<PointXYZIT>,
    pub cloud_output: Option<Vec<PointXYZIT>>,
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
        config: Option<&UndistortParameters>,
    ) -> (Isometry3<f64>, SE3) {
        self.integrate_gyr(imu_data, start_time);

        // only use translation in integration if we have calibration
        /*let mut t_l_be = if self.intrinsic_calibration_given {
            self.gyr_int.get_pose()
        } else {*/

        let mut t_l_be = SE3::from_rotation_and_translation(
            self.gyr_int.get_rot(),
            Vector3::new(0.0, 0.0, 0.0), // too unstable for accurate undistortion
        );

        if let Some(config) = config {
            let transform_imu_lidar = SE3::from_rotation_and_translation(
                quaternion_to_so3_exp(&config.lidar_to_imu.rotation),
                Vector3::new(
                    config.lidar_to_imu.translation.x,
                    config.lidar_to_imu.translation.y,
                    config.lidar_to_imu.translation.z,
                ),
            );

            t_l_be = transform_imu_lidar
                .inverse()
                .group_mul(t_l_be)
                .group_mul(transform_imu_lidar)
        }

        // quat from rot mat
        let rot3 = Rotation3::from_matrix(&t_l_be.rotation().matrix());
        let quat = UnitQuaternion::from_rotation_matrix(&rot3);
        let trans = Translation3::from(t_l_be.translation());
        (Isometry3::from_parts(trans, quat), t_l_be)
    }

    pub fn process(
        &'_ mut self,
        mut meas: MeasureGroup,
        imu_pose_prediction: &Option<Box<ImuPredictorUndistort>>,
    ) -> Option<ImuProcessedCloud<'_>> {
        debug_assert!(!meas.imu_buffer.is_empty());
        debug_assert!(!meas.cloud.is_empty());

        self.runtime_eval.start("deskew").unwrap();

        let estimated_pose = match imu_pose_prediction.as_ref().map(|f| {
            f(
                &meas.imu_buffer,
                &meas.odom_buffer,
                meas.last_lidar_end_stamp,
                meas.lidar_start_stamp,
                meas.lidar_end_stamp,
                &mut meas.cloud,
                meas.cloud_passthrough.as_mut(),
            )
        }) {
            Some(Left(pose)) => {
                let rot = Rotation3::from_matrix(pose.pose.rotation.to_rotation_matrix().matrix());
                let quat = UnitQuaternion::from_rotation_matrix(&rot);
                let iso = Isometry3::from_parts(pose.pose.translation, quat);
                // let sophus = SE3::from_rotation_and_translation(
                //     quaternion_to_so3_exp(&quat),
                //     pose.pose.translation.vector,
                // );
                // let deskewed_cloud = self.undistort_cloud_livox(
                //     meas.cloud,
                //     meas.lidar_end_stamp - meas.lidar_start_stamp,
                //     sophus,
                // );
                iso
            }
            Some(Right(pose)) => pose,
            /*None => self.integrate_imu_for_undistortion(
                &meas.imu_buffer,
                meas.lidar_start_stamp,
                Some(&config),
            ),*/
            None => Isometry3::identity(),
        };

        let deskewed_cloud = meas.cloud;
        let deskewed_cloud_output = meas.cloud_passthrough;

        let n_last_imu = meas.imu_buffer.last().unwrap().clone();

        /*let my_integration =
        naive_preint::integrate_imu_up_to(sensor_data, &config.imu_preint_options);*/

        self.runtime_eval.stop("deskew").unwrap();

        self.last_lidar_stamp = Some(meas.lidar_end_stamp);
        self.last_imu = Some(n_last_imu.clone());
        Some(ImuProcessedCloud {
            cloud: deskewed_cloud,
            cloud_output: deskewed_cloud_output,
            last_covariance: imu_msg_to_cov(&n_last_imu),
            last_twist: imu_msg_to_twist(&n_last_imu),
            integrated_covariance: self.gyr_int.get_kum_covariances(),
            end_of_frame_guess: estimated_pose,
            end_of_frame_guess_time: time_msg_from_secs(meas.lidar_end_stamp), // TODO impl copy for time?
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
        cloud_in_out: Vec<PointXYZIT>,
        delta_time: f64,
        pose_end_of_frame: SE3,
    ) -> Vec<ros_pointcloud2::points::PointXYZI> {
        let tbe = pose_end_of_frame.translation();
        let rso3_be = pose_end_of_frame.rotation().log();
        let iter = {
            #[cfg(feature = "parallel")]
            {
                cloud_in_out.into_par_iter()
            }
            #[cfg(not(feature = "parallel"))]
            {
                cloud_in_out.into_iter()
            }
        };
        iter.map(|p| {
            let ring = p.timestamp.floor() as i64;
            let dt_bi = p.timestamp - ring as f32;
            let ratio_bi = dt_bi / delta_time as f32;
            let ratio_ie = 1.0 - ratio_bi;

            let rso3_ie = ratio_ie * rso3_be.cast();
            let rie = SO3::exp(rso3_ie.cast());

            let tie = ratio_ie * tbe.cast();
            let v_pt_i = Vector3::new(p.x as f64, p.y as f64, p.z as f64);
            let v_pt_comp_e = rie.inverse().matrix() * (v_pt_i - tie.cast());
            ros_pointcloud2::points::PointXYZI::new(
                v_pt_comp_e.x as f32,
                v_pt_comp_e.y as f32,
                v_pt_comp_e.z as f32,
                p.intensity,
            )
        })
        .collect()
    }
}

use log::{error, warn};
use nalgebra::*;
use pelorus_core::{
    KinematicModel, OdomMsg, log_debouncer::should_log_once_default, timing::RuntimeEvaluator,
};
use prelio::timestamp_to_secs;

use sophus_lie::prelude::*;

pub type SO3 = sophus_lie::Rotation3F64;
pub type SE3 = sophus_lie::Isometry3F64;

// TODO use config, needs some refactor
const GRAVITY_DEFAULT: f64 = 9.81;

#[derive(Clone, Debug)]
pub struct StateIKFom {
    pub pose: SE3,                    // T_W_I: world frame pose of IMU
    pub lidar_to_imu_r: SO3,          // R_I_L: lidar to IMU rotation (extrinsic calibration)
    pub lidar_to_imu_t: Vector3<f64>, // t_I_L: lidar to IMU translation (extrinsic calibration)
    pub vel: Vector3<f64>,            // velocity in world frame
    pub bg: Vector3<f64>,             // estimated gyro bias
    pub ba: Vector3<f64>,             // accelerometer bias
    pub grav: Vector3<f64>,           // gravity vector in world frame
}

impl StateIKFom {
    pub fn new(gravity: f64) -> Self {
        StateIKFom {
            pose: SE3::identity(),
            lidar_to_imu_r: SO3::identity(),
            lidar_to_imu_t: Vector3::zeros(),
            vel: Vector3::zeros(),
            bg: Vector3::zeros(),
            ba: Vector3::zeros(),
            grav: Vector3::new(0.0, 0.0, -gravity),
        }
    }
}

impl Default for StateIKFom {
    fn default() -> Self {
        StateIKFom {
            pose: SE3::identity(),
            lidar_to_imu_r: SO3::identity(),
            lidar_to_imu_t: Vector3::zeros(),
            vel: Vector3::zeros(),
            bg: Vector3::zeros(),
            ba: Vector3::zeros(),
            grav: Vector3::new(0.0, 0.0, -GRAVITY_DEFAULT),
        }
    }
}

pub struct InputIKFom {
    pub acc: Vector3<f64>,
    pub gyro: Vector3<f64>,
}

impl InputIKFom {
    pub fn new(acc: Vector3<f64>, gyro: Vector3<f64>) -> Self {
        Self { acc, gyro }
    }

    pub fn init() -> Self {
        Self {
            acc: Vector3::zeros(),
            gyro: Vector3::zeros(),
        }
    }
}

/// Get the state transition function f(x, u) Matrix
fn get_f(s: &StateIKFom, inp: &InputIKFom) -> SMatrix<f64, 18, 1> {
    let angular_vel = inp.gyro - s.bg; // bg is the estimated gyro bias
    // Use the rotation component from the SE3 pose
    let acc_inertial = s.pose.rotation().matrix() * (inp.acc - s.ba); // transform to world
    let acc_grav = acc_inertial + s.grav; // acceleration added gravity

    let mut res = SMatrix::<f64, 18, 1>::zeros();
    res.fixed_rows_mut::<3>(0).copy_from(&angular_vel); // pos_dot = vel
    res.fixed_rows_mut::<3>(3).copy_from(&s.vel); // rot_dot = angular_vel
    res.fixed_rows_mut::<3>(6).copy_from(&acc_grav); // vel_dot = acc_grav

    res
}

// Given a vector `q = [x, y, z]`, the skew-symmetric matrix is:
// ```
// [  0, -z,  y ]
// [  z,  0, -x ]
// [ -y,  x,  0 ]
// ```
pub fn skew_symmetric<T>(q: &Vector3<T>) -> Matrix3<T>
where
    T: nalgebra::Scalar + Copy + From<f64> + std::ops::Neg<Output = T>,
{
    Matrix3::new(
        T::from(0.0),
        -q[2],
        q[1],
        q[2],
        T::from(0.0),
        -q[0],
        -q[1],
        q[0],
        T::from(0.0),
    )
}

/// Get the Jacobian of the state transition function f(x, u) with respect to x. x is the state
pub fn df_dx(s: &StateIKFom, inp: &InputIKFom, dt: f64) -> SMatrix<f64, 18, 18> {
    let mut cov = SMatrix::<f64, 18, 18>::identity();

    // rot
    cov.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(Matrix3::identity() + s.pose.rotation().matrix() * skew_symmetric(&s.bg)));
    cov.fixed_view_mut::<3, 3>(0, 9)
        .copy_from(&(-s.pose.rotation().matrix() * dt));

    // pos
    cov.fixed_view_mut::<3, 3>(3, 0).copy_from(
        &(skew_symmetric(&(s.pose.translation() + s.vel * dt))
            * s.pose.rotation().matrix()
            * skew_symmetric(&s.bg)),
    );
    cov.fixed_view_mut::<3, 3>(3, 6)
        .copy_from(&(Matrix3::identity() * dt));
    cov.fixed_view_mut::<3, 3>(3, 9).copy_from(
        &(-dt * skew_symmetric(&(s.pose.translation() + s.vel * dt)) * s.pose.rotation().matrix()),
    );

    // vel
    cov.fixed_view_mut::<3, 3>(6, 0).copy_from(
        &(dt * s.pose.rotation().matrix() * skew_symmetric(&s.ba)
            - skew_symmetric(&(dt * s.pose.rotation().matrix() * inp.acc))
            + skew_symmetric(&s.vel) * s.pose.rotation().matrix() * skew_symmetric(&s.bg)
            + skew_symmetric(&((s.pose.rotation().matrix() * (inp.acc - s.ba) + s.grav) * dt))
                * (Matrix3::identity() + s.pose.rotation().matrix() * skew_symmetric(&s.bg))),
    );
    cov.fixed_view_mut::<3, 3>(6, 9).copy_from(
        &(-dt
            * skew_symmetric(
                &(s.vel + (s.pose.rotation().matrix() * (inp.acc - s.ba) + s.grav) * dt),
            )
            * s.pose.rotation().matrix()),
    );
    cov.fixed_view_mut::<3, 3>(6, 12)
        .copy_from(&(-dt * s.pose.rotation().matrix()));
    cov.fixed_view_mut::<3, 3>(6, 15)
        .copy_from(&(dt * Matrix3::identity()));

    // bg
    cov.fixed_view_mut::<3, 3>(9, 0)
        .copy_from(&(skew_symmetric(&s.bg) * s.pose.rotation().matrix() * skew_symmetric(&s.bg)));
    cov.fixed_view_mut::<3, 3>(9, 9).copy_from(
        &(Matrix3::identity() - dt * skew_symmetric(&s.bg) * s.pose.rotation().matrix()),
    );

    // ba
    cov.fixed_view_mut::<3, 3>(12, 0)
        .copy_from(&(skew_symmetric(&s.ba) * s.pose.rotation().matrix() * skew_symmetric(&s.bg)));
    cov.fixed_view_mut::<3, 3>(12, 9)
        .copy_from(&(-dt * skew_symmetric(&s.ba) * s.pose.rotation().matrix()));

    // g
    cov.fixed_view_mut::<3, 3>(15, 0)
        .copy_from(&(skew_symmetric(&s.grav) * s.pose.rotation().matrix() * skew_symmetric(&s.bg)));
    cov.fixed_view_mut::<3, 3>(15, 9)
        .copy_from(&(-dt * skew_symmetric(&s.grav) * s.pose.rotation().matrix()));

    cov
}

/// Get the Jacobian of the state transition function f(x, u) with respect to w. w is the noise
pub fn df_dw(s: &StateIKFom, inp: &InputIKFom, dt: f64) -> SMatrix<f64, 18, 12> {
    let mut cov = SMatrix::<f64, 18, 12>::zeros();

    // rot
    cov.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(-s.pose.rotation().matrix()));

    // pos
    cov.fixed_view_mut::<3, 3>(3, 0).copy_from(
        &(-skew_symmetric(&(s.pose.translation() + s.vel * dt)) * s.pose.rotation().matrix()),
    );

    // vel
    cov.fixed_view_mut::<3, 3>(6, 0).copy_from(
        &(-skew_symmetric(
            &(s.vel + (s.pose.rotation().matrix() * (inp.acc - s.ba) + s.grav) * dt),
        ) * s.pose.rotation().matrix()),
    );
    cov.fixed_view_mut::<3, 3>(6, 3)
        .copy_from(&(-s.pose.rotation().matrix()));

    // bg
    cov.fixed_view_mut::<3, 3>(9, 0)
        .copy_from(&(-skew_symmetric(&s.bg) * s.pose.rotation().matrix()));
    cov.fixed_view_mut::<3, 3>(9, 6)
        .copy_from(&(Matrix3::identity()));

    // ba
    cov.fixed_view_mut::<3, 3>(12, 0)
        .copy_from(&(-skew_symmetric(&s.ba) * s.pose.rotation().matrix()));
    cov.fixed_view_mut::<3, 3>(12, 9)
        .copy_from(&(Matrix3::identity()));

    // g
    cov.fixed_view_mut::<3, 3>(15, 0)
        .copy_from(&(-skew_symmetric(&s.grav) * s.pose.rotation().matrix()));

    cov
}

pub struct Iekf {
    pub state: StateIKFom,
    pub cov: SMatrix<f64, 18, 18>,
    pub initialized: bool,
    pub last_time: f64,
    pub h: DVector<f64>,
    pub h_x: DMatrix<f64>,
}

/// Adding the state and the perturbation, stay on the manifold
fn box_plus(x: &StateIKFom, f: &SMatrix<f64, 18, 1>) -> StateIKFom {
    let mut x_r = x.clone();

    let delta_rot_vec = f.fixed_rows::<3>(0);
    let delta_pos = f.fixed_rows::<3>(3);
    x_r.pose.set_translation(x.pose.translation() + delta_pos);
    x_r.pose.set_rotation(
        x.pose
            .rotation()
            .group_mul(SO3::exp(delta_rot_vec.clone_owned())),
    );

    x_r.vel = x.vel + f.fixed_rows::<3>(6);
    x_r.bg = x.bg + f.fixed_rows::<3>(9);
    x_r.ba = x.ba + f.fixed_rows::<3>(12);
    x_r.grav = x.grav + f.fixed_rows::<3>(15);

    x_r
}

/// Difference between two states, stay on the manifold
#[allow(dead_code)]
fn box_minus(x1: &StateIKFom, x2: &StateIKFom) -> SMatrix<f64, 18, 1> {
    let mut x_r = SMatrix::<f64, 18, 1>::zeros();

    x_r.fixed_rows_mut::<3>(0).copy_from(
        &x2.pose
            .rotation()
            .inverse()
            .group_mul(x1.pose.rotation())
            .log(),
    );
    x_r.fixed_rows_mut::<3>(3)
        .copy_from(&(x1.pose.translation() - x2.pose.translation()));

    x_r.fixed_rows_mut::<3>(6).copy_from(&(x1.vel - x2.vel));
    x_r.fixed_rows_mut::<3>(9).copy_from(&(x1.bg - x2.bg));
    x_r.fixed_rows_mut::<3>(12).copy_from(&(x1.ba - x2.ba));
    x_r.fixed_rows_mut::<3>(15).copy_from(&(x1.grav - x2.grav));

    x_r
}

impl Iekf {
    pub fn new(g: f64) -> Self {
        Self {
            state: StateIKFom::new(g),
            cov: SMatrix::<f64, 18, 18>::identity(),
            initialized: false,
            last_time: 0.0,
            h: DVector::<f64>::zeros(0),
            h_x: DMatrix::<f64>::zeros(0, 18),
        }
    }

    pub fn predict(&mut self, dt: f64, q: &SMatrix<f64, 12, 12>, i_in: &InputIKFom) {
        self.last_time += dt;

        let f = get_f(&self.state, i_in);
        let f_x = df_dx(&self.state, i_in, dt);
        let f_w = df_dw(&self.state, i_in, dt);

        // State update
        self.state = box_plus(&self.state, &(f * dt));
        // NOTE maybe try with it
        // f_x = SMatrix::<f64, 18, 18>::identity() + f_x * dt;

        // Cov update (we get less secure about our state after prediction without correction)
        self.cov = f_x * self.cov * f_x.transpose() + dt * dt * f_w * q * f_w.transpose();
    }

    pub fn correct_with_body_velocities(
        &mut self,
        odom: OdomMsg,
        kinematic: KinematicModel,
        cov_override: Option<f64>,
    ) {
        if !self.initialized {
            return;
        }

        let t = timestamp_to_secs(&odom.stamp);
        let dt_gap = t - self.last_time;
        if dt_gap < 0.0 {
            // use global log debouncer to avoid spamming
            if should_log_once_default("odometry_msg_older_than_filter") {
                warn!(
                    "Odometry message time {} is older than filter time {}. Skipping.",
                    t, self.last_time
                );
            }
            return;
        }
        self.last_time = t;

        let base_cov = cov_override.unwrap_or_else(|| {
            let cov_x = odom.twist_cov[0];
            let cov_y = odom.twist_cov[4];
            let cov_z = odom.twist_cov[8];
            cov_x.max(cov_y).max(cov_z).max(0.001)
        });

        let adjusted_base_cov = base_cov * (1.0 + dt_gap * 5.0);
        let r_wb = self.state.pose.rotation().matrix();
        let r_bw = r_wb.transpose();

        let v_meas_world = r_bw * odom.twist_linear;

        match kinematic {
            KinematicModel::Differential | KinematicModel::Ackermann { .. } => {
                self.correct_velocity_component(0, v_meas_world.x, adjusted_base_cov);
            }

            KinematicModel::Holonomic2D => {
                self.correct_velocity_component(0, v_meas_world.x, adjusted_base_cov);
                self.correct_velocity_component(1, v_meas_world.y, adjusted_base_cov);
            }

            KinematicModel::Full3D => {
                self.correct_velocity_component(0, v_meas_world.x, adjusted_base_cov);
                self.correct_velocity_component(1, v_meas_world.y, adjusted_base_cov);
                self.correct_velocity_component(2, v_meas_world.z, adjusted_base_cov);
            }
        }
    }

    fn apply_scalar_constraint(&mut self, h: SVector<f64, 18>, residual: f64, r_var: f64) {
        let s = (h.transpose() * self.cov * h)[0] + r_var;
        if !s.is_finite() || s <= 0.0 {
            warn!("Invalid innovation variance in kinematic constraint");
            return;
        }

        let mut k = (self.cov * h) / s;

        // zero out parts that must never be changed by this measurement
        // (rotation and translation live in the first 6 entries)
        k.fixed_rows_mut::<3>(0).fill(0.0); // rotation
        k.fixed_rows_mut::<3>(3).fill(0.0); // translation

        // automatic damping
        // cap the norm of the gain (prevents gigantic jumps)
        const K_MAX: f64 = 0.8;
        let gain_norm = k.norm();
        if gain_norm > K_MAX {
            let scale = K_MAX / gain_norm;
            k *= scale;
        }

        let dx = k * residual;
        self.state = box_plus(&self.state, &dx);

        let i_kh = SMatrix::<f64, 18, 18>::identity() - k * h.transpose();
        self.cov = i_kh * self.cov * i_kh.transpose() + k * r_var * k.transpose();
    }

    fn correct_velocity_component(
        &mut self,
        component: usize,
        measured_value: f64,
        noise_cov: f64,
    ) {
        if component > 2 {
            return;
        }

        let residual = measured_value - self.state.vel[component];

        let p_comp = self.cov[(6 + component, 6 + component)];
        let innovation_var = p_comp + noise_cov;

        if !innovation_var.is_finite() || innovation_var <= 0.0 {
            warn!(
                "Invalid innovation variance in velocity[{}] correction",
                component
            );
            return;
        }

        let gain_scalar = p_comp / innovation_var;

        let mut kalman_gain = SVector::<f64, 18>::zeros();
        for i in 0..18 {
            kalman_gain[i] = self.cov[(i, 6 + component)] * gain_scalar;
        }
        let delta_v = gain_scalar * residual;

        if !delta_v.is_finite() {
            warn!("NaN in velocity[{}] update", component);
            return;
        }

        self.state.vel[component] += delta_v;

        for i in 0..18 {
            for j in 0..18 {
                self.cov[(i, j)] -= kalman_gain[i] * self.cov[(6 + component, j)];
            }
        }
    }

    pub fn correct_kinematics(
        &mut self,
        model_type: KinematicModel,
        gyro_measurement: Vector3<f64>,
        constraint_noise_std: f64,
    ) {
        if let KinematicModel::Full3D = model_type {
            return;
        }

        let r_w_i = self.state.pose.rotation().matrix();
        let r_i_w = r_w_i.transpose();
        let v_body = r_i_w * self.state.vel;
        let r_var = constraint_noise_std * constraint_noise_std;

        match model_type {
            KinematicModel::Differential => {
                // lateral (y) constraint
                let mut h_lat = SVector::<f64, 18>::zeros();
                for j in 0..3 {
                    h_lat[6 + j] = r_i_w[(1, j)];
                }
                self.apply_scalar_constraint(h_lat, -v_body.y, r_var);

                // vertical (z) constraint
                let mut h_vert = SVector::<f64, 18>::zeros();
                for j in 0..3 {
                    h_vert[6 + j] = r_i_w[(2, j)];
                }
                self.apply_scalar_constraint(h_vert, -v_body.z, r_var);
            }
            KinematicModel::Holonomic2D => {
                // Only vertical (z) constraint: robot stays on ground
                let mut h_vert = SVector::<f64, 18>::zeros();
                for j in 0..3 {
                    h_vert[6 + j] = r_i_w[(2, j)];
                }
                self.apply_scalar_constraint(h_vert, -v_body.z, r_var);
            }

            KinematicModel::Ackermann { imu_to_rear_axle } => {
                // rear‑axle lateral constraint
                let v_rear = v_body + gyro_measurement.cross(&imu_to_rear_axle);
                let mut h_lat = SVector::<f64, 18>::zeros();
                for j in 0..3 {
                    h_lat[6 + j] = r_i_w[(1, j)];
                }
                self.apply_scalar_constraint(h_lat, -v_rear.y, r_var);

                // vertical constraint
                let mut h_vert = SVector::<f64, 18>::zeros();
                for j in 0..3 {
                    h_vert[6 + j] = r_i_w[(2, j)];
                }
                self.apply_scalar_constraint(h_vert, -v_body.z, r_var);
            }

            KinematicModel::Full3D => {}
        }
    }

    pub fn correct(
        &mut self,
        measurement_noise_cov: f64,
        measurement_time: f64,
        max_iter: usize,
        eval: std::sync::Arc<std::sync::RwLock<&mut RuntimeEvaluator>>,
        mut model: impl FnMut(
            &StateIKFom,
            &mut bool,         // valid
            bool,              // converge
            &mut DVector<f64>, // residual h
            &mut DMatrix<f64>, // jacobian H
        ),
    ) {
        self.last_time = measurement_time;

        if measurement_noise_cov <= 0.0 {
            error!("IEKF: Non‑positive measurement noise covariance, skipping update.");
            return;
        }

        let state_propagated = self.state.clone();
        let p_propagated = self.cov;

        let p_propagated_inv = match p_propagated.try_inverse() {
            Some(inv) => inv,
            None => {
                warn!("IEKF: Propagated covariance is singular, skipping update.");
                return;
            }
        };

        let r_inv = 1.0 / measurement_noise_cov;

        let mut final_kh = SMatrix::<f64, 18, 18>::zeros();

        for iter in 0..max_iter {
            let mut valid = true;
            let converge_flag = true;

            {
                let mut eval = eval.write().unwrap();
                eval.start("ekf_correct/model").unwrap();
            }

            model(
                &self.state,
                &mut valid,
                converge_flag,
                &mut self.h,
                &mut self.h_x,
            );

            let mut eval = eval.write().unwrap();
            eval.stop("ekf_correct/model").unwrap();

            if !valid || self.h_x.nrows() == 0 {
                break;
            }

            eval.start("ekf_correct/info_mat").unwrap();
            // H: m x 18,  h: m x 1
            let h_t = self.h_x.transpose(); // 18 x m
            let hth = &h_t * &self.h_x; // 18 x 18
            eval.stop("ekf_correct/info_mat").unwrap();

            let lhs = hth * r_inv + p_propagated_inv;
            let lhs_inv = match lhs.try_inverse() {
                Some(inv) => inv,
                None => {
                    warn!("IEKF: LHS solve failed in iteration {}, skipping.", iter);
                    break;
                }
            };

            eval.start("ekf_correct/gain").unwrap();
            let k = &lhs_inv * &h_t * r_inv;

            let kh = &k * &self.h;

            let kh_norm = kh.norm();
            if !kh_norm.is_finite() {
                warn!(
                    "IEKF: Non‑finite update norm in iteration {}, aborting.",
                    iter
                );
                self.state = state_propagated;
                self.cov = p_propagated;
                eval.stop("ekf_correct/gain").unwrap();
                return;
            }
            eval.stop("ekf_correct/gain").unwrap();
            eval.start("ekf_correct/state").unwrap();

            let mut s = SMatrix::<f64, 8, 8>::zeros();
            // rotation increment
            s.fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&skew_symmetric(&kh.fixed_rows::<3>(0).into_owned()));
            // translation increment
            s.fixed_view_mut::<3, 1>(0, 3)
                .copy_from(&kh.fixed_rows::<3>(3).into_owned());
            // velocity increment
            s.fixed_view_mut::<3, 1>(0, 4)
                .copy_from(&kh.fixed_rows::<3>(6).into_owned());
            // bg increment
            s.fixed_view_mut::<3, 1>(0, 5)
                .copy_from(&kh.fixed_rows::<3>(9).into_owned());
            // ba increment
            s.fixed_view_mut::<3, 1>(0, 6)
                .copy_from(&kh.fixed_rows::<3>(12).into_owned());
            // grav increment
            s.fixed_view_mut::<3, 1>(0, 7)
                .copy_from(&kh.fixed_rows::<3>(15).into_owned());

            let (a, b) = if kh_norm > 1e-6 {
                (
                    (1.0 - kh_norm.cos()) / (kh_norm * kh_norm),
                    (kh_norm - kh_norm.sin()) / (kh_norm.powi(3)),
                )
            } else {
                (0.5, 1.0 / 6.0)
            };

            let s2 = s * s;
            let s3 = s2 * s;
            let exps = SMatrix::<f64, 8, 8>::identity() + s + s2 * a + s3 * b;

            let mut x_mat = SMatrix::<f64, 8, 8>::identity();
            x_mat
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&state_propagated.pose.rotation().matrix());
            x_mat
                .fixed_view_mut::<3, 1>(0, 3)
                .copy_from(&state_propagated.pose.translation());
            x_mat
                .fixed_view_mut::<3, 1>(0, 4)
                .copy_from(&state_propagated.vel);
            x_mat
                .fixed_view_mut::<3, 1>(0, 5)
                .copy_from(&state_propagated.bg);
            x_mat
                .fixed_view_mut::<3, 1>(0, 6)
                .copy_from(&state_propagated.ba);
            x_mat
                .fixed_view_mut::<3, 1>(0, 7)
                .copy_from(&state_propagated.grav);

            let new_x_mat = exps * x_mat;

            if !new_x_mat.iter().all(|x| x.is_finite()) {
                warn!("IEKF: NaN detected in iteration {}, aborting.", iter);
                self.state = state_propagated;
                self.cov = p_propagated;
                eval.stop("ekf_correct/state").unwrap();
                return;
            }

            // Extract rotation and state fields
            let new_rot_mat = new_x_mat.fixed_view::<3, 3>(0, 0).into_owned();
            let new_quat = UnitQuaternion::from_matrix(&new_rot_mat);
            let new_so3 = match SO3::try_from_mat(new_quat.to_rotation_matrix().matrix()) {
                Some(so3) => so3,
                None => {
                    warn!("IEKF: Invalid SO3 in iteration {}", iter);
                    break;
                }
            };

            self.state.pose.set_rotation(new_so3);
            self.state
                .pose
                .set_translation(new_x_mat.fixed_view::<3, 1>(0, 3).into_owned());
            self.state.vel = new_x_mat.fixed_view::<3, 1>(0, 4).into_owned();
            self.state.bg = new_x_mat.fixed_view::<3, 1>(0, 5).into_owned();
            self.state.ba = new_x_mat.fixed_view::<3, 1>(0, 6).into_owned();
            self.state.grav = new_x_mat.fixed_view::<3, 1>(0, 7).into_owned();

            // KH = K * H  (18 x m * m x 18 = 18 x 18, fixed-size)
            let kh_mat = &k * &self.h_x;
            debug_assert!(kh_mat.nrows() == 18 && kh_mat.ncols() == 18);
            let mut kh_mat_fixed = SMatrix::<f64, 18, 18>::zeros();
            for i in 0..18 {
                for j in 0..18 {
                    kh_mat_fixed[(i, j)] = kh_mat[(i, j)];
                }
            }
            final_kh = kh_mat_fixed;

            eval.stop("ekf_correct/state").unwrap();
            //dbg!(iter, kh_norm);
            if kh_norm < 0.001 {
                break;
            }
        }

        self.cov = p_propagated - final_kh * p_propagated;
    }
}

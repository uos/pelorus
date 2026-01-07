use faer::linalg::matmul::matmul;
use faer::linalg::solvers::DenseSolveCore;
use faer::prelude::{Reborrow, ReborrowMut};
use faer::{Col, Mat};
use log::{error, warn};
use nalgebra::Vector3; // Keep for sophus_lie interface
use pelorus_core::{
    KinematicModel, OdomMsg, log_debouncer::should_log_once_default, timing::RuntimeEvaluator,
};
use prelio::timestamp_to_secs;

use sophus_lie::prelude::*;

pub type SO3 = sophus_lie::Rotation3F64;
pub type SE3 = sophus_lie::Isometry3F64;

// Helper to convert nalgebra Matrix3 from sophus to faer
#[inline]
fn mat3_to_faer(m: &nalgebra::Matrix3<f64>) -> Mat<f64> {
    Mat::from_fn(3, 3, |i, j| m[(i, j)])
}

// TODO use config, needs some refactor
const GRAVITY_DEFAULT: f64 = 9.81;

#[derive(Clone, Debug, Copy)]
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
fn get_f(s: &StateIKFom, inp: &InputIKFom) -> Mat<f64> {
    let angular_vel = inp.gyro - s.bg; // bg is the estimated gyro bias
    // Use the rotation component from the SE3 pose
    let rot_mat = mat3_to_faer(&s.pose.rotation().matrix());
    let acc_diff = inp.acc - s.ba;
    // Convert to faer col for matrix multiply
    let acc_diff_col = faer::col![acc_diff.x, acc_diff.y, acc_diff.z];
    let acc_inertial = &rot_mat * &acc_diff_col; // transform to world
    let grav_col = faer::col![s.grav.x, s.grav.y, s.grav.z];
    let acc_grav = &acc_inertial + &grav_col; // acceleration added gravity

    let mut res = Mat::<f64>::zeros(18, 1);
    // Copy angular_vel to rows 0-2
    res[(0, 0)] = angular_vel.x;
    res[(1, 0)] = angular_vel.y;
    res[(2, 0)] = angular_vel.z;
    // Copy vel to rows 3-5
    res[(3, 0)] = s.vel.x;
    res[(4, 0)] = s.vel.y;
    res[(5, 0)] = s.vel.z;
    // Copy acc_grav to rows 6-8
    res[(6, 0)] = acc_grav[0];
    res[(7, 0)] = acc_grav[1];
    res[(8, 0)] = acc_grav[2];

    res
}

// Given a vector `q = [x, y, z]`, the skew-symmetric matrix is:
// ```
// [  0, -z,  y ]
// [  z,  0, -x ]
// [ -y,  x,  0 ]
// ```
pub fn skew_symmetric(q: &Vector3<f64>) -> Mat<f64> {
    let mut m = Mat::<f64>::zeros(3, 3);
    m[(0, 1)] = -q[2];
    m[(0, 2)] = q[1];
    m[(1, 0)] = q[2];
    m[(1, 2)] = -q[0];
    m[(2, 0)] = -q[1];
    m[(2, 1)] = q[0];
    m
}

/// Get the Jacobian of the state transition function f(x, u) with respect to x. x is the state
pub fn df_dx(s: &StateIKFom, inp: &InputIKFom, dt: f64) -> Mat<f64> {
    let mut cov = Mat::<f64>::identity(18, 18);

    // Get rotation matrix from sophus
    let rot_mat = mat3_to_faer(&s.pose.rotation().matrix());
    let id3 = Mat::<f64>::identity(3, 3);

    // Helper to copy 3x3 block into larger matrix
    let copy_block = |dest: &mut Mat<f64>, row: usize, col: usize, src: &Mat<f64>| {
        for i in 0..3 {
            for j in 0..3 {
                dest[(row + i, col + j)] = src[(i, j)];
            }
        }
    };

    // rot block (0, 0)
    let rot_block = &id3 + &rot_mat * &skew_symmetric(&s.bg);
    copy_block(&mut cov, 0, 0, &rot_block);

    // rot block (0, 9)
    let rot_block2 = &rot_mat * (-dt);
    copy_block(&mut cov, 0, 9, &rot_block2);

    // pos block (3, 0)
    let trans_vel = s.pose.translation() + s.vel * dt;
    let pos_block = &skew_symmetric(&trans_vel) * &rot_mat * &skew_symmetric(&s.bg);
    copy_block(&mut cov, 3, 0, &pos_block);

    // pos block (3, 6)
    let pos_block2 = &id3 * dt;
    copy_block(&mut cov, 3, 6, &pos_block2);

    // pos block (3, 9)
    let pos_block3 = &skew_symmetric(&trans_vel) * &rot_mat * (-dt);
    copy_block(&mut cov, 3, 9, &pos_block3);

    // vel blocks are more complex
    let acc_diff = inp.acc - s.ba;
    let rot_acc = &s.pose.rotation().matrix() * acc_diff;
    let acc_grav_dt = (rot_acc + s.grav) * dt;

    // vel block (6, 0)
    let rot_inp_acc_dt = &s.pose.rotation().matrix() * inp.acc * dt;
    let vel_block = &rot_mat * &skew_symmetric(&s.ba) * dt - &skew_symmetric(&rot_inp_acc_dt)
        + &skew_symmetric(&s.vel) * &rot_mat * &skew_symmetric(&s.bg)
        + &skew_symmetric(&acc_grav_dt) * (&id3 + &rot_mat * &skew_symmetric(&s.bg));
    copy_block(&mut cov, 6, 0, &vel_block);

    // vel block (6, 9)
    let vel_grav = s.vel + acc_grav_dt;
    let vel_block2 = &skew_symmetric(&vel_grav) * &rot_mat * (-dt);
    copy_block(&mut cov, 6, 9, &vel_block2);

    // vel block (6, 12)
    let vel_block3 = &rot_mat * (-dt);
    copy_block(&mut cov, 6, 12, &vel_block3);

    // vel block (6, 15)
    let vel_block4 = &id3 * dt;
    copy_block(&mut cov, 6, 15, &vel_block4);

    // bg block (9, 0)
    let bg_block = &skew_symmetric(&s.bg) * &rot_mat * &skew_symmetric(&s.bg);
    copy_block(&mut cov, 9, 0, &bg_block);

    // bg block (9, 9)
    let bg_block2 = &id3 - &skew_symmetric(&s.bg) * &rot_mat * dt;
    copy_block(&mut cov, 9, 9, &bg_block2);

    // ba block (12, 0)
    let ba_block = &skew_symmetric(&s.ba) * &rot_mat * &skew_symmetric(&s.bg);
    copy_block(&mut cov, 12, 0, &ba_block);

    // ba block (12, 9)
    let ba_block2 = &skew_symmetric(&s.ba) * &rot_mat * (-dt);
    copy_block(&mut cov, 12, 9, &ba_block2);

    // g block (15, 0)
    let g_block = &skew_symmetric(&s.grav) * &rot_mat * &skew_symmetric(&s.bg);
    copy_block(&mut cov, 15, 0, &g_block);

    // g block (15, 9)
    let g_block2 = &skew_symmetric(&s.grav) * &rot_mat * (-dt);
    copy_block(&mut cov, 15, 9, &g_block2);

    cov
}

/// Get the Jacobian of the state transition function f(x, u) with respect to w. w is the noise
pub fn df_dw(s: &StateIKFom, inp: &InputIKFom, dt: f64) -> Mat<f64> {
    let mut cov = Mat::<f64>::zeros(18, 12);

    let rot_mat = mat3_to_faer(&s.pose.rotation().matrix());
    let id3 = Mat::<f64>::identity(3, 3);

    // Helper to copy 3x3 block
    let copy_block = |dest: &mut Mat<f64>, row: usize, col: usize, src: &Mat<f64>| {
        for i in 0..3 {
            for j in 0..3 {
                dest[(row + i, col + j)] = src[(i, j)];
            }
        }
    };

    // rot block (0, 0)
    let rot_block = &rot_mat * (-1.0);
    copy_block(&mut cov, 0, 0, &rot_block);

    // pos block (3, 0)
    let trans_vel = s.pose.translation() + s.vel * dt;
    let pos_block = &skew_symmetric(&trans_vel) * &rot_mat * (-1.0);
    copy_block(&mut cov, 3, 0, &pos_block);

    // vel block (6, 0)
    let acc_diff = inp.acc - s.ba;
    let rot_acc = &s.pose.rotation().matrix() * acc_diff;
    let vel_grav_dt = s.vel + (rot_acc + s.grav) * dt;
    let vel_block = &skew_symmetric(&vel_grav_dt) * &rot_mat * (-1.0);
    copy_block(&mut cov, 6, 0, &vel_block);

    // vel block (6, 3)
    let vel_block2 = &rot_mat * (-1.0);
    copy_block(&mut cov, 6, 3, &vel_block2);

    // bg block (9, 0)
    let bg_block = &skew_symmetric(&s.bg) * &rot_mat * (-1.0);
    copy_block(&mut cov, 9, 0, &bg_block);

    // bg block (9, 6)
    copy_block(&mut cov, 9, 6, &id3);

    // ba block (12, 0)
    let ba_block = &skew_symmetric(&s.ba) * &rot_mat * (-1.0);
    copy_block(&mut cov, 12, 0, &ba_block);

    // ba block (12, 9)
    copy_block(&mut cov, 12, 9, &id3);

    // g block (15, 0)
    let g_block = &skew_symmetric(&s.grav) * &rot_mat * (-1.0);
    copy_block(&mut cov, 15, 0, &g_block);

    cov
}

pub struct Iekf {
    pub state: StateIKFom,
    pub cov: Mat<f64>,           // 18x18 covariance matrix in faer
    pub f_w_q_w_t_dt2: Mat<f64>, // 18x18 temporary for cov update in predict
    pub fxtcov: Mat<f64>,        // 18x18 temporary for cov update in predict
    pub fxcov: Mat<f64>,         // 18x18 temporary for cov update in predict
    pub q_f_w_t: Mat<f64>,       // 12x18 temporary for cov update in predict
    pub id: Mat<f64>,            // 18x18 temporary for cov update in scalar constraint
    pub k_mat: Mat<f64>,         // 18x1 temporary for cov update in scalar constraint
    pub h_t_mat: Mat<f64>,       // 1x18 temporary for cov update in scalar constraint
    pub k_h_t: Mat<f64>,         // 18x18 temporary for cov update in scalar constraint
    pub k_k: Mat<f64>,           // 18x18 temporary for cov update in scalar constraint
    pub cov_i_kh: Mat<f64>,      // 18x18 temporary for cov update in scalar constraint
    pub i_cov_i_kh: Mat<f64>,    // 18x18 temporary for cov update in scalar constraint
    pub initialized: bool,
    pub last_time: f64,
    pub h: Col<f64>,   // measurement residual vector
    pub h_x: Mat<f64>, // measurement Jacobian matrix
    pub k: Mat<f64>,
    pub k_h: Mat<f64>,
}

/// Adding the state and the perturbation, stay on the manifold
fn box_plus(x: &StateIKFom, f: &Mat<f64>) -> StateIKFom {
    let mut x_r = *x;

    // Extract deltas from the faer matrix (f is 18x1)
    let delta_rot_vec = Vector3::new(f[(0, 0)], f[(1, 0)], f[(2, 0)]);
    let delta_pos = Vector3::new(f[(3, 0)], f[(4, 0)], f[(5, 0)]);

    x_r.pose.set_translation(x.pose.translation() + delta_pos);
    x_r.pose
        .set_rotation(x.pose.rotation().group_mul(SO3::exp(delta_rot_vec)));

    x_r.vel = x.vel + Vector3::new(f[(6, 0)], f[(7, 0)], f[(8, 0)]);
    x_r.bg = x.bg + Vector3::new(f[(9, 0)], f[(10, 0)], f[(11, 0)]);
    x_r.ba = x.ba + Vector3::new(f[(12, 0)], f[(13, 0)], f[(14, 0)]);
    x_r.grav = x.grav + Vector3::new(f[(15, 0)], f[(16, 0)], f[(17, 0)]);

    x_r
}

/// Difference between two states, stay on the manifold
#[allow(dead_code)]
fn box_minus(x1: &StateIKFom, x2: &StateIKFom) -> Mat<f64> {
    let mut x_r = Mat::<f64>::zeros(18, 1);

    let rot_diff = x2
        .pose
        .rotation()
        .inverse()
        .group_mul(x1.pose.rotation())
        .log();

    // Copy rotation difference
    x_r[(0, 0)] = rot_diff.x;
    x_r[(1, 0)] = rot_diff.y;
    x_r[(2, 0)] = rot_diff.z;

    // Position difference
    let pos_diff = x1.pose.translation() - x2.pose.translation();
    x_r[(3, 0)] = pos_diff.x;
    x_r[(4, 0)] = pos_diff.y;
    x_r[(5, 0)] = pos_diff.z;

    // Velocity difference
    let vel_diff = x1.vel - x2.vel;
    x_r[(6, 0)] = vel_diff.x;
    x_r[(7, 0)] = vel_diff.y;
    x_r[(8, 0)] = vel_diff.z;

    // Gyro bias difference
    let bg_diff = x1.bg - x2.bg;
    x_r[(9, 0)] = bg_diff.x;
    x_r[(10, 0)] = bg_diff.y;
    x_r[(11, 0)] = bg_diff.z;

    // Accelerometer bias difference
    let ba_diff = x1.ba - x2.ba;
    x_r[(12, 0)] = ba_diff.x;
    x_r[(13, 0)] = ba_diff.y;
    x_r[(14, 0)] = ba_diff.z;

    // Gravity difference
    let grav_diff = x1.grav - x2.grav;
    x_r[(15, 0)] = grav_diff.x;
    x_r[(16, 0)] = grav_diff.y;
    x_r[(17, 0)] = grav_diff.z;

    x_r
}

impl Iekf {
    pub fn new(g: f64) -> Self {
        Self {
            state: StateIKFom::new(g),
            cov: Mat::<f64>::identity(18, 18),
            initialized: false,
            last_time: 0.0,
            h: Col::<f64>::zeros(0),
            h_x: Mat::<f64>::zeros(0, 18),
            k: Mat::<f64>::zeros(18, 10000),
            k_h: Mat::<f64>::zeros(18, 18),
            f_w_q_w_t_dt2: Mat::<f64>::zeros(18, 18),
            fxtcov: Mat::<f64>::zeros(18, 18),
            fxcov: Mat::<f64>::zeros(18, 18),
            q_f_w_t: Mat::<f64>::zeros(12, 18),
            id: Mat::<f64>::identity(18, 18),
            k_mat: Mat::<f64>::zeros(18, 1),
            h_t_mat: Mat::<f64>::zeros(1, 18),
            k_h_t: Mat::<f64>::zeros(18, 18),
            k_k: Mat::<f64>::zeros(18, 18),
            cov_i_kh: Mat::<f64>::zeros(18, 18),
            i_cov_i_kh: Mat::<f64>::zeros(18, 18),
        }
    }

    pub fn predict(&mut self, dt: f64, q: &Mat<f64>, i_in: &InputIKFom) {
        self.last_time += dt;

        let f = get_f(&self.state, i_in);
        let f_x = df_dx(&self.state, i_in, dt);
        let f_w = df_dw(&self.state, i_in, dt);

        // State update: f * dt
        let f_dt = &f * dt;
        self.state = box_plus(&self.state, &f_dt); // seems fast enough

        // Cov update
        // P = F * P * F^T + dt^2 * F_w * Q * F_w^T
        let f_x_t = f_x.transpose();
        let f_w_t = f_w.transpose();

        matmul(
            self.q_f_w_t.rb_mut(),
            faer::Accum::Replace,
            q.rb(),
            f_w_t.rb(),
            1.0,
            faer::Par::Seq,
        );
        matmul(
            self.f_w_q_w_t_dt2.rb_mut(),
            faer::Accum::Replace,
            f_w.rb(),
            self.q_f_w_t.rb(),
            dt * dt,
            faer::Par::Seq,
        );
        matmul(
            self.fxtcov.rb_mut(),
            faer::Accum::Replace,
            self.cov.rb(),
            f_x_t.rb(),
            1.0,
            faer::Par::Seq,
        );
        matmul(
            self.fxcov.rb_mut(),
            faer::Accum::Replace,
            f_x.rb(),
            self.fxtcov.rb(),
            1.0,
            faer::Par::Seq,
        );
        // self.cov = &f_x * &self.cov * &f_x_t + dt * dt * (&f_w * q * &f_w_t);
        self.cov = self.fxcov.rb() + self.f_w_q_w_t_dt2.rb();
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

    fn apply_scalar_constraint(&mut self, h: Col<f64>, residual: f64, r_var: f64) {
        // Compute innovation variance: h^T * P * h + r_var
        let h_t = h.transpose();
        let cov_h = &self.cov * &h;
        // h_t is 1x18, cov_h is Col (18x1), result is a scalar
        // Compute h^T * cov_h manually
        let mut s = 0.0;
        for i in 0..18 {
            s += h_t[i] * cov_h[i];
        }
        s += r_var;

        if !s.is_finite() || s <= 0.0 {
            warn!("Invalid innovation variance in kinematic constraint");
            return;
        }

        // Kalman gain: K = P * h / s (result is 18x1)
        let mut k = &cov_h / s;

        // zero out parts that must never be changed by this measurement
        // (rotation and translation live in the first 6 entries)
        for i in 0..3 {
            k[i] = 0.0; // rotation
        }
        for i in 3..6 {
            k[i] = 0.0; // translation
        }

        // automatic damping - cap the norm of the gain (prevents gigantic jumps)
        const K_MAX: f64 = 0.8;
        let gain_norm = k.norm_l2();
        if gain_norm > K_MAX {
            let scale = K_MAX / gain_norm;
            k = &k * scale;
        }

        // Update state: dx = k * residual (k is 18x1, result is 18x1)
        let dx = &k * residual;

        // Convert Col to Mat for box_plus
        let mut dx_mat = Mat::<f64>::zeros(18, 1);
        for i in 0..18 {
            dx_mat[(i, 0)] = dx[i];
        }
        self.state = box_plus(&self.state, &dx_mat);

        // Update covariance: P = (I - K*h^T) * P * (I - K*h^T)^T + K * r_var * K^T
        for i in 0..18 {
            self.k_mat[(i, 0)] = k[i];
        }
        for j in 0..18 {
            self.h_t_mat[(0, j)] = h_t[j];
        }
        matmul(
            self.k_h_t.rb_mut(),
            faer::Accum::Replace,
            self.k_mat.rb(),
            self.h_t_mat.rb(),
            1.0,
            faer::Par::Seq,
        );
        let i_kh = &self.id - &self.k_h_t;
        matmul(
            self.k_k.rb_mut(),
            faer::Accum::Replace,
            self.k_mat.rb(),
            self.k_mat.transpose().rb(),
            r_var,
            faer::Par::Seq,
        );
        matmul(
            self.cov_i_kh.rb_mut(),
            faer::Accum::Replace,
            self.cov.rb(),
            i_kh.transpose().rb(),
            1.0,
            faer::Par::Seq,
        );
        matmul(
            self.i_cov_i_kh.rb_mut(),
            faer::Accum::Replace,
            i_kh.rb(),
            self.cov_i_kh.rb(),
            1.0,
            faer::Par::Seq,
        );
        // self.cov = i_kh * self.cov * i_kh.transpose() + k * r_var * k.transpose();
        self.cov = &self.i_cov_i_kh + &self.k_k;
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

        let mut kalman_gain = Col::<f64>::zeros(18);
        for i in 0..18 {
            kalman_gain[i] = self.cov[(i, 6 + component)] * gain_scalar;
        }
        let delta_v = gain_scalar * residual;

        if !delta_v.is_finite() {
            warn!("NaN in velocity[{}] update", component);
            return;
        }

        self.state.vel[component] += delta_v;

        // Update covariance
        for i in 0..18 {
            for j in 0..18 {
                let old_val = self.cov[(i, j)];
                let update = kalman_gain[i] * self.cov[(6 + component, j)];
                self.cov[(i, j)] = old_val - update;
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
                let mut h_lat = Col::<f64>::zeros(18);
                for j in 0..3 {
                    h_lat[6 + j] = r_i_w[(1, j)];
                }
                self.apply_scalar_constraint(h_lat, -v_body.y, r_var);

                // vertical (z) constraint
                let mut h_vert = Col::<f64>::zeros(18);
                for j in 0..3 {
                    h_vert[6 + j] = r_i_w[(2, j)];
                }
                self.apply_scalar_constraint(h_vert, -v_body.z, r_var);
            }
            KinematicModel::Holonomic2D => {
                // Only vertical (z) constraint: robot stays on ground
                let mut h_vert = Col::<f64>::zeros(18);
                for j in 0..3 {
                    h_vert[6 + j] = r_i_w[(2, j)];
                }
                self.apply_scalar_constraint(h_vert, -v_body.z, r_var);
            }

            KinematicModel::Ackermann { imu_to_rear_axle } => {
                // rear‑axle lateral constraint
                let v_rear = v_body + gyro_measurement.cross(&imu_to_rear_axle);
                let mut h_lat = Col::<f64>::zeros(18);
                for j in 0..3 {
                    h_lat[6 + j] = r_i_w[(1, j)];
                }
                self.apply_scalar_constraint(h_lat, -v_rear.y, r_var);

                // vertical constraint
                let mut h_vert = Col::<f64>::zeros(18);
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
            &mut bool,     // valid
            bool,          // converge
            &mut Col<f64>, // residual h (updated to faer)
            &mut Mat<f64>, // jacobian H (updated to faer)
        ),
    ) {
        self.last_time = measurement_time;

        if measurement_noise_cov <= 0.0 {
            error!("IEKF: Non‑positive measurement noise covariance, skipping update.");
            return;
        }

        let state_propagated = self.state;
        let p_propagated = self.cov.clone();

        let p_propagated_inv = p_propagated.partial_piv_lu().inverse();

        let r_inv = 1.0 / measurement_noise_cov;

        let mut hth = Mat::<f64>::zeros(18, 18);

        // Build 8x8 matrix s for state update
        let mut s = Mat::<f64>::zeros(8, 8);

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
            eval.start("ekf_correct/info_mat").unwrap();

            if !valid || self.h_x.nrows() == 0 {
                eval.stop("ekf_correct/info_mat").unwrap();
                break;
            }

            // pre-allocate k if needed
            if self.k.ncols() < self.h_x.nrows() {
                self.k.reserve(18, self.h_x.nrows());
            }
            if self.k.ncols() != self.h_x.nrows() {
                self.k.resize_with(18, self.h_x.nrows(), |_, _| 0.0);
            }

            // H: m x 18,  h: m x 1
            let h_t = self.h_x.transpose(); // 18 x m, a view change, basically free
            matmul(
                hth.rb_mut(), // 18 x 18 mat
                faer::Accum::Replace,
                h_t.rb(),
                self.h_x.rb(),
                1.0,
                faer::Par::Seq, // single thread, TODO measure when to go to par and rayon feature enabled
            );

            eval.stop("ekf_correct/info_mat").unwrap();
            eval.start("ekf_correct/lhs").unwrap();

            let lhs = &hth * r_inv + &p_propagated_inv; // free
            let lhs_inv = lhs.partial_piv_lu().inverse(); // 18 x 18, free

            matmul(
                self.k.rb_mut(),
                faer::Accum::Replace,
                lhs_inv.rb(),
                h_t.rb(),
                r_inv,
                faer::Par::Seq, // single thread
            );
            eval.stop("ekf_correct/lhs").unwrap();

            eval.start("ekf_correct/gain").unwrap();
            let kh = &self.k * &self.h; // 18 x 18, multiplies a lot but automatically fast enough (100us)
            let kh_norm = kh.norm_l2();
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

            // Extract kh components (kh is a Col<f64>, indexed with single index)
            let kh_rot = Vector3::new(kh[0], kh[1], kh[2]);
            let kh_trans = Vector3::new(kh[3], kh[4], kh[5]);
            let kh_vel = Vector3::new(kh[6], kh[7], kh[8]);
            let kh_bg = Vector3::new(kh[9], kh[10], kh[11]);
            let kh_ba = Vector3::new(kh[12], kh[13], kh[14]);
            let kh_grav = Vector3::new(kh[15], kh[16], kh[17]);

            // rotation increment (skew symmetric)
            let skew_rot = skew_symmetric(&kh_rot);
            for i in 0..3 {
                for j in 0..3 {
                    s[(i, j)] = skew_rot[(i, j)];
                }
            }

            // translation increment
            for i in 0..3 {
                s[(i, 3)] = kh_trans[i];
            }

            // velocity increment
            for i in 0..3 {
                s[(i, 4)] = kh_vel[i];
            }

            // bg increment
            for i in 0..3 {
                s[(i, 5)] = kh_bg[i];
            }

            // ba increment
            for i in 0..3 {
                s[(i, 6)] = kh_ba[i];
            }

            // grav increment
            for i in 0..3 {
                s[(i, 7)] = kh_grav[i];
            }

            let (a, b) = if kh_norm > 1e-6 {
                (
                    (1.0 - kh_norm.cos()) / (kh_norm * kh_norm),
                    (kh_norm - kh_norm.sin()) / (kh_norm.powi(3)),
                )
            } else {
                (0.5, 1.0 / 6.0)
            };

            let s2 = &s * &s;
            let s3 = &s2 * &s;
            let id8 = Mat::<f64>::identity(8, 8);
            let exps = &id8 + &s + &s2 * a + &s3 * b;

            let mut x_mat = Mat::<f64>::identity(8, 8);
            let rot_mat = mat3_to_faer(&state_propagated.pose.rotation().matrix());
            for i in 0..3 {
                for j in 0..3 {
                    x_mat[(i, j)] = rot_mat[(i, j)];
                }
            }
            let trans = state_propagated.pose.translation();
            for i in 0..3 {
                x_mat[(i, 3)] = trans[i];
                x_mat[(i, 4)] = state_propagated.vel[i];
                x_mat[(i, 5)] = state_propagated.bg[i];
                x_mat[(i, 6)] = state_propagated.ba[i];
                x_mat[(i, 7)] = state_propagated.grav[i];
            }

            let new_x_mat = &exps * &x_mat;

            // Check for NaN - iterate over matrix elements
            let mut has_nan = false;
            for i in 0..8 {
                for j in 0..8 {
                    if !new_x_mat[(i, j)].is_finite() {
                        has_nan = true;
                        break;
                    }
                }
            }
            if has_nan {
                warn!("IEKF: NaN detected in iteration {}, aborting.", iter);
                self.state = state_propagated;
                self.cov = p_propagated;
                eval.stop("ekf_correct/state").unwrap();
                eval.stop("ekf_correct/state/x").unwrap();
                return;
            }

            // Extract rotation and convert back to nalgebra for sophus
            let mut new_rot_na = nalgebra::Matrix3::zeros();
            for i in 0..3 {
                for j in 0..3 {
                    new_rot_na[(i, j)] = new_x_mat[(i, j)];
                }
            }
            let new_quat = nalgebra::UnitQuaternion::from_matrix(&new_rot_na);
            let new_so3 = match SO3::try_from_mat(new_quat.to_rotation_matrix().matrix()) {
                Some(so3) => so3,
                None => {
                    warn!("IEKF: Invalid SO3 in iteration {}", iter);
                    break;
                }
            };

            self.state.pose.set_rotation(new_so3);
            self.state.pose.set_translation(Vector3::new(
                new_x_mat[(0, 3)],
                new_x_mat[(1, 3)],
                new_x_mat[(2, 3)],
            ));
            self.state.vel = Vector3::new(new_x_mat[(0, 4)], new_x_mat[(1, 4)], new_x_mat[(2, 4)]);
            self.state.bg = Vector3::new(new_x_mat[(0, 5)], new_x_mat[(1, 5)], new_x_mat[(2, 5)]);
            self.state.ba = Vector3::new(new_x_mat[(0, 6)], new_x_mat[(1, 6)], new_x_mat[(2, 6)]);
            self.state.grav = Vector3::new(new_x_mat[(0, 7)], new_x_mat[(1, 7)], new_x_mat[(2, 7)]);

            // KH = K * H  (18 x m * m x 18 = 18 x 18)
            matmul(
                self.k_h.rb_mut(),
                faer::Accum::Replace,
                self.k.rb(),
                self.h_x.rb(),
                1.0,
                faer::Par::Seq,
            );

            eval.stop("ekf_correct/state").unwrap();
            if kh_norm < 0.001 {
                break;
            }
        }

        self.cov = &p_propagated - &self.k_h * &p_propagated;
    }
}

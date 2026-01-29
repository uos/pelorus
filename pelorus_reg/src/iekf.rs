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

/// 24-dimensional state following S-FAST_LIO ordering:
/// - pos (0-2): position in world frame
/// - rot (3-5): rotation SO3 (world to IMU)
/// - offset_R_L_I (6-8): lidar-to-IMU rotation extrinsic
/// - offset_T_L_I (9-11): lidar-to-IMU translation extrinsic
/// - vel (12-14): velocity in world frame
/// - bg (15-17): gyro bias
/// - ba (18-20): accelerometer bias
/// - grav (21-23): gravity vector in world frame
#[derive(Clone, Debug, Copy)]
pub struct StateIKFom {
    pub pos: Vector3<f64>,          // position in world frame
    pub rot: SO3,                   // rotation: world to IMU (T_W_I rotation part)
    pub offset_r_l_i: SO3,          // R_I_L: lidar to IMU rotation (extrinsic calibration)
    pub offset_t_l_i: Vector3<f64>, // t_I_L: lidar to IMU translation (extrinsic calibration)
    pub vel: Vector3<f64>,          // velocity in world frame
    pub bg: Vector3<f64>,           // estimated gyro bias
    pub ba: Vector3<f64>,           // accelerometer bias
    pub grav: Vector3<f64>,         // gravity vector in world frame
}

impl StateIKFom {
    pub fn new(gravity: f64) -> Self {
        StateIKFom {
            pos: Vector3::zeros(),
            rot: SO3::identity(),
            offset_r_l_i: SO3::identity(),
            offset_t_l_i: Vector3::zeros(),
            vel: Vector3::zeros(),
            bg: Vector3::zeros(),
            ba: Vector3::zeros(),
            grav: Vector3::new(0.0, 0.0, -gravity),
        }
    }

    /// Helper to get pose as SE3 (for compatibility)
    pub fn pose(&self) -> SE3 {
        SE3::from_rotation_and_translation(self.rot, self.pos)
    }

    /// Set pose from SE3 (for compatibility)
    pub fn set_pose(&mut self, pose: SE3) {
        self.pos = pose.translation();
        self.rot = pose.rotation();
    }
}

impl Default for StateIKFom {
    fn default() -> Self {
        StateIKFom {
            pos: Vector3::zeros(),
            rot: SO3::identity(),
            offset_r_l_i: SO3::identity(),
            offset_t_l_i: Vector3::zeros(),
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

/// Get the state transition function f(x, u) Matrix (24x1)
/// Following S-FAST_LIO use-ikfom.hpp get_f() ordering:
/// - (0-2): vel (position derivative)
/// - (3-5): omega (rotation derivative)
/// - (6-8): zeros (offset_R_L_I derivative - not estimated online typically)
/// - (9-11): zeros (offset_T_L_I derivative - not estimated online typically)
/// - (12-14): a_inertial + grav (velocity derivative)
/// - (15-17): zeros (bg derivative)
/// - (18-20): zeros (ba derivative)
/// - (21-23): zeros (grav derivative)
fn get_f(s: &StateIKFom, inp: &InputIKFom) -> Mat<f64> {
    let omega = inp.gyro - s.bg; // angular velocity minus bias
    let rot_mat = mat3_to_faer(&s.rot.matrix());
    let acc_diff = inp.acc - s.ba;
    // Convert to faer col for matrix multiply
    let acc_diff_col = faer::col![acc_diff.x, acc_diff.y, acc_diff.z];
    let acc_inertial = &rot_mat * &acc_diff_col; // transform to world
    let grav_col = faer::col![s.grav.x, s.grav.y, s.grav.z];
    let acc_grav = &acc_inertial + &grav_col; // acceleration added gravity

    let mut res = Mat::<f64>::zeros(24, 1);
    // vel derivative -> position derivative (rows 0-2)
    res[(0, 0)] = s.vel.x;
    res[(1, 0)] = s.vel.y;
    res[(2, 0)] = s.vel.z;
    // omega -> rotation derivative (rows 3-5)
    res[(3, 0)] = omega.x;
    res[(4, 0)] = omega.y;
    res[(5, 0)] = omega.z;
    // offset_R_L_I derivative (rows 6-8) - zeros
    // offset_T_L_I derivative (rows 9-11) - zeros
    // acc_grav -> velocity derivative (rows 12-14)
    res[(12, 0)] = acc_grav[0];
    res[(13, 0)] = acc_grav[1];
    res[(14, 0)] = acc_grav[2];
    // bg, ba, grav derivatives (rows 15-23) - zeros

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

pub fn skew_symmetric_na<T>(q: &Vector3<T>) -> nalgebra::Matrix3<T>
where
    T: nalgebra::Scalar + Copy + From<f64> + std::ops::Neg<Output = T>,
{
    nalgebra::Matrix3::new(
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

/// Get the Jacobian of the state transition function f(x, u) with respect to x.
/// Following S-FAST_LIO use-ikfom.hpp df_dx() - returns 24x24 matrix
/// Note: This returns the matrix WITHOUT the identity and dt multiplication
/// (those are applied in predict())
pub fn df_dx(s: &StateIKFom, inp: &InputIKFom) -> Mat<f64> {
    let mut cov = Mat::<f64>::zeros(24, 24);

    let rot_mat = mat3_to_faer(&s.rot.matrix());
    let acc_diff = inp.acc - s.ba;
    let acc_skew = skew_symmetric(&acc_diff);

    // Helper to copy 3x3 block into larger matrix
    let copy_block = |dest: &mut Mat<f64>, row: usize, col: usize, src: &Mat<f64>| {
        for i in 0..3 {
            for j in 0..3 {
                dest[(row + i, col + j)] = src[(i, j)];
            }
        }
    };

    // d(pos)/d(vel) = I  -> (0, 12)
    let id3 = Mat::<f64>::identity(3, 3);
    copy_block(&mut cov, 0, 12, &id3);

    // d(vel)/d(rot) = -R * skew(acc - ba)  -> (12, 3)
    let vel_rot_block = &rot_mat * &acc_skew * (-1.0);
    copy_block(&mut cov, 12, 3, &vel_rot_block);

    // d(vel)/d(ba) = -R  -> (12, 18)
    let vel_ba_block = &rot_mat * (-1.0);
    copy_block(&mut cov, 12, 18, &vel_ba_block);

    // d(vel)/d(grav) = I  -> (12, 21)
    copy_block(&mut cov, 12, 21, &id3);

    // d(rot)/d(bg) = -I  -> (3, 15)
    let neg_id3 = &id3 * (-1.0);
    copy_block(&mut cov, 3, 15, &neg_id3);

    cov
}

/// Get the Jacobian of the state transition function f(x, u) with respect to w (noise).
/// Following S-FAST_LIO use-ikfom.hpp df_dw() - returns 24x12 matrix
/// Note: This returns the matrix WITHOUT the dt multiplication (applied in predict())
pub fn df_dw(s: &StateIKFom, _inp: &InputIKFom) -> Mat<f64> {
    let mut cov = Mat::<f64>::zeros(24, 12);

    let rot_mat = mat3_to_faer(&s.rot.matrix());
    let id3 = Mat::<f64>::identity(3, 3);

    // Helper to copy 3x3 block
    let copy_block = |dest: &mut Mat<f64>, row: usize, col: usize, src: &Mat<f64>| {
        for i in 0..3 {
            for j in 0..3 {
                dest[(row + i, col + j)] = src[(i, j)];
            }
        }
    };

    // d(vel)/d(n_a) = -R  -> (12, 3)
    let vel_na_block = &rot_mat * (-1.0);
    copy_block(&mut cov, 12, 3, &vel_na_block);

    // d(rot)/d(n_g) = -I  -> (3, 0)
    let neg_id3 = &id3 * (-1.0);
    copy_block(&mut cov, 3, 0, &neg_id3);

    // d(bg)/d(n_bg) = I  -> (15, 6)
    copy_block(&mut cov, 15, 6, &id3);

    // d(ba)/d(n_ba) = I  -> (18, 9)
    copy_block(&mut cov, 18, 9, &id3);

    cov
}

pub struct Iekf {
    pub state: StateIKFom,
    pub cov: Mat<f64>,           // 24x24 covariance matrix in faer
    pub f_w_q_w_t_dt2: Mat<f64>, // 24x24 temporary for cov update in predict
    pub fxtcov: Mat<f64>,        // 24x24 temporary for cov update in predict
    pub fxcov: Mat<f64>,         // 24x24 temporary for cov update in predict
    pub q_f_w_t: Mat<f64>,       // 12x24 temporary for cov update in predict
    pub id: Mat<f64>,            // 24x24 temporary for cov update in scalar constraint
    pub k_mat: Mat<f64>,         // 24x1 temporary for cov update in scalar constraint
    pub h_t_mat: Mat<f64>,       // 1x24 temporary for cov update in scalar constraint
    pub k_h_t: Mat<f64>,         // 24x24 temporary for cov update in scalar constraint
    pub k_k: Mat<f64>,           // 24x24 temporary for cov update in scalar constraint
    pub cov_i_kh: Mat<f64>,      // 24x24 temporary for cov update in scalar constraint
    pub i_cov_i_kh: Mat<f64>,    // 24x24 temporary for cov update in scalar constraint
    pub initialized: bool,
    pub last_time: f64,
    pub h: Col<f64>,   // measurement residual vector
    pub h_x: Mat<f64>, // measurement Jacobian matrix (m x 12, sparse - only first 12 cols non-zero)
    pub k: Mat<f64>,   // Kalman gain (24 x m)
    pub k_h: Mat<f64>, // K * H (24 x 24)
    pub hth: Mat<f64>, // H^T * H (24 x 24, but only 12x12 block is non-zero)
}

/// Adding the state and the perturbation, stay on the manifold
/// pos(0-2), rot(3-5), offset_R_L_I(6-8), offset_T_L_I(9-11), vel(12-14), bg(15-17), ba(18-20), grav(21-23)
fn box_plus(x: &StateIKFom, f: &Mat<f64>) -> StateIKFom {
    let mut x_r = *x;

    // pos += delta_pos (rows 0-2)
    let delta_pos = Vector3::new(f[(0, 0)], f[(1, 0)], f[(2, 0)]);
    x_r.pos = x.pos + delta_pos;

    // rot = rot * exp(delta_rot) (rows 3-5)
    let delta_rot_vec = Vector3::new(f[(3, 0)], f[(4, 0)], f[(5, 0)]);
    x_r.rot = x.rot.group_mul(SO3::exp(delta_rot_vec));

    // offset_R_L_I = offset_R_L_I * exp(delta_offset_R) (rows 6-8)
    let delta_offset_r = Vector3::new(f[(6, 0)], f[(7, 0)], f[(8, 0)]);
    x_r.offset_r_l_i = x.offset_r_l_i.group_mul(SO3::exp(delta_offset_r));

    // offset_T_L_I += delta_offset_T (rows 9-11)
    let delta_offset_t = Vector3::new(f[(9, 0)], f[(10, 0)], f[(11, 0)]);
    x_r.offset_t_l_i = x.offset_t_l_i + delta_offset_t;

    // vel += delta_vel (rows 12-14)
    x_r.vel = x.vel + Vector3::new(f[(12, 0)], f[(13, 0)], f[(14, 0)]);

    // bg += delta_bg (rows 15-17)
    x_r.bg = x.bg + Vector3::new(f[(15, 0)], f[(16, 0)], f[(17, 0)]);

    // ba += delta_ba (rows 18-20)
    x_r.ba = x.ba + Vector3::new(f[(18, 0)], f[(19, 0)], f[(20, 0)]);

    // grav += delta_grav (rows 21-23)
    x_r.grav = x.grav + Vector3::new(f[(21, 0)], f[(22, 0)], f[(23, 0)]);

    x_r
}

/// Difference between two states, stay on the manifold
fn box_minus(x1: &StateIKFom, x2: &StateIKFom) -> Mat<f64> {
    let mut x_r = Mat::<f64>::zeros(24, 1);

    // pos difference (rows 0-2)
    let pos_diff = x1.pos - x2.pos;
    x_r[(0, 0)] = pos_diff.x;
    x_r[(1, 0)] = pos_diff.y;
    x_r[(2, 0)] = pos_diff.z;

    // rot difference: log(x2.rot^-1 * x1.rot) (rows 3-5)
    let rot_diff = x2.rot.inverse().group_mul(x1.rot).log();
    x_r[(3, 0)] = rot_diff.x;
    x_r[(4, 0)] = rot_diff.y;
    x_r[(5, 0)] = rot_diff.z;

    // offset_R_L_I difference (rows 6-8)
    let offset_r_diff = x2.offset_r_l_i.inverse().group_mul(x1.offset_r_l_i).log();
    x_r[(6, 0)] = offset_r_diff.x;
    x_r[(7, 0)] = offset_r_diff.y;
    x_r[(8, 0)] = offset_r_diff.z;

    // offset_T_L_I difference (rows 9-11)
    let offset_t_diff = x1.offset_t_l_i - x2.offset_t_l_i;
    x_r[(9, 0)] = offset_t_diff.x;
    x_r[(10, 0)] = offset_t_diff.y;
    x_r[(11, 0)] = offset_t_diff.z;

    // vel difference (rows 12-14)
    let vel_diff = x1.vel - x2.vel;
    x_r[(12, 0)] = vel_diff.x;
    x_r[(13, 0)] = vel_diff.y;
    x_r[(14, 0)] = vel_diff.z;

    // bg difference (rows 15-17)
    let bg_diff = x1.bg - x2.bg;
    x_r[(15, 0)] = bg_diff.x;
    x_r[(16, 0)] = bg_diff.y;
    x_r[(17, 0)] = bg_diff.z;

    // ba difference (rows 18-20)
    let ba_diff = x1.ba - x2.ba;
    x_r[(18, 0)] = ba_diff.x;
    x_r[(19, 0)] = ba_diff.y;
    x_r[(20, 0)] = ba_diff.z;

    // grav difference (rows 21-23)
    let grav_diff = x1.grav - x2.grav;
    x_r[(21, 0)] = grav_diff.x;
    x_r[(22, 0)] = grav_diff.y;
    x_r[(23, 0)] = grav_diff.z;

    x_r
}

impl Iekf {
    pub fn new(g: f64) -> Self {
        Self {
            state: StateIKFom::new(g),
            cov: Mat::<f64>::identity(24, 24),
            initialized: false,
            last_time: 0.0,
            h: Col::<f64>::zeros(0),
            h_x: Mat::<f64>::zeros(0, 12), // Sparse: only first 12 cols non-zero
            k: Mat::<f64>::zeros(24, 10000),
            k_h: Mat::<f64>::zeros(24, 24),
            f_w_q_w_t_dt2: Mat::<f64>::zeros(24, 24),
            fxtcov: Mat::<f64>::zeros(24, 24),
            fxcov: Mat::<f64>::zeros(24, 24),
            q_f_w_t: Mat::<f64>::zeros(12, 24),
            id: Mat::<f64>::identity(24, 24),
            k_mat: Mat::<f64>::zeros(24, 1),
            h_t_mat: Mat::<f64>::zeros(1, 24),
            k_h_t: Mat::<f64>::zeros(24, 24),
            k_k: Mat::<f64>::zeros(24, 24),
            cov_i_kh: Mat::<f64>::zeros(24, 24),
            i_cov_i_kh: Mat::<f64>::zeros(24, 24),
            hth: Mat::<f64>::zeros(24, 24),
        }
    }

    pub fn predict(&mut self, dt: f64, q: &Mat<f64>, i_in: &InputIKFom) {
        self.last_time += dt;

        let f = get_f(&self.state, i_in);
        let f_x_raw = df_dx(&self.state, i_in);
        let f_w_raw = df_dw(&self.state, i_in);

        // State update: x = boxplus(x, f * dt)
        let f_dt = &f * dt;
        self.state = box_plus(&self.state, &f_dt);

        let id24 = Mat::<f64>::identity(24, 24);
        let f_x = &id24 + &f_x_raw * dt;
        let f_x_t = f_x.transpose();

        // F_w = f_w * dt
        let f_w = &f_w_raw * dt;
        let f_w_t = f_w.transpose();

        // Cov update: P = F_x * P * F_x^T + F_w * Q * F_w^T
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
            1.0,
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
        let r_wb = self.state.rot.matrix();
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
        // h_t is 1x24, cov_h is Col (24x1), result is a scalar
        // Compute h^T * cov_h manually
        let mut s = 0.0;
        for i in 0..24 {
            s += h_t[i] * cov_h[i];
        }
        s += r_var;

        if !s.is_finite() || s <= 0.0 {
            warn!("Invalid innovation variance in kinematic constraint");
            return;
        }

        // Kalman gain: K = P * h / s (result is 24x1)
        let mut k = &cov_h / s;

        // zero out parts that must never be changed by this measurement
        // (pos and rotation live in the first 6 entries)
        for i in 0..3 {
            k[i] = 0.0; // pos
        }
        for i in 3..6 {
            k[i] = 0.0; // rotation
        }

        // automatic damping - cap the norm of the gain (prevents gigantic jumps)
        const K_MAX: f64 = 0.8;
        let gain_norm = k.norm_l2();
        if gain_norm > K_MAX {
            let scale = K_MAX / gain_norm;
            k = &k * scale;
        }

        // Update state: dx = k * residual (k is 24x1, result is 24x1)
        let dx = &k * residual;

        // Convert Col to Mat for box_plus
        let mut dx_mat = Mat::<f64>::zeros(24, 1);
        for i in 0..24 {
            dx_mat[(i, 0)] = dx[i];
        }
        self.state = box_plus(&self.state, &dx_mat);

        // Update covariance: P = (I - K*h^T) * P * (I - K*h^T)^T + K * r_var * K^T
        for i in 0..24 {
            self.k_mat[(i, 0)] = k[i];
        }
        for j in 0..24 {
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

        // Velocity is at indices 12-14 in the 24-dim state
        let vel_idx = 12 + component;
        let p_comp = self.cov[(vel_idx, vel_idx)];
        let innovation_var = p_comp + noise_cov;

        if !innovation_var.is_finite() || innovation_var <= 0.0 {
            warn!(
                "Invalid innovation variance in velocity[{}] correction",
                component
            );
            return;
        }

        let gain_scalar = p_comp / innovation_var;

        let mut kalman_gain = Col::<f64>::zeros(24);
        for i in 0..24 {
            kalman_gain[i] = self.cov[(i, vel_idx)] * gain_scalar;
        }
        let delta_v = gain_scalar * residual;

        if !delta_v.is_finite() {
            warn!("NaN in velocity[{}] update", component);
            return;
        }

        self.state.vel[component] += delta_v;

        // Update covariance
        for i in 0..24 {
            for j in 0..24 {
                let old_val = self.cov[(i, j)];
                let update = kalman_gain[i] * self.cov[(vel_idx, j)];
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

        let r_w_i = self.state.rot.matrix();
        let r_i_w = r_w_i.transpose();
        let v_body = r_i_w * self.state.vel;
        let r_var = constraint_noise_std * constraint_noise_std;

        match model_type {
            KinematicModel::Differential => {
                // lateral (y) constraint - velocity index is 12-14 in 24-dim state
                let mut h_lat = Col::<f64>::zeros(24);
                for j in 0..3 {
                    h_lat[12 + j] = r_i_w[(1, j)];
                }
                self.apply_scalar_constraint(h_lat, -v_body.y, r_var);

                // vertical (z) constraint
                let mut h_vert = Col::<f64>::zeros(24);
                for j in 0..3 {
                    h_vert[12 + j] = r_i_w[(2, j)];
                }
                self.apply_scalar_constraint(h_vert, -v_body.z, r_var);
            }
            KinematicModel::Holonomic2D => {
                // Only vertical (z) constraint: robot stays on ground
                let mut h_vert = Col::<f64>::zeros(24);
                for j in 0..3 {
                    h_vert[12 + j] = r_i_w[(2, j)];
                }
                self.apply_scalar_constraint(h_vert, -v_body.z, r_var);
            }

            KinematicModel::Ackermann { imu_to_rear_axle } => {
                // rear‑axle lateral constraint
                let v_rear = v_body + gyro_measurement.cross(&imu_to_rear_axle);
                let mut h_lat = Col::<f64>::zeros(24);
                for j in 0..3 {
                    h_lat[12 + j] = r_i_w[(1, j)];
                }
                self.apply_scalar_constraint(h_lat, -v_rear.y, r_var);

                // vertical constraint
                let mut h_vert = Col::<f64>::zeros(24);
                for j in 0..3 {
                    h_vert[12 + j] = r_i_w[(2, j)];
                }
                self.apply_scalar_constraint(h_vert, -v_body.z, r_var);
            }

            KinematicModel::Full3D => {}
        }
    }

    /// State ordering (24-dim): pos(0-2), rot(3-5), offset_R_L_I(6-8), offset_T_L_I(9-11),
    ///                          vel(12-14), bg(15-17), ba(18-20), grav(21-23)
    pub fn correct(
        &mut self,
        measurement_noise_cov: f64,
        measurement_time: f64,
        max_iter: usize,
        eval: std::sync::Arc<std::sync::RwLock<&mut RuntimeEvaluator>>,
        mut model: impl FnMut(
            &StateIKFom,
            &mut bool,     // valid
            bool,          // converge (for re-searching correspondences if needed)
            &mut Col<f64>, // residual h
            &mut Mat<f64>, // jacobian H (m x 12, sparse)
            &mut usize,    // num_measurements - actual number of valid measurements
        ),
    ) {
        self.last_time = measurement_time;

        if measurement_noise_cov <= 0.0 {
            error!("IEKF: Non‑positive measurement noise covariance, skipping update.");
            return;
        }

        // Save propagated state and covariance
        let state_propagated = self.state;
        let p_propagated = self.cov.clone();

        let r_inv = 1.0 / measurement_noise_cov;

        let mut converge_count = 0;
        let mut converge_flag = true;

        // Iterate from -1 to max_iter (following S-FAST_LIO convention)
        // -1 iteration is just to compute the first Jacobian
        for iter in 0..=max_iter {
            let mut valid = true;
            let mut num_measurements = 0usize;

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
                &mut num_measurements,
            );

            let mut eval = eval.write().unwrap();
            eval.stop("ekf_correct/model").unwrap();

            if !valid || num_measurements == 0 {
                continue;
            }

            // Use actual number of measurements, not buffer size
            let m = num_measurements;

            eval.start("ekf_correct/info_mat").unwrap();

            // dx_new = boxminus(x_, x_propagated) - equation (18): x^k - x^
            let dx_new = box_minus(&self.state, &state_propagated);

            // H is m x 12 (sparse - only first 12 columns non-zero)
            // We only use the first m rows of h_x (the actual measurements)
            // HTH is 24 x 24 but only 12x12 top-left block is non-zero
            let h_full = &self.h_x;
            // Create a view of only the first m rows
            let h = h_full.subrows(0, m);
            let h_t = h.transpose(); // 12 x m

            // Compute H^T * H (12 x 12)
            let mut hth_12x12 = Mat::<f64>::zeros(12, 12);
            matmul(
                hth_12x12.rb_mut(),
                faer::Accum::Replace,
                h_t.rb(),
                h.rb(),
                1.0,
                faer::Par::Seq,
            );

            // Build 24x24 HTH with sparse structure
            for i in 0..24 {
                for j in 0..24 {
                    self.hth[(i, j)] = 0.0;
                }
            }
            for i in 0..12 {
                for j in 0..12 {
                    self.hth[(i, j)] = hth_12x12[(i, j)];
                }
            }

            eval.stop("ekf_correct/info_mat").unwrap();
            eval.start("ekf_correct/lhs").unwrap();

            // K_front = (HTH / R + P^-1)^-1
            let p_inv = p_propagated.partial_piv_lu().inverse();
            let lhs = &self.hth * r_inv + &p_inv;
            let k_front = lhs.partial_piv_lu().inverse();

            // K = K_front[:, 0:12] * H^T / R  (24 x m)
            // Pre-allocate k if needed
            if self.k.ncols() < m {
                self.k.reserve(24, m);
            }
            if self.k.ncols() != m {
                self.k.resize_with(24, m, |_, _| 0.0);
            }

            // K_front_12 = K_front[:, 0:12] (24 x 12)
            let mut k_front_12 = Mat::<f64>::zeros(24, 12);
            for i in 0..24 {
                for j in 0..12 {
                    k_front_12[(i, j)] = k_front[(i, j)];
                }
            }

            // K = K_front_12 * H^T / R  (24 x 12 * 12 x m = 24 x m)
            matmul(
                self.k.rb_mut(),
                faer::Accum::Replace,
                k_front_12.rb(),
                h_t.rb(),
                r_inv,
                faer::Par::Seq,
            );

            eval.stop("ekf_correct/lhs").unwrap();
            eval.start("ekf_correct/gain").unwrap();

            // KH = K * H (24 x m * m x 12 = 24 x 12, but we store as 24 x 24 with zeros)
            let mut kh_12 = Mat::<f64>::zeros(24, 12);
            matmul(
                kh_12.rb_mut(),
                faer::Accum::Replace,
                self.k.rb(),
                h.rb(),
                1.0,
                faer::Par::Seq,
            );

            // Build full 24x24 KH matrix (sparse)
            for i in 0..24 {
                for j in 0..24 {
                    self.k_h[(i, j)] = 0.0;
                }
            }
            for i in 0..24 {
                for j in 0..12 {
                    self.k_h[(i, j)] = kh_12[(i, j)];
                }
            }

            // dx_ = K * h + (KH - I) * dx_new  (equation 18)
            // K * h (24 x 1)
            let mut kh_vec = Mat::<f64>::zeros(24, 1);
            let h_col = Mat::from_fn(m, 1, |i, _| self.h[i]);
            matmul(
                kh_vec.rb_mut(),
                faer::Accum::Replace,
                self.k.rb(),
                h_col.rb(),
                1.0,
                faer::Par::Seq,
            );

            // (KH - I) * dx_new
            let id24 = Mat::<f64>::identity(24, 24);
            let kh_minus_i = &self.k_h - &id24;
            let kh_i_dx = &kh_minus_i * &dx_new;

            // dx_ = K * h + (KH - I) * dx_new
            let dx_ = &kh_vec + &kh_i_dx;

            eval.stop("ekf_correct/gain").unwrap();
            eval.start("ekf_correct/state").unwrap();

            // Check for NaN
            let mut has_nan = false;
            for i in 0..24 {
                if !dx_[(i, 0)].is_finite() {
                    has_nan = true;
                    break;
                }
            }
            if has_nan {
                warn!("IEKF: NaN detected in iteration {}, aborting.", iter);
                self.state = state_propagated;
                self.cov = p_propagated;
                eval.stop("ekf_correct/state").unwrap();
                return;
            }

            // x_ = boxplus(x_, dx_)
            self.state = box_plus(&self.state, &dx_);

            // Check convergence: if all |dx_[j]| < EPSI, converged
            converge_flag = true;
            for j in 0..24 {
                if dx_[(j, 0)].abs() > measurement_noise_cov {
                    converge_flag = false;
                    break;
                }
            }

            if converge_flag {
                converge_count += 1;
            }

            // If not converged yet and this is second-to-last iteration, force converge
            // to re-search correspondences
            if converge_count == 0 && iter == max_iter - 1 {
                converge_flag = true;
            }

            eval.stop("ekf_correct/state").unwrap();

            // Exit if converged twice or at max iteration
            if converge_count > 1 || iter == max_iter {
                break;
            }
        }

        // Covariance update: P = (I - KH) * P  (equation 19)
        let id24 = Mat::<f64>::identity(24, 24);
        let i_kh = &id24 - &self.k_h;
        self.cov = &i_kh * &p_propagated;
    }
}

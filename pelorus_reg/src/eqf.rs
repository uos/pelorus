//! Iterated equivariant filter (EqF) for tightly coupled Lidar-inertial odometry.
//!
//! Follows "Equivariant Filter for Tightly Coupled LiDAR-Inertial Odometry"
//! (Xu et al., ICRA 2025, <https://arxiv.org/abs/2409.06948>) and its reference
//! implementation <https://github.com/taoxiom/Eq-LIO>.
//!
//! The state and measurement model follow a FAST-LIO style iterated EKF. The
//! error is a group element of the semi-direct product symmetry coupling the
//! navigation state, IMU biases, and Lidar extrinsics. The correction acts on
//! the left:
//!
//! ```text
//! R <- exp(dtheta) R,   p <- dp + exp(dtheta) p,   v <- dv + exp(dtheta) v
//! ```
//!
//! with the biases carried along by `R^T` (see [`compensate`]). Because the group
//! action, not the raw IMU reading, drives the linearisation, the error dynamics
//! depend on the state only through equivariant quantities, which is what reduces
//! the linearisation error and keeps the filter consistent through abrupt motion.

use faer::linalg::matmul::matmul;
use faer::linalg::solvers::DenseSolveCore;
use faer::prelude::{Reborrow, ReborrowMut};
use faer::{Col, Mat};
use log::{error, warn};
use nalgebra::Vector3; // Keep for sophus_lie interface
use pelorus_core::{
    KinematicModel, OdomMsg, log_debouncer::should_log_once_default, timing::RuntimeEvaluator,
};
use pelorus_sync::timestamp_to_secs;

pub use pelorus_core::{SE3, SO3};

// Helper to convert nalgebra Matrix3 from sophus to faer
#[inline]
fn mat3_to_faer(m: &nalgebra::Matrix3<f64>) -> Mat<f64> {
    Mat::from_fn(3, 3, |i, j| m[(i, j)])
}

// TODO use config, needs some refactor
const GRAVITY_DEFAULT: f64 = 9.81;

/// Gain cap for the non-holonomic kinematic constraints.
const KINEMATIC_MAX_GAIN: f64 = 0.8;

/// Filter state. The index ranges are the layout of the 24-dimensional error
/// vector used by the covariance, the Jacobians, [`compensate`] and [`box_minus`].
///
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

/// Given a vector `q = [x, y, z]`, the skew-symmetric matrix is:
/// ```text
/// [  0, -z,  y ]
/// [  z,  0, -x ]
/// [ -y,  x,  0 ]
/// ```
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

/// Jacobian of the error dynamics with respect to the error state (Eq-LIO eq. (7)).
///
/// The equivariant error uses the global frame of the semi-direct product. Its
/// linearisation is driven by `X = hat(R (omega - bg))`. The blocks below depend
/// on quantities that transform covariantly with the group action.
///
/// Returns the matrix WITHOUT the identity and the `dt` factor. Both are applied
/// in `predict()`.
pub fn df_dx(
    s: &StateIKFom,
    inp: &InputIKFom,
    extrinsic_est: bool,
    anchor: &Vector3<f64>,
) -> Mat<f64> {
    let mut cov = Mat::<f64>::zeros(24, 24);
    let pos_a = s.pos - anchor;

    let id3 = nalgebra::Matrix3::<f64>::identity();
    let rot_mat = s.rot.matrix();

    // Generator of the equivariant error dynamics: the bias-corrected angular
    // rate expressed in the world frame.
    let omega_world = rot_mat * (inp.gyro - s.bg);
    let x_blk = skew_symmetric_na(&omega_world);

    let mut copy_block = |row: usize, col: usize, src: &nalgebra::Matrix3<f64>| {
        for i in 0..3 {
            for j in 0..3 {
                cov[(row + i, col + j)] = src[(i, j)];
            }
        }
    };

    copy_block(0, 12, &id3); // pos -- vel
    copy_block(0, 15, &(-skew_symmetric_na(&pos_a))); // pos -- gyro bias
    copy_block(3, 15, &(-id3)); // rot -- gyro bias

    if extrinsic_est {
        // Couples the lever arm to the extrinsic rotation error.
        let z_blk = skew_symmetric_na(&(skew_symmetric_na(&pos_a) * omega_world + s.vel));
        copy_block(6, 6, &x_blk); // extrinsic rot -- extrinsic rot
        copy_block(9, 6, &z_blk); // lever arm -- extrinsic rot
        copy_block(9, 9, &x_blk); // lever arm -- lever arm
    }

    copy_block(12, 3, &skew_symmetric_na(&s.grav)); // vel -- rot
    copy_block(12, 18, &(-id3)); // vel -- accel bias
    copy_block(12, 21, &id3); // vel -- gravity

    copy_block(15, 15, &x_blk); // gyro bias -- gyro bias

    // Couples the accelerometer bias to the gyro bias error.
    let y_blk = skew_symmetric_na(
        &(rot_mat * (inp.acc - s.ba) + skew_symmetric_na(&s.vel) * omega_world + s.grav),
    );
    copy_block(18, 15, &y_blk); // accel bias -- gyro bias
    copy_block(18, 18, &x_blk); // accel bias -- accel bias

    cov
}

/// Jacobian of the error dynamics with respect to the IMU noise.
///
/// Every block carries the world rotation because the noise enters in the body
/// frame while the equivariant error state is expressed globally.
pub fn df_dw(s: &StateIKFom, _inp: &InputIKFom, anchor: &Vector3<f64>) -> Mat<f64> {
    let mut cov = Mat::<f64>::zeros(24, 12);

    let rot_mat = s.rot.matrix();
    let pos_skew_rot = skew_symmetric_na(&(s.pos - anchor)) * rot_mat;
    let vel_skew_rot = skew_symmetric_na(&s.vel) * rot_mat;

    let mut copy_block = |row: usize, col: usize, src: &nalgebra::Matrix3<f64>| {
        for i in 0..3 {
            for j in 0..3 {
                cov[(row + i, col + j)] = src[(i, j)];
            }
        }
    };

    copy_block(0, 0, &pos_skew_rot); // pos -- gyro noise
    copy_block(3, 0, &rot_mat); // rot -- gyro noise
    copy_block(12, 0, &vel_skew_rot); // vel -- gyro noise
    copy_block(12, 3, &rot_mat); // vel -- accel noise
    copy_block(15, 6, &(-rot_mat)); // gyro bias -- gyro bias noise
    copy_block(18, 6, &(-vel_skew_rot)); // accel bias -- gyro bias noise
    copy_block(18, 9, &(-rot_mat)); // accel bias -- accel bias noise

    cov
}

pub struct Eqf {
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
    /// Origin the equivariant error is measured about. The global rotation error
    /// displaces a point by its distance from this anchor, so leaving it at the
    /// world origin makes both the Jacobians and the covariance scale with distance
    /// travelled and destroys the conditioning of the update. Kept near the robot.
    pub anchor: Vector3<f64>,
    /// Estimate the Lidar-to-IMU extrinsics online. Off by default. When on,
    /// the extrinsic blocks of the equivariant error dynamics and the
    /// corresponding measurement Jacobian columns become active.
    pub extrinsic_est: bool,
    pub last_time: f64,
    pub h: Col<f64>,   // measurement residual vector
    pub h_x: Mat<f64>, // measurement Jacobian matrix (m x 12, sparse - only first 12 cols non-zero)
    pub k: Mat<f64>,   // Kalman gain (24 x m)
    pub k_h: Mat<f64>, // K * H (24 x 24)
    pub hth: Mat<f64>, // H^T * H (24 x 24, but only 12x12 block is non-zero)
    pub base_to_lidar: nalgebra::Isometry3<f64>,
}

/// Forward propagation of the state along the IMU kinematics.
///
/// Ordering: pos(0-2), rot(3-5), offset_R_L_I(6-8), offset_T_L_I(9-11),
/// vel(12-14), bg(15-17), ba(18-20), grav(21-23).
///
/// Propagation is unaffected by the choice of error frame, so this stays the
/// plain right-multiplied manifold update. The equivariance shows up in
/// `compensate` and `box_minus`.
fn propagate(x: &StateIKFom, f: &Mat<f64>) -> StateIKFom {
    let mut x_r = *x;

    x_r.pos = x.pos + Vector3::new(f[(0, 0)], f[(1, 0)], f[(2, 0)]);
    x_r.rot = x
        .rot
        .group_mul(SO3::exp(Vector3::new(f[(3, 0)], f[(4, 0)], f[(5, 0)])));
    x_r.offset_r_l_i =
        x.offset_r_l_i
            .group_mul(SO3::exp(Vector3::new(f[(6, 0)], f[(7, 0)], f[(8, 0)])));
    x_r.offset_t_l_i = x.offset_t_l_i + Vector3::new(f[(9, 0)], f[(10, 0)], f[(11, 0)]);
    x_r.vel = x.vel + Vector3::new(f[(12, 0)], f[(13, 0)], f[(14, 0)]);
    x_r.bg = x.bg + Vector3::new(f[(15, 0)], f[(16, 0)], f[(17, 0)]);
    x_r.ba = x.ba + Vector3::new(f[(18, 0)], f[(19, 0)], f[(20, 0)]);
    x_r.grav = x.grav;

    x_r
}

/// Inject an equivariant error `f` into the state.
///
/// The error lives in the global frame, so the rotation error acts on the left
/// and drags position, velocity and both biases along with it. This is the
/// inverse of [`box_minus`].
fn compensate(
    x: &StateIKFom,
    f: &Mat<f64>,
    extrinsic_est: bool,
    anchor: &Vector3<f64>,
) -> StateIKFom {
    let delta = |i: usize| Vector3::new(f[(i, 0)], f[(i + 1, 0)], f[(i + 2, 0)]);

    let d_bg = delta(15);
    let rot_error = SO3::exp(delta(3));
    let rot_error_mat = rot_error.matrix();
    let rot_inv_mat = x.rot.matrix().transpose();

    let mut x_r = *x;
    x_r.rot = rot_error.group_mul(x.rot);
    x_r.pos = anchor + delta(0) + rot_error_mat * (x.pos - anchor);
    x_r.vel = delta(12) + rot_error_mat * x.vel;

    if extrinsic_est {
        let offset_r_error = SO3::exp(delta(6));
        x_r.offset_r_l_i = x
            .rot
            .inverse()
            .group_mul(offset_r_error)
            .group_mul(x.rot)
            .group_mul(x.offset_r_l_i);
        x_r.offset_t_l_i =
            x_r.offset_r_l_i.matrix() * x.offset_r_l_i.matrix().transpose() * x.offset_t_l_i
                + rot_inv_mat * delta(9)
                - rot_inv_mat
                    * (nalgebra::Matrix3::identity() - offset_r_error.matrix())
                    * (x.pos - anchor);
    }

    x_r.bg = x.bg + rot_inv_mat * d_bg;
    x_r.ba = x.ba + rot_inv_mat * (delta(18) - skew_symmetric_na(&x.vel) * d_bg);
    x_r.grav = x.grav + delta(21);

    x_r
}

/// Jacobian of the error re-referencing, i.e. the "reset step".
///
/// After the update moves the estimate by `dx`, the posterior covariance is still
/// expressed about the *old* reference. Pushing it onto the new one is not optional:
/// omitting it leaves the covariance systematically wrong by a little on every
/// frame, which is invisible locally and accumulates into drift over a long run.
///
/// It matters more here than in a body-frame EKF. There, position and velocity are
/// corrected additively and their reset blocks are exactly identity, leaving only
/// the rotation block. Here the group action folds the rotation correction into
/// position and velocity too, which measures ~3x further from identity.
///
/// Computed by central differences: a closed form exists but this is exact to
/// ~1e-10, costs 48 cheap evaluations once per correction, and is far harder to get
/// subtly wrong.
fn reset_jacobian(
    base: &StateIKFom,
    dx: &Mat<f64>,
    updated: &StateIKFom,
    extrinsic_est: bool,
    anchor: &Vector3<f64>,
) -> Mat<f64> {
    const EPS: f64 = 1e-6;
    let mut j = Mat::<f64>::identity(24, 24);
    let mut probe = dx.clone();

    for c in 0..24 {
        probe[(c, 0)] = dx[(c, 0)] + EPS;
        let hi = box_minus(
            &compensate(base, &probe, extrinsic_est, anchor),
            updated,
            extrinsic_est,
            anchor,
        );
        probe[(c, 0)] = dx[(c, 0)] - EPS;
        let lo = box_minus(
            &compensate(base, &probe, extrinsic_est, anchor),
            updated,
            extrinsic_est,
            anchor,
        );
        probe[(c, 0)] = dx[(c, 0)];

        for r in 0..24 {
            j[(r, c)] = (hi[(r, 0)] - lo[(r, 0)]) / (2.0 * EPS);
        }
    }

    if !extrinsic_est {
        // Preserve fixed extrinsics across the reset to keep covariance nonsingular.
        for r in 6..12 {
            for c in 0..24 {
                j[(r, c)] = if r == c { 1.0 } else { 0.0 };
            }
            for rr in 0..24 {
                j[(rr, r)] = if rr == r { 1.0 } else { 0.0 };
            }
        }
    }

    j
}

/// Equivariant difference between two states, the inverse of [`compensate`].
fn box_minus(
    x1: &StateIKFom,
    x2: &StateIKFom,
    extrinsic_est: bool,
    anchor: &Vector3<f64>,
) -> Mat<f64> {
    let rot_err = x1.rot.group_mul(x2.rot.inverse());
    let rot_err_mat = rot_err.matrix();
    let rot2_mat = x2.rot.matrix();
    let d_bg = rot2_mat * (x1.bg - x2.bg);

    let mut x_r = Mat::<f64>::zeros(24, 1);
    let mut set = |i: usize, v: Vector3<f64>| {
        x_r[(i, 0)] = v.x;
        x_r[(i + 1, 0)] = v.y;
        x_r[(i + 2, 0)] = v.z;
    };

    set(0, (x1.pos - anchor) - rot_err_mat * (x2.pos - anchor));
    set(3, rot_err.log());

    if extrinsic_est {
        // The extrinsic error is expressed in the same global frame as the rest
        // of the correction, mirroring `compensate`. Eq-LIO derives this branch
        // but then unconditionally overwrites it with the body-frame form below,
        // which leaves its `compensate` and `boxminus` inconsistent whenever
        // online extrinsic estimation is enabled.
        let ext_err = x2
            .rot
            .group_mul(x1.offset_r_l_i)
            .group_mul(x2.offset_r_l_i.inverse())
            .group_mul(x2.rot.inverse());
        let ext_err_mat = ext_err.matrix();
        set(6, ext_err.log());
        set(
            9,
            (nalgebra::Matrix3::identity() - ext_err_mat) * (x2.pos - anchor)
                + rot2_mat * x1.offset_t_l_i
                - ext_err_mat * rot2_mat * x2.offset_t_l_i,
        );
    } else {
        set(
            6,
            x2.offset_r_l_i.inverse().group_mul(x1.offset_r_l_i).log(),
        );
        set(9, x1.offset_t_l_i - x2.offset_t_l_i);
    }

    set(12, x1.vel - rot_err_mat * x2.vel);
    set(15, d_bg);
    set(
        18,
        rot2_mat * (x1.ba - x2.ba) + skew_symmetric_na(&x2.vel) * d_bg,
    );
    set(21, x1.grav - x2.grav);

    x_r
}

impl Eqf {
    pub fn new(g: f64, base_to_lidar: nalgebra::Isometry3<f64>) -> Self {
        Self {
            state: StateIKFom::new(g),
            cov: Mat::<f64>::identity(24, 24),
            initialized: false,
            anchor: Vector3::zeros(),
            extrinsic_est: false,
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
            base_to_lidar,
        }
    }

    /// Move the anchor. Re-anchoring is a group action, so the covariance is
    /// transformed by an exact congruence: `dp` measured about `a'` equals `dp`
    /// about `a` minus `S(a' - a) dtheta`. No approximation is involved.
    pub fn set_anchor(&mut self, new_anchor: Vector3<f64>) {
        let shift = new_anchor - self.anchor;
        if shift.norm_squared() == 0.0 {
            return;
        }
        let s = skew_symmetric_na(&shift);
        let mut ad = Mat::<f64>::identity(24, 24);
        for i in 0..3 {
            for j in 0..3 {
                ad[(i, 3 + j)] = -s[(i, j)];
            }
        }
        self.cov = &(&ad * &self.cov) * ad.transpose();
        self.anchor = new_anchor;
    }

    pub fn predict(&mut self, dt: f64, q: &Mat<f64>, i_in: &InputIKFom) {
        self.last_time += dt;

        let f = get_f(&self.state, i_in);
        let f_x_raw = df_dx(&self.state, i_in, self.extrinsic_est, &self.anchor);
        let f_w_raw = df_dw(&self.state, i_in, &self.anchor);

        // State update: x = propagate(x, f * dt)
        let f_dt = &f * dt;
        self.state = propagate(&self.state, &f_dt);

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

    /// Move a posterior covariance from error coordinates about `base` into
    /// error coordinates about the current state after injecting `dx`.
    fn reset_covariance_after_update(&mut self, base: &StateIKFom, dx: &Mat<f64>) {
        let j_reset = reset_jacobian(base, dx, &self.state, self.extrinsic_est, &self.anchor);
        self.cov = &(&j_reset * &self.cov) * j_reset.transpose();
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
        let r_wi = self.state.rot.matrix();

        // Current Lidar-to-IMU rotation (R_IL from state)
        let r_il = self.state.offset_r_l_i.matrix();

        // Static Base-to-Lidar (R_BL from config)
        let r_bl_rot = self.base_to_lidar.rotation.to_rotation_matrix();
        let r_bl = r_bl_rot.matrix();

        // Compute Base-to-IMU rotation: R_BI = R_BL * R_IL^T
        let r_bi = r_bl * r_il.transpose();

        // Compute Base-to-World rotation: R_WB = R_WI * R_BI^T
        let r_wb = r_wi * r_bi.transpose();

        // Measurement is in Base frame, transform to World frame
        let v_meas_world = r_wb * odom.twist_linear;

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

    /// Fuse one scalar measurement whose Jacobian is `h`.
    ///
    /// Lidar correction owns position and rotation. `max_gain` can damp the
    /// model-based non-holonomic constraints.
    fn apply_scalar_constraint(
        &mut self,
        h: Col<f64>,
        residual: f64,
        r_var: f64,
        max_gain: Option<f64>,
    ) {
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
        if let Some(k_max) = max_gain {
            let gain_norm = k.norm_l2();
            if gain_norm > k_max {
                k = &k * (k_max / gain_norm);
            }
        }

        // Update state: dx = k * residual (k is 24x1, result is 24x1)
        let dx = &k * residual;

        // Convert Col to Mat for compensate
        let mut dx_mat = Mat::<f64>::zeros(24, 1);
        for i in 0..24 {
            dx_mat[(i, 0)] = dx[i];
        }
        let state_before = self.state;
        self.state = compensate(&state_before, &dx_mat, self.extrinsic_est, &self.anchor);

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
        self.reset_covariance_after_update(&state_before, &dx_mat);
    }

    /// Fuse one component of a world-frame velocity measurement.
    ///
    /// Under the equivariant error the world velocity perturbs as
    /// `v <- dv + exp(dtheta) v`, so the measurement couples to the rotation
    /// error through `-hat(v)` as well as to the velocity error itself. Routing
    /// this through [`Self::apply_scalar_constraint`] keeps the injection in the
    /// same frame as the rest of the filter.
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
        if !residual.is_finite() {
            warn!("Non-finite residual in velocity[{component}] correction");
            return;
        }

        let vel_skew = skew_symmetric_na(&self.state.vel);
        let mut h = Col::<f64>::zeros(24);
        for j in 0..3 {
            h[3 + j] = -vel_skew[(component, j)];
        }
        h[12 + component] = 1.0;

        self.apply_scalar_constraint(h, residual, noise_cov, None);
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

        // Current IMU orientation in World frame (R_WI)
        let r_wi = self.state.rot.matrix();
        let r_iw = r_wi.transpose();

        // Current Lidar-to-IMU rotation (R_IL from state)
        let r_il = self.state.offset_r_l_i.matrix();

        // Static Base-to-Lidar (R_BL from config)
        let r_bl_rot = self.base_to_lidar.rotation.to_rotation_matrix();
        let r_bl = r_bl_rot.matrix();

        // Compute Base-to-IMU rotation: R_BI = R_BL * R_LI = R_BL * R_IL^T
        let r_bi = r_bl * r_il.transpose();

        // Compute World-to-Base rotation: R_BW = R_BI * R_IW
        let r_bw = r_bi * r_iw;

        // Velocity of IMU in World frame
        let v_i_world = self.state.vel;

        // Vector from IMU to reference point (in IMU frame)
        let p_i_ref = match model_type {
            KinematicModel::Ackermann { imu_to_rear_axle } => imu_to_rear_axle,
            _ => {
                // For other models, the reference point is the base_link center.
                // p_IB = t_IL + R_IL * p_LB = t_IL - R_IL * R_BL^T * t_BL
                let t_il = self.state.offset_t_l_i;
                let t_bl = self.base_to_lidar.translation.vector;
                t_il - r_il * r_bl.transpose() * t_bl
            }
        };

        // Velocity of reference point in Base frame
        // v_ref_base = R_BI * ( R_WI^T * v_i_world + omega_i_imu x p_i_ref )
        let v_ref_base = r_bi * (r_iw * v_i_world + gyro_measurement.cross(&p_i_ref));

        let r_var = constraint_noise_std * constraint_noise_std;

        match model_type {
            KinematicModel::Differential | KinematicModel::Ackermann { .. } => {
                // lateral (y) constraint - velocity index is 12-14 in 24-dim state
                let mut h_lat = Col::<f64>::zeros(24);
                // Jacobian H = d(v_ref_base.y) / d(v_i_world) = e_y^T * R_BI * R_WI^T = row 1 of R_BW
                for j in 0..3 {
                    h_lat[12 + j] = r_bw[(1, j)];
                }
                self.apply_scalar_constraint(h_lat, -v_ref_base.y, r_var, Some(KINEMATIC_MAX_GAIN));

                // vertical (z) constraint
                let mut h_vert = Col::<f64>::zeros(24);
                // Jacobian H = d(v_ref_base.z) / d(v_i_world) = e_z^T * R_BI * R_WI^T = row 2 of R_BW
                for j in 0..3 {
                    h_vert[12 + j] = r_bw[(2, j)];
                }
                self.apply_scalar_constraint(
                    h_vert,
                    -v_ref_base.z,
                    r_var,
                    Some(KINEMATIC_MAX_GAIN),
                );
            }
            KinematicModel::Holonomic2D => {
                // Only vertical (z) constraint: robot stays on ground
                let mut h_vert = Col::<f64>::zeros(24);
                for j in 0..3 {
                    h_vert[12 + j] = r_bw[(2, j)];
                }
                self.apply_scalar_constraint(
                    h_vert,
                    -v_ref_base.z,
                    r_var,
                    Some(KINEMATIC_MAX_GAIN),
                );
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

        // Keep Jacobian moment arms local to the robot for numerical precision.
        self.set_anchor(self.state.pos);

        if measurement_noise_cov <= 0.0 {
            error!("EqF: Non‑positive measurement noise covariance, skipping update.");
            return;
        }

        {
            let mut eval = eval.write().unwrap();
            eval.start("ekf_correct/it").unwrap();
        }
        // Save propagated state and covariance
        let state_propagated = self.state;
        let p_propagated = self.cov.clone();

        let r_inv = 1.0 / measurement_noise_cov;

        let mut converge_count = 0;
        let mut converge_flag = true;

        // The initial iteration computes the first Jacobian.
        for iter in 0..=max_iter {
            let mut valid = true;
            let mut num_measurements = 0usize;

            {
                let mut eval = eval.write().unwrap();
                eval.start("ekf_correct/it/model").unwrap();
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
            eval.stop("ekf_correct/it/model").unwrap();

            if !valid || num_measurements == 0 {
                continue;
            }

            // Use actual number of measurements, not buffer size
            let m = num_measurements;

            eval.start("ekf_correct/it/info_mat").unwrap();

            // dx_new = boxminus(x_, x_propagated) - equation (18): x^k - x^
            let dx_new = box_minus(
                &self.state,
                &state_propagated,
                self.extrinsic_est,
                &self.anchor,
            );

            // H is m x 12 (sparse - only first 12 columns non-zero)
            // We only use the first m rows of h_x (the actual measurements)
            // HTH is 24 x 24 but only 12x12 top-left block is non-zero
            let h_full = &self.h_x;
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

            eval.stop("ekf_correct/it/info_mat").unwrap();
            eval.start("ekf_correct/it/lhs").unwrap();

            // K_front = (HTH / R + P^-1)^-1
            let p_inv = p_propagated.partial_piv_lu().inverse();
            let lhs = &self.hth * r_inv + &p_inv;
            let k_front = lhs.partial_piv_lu().inverse();

            // K = K_front[:, 0:12] * H^T / R  (24 x m)
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

            eval.stop("ekf_correct/it/lhs").unwrap();
            eval.start("ekf_correct/it/gain").unwrap();

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
            eval.stop("ekf_correct/it/gain").unwrap();
            eval.start("ekf_correct/it/state").unwrap();

            // Check for NaN
            let mut has_nan = false;
            for i in 0..24 {
                if !dx_[(i, 0)].is_finite() {
                    has_nan = true;
                    break;
                }
            }
            if has_nan {
                warn!("EqF: NaN detected in iteration {}, aborting.", iter);
                self.state = state_propagated;
                self.cov = p_propagated;
                eval.stop("ekf_correct/it/state").unwrap();
                eval.stop("ekf_correct/it").unwrap();
                return;
            }

            self.state = compensate(&self.state, &dx_, self.extrinsic_est, &self.anchor);

            // Check convergence: if all |dx_[j]| < EPSI, converged
            converge_flag = true;
            for j in 0..24 {
                if dx_[(j, 0)].abs() > measurement_noise_cov {
                    converge_flag = false;
                    eval.stop("ekf_correct/it/state").unwrap();
                    break;
                }
            }

            if converge_flag {
                converge_count += 1;
            }
            // Force a final correspondence search when convergence has stalled.
            if converge_count == 0 && iter == max_iter - 1 {
                converge_flag = true;
            }

            eval.stop("ekf_correct/it/state").unwrap();

            // Exit if converged twice or at max iteration
            if converge_count > 1 || iter == max_iter {
                break;
            }
        }
        let mut eval = eval.write().unwrap();
        eval.stop("ekf_correct/it").unwrap();
        // Covariance update: P = (I - KH) * P  (equation 19)
        eval.start("ekf_correct/cov").unwrap();
        let id24 = Mat::<f64>::identity(24, 24);
        let i_kh = &id24 - &self.k_h;
        self.cov = &i_kh * &p_propagated;

        // The posterior above is expressed about the propagated state, but the
        // estimate has since moved to `self.state`. Re-reference it.
        let dx_total = box_minus(
            &self.state,
            &state_propagated,
            self.extrinsic_est,
            &self.anchor,
        );
        self.reset_covariance_after_update(&state_propagated, &dx_total);
        eval.stop("ekf_correct/cov").unwrap();
    }
}

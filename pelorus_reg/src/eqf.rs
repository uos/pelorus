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

use alloc::vec::Vec;
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::DenseSolveCore;
use faer::prelude::{Reborrow, ReborrowMut};
use faer::{Col, Mat, MatRef};
use log::{error, warn};
// Float maths through libm without std. With std the inherent methods take precedence.
#[allow(unused_imports)]
use nalgebra::ComplexField;
use nalgebra::{DMatrix, Vector3}; // Keep Vector3 for sophus_lie interface
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
/// Rotation indices use target frame first, source frame second:
/// `v_A = R_AB * v_B`, so `R_AB` maps coordinates from frame B to frame A.
///
/// - pos (0-2): position in world frame
/// - rot (3-5): rotation SO3 (IMU to world, R_WI)
/// - offset_R_L_I (6-8): lidar-to-IMU rotation extrinsic
/// - offset_T_L_I (9-11): lidar-to-IMU translation extrinsic
/// - vel (12-14): velocity in world frame
/// - bg (15-17): gyro bias
/// - ba (18-20): accelerometer bias
/// - grav (21-23): gravity vector in world frame
#[derive(Clone, Debug, Copy)]
pub struct StateIKFom {
    pub pos: Vector3<f64>,          // position in world frame
    pub rot: SO3,                   // R_W_I: IMU to world rotation (T_W_I rotation part)
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
    T: nalgebra::Scalar + Copy + From<f64> + ::core::ops::Neg<Output = T>,
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

/// The nonzero 3x3 blocks of a block-sparse matrix, each with its block row
/// and block column. The process Jacobians below have at most 12 of them.
#[derive(Clone, Copy)]
pub struct Blocks {
    list: [(usize, usize, nalgebra::Matrix3<f64>); 12],
    len: usize,
}

impl Blocks {
    fn new() -> Self {
        Self {
            list: [(0, 0, nalgebra::Matrix3::zeros()); 12],
            len: 0,
        }
    }

    fn push(&mut self, row: usize, col: usize, block: nalgebra::Matrix3<f64>) {
        self.list[self.len] = (row, col, block);
        self.len += 1;
    }

    /// The blocks as (block row, block column, block).
    pub fn iter(&self) -> impl Iterator<Item = &(usize, usize, nalgebra::Matrix3<f64>)> {
        self.list[..self.len].iter()
    }

    /// The dense `rows x cols` matrix.
    pub fn to_dense(&self, rows: usize, cols: usize) -> Mat<f64> {
        let mut dense = Mat::<f64>::zeros(rows, cols);
        for (row, col, block) in self.iter() {
            for i in 0..3 {
                for j in 0..3 {
                    dense[(3 * row + i, 3 * col + j)] = block[(i, j)];
                }
            }
        }
        dense
    }

    /// `out += scale * self * x`.
    fn add_left_product(&self, out: &mut Mat<f64>, x: &Mat<f64>, scale: f64) {
        for (row, col, block) in self.iter() {
            for c in 0..x.ncols() {
                for i in 0..3 {
                    let mut sum = 0.0;
                    for k in 0..3 {
                        sum += block[(i, k)] * x[(3 * col + k, c)];
                    }
                    out[(3 * row + i, c)] += scale * sum;
                }
            }
        }
    }

    /// `out += scale * x * self^T`.
    fn add_right_product_transposed(&self, out: &mut Mat<f64>, x: &Mat<f64>, scale: f64) {
        for (row, col, block) in self.iter() {
            for r in 0..x.nrows() {
                for i in 0..3 {
                    let mut sum = 0.0;
                    for k in 0..3 {
                        sum += x[(r, 3 * col + k)] * block[(i, k)];
                    }
                    out[(r, 3 * row + i)] += scale * sum;
                }
            }
        }
    }
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
    df_dx_blocks(s, inp, extrinsic_est, anchor).to_dense(24, 24)
}

/// The nonzero 3x3 blocks of [`df_dx`], in block rows and columns of the
/// ordering pos, rot, offset_R_L_I, offset_T_L_I, vel, bg, ba, grav.
pub fn df_dx_blocks(
    s: &StateIKFom,
    inp: &InputIKFom,
    extrinsic_est: bool,
    anchor: &Vector3<f64>,
) -> Blocks {
    let mut blocks = Blocks::new();
    let pos_a = s.pos - anchor;

    let id3 = nalgebra::Matrix3::<f64>::identity();
    let rot_mat = s.rot.matrix();

    // Generator of the equivariant error dynamics: the bias-corrected angular
    // rate expressed in the world frame.
    let omega_world = rot_mat * (inp.gyro - s.bg);
    let x_blk = skew_symmetric_na(&omega_world);

    blocks.push(0, 4, id3); // pos -- vel
    blocks.push(0, 5, -skew_symmetric_na(&pos_a)); // pos -- gyro bias
    blocks.push(1, 5, -id3); // rot -- gyro bias

    if extrinsic_est {
        // Couples the lever arm to the extrinsic rotation error.
        let z_blk = skew_symmetric_na(&(skew_symmetric_na(&pos_a) * omega_world + s.vel));
        blocks.push(2, 2, x_blk); // extrinsic rot -- extrinsic rot
        blocks.push(3, 2, z_blk); // lever arm -- extrinsic rot
        blocks.push(3, 3, x_blk); // lever arm -- lever arm
    }

    blocks.push(4, 1, skew_symmetric_na(&s.grav)); // vel -- rot
    blocks.push(4, 6, -id3); // vel -- accel bias
    blocks.push(4, 7, id3); // vel -- gravity

    blocks.push(5, 5, x_blk); // gyro bias -- gyro bias

    // Couples the accelerometer bias to the gyro bias error.
    let y_blk = skew_symmetric_na(
        &(rot_mat * (inp.acc - s.ba) + skew_symmetric_na(&s.vel) * omega_world + s.grav),
    );
    blocks.push(6, 5, y_blk); // accel bias -- gyro bias
    blocks.push(6, 6, x_blk); // accel bias -- accel bias

    blocks
}

/// Jacobian of the error dynamics with respect to the IMU noise.
///
/// Every block carries the world rotation because the noise enters in the body
/// frame while the equivariant error state is expressed globally.
pub fn df_dw(s: &StateIKFom, inp: &InputIKFom, anchor: &Vector3<f64>) -> Mat<f64> {
    df_dw_blocks(s, inp, anchor).to_dense(24, 12)
}

/// The nonzero 3x3 blocks of [`df_dw`]. The block columns are the gyro, accel,
/// gyro bias and accel bias noise.
pub fn df_dw_blocks(s: &StateIKFom, _inp: &InputIKFom, anchor: &Vector3<f64>) -> Blocks {
    let mut blocks = Blocks::new();

    let rot_mat = s.rot.matrix();
    let pos_skew_rot = skew_symmetric_na(&(s.pos - anchor)) * rot_mat;
    let vel_skew_rot = skew_symmetric_na(&s.vel) * rot_mat;

    blocks.push(0, 0, pos_skew_rot); // pos -- gyro noise
    blocks.push(1, 0, rot_mat); // rot -- gyro noise
    blocks.push(4, 0, vel_skew_rot); // vel -- gyro noise
    blocks.push(4, 1, rot_mat); // vel -- accel noise
    blocks.push(5, 2, -rot_mat); // gyro bias -- gyro bias noise
    blocks.push(6, 2, -vel_skew_rot); // accel bias -- gyro bias noise
    blocks.push(6, 3, -rot_mat); // accel bias -- accel bias noise

    blocks
}

/// The first `active` columns of the gain front `(P^-1 + C)^-1`, where `C` is
/// `info` in its top-left `active x active` block and zero elsewhere.
///
/// By the push-through identity they equal `P[:, ..active] * (I + C_aa * P_aa)^-1`,
/// which needs one `active x active` inverse instead of inverting `P` and the
/// 24 x 24 sum.
fn gain_front_columns(p: &Mat<f64>, info: MatRef<f64>, active: usize) -> Mat<f64> {
    let mut lhs = Mat::<f64>::identity(active, active);
    matmul(
        lhs.rb_mut(),
        faer::Accum::Add,
        info,
        p.submatrix(0, 0, active, active),
        1.0,
        faer::Par::Seq,
    );
    let lhs_inv = lhs.partial_piv_lu().inverse();
    let mut out = Mat::<f64>::zeros(p.nrows(), active);
    matmul(
        out.rb_mut(),
        faer::Accum::Replace,
        p.subcols(0, active),
        lhs_inv.rb(),
        1.0,
        faer::Par::Seq,
    );
    out
}

pub struct Eqf {
    pub state: StateIKFom,
    pub cov: Mat<f64>,        // 24x24 covariance matrix in faer
    pub fxtcov: Mat<f64>,     // 24x24 temporary for cov update in predict
    pub fxcov: Mat<f64>,      // 24x24 temporary for cov update in predict
    pub f_w_q: Mat<f64>,      // 24x12 temporary for cov update in predict
    pub id: Mat<f64>,         // 24x24 temporary for cov update in scalar constraint
    pub k_mat: Mat<f64>,      // 24x1 temporary for cov update in scalar constraint
    pub h_t_mat: Mat<f64>,    // 1x24 temporary for cov update in scalar constraint
    pub k_h_t: Mat<f64>,      // 24x24 temporary for cov update in scalar constraint
    pub k_k: Mat<f64>,        // 24x24 temporary for cov update in scalar constraint
    pub cov_i_kh: Mat<f64>,   // 24x24 temporary for cov update in scalar constraint
    pub i_cov_i_kh: Mat<f64>, // 24x24 temporary for cov update in scalar constraint
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
    pub k_h: Mat<f64>, // K * H (24 x 24)
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
            k_h: Mat::<f64>::zeros(24, 24),
            fxtcov: Mat::<f64>::zeros(24, 24),
            fxcov: Mat::<f64>::zeros(24, 24),
            f_w_q: Mat::<f64>::zeros(24, 12),
            id: Mat::<f64>::identity(24, 24),
            k_mat: Mat::<f64>::zeros(24, 1),
            h_t_mat: Mat::<f64>::zeros(1, 24),
            k_h_t: Mat::<f64>::zeros(24, 24),
            k_k: Mat::<f64>::zeros(24, 24),
            cov_i_kh: Mat::<f64>::zeros(24, 24),
            i_cov_i_kh: Mat::<f64>::zeros(24, 24),
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
        // Both Jacobians are block-sparse, so the covariance update multiplies
        // only their nonzero 3x3 blocks.
        let f_x = df_dx_blocks(&self.state, i_in, self.extrinsic_est, &self.anchor);
        let f_w = df_dw_blocks(&self.state, i_in, &self.anchor);

        // State update: x = propagate(x, f * dt)
        let f_dt = &f * dt;
        self.state = propagate(&self.state, &f_dt);

        // Cov update: P = F_x * P * F_x^T + F_w * Q * F_w^T with
        // F_x = I + f_x * dt and F_w = f_w * dt.
        // P * F_x^T = P + dt * P * f_x^T
        self.fxtcov.copy_from(&self.cov);
        f_x.add_right_product_transposed(&mut self.fxtcov, &self.cov, dt);
        // F_x * (P * F_x^T) = (P * F_x^T) + dt * f_x * (P * F_x^T)
        self.fxcov.copy_from(&self.fxtcov);
        f_x.add_left_product(&mut self.fxcov, &self.fxtcov, dt);
        // F_w * Q * F_w^T = dt^2 * (f_w * Q) * f_w^T
        self.f_w_q.fill(0.0);
        f_w.add_left_product(&mut self.f_w_q, q, dt);
        f_w.add_right_product_transposed(&mut self.fxcov, &self.f_w_q, dt);
        self.cov.copy_from(&self.fxcov);
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
        gyro_measurement: Vector3<f64>,
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

        let components: &[usize] = match kinematic {
            KinematicModel::Differential | KinematicModel::Ackermann { .. } => &[0],
            KinematicModel::Holonomic2D => &[0, 1],
            KinematicModel::Full3D => &[0, 1, 2],
        };

        let covariance_scale = 1.0 + dt_gap * 5.0;
        let mut measurement_cov = DMatrix::<f64>::zeros(components.len(), components.len());
        if let Some(variance) = cov_override {
            if !variance.is_finite() || variance <= 0.0 {
                warn!("Invalid wheel odometry covariance override");
                return;
            }
            for diagonal in 0..components.len() {
                measurement_cov[(diagonal, diagonal)] = variance * covariance_scale;
            }
        } else {
            // geometry_msgs/TwistWithCovariance is row-major in the order
            // [linear xyz, angular xyz], so the linear diagonal is 0, 7, 14.
            for (row, &source_row) in components.iter().enumerate() {
                for (col, &source_col) in components.iter().enumerate() {
                    let lhs = odom.twist_cov[source_row * 6 + source_col];
                    let rhs = odom.twist_cov[source_col * 6 + source_row];
                    measurement_cov[(row, col)] = if lhs.is_finite() && rhs.is_finite() {
                        0.5 * (lhs + rhs) * covariance_scale
                    } else {
                        0.0
                    };
                }
                if !measurement_cov[(row, row)].is_finite() || measurement_cov[(row, row)] < 0.001 {
                    measurement_cov[(row, row)] = 0.001 * covariance_scale;
                }
            }
        }

        // Whiten the selected velocity observation so correlated covariance can
        // still be applied through the scalar Joseph-form correction below.
        let whitener = measurement_cov
            .clone()
            .cholesky()
            .and_then(|chol| chol.l().try_inverse())
            .unwrap_or_else(|| {
                if should_log_once_default("wheel_odom_covariance_not_positive_definite") {
                    warn!("Wheel odometry covariance is not positive definite; using its diagonal");
                }
                DMatrix::from_diagonal(&measurement_cov.diagonal().map(|v| 1.0 / v.sqrt()))
            });

        for whitened_row in 0..components.len() {
            // Re-linearize after every scalar update. This preserves the usual
            // sequential-EKF behavior while supporting a correlated input covariance.
            let predicted = self.predict_odom_velocity(&odom, kinematic, gyro_measurement);
            let mut residual = 0.0;
            for (col, &component) in components.iter().enumerate() {
                residual += whitener[(whitened_row, col)]
                    * (odom.twist_linear[component] - predicted[component]);
            }

            let h = self.odom_velocity_jacobian_row(
                &odom,
                kinematic,
                gyro_measurement,
                components,
                whitener.row(whitened_row).iter().copied(),
            );
            self.apply_scalar_constraint(h, residual, 1.0, None, false);
        }
    }

    /// Predict the velocity reported by wheel odometry in its own measurement frame.
    fn predict_odom_velocity(
        &self,
        odom: &OdomMsg,
        kinematic: KinematicModel,
        gyro_measurement: Vector3<f64>,
    ) -> Vector3<f64> {
        self.predict_odom_velocity_for_state(&self.state, odom, kinematic, gyro_measurement)
    }

    fn predict_odom_velocity_for_state(
        &self,
        state: &StateIKFom,
        odom: &OdomMsg,
        kinematic: KinematicModel,
        gyro_measurement: Vector3<f64>,
    ) -> Vector3<f64> {
        let r_il = state.offset_r_l_i.matrix();
        let r_bl = self.base_to_lidar.rotation.to_rotation_matrix();
        let p_i_ref = match kinematic {
            KinematicModel::Ackermann { imu_to_rear_axle } => imu_to_rear_axle,
            _ => {
                let t_bl = self.base_to_lidar.translation.vector;
                state.offset_t_l_i - r_il * r_bl.matrix().transpose() * t_bl
            }
        };

        let omega_i = gyro_measurement - state.bg;
        let v_ref_i = state.rot.matrix().transpose() * state.vel + omega_i.cross(&p_i_ref);
        odom.twist_frame_to_imu.transpose() * v_ref_i
    }

    /// Linearize a whitened wheel-velocity row in the EqF error coordinates.
    ///
    /// Numerical differentiation is intentional here: it includes the cancellation
    /// of global rotation error in a body-velocity observation as well as gyro-bias,
    /// lever-arm, and online-extrinsic terms under the exact `compensate` action.
    fn odom_velocity_jacobian_row(
        &self,
        odom: &OdomMsg,
        kinematic: KinematicModel,
        gyro_measurement: Vector3<f64>,
        components: &[usize],
        weights: impl Iterator<Item = f64>,
    ) -> Col<f64> {
        const EPS: f64 = 1e-6;
        let weights: Vec<f64> = weights.collect();
        let mut h = Col::<f64>::zeros(24);
        let mut delta = Mat::<f64>::zeros(24, 1);

        for column in 0..24 {
            delta[(column, 0)] = EPS;
            let high_state = compensate(&self.state, &delta, self.extrinsic_est, &self.anchor);
            let high = self.predict_odom_velocity_for_state(
                &high_state,
                odom,
                kinematic,
                gyro_measurement,
            );

            delta[(column, 0)] = -EPS;
            let low_state = compensate(&self.state, &delta, self.extrinsic_est, &self.anchor);
            let low =
                self.predict_odom_velocity_for_state(&low_state, odom, kinematic, gyro_measurement);
            delta[(column, 0)] = 0.0;

            for (&component, &weight) in components.iter().zip(weights.iter()) {
                h[column] += weight * (high[component] - low[component]) / (2.0 * EPS);
            }
        }
        h
    }

    /// Fuse one scalar measurement whose Jacobian is `h`.
    ///
    /// `lock_pose` keeps model-only non-holonomic constraints from directly
    /// moving the Lidar pose. `max_gain` can damp those synthetic constraints.
    fn apply_scalar_constraint(
        &mut self,
        h: Col<f64>,
        residual: f64,
        r_var: f64,
        max_gain: Option<f64>,
        lock_pose: bool,
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

        if lock_pose {
            // Model-only non-holonomic constraints should not directly move the
            // Lidar pose. Actual wheel observations use the complete Kalman gain.
            for i in 0..6 {
                k[i] = 0.0;
            }
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

        // Static Lidar-to-Base rotation (R_BL from config)
        let r_bl_rot = self.base_to_lidar.rotation.to_rotation_matrix();
        let r_bl = r_bl_rot.matrix();

        // Compute IMU-to-Base rotation: R_BI = R_BL * R_LI = R_BL * R_IL^T
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
                self.apply_scalar_constraint(
                    h_lat,
                    -v_ref_base.y,
                    r_var,
                    Some(KINEMATIC_MAX_GAIN),
                    true,
                );

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
                    true,
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
                    true,
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
        eval: alloc::sync::Arc<pelorus_core::sync::RwLock<&mut RuntimeEvaluator>>,
        mut model: impl FnMut(
            &StateIKFom,
            &mut bool,     // valid
            bool,          // converge (for re-searching correspondences if needed)
            &mut Col<f64>, // residual h
            &mut Mat<f64>, // jacobian H (m x 12, sparse)
            &mut usize,    // num_measurements - actual number of valid measurements
        ),
    ) {
        pelorus_core::complete_synchronously(self.correct_async(
            measurement_noise_cov,
            measurement_time,
            max_iter,
            eval,
            async |state, valid, converged, h, h_x, count| {
                model(state, valid, converged, h, h_x, count);
            },
        ));
    }

    pub async fn correct_async(
        &mut self,
        measurement_noise_cov: f64,
        measurement_time: f64,
        max_iter: usize,
        eval: alloc::sync::Arc<pelorus_core::sync::RwLock<&mut RuntimeEvaluator>>,
        mut model: impl AsyncFnMut(
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
            let mut eval = eval.write();
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
                let mut eval = eval.write();
                eval.start("ekf_correct/it/model").unwrap();
            }

            model(
                &self.state,
                &mut valid,
                converge_flag,
                &mut self.h,
                &mut self.h_x,
                &mut num_measurements,
            )
            .await;

            let mut eval = eval.write();
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
            // The extrinsic columns 6..12 are only active with online extrinsic
            // estimation. Without it they are zero and add nothing to H^T * H
            // or H^T * h, so only the first 6 columns enter the products.
            let active = if self.extrinsic_est { 12 } else { 6 };
            let h_full = &self.h_x;
            let h = h_full.subrows(0, m).subcols(0, active);
            let h_t = h.transpose(); // active x m

            // Compute H^T * H (12 x 12)
            let mut hth_12x12 = Mat::<f64>::zeros(12, 12);
            matmul(
                hth_12x12.as_mut().submatrix_mut(0, 0, active, active),
                faer::Accum::Replace,
                h_t.rb(),
                h.rb(),
                1.0,
                faer::Par::Seq,
            );

            // H^T * h (12 x 1)
            let mut ht_residual = Mat::<f64>::zeros(12, 1);
            for j in 0..active {
                let column = h.col(j);
                let mut sum = 0.0;
                for i in 0..m {
                    sum += column[i] * self.h[i];
                }
                ht_residual[(j, 0)] = sum;
            }

            eval.stop("ekf_correct/it/info_mat").unwrap();
            eval.start("ekf_correct/it/lhs").unwrap();

            // C = H^T * H / R in its active block.
            let info = hth_12x12.submatrix(0, 0, active, active) * r_inv;
            // K_front = (P^-1 + C)^-1, needed only in its active columns.
            let k_front_active = gain_front_columns(&p_propagated, info.rb(), active);

            eval.stop("ekf_correct/it/lhs").unwrap();
            eval.start("ekf_correct/it/gain").unwrap();

            // The gain K = K_front * H^T / R is 24 x m, but only its products
            // with H and h are needed. Both go through H^T, so they come from
            // the products above and nothing here grows with m:
            // K * H = K_front * C and K * h = K_front * (H^T * h) / R.
            for i in 0..24 {
                for j in 0..24 {
                    self.k_h[(i, j)] = 0.0;
                }
            }
            matmul(
                self.k_h.as_mut().submatrix_mut(0, 0, 24, active),
                faer::Accum::Replace,
                k_front_active.rb(),
                info.rb(),
                1.0,
                faer::Par::Seq,
            );

            // dx_ = K * h + (KH - I) * dx_new  (equation 18)
            // K * h (24 x 1)
            let mut kh_vec = Mat::<f64>::zeros(24, 1);
            matmul(
                kh_vec.rb_mut(),
                faer::Accum::Replace,
                k_front_active.rb(),
                ht_residual.subrows(0, active),
                r_inv,
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
        let mut eval = eval.write();
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Isometry3, Matrix3};
    use pelorus_core::time_msg_from_secs;

    fn odom(linear: Vector3<f64>, frame_to_imu: Matrix3<f64>) -> OdomMsg {
        let mut covariance = [0.0; 36];
        covariance[0] = 0.01;
        covariance[7] = 0.01;
        covariance[14] = 0.01;
        OdomMsg {
            twist_linear: linear,
            twist_angular: Vector3::zeros(),
            twist_cov: covariance,
            twist_frame_to_imu: frame_to_imu,
            stamp: time_msg_from_secs(0.0),
        }
    }

    fn initialized_filter() -> Eqf {
        let mut filter = Eqf::new(9.81, Isometry3::identity());
        filter.initialized = true;
        filter
    }

    #[test]
    fn gain_front_columns_match_dense_inverse() {
        let p = Mat::from_fn(24, 24, |i, j| {
            let (a, b) = (i.min(j) as f64, i.max(j) as f64);
            0.01 * (a * 0.37 + b * 0.11).sin() + if i == j { 0.5 + 0.01 * a } else { 0.0 }
        });
        for active in [6, 12] {
            let h = Mat::from_fn(40, active, |i, j| ((i * 7 + j * 3) as f64 * 0.13).cos());
            let info = h.transpose() * &h * 1000.0;
            let mut full_info = Mat::<f64>::zeros(24, 24);
            for i in 0..active {
                for j in 0..active {
                    full_info[(i, j)] = info[(i, j)];
                }
            }
            let expected = (p.partial_piv_lu().inverse() + &full_info)
                .partial_piv_lu()
                .inverse();

            // The dense reference inverts twice, so it carries more rounding.
            let scale = (0..24)
                .flat_map(|i| (0..active).map(move |j| (i, j)))
                .map(|(i, j)| expected[(i, j)].abs())
                .fold(0.0, f64::max);
            let columns = gain_front_columns(&p, info.rb(), active);
            for i in 0..24 {
                for j in 0..active {
                    let difference = (columns[(i, j)] - expected[(i, j)]).abs();
                    assert!(
                        difference < 1e-10 * scale,
                        "({i}, {j}) differs by {difference}"
                    );
                }
            }
        }
    }

    #[test]
    fn block_sparse_prediction_matches_dense_products() {
        let mut filter = initialized_filter();
        filter.extrinsic_est = true;
        filter.state.rot = SO3::exp(Vector3::new(0.2, -0.1, 0.7));
        filter.state.pos = Vector3::new(4.0, 1.0, -2.0);
        filter.state.vel = Vector3::new(2.0, -0.4, 0.3);
        filter.cov = Mat::from_fn(24, 24, |i, j| {
            let (a, b) = (i.min(j) as f64, i.max(j) as f64);
            0.01 * (a * 0.37 + b * 0.11).sin() + if i == j { 1.0 } else { 0.0 }
        });
        let q = Mat::from_fn(12, 12, |i, j| {
            if i / 3 == j / 3 {
                0.01 * (1.0 + i as f64) + 0.001 * j as f64
            } else {
                0.0
            }
        });
        let input = InputIKFom {
            acc: Vector3::new(0.3, -0.2, 9.7),
            gyro: Vector3::new(0.05, 0.1, -0.2),
        };
        let dt = 0.005;

        let f_x =
            Mat::<f64>::identity(24, 24) + df_dx(&filter.state, &input, true, &filter.anchor) * dt;
        let f_w = df_dw(&filter.state, &input, &filter.anchor) * dt;
        let fp = &f_x * &filter.cov;
        let fw_q = &f_w * &q;
        let expected = &fp * f_x.transpose() + &fw_q * f_w.transpose();

        filter.predict(dt, &q, &input);
        for i in 0..24 {
            for j in 0..24 {
                let difference = (filter.cov[(i, j)] - expected[(i, j)]).abs();
                assert!(difference < 1e-12, "({i}, {j}) differs by {difference}");
            }
        }
    }

    #[test]
    fn body_velocity_jacobian_cancels_global_rotation_error() {
        let mut filter = initialized_filter();
        filter.state.rot = SO3::exp(Vector3::new(0.2, -0.1, 0.7));
        filter.state.vel = Vector3::new(2.0, -0.4, 0.3);
        let measurement = odom(Vector3::zeros(), Matrix3::identity());

        let h = filter.odom_velocity_jacobian_row(
            &measurement,
            KinematicModel::Differential,
            Vector3::zeros(),
            &[0],
            ::core::iter::once(1.0),
        );

        for index in 3..6 {
            assert!(
                h[index].abs() < 1e-8,
                "rotation column {index}: {}",
                h[index]
            );
        }
        let expected_matrix = filter.state.rot.matrix().transpose();
        let expected = expected_matrix.row(0);
        for index in 0..3 {
            assert!((h[12 + index] - expected[index]).abs() < 1e-8);
        }
    }

    #[test]
    fn differential_odom_corrects_forward_axis_not_world_x() {
        let mut filter = initialized_filter();
        filter.state.rot = SO3::exp(Vector3::new(0.0, 0.0, ::core::f64::consts::FRAC_PI_2));
        let measurement = odom(Vector3::new(1.0, 0.0, 0.0), Matrix3::identity());

        filter.correct_with_body_velocities(
            measurement,
            KinematicModel::Differential,
            None,
            Vector3::zeros(),
        );

        assert!(filter.state.vel.y > 0.9);
        assert!(filter.state.vel.x.abs() < 1e-8);
    }

    #[test]
    fn configured_odom_frame_rotation_is_respected() {
        let mut filter = initialized_filter();
        let frame_to_imu = nalgebra::Rotation3::from_axis_angle(
            &Vector3::z_axis(),
            ::core::f64::consts::FRAC_PI_2,
        )
        .into_inner();
        let measurement = odom(Vector3::new(1.0, 0.0, 0.0), frame_to_imu);

        filter.correct_with_body_velocities(
            measurement,
            KinematicModel::Differential,
            None,
            Vector3::zeros(),
        );

        assert!(filter.state.vel.y > 0.9);
        assert!(filter.state.vel.x.abs() < 1e-8);
    }

    #[test]
    fn linear_y_variance_comes_from_ros_covariance_index_seven() {
        let mut filter = initialized_filter();
        let mut measurement = odom(Vector3::new(0.0, 1.0, 0.0), Matrix3::identity());
        measurement.twist_cov[4] = 1e-6;
        measurement.twist_cov[7] = 100.0;

        filter.correct_with_body_velocities(
            measurement,
            KinematicModel::Holonomic2D,
            None,
            Vector3::zeros(),
        );

        assert!(filter.state.vel.y > 0.0);
        assert!(filter.state.vel.y < 0.02);
    }
}

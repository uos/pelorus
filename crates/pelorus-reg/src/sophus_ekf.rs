use log::warn;
use nalgebra::*;
use sophus_lie::traits::IsTranslationProductGroup;

pub type SO3 = sophus_lie::Rotation3<f64, 1>;
pub type SE3 = sophus_lie::Isometry3<f64, 1>;

// TODO use config, needs some refactor
const GRAVITY_DEFAULT: f64 = 9.81;

#[derive(Clone, Debug)]
pub struct StateIKFom {
    pub pose: SE3,
    pub offset_r_l_i: SO3,
    pub offset_t_l_i: Vector3<f64>,
    pub vel: Vector3<f64>,
    pub bg: Vector3<f64>,   // estimated gyro bias
    pub ba: Vector3<f64>,   // accelerometer bias (TODO never set)
    pub grav: Vector3<f64>, // gravity vector
}

impl StateIKFom {
    pub fn new(gravity: f64) -> Self {
        StateIKFom {
            pose: SE3::identity(),
            offset_r_l_i: SO3::identity(),
            offset_t_l_i: Vector3::zeros(),
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
        &(-dt * skew_symmetric(&(s.pose.translation() + &s.vel * dt)) * s.pose.rotation().matrix()),
    );

    // vel
    cov.fixed_view_mut::<3, 3>(6, 0).copy_from(
        &(dt * s.pose.rotation().matrix() * skew_symmetric(&s.ba)
            - skew_symmetric(&(dt * s.pose.rotation().matrix() * &inp.acc))
            + skew_symmetric(&s.vel) * s.pose.rotation().matrix() * skew_symmetric(&s.bg)
            + skew_symmetric(&((s.pose.rotation().matrix() * (&inp.acc - &s.ba) + &s.grav) * dt))
                * (Matrix3::identity() + s.pose.rotation().matrix() * skew_symmetric(&s.bg))),
    );
    cov.fixed_view_mut::<3, 3>(6, 9).copy_from(
        &(-dt
            * skew_symmetric(
                &(&s.vel + (s.pose.rotation().matrix() * (&inp.acc - s.ba) + &s.grav) * dt),
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

    return cov;
}

/// Get the Jacobian of the state transition function f(x, u) with respect to w. w is the noise
pub fn df_dw(s: &StateIKFom, inp: &InputIKFom, dt: f64) -> SMatrix<f64, 18, 12> {
    let mut cov = SMatrix::<f64, 18, 12>::zeros();

    // rot
    cov.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(-s.pose.rotation().matrix()));

    // pos
    cov.fixed_view_mut::<3, 3>(3, 0).copy_from(
        &(-skew_symmetric(&(s.pose.translation() + &s.vel * dt)) * s.pose.rotation().matrix()),
    );

    // vel
    cov.fixed_view_mut::<3, 3>(6, 0).copy_from(
        &(-skew_symmetric(
            &(&s.vel + (s.pose.rotation().matrix() * (&inp.acc - &s.ba) + &s.grav) * dt),
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

    return cov;
}

pub struct Esekf {
    pub state: StateIKFom,
    pub cov: SMatrix<f64, 18, 18>,
}

/// Adding the state and the perturbation, stay on the manifold
fn box_plus(x: &StateIKFom, f: &SMatrix<f64, 18, 1>) -> StateIKFom {
    let mut x_r = x.clone();

    let delta_rot_vec = f.fixed_rows::<3>(0);
    let delta_pos = f.fixed_rows::<3>(3);
    x_r.pose
        .set_translation(&(x.pose.translation() + &delta_pos));
    x_r.pose.set_rotation(
        &x.pose
            .rotation()
            .group_mul(&SO3::exp(&delta_rot_vec.clone_owned())),
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
            .group_mul(&x1.pose.rotation())
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

impl Esekf {
    pub fn new(g: f64) -> Self {
        Self {
            state: StateIKFom::new(g),
            cov: SMatrix::<f64, 18, 18>::identity(),
        }
    }

    pub fn predict(&mut self, dt: f64, q: &SMatrix<f64, 12, 12>, i_in: &InputIKFom) {
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

    pub fn correct(
        &mut self,
        measurement_noise_cov: f64,
        // TODO check change to FN
        mut model: impl FnMut(&StateIKFom, &mut bool, bool, &mut DVector<f64>, &mut DMatrix<f64>),
    ) {
        let converge = true;
        let mut h = DVector::<f64>::default();
        let mut h_x = DMatrix::<f64>::default();
        let state_propagated = self.state.clone(); // state x
        let p_propagated = self.cov.clone();

        let mut valid = true;

        model(&self.state, &mut valid, converge, &mut h, &mut h_x);

        if !valid {
            warn!("iteration not valid");
            return;
        }

        let hth = h_x.transpose() * &h_x;
        let k = (hth / measurement_noise_cov + p_propagated.try_inverse().unwrap())
            .try_inverse()
            .unwrap()
            * h_x.transpose()
            / measurement_noise_cov;

        let kh = &k * &h;
        let kh_norm = kh.norm();
        let mut s = nalgebra::SMatrix::<f64, 8, 8>::zeros();
        s.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&skew_symmetric(&kh.fixed_view::<3, 1>(0, 0).xyz()));
        s.fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&(&kh.fixed_view::<3, 1>(3, 0).xyz()));
        s.fixed_view_mut::<3, 1>(0, 4)
            .copy_from(&(&kh.fixed_view::<3, 1>(6, 0).xyz()));
        s.fixed_view_mut::<3, 1>(0, 5)
            .copy_from(&(&kh.fixed_view::<3, 1>(9, 0).xyz()));
        s.fixed_view_mut::<3, 1>(0, 6)
            .copy_from(&(&kh.fixed_view::<3, 1>(12, 0).xyz()));
        s.fixed_view_mut::<3, 1>(0, 7)
            .copy_from(&(&kh.fixed_view::<3, 1>(15, 0).xyz()));

        let exps = nalgebra::SMatrix::<f64, 8, 8>::identity()
            + s
            + (1.0 - kh_norm.cos()) / (kh_norm * kh_norm) * s * s
            + (kh_norm - kh_norm.sin()) / (kh_norm * kh_norm * kh_norm) * s * s * s;

        let mut x_mat = nalgebra::SMatrix::<f64, 8, 8>::identity();
        x_mat
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(state_propagated.pose.rotation().matrix()));
        x_mat
            .fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&(state_propagated.pose.translation()));
        x_mat
            .fixed_view_mut::<3, 1>(0, 4)
            .copy_from(&(state_propagated.vel));
        x_mat
            .fixed_view_mut::<3, 1>(0, 5)
            .copy_from(&(state_propagated.bg));
        x_mat
            .fixed_view_mut::<3, 1>(0, 6)
            .copy_from(&(state_propagated.ba));
        x_mat
            .fixed_view_mut::<3, 1>(0, 7)
            .copy_from(&(state_propagated.grav));

        let new_state = exps * x_mat;
        // self.state.pose.set_rotation(
        //     &SO3::try_from_mat(&new_state.fixed_view::<3, 3>(0, 0).into_owned()).unwrap(),
        // );
        let new_rot_matrix = new_state.fixed_view::<3, 3>(0, 0).into_owned();
        let new_quat = UnitQuaternion::from_matrix(&new_rot_matrix);
        let new_so3 = SO3::try_from_mat(&new_quat.to_rotation_matrix().matrix()).unwrap();
        self.state.pose.set_rotation(&new_so3);
        self.state
            .pose
            .set_translation(&new_state.fixed_view::<3, 1>(0, 3).xyz());
        self.state.vel = new_state.fixed_view::<3, 1>(0, 4).xyz();
        self.state.bg = new_state.fixed_view::<3, 1>(0, 5).xyz();
        self.state.ba = new_state.fixed_view::<3, 1>(0, 6).xyz();
        self.state.grav = new_state.fixed_view::<3, 1>(0, 7).xyz();

        self.cov = p_propagated - k * &h_x * p_propagated;
    }
}

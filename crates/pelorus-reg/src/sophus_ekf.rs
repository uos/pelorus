/// Extended Kalman Filter on the SO(3) manifold.
//use crate::ekf_model::CustomModel;
use nalgebra::*;

pub type SO3 = sophus_lie::Rotation3<f64, 1>;

const CONVERGE_EPSILON: f64 = 0.001;
const GRAVITY_DEFAULT: f64 = 9.81;

#[derive(Clone, Debug)]
pub struct StateIKFom {
    pub pos: Vector3<f64>,
    pub rot: SO3,
    pub offset_r_l_i: SO3,
    pub offset_t_l_i: Vector3<f64>,
    pub vel: Vector3<f64>,
    pub bg: Vector3<f64>,   // estimated gyro bias
    pub ba: Vector3<f64>,   // accelerometer bias (TODO never set)
    pub grav: Vector3<f64>, // gravity vector
}

impl StateIKFom {
    pub fn new(gravity: f64) -> Self {
        let mut res = StateIKFom::default();
        res.grav = Vector3::new(0.0, 0.0, -gravity);
        res
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

/// Get the state transition function f(x, u) Matrix
fn get_f(s: &StateIKFom, inp: &InputIKFom) -> SMatrix<f64, 24, 1> {
    let angular_vel = inp.gyro - s.bg; // bg is the estimated gyro bias
    let acc_inertial = s.rot.matrix() * (inp.acc - s.ba); // transform to world
    let acc_grav = acc_inertial + s.grav; // acceleration added gravity

    let mut res = SMatrix::<f64, 24, 1>::zeros();
    res.fixed_rows_mut::<3>(0).copy_from(&s.vel);
    res.fixed_rows_mut::<3>(3).copy_from(&angular_vel);
    res.fixed_rows_mut::<3>(12).copy_from(&acc_grav);

    res
}

/// Get the Jacobian of the state transition function f(x, u) with respect to x. x is the state
pub fn df_dx(s: &StateIKFom, inp: &InputIKFom) -> SMatrix<f64, 24, 24> {
    let mut cov = SMatrix::<f64, 24, 24>::zeros();
    cov.fixed_view_mut::<3, 3>(0, 12)
        .copy_from(&Matrix3::identity());
    let acc = inp.acc - s.ba;

    let r_neg = -s.rot.matrix();
    cov.fixed_view_mut::<3, 3>(12, 3)
        .copy_from(&(r_neg * SO3::hat(&acc)));
    cov.fixed_view_mut::<3, 3>(12, 18).copy_from(&r_neg);

    cov.fixed_view_mut::<3, 3>(12, 21)
        .copy_from(&Matrix3::identity());
    cov.fixed_view_mut::<3, 3>(3, 15)
        .copy_from(&(-Matrix3::identity()));

    return cov;
}

/// Get the Jacobian of the state transition function f(x, u) with respect to w. w is the noise
pub fn df_dw(s: &StateIKFom, _inp: &InputIKFom) -> SMatrix<f64, 24, 12> {
    let mut cov = SMatrix::<f64, 24, 12>::zeros();
    cov.fixed_view_mut::<3, 3>(12, 3)
        .copy_from(&(-s.rot.matrix()));
    cov.fixed_view_mut::<3, 3>(3, 0)
        .copy_from(&(-Matrix3::identity()));
    cov.fixed_view_mut::<3, 3>(15, 6)
        .copy_from(&(Matrix3::identity()));
    cov.fixed_view_mut::<3, 3>(18, 9)
        .copy_from(&(Matrix3::identity()));

    return cov;
}

pub struct Esekf {
    pub state: StateIKFom,
    pub cov: SMatrix<f64, 24, 24>,
}

/// Adding the state and the perturbation, stay on the manifold
fn box_plus(x: &StateIKFom, f: &SMatrix<f64, 24, 1>) -> StateIKFom {
    let mut x_r = StateIKFom::default();
    x_r.pos = x.pos + f.fixed_rows::<3>(0);
    x_r.rot = x
        .rot
        .group_mul(&SO3::exp(&f.fixed_rows::<3>(3).clone_owned()));
    x_r.offset_r_l_i = x
        .offset_r_l_i
        .group_mul(&SO3::exp(&f.fixed_rows::<3>(6).clone_owned()));
    x_r.offset_t_l_i = x.offset_t_l_i + f.fixed_rows::<3>(9);
    x_r.vel = x.vel + f.fixed_rows::<3>(12);
    x_r.bg = x.bg + f.fixed_rows::<3>(15);
    x_r.ba = x.ba + f.fixed_rows::<3>(18);
    x_r.grav = x.grav + f.fixed_rows::<3>(21);

    x_r
}

/// Difference between two states, stay on the manifold
fn box_minus(x1: &StateIKFom, x2: &StateIKFom) -> SMatrix<f64, 24, 1> {
    let mut x_r = SMatrix::<f64, 24, 1>::zeros();

    x_r.fixed_rows_mut::<3>(0).copy_from(&(x1.pos - x2.pos));
    x_r.fixed_rows_mut::<3>(3)
        .copy_from(&x2.rot.inverse().group_mul(&x1.rot).log());
    x_r.fixed_rows_mut::<3>(6)
        .copy_from(&x2.offset_r_l_i.inverse().group_mul(&x1.offset_r_l_i).log());

    x_r.fixed_rows_mut::<3>(9)
        .copy_from(&(x1.offset_t_l_i - x2.offset_t_l_i));
    x_r.fixed_rows_mut::<3>(12).copy_from(&(x1.vel - x2.vel));
    x_r.fixed_rows_mut::<3>(15).copy_from(&(x1.bg - x2.bg));
    x_r.fixed_rows_mut::<3>(18).copy_from(&(x1.ba - x2.ba));
    x_r.fixed_rows_mut::<3>(21).copy_from(&(x1.grav - x2.grav));

    x_r
}

impl Esekf {
    pub fn new(g: f64) -> Self {
        Self {
            state: StateIKFom::new(g),
            cov: SMatrix::<f64, 24, 24>::identity(),
        }
    }

    pub fn predict(&mut self, dt: f64, q: &SMatrix<f64, 12, 12>, i_in: &InputIKFom) {
        let f = get_f(&self.state, i_in);
        let mut f_x = df_dx(&self.state, i_in);
        let f_w = df_dw(&self.state, i_in);

        // State update
        self.state = box_plus(&self.state, &(f * dt));
        f_x = SMatrix::<f64, 24, 24>::identity() + f_x * dt;

        // Cov update (we get less secure about our state after prediction without correction)
        let dt_fw = dt * f_w;
        self.cov = f_x * self.cov * f_x.transpose() + dt_fw * q * dt_fw.transpose();
    }

    pub fn correct(
        &mut self,
        measurement_noise_cov: f64,
        max_iter: i8,
        extrinsic_estimation: bool,
        mut model: impl FnMut(&StateIKFom, &mut bool, bool, &mut DVector<f64>, &mut DMatrix<f64>, bool),
    ) {
        let mut converge = true;
        let mut h = DVector::<f64>::default();
        let mut h_x = DMatrix::<f64>::default();
        let mut t = 0;
        let state_propagated = self.state.clone(); // state x

        // TODO strange loop starting point, fix when everything is working
        for i in -1..max_iter {
            let mut valid = true;

            model(
                &self.state,
                &mut valid,
                converge,
                &mut h,
                &mut h_x,
                extrinsic_estimation,
            );

            if !valid {
                // dbg!("not valid");
                continue; // TODO could also break here since no change is made to the state
            }

            // state differences made by our model, stay on the manifold
            let dx_new = box_minus(&self.state, &state_propagated);

            let hessian_t = &h_x.transpose();
            let hth = {
                let mut hth = SMatrix::<f64, 24, 24>::zeros();
                let gramian = hessian_t * &h_x;
                debug_assert!(gramian.nrows() == 12);
                debug_assert!(gramian.ncols() == 12);
                hth.fixed_view_mut::<12, 12>(0, 0).copy_from(&(gramian));
                hth
            };

            let kalman_gain_pre = (hth / measurement_noise_cov + self.cov.try_inverse().unwrap())
                .try_inverse()
                .unwrap();
            let k = kalman_gain_pre.fixed_view::<24, 12>(0, 0) * hessian_t / measurement_noise_cov;
            let kh = {
                let mut kh = SMatrix::<f64, 24, 24>::zeros();
                let khessian = &k * &h_x;
                debug_assert!(khessian.nrows() == 24);
                debug_assert!(khessian.ncols() == 12);
                kh.fixed_view_mut::<24, 12>(0, 0).copy_from(&khessian);
                kh
            };
            // avg of all k
            // let k_avg = k.iter().sum::<f64>() / k.len() as f64;
            // let h_avg = h.iter().sum::<f64>() / h.len() as f64;

            let khh = &k * &h;
            let state_update = k * &h + (kh - SMatrix::<f64, 24, 24>::identity()) * dx_new;

            // khh is too big in x,y, k is really small overall
            // maybe a specific matmul only affects x,y because rest is small
            //
            // if i == -1 {
            //     dbg!(&state_update, khh, h_avg, k_avg);
            // }
            // dbg!(khh);

            // apply state update towards convergence
            self.state = box_plus(&self.state, &state_update);

            converge = state_update
                .column_iter()
                .all(|x| x.iter().all(|y| y.abs() <= CONVERGE_EPSILON));

            // check if all converged, go through all and print with index which one is not converged
            // converge = true;
            // for x in state_update.column_iter() {
            //     for (idx, y) in x.iter().enumerate() {
            //         if y.abs() > CONVERGE_EPSILON {
            //             dbg!(state_index_to_reason(idx), &self.cov.diagonal(), t);
            //             converge = false;
            //             break;
            //         }
            //     }
            // }

            if converge {
                t += 1;
            }

            // dbg!(t, i);

            // never converged
            if t == 0 {
                // if we are in the last iteration, we give a last change for the model to converge
                if i == max_iter - 2 {
                    converge = true;
                }
            } else {
                // TODO still testing if t >= 1 or t > 1 better. t > 1 seems to be way slower with same results
                // reduce covariance of our state after some converged iterations using the certainty of the last iteration
                if t >= 1 || i == max_iter - 1 {
                    self.cov = (SMatrix::<f64, 24, 24>::identity() - kh) * self.cov;
                    // dbg!(&self.cov.diagonal(), i);
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    /// Testing box_minus because the Rust SO3 API differs from C++ with Eigen.
    /// We need to make sure that log difference yields the same result given the same input.
    fn test_box_minus() {
        let x1 = StateIKFom {
            pos: Vector3::new(-0.00033805, -0.000889051, 0.00938266),
            rot: SO3::try_from_mat(&Matrix3::new(
                1.0,
                -0.000811369,
                -0.000536165,
                0.000809094,
                0.999991,
                -0.00422844,
                0.000539591,
                0.004228,
                0.999991,
            ))
            .unwrap(),
            offset_r_l_i: SO3::try_from_mat(&Matrix3::new(
                1.0,
                -3.58198e-08,
                -1.55174e-08,
                3.58198e-08,
                1.0,
                -2.29495e-09,
                1.55174e-08,
                2.29495e-09,
                1.0,
            ))
            .unwrap(),
            offset_t_l_i: Vector3::new(-2.29524e-09, -8.60487e-09, -0.0999999),
            vel: Vector3::new(0.00100574, -0.00287086, -0.0415167),
            bg: Vector3::new(-0.00306821, -0.0160729, 0.0246286),
            ba: Vector3::new(9.91917e-10, 3.81246e-09, -5.09075e-08),
            grav: Vector3::new(-0.344021, -0.210795, -9.8017),
        };

        let x2 = StateIKFom {
            pos: Vector3::new(-3.91725e-05, 3.46329e-05, -0.00277273),
            rot: SO3::try_from_mat(&Matrix3::new(
                0.999996,
                0.0027701,
                0.00101921,
                -0.00276596,
                0.999988,
                -0.0040398,
                -0.00103039,
                0.00403696,
                0.999991,
            ))
            .unwrap(),
            offset_r_l_i: SO3::try_from_mat(&Matrix3::new(
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            ))
            .unwrap(),
            offset_t_l_i: Vector3::new(0.0, 0.0, -0.1),
            vel: Vector3::new(0.00257744, -0.00273267, -0.0427957),
            bg: Vector3::new(-0.00306821, -0.0160729, 0.0246287),
            ba: Vector3::new(0.0, 0.0, 0.0),
            grav: Vector3::new(-0.344021, -0.210795, -9.8017),
        };
        let res = super::box_minus(&x1, &x2);

        let expected = SMatrix::<f64, 24, 1>::from_vec(vec![
            -0.000298878,
            -0.000923684,
            0.0121554,
            0.000190165,
            -0.00155519,
            0.00358152,
            2.29495e-09,
            -1.55174e-08,
            3.58198e-08,
            -2.29524e-09,
            -8.60487e-09,
            1.20337e-07,
            -0.00157169,
            -0.00013819,
            0.00127899,
            -1.55543e-09,
            1.55399e-08,
            -3.62603e-08,
            9.91917e-10,
            3.81246e-09,
            -5.09075e-08,
            -9.85312e-12,
            -3.8593e-11,
            5.09022e-10,
        ]);

        for (idx, x) in res.iter().enumerate() {
            assert!((x - expected[idx]).abs() < 1e-7);
        }
    }
}

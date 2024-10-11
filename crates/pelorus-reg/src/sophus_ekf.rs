/// Extended Kalman Filter on the SO(3) manifold.
//use crate::ekf_model::CustomModel;
use nalgebra::*;

type SO3 = sophus_lie::Rotation3<f64, 1>;

const CONVERGE_EPSILON: f64 = 0.001;

#[derive(Clone)]
pub struct StateIKFom {
    pub pos: Vector3<f64>,
    pub rot: SO3,
    pub offset_r_l_i: SO3,
    pub offset_t_l_i: Vector3<f64>,
    pub vel: Vector3<f64>,
    pub bg: Vector3<f64>,
    pub ba: Vector3<f64>,
    pub grav: Vector3<f64>,
}

impl StateIKFom {
    pub fn new(gravity: f64) -> Self {
        let mut res = StateIKFom::default();
        res.grav = Vector3::new(0.0, 0.0, gravity);
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
            grav: Vector3::new(0.0, 0.0, 9.81),
        }
    }
}

pub struct InputIKFom {
    pub acc: Vector3<f64>,
    pub gyro: Vector3<f64>,
}

fn get_f(s: &StateIKFom, inp: &InputIKFom) -> SMatrix<f64, 24, 1> {
    let omega = inp.gyro - s.bg;
    let a_inertial = s.rot.matrix().fixed_view::<3, 3>(0, 0) * (inp.acc - s.ba);

    let mut res = SMatrix::<f64, 24, 1>::zeros();
    res.fixed_rows_mut::<3>(0).copy_from(&s.vel);
    res.fixed_rows_mut::<3>(3).copy_from(&omega);
    res.fixed_rows_mut::<3>(12)
        .copy_from(&(a_inertial + s.grav));

    res
}

pub fn df_dx(s: &StateIKFom, inp: &InputIKFom) -> SMatrix<f64, 24, 24> {
    let mut cov = SMatrix::<f64, 24, 24>::zeros();
    cov.fixed_view_mut::<3, 3>(0, 12)
        .copy_from(&Matrix3::identity());
    let acc = inp.acc - s.ba;

    let r_neg = -s.rot.matrix().fixed_view::<3, 3>(0, 0);
    cov.fixed_view_mut::<3, 3>(12, 3)
        .copy_from(&(r_neg * SO3::hat(&acc)));
    cov.fixed_view_mut::<3, 3>(12, 18).copy_from(&r_neg);

    cov.fixed_view_mut::<3, 3>(12, 21)
        .copy_from(&Matrix3::identity());
    cov.fixed_view_mut::<3, 3>(3, 15)
        .copy_from(&Matrix3::identity());

    return cov;
}

pub fn df_dw(s: &StateIKFom, _inp: &InputIKFom) -> SMatrix<f64, 24, 12> {
    let mut cov = SMatrix::<f64, 24, 12>::zeros();
    cov.fixed_view_mut::<3, 3>(12, 3)
        .copy_from(&(-s.rot.matrix().fixed_view::<3, 3>(0, 0)));
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

fn box_plus(x: &StateIKFom, f: SMatrix<f64, 24, 1>) -> StateIKFom {
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

/*
TODO test if
Sophus::SO3(x2.rot.matrix().transpose() * x1.rot.matrix()).log();
and
x_r.fixed_rows_mut::<3>(3).copy_from(&(SO3::log(&x2.rot.inverse().group_mul(&x1.rot))));
do the same thing
*/
fn box_minus(x1: &StateIKFom, x2: &StateIKFom) -> SMatrix<f64, 24, 1> {
    let mut x_r = SMatrix::<f64, 24, 1>::zeros();

    x_r.fixed_rows_mut::<3>(0).copy_from(&(x1.pos - x2.pos));
    x_r.fixed_rows_mut::<3>(3)
        .copy_from(&(SO3::log(&x2.rot.inverse().group_mul(&x1.rot))));
    x_r.fixed_rows_mut::<3>(6)
        .copy_from(&(SO3::log(&x2.offset_r_l_i.inverse().group_mul(&x1.offset_r_l_i))));

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

        self.state = box_plus(&self.state, f * dt);
        f_x = SMatrix::<f64, 24, 24>::identity() + f_x * dt;
        let dt_fw = dt * f_w;
        self.cov = f_x * self.cov * f_x.transpose() + dt_fw * q * dt_fw.transpose();
    }

    pub fn update(
        &mut self,
        r: f64,
        max_iter: i8,
        extrinsic_estimation: bool,
        mut model: impl FnMut(&StateIKFom, &mut bool, bool, &mut DVector<f64>, &mut DMatrix<f64>, bool),
    ) {
        let mut converge = true;
        let mut h = DVector::<f64>::default();
        let mut h_x = DMatrix::<f64>::default();
        let mut t = 0;
        let state_propagated = self.state.clone();

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
                break;
            }

            let dx_new = box_minus(&self.state, &state_propagated);

            let hessian = &h_x;
            let hessian_t = &hessian.transpose();
            let hth = {
                let mut hth = SMatrix::<f64, 24, 24>::zeros();
                hth.fixed_view_mut::<12, 12>(0, 0)
                    .copy_from(&(hessian_t * hessian));
                hth
            };

            let k_front = (hth / r + self.cov.try_inverse().unwrap())
                .try_inverse()
                .unwrap();

            let k = k_front.fixed_view::<24, 12>(0, 0) * hessian_t / r;
            let kh = {
                let mut kh = SMatrix::<f64, 24, 24>::zeros();
                kh.fixed_view_mut::<24, 12>(0, 0).copy_from(&(&k * hessian));
                kh
            };
            let dx_ = k * &h + (kh - SMatrix::<f64, 24, 24>::identity()) * dx_new;
            self.state = box_plus(&self.state, dx_);

            converge = true;
            for j in 0..24 {
                if dx_[j].abs() > CONVERGE_EPSILON {
                    converge = false;
                    break;
                }
            }

            if converge {
                t += 1;
            }

            if t == 0 && i == max_iter - 2 {
                converge = true;
            }

            if t > 1 || i == max_iter - 1 {
                self.cov = (SMatrix::<f64, 24, 24>::identity() - kh) * self.cov;
                return;
            }
        }
    }
}

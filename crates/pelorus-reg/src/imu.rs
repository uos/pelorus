use nalgebra::*;

use crate::sophus_ekf::Esekf;

const MAX_INI_COUNT: usize = 10;
const COV_IMU_RAW_DEFAULT: f64 = 0.0001;
const COV_IMU_BIAS_DEFAULT: f64 = 0.00001;

pub struct ImuProcessing {
    ekf_inited: bool,
    last_imu: crate::ImuMsg,
    init_iter_num: usize,
    cov_acc: Vector3<f64>,
    cov_acc_bias: Vector3<f64>,
    cov_gyr: Vector3<f64>,
    cov_gyr_bias: Vector3<f64>,
    imu_need_init: bool,
    mean_acc: Vector3<f64>,
    mean_gyr: Vector3<f64>,
    first_frame: bool,
    lidar_t_wrt_imu: Vector3<f64>,
    lidar_r_wrt_imu: Matrix3<f64>,
    pub ekf_cov_gyr_bias: Matrix3<f64>,
    pub ekf_cov_acc_bias: Matrix3<f64>,
    pub ekf_cov_acc: Matrix3<f64>,
    pub ekf_cov_gyr: Matrix3<f64>,
}

impl Default for ImuProcessing {
    fn default() -> Self {
        Self {
            ekf_inited: false,
            imu_need_init: true,
            init_iter_num: 1,
            cov_acc: Vector3::new(
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
            ),
            cov_gyr: Vector3::new(
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
            ),
            last_imu: crate::ImuMsg::default(),
            mean_acc: Vector3::zeros(),
            mean_gyr: Vector3::zeros(),
            first_frame: true,
            lidar_t_wrt_imu: Vector3::zeros(),
            lidar_r_wrt_imu: Matrix3::identity(),
            ekf_cov_gyr_bias: Matrix3::zeros(),
            ekf_cov_acc_bias: Matrix3::zeros(),
            ekf_cov_acc: Matrix3::zeros(),
            ekf_cov_gyr: Matrix3::zeros(),
            cov_acc_bias: Vector3::new(
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
            ),
            cov_gyr_bias: Vector3::new(
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
            ),
        }
    }
}

#[derive(Clone, Debug, Copy)]
pub struct ImuParameters {
    pub cov_acc: Vector3<f64>,
    pub cov_gyr: Vector3<f64>,
    pub cov_gyr_bias: Vector3<f64>,
    pub cov_acc_bias: Vector3<f64>,
}

impl Default for ImuParameters {
    fn default() -> Self {
        Self {
            cov_acc: Vector3::new(
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
            ),
            cov_gyr: Vector3::new(
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
                COV_IMU_RAW_DEFAULT,
            ),
            cov_gyr_bias: Vector3::new(
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
            ),
            cov_acc_bias: Vector3::new(
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
                COV_IMU_BIAS_DEFAULT,
            ),
        }
    }
}

impl ImuProcessing {
    pub fn new(params: &ImuParameters, lidar_imu_extr: &Isometry3<f64>) -> Self {
        let mut s = Self::default();

        s.ekf_cov_acc[(0, 0)] = params.cov_acc[0];
        s.ekf_cov_acc[(1, 1)] = params.cov_acc[1];
        s.ekf_cov_acc[(2, 2)] = params.cov_acc[2];

        s.ekf_cov_gyr[(0, 0)] = params.cov_gyr[0];
        s.ekf_cov_gyr[(1, 1)] = params.cov_gyr[1];
        s.ekf_cov_gyr[(2, 2)] = params.cov_gyr[2];

        s.ekf_cov_gyr_bias[(0, 0)] = params.cov_gyr_bias[0];
        s.ekf_cov_gyr_bias[(1, 1)] = params.cov_gyr_bias[1];
        s.ekf_cov_gyr_bias[(2, 2)] = params.cov_gyr_bias[2];

        s.ekf_cov_acc_bias[(0, 0)] = params.cov_acc_bias[0];
        s.ekf_cov_acc_bias[(1, 1)] = params.cov_acc_bias[1];
        s.ekf_cov_acc_bias[(2, 2)] = params.cov_acc_bias[2];

        s.cov_acc = params.cov_acc;
        s.cov_gyr = params.cov_gyr;
        s.cov_gyr_bias = params.cov_gyr_bias;
        s.cov_acc_bias = params.cov_acc_bias;

        s.lidar_r_wrt_imu = *lidar_imu_extr.rotation.to_rotation_matrix().matrix();
        s.lidar_t_wrt_imu = lidar_imu_extr.translation.vector;

        s
    }

    pub fn reset(&mut self) {
        self.mean_acc = Vector3::new(0.0, 0.0, -1.0);
        self.mean_gyr = Vector3::zeros();
        self.imu_need_init = true;
        self.init_iter_num = 1;
        self.last_imu = crate::ImuMsg::default();
    }

    /*pub fn set_extrinsic(&mut self, t: &SMatrix<f64, 4, 4>) {
        self.lidar_t_wrt_imu = Vector3::new(t[(0, 3)], t[(1, 3)], t[(2, 3)]);
        self.lidar_r_wrt_imu = Matrix3::new(
            t[(0, 0)],
            t[(0, 1)],
            t[(0, 2)],
            t[(1, 0)],
            t[(1, 1)],
            t[(1, 2)],
            t[(2, 0)],
            t[(2, 1)],
            t[(2, 2)],
        );
    }*/

    pub fn initialize(
        &mut self,
        msgs: &[crate::ImuMsg],
        ekf: &mut crate::sophus_ekf::Esekf,
    ) -> Vector3<f64> {
        if !self.imu_need_init {
            self.ekf_inited = true; // TODO maybe remove, since imu_need_init already does the same thing
            return self.mean_acc;
        }

        let mut n = 0.0;
        if self.first_frame {
            self.reset();
            n = 1.0;
            self.first_frame = false;
            self.mean_acc = msgs[0].linear_acceleration;
            self.mean_gyr = msgs[0].angular_velocity;
        }

        for msg in msgs {
            let cur_acc = msg.linear_acceleration;
            let cur_gyr = msg.angular_velocity;

            self.mean_acc += (cur_acc - self.mean_acc) / (n as f64);
            self.mean_gyr += (cur_gyr - self.mean_gyr) / (n as f64);

            self.cov_acc = self.cov_acc * (n - 1.0) / n
                + (cur_acc - self.mean_acc).component_mul(&(cur_acc - self.mean_acc)) * (n - 1.0)
                    / (n * n);
            self.cov_gyr = self.cov_gyr * (n - 1.0) / n
                + (cur_gyr - self.mean_gyr).component_mul(&(cur_gyr - self.mean_gyr)) * (n - 1.0)
                    / (n * n);

            n += 1.0;
        }

        ekf.state.grav = -self.mean_acc / self.mean_acc.norm() * 9.81; // TODO PARAM
        ekf.state.bg = self.mean_gyr;
        ekf.state.offset_t_l_i = self.lidar_t_wrt_imu;
        ekf.state.offset_r_l_i =
            sophus_lie::Rotation3::try_from_mat(&self.lidar_r_wrt_imu).unwrap();

        const COV1: f64 = 0.00001;
        const COV2: f64 = 0.0001;
        const COV3: f64 = 0.001;

        ekf.cov.fill_with_identity();
        ekf.cov[(6, 6)] = COV1;
        ekf.cov[(7, 7)] = COV1;
        ekf.cov[(8, 8)] = COV1;
        ekf.cov[(9, 9)] = COV1;
        ekf.cov[(10, 10)] = COV1;
        ekf.cov[(11, 11)] = COV1;

        ekf.cov[(15, 15)] = COV2;
        ekf.cov[(16, 16)] = COV2;
        ekf.cov[(17, 17)] = COV2;

        ekf.cov[(18, 18)] = COV3;
        ekf.cov[(19, 19)] = COV3;
        ekf.cov[(20, 20)] = COV3;

        ekf.cov[(21, 21)] = COV1;
        ekf.cov[(22, 22)] = COV1;

        self.last_imu = msgs[msgs.len() - 1].clone();

        self.imu_need_init = self.init_iter_num <= MAX_INI_COUNT;
        self.ekf_inited = true;

        self.mean_acc
    }

    pub fn predict(
        &self,
        ekf: &mut Esekf,
        acc: [f64; 3],
        gyro: [f64; 3],
        dt: f64,
    ) -> anyhow::Result<crate::ImuEkfPrediction> {
        let inp = crate::sophus_ekf::InputIKFom {
            acc: Vector3::new(acc[0], acc[1], acc[2]),
            gyro: Vector3::new(gyro[0], gyro[1], gyro[2]),
        };

        let q = {
            let mut q = SMatrix::<f64, 12, 12>::zeros();
            q.fixed_view_mut::<3, 3>(0, 0).copy_from(&self.ekf_cov_gyr);
            q.fixed_view_mut::<3, 3>(3, 3).copy_from(&self.ekf_cov_acc);
            q.fixed_view_mut::<3, 3>(6, 6)
                .copy_from(&self.ekf_cov_gyr_bias);
            q.fixed_view_mut::<3, 3>(9, 9)
                .copy_from(&self.ekf_cov_acc_bias);

            q
        };

        dbg!(q);

        if !self.ekf_inited {
            anyhow::bail!("EKF not initialized");
        }

        ekf.predict(dt, &q, &inp);

        Ok(crate::ImuEkfPrediction {
            pose: Isometry3::from_parts(
                Translation3::from(ekf.state.pos),
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                    &ekf.state.rot.matrix(),
                )),
            ),
            linear_velocity: ekf.state.vel,
            lidar_imu_extrinsic: Isometry3::from_parts(
                Translation3::from(ekf.state.offset_t_l_i),
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                    &ekf.state.offset_r_l_i.matrix(),
                )),
            ),
        })
    }
}

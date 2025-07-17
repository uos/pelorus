#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

impl From<[[f64; 3usize]; 3usize]> for Matrix3 {
    fn from(data: [[f64; 3usize]; 3usize]) -> Self {
        Self { data }
    }
}

impl From<[f64; 9usize]> for Matrix3 {
    fn from(data: [f64; 9usize]) -> Self {
        Self {
            data: [
                [data[0], data[1], data[2]],
                [data[3], data[4], data[5]],
                [data[6], data[7], data[8]],
            ],
        }
    }
}

impl From<nalgebra::Matrix3<f64>> for Matrix3 {
    fn from(matrix: nalgebra::Matrix3<f64>) -> Self {
        Self {
            data: [
                [matrix[(0, 0)], matrix[(0, 1)], matrix[(0, 2)]],
                [matrix[(1, 0)], matrix[(1, 1)], matrix[(1, 2)]],
                [matrix[(2, 0)], matrix[(2, 1)], matrix[(2, 2)]],
            ],
        }
    }
}

impl From<Matrix3> for nalgebra::Matrix3<f64> {
    fn from(matrix: Matrix3) -> Self {
        nalgebra::Matrix3::new(
            matrix.data[0][0],
            matrix.data[0][1],
            matrix.data[0][2],
            matrix.data[1][0],
            matrix.data[1][1],
            matrix.data[1][2],
            matrix.data[2][0],
            matrix.data[2][1],
            matrix.data[2][2],
        )
    }
}

impl From<Vector3> for nalgebra::Vector3<f64> {
    fn from(vec: Vector3) -> Self {
        nalgebra::Vector3::new(vec.x, vec.y, vec.z)
    }
}

#[derive(Debug, Clone)]
pub struct ImuEkfPredictionNa {
    pub pose: nalgebra::Isometry3<f64>,
    pub linear_velocity: nalgebra::Vector3<f64>,
    pub lidar_imu_extrinsic_rotation: nalgebra::Matrix3<f64>,
    pub lidar_imu_extrinsic_translation: nalgebra::Vector3<f64>,
}

impl ImuEkfPredictionNa {
    pub fn new(
        pose: nalgebra::Isometry3<f64>,
        linear_velocity: nalgebra::Vector3<f64>,
        lidar_imu_extrinsic_rotation: nalgebra::Matrix3<f64>,
        lidar_imu_extrinsic_translation: nalgebra::Vector3<f64>,
    ) -> Self {
        Self {
            pose,
            linear_velocity,
            lidar_imu_extrinsic_rotation,
            lidar_imu_extrinsic_translation,
        }
    }

    pub fn into_jlio(self) -> ImuEkfPrediction {
        ImuEkfPrediction {
            pose: Pose {
                position: self.pose.translation.into(),
                orientation: self.pose.rotation.into(),
            },
            linear_velocity: self.linear_velocity.into(),
            lidar_imu_extrinsic_rotation: self.lidar_imu_extrinsic_rotation.into(),
            lidar_imu_extrinsic_translation: self.lidar_imu_extrinsic_translation.into(),
        }
    }

    pub fn from_jlio(prediction: ImuEkfPrediction) -> Self {
        Self {
            pose: nalgebra::Isometry3::from_parts(
                prediction.pose.position.into(),
                prediction.pose.orientation.into(),
            ),
            linear_velocity: prediction.linear_velocity.into(),
            lidar_imu_extrinsic_rotation: prediction.lidar_imu_extrinsic_rotation.into(),
            lidar_imu_extrinsic_translation: prediction.lidar_imu_extrinsic_translation.into(),
        }
    }
}

impl From<ImuEkfPredictionNa> for ImuEkfPrediction {
    fn from(prediction: ImuEkfPredictionNa) -> Self {
        prediction.into_jlio()
    }
}

impl From<ImuEkfPrediction> for ImuEkfPredictionNa {
    fn from(prediction: ImuEkfPrediction) -> Self {
        ImuEkfPredictionNa::from_jlio(prediction)
    }
}

impl ImuEkfPrediction {
    pub fn new(
        pose: Pose,
        linear_velocity: Vector3,
        lidar_imu_extrinsic_rotation: Matrix3,
        lidar_imu_extrinsic_translation: Vector3,
    ) -> Self {
        Self {
            pose,
            linear_velocity,
            lidar_imu_extrinsic_rotation,
            lidar_imu_extrinsic_translation,
        }
    }

    pub fn from_nalgebra(
        pose: nalgebra::Isometry3<f64>,
        linear_velocity: nalgebra::Vector3<f64>,
        lidar_imu_extrinsic_rotation: nalgebra::Matrix3<f64>,
        lidar_imu_extrinsic_translation: nalgebra::Vector3<f64>,
    ) -> Self {
        Self {
            pose: Pose {
                position: pose.translation.into(),
                orientation: pose.rotation.into(),
            },
            linear_velocity: linear_velocity.into(),
            lidar_imu_extrinsic_rotation: lidar_imu_extrinsic_rotation.into(),
            lidar_imu_extrinsic_translation: lidar_imu_extrinsic_translation.into(),
        }
    }

    pub fn into_nalgebra(self) -> ImuEkfPredictionNa {
        ImuEkfPredictionNa {
            pose: nalgebra::Isometry3::from_parts(
                self.pose.position.into(),
                self.pose.orientation.into(),
            ),
            linear_velocity: self.linear_velocity.into(),
            lidar_imu_extrinsic_rotation: self.lidar_imu_extrinsic_rotation.into(),
            lidar_imu_extrinsic_translation: self.lidar_imu_extrinsic_translation.into(),
        }
    }
}

impl Quaternion {
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }
}

impl Default for Quaternion {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }
}

impl From<nalgebra::UnitQuaternion<f64>> for Quaternion {
    fn from(quat: nalgebra::UnitQuaternion<f64>) -> Self {
        Self {
            x: quat.coords.x,
            y: quat.coords.y,
            z: quat.coords.z,
            w: quat.coords.w,
        }
    }
}

impl From<Quaternion> for nalgebra::UnitQuaternion<f64> {
    fn from(quat: Quaternion) -> Self {
        nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
            quat.w, quat.x, quat.y, quat.z,
        ))
    }
}

impl From<nalgebra::Vector3<f64>> for Vector3 {
    fn from(vec: nalgebra::Vector3<f64>) -> Self {
        Self {
            x: vec.x,
            y: vec.y,
            z: vec.z,
        }
    }
}

impl From<&nalgebra::Vector3<f64>> for Vector3 {
    fn from(vec: &nalgebra::Vector3<f64>) -> Self {
        Self {
            x: vec.x,
            y: vec.y,
            z: vec.z,
        }
    }
}

impl From<nalgebra::Translation3<f64>> for Vector3 {
    fn from(vec: nalgebra::Translation3<f64>) -> Self {
        Self {
            x: vec.vector.x,
            y: vec.vector.y,
            z: vec.vector.z,
        }
    }
}

impl From<&nalgebra::Translation3<f64>> for Vector3 {
    fn from(vec: &nalgebra::Translation3<f64>) -> Self {
        Self {
            x: vec.vector.x,
            y: vec.vector.y,
            z: vec.vector.z,
        }
    }
}

impl From<Vector3> for nalgebra::Translation3<f64> {
    fn from(vec: Vector3) -> Self {
        nalgebra::Translation3::new(vec.x, vec.y, vec.z)
    }
}

impl Vector3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

impl Default for Vector3 {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

impl ImuMsg {
    pub fn new(
        timestamp: f64,
        orientation: Quaternion,
        orientation_cov: [f64; 9],
        angular_velocity: Vector3,
        angular_velocity_cov: [f64; 9],
        linear_acceleration: Vector3,
        linear_acceleration_cov: [f64; 9],
    ) -> Self {
        Self {
            timestamp_sec: timestamp,
            orientation,
            orientation_covariance: orientation_cov,
            angular_velocity,
            angular_velocity_covariance: angular_velocity_cov,
            linear_acceleration,
            linear_acceleration_covariance: linear_acceleration_cov,
        }
    }

    pub fn new_with_cov_diag(
        timestamp: f64,
        orientation: Quaternion,
        orientation_cov: Vector3,
        angular_velocity: Vector3,
        angular_velocity_cov: Vector3,
        linear_acceleration: Vector3,
        linear_acceleration_cov: Vector3,
    ) -> Self {
        Self {
            timestamp_sec: timestamp,
            orientation,
            orientation_covariance: [
                orientation_cov.x,
                0.0,
                0.0,
                0.0,
                orientation_cov.y,
                0.0,
                0.0,
                0.0,
                orientation_cov.z,
            ],
            angular_velocity,
            angular_velocity_covariance: [
                angular_velocity_cov.x,
                0.0,
                0.0,
                0.0,
                angular_velocity_cov.y,
                0.0,
                0.0,
                0.0,
                angular_velocity_cov.z,
            ],
            linear_acceleration,
            linear_acceleration_covariance: [
                linear_acceleration_cov.x,
                0.0,
                0.0,
                0.0,
                linear_acceleration_cov.y,
                0.0,
                0.0,
                0.0,
                linear_acceleration_cov.z,
            ],
        }
    }

    pub fn new_with_cov_diag_scalar(
        timestamp: f64,
        orientation: Quaternion,
        orientation_cov: f64,
        angular_velocity: Vector3,
        angular_velocity_cov: f64,
        linear_acceleration: Vector3,
        linear_acceleration_cov: f64,
    ) -> Self {
        Self::new_with_cov_diag(
            timestamp,
            orientation,
            Vector3::new(orientation_cov, orientation_cov, orientation_cov),
            angular_velocity,
            Vector3::new(
                angular_velocity_cov,
                angular_velocity_cov,
                angular_velocity_cov,
            ),
            linear_acceleration,
            Vector3::new(
                linear_acceleration_cov,
                linear_acceleration_cov,
                linear_acceleration_cov,
            ),
        )
    }

    pub fn from_nalgebra(
        timestamp: f64,
        orientation: nalgebra::UnitQuaternion<f64>,
        orientation_cov: nalgebra::Matrix3<f64>,
        angular_velocity: nalgebra::Vector3<f64>,
        angular_velocity_cov: nalgebra::Matrix3<f64>,
        linear_acceleration: nalgebra::Vector3<f64>,
        linear_acceleration_cov: nalgebra::Matrix3<f64>,
    ) -> Self {
        Self {
            timestamp_sec: timestamp,
            orientation: orientation.into(),
            orientation_covariance: [
                orientation_cov[(0, 0)],
                orientation_cov[(0, 1)],
                orientation_cov[(0, 2)],
                orientation_cov[(1, 0)],
                orientation_cov[(1, 1)],
                orientation_cov[(1, 2)],
                orientation_cov[(2, 0)],
                orientation_cov[(2, 1)],
                orientation_cov[(2, 2)],
            ],
            angular_velocity: angular_velocity.into(),
            angular_velocity_covariance: [
                angular_velocity_cov[(0, 0)],
                angular_velocity_cov[(0, 1)],
                angular_velocity_cov[(0, 2)],
                angular_velocity_cov[(1, 0)],
                angular_velocity_cov[(1, 1)],
                angular_velocity_cov[(1, 2)],
                angular_velocity_cov[(2, 0)],
                angular_velocity_cov[(2, 1)],
                angular_velocity_cov[(2, 2)],
            ],
            linear_acceleration: linear_acceleration.into(),
            linear_acceleration_covariance: [
                linear_acceleration_cov[(0, 0)],
                linear_acceleration_cov[(0, 1)],
                linear_acceleration_cov[(0, 2)],
                linear_acceleration_cov[(1, 0)],
                linear_acceleration_cov[(1, 1)],
                linear_acceleration_cov[(1, 2)],
                linear_acceleration_cov[(2, 0)],
                linear_acceleration_cov[(2, 1)],
                linear_acceleration_cov[(2, 2)],
            ],
        }
    }
}

impl From<nalgebra::Isometry3<f64>> for Pose {
    fn from(isometry: nalgebra::Isometry3<f64>) -> Self {
        Self {
            position: isometry.translation.into(),
            orientation: isometry.rotation.into(),
        }
    }
}

impl From<Pose> for nalgebra::Isometry3<f64> {
    fn from(pose: Pose) -> Self {
        nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::from(pose.position),
            nalgebra::UnitQuaternion::from(pose.orientation),
        )
    }
}

impl Default for Parameter {
    fn default() -> Self {
        Self {
            imu_acc_cov: 0.01,
            imu_acc_bias_cov: 0.0001,
            imu_gyro_cov: 0.01,
            imu_gyro_bias_cov: 0.0001,
        }
    }
}

impl Parameter {
    pub fn new(
        imu_acc_cov: f64,
        imu_acc_bias_cov: f64,
        imu_gyro_cov: f64,
        imu_gyro_bias_cov: f64,
    ) -> Self {
        Self {
            imu_acc_cov,
            imu_acc_bias_cov,
            imu_gyro_cov,
            imu_gyro_bias_cov,
        }
    }
}

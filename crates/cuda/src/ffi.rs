#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use core::panic;

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

#[derive(Debug, Copy, Clone, Default)]
pub enum NeighborType {
    Center,
    Neighbor6,
    #[default]
    Neighbor18,
    Neighbor26,
}

impl From<NearbyType> for NeighborType {
    fn from(value: NearbyType) -> Self {
        match value {
            NearbyType_Center => Self::Center,
            NearbyType_Nearby18 => Self::Neighbor18,
            NearbyType_Nearby26 => Self::Neighbor26,
            NearbyType_Nearby6 => Self::Neighbor6,
            _ => {
                panic!("unknown nearby type")
            }
        }
    }
}

impl From<NeighborType> for NearbyType {
    fn from(value: NeighborType) -> Self {
        match value {
            NeighborType::Center => NearbyType_Center,
            NeighborType::Neighbor6 => NearbyType_Nearby6,
            NeighborType::Neighbor18 => NearbyType_Nearby18,
            NeighborType::Neighbor26 => NearbyType_Nearby26,
        }
    }
}

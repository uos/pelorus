#[derive(Debug, Clone, Copy, Default)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

impl From<nalgebra::Vector3<f64>> for Vector3 {
    fn from(v: nalgebra::Vector3<f64>) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl From<&nalgebra::Vector3<f64>> for Vector3 {
    fn from(v: &nalgebra::Vector3<f64>) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl From<nalgebra::Translation3<f64>> for Vector3 {
    fn from(t: nalgebra::Translation3<f64>) -> Self {
        Self {
            x: t.vector.x,
            y: t.vector.y,
            z: t.vector.z,
        }
    }
}

impl From<&nalgebra::Translation3<f64>> for Vector3 {
    fn from(t: &nalgebra::Translation3<f64>) -> Self {
        Self {
            x: t.vector.x,
            y: t.vector.y,
            z: t.vector.z,
        }
    }
}

impl From<Vector3> for nalgebra::Vector3<f64> {
    fn from(v: Vector3) -> Self {
        nalgebra::Vector3::new(v.x, v.y, v.z)
    }
}

impl From<Vector3> for nalgebra::Translation3<f64> {
    fn from(v: Vector3) -> Self {
        nalgebra::Translation3::new(v.x, v.y, v.z)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { w, x, y, z }
    }
}

impl Default for Quaternion {
    fn default() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

impl From<nalgebra::UnitQuaternion<f64>> for Quaternion {
    fn from(q: nalgebra::UnitQuaternion<f64>) -> Self {
        Self {
            w: q.coords.w,
            x: q.coords.x,
            y: q.coords.y,
            z: q.coords.z,
        }
    }
}

impl From<Quaternion> for nalgebra::UnitQuaternion<f64> {
    fn from(q: Quaternion) -> Self {
        nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(q.w, q.x, q.y, q.z))
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Matrix3 {
    pub data: [[f64; 3]; 3],
}

impl From<[[f64; 3]; 3]> for Matrix3 {
    fn from(data: [[f64; 3]; 3]) -> Self {
        Self { data }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Pose {
    pub position: Vector3,
    pub orientation: Quaternion,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NeighborType {
    Center,
    Neighbor6,
    #[default]
    Neighbor18,
    Neighbor26,
}

impl NeighborType {
    pub fn as_u32(self) -> u32 {
        match self {
            NeighborType::Center => 0,
            NeighborType::Neighbor6 => 6,
            NeighborType::Neighbor18 => 18,
            NeighborType::Neighbor26 => 26,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub i: f32,
    pub t: f32,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct PointNormal {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub _pad: f32,
}

impl From<PointXYZIT> for Point {
    fn from(p: PointXYZIT) -> Self {
        Self {
            x: p.x,
            y: p.y,
            z: p.z,
            i: p.intensity,
            t: p.timestamp,
        }
    }
}

impl From<Point> for PointXYZIT {
    fn from(p: Point) -> Self {
        PointXYZIT {
            x: p.x,
            y: p.y,
            z: p.z,
            intensity: p.i,
            timestamp: p.t,
        }
    }
}

impl From<PointNormal> for PointXYZINormal {
    fn from(p: PointNormal) -> Self {
        PointXYZINormal {
            x: p.x,
            y: p.y,
            z: p.z,
            intensity: p.intensity,
            normal_x: p.nx,
            normal_y: p.ny,
            normal_z: p.nz,
        }
    }
}
use bytemuck::{Pod, Zeroable};
use pelorus_core::PointXYZIT;
use ros_pointcloud2::points::PointXYZINormal;

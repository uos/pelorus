use nalgebra::{
    Isometry3, Matrix3, Matrix4, Point3, Translation3, UnitQuaternion, Vector3, Vector6,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SO3(pub UnitQuaternion<f64>);

impl Default for SO3 {
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}

impl SO3 {
    #[inline]
    pub fn identity() -> Self {
        Self(UnitQuaternion::identity())
    }

    #[inline]
    pub fn exp(omega: Vector3<f64>) -> Self {
        Self(UnitQuaternion::from_scaled_axis(omega))
    }

    #[inline]
    pub fn log(&self) -> Vector3<f64> {
        self.0.scaled_axis()
    }

    #[inline]
    pub fn matrix(&self) -> Matrix3<f64> {
        self.0.to_rotation_matrix().into_inner()
    }

    #[inline]
    pub fn inverse(&self) -> Self {
        Self(self.0.inverse())
    }

    #[inline]
    pub fn group_mul(&self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    #[inline]
    pub fn try_from_mat(m: &Matrix3<f64>) -> Option<Self> {
        Some(Self(UnitQuaternion::from_matrix(m)))
    }

    #[inline]
    pub fn transform(&self, v: Vector3<f64>) -> Vector3<f64> {
        self.0.transform_vector(&v)
    }

    #[inline]
    pub fn quaternion(&self) -> &UnitQuaternion<f64> {
        &self.0
    }
}

impl From<UnitQuaternion<f64>> for SO3 {
    #[inline]
    fn from(q: UnitQuaternion<f64>) -> Self {
        Self(q)
    }
}

impl From<SO3> for UnitQuaternion<f64> {
    #[inline]
    fn from(rot: SO3) -> Self {
        rot.0
    }
}

impl std::ops::Mul<SO3> for SO3 {
    type Output = SO3;
    #[inline]
    fn mul(self, rhs: SO3) -> SO3 {
        self.group_mul(rhs)
    }
}

impl std::ops::Mul<Vector3<f64>> for SO3 {
    type Output = Vector3<f64>;
    #[inline]
    fn mul(self, rhs: Vector3<f64>) -> Vector3<f64> {
        self.transform(rhs)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SE3(pub Isometry3<f64>);

impl Default for SE3 {
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}

impl SE3 {
    #[inline]
    pub fn identity() -> Self {
        Self(Isometry3::identity())
    }

    #[inline]
    pub fn from_rotation_and_translation(rot: SO3, trans: Vector3<f64>) -> Self {
        Self(Isometry3::from_parts(Translation3::from(trans), rot.0))
    }

    #[inline]
    pub fn exp(xi: Vector6<f64>) -> Self {
        let trans = Vector3::new(xi[0], xi[1], xi[2]);
        let rot = SO3::exp(Vector3::new(xi[3], xi[4], xi[5]));
        Self::from_rotation_and_translation(rot, trans)
    }

    #[inline]
    pub fn inverse(&self) -> Self {
        Self(self.0.inverse())
    }

    #[inline]
    pub fn group_mul(&self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    #[inline]
    pub fn translation(&self) -> Vector3<f64> {
        self.0.translation.vector
    }

    #[inline]
    pub fn rotation(&self) -> SO3 {
        SO3(self.0.rotation)
    }

    #[inline]
    pub fn matrix(&self) -> Matrix4<f64> {
        self.0.to_homogeneous()
    }

    #[inline]
    pub fn transform(&self, p: Vector3<f64>) -> Vector3<f64> {
        self.0.transform_point(&Point3::from(p)).coords
    }

    #[inline]
    pub fn isometry(&self) -> &Isometry3<f64> {
        &self.0
    }
}

impl From<Isometry3<f64>> for SE3 {
    #[inline]
    fn from(iso: Isometry3<f64>) -> Self {
        Self(iso)
    }
}

impl From<SE3> for Isometry3<f64> {
    #[inline]
    fn from(pose: SE3) -> Self {
        pose.0
    }
}

impl std::ops::Mul<SE3> for SE3 {
    type Output = SE3;
    #[inline]
    fn mul(self, rhs: SE3) -> SE3 {
        self.group_mul(rhs)
    }
}

impl std::ops::Mul<Vector3<f64>> for SE3 {
    type Output = Vector3<f64>;
    #[inline]
    fn mul(self, rhs: Vector3<f64>) -> Vector3<f64> {
        self.transform(rhs)
    }
}

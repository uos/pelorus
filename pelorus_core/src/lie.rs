use nalgebra::{
    Isometry3, Matrix3, Matrix4, Point3, Translation3, UnitQuaternion, Vector3, Vector6,
};
use serde::{Deserialize, Serialize};

use crate::Real;

/// A product of unit quaternions leaves the unit sphere by about one rounding
/// step. The filter chains thousands of them per minute, which an `f32` state
/// turns into a visible scale error of every rotation.
#[inline]
fn renormalized(mut q: UnitQuaternion<Real>) -> UnitQuaternion<Real> {
    if crate::REAL_IS_F32 {
        q.renormalize_fast();
    }
    q
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SO3(pub UnitQuaternion<Real>);

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
    pub fn exp(omega: Vector3<Real>) -> Self {
        Self(crate::math::quaternion_from_scaled_axis(omega))
    }

    #[inline]
    pub fn log(&self) -> Vector3<Real> {
        self.0.scaled_axis()
    }

    #[inline]
    pub fn matrix(&self) -> Matrix3<Real> {
        self.0.to_rotation_matrix().into_inner()
    }

    #[inline]
    pub fn inverse(&self) -> Self {
        Self(self.0.inverse())
    }

    #[inline]
    pub fn group_mul(&self, other: Self) -> Self {
        Self(renormalized(self.0 * other.0))
    }

    #[inline]
    pub fn try_from_mat(m: &Matrix3<Real>) -> Option<Self> {
        Some(Self(crate::math::quaternion_from_matrix(m)))
    }

    #[inline]
    pub fn transform(&self, v: Vector3<Real>) -> Vector3<Real> {
        self.0.transform_vector(&v)
    }

    #[inline]
    pub fn quaternion(&self) -> &UnitQuaternion<Real> {
        &self.0
    }
}

impl From<UnitQuaternion<Real>> for SO3 {
    #[inline]
    fn from(q: UnitQuaternion<Real>) -> Self {
        Self(q)
    }
}

impl From<SO3> for UnitQuaternion<Real> {
    #[inline]
    fn from(rot: SO3) -> Self {
        rot.0
    }
}

impl core::ops::Mul<SO3> for SO3 {
    type Output = SO3;
    #[inline]
    fn mul(self, rhs: SO3) -> SO3 {
        self.group_mul(rhs)
    }
}

impl core::ops::Mul<Vector3<Real>> for SO3 {
    type Output = Vector3<Real>;
    #[inline]
    fn mul(self, rhs: Vector3<Real>) -> Vector3<Real> {
        self.transform(rhs)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SE3(pub Isometry3<Real>);

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
    pub fn from_rotation_and_translation(rot: SO3, trans: Vector3<Real>) -> Self {
        Self(Isometry3::from_parts(Translation3::from(trans), rot.0))
    }

    #[inline]
    pub fn exp(xi: Vector6<Real>) -> Self {
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
        let mut product = self.0 * other.0;
        product.rotation = renormalized(product.rotation);
        Self(product)
    }

    #[inline]
    pub fn translation(&self) -> Vector3<Real> {
        self.0.translation.vector
    }

    #[inline]
    pub fn rotation(&self) -> SO3 {
        SO3(self.0.rotation)
    }

    #[inline]
    pub fn matrix(&self) -> Matrix4<Real> {
        self.0.to_homogeneous()
    }

    #[inline]
    pub fn transform(&self, p: Vector3<Real>) -> Vector3<Real> {
        self.0.transform_point(&Point3::from(p)).coords
    }

    #[inline]
    pub fn isometry(&self) -> &Isometry3<Real> {
        &self.0
    }
}

impl From<Isometry3<Real>> for SE3 {
    #[inline]
    fn from(iso: Isometry3<Real>) -> Self {
        Self(iso)
    }
}

impl From<SE3> for Isometry3<Real> {
    #[inline]
    fn from(pose: SE3) -> Self {
        pose.0
    }
}

impl core::ops::Mul<SE3> for SE3 {
    type Output = SE3;
    #[inline]
    fn mul(self, rhs: SE3) -> SE3 {
        self.group_mul(rhs)
    }
}

impl core::ops::Mul<Vector3<Real>> for SE3 {
    type Output = Vector3<Real>;
    #[inline]
    fn mul(self, rhs: Vector3<Real>) -> Vector3<Real> {
        self.transform(rhs)
    }
}

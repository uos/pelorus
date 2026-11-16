//! Sine and cosine in [`Real`], and the rotations nalgebra builds from them.
//!
//! libm's single-precision `sinf` and `cosf` reduce their argument and evaluate
//! their polynomials in `f64`. A core without a double-precision FPU runs those
//! in software, so the `f32` build takes the single-precision versions below.
//! The `f64` build calls the usual functions and computes what nalgebra
//! computes.

use nalgebra::{Matrix3, Rotation3, Unit, UnitQuaternion, Vector3};

use crate::Real;

/// Sine and cosine of an `f32`, computed in `f32` only.
///
/// The argument is reduced by π/4 in three parts (Cody and Waite) and the
/// polynomials are the single-precision minimax ones of Cephes. Both results
/// are within about one unit in the last place for `|x|` below 8192. Larger
/// arguments lose precision in the reduction. Angles in Pelorus stay far
/// below that.
pub fn sin_cos_f32(x: f32) -> (f32, f32) {
    const FOUR_OVER_PI: f32 = 1.273_239_5;
    const DP1: f32 = 0.785_156_25;
    const DP2: f32 = 2.418_756_5e-4;
    const DP3: f32 = 3.774_895e-8;

    if !x.is_finite() {
        return (f32::NAN, f32::NAN);
    }
    let magnitude = x.abs();
    let mut octant = (magnitude * FOUR_OVER_PI) as i32;
    let mut y = octant as f32;
    if octant & 1 == 1 {
        octant += 1;
        y += 1.0;
    }
    octant &= 7;
    let mut sin_negative = x.is_sign_negative();
    let mut cos_negative = false;
    if octant > 3 {
        octant -= 4;
        sin_negative = !sin_negative;
        cos_negative = !cos_negative;
    }
    if octant > 1 {
        cos_negative = !cos_negative;
    }
    let r = ((magnitude - y * DP1) - y * DP2) - y * DP3;
    let z = r * r;
    let sin_r = ((-1.951_529_6e-4 * z + 8.332_161e-3) * z - 1.666_665_5e-1) * z * r + r;
    let cos_r =
        ((2.443_315_7e-5 * z - 1.388_731_6e-3) * z + 4.166_664_6e-2) * z * z - 0.5 * z + 1.0;
    let (sin, cos) = if octant == 1 || octant == 2 {
        (cos_r, sin_r)
    } else {
        (sin_r, cos_r)
    };
    (
        if sin_negative { -sin } else { sin },
        if cos_negative { -cos } else { cos },
    )
}

/// Sine of an `f32` without any `f64` in the `f32` build. The `f64` build
/// calls what `f32::sin` calls, the system's maths with std and libm without.
#[inline]
pub fn sinf(x: f32) -> f32 {
    #[cfg(feature = "f32")]
    {
        sin_cos_f32(x).0
    }
    #[cfg(not(feature = "f32"))]
    {
        nalgebra::ComplexField::sin(x)
    }
}

/// Cosine of an `f32`, like [`sinf`].
#[inline]
pub fn cosf(x: f32) -> f32 {
    #[cfg(feature = "f32")]
    {
        sin_cos_f32(x).1
    }
    #[cfg(not(feature = "f32"))]
    {
        nalgebra::ComplexField::cos(x)
    }
}

/// Sine of `x`.
#[inline]
pub fn sin(x: Real) -> Real {
    #[cfg(feature = "f32")]
    {
        sin_cos_f32(x).0
    }
    #[cfg(not(feature = "f32"))]
    {
        nalgebra::ComplexField::sin(x)
    }
}

/// Cosine of `x`.
#[inline]
pub fn cos(x: Real) -> Real {
    #[cfg(feature = "f32")]
    {
        sin_cos_f32(x).1
    }
    #[cfg(not(feature = "f32"))]
    {
        nalgebra::ComplexField::cos(x)
    }
}

/// Sine and cosine of `x`.
#[inline]
pub fn sin_cos(x: Real) -> (Real, Real) {
    #[cfg(feature = "f32")]
    {
        sin_cos_f32(x)
    }
    #[cfg(not(feature = "f32"))]
    {
        nalgebra::ComplexField::sin_cos(x)
    }
}

/// [`UnitQuaternion::from_scaled_axis`].
pub fn quaternion_from_scaled_axis(axisangle: Vector3<Real>) -> UnitQuaternion<Real> {
    #[cfg(feature = "f32")]
    {
        // `Quaternion::from_imag(axisangle / 2).exp()`, whose scalar part is
        // zero, so its exponential is one.
        let half = axisangle / 2.0;
        let nn = half.norm_squared();
        if nn <= Real::EPSILON * Real::EPSILON {
            return UnitQuaternion::identity();
        }
        let n = nalgebra::ComplexField::sqrt(nn);
        let (sin, cos) = sin_cos(n);
        UnitQuaternion::new_unchecked(nalgebra::Quaternion::from_parts(cos, half * (sin / n)))
    }
    #[cfg(not(feature = "f32"))]
    {
        UnitQuaternion::from_scaled_axis(axisangle)
    }
}

/// [`Rotation3::from_axis_angle`].
pub fn rotation_from_axis_angle(axis: &Unit<Vector3<Real>>, angle: Real) -> Rotation3<Real> {
    #[cfg(feature = "f32")]
    {
        if angle == 0.0 {
            return Rotation3::identity();
        }
        let (ux, uy, uz) = (axis[0], axis[1], axis[2]);
        let (sqx, sqy, sqz) = (ux * ux, uy * uy, uz * uz);
        let (sin, cos) = sin_cos(angle);
        let one_m_cos = 1.0 - cos;
        Rotation3::from_matrix_unchecked(Matrix3::new(
            sqx + (1.0 - sqx) * cos,
            ux * uy * one_m_cos - uz * sin,
            ux * uz * one_m_cos + uy * sin,
            ux * uy * one_m_cos + uz * sin,
            sqy + (1.0 - sqy) * cos,
            uy * uz * one_m_cos - ux * sin,
            ux * uz * one_m_cos - uy * sin,
            uy * uz * one_m_cos + ux * sin,
            sqz + (1.0 - sqz) * cos,
        ))
    }
    #[cfg(not(feature = "f32"))]
    {
        Rotation3::from_axis_angle(axis, angle)
    }
}

/// [`Rotation3::from_scaled_axis`].
pub fn rotation_from_scaled_axis(axisangle: Vector3<Real>) -> Rotation3<Real> {
    #[cfg(feature = "f32")]
    {
        let (axis, angle) = Unit::new_and_get(axisangle);
        rotation_from_axis_angle(&axis, angle)
    }
    #[cfg(not(feature = "f32"))]
    {
        Rotation3::from_scaled_axis(axisangle)
    }
}

/// [`Rotation3::from_matrix`]: the rotation closest to `m`, by the iteration
/// of Müller et al., "A Robust Method to Extract the Rotational Part of
/// Deformations".
pub fn rotation_from_matrix(m: &Matrix3<Real>) -> Rotation3<Real> {
    #[cfg(feature = "f32")]
    {
        let eps = Real::EPSILON;
        let eps_disturbance = nalgebra::ComplexField::sqrt(eps).max(eps * eps);
        let mut perturbation_axes = Vector3::x_axis();
        let mut rot = Matrix3::identity();
        loop {
            let axis = rot.column(0).cross(&m.column(0))
                + rot.column(1).cross(&m.column(1))
                + rot.column(2).cross(&m.column(2));
            let denom = rot.column(0).dot(&m.column(0))
                + rot.column(1).dot(&m.column(1))
                + rot.column(2).dot(&m.column(2));
            let axisangle = axis / (denom.abs() + eps);
            match Unit::try_new_and_get(axisangle, eps) {
                Some((axis, angle)) => {
                    rot = rotation_from_axis_angle(&axis, angle).into_inner() * rot;
                }
                None => {
                    // Leave a maximum of `(m - rot).norm()` by a small turn.
                    let mut perturbed = rot;
                    let norm_squared = (m - rot).norm_squared();
                    let mut new_norm_squared;
                    loop {
                        perturbed *= rotation_from_axis_angle(&perturbation_axes, eps_disturbance)
                            .into_inner();
                        new_norm_squared = (m - perturbed).norm_squared();
                        if (norm_squared - new_norm_squared).abs() > eps {
                            break;
                        }
                    }
                    if norm_squared < new_norm_squared {
                        break;
                    }
                    perturbation_axes = Unit::new_unchecked(perturbation_axes.yzx());
                    rot = perturbed;
                }
            }
        }
        Rotation3::from_matrix_unchecked(rot)
    }
    #[cfg(not(feature = "f32"))]
    {
        Rotation3::from_matrix(m)
    }
}

/// [`UnitQuaternion::from_matrix`].
pub fn quaternion_from_matrix(m: &Matrix3<Real>) -> UnitQuaternion<Real> {
    UnitQuaternion::from_rotation_matrix(&rotation_from_matrix(m))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distance in units of the last place between `a` and the `f32` nearest
    /// to `exact`.
    fn ulps(a: f32, exact: f64) -> u32 {
        let b = exact as f32;
        if a == b {
            return 0;
        }
        let (a, b) = (a.to_bits() as i64, b.to_bits() as i64);
        (a - b).unsigned_abs() as u32
    }

    #[test]
    fn single_precision_sine_and_cosine_are_within_two_ulps() {
        let mut worst = (0, 0);
        for i in -200_000..=200_000 {
            let x = i as f32 * 1.0e-3;
            let (sin, cos) = sin_cos_f32(x);
            let (exact_sin, exact_cos) = (f64::from(x).sin(), f64::from(x).cos());
            // Near a zero the relative error grows without bound, so compare
            // there against the spacing of the larger value.
            let tolerance = |exact: f64| if exact.abs() < 1e-3 { 1e-9 } else { 0.0 };
            if (f64::from(sin) - exact_sin).abs() > tolerance(exact_sin) {
                worst.0 = worst.0.max(ulps(sin, exact_sin));
            }
            if (f64::from(cos) - exact_cos).abs() > tolerance(exact_cos) {
                worst.1 = worst.1.max(ulps(cos, exact_cos));
            }
        }
        assert!(
            worst.0 <= 2 && worst.1 <= 2,
            "sine {} ulps, cosine {} ulps",
            worst.0,
            worst.1
        );
    }

    #[test]
    fn special_values_propagate() {
        assert!(sin_cos_f32(f32::NAN).0.is_nan());
        assert!(sin_cos_f32(f32::INFINITY).1.is_nan());
        assert_eq!(sin_cos_f32(0.0), (0.0, 1.0));
        assert_eq!(sin_cos_f32(-0.0).0.to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn rotations_match_nalgebra() {
        // `angle_to` takes an arccosine near one, which an `f32` resolves only
        // to about 3e-4 rad, so compare the entries instead.
        let axisangle = Vector3::new(0.3, -1.1, 0.7);
        let ours = quaternion_from_scaled_axis(axisangle);
        let theirs = UnitQuaternion::from_scaled_axis(axisangle);
        assert!((ours.coords - theirs.coords).norm() < 1e-6);

        let ours = rotation_from_scaled_axis(axisangle);
        let theirs = Rotation3::from_scaled_axis(axisangle);
        assert!((ours.matrix() - theirs.matrix()).norm() < 1e-6);
        assert_eq!(
            rotation_from_scaled_axis(Vector3::zeros()),
            Rotation3::identity()
        );

        let skewed = Rotation3::from_euler_angles(0.2, -0.4, 1.3).into_inner()
            + Matrix3::new(1e-3, 0.0, 2e-3, 0.0, -1e-3, 0.0, 1e-3, 0.0, 0.0);
        let ours = rotation_from_matrix(&skewed);
        let theirs = Rotation3::from_matrix(&skewed);
        assert!((ours.matrix() - theirs.matrix()).norm() < 1e-5);
    }
}

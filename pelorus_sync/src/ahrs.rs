//! The attitude and heading reference system of x-io's Fusion library, in the
//! form of the `imu-fusion` crate, for the case Pelorus uses: north-west-up,
//! the default settings and no magnetometer.
//!
//! The crate's sine and cosine come from libm, whose single-precision versions
//! compute in `f64`. The `f32` build takes [`pelorus_core::math`]'s instead and
//! stays in `f32`. The `f64` build calls libm, so it computes what the crate
//! computes, bit for bit.
//!
//! Ported from imu-fusion 0.2.6 (<https://github.com/mark2b/imu-fusion-rs>):
//!
//! MIT License
//!
//! Copyright (c) 2023 Mark Berner
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.

use pelorus_core::REAL_IS_F32;
use pelorus_core::math::sin_cos_f32;

fn sinf(x: f32) -> f32 {
    if REAL_IS_F32 {
        sin_cos_f32(x).0
    } else {
        libm::sinf(x)
    }
}

fn cosf(x: f32) -> f32 {
    if REAL_IS_F32 {
        sin_cos_f32(x).1
    } else {
        libm::cosf(x)
    }
}

/// Gain while the filter initializes.
const INITIAL_GAIN: f32 = 10.0;
/// Length of the initialization in seconds.
const INITIALISATION_PERIOD: f32 = 3.0;
/// The crate's default settings. Its constructor does not run them through
/// `update_settings`, so the gyroscope range stays zero and the rejection
/// stays in degrees.
const GAIN: f32 = 0.5;
const GYROSCOPE_RANGE: f32 = 0.0;
const ACCELERATION_REJECTION: f32 = 90.0;
const RECOVERY_TRIGGER_PERIOD: i32 = 0;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vector {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn is_zero(&self) -> bool {
        self.x == 0.0 && self.y == 0.0 && self.z == 0.0
    }

    fn cross(&self, rhs: &Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    fn dot(&self, rhs: &Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// The squared length. The crate names it `magnitude` and compares it with
    /// the rejection threshold as such.
    fn magnitude(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    fn normalize(&self) -> Self {
        self.scale(fast_inverse_sqrt(self.magnitude()))
    }

    fn scale(&self, factor: f32) -> Self {
        Self::new(self.x * factor, self.y * factor, self.z * factor)
    }

    fn add(&self, rhs: &Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    fn sub(&self, rhs: &Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quaternion {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Quaternion {
    const IDENTITY: Self = Self {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    fn add(&self, rhs: &Self) -> Self {
        Self {
            w: self.w + rhs.w,
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }

    fn mul(&self, rhs: &Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }

    /// The product with the pure quaternion `(0, rhs)`.
    fn mul_vector(&self, rhs: &Vector) -> Self {
        Self {
            w: -self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x,
        }
    }

    fn normalize(&self) -> Self {
        let factor = fast_inverse_sqrt(
            self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z,
        );
        Self {
            w: self.w * factor,
            x: self.x * factor,
            y: self.y * factor,
            z: self.z * factor,
        }
    }
}

/// The filter state.
pub struct Ahrs {
    pub quaternion: Quaternion,
    acc: Vector,
    initialising: bool,
    ramped_gain: f32,
    ramped_gain_step: f32,
    angular_rate_recovery: bool,
    half_accelerometer_feedback: Vector,
    accelerometer_ignored: bool,
    acceleration_recovery_trigger: i32,
    acceleration_recovery_timeout: i32,
}

impl Default for Ahrs {
    fn default() -> Self {
        Self::new()
    }
}

impl Ahrs {
    pub fn new() -> Self {
        Self {
            quaternion: Quaternion::IDENTITY,
            acc: Vector::ZERO,
            initialising: true,
            ramped_gain: INITIAL_GAIN,
            ramped_gain_step: (INITIAL_GAIN - GAIN) / INITIALISATION_PERIOD,
            angular_rate_recovery: false,
            half_accelerometer_feedback: Vector::ZERO,
            accelerometer_ignored: false,
            acceleration_recovery_trigger: 0,
            acceleration_recovery_timeout: RECOVERY_TRIGGER_PERIOD,
        }
    }

    fn reset(&mut self) {
        self.quaternion = Quaternion::IDENTITY;
        self.acc = Vector::ZERO;
        self.initialising = true;
        self.ramped_gain = INITIAL_GAIN;
        self.angular_rate_recovery = false;
        self.half_accelerometer_feedback = Vector::ZERO;
        self.accelerometer_ignored = false;
        self.acceleration_recovery_trigger = 0;
        self.acceleration_recovery_timeout = RECOVERY_TRIGGER_PERIOD;
    }

    /// One update with the gyroscope in degrees per second, the accelerometer
    /// in g and the time step in seconds. The heading stays zero while the
    /// filter initializes.
    pub fn update_no_mag(&mut self, gyr: Vector, acc: Vector, dt: f32) {
        self.update(gyr, acc, dt);
        if self.initialising {
            self.set_heading(0.0);
        }
    }

    fn update(&mut self, gyr: Vector, acc: Vector, dt: f32) {
        self.acc = acc;
        // Reinitialise if the gyroscope range is exceeded.
        if gyr.x.abs() > GYROSCOPE_RANGE
            || gyr.y.abs() > GYROSCOPE_RANGE
            || gyr.z.abs() > GYROSCOPE_RANGE
        {
            let quaternion = self.quaternion;
            self.reset();
            self.quaternion = quaternion;
            self.angular_rate_recovery = true;
        }
        // Ramp the gain down during the initialization.
        if self.initialising {
            self.ramped_gain -= self.ramped_gain_step * dt;
            if self.ramped_gain < GAIN || GAIN == 0.0 {
                self.ramped_gain = GAIN;
                self.initialising = false;
                self.angular_rate_recovery = false;
            }
        }
        let half_gravity = self.half_gravity();

        let mut half_accelerometer_feedback = Vector::ZERO;
        self.accelerometer_ignored = true;
        if !self.acc.is_zero() {
            self.half_accelerometer_feedback = feedback(self.acc.normalize(), half_gravity);
            if self.initialising
                || self.half_accelerometer_feedback.magnitude() <= ACCELERATION_REJECTION
            {
                self.accelerometer_ignored = false;
                self.acceleration_recovery_trigger -= 9;
            } else {
                self.acceleration_recovery_trigger += 1;
            }
            if self.acceleration_recovery_trigger > self.acceleration_recovery_timeout {
                self.acceleration_recovery_timeout = 0;
                self.accelerometer_ignored = false;
            } else {
                self.acceleration_recovery_timeout = RECOVERY_TRIGGER_PERIOD;
            }
            self.acceleration_recovery_trigger = self
                .acceleration_recovery_trigger
                .clamp(0, RECOVERY_TRIGGER_PERIOD);
            if !self.accelerometer_ignored {
                half_accelerometer_feedback = self.half_accelerometer_feedback;
            }
        }
        // Without a magnetometer its feedback is zero. The sum keeps the
        // crate's rounding.
        let half_magnetometer_feedback = Vector::ZERO;

        let half_gyroscope = gyr.scale(degrees_to_radians(0.5));
        let adjusted_half_gyroscope = half_gyroscope.add(
            &half_accelerometer_feedback
                .add(&half_magnetometer_feedback)
                .scale(self.ramped_gain),
        );
        self.quaternion = self.quaternion.add(
            &self
                .quaternion
                .mul_vector(&adjusted_half_gyroscope.scale(dt)),
        );
        self.quaternion = self.quaternion.normalize();
    }

    fn half_gravity(&self) -> Vector {
        let q = self.quaternion;
        Vector {
            x: q.x * q.z - q.w * q.y,
            y: q.y * q.z + q.w * q.x,
            z: q.w * q.w - 0.5 + q.z * q.z,
        }
    }

    fn set_heading(&mut self, heading: f32) {
        let q = self.quaternion;
        let yaw = libm::atan2f(q.w * q.z + q.x * q.y, 0.5 - q.y * q.y - q.z * q.z);
        let half_yaw_minus_heading = 0.5 * (yaw - degrees_to_radians(heading));
        let rotation = Quaternion {
            w: cosf(half_yaw_minus_heading),
            x: 0.0,
            y: 0.0,
            z: -sinf(half_yaw_minus_heading),
        };
        self.quaternion = rotation.mul(&self.quaternion);
    }

    /// The accelerometer measurement without gravity, in the sensor frame, in g.
    pub fn linear_acc(&self) -> Vector {
        let q = self.quaternion;
        let gravity = Vector {
            x: 2.0 * (q.x * q.z - q.w * q.y),
            y: 2.0 * (q.y * q.z + q.w * q.x),
            z: 2.0 * (q.w * q.w - 0.5 + q.z * q.z),
        };
        self.acc.sub(&gravity)
    }
}

fn feedback(sensor: Vector, reference: Vector) -> Vector {
    // The cross product, normalized once the error exceeds 90 degrees.
    if sensor.dot(&reference) < 0.0 {
        sensor.cross(&reference).normalize()
    } else {
        sensor.cross(&reference)
    }
}

fn degrees_to_radians(degrees: f32) -> f32 {
    degrees * (core::f32::consts::PI / 180.0)
}

/// The inverse square root of Fusion, a bit trick and one Newton step.
fn fast_inverse_sqrt(x: f32) -> f32 {
    let estimate = f32::from_bits((0x5F1F_1412 - ((x.to_bits() as i32) >> 1)) as u32);
    estimate * (1.690_002_3 - 0.714_158_2 * x * estimate * estimate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_imu_fusion_bit_for_bit() {
        // The `f32` build takes its own sine and cosine, which differ from
        // libm's in the last bit.
        if pelorus_core::REAL_IS_F32 {
            return;
        }
        let mut ours = Ahrs::new();
        let mut theirs = imu_fusion::FusionAhrs::new();
        for i in 0..5000 {
            let t = i as f32 * 0.005;
            let gyr = Vector::new(
                20.0 * (t * 1.3).sin(),
                -15.0 * (t * 0.7).cos(),
                5.0 * t.sin(),
            );
            let acc = Vector::new(
                0.1 * (t * 2.1).sin(),
                0.05 * t.cos(),
                1.0 + 0.02 * (t * 3.0).sin(),
            );
            // A zero gyroscope reading takes the other branch of the range check.
            let gyr = if i % 97 == 0 { Vector::ZERO } else { gyr };
            ours.update_no_mag(gyr, acc, 0.005);
            theirs.update_no_mag(
                imu_fusion::FusionVector::new(gyr.x, gyr.y, gyr.z),
                imu_fusion::FusionVector::new(acc.x, acc.y, acc.z),
                0.005,
            );
            let (q, p) = (ours.quaternion, theirs.quaternion);
            assert_eq!(
                [q.w, q.x, q.y, q.z].map(f32::to_bits),
                [p.w, p.x, p.y, p.z].map(f32::to_bits),
                "step {i}"
            );
            let (a, b) = (ours.linear_acc(), theirs.linear_acc());
            assert_eq!(
                [a.x, a.y, a.z].map(f32::to_bits),
                [b.x, b.y, b.z].map(f32::to_bits)
            );
        }
    }
}

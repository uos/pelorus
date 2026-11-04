//! Running the odometry without an IMU, by synthesizing one.
use alloc::string::{String, ToString};
use alloc::vec::Vec;
// Float maths through libm without std. With std the inherent methods take precedence.
#[allow(unused_imports)]
use nalgebra::{ComplexField, RealField};
use nalgebra::{Isometry3, Rotation3, Vector3};
use pelorus_core::{ImuMsg, Stamp};

/// The rotation vector taking `from` to `to`, in the `from` body frame.
fn rotation_increment(from: &Rotation3<f64>, to: &Rotation3<f64>) -> Vector3<f64> {
    let delta = nalgebra::UnitQuaternion::from_rotation_matrix(&(from.inverse() * to));

    // q and −q are the same rotation. The positive-scalar one is the short way.
    let (scalar, vector) = if delta.w < 0.0 {
        (-delta.w, -delta.imag())
    } else {
        (delta.w, delta.imag())
    };

    let sine = vector.norm();
    if sine < 1e-12 {
        return vector * 2.0;
    }
    vector * (2.0 * sine.atan2(scalar) / sine)
}

/// Longest plausible interval for a roughly 10 Hz scan stream.
const MAX_INTERVAL: f64 = 1.0;

/// Interval assumed when the one on offer makes no sense.
const NOMINAL_INTERVAL: f64 = 0.1;

/// Caps allocation when timestamps are invalid.
const MAX_STEPS: usize = 4096;

/// The filter's estimates of the IMU errors.
///
/// The filter subtracts the biases from every reading and adds its gravity
/// vector. A synthetic reading has to carry the same values, or the filter
/// turns them into a false rotation and acceleration between scans.
#[derive(Clone, Copy, Debug)]
pub struct ImuEstimates {
    pub gyro_bias: Vector3<f64>,
    pub accel_bias: Vector3<f64>,
    /// Gravity in the world frame, pointing down.
    pub gravity: Vector3<f64>,
}

/// Generates the IMU stream a rigidly-moving body would have produced under a
/// constant-twist model.
pub struct SyntheticImu {
    gravity: f64,
    rate: f64,
    frame_id: String,
    /// Rotation and time of the previous propagation, which is what the
    /// angular rate is measured across.
    previous: Option<(Isometry3<f64>, Stamp)>,
}

impl SyntheticImu {
    /// `rate` is how densely to sample the interval. It only has to be fine
    /// enough that the rotation between samples is small.
    pub fn new(gravity: f64, rate: f64) -> Self {
        Self {
            gravity,
            rate: rate.max(1.0),
            frame_id: "imu_synthetic".to_string(),
            previous: None,
        }
    }

    /// A stationary, level buffer used to initialize gravity and attitude.
    pub fn initialization_buffer(&self, samples: usize) -> Vec<ImuMsg> {
        (0..samples.max(2))
            .map(|i| {
                self.message(
                    Stamp::from_secs(i as f64 / self.rate),
                    Vector3::zeros(),
                    Vector3::new(0.0, 0.0, self.gravity),
                )
            })
            .collect()
    }

    /// The IMU stream for the interval `(from, to]`, given where the filter
    /// currently thinks it is and its IMU error estimates. The sample at `from`
    /// belongs to the previous interval and is retained by the integrator as
    /// its boundary sample.
    pub fn generate(
        &mut self,
        pose: &Isometry3<f64>,
        estimates: &ImuEstimates,
        from: Stamp,
        to: Stamp,
    ) -> (Vec<ImuMsg>, Option<Vector3<f64>>) {
        let rotation = pose.rotation.to_rotation_matrix();

        let (omega, linear_velocity) = match self.previous {
            Some((previous, previous_time)) if from > previous_time => {
                let dt = from.secs_since(previous_time);
                (
                    rotation_increment(&previous.rotation.to_rotation_matrix(), &rotation) / dt,
                    Some((pose.translation.vector - previous.translation.vector) / dt),
                )
            }
            // The first interval has no previous pose to difference, so it
            // propagates as pure translation and lets the Lidar correct it.
            _ => (Vector3::zeros(), None),
        };
        self.previous = Some((*pose, from));

        // Guarantee one sample and cap bad timestamp intervals.
        let span = match to.secs_since(from) {
            span if span.is_finite() && span > 0.0 && span <= MAX_INTERVAL => span,
            _ => NOMINAL_INTERVAL,
        };
        let from = to.add_secs(-span);

        let steps = ((span * self.rate).ceil() as usize).clamp(1, MAX_STEPS);
        let messages = (1..=steps)
            .map(|k| {
                let tau = span * k as f64 / steps as f64;
                // Rotate gravity with attitude to keep modelled acceleration zero.
                let attitude = rotation * Rotation3::from_scaled_axis(omega * tau);
                let specific_force = attitude.inverse() * -estimates.gravity;
                // The integrator scales accelerations by 9.81 over the mean
                // of the initialization buffer, whose magnitude is `gravity`.
                let reading = (specific_force + estimates.accel_bias) * (self.gravity / 9.81);
                self.message(from.add_secs(tau), omega + estimates.gyro_bias, reading)
            })
            .collect();
        (messages, linear_velocity)
    }

    fn message(&self, stamp: Stamp, gyro: Vector3<f64>, accel: Vector3<f64>) -> ImuMsg {
        ImuMsg {
            orientation: [0.0, 0.0, 0.0, 1.0],
            orientation_covariance: [0.0; 9],
            angular_velocity: [gyro.x, gyro.y, gyro.z],
            angular_velocity_covariance: [0.0; 9],
            linear_acceleration: [accel.x, accel.y, accel.z],
            linear_acceleration_covariance: [0.0; 9],
            frame_id: self.frame_id.clone(),
            stamp: stamp.to_msg(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pelorus_core::timestamp_to_secs;

    const LEVEL: ImuEstimates = ImuEstimates {
        gyro_bias: Vector3::new(0.0, 0.0, 0.0),
        accel_bias: Vector3::new(0.0, 0.0, 0.0),
        gravity: Vector3::new(0.0, 0.0, -9.81),
    };

    #[test]
    fn readings_minus_the_estimated_errors_are_the_constant_twist() {
        let mut synthetic = SyntheticImu::new(9.81, 200.0);
        let estimates = ImuEstimates {
            gyro_bias: Vector3::new(0.01, -0.02, 0.005),
            accel_bias: Vector3::new(0.1, 0.05, -0.2),
            gravity: Vector3::new(0.3, -0.2, -9.8),
        };
        let first = Isometry3::identity();
        let second = Isometry3::from_parts(
            nalgebra::Translation3::identity(),
            nalgebra::UnitQuaternion::from_euler_angles(0.0, 0.0, 0.05),
        );
        synthetic.generate(
            &first,
            &estimates,
            Stamp::from_secs(10.0),
            Stamp::from_secs(10.1),
        );
        let (messages, _) = synthetic.generate(
            &second,
            &estimates,
            Stamp::from_secs(10.1),
            Stamp::from_secs(10.2),
        );

        for (k, message) in messages.iter().enumerate() {
            let omega = Vector3::from(message.angular_velocity) - estimates.gyro_bias;
            assert!((omega - Vector3::new(0.0, 0.0, 0.5)).norm() < 1e-9);
            // What the filter integrates: the rotated, bias-free reading plus gravity.
            let tau = 0.1 * (k + 1) as f64 / messages.len() as f64;
            let attitude =
                second.rotation.to_rotation_matrix() * Rotation3::from_scaled_axis(omega * tau);
            let acceleration = attitude
                * (Vector3::from(message.linear_acceleration) - estimates.accel_bias)
                + estimates.gravity;
            assert!(acceleration.norm() < 1e-9, "sample {k}: {acceleration}");
        }
    }

    #[test]
    fn consecutive_intervals_do_not_repeat_boundary_timestamp() {
        let mut synthetic = SyntheticImu::new(9.81, 200.0);
        let pose = Isometry3::identity();

        let (first, _) = synthetic.generate(
            &pose,
            &LEVEL,
            Stamp::from_secs(10.0),
            Stamp::from_secs(10.1),
        );
        let (second, _) = synthetic.generate(
            &pose,
            &LEVEL,
            Stamp::from_secs(10.1),
            Stamp::from_secs(10.2),
        );

        assert!(!first.is_empty());
        assert!(!second.is_empty());
        assert!(
            timestamp_to_secs(&first.last().unwrap().stamp)
                < timestamp_to_secs(&second.first().unwrap().stamp)
        );
        assert!((timestamp_to_secs(&second.last().unwrap().stamp) - 10.2).abs() < 2e-9);
    }

    #[test]
    fn translation_between_corrected_poses_becomes_world_velocity() {
        let mut synthetic = SyntheticImu::new(9.81, 200.0);
        let first_pose = Isometry3::translation(1.0, 2.0, 3.0);
        let second_pose = Isometry3::translation(1.2, 1.9, 3.4);

        let (_, first_velocity) = synthetic.generate(
            &first_pose,
            &LEVEL,
            Stamp::from_secs(10.0),
            Stamp::from_secs(10.1),
        );
        let (_, second_velocity) = synthetic.generate(
            &second_pose,
            &LEVEL,
            Stamp::from_secs(10.1),
            Stamp::from_secs(10.2),
        );

        assert!(first_velocity.is_none());
        let velocity = second_velocity.unwrap();
        assert!((velocity - Vector3::new(2.0, -1.0, 4.0)).norm() < 1e-12);
    }
}

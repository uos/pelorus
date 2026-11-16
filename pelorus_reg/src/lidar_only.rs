//! Running the odometry without an IMU, by synthesizing one.
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use pelorus_core::{Real, double};
// Float maths through libm without std. With std the inherent methods take precedence.
#[allow(unused_imports)]
use nalgebra::{ComplexField, RealField};
use nalgebra::{Isometry3, Rotation3, Vector3};
use pelorus_core::{ImuMsg, Stamp};

/// The rotation vector taking `from` to `to`, in the `from` body frame.
fn rotation_increment(from: &Rotation3<Real>, to: &Rotation3<Real>) -> Vector3<Real> {
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
const MAX_INTERVAL: Real = 1.0;

/// Interval assumed when the one on offer makes no sense.
const NOMINAL_INTERVAL: Real = 0.1;

/// Caps allocation when timestamps are invalid.
const MAX_STEPS: usize = 4096;

/// The filter's estimates of the IMU errors.
///
/// The filter subtracts the biases from every reading and adds its gravity
/// vector. A synthetic reading has to carry the same values, or the filter
/// turns them into a false rotation and acceleration between scans.
#[derive(Clone, Copy, Debug)]
pub struct ImuEstimates {
    pub gyro_bias: Vector3<Real>,
    pub accel_bias: Vector3<Real>,
    /// Gravity in the world frame, pointing down.
    pub gravity: Vector3<Real>,
}

/// Generates the IMU stream a rigidly-moving body would have produced under a
/// constant-twist model.
pub struct SyntheticImu {
    gravity: Real,
    rate: Real,
    frame_id: String,
    /// Rotation and time of the previous propagation, which is what the
    /// angular rate is measured across.
    previous: Option<(Isometry3<Real>, Stamp)>,
    /// The map cells of the computation frame that `previous` is in.
    cells: [i64; 3],
}

impl SyntheticImu {
    /// `rate` is how densely to sample the interval. It only has to be fine
    /// enough that the rotation between samples is small.
    pub fn new(gravity: Real, rate: Real) -> Self {
        Self {
            gravity,
            rate: rate.max(1.0),
            frame_id: "imu_synthetic".to_string(),
            previous: None,
            cells: [0; 3],
        }
    }

    /// Moves the previous pose into the current computation frame, so the
    /// next twist does not include the frame's own move.
    pub fn follow(&mut self, origin: &pelorus_core::Origin) {
        if origin.cells == self.cells {
            return;
        }
        let shift = [0, 1, 2].map(|axis| origin.cells[axis] - self.cells[axis]);
        if let Some((pose, _)) = &mut self.previous {
            pose.translation.vector -= origin.shift_in_metres(shift);
        }
        self.cells = origin.cells;
    }

    /// A stationary, level buffer used to initialize gravity and attitude.
    pub fn initialization_buffer(&self, samples: usize) -> Vec<ImuMsg> {
        (0..samples.max(2))
            .map(|i| {
                self.message(
                    Stamp::from_secs(double(i as Real / self.rate)),
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
        pose: &Isometry3<Real>,
        estimates: &ImuEstimates,
        from: Stamp,
        to: Stamp,
    ) -> (Vec<ImuMsg>, Option<Vector3<Real>>) {
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
            // propagates as pure translation and lets the lidar correct it.
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
                let tau = span * k as Real / steps as Real;
                // Rotate gravity with attitude to keep modelled acceleration zero.
                let attitude =
                    rotation * pelorus_core::math::rotation_from_scaled_axis(omega * tau);
                let specific_force = attitude.inverse() * -estimates.gravity;
                // The integrator scales accelerations by 9.81 over the mean
                // of the initialization buffer, whose magnitude is `gravity`.
                let reading = (specific_force + estimates.accel_bias) * (self.gravity / 9.81);
                self.message(from.add_secs(tau), omega + estimates.gyro_bias, reading)
            })
            .collect();
        (messages, linear_velocity)
    }

    fn message(&self, stamp: Stamp, gyro: Vector3<Real>, accel: Vector3<Real>) -> ImuMsg {
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
            let omega = message.gyro() - estimates.gyro_bias;
            assert!((omega - Vector3::new(0.0, 0.0, 0.5)).norm() < crate::tolerance(1e-9));
            // What the filter integrates: the rotated, bias-free reading plus gravity.
            let tau = 0.1 * (k + 1) as Real / messages.len() as Real;
            let attitude = second.rotation.to_rotation_matrix()
                * pelorus_core::math::rotation_from_scaled_axis(omega * tau);
            let acceleration =
                attitude * (message.acc() - estimates.accel_bias) + estimates.gravity;
            assert!(
                acceleration.norm() < crate::tolerance(1e-9),
                "sample {k}: {acceleration}"
            );
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
        assert!((velocity - Vector3::new(2.0, -1.0, 4.0)).norm() < crate::tolerance(1e-12));
    }

    #[test]
    fn a_moved_computation_frame_is_no_velocity() {
        let mut synthetic = SyntheticImu::new(9.81, 200.0);
        let mut origin = pelorus_core::Origin::new(0.5);
        synthetic.generate(
            &Isometry3::translation(70.0, 0.0, 0.0),
            &LEVEL,
            Stamp::from_secs(10.0),
            Stamp::from_secs(10.1),
        );
        // The frame moves 128 cells along x, and the robot 0.2 m.
        origin.cells[0] += 128;
        synthetic.follow(&origin);
        let (_, velocity) = synthetic.generate(
            &Isometry3::translation(6.2, 0.0, 0.0),
            &LEVEL,
            Stamp::from_secs(10.1),
            Stamp::from_secs(10.2),
        );

        let velocity = velocity.unwrap();
        assert!((velocity - Vector3::new(2.0, 0.0, 0.0)).norm() < crate::tolerance(1e-9));
    }
}

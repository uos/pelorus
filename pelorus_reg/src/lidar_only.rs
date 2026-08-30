//! Running the odometry without an IMU, by synthesizing one.
use nalgebra::{Isometry3, Rotation3, Vector3};
use pelorus_core::{ImuMsg, time_msg_from_secs};

/// The rotation vector taking `from` to `to`, in the `from` body frame.
///
/// Deliberately not `Rotation3::scaled_axis`, whose angle is an unclamped
/// `acos((trace − 1) / 2)`: between two scans of a slowly turning platform the
/// trace is about 3, rounding pushes the argument past 1, and `acos` returns
/// NaN. A NaN angular rate here would go straight into the filter.
fn rotation_increment(from: &Rotation3<f64>, to: &Rotation3<f64>) -> Vector3<f64> {
    let delta = nalgebra::UnitQuaternion::from_rotation_matrix(&(from.inverse() * to));

    // q and −q are the same rotation; the positive-scalar one is the short way.
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

/// Generates the IMU stream a rigidly-moving body would have produced under a
/// constant-twist model.
pub struct SyntheticImu {
    gravity: f64,
    rate: f64,
    frame_id: String,
    /// Rotation and time of the previous propagation, which is what the
    /// angular rate is measured across.
    previous: Option<(Rotation3<f64>, f64)>,
}

impl SyntheticImu {
    /// `rate` is how densely to sample the interval; it only has to be fine
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
                    i as f64 / self.rate,
                    Vector3::zeros(),
                    &Rotation3::identity(),
                )
            })
            .collect()
    }

    /// The IMU stream for the interval `[from, to]`, given where the filter
    /// currently thinks it is.
    pub fn generate(&mut self, pose: &Isometry3<f64>, from: f64, to: f64) -> Vec<ImuMsg> {
        let rotation = pose.rotation.to_rotation_matrix();

        let omega = match self.previous {
            Some((previous, previous_time)) if from > previous_time => {
                rotation_increment(&previous, &rotation) / (from - previous_time)
            }
            // The first interval has no previous pose to difference, so it
            // propagates as pure translation and lets the LiDAR correct it.
            _ => Vector3::zeros(),
        };
        self.previous = Some((rotation, from));

        // Guarantee one sample and cap bad timestamp intervals.
        let span = match to - from {
            span if span.is_finite() && span > 0.0 && span <= MAX_INTERVAL => span,
            _ => NOMINAL_INTERVAL,
        };
        let from = to - span;

        let steps = ((span * self.rate).ceil() as usize).clamp(1, MAX_STEPS);
        (0..=steps)
            .map(|k| {
                let tau = span * k as f64 / steps as f64;
                // Rotate gravity with attitude to keep modelled acceleration zero.
                let attitude = rotation * Rotation3::from_scaled_axis(omega * tau);
                self.message(from + tau, omega, &attitude)
            })
            .collect()
    }

    fn message(&self, stamp: f64, omega: Vector3<f64>, attitude: &Rotation3<f64>) -> ImuMsg {
        let specific_force = attitude.inverse() * Vector3::new(0.0, 0.0, self.gravity);
        ImuMsg {
            orientation: [0.0, 0.0, 0.0, 1.0],
            orientation_covariance: [0.0; 9],
            angular_velocity: [omega.x, omega.y, omega.z],
            angular_velocity_covariance: [0.0; 9],
            linear_acceleration: [specific_force.x, specific_force.y, specific_force.z],
            linear_acceleration_covariance: [0.0; 9],
            frame_id: self.frame_id.clone(),
            stamp: time_msg_from_secs(stamp),
        }
    }
}

use nalgebra::*;
use pelorus_core::{ImuMsg, Real, UndistortParameters, real};

// TODO gravity is hardcoded on axis z, should be like eth with global gravity
#[allow(dead_code, unused_variables)]
pub fn integrate_imu(
    imu_data: &[ImuMsg],
    start_time: Real,
    config: Option<&UndistortParameters>,
    gravity: Real,
) -> Isometry3<Real> {
    let mut pose: Isometry<Real, Unit<Quaternion<Real>>, 3> = Isometry3::identity();
    let mut velocity = Vector3::zeros();
    let mut position = Vector3::zeros();

    let mut prev_time = start_time;
    let mut prev_accel = Vector3::zeros();

    let gravity = Vector3::new(0.0, 0.0, -gravity);

    for (i, imu_entry) in imu_data.iter().enumerate() {
        let accel_sample = &imu_entry.linear_acceleration;
        let gyro_sample = &imu_entry.angular_velocity;
        let t = real(crate::timestamp_to_secs(&imu_entry.stamp));

        let dt = if i == 0 { 0.0 } else { t - prev_time };

        let accel = Vector3::new(accel_sample[0], accel_sample[1], accel_sample[2]) + gravity;

        let gyro = Vector3::new(gyro_sample[0], gyro_sample[1], gyro_sample[2]);

        // Integrate rotation using the gyroscope data
        if dt > 0.0 {
            let angular_velocity = gyro * dt;
            let rotation: Unit<Quaternion<Real>> =
                pelorus_core::math::quaternion_from_scaled_axis(angular_velocity);
            pose.rotation *= rotation;
        }

        // Integrate velocity and position using the accelerometer data
        if dt > 0.0 {
            let current_accel = pose.rotation * accel;
            velocity += 0.5 * (prev_accel + current_accel) * dt;
            position += velocity * dt + 0.5 * current_accel * dt * dt;
            prev_accel = current_accel;
        }

        prev_time = t;
    }

    let translation = Translation3::from(position);

    Isometry3::from_parts(translation, pose.rotation)
}

use crate::config;
use crate::imu::ImuMsg;
use nalgebra::*;

type SO3 = sophus_lie::Rotation3<f64, 1>;

#[allow(dead_code, unused_variables)]
pub fn integrate_imu(
    imu_data: &Vec<ImuMsg>,
    start_time: f64,
    config: Option<&config::UndistortParameters>,
    use_ahrs_orientation: bool,
    gravity: f64,
) -> Isometry3<f64> {
    let k_gravity = Vector3::new(0.0, 0.0, -gravity);

    let mut init = Isometry3::identity();
    let mut kum_angular_velocity = Vector3::new(0.0, 0.0, 0.0);
    let mut kum_linear_velocity = Vector3::new(0.0, 0.0, 0.0);

    // bias is already removed when incoming step
    for imu_msg in imu_data {
        let sec_time = crate::timestamp_to_secs(&imu_msg.stamp);

        if sec_time < start_time {
            continue;
        }

        let delta_time = sec_time - start_time;

        let angular_velocity = Vector3::new(
            imu_msg.angular_velocity[0],
            imu_msg.angular_velocity[1],
            imu_msg.angular_velocity[2],
        );

        let linear_acceleration = Vector3::new(
            imu_msg.linear_acceleration[0],
            imu_msg.linear_acceleration[1],
            imu_msg.linear_acceleration[2],
        );

        let global_gravity = if use_ahrs_orientation {
            let orientation = nalgebra::Quaternion::new(
                imu_msg.orientation[0],
                imu_msg.orientation[1],
                imu_msg.orientation[2],
                imu_msg.orientation[3],
            );
            let orientation = nalgebra::UnitQuaternion::new_normalize(orientation);
            orientation.inverse().transform_vector(&k_gravity)
        } else {
            let delta_angle = delta_time * (angular_velocity + kum_angular_velocity) / 2.0;
            let half_exp = delta_angle / 2.0;
            let half_delta_rotation = SO3::exp(&half_exp);

            half_delta_rotation.inverse().transform(&k_gravity)
        };

        kum_angular_velocity = angular_velocity;

        let delta_linear_velocity = delta_time * (linear_acceleration + global_gravity);

        let added =
            init.rotation * (delta_time * (kum_linear_velocity + delta_linear_velocity / 2.0));

        init.translation.vector += added;

        kum_linear_velocity += delta_linear_velocity;
    }

    let final_angular_velocity = kum_angular_velocity;
    let final_linear_velocity = kum_linear_velocity;

    //(init, final_angular_velocity, final_linear_velocity)
    init
}

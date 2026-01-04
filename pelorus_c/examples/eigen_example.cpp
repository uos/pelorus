#include <iostream>
#include <iomanip>
#include "../pelorus.hpp"

int main()
{
    std::cout << "Version: " << pelorus::Pelorus::version() << "\n\n";

    try
    {
        // Create Pelorus instance
        pelorus::Pelorus pelorus;
        std::cout << "Created Pelorus instance\n\n";

        // For demonstration, we'll create a mock odometry result
        // In real usage, you would get this from pelorus.recv() or pelorus.try_recv()

        // Example: Creating test data
        pelorus_odometry test_odom = {0};
        test_odom.timestamp_sec = 1000;
        test_odom.timestamp_nanosec = 500000000;

        // Position: 1m forward, 0.5m left, 0.2m up
        test_odom.position[0] = 1.0;
        test_odom.position[1] = 0.5;
        test_odom.position[2] = 0.2;

        // Orientation: slight rotation around Z axis (yaw ~10 degrees)
        // Quaternion for ~10 degree rotation around Z: [w, x, y, z]
        test_odom.orientation[0] = 0.9962; // w
        test_odom.orientation[1] = 0.0;    // x
        test_odom.orientation[2] = 0.0;    // y
        test_odom.orientation[3] = 0.0872; // z

        // Linear velocity: 0.5 m/s forward
        test_odom.linear_velocity[0] = 0.5;
        test_odom.linear_velocity[1] = 0.0;
        test_odom.linear_velocity[2] = 0.0;

        // Angular velocity: 0.1 rad/s around Z
        test_odom.angular_velocity[0] = 0.0;
        test_odom.angular_velocity[1] = 0.0;
        test_odom.angular_velocity[2] = 0.1;

        // Identity covariances for simplicity
        for (int i = 0; i < 36; i++)
        {
            test_odom.pose_covariance[i] = (i % 7 == 0) ? 0.01 : 0.0;
            test_odom.twist_covariance[i] = (i % 7 == 0) ? 0.001 : 0.0;
        }

        // Wrap in Odometry object
        pelorus::Odometry odom(test_odom);

        std::cout << "1. Basic Conversions\n";
        std::cout << "--------------------\n";

        // Position
        Eigen::Vector3d pos = pelorus::eigen::to_position(odom.position());
        std::cout << "Position: [" << pos.x() << ", " << pos.y() << ", " << pos.z() << "]\n";

        // Orientation
        Eigen::Quaterniond quat = pelorus::eigen::to_quaternion(odom.orientation());
        std::cout << "Orientation (quat): w=" << quat.w()
                  << ", x=" << quat.x()
                  << ", y=" << quat.y()
                  << ", z=" << quat.z() << "\n";

        // Velocities
        Eigen::Vector3d lin_vel = pelorus::eigen::to_linear_velocity(odom.linear_velocity());
        Eigen::Vector3d ang_vel = pelorus::eigen::to_angular_velocity(odom.angular_velocity());
        std::cout << "Linear velocity: [" << lin_vel.x() << ", " << lin_vel.y() << ", " << lin_vel.z() << "]\n";
        std::cout << "Angular velocity: [" << ang_vel.x() << ", " << ang_vel.y() << ", " << ang_vel.z() << "]\n\n";

        std::cout << "2. Transformation Matrices\n";
        std::cout << "---------------------------\n";

        // Isometry3d (most common for rigid transforms)
        Eigen::Isometry3d transform = pelorus::eigen::to_transform(odom);
        std::cout << "Isometry3d Transform:\n"
                  << transform.matrix() << "\n\n";

        // Alternative syntax using free function
        Eigen::Isometry3d transform2 = pelorus::eigen::to_transform(odom);
        std::cout << "Using free function (same result):\n"
                  << transform2.matrix() << "\n\n";

        // Affine3d (more general)
        Eigen::Affine3d affine = pelorus::eigen::to_affine(odom);
        std::cout << "Affine3d Transform:\n"
                  << affine.matrix() << "\n\n";

        // Direct 4x4 matrix
        Eigen::Matrix4d mat = pelorus::eigen::to_matrix(odom);
        std::cout << "4x4 Matrix:\n"
                  << mat << "\n\n";

        std::cout << "3. Complete State\n";
        std::cout << "-----------------\n";

        // Get all odometry as Eigen types in one call
        auto state = pelorus::eigen::OdometryState::from_pelorus(odom);
        std::cout << "Timestamp: " << state.timestamp_sec << "."
                  << std::setfill('0') << std::setw(9) << state.timestamp_nanosec << "\n";
        std::cout << "Transform translation: " << state.transform.translation().transpose() << "\n";
        std::cout << "Linear velocity: " << state.linear_velocity.transpose() << "\n";
        std::cout << "Angular velocity: " << state.angular_velocity.transpose() << "\n";
        std::cout << "Pose covariance (first row): " << state.pose_cov.row(0) << "\n";
        std::cout << "Twist covariance (first row): " << state.twist_cov.row(0) << "\n\n";

        std::cout << "4. Practical Usage\n";
        std::cout << "------------------\n";

        // Example: Transform a point from sensor frame to world frame
        Eigen::Vector3d point_in_sensor(1.0, 0.0, 0.0); // 1m in front of sensor
        Eigen::Vector3d point_in_world = transform * point_in_sensor;
        std::cout << "Point in sensor: " << point_in_sensor.transpose() << "\n";
        std::cout << "Point in world: " << point_in_world.transpose() << "\n\n";

        // Example: Compose transformations
        Eigen::Isometry3d base_to_sensor = Eigen::Isometry3d::Identity();
        base_to_sensor.translation() = Eigen::Vector3d(0.1, 0.0, 0.5); // Sensor offset

        Eigen::Isometry3d world_to_base = transform;
        Eigen::Isometry3d world_to_sensor = world_to_base * base_to_sensor;

        std::cout << "World to sensor transform:\n"
                  << world_to_sensor.matrix() << "\n\n";
    }
    catch (const pelorus::Exception &e)
    {
        std::cerr << "Pelorus error: " << e.what() << "\n";
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

#include <iostream>
#include <iomanip>
#include <cmath>

#include "../pelorus.hpp"
#include "../pelorus_pcl.hpp"
#include "../pelorus_eigen.hpp"

static void fill_identity_cov(double* cov, double diag) {
    for (size_t i = 0; i < 9; i++) {
        cov[i] = (i % 4 == 0) ? diag : 0.0;
    }
}

int main() {
    std::cout << "Version: " << pelorus::Pelorus::version() << "\n\n";

    try {
        pelorus::Pelorus pelorus;
        std::cout << "Created Pelorus instance with default config\n\n";

        const int32_t base_sec = 1000;

        std::cout << "Sending 200 IMU messages for initialization...\n";
        for (int i = 0; i < 200; i++) {
            pelorus::ImuMsg imu;
            imu.timestamp_sec = base_sec;
            imu.timestamp_nanosec = i * 5000000;  // 5ms
            imu.frame_id = "imu";

            imu.orientation[0] = 1.0;  // w
            imu.angular_velocity[0] = 0.001 * (i % 3 - 1);
            imu.angular_velocity[1] = 0.001 * ((i + 1) % 3 - 1);
            imu.angular_velocity[2] = 0.002 * ((i + 2) % 3 - 1);
            imu.linear_acceleration[0] = 0.01 * (i % 5 - 2);
            imu.linear_acceleration[1] = 0.01 * ((i + 1) % 5 - 2);
            imu.linear_acceleration[2] = 9.81 + 0.01 * ((i + 2) % 5 - 2);

            fill_identity_cov(imu.orientation_covariance, 0.01);
            fill_identity_cov(imu.angular_velocity_covariance, 0.001);
            fill_identity_cov(imu.linear_acceleration_covariance, 0.01);

            pelorus.add_imu(imu);
        }

        constexpr int NUM_POINTS = 64;

        // Scan 1
        auto scan1 = ::pcl::PointCloud<pelorus::pcl::PointXYZIT>::Ptr(
            new ::pcl::PointCloud<pelorus::pcl::PointXYZIT>());
        scan1->points.resize(NUM_POINTS);
        scan1->width = NUM_POINTS;
        scan1->height = 1;
        scan1->is_dense = false;
        for (int i = 0; i < NUM_POINTS; i++) {
            const float angle = (float)i / NUM_POINTS * 6.28318f;
            const float range = 5.0f + 0.5f * (i % 8);

            auto& p = scan1->points[i];
            p.x = range * std::cos(angle);
            p.y = range * std::sin(angle);
            p.z = 0.2f * std::sin(angle * 3);
            p.intensity = 0.0f;
            p.t = static_cast<uint32_t>(((float)i / NUM_POINTS) * 100000000.0f);
        }
        pelorus::pcl::add_pointcloud(pelorus, scan1, base_sec, 600000000, "lidar");

        // Some more IMU
        for (int i = 200; i < 300; i++) {
            pelorus::ImuMsg imu;
            imu.timestamp_sec = base_sec;
            imu.timestamp_nanosec = i * 5000000;
            imu.frame_id = "imu";

            imu.orientation[0] = 1.0;
            imu.angular_velocity[0] = 0.001 * (i % 3 - 1);
            imu.angular_velocity[1] = 0.001 * ((i + 1) % 3 - 1);
            imu.angular_velocity[2] = 0.002 * ((i + 2) % 3 - 1);
            imu.linear_acceleration[0] = 0.01 * (i % 5 - 2);
            imu.linear_acceleration[1] = 0.01 * ((i + 1) % 5 - 2);
            imu.linear_acceleration[2] = 9.81 + 0.01 * ((i + 2) % 5 - 2);

            fill_identity_cov(imu.orientation_covariance, 0.01);
            fill_identity_cov(imu.angular_velocity_covariance, 0.001);
            fill_identity_cov(imu.linear_acceleration_covariance, 0.01);

            pelorus.add_imu(imu);
        }

        // Scan 2 (slightly moved)
        auto scan2 = ::pcl::PointCloud<pelorus::pcl::PointXYZIT>::Ptr(
            new ::pcl::PointCloud<pelorus::pcl::PointXYZIT>());
        scan2->points.resize(NUM_POINTS);
        scan2->width = NUM_POINTS;
        scan2->height = 1;
        scan2->is_dense = false;
        for (int i = 0; i < NUM_POINTS; i++) {
            const float angle = (float)i / NUM_POINTS * 6.28318f + 0.05f;
            const float range = 5.0f + 0.5f * (i % 8) + 0.05f;

            auto& p = scan2->points[i];
            p.x = range * std::cos(angle);
            p.y = range * std::sin(angle);
            p.z = 0.2f * std::sin(angle * 3);
            p.intensity = 0.0f;
            p.t = static_cast<uint32_t>(((float)i / NUM_POINTS) * 100000000.0f);
        }
        pelorus::pcl::add_pointcloud(pelorus, scan2, base_sec + 1, 200000000, "lidar");

        std::cout << "Waiting for odometry (blocking)...\n\n";
        auto odom = pelorus.recv();
        if (!odom.valid()) {
            std::cerr << "Error receiving odometry\n";
            return 1;
        }

        std::cout << std::fixed;
        std::cout << "  Position: [" << std::setprecision(3)
                  << odom.position()[0] << ", "
                  << odom.position()[1] << ", "
                  << odom.position()[2] << "]\n";
        std::cout << "  Orientation [w,x,y,z]: [" << std::setprecision(3)
              << odom.orientation()[0] << ", "
                  << odom.orientation()[1] << ", "
                  << odom.orientation()[2] << ", "
                  << odom.orientation()[3] << "]\n";
        std::cout << "  Linear velocity: [" << std::setprecision(3)
              << odom.linear_velocity()[0] << ", "
                  << odom.linear_velocity()[1] << ", "
                  << odom.linear_velocity()[2] << "]\n";
        std::cout << "  Angular velocity: [" << std::setprecision(3)
              << odom.angular_velocity()[0] << ", "
                  << odom.angular_velocity()[1] << ", "
                  << odom.angular_velocity()[2] << "]\n";
        std::cout << "  Point cloud: " << odom.num_points() << " points\n";

        for (size_t j = 0; j < odom.num_points() && j < 3; j++) {
            const auto* p = &odom.points()[j];
            std::cout << "    Point " << j << ": ["
                      << std::setprecision(3)
                      << p->x << ", " << p->y << ", " << p->z
                      << "] I=" << std::setprecision(3) << p->intensity << "\n";
        }

        // Eigen demo: SE(3) transform from odometry pose and apply it to a point.
        const Eigen::Isometry3d T_world_sensor = pelorus::eigen::to_transform(odom);
        const Eigen::Vector3d sample_point_sensor(1.0, 0.0, 0.0);
        const Eigen::Vector3d sample_point_world = T_world_sensor * sample_point_sensor;

        std::cout << "\nEigen demo\n";
        std::cout << "  T_world_sensor (matrix):\n" << std::setprecision(3) << T_world_sensor.matrix() << "\n";
        std::cout << "  point_sensor: " << std::setprecision(3) << sample_point_sensor.transpose() << "\n";
        std::cout << "  point_world:  " << std::setprecision(3) << sample_point_world.transpose() << "\n";

    } catch (const pelorus::Exception& e) {
        std::cerr << "Pelorus error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

#include <iostream>
#include <iomanip>
#include <cmath>
#include <utility>
#include <vector>

#include "../pelorus.hpp"

static void fill_identity_cov(double *cov, double diag)
{
    for (size_t i = 0; i < 9; i++)
    {
        cov[i] = (i % 4 == 0) ? diag : 0.0;
    }
}

int main()
{
    std::cout << "Version: " << pelorus::Pelorus::version() << "\n\n";

    try
    {
        pelorus::Pelorus pelorus("./pico-equality.rl");
        std::cout << "Created Pelorus instance with comparison config\n\n";

        const int32_t base_sec = 1000;

        std::cout << "Sending 200 IMU messages for initialization...\n";
        for (int i = 0; i < 200; i++)
        {
            pelorus::ImuMsg imu;
            imu.timestamp_sec = base_sec;
            imu.timestamp_nanosec = i * 5000000; // 5ms
            imu.frame_id = "";

            imu.orientation[0] = 1.0; // w
            imu.angular_velocity[0] = 0.004 + 0.001 * (i % 3 - 1);
            imu.angular_velocity[1] = -0.002 + 0.001 * ((i + 1) % 3 - 1);
            imu.angular_velocity[2] = 0.010 + 0.002 * ((i + 2) % 3 - 1);
            imu.linear_acceleration[0] = 0.03 + 0.01 * (i % 5 - 2);
            imu.linear_acceleration[1] = -0.015 + 0.01 * ((i + 1) % 5 - 2);
            imu.linear_acceleration[2] = 9.81 + 0.02 + 0.01 * ((i + 2) % 5 - 2);

            fill_identity_cov(imu.orientation_covariance, 0.01);
            fill_identity_cov(imu.angular_velocity_covariance, 0.001);
            fill_identity_cov(imu.linear_acceleration_covariance, 0.01);

            pelorus.add_imu(imu);
        }

        constexpr int NUM_POINTS = 512;
        const auto add_cloud = [&](const ::pcl::PointCloud<pelorus::pcl::PointXYZIT>::Ptr &cloud,
                                   int32_t ts_sec,
                                   uint32_t ts_nsec) -> uint64_t
        {
            std::vector<float> points;
            points.reserve(cloud->size() * 5);
            for (const auto &pt : cloud->points)
            {
                points.push_back(pt.x);
                points.push_back(pt.y);
                points.push_back(pt.z);
                points.push_back(pt.intensity);
                points.push_back(static_cast<float>(pt.t));
            }
            return pelorus.add_points_xyzit(points.data(), cloud->size(), ts_sec, ts_nsec);
        };

        // Scan 1
        auto scan1 = ::pcl::PointCloud<pelorus::pcl::PointXYZIT>::Ptr(
            new ::pcl::PointCloud<pelorus::pcl::PointXYZIT>());
        scan1->points.resize(NUM_POINTS);
        scan1->width = NUM_POINTS;
        scan1->height = 1;
        scan1->is_dense = false;
        for (int i = 0; i < NUM_POINTS; i++)
        {
            float x, y, z;
            if (i < NUM_POINTS / 2)
            {
                const int u = i % 16;
                const int v = i / 16;
                x = 4.0f + 0.12f * u;
                y = -1.0f + 0.12f * v;
                z = 0.01f * ((u + v) % 2);
            }
            else
            {
                const int j = i - NUM_POINTS / 2;
                const int u = j % 16;
                const int v = j / 16;
                x = 6.0f + 0.005f * ((u + v) % 2);
                y = -1.0f + 0.12f * u;
                z = 0.10f + 0.12f * v;
            }

            auto &p = scan1->points[i];
            p.x = x;
            p.y = y;
            p.z = z;
            p.intensity = 0.0f;
            p.t = static_cast<uint32_t>(((float)i / NUM_POINTS) * 100000000.0f);
        }
        const uint64_t scan1_id = add_cloud(scan1, base_sec, 600000000);
        std::cout << "Added scan id=" << scan1_id << "\n";

        // Some more IMU
        for (int i = 200; i < 300; i++)
        {
            pelorus::ImuMsg imu;
            imu.timestamp_sec = base_sec;
            imu.timestamp_nanosec = i * 5000000;
            imu.frame_id = "";

            imu.orientation[0] = 1.0;
            imu.angular_velocity[0] = 0.004 + 0.001 * (i % 3 - 1);
            imu.angular_velocity[1] = -0.002 + 0.001 * ((i + 1) % 3 - 1);
            imu.angular_velocity[2] = 0.010 + 0.002 * ((i + 2) % 3 - 1);
            imu.linear_acceleration[0] = 0.03 + 0.01 * (i % 5 - 2);
            imu.linear_acceleration[1] = -0.015 + 0.01 * ((i + 1) % 5 - 2);
            imu.linear_acceleration[2] = 9.81 + 0.02 + 0.01 * ((i + 2) % 5 - 2);

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
        for (int i = 0; i < NUM_POINTS; i++)
        {
            float x, y, z;
            if (i < NUM_POINTS / 2)
            {
                const int u = i % 16;
                const int v = i / 16;
                x = 4.0f + 0.12f * u;
                y = -1.0f + 0.12f * v;
                z = 0.01f * ((u + v) % 2);
            }
            else
            {
                const int j = i - NUM_POINTS / 2;
                const int u = j % 16;
                const int v = j / 16;
                x = 6.0f + 0.005f * ((u + v) % 2);
                y = -1.0f + 0.12f * u;
                z = 0.10f + 0.12f * v;
            }
            const float c = 0.996802f; // cos(0.08 rad)
            const float s = 0.079915f; // sin(0.08 rad)

            auto &p = scan2->points[i];
            p.x = c * x - s * y + 0.18f;
            p.y = s * x + c * y - 0.12f;
            p.z = z + 0.03f;
            p.intensity = 0.0f;
            p.t = static_cast<uint32_t>(((float)i / NUM_POINTS) * 100000000.0f);
        }
        const uint64_t scan2_id = add_cloud(scan2, base_sec + 1, 200000000);
        std::cout << "Added scan id=" << scan2_id << "\n";

        std::cout << "Waiting for odometry for scan id " << scan2_id << " (blocking)...\n\n";
        pelorus::Odometry odom;
        while (true)
        {
            auto recv_res = pelorus.recv();
            if (recv_res.status != pelorus::Pelorus::STATUS_OK)
            {
                std::cerr << "Error receiving odometry, status: " << recv_res.status << "\n";
                return 1;
            }
            if (recv_res.odom.scan_id() == scan2_id)
            {
                odom = std::move(recv_res.odom);
                break;
            }
        }

        // Match async_callback output style
        std::cout << std::fixed;
        std::cout << "Callback invoked! Odometry #1 received (scan_id=" << odom.scan_id() << ")\n";
        std::cout << "  Timestamp: " << odom.timestamp_sec() << "."
                  << std::setw(9) << std::setfill('0') << odom.timestamp_nanosec() << "\n";
        std::cout << std::setfill(' ');
        std::cout << "  Position: [" << std::setprecision(3)
                  << odom.position()[0] << ", "
                  << odom.position()[1] << ", "
                  << odom.position()[2] << "]\n";
        std::cout << "  Linear velocity: [" << std::setprecision(3)
                  << odom.linear_velocity()[0] << ", "
                  << odom.linear_velocity()[1] << ", "
                  << odom.linear_velocity()[2] << "]\n";
        std::cout << "  Point cloud: " << odom.num_points() << " points\n";

        if (std::getenv("PELORUS_DUMP_ALL_POINTS") != nullptr)
        {
            for (size_t j = 0; j < odom.num_points(); j++)
            {
                const auto *p = &odom.points()[j];
                std::cout << "    Point " << j << ": ["
                          << std::setprecision(6) << p->x << ", " << p->y << ", " << p->z
                          << "] I=" << std::setprecision(6) << p->intensity << "\n";
            }
        }
        else
        {
            for (size_t j = 0; j < odom.num_points() && j < 3; j++)
            {
                const auto *p = &odom.points()[j];
                std::cout << "    Point " << j << ": ["
                          << std::setprecision(3)
                          << p->x << ", " << p->y << ", " << p->z
                          << "] I=" << std::setprecision(3) << p->intensity << "\n";
            }
        }
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

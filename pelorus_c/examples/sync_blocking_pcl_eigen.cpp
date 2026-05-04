#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../pelorus.hpp"
#include "synthetic_dataset.h"

static void fill_identity_cov(double *cov, double diag)
{
    for (size_t i = 0; i < 9; i++) {
        cov[i] = (i % 4 == 0) ? diag : 0.0;
    }
}

static void add_imu(pelorus::Pelorus &lio, const synthetic_imu_t &src)
{
    pelorus::ImuMsg imu;
    synthetic_split_time(src.timestamp, &imu.timestamp_sec, &imu.timestamp_nanosec);
    imu.frame_id = "";
    imu.orientation[0] = 1.0;
    imu.orientation[1] = 0.0;
    imu.orientation[2] = 0.0;
    imu.orientation[3] = 0.0;
    imu.angular_velocity[0] = src.gyro[0];
    imu.angular_velocity[1] = src.gyro[1];
    imu.angular_velocity[2] = src.gyro[2];
    imu.linear_acceleration[0] = src.accel[0];
    imu.linear_acceleration[1] = src.accel[1];
    imu.linear_acceleration[2] = src.accel[2];
    fill_identity_cov(imu.orientation_covariance, 0.01);
    fill_identity_cov(imu.angular_velocity_covariance, 0.001);
    fill_identity_cov(imu.linear_acceleration_covariance, 0.01);
    lio.add_imu(imu);
}

static uint64_t add_cloud(pelorus::Pelorus &lio, const synthetic_cloud_t &cloud)
{
    int32_t sec;
    uint32_t nsec;
    synthetic_split_time(cloud.start_time, &sec, &nsec);
    std::vector<float> points;
    points.reserve(static_cast<size_t>(cloud.n_points) * 5);
    for (int32_t i = 0; i < cloud.n_points; i++) {
        points.push_back(cloud.points[i].x);
        points.push_back(cloud.points[i].y);
        points.push_back(cloud.points[i].z);
        points.push_back(cloud.points[i].intensity);
        points.push_back(cloud.points[i].timestamp * 1000000000.0f);
    }
    return lio.add_points_xyzit(points.data(), static_cast<size_t>(cloud.n_points), sec, nsec);
}

static void print_odom(const pelorus::Odometry &odom)
{
    std::cout << "Callback invoked! Odometry #1 received (scan_id=" << odom.scan_id() << ")\n";
    std::cout << "  Timestamp: " << odom.timestamp_sec() << "."
              << std::setw(9) << std::setfill('0') << odom.timestamp_nanosec() << std::setfill(' ') << "\n";
    auto pos = odom.position();
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Position: [" << pos[0] << ", " << pos[1] << ", " << pos[2] << "]\n";
    auto lv = odom.linear_velocity();
    std::cout << "  Linear velocity: [" << lv[0] << ", " << lv[1] << ", " << lv[2] << "]\n";
    std::cout << "  Point cloud: " << odom.num_points() << " points\n";

    size_t limit = std::getenv("PELORUS_DUMP_ALL_POINTS") ? odom.num_points() : std::min<size_t>(3, odom.num_points());
    for (size_t i = 0; i < limit; i++) {
        const pelorus_point_xyzi &p = odom.points()[i];
        if (std::getenv("PELORUS_DUMP_ALL_POINTS")) {
            std::cout << "    Point " << i << ": [" << std::setprecision(6) << p.x << ", "
                      << p.y << ", " << p.z << "] I=" << p.intensity << "\n";
        } else {
            std::cout << "    Point " << i << ": [" << std::setprecision(2) << p.x << ", "
                      << p.y << ", " << p.z << "] I=" << std::setprecision(1) << p.intensity << "\n";
        }
    }
}

int main(int argc, char **argv)
{
    const char *dataset_path = argc >= 2 ? argv[1] : "./synthetic_dataset.bin";

    try {
        pelorus::Pelorus lio("./no-accum.rl");
        FILE *f = fopen(dataset_path, "rb");
        if (!f) {
            perror(dataset_path);
            return 1;
        }

        uint64_t target_scan_id = 0;
        synthetic_record_t rec;
        while (synthetic_read_record(f, &rec) > 0) {
            if (rec.tag == SYNTH_TAG_IMU) {
                add_imu(lio, rec.imu);
            } else if (rec.tag == SYNTH_TAG_CLOUD) {
                target_scan_id = add_cloud(lio, rec.cloud);
                std::cout << "Added scan id=" << target_scan_id << "\n";
            }
            synthetic_free_record(&rec);
        }
        fclose(f);

        while (true) {
            auto result = lio.recv();
            if (result.status == pelorus::Pelorus::STATUS_OK && result.odom.scan_id() == target_scan_id) {
                print_odom(result.odom);
                break;
            }
            if (result.status != pelorus::Pelorus::STATUS_OK) {
                std::cerr << "Failed to receive odometry\n";
                return 1;
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}

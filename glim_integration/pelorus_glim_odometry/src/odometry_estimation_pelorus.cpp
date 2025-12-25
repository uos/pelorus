#include <glim/odometry/odometry_estimation_base.hpp>

#include <glim/odometry/estimation_frame.hpp>
#include <glim/util/config.hpp>

#include <gtsam_points/types/point_cloud_cpu.hpp>

#include <Eigen/Geometry>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>

#include "pelorus.hpp"
#include "pelorus_eigen.hpp"

namespace glim {

class OdometryEstimationPelorus final : public OdometryEstimationBase {
public:
  OdometryEstimationPelorus() {
    // Reuse GLIM's normal config lookup; user should point global.config_odometry to a file
    // that contains these pelorus-specific params.
    Config cfg(GlobalConfig::get_config_path("config_odometry"));

    max_wait_ms_ = cfg.param<int>("odometry_estimation", "max_wait_ms", 200);
    const std::string rl_path = cfg.param<std::string>("odometry_estimation", "pelorus_rl_path", "");

    // Load sensor extrinsics (used to fill EstimationFrame fields)
    {
      Config sensors_cfg(GlobalConfig::get_config_path("config_sensors"));
      T_lidar_imu_ = sensors_cfg.param<Eigen::Isometry3d>("sensors", "T_lidar_imu", Eigen::Isometry3d::Identity());
    }

    try {
      if (rl_path.empty()) {
        pelorus_ = std::make_unique<pelorus::Pelorus>();
      } else {
        pelorus_create_options opts = PELORUS_CREATE_OPTIONS_DEFAULT;
        opts.config_path = rl_path.c_str();
        pelorus_ = std::make_unique<pelorus::Pelorus>(&opts);
      }
    } catch (const pelorus::Exception& e) {
      spdlog::error("[odometry_estimation_pelorus] Pelorus init failed: {}", e.what());
    }

    last_T_world_lidar_.setIdentity();
  }

  ~OdometryEstimationPelorus() override = default;

  bool requires_imu() const override { return true; }

  void insert_imu(const double stamp, const Eigen::Vector3d& linear_acc, const Eigen::Vector3d& angular_vel) override {
    OdometryEstimationBase::insert_imu(stamp, linear_acc, angular_vel);

    if (!pelorus_) {
      return;
    }

    pelorus::ImuMsg imu;
    stamp_to_sec_nsec(stamp, imu.timestamp_sec, imu.timestamp_nanosec);
    imu.frame_id = "";

    imu.angular_velocity[0] = angular_vel.x();
    imu.angular_velocity[1] = angular_vel.y();
    imu.angular_velocity[2] = angular_vel.z();

    imu.linear_acceleration[0] = linear_acc.x();
    imu.linear_acceleration[1] = linear_acc.y();
    imu.linear_acceleration[2] = linear_acc.z();

    pelorus_->add_imu(imu);
  }

  EstimationFrame::ConstPtr insert_frame(const PreprocessedFrame::Ptr& frame, std::vector<EstimationFrame::ConstPtr>& marginalized_states) override {
    // Keep GLIM callback semantics (extensions can listen to these regardless of backend).
    OdometryEstimationBase::insert_frame(frame, marginalized_states);

    // Feed points into Pelorus (if available). We use the raw preprocessed LiDAR frame.
    if (pelorus_ && frame) {
      int32_t sec = 0;
      uint32_t nsec = 0;
      stamp_to_sec_nsec(frame->stamp, sec, nsec);

      const size_t n = static_cast<size_t>(frame->points.size());
      std::vector<float> xyzit;
      xyzit.resize(n * 5);

      for (size_t i = 0; i < n; i++) {
        const Eigen::Vector4d& p = frame->points[i];
        const float intensity = (i < frame->intensities.size()) ? static_cast<float>(frame->intensities[i]) : 0.0f;

        double t_sec = (i < frame->times.size()) ? frame->times[i] : 0.0;
        if (t_sec < 0.0) {
          t_sec = 0.0;
        }
        const double t_ns = std::min(4294967295.0, t_sec * 1e9);

        xyzit[i * 5 + 0] = static_cast<float>(p.x());
        xyzit[i * 5 + 1] = static_cast<float>(p.y());
        xyzit[i * 5 + 2] = static_cast<float>(p.z());
        xyzit[i * 5 + 3] = intensity;
        xyzit[i * 5 + 4] = static_cast<float>(t_ns);
      }

      pelorus_->add_points_xyzit(xyzit.data(), n, sec, nsec, "");
    }

    // Get Pelorus odometry result for this scan.
    bool got = false;
    pelorus::Odometry odom;

    if (pelorus_) {
      const auto start = std::chrono::steady_clock::now();
      while (true) {
        odom = pelorus_->try_recv();
        if (odom.valid()) {
          got = true;
          break;
        }
        const auto now = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (ms >= max_wait_ms_) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }

    EstimationFrame::Ptr out(new EstimationFrame);
    out->id = next_id_++;
    out->stamp = frame ? frame->stamp : 0.0;
    out->T_lidar_imu = T_lidar_imu_;

    // Fill point data: GLIM mapping expects a non-null `frame` for keyframing/overlap.
    // We use the incoming frame points (LiDAR frame) as the deskewed cloud baseline.
    if (frame) {
      out->raw_frame = frame;
      out->frame_id = FrameID::LIDAR;
      out->frame = std::make_shared<gtsam_points::PointCloudCPU>(frame->points);
      if (!frame->intensities.empty()) {
        out->frame->add_intensities(frame->intensities);
      }
    } else {
      out->frame_id = FrameID::LIDAR;
    }

    Eigen::Isometry3d T_world_lidar = last_T_world_lidar_;
    if (got) {
      T_world_lidar = pelorus::eigen::to_transform(odom);
      last_T_world_lidar_ = T_world_lidar;
    }

    out->T_world_lidar = T_world_lidar;
    out->T_world_imu = T_world_lidar * T_lidar_imu_;

    out->v_world_imu << 0.0, 0.0, 0.0;
    out->imu_bias.setZero();

    if (got) {
      const auto v = odom.linear_velocity();
      out->v_world_imu << v[0], v[1], v[2];
    }

    return out;
  }

private:
  static void stamp_to_sec_nsec(const double stamp, int32_t& sec, uint32_t& nsec) {
    const double s_floor = std::floor(stamp);
    sec = static_cast<int32_t>(s_floor);
    const double frac = stamp - s_floor;
    const double ns = frac * 1e9;
    if (ns <= 0.0) {
      nsec = 0;
    } else if (ns >= 4294967295.0) {
      nsec = 4294967295u;
    } else {
      nsec = static_cast<uint32_t>(ns);
    }
  }

private:
  std::unique_ptr<pelorus::Pelorus> pelorus_;

  int max_wait_ms_{200};
  long next_id_{0};

  Eigen::Isometry3d T_lidar_imu_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d last_T_world_lidar_{Eigen::Isometry3d::Identity()};
};

extern "C" OdometryEstimationBase* create_odometry_estimation_module() {
  return new OdometryEstimationPelorus();
}

}  // namespace glim

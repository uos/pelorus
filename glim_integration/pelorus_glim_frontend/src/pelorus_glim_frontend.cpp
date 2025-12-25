#include <glim/odometry/callbacks.hpp>
#include <glim/util/config.hpp>
#include <glim/util/extension_module.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/noiseModel/Diagonal.h>

#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <vector>

#include <spdlog/spdlog.h>

#include "pelorus.hpp"
#include "pelorus_eigen.hpp"

namespace glim {

class PelorusGlimFrontend final : public ExtensionModule {
public:
  PelorusGlimFrontend() {
    // Optional config file at: <glim config_root>/config_pelorus_frontend.json
    // (or override via config.json global mapping: "config_pelorus_frontend": "..." )
    Config cfg(GlobalConfig::get_config_path("config_pelorus_frontend"));
    const std::string rl_path = cfg.param<std::string>("pelorus_frontend", "pelorus_rl_path", "");

    try {
      if (rl_path.empty()) {
        pelorus_ = std::make_unique<pelorus::Pelorus>();
      } else {
        pelorus_create_options opts = PELORUS_CREATE_OPTIONS_DEFAULT;
        opts.config_path = rl_path.c_str();
        pelorus_ = std::make_unique<pelorus::Pelorus>(&opts);
      }
    } catch (const pelorus::Exception& e) {
      spdlog::warn("[pelorus_frontend] Pelorus init failed: {}", e.what());
    }

    // Constraint parameters (kept simple and robust by default)
    stamp_match_tolerance_sec_ = cfg.param<double>("pelorus_frontend", "stamp_match_tolerance_sec", 0.02);
    between_sigma_trans_ = cfg.param<double>("pelorus_frontend", "between_sigma_trans", 0.2);
    between_sigma_rot_rad_ = cfg.param<double>("pelorus_frontend", "between_sigma_rot_rad", 0.15);
    max_frame_buffer_ = cfg.param<int>("pelorus_frontend", "max_frame_buffer", 400);

    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    using std::placeholders::_4;

    OdometryEstimationCallbacks::on_insert_imu.add(std::bind(&PelorusGlimFrontend::on_insert_imu, this, _1, _2, _3));
    OdometryEstimationCallbacks::on_insert_frame.add(std::bind(&PelorusGlimFrontend::on_insert_frame, this, _1));
    OdometryEstimationCallbacks::on_new_frame.add(std::bind(&PelorusGlimFrontend::on_new_frame, this, _1));
    OdometryEstimationCallbacks::on_smoother_update.add(std::bind(&PelorusGlimFrontend::on_smoother_update, this, _1, _2, _3, _4));
  }

  ~PelorusGlimFrontend() override = default;

  bool ok() const override { return true; }

private:
  using gtsam::symbol_shorthand::X;

  struct FrameInfo {
    long id;
    double stamp;
  };

  static void stamp_to_sec_nsec(const double stamp, int32_t& sec, uint32_t& nsec) {
    // GLIM uses double timestamps (seconds). Convert conservatively.
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

  static double sec_nsec_to_stamp(const int32_t sec, const uint32_t nsec) {
    return static_cast<double>(sec) + 1e-9 * static_cast<double>(nsec);
  }

  // Use pelorus_eigen.hpp helpers for pose conversions.

  void on_new_frame(const EstimationFrame::ConstPtr& frame) {
    if (!frame) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    frame_buffer_.push_back(FrameInfo{frame->id, frame->stamp});
    while (static_cast<int>(frame_buffer_.size()) > max_frame_buffer_) {
      frame_buffer_.pop_front();
    }
  }

  void on_insert_imu(const double stamp, const Eigen::Vector3d& linear_acc, const Eigen::Vector3d& angular_vel) {
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

  void on_insert_frame(const PreprocessedFrame::Ptr& frame) {
    if (!pelorus_ || !frame) {
      return;
    }

    const auto raw = frame->raw_points;
    if (!raw) {
      return;
    }

    int32_t sec = 0;
    uint32_t nsec = 0;
    stamp_to_sec_nsec(raw->stamp, sec, nsec);

    const size_t n = static_cast<size_t>(raw->points.size());
    std::vector<float> xyzit;
    xyzit.resize(n * 5);

    for (size_t i = 0; i < n; i++) {
      const Eigen::Vector4d& p = raw->points[i];
      const float intensity = (i < raw->intensities.size()) ? static_cast<float>(raw->intensities[i]) : 0.0f;

      double t_sec = (i < raw->times.size()) ? raw->times[i] : 0.0;
      if (t_sec < 0.0) {
        t_sec = 0.0;
      }

      // Pelorus expects `t` as nanoseconds offset (non-negative) and casts to uint32.
      const double t_ns = std::min(4294967295.0, t_sec * 1e9);

      xyzit[i * 5 + 0] = static_cast<float>(p.x());
      xyzit[i * 5 + 1] = static_cast<float>(p.y());
      xyzit[i * 5 + 2] = static_cast<float>(p.z());
      xyzit[i * 5 + 3] = intensity;
      xyzit[i * 5 + 4] = static_cast<float>(t_ns);
    }

    pelorus_->add_points_xyzit(xyzit.data(), n, sec, nsec, "");
  }

  void on_smoother_update(
    gtsam_points::IncrementalFixedLagSmootherExtWithFallback& /*smoother*/,
    gtsam::NonlinearFactorGraph& new_factors,
    gtsam::Values& /*new_values*/,
    std::map<std::uint64_t, double>& /*new_stamps*/)
  {
    if (!pelorus_) {
      return;
    }

    // Drain Pelorus outputs and associate them to GLIM frame IDs by timestamp.
    std::vector<std::pair<long, Eigen::Isometry3d>> matched;
    while (true) {
      pelorus::Odometry odom = pelorus_->try_recv();
      if (!odom.valid()) {
        break;
      }
      const double stamp = sec_nsec_to_stamp(odom.timestamp_sec(), odom.timestamp_nanosec());
      const Eigen::Isometry3d T_world_lidar = pelorus::eigen::to_transform(odom);

      std::lock_guard<std::mutex> lock(mutex_);

      long best_id = -1;
      double best_dt = 1e9;
      for (const auto& f : frame_buffer_) {
        const double dt = std::abs(f.stamp - stamp);
        if (dt < best_dt) {
          best_dt = dt;
          best_id = f.id;
        }
      }

      if (best_id >= 0 && best_dt <= stamp_match_tolerance_sec_) {
        pelorus_pose_by_id_[best_id] = T_world_lidar;
        matched.emplace_back(best_id, T_world_lidar);
      }
    }

    // Add BetweenFactor constraints between consecutive Pelorus poses.
    // This is the ORB-SLAM-style integration: Pelorus acts as a constraint source.
    if (between_sigma_trans_ <= 0.0 || between_sigma_rot_rad_ <= 0.0) {
      return;
    }

    const auto noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) <<
        between_sigma_rot_rad_, between_sigma_rot_rad_, between_sigma_rot_rad_,
        between_sigma_trans_, between_sigma_trans_, between_sigma_trans_)
        .finished());

    std::sort(matched.begin(), matched.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [id, _T] : matched) {
      const long prev_id = id - 1;
      if (inserted_until_id_ >= id) {
        continue;
      }
      const auto prev_it = pelorus_pose_by_id_.find(prev_id);
      const auto cur_it = pelorus_pose_by_id_.find(id);
      if (prev_it == pelorus_pose_by_id_.end() || cur_it == pelorus_pose_by_id_.end()) {
        continue;
      }

      const Eigen::Isometry3d delta = prev_it->second.inverse() * cur_it->second;
      const gtsam::Pose3 meas(delta.matrix());
      new_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(prev_id), X(id), meas, noise);
      inserted_until_id_ = id;
    }
  }

private:
  std::unique_ptr<pelorus::Pelorus> pelorus_;

  // Matching + factor insertion state
  std::mutex mutex_;
  std::deque<FrameInfo> frame_buffer_;
  std::map<long, Eigen::Isometry3d> pelorus_pose_by_id_;
  long inserted_until_id_{-1};

  double stamp_match_tolerance_sec_{0.02};
  double between_sigma_trans_{0.2};
  double between_sigma_rot_rad_{0.15};
  int max_frame_buffer_{400};
};

extern "C" ExtensionModule* create_extension_module() {
  return new PelorusGlimFrontend();
}

}  // namespace glim

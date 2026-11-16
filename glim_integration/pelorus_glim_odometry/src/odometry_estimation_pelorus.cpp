#include <glim/odometry/odometry_estimation_base.hpp>

#include <glim/common/cloud_covariance_estimation.hpp>
#include <glim/odometry/callbacks.hpp>
#include <glim/odometry/estimation_frame.hpp>
#include <glim/util/config.hpp>

#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>

#include <Eigen/Cholesky>
#include <Eigen/Geometry>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <spdlog/spdlog.h>

#include "pelorus.hpp"

namespace glim {

class OdometryEstimationPelorus final : public OdometryEstimationBase {
public:
  OdometryEstimationPelorus() {
    Config cfg(GlobalConfig::get_config_path("config_odometry"));

    const std::string rl_path =
        cfg.param<std::string>("odometry_estimation", "pelorus_rl_path", "");
    const bool use_glim_config =
        cfg.param<bool>("odometry_estimation", "use_glim_config", true);
    const double query_resolution_factor = cfg.param<double>(
        "odometry_estimation", "query_resolution_factor", 0.5);
    const double map_resolution_m =
        cfg.param<double>("odometry_estimation", "map_resolution", 0.5);

    // Load sensor extrinsics
    {
      Config sensors_cfg(GlobalConfig::get_config_path("config_sensors"));
      T_lidar_imu_ = sensors_cfg.param<Eigen::Isometry3d>(
          "sensors", "T_lidar_imu", Eigen::Isometry3d::Identity());
    }

    // Generate temporary config if requested and path is empty
    std::string effective_rl_path = rl_path;
    if (use_glim_config && effective_rl_path.empty()) {
      try {
        // GLIM's `T_lidar_imu_` variable is IMU->lidar per config comment:
        // "T_lidar_imu transforms points in IMU into lidar"
        // Pelorus expects `from_lidar_transform` as lidar->IMU.
        const Eigen::Isometry3d T_imu_lidar = T_lidar_imu_.inverse();
        const Eigen::Vector3d t = T_imu_lidar.translation();
        const Eigen::Matrix3d R = T_imu_lidar.linear();

        const auto tmp_dir = std::filesystem::temp_directory_path();
        std::ostringstream name;
        name << "pelorus_glim_odometry_glim_config_" << std::hex
             << reinterpret_cast<std::uintptr_t>(this) << ".rl";
        generated_rl_path_ = (tmp_dir / name.str()).string();

        std::ofstream ofs(generated_rl_path_, std::ios::out | std::ios::trunc);
        if (ofs) {
          ofs << std::fixed << std::setprecision(16);
          ofs << "imu {\n";
          ofs << "  from_lidar_transform {\n";
          ofs << "    use_tf_tree = false\n";
          ofs << "    translation = [" << t.x() << "m, " << t.y() << "m, "
              << t.z() << "m]\n";
          ofs << "    rotation = [ [" << R(0, 0) << ", " << R(0, 1) << ", "
              << R(0, 2) << "],\n";
          ofs << "                 [" << R(1, 0) << ", " << R(1, 1) << ", "
              << R(1, 2) << "],\n";
          ofs << "                 [" << R(2, 0) << ", " << R(2, 1) << ", "
              << R(2, 2) << "] ]\n";
          ofs << "  }\n";
          ofs << "}\n";
          ofs << "lidar.as_sync = true\n";
          ofs << "preprocessing.downsample.bypass_output_downsample = true\n";
          ofs << "preprocessing.downsample.backend = Backend\n";
          ofs << "preprocessing.downsample.query_resolution_factor = "
              << query_resolution_factor << "\n";
          if (map_resolution_m > 0.0)
            ofs << "registration.map.resolution = " << map_resolution_m
                << "m\n";

          effective_rl_path = generated_rl_path_;
          logger->info(
              "[odometry_estimation_pelorus] query_resolution_factor={}, "
              "map_resolution={}m",
              query_resolution_factor, map_resolution_m);
        }
      } catch (const std::exception &e) {
        logger->error("[odometry_estimation_pelorus] Config gen failed: {}",
                      e.what());
      }
    }

    // Initialize Pelorus
    try {
      if (effective_rl_path.empty()) {
        pelorus_ = std::make_unique<pelorus::Pelorus>();
      } else {
        pelorus_create_options opts = PELORUS_CREATE_OPTIONS_DEFAULT;
        opts.config_path = effective_rl_path.c_str();
        pelorus_ = std::make_unique<pelorus::Pelorus>(&opts);
      }

      // REGISTER CALLBACK
      // We set the callback immediately. Note: 'this' is safe here because the
      // object is constructed. We must unset it in the destructor.
      {
        pelorus_->set_callback_with_bias(
            [this](const pelorus_odometry *odom_c,
                   const pelorus_imu_bias *bias_c) {
              if (odom_c && bias_c) {
                this->on_odometry_callback(odom_c, bias_c);
              }
            });
      }
    } catch (const pelorus::Exception &e) {
      logger->error("[odometry_estimation_pelorus] Pelorus init failed: {}",
                    e.what());
    }

    covariance_estimation_ = std::make_unique<CloudCovarianceEstimation>();
    last_T_world_lidar_.setIdentity();
  }

  ~OdometryEstimationPelorus() override {
    if (pelorus_) {
      // Disable callback first to prevent use-after-free of 'this' if pelorus
      // thread is still active
      pelorus_->set_callback_with_bias(nullptr);
    }
    // generated_rl_path_ could be deleted here if desired, but keeping it for
    // debug is often useful.
  }

  bool requires_imu() const override { return true; }

  void insert_imu(const double stamp, const Eigen::Vector3d &linear_acc,
                  const Eigen::Vector3d &angular_vel) override {
    OdometryEstimationBase::insert_imu(stamp, linear_acc, angular_vel);

    if (!pelorus_)
      return;

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

  EstimationFrame::ConstPtr insert_frame(
      const PreprocessedFrame::Ptr &frame,
      std::vector<EstimationFrame::ConstPtr> &marginalized_states) override {
    OdometryEstimationBase::insert_frame(frame, marginalized_states);
    drain_marginalized_frames(marginalized_states);

    if (!pelorus_ || !frame) {
      return EstimationFrame::ConstPtr();
    }

    // 1. Prepare Data for Pelorus
    int32_t sec = 0;
    uint32_t nsec = 0;
    stamp_to_sec_nsec(frame->stamp, sec, nsec);

    const size_t n = frame->points.size();
    std::vector<float> xyzit;
    xyzit.resize(n * 5);

    for (size_t i = 0; i < n; i++) {
      const Eigen::Vector4d &p = frame->points[i];
      const float intensity = (i < frame->intensities.size())
                                  ? static_cast<float>(frame->intensities[i])
                                  : 0.0f;

      double t_sec = (i < frame->times.size()) ? frame->times[i] : 0.0;
      if (t_sec < 0.0)
        t_sec = 0.0;
      const double t_ns = std::min(4294967295.0, t_sec * 1e9);

      xyzit[i * 5 + 0] = static_cast<float>(p.x());
      xyzit[i * 5 + 1] = static_cast<float>(p.y());
      xyzit[i * 5 + 2] = static_cast<float>(p.z());
      xyzit[i * 5 + 3] = intensity;
      xyzit[i * 5 + 4] = static_cast<float>(t_ns);
    }

    uint64_t scan_id = pelorus_->add_points_xyzit(xyzit.data(), n, sec, nsec);

    if (scan_id == 0) {
      logger->error(
          "[odometry_estimation_pelorus] add_points_xyzit returned an error.");
      return EstimationFrame::ConstPtr();
    }

    EstimationFrame::Ptr out(new EstimationFrame);
    out->id = next_id_++;
    out->stamp = frame->stamp;
    out->T_lidar_imu = T_lidar_imu_;
    out->raw_frame = frame;
    out->frame_id = FrameID::LIDAR;

    // Populate cloud data
    auto cloud = std::make_shared<gtsam_points::PointCloudCPU>(frame->points);
    if (!frame->intensities.empty())
      cloud->add_intensities(frame->intensities.data(),
                             static_cast<int>(frame->intensities.size()));
    if (!frame->times.empty())
      cloud->add_times(frame->times);
    out->frame = cloud;

    EstimationFrame::Ptr completed_frame;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);

      // Check if the result arrived extremely fast and is already cached
      auto cache_it = cached_results_.find(scan_id);
      if (cache_it != cached_results_.end()) {
        // HIT: Apply cached result immediately
        apply_result_to_frame(out, cache_it->second);

        cached_results_.erase(cache_it);
        // Cleanup cache order tracker
        for (auto it = cache_order_.begin(); it != cache_order_.end(); ++it) {
          if (*it == scan_id) {
            cache_order_.erase(it);
            break;
          }
        }

        completed_frame = out;
      } else {
        // MISS: The callback hasn't arrived yet (normal case). Register as
        // pending.
        pending_frames_[scan_id] = out;
      }
    }

    if (completed_frame) {
      complete_frame(completed_frame);
      drain_marginalized_frames(marginalized_states);
      return completed_frame;
    }

    // Pelorus completes asynchronously. The callback publishes the current
    // estimate, and a later insert_frame() (or get_remaining_frames() at end of
    // sequence) hands finalized frames to GLIM's submapping path.
    return EstimationFrame::ConstPtr();
  }

  std::vector<EstimationFrame::ConstPtr> get_remaining_frames() override {
    std::vector<EstimationFrame::ConstPtr> remaining_frames;

    {
      std::unique_lock<std::mutex> lock(data_mutex_);
      std::uint64_t observed_generation = result_generation_;

      while (!pending_frames_.empty()) {
        const bool made_progress = results_cv_.wait_for(
            lock, kResultStallTimeout, [this, observed_generation] {
              return pending_frames_.empty() ||
                     result_generation_ != observed_generation;
            });

        if (!made_progress) {
          logger->warn(
              "[odometry_estimation_pelorus] Timed out waiting for {} pending "
              "Pelorus result(s) at end of sequence.",
              pending_frames_.size());
          break;
        }

        observed_generation = result_generation_;
      }

      while (!marginalized_frames_.empty()) {
        remaining_frames.push_back(marginalized_frames_.front());
        marginalized_frames_.pop_front();
      }

      while (!active_frames_.empty()) {
        remaining_frames.push_back(active_frames_.front());
        active_frames_.pop_front();
      }
    }

    if (!remaining_frames.empty()) {
      OdometryEstimationCallbacks::on_marginalized_frames(remaining_frames);
    }

    return remaining_frames;
  }

private:
  struct CachedResult {
    Eigen::Isometry3d T_world_lidar;
    Eigen::Vector3d velocity;
    Eigen::Vector3d accel_bias;
    Eigen::Vector3d gyro_bias;
    std::shared_ptr<gtsam_points::PointCloudCPU> cloud;
  };

  // Triggered from background Pelorus thread (C context)
  void on_odometry_callback(const pelorus_odometry *odom_c,
                            const pelorus_imu_bias *bias_c) {
    if (!odom_c || !bias_c)
      return;

    uint64_t scan_id = odom_c->scan_id;
    CachedResult result;

    // Pelorus poses and deskewed clouds are always expressed in the lidar
    // frame.
    const double *p = odom_c->position;
    const double *q = odom_c->orientation; // w, x, y, z

    Eigen::Vector3d trans(p[0], p[1], p[2]);
    Eigen::Quaterniond quat(q[0], q[1], q[2], q[3]);
    result.T_world_lidar = Eigen::Isometry3d::Identity();
    result.T_world_lidar.linear() = quat.toRotationMatrix();
    result.T_world_lidar.translation() = trans;

    const double *v = odom_c->linear_velocity;
    result.velocity =
        result.T_world_lidar.linear() * Eigen::Vector3d(v[0], v[1], v[2]);
    result.accel_bias =
        Eigen::Vector3d(bias_c->accel_bias[0], bias_c->accel_bias[1],
                        bias_c->accel_bias[2]);
    result.gyro_bias =
        Eigen::Vector3d(bias_c->gyro_bias[0], bias_c->gyro_bias[1],
                        bias_c->gyro_bias[2]);

    result.cloud = od_to_cloud_raw(odom_c);

    // Prepare holders for notifications we will perform outside the lock
    EstimationFrame::Ptr frame_to_publish = nullptr;
    std::vector<EstimationFrame::Ptr> frames_to_publish;
    EstimationFrame::Ptr init_finished_frame;

    {
      std::lock_guard<std::mutex> lock(data_mutex_);

      last_T_world_lidar_ = result.T_world_lidar;
      result_generation_++;

      // Pelorus's initial map-only scans do not produce odometry results. The
      // first callback therefore marks initialization complete. Discard older
      // pending input frames because no result will arrive for them.
      if (!pelorus_initialized_) {
        for (auto it = pending_frames_.begin(); it != pending_frames_.end();) {
          if (it->first < scan_id) {
            it = pending_frames_.erase(it);
          } else {
            ++it;
          }
        }

        pelorus_initialized_ = true;

        // Compose an estimated frame to signal 'finished' to GLIM
        EstimationFrame::Ptr estimated(new EstimationFrame);
        // Prefer using the pending frame's stamp if available
        if (auto pit = pending_frames_.find(scan_id);
            pit != pending_frames_.end() && pit->second) {
          estimated->stamp = pit->second->stamp;
          estimated->raw_frame = pit->second->raw_frame;
        } else {
          const double stamp =
              static_cast<double>(odom_c->timestamp_sec) +
              static_cast<double>(odom_c->timestamp_nanosec) * 1e-9;
          estimated->stamp = stamp;
        }

        estimated->id = -1;
        estimated->T_lidar_imu = T_lidar_imu_;
        estimated->T_world_lidar = result.T_world_lidar;
        estimated->T_world_imu = result.T_world_lidar * T_lidar_imu_;
        estimated->v_world_imu = result.velocity;
        estimated->imu_bias << result.accel_bias, result.gyro_bias;
        estimated->frame_id = FrameID::LIDAR;

        init_finished_frame = estimated;

        logger->info("[odometry_estimation_pelorus] Pelorus reported "
                     "initialization (scan_id={}).",
                     scan_id);

        // Associate the first result with its input frame. If the callback won
        // the race with insert_frame(), cache it for the normal cache-hit path.
        auto pending_it_curr = pending_frames_.find(scan_id);
        if (pending_it_curr != pending_frames_.end()) {
          apply_result_to_frame(pending_it_curr->second, result);
          frames_to_publish.push_back(pending_it_curr->second);
          pending_frames_.erase(pending_it_curr);
        } else {
          cached_results_[scan_id] = result;
          cache_order_.push_back(scan_id);
        }

        // Update last pose again
        last_T_world_lidar_ = result.T_world_lidar;
      } else {
        // Normal behavior after initialization
        auto pending_it = pending_frames_.find(scan_id);
        if (pending_it != pending_frames_.end()) {
          frame_to_publish = pending_it->second;
          apply_result_to_frame(frame_to_publish, result);
          pending_frames_.erase(pending_it);
        } else {
          // No frame waiting yet (unlikely if loop is fast, but possible), or
          // we are ahead. Store in cache for when insert_frame is called.
          cached_results_[scan_id] = result;
          cache_order_.push_back(scan_id);

          // Simple cache eviction
          if (cache_order_.size() > kCacheMax) {
            uint64_t old_id = cache_order_.front();
            cache_order_.pop_front();
            cached_results_.erase(old_id);
          }
        }
      }

      if (frame_to_publish) {
        queue_completed_frame_locked(frame_to_publish);
      }
      for (const auto &frame : frames_to_publish) {
        queue_completed_frame_locked(frame);
      }
    }

    results_cv_.notify_all();

    // Perform notifications outside the lock
    if (init_finished_frame) {
      IMUStateInitializationCallbacks::on_finished(init_finished_frame);
    }

    if (frame_to_publish) {
      OdometryEstimationCallbacks::on_new_frame(frame_to_publish);
    }

    for (const auto &f : frames_to_publish) {
      OdometryEstimationCallbacks::on_new_frame(f);
    }
  }

  void complete_frame(const EstimationFrame::Ptr &frame) {
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      queue_completed_frame_locked(frame);
    }

    OdometryEstimationCallbacks::on_new_frame(frame);
  }

  void queue_completed_frame_locked(const EstimationFrame::ConstPtr &frame) {
    active_frames_.push_back(frame);

    while (active_frames_.size() > kMaxActiveFrames) {
      marginalized_frames_.push_back(active_frames_.front());
      active_frames_.pop_front();
    }
  }

  void drain_marginalized_frames(
      std::vector<EstimationFrame::ConstPtr> &marginalized_states) {
    std::vector<EstimationFrame::ConstPtr> newly_marginalized;

    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      while (!marginalized_frames_.empty()) {
        newly_marginalized.push_back(marginalized_frames_.front());
        marginalized_frames_.pop_front();
      }
    }

    marginalized_states.insert(marginalized_states.end(),
                               newly_marginalized.begin(),
                               newly_marginalized.end());

    if (!newly_marginalized.empty()) {
      OdometryEstimationCallbacks::on_marginalized_frames(newly_marginalized);
    }
  }

  // Helper: Apply result struct to frame object
  void apply_result_to_frame(EstimationFrame::Ptr &frame,
                             const CachedResult &res) {
    frame->T_world_lidar = res.T_world_lidar;
    frame->T_world_imu = res.T_world_lidar * T_lidar_imu_;
    frame->v_world_imu = res.velocity;
    frame->imu_bias << res.accel_bias, res.gyro_bias;
    frame->frame_id = FrameID::LIDAR;

    // Keep the returned deskewed points in the lidar frame so the cloud and
    // T_world_lidar use the same coordinates.
    if (res.cloud) {
      std::vector<Eigen::Vector4d> points_lidar(res.cloud->size());
      for (std::size_t i = 0; i < res.cloud->size(); i++) {
        points_lidar[i] = res.cloud->points[i];
      }

      auto cloud = std::make_shared<gtsam_points::PointCloudCPU>(points_lidar);
      if (res.cloud->intensities_storage.size() == points_lidar.size()) {
        cloud->add_intensities(res.cloud->intensities_storage);
      }

      if (!points_lidar.empty()) {
        const int requested_neighbors =
            frame->raw_frame ? frame->raw_frame->k_neighbors
                             : kDefaultCovarianceNeighbors;
        const std::vector<int> neighbors =
            find_neighbors(points_lidar, requested_neighbors);

        std::vector<Eigen::Vector4d> normals;
        std::vector<Eigen::Matrix4d> covariances;
        covariance_estimation_->estimate(points_lidar, neighbors, normals,
                                         covariances);
        cloud->add_normals(normals);
        cloud->add_covs(covariances);
      }

      frame->frame = cloud;
    }
  }

  static std::vector<int>
  find_neighbors(const std::vector<Eigen::Vector4d> &points,
                 const int requested_neighbors) {
    if (points.empty()) {
      return {};
    }

    const int num_points = static_cast<int>(points.size());
    const int k = std::clamp(requested_neighbors, 1, num_points);
    const gtsam_points::KdTree tree(points.data(), num_points);
    std::vector<int> neighbors(points.size() * static_cast<std::size_t>(k));

    for (std::size_t i = 0; i < points.size(); i++) {
      std::vector<std::size_t> indices(static_cast<std::size_t>(k), i);
      std::vector<double> squared_distances(static_cast<std::size_t>(k));
      tree.knn_search(points[i].data(), static_cast<std::size_t>(k),
                      indices.data(), squared_distances.data());

      const std::size_t offset = i * static_cast<std::size_t>(k);
      std::transform(indices.begin(), indices.end(),
                     neighbors.begin() + offset, [](const std::size_t index) {
                       return static_cast<int>(index);
                     });
    }

    return neighbors;
  }

  // Helper: Manual conversion from raw C struct to GTSAM cloud
  std::shared_ptr<gtsam_points::PointCloudCPU>
  od_to_cloud_raw(const pelorus_odometry *odom) {
    if (!odom || odom->num_points == 0)
      return nullptr;

    std::vector<Eigen::Vector4d> pts;
    pts.reserve(odom->num_points);
    std::vector<float> ints;
    ints.reserve(odom->num_points);

    const pelorus_point_xyzi *pts_ptr = odom->points;
    for (size_t i = 0; i < odom->num_points; ++i) {
      const auto &p = pts_ptr[i];
      pts.emplace_back(p.x, p.y, p.z, 1.0);
      ints.push_back(p.intensity);
    }

    auto pc = std::make_shared<gtsam_points::PointCloudCPU>(pts);
    pc->add_intensities(ints.data(), static_cast<int>(ints.size()));
    return pc;
  }

  static void stamp_to_sec_nsec(const double stamp, int32_t &sec,
                                uint32_t &nsec) {
    const double s_floor = std::floor(stamp);
    sec = static_cast<int32_t>(s_floor);
    const double frac = stamp - s_floor;
    const double ns = frac * 1e9;
    nsec = (ns < 0)
               ? 0
               : (ns >= 4294967295.0 ? 4294967295u : static_cast<uint32_t>(ns));
  }

private:
  std::unique_ptr<pelorus::Pelorus> pelorus_;
  std::unique_ptr<CloudCovarianceEstimation> covariance_estimation_;
  std::string generated_rl_path_;

  long next_id_{0};
  Eigen::Isometry3d T_lidar_imu_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d last_T_world_lidar_{Eigen::Isometry3d::Identity()};
  // Set to true once Pelorus returns a scan_id != 1 (i.e., its internal
  // initialization is complete)
  bool pelorus_initialized_{false};

  // Shared State Data
  std::mutex data_mutex_;
  std::condition_variable results_cv_;
  std::uint64_t result_generation_{0};

  // Frames submitted to Pelorus but not yet returned
  std::unordered_map<uint64_t, EstimationFrame::Ptr> pending_frames_;

  // Results returned by Pelorus but not yet picked up by insert_frame
  std::unordered_map<uint64_t, CachedResult> cached_results_;

  // Cache eviction management
  std::deque<uint64_t> cache_order_;
  std::deque<EstimationFrame::ConstPtr> active_frames_;
  std::deque<EstimationFrame::ConstPtr> marginalized_frames_;

  static constexpr size_t kMaxActiveFrames = 10;
  static constexpr size_t kCacheMax = 64;
  static constexpr int kDefaultCovarianceNeighbors = 8;
  static constexpr std::chrono::seconds kResultStallTimeout{5};
};

extern "C" OdometryEstimationBase *create_odometry_estimation_module() {
  return new OdometryEstimationPelorus();
}

} // namespace glim

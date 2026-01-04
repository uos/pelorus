#include <glim/odometry/odometry_estimation_base.hpp>

#include <glim/odometry/estimation_frame.hpp>
#include <glim/odometry/callbacks.hpp>
#include <glim/util/config.hpp>

#include <gtsam_points/types/point_cloud_cpu.hpp>

#include <Eigen/Geometry>
#include <Eigen/Cholesky>

#include <atomic>
#include <mutex>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <deque>

#include <spdlog/spdlog.h>

#include "pelorus.hpp"

namespace glim
{

  class OdometryEstimationPelorus final : public OdometryEstimationBase
  {
  public:
    OdometryEstimationPelorus()
    {
      Config cfg(GlobalConfig::get_config_path("config_odometry"));

      const std::string rl_path = cfg.param<std::string>("odometry_estimation", "pelorus_rl_path", "");
      const bool use_glim_config = cfg.param<bool>("odometry_estimation", "use_glim_config", true);

      // Load sensor extrinsics
      {
        Config sensors_cfg(GlobalConfig::get_config_path("config_sensors"));
        T_lidar_imu_ = sensors_cfg.param<Eigen::Isometry3d>("sensors", "T_lidar_imu", Eigen::Isometry3d::Identity());
      }

      // Generate temporary config if requested and path is empty
      std::string effective_rl_path = rl_path;
      if (use_glim_config && effective_rl_path.empty())
      {
        try
        {
          // GLIM's `T_lidar_imu_` variable is IMU->Lidar per config comment:
          // "T_lidar_imu transforms points in IMU into Lidar"
          // Pelorus expects `from_lidar_transform` as Lidar->IMU.
          // Therefore write the inverse here so Pelorus gets Lidar->IMU.
          const Eigen::Isometry3d T_imu_lidar = T_lidar_imu_.inverse();
          const Eigen::Vector3d t = T_imu_lidar.translation();
          const Eigen::Matrix3d R = T_imu_lidar.linear();

          double preprocess_downsample_resolution_m = 0.5;
          double map_resolution_m = 0.5;
          try
          {
            Config pelorus_cfg(GlobalConfig::get_config_path("config_odometry"));
            preprocess_downsample_resolution_m = pelorus_cfg.param<double>("odometry_estimation", "input_resolution", 0.5);
            map_resolution_m = pelorus_cfg.param<double>("odometry_estimation", "map_resolution", 0.5);
          }
          catch (...)
          {
          }

          if (preprocess_downsample_resolution_m > map_resolution_m)
          {
            map_resolution_m = preprocess_downsample_resolution_m;
          }

          const auto tmp_dir = std::filesystem::temp_directory_path();
          std::ostringstream name;
          name << "pelorus_glim_odometry_glim_config_" << std::hex << reinterpret_cast<std::uintptr_t>(this) << ".rl";
          generated_rl_path_ = (tmp_dir / name.str()).string();

          std::ofstream ofs(generated_rl_path_, std::ios::out | std::ios::trunc);
          if (ofs)
          {
            ofs << std::fixed << std::setprecision(16);
            ofs << "imu {\n";
            ofs << "  from_lidar_transform {\n";
            ofs << "    use_tf_tree = false\n";
            ofs << "    translation = [" << t.x() << "m, " << t.y() << "m, " << t.z() << "m]\n";
            ofs << "    rotation = [ [" << R(0, 0) << ", " << R(0, 1) << ", " << R(0, 2) << "],\n";
            ofs << "                 [" << R(1, 0) << ", " << R(1, 1) << ", " << R(1, 2) << "],\n";
            ofs << "                 [" << R(2, 0) << ", " << R(2, 1) << ", " << R(2, 2) << "] ]\n";
            ofs << "  }\n";
            ofs << "}\n";
            ofs << "lidar.as_sync = true\n";
            if (preprocess_downsample_resolution_m > 0.0)
              ofs << "preprocessing.downsample.resolution = " << preprocess_downsample_resolution_m << "m\n";
            if (map_resolution_m > 0.0)
              ofs << "registration.map.resolution = " << map_resolution_m << "m\n";

            effective_rl_path = generated_rl_path_;
            logger->info("[odometry_estimation_pelorus] input_resolution={}m, map_resolution={}m", preprocess_downsample_resolution_m, map_resolution_m);
          }
        }
        catch (const std::exception &e)
        {
          logger->error("[odometry_estimation_pelorus] Config gen failed: {}", e.what());
        }
      }

      // Initialize Pelorus
      try
      {
        if (effective_rl_path.empty())
        {
          pelorus_ = std::make_unique<pelorus::Pelorus>();
        }
        else
        {
          pelorus_create_options opts = PELORUS_CREATE_OPTIONS_DEFAULT;
          opts.config_path = effective_rl_path.c_str();
          pelorus_ = std::make_unique<pelorus::Pelorus>(&opts);
        }

        // REGISTER CALLBACK
        // We set the callback immediately. Note: 'this' is safe here because the object is constructed.
        // We must unset it in the destructor.
        {
          pelorus_->set_callback([this](const pelorus_odometry *odom_c)
                                 {
            if (odom_c) {
              this->on_odometry_callback(odom_c);
            } });
        }
      }
      catch (const pelorus::Exception &e)
      {
        logger->error("[odometry_estimation_pelorus] Pelorus init failed: {}", e.what());
      }

      last_T_world_lidar_.setIdentity();
    }

    ~OdometryEstimationPelorus() override
    {
      if (pelorus_)
      {
        // Disable callback first to prevent use-after-free of 'this' if pelorus thread is still active
        pelorus_->set_callback(nullptr);
      }
      // generated_rl_path_ could be deleted here if desired, but keeping it for debug is often useful.
    }

    bool requires_imu() const override { return true; }

    void insert_imu(const double stamp, const Eigen::Vector3d &linear_acc, const Eigen::Vector3d &angular_vel) override
    {
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

    EstimationFrame::ConstPtr insert_frame(const PreprocessedFrame::Ptr &frame, std::vector<EstimationFrame::ConstPtr> &marginalized_states) override
    {
      OdometryEstimationBase::insert_frame(frame, marginalized_states);

      if (!pelorus_ || !frame)
      {
        return EstimationFrame::ConstPtr();
      }

      // 1. Prepare Data for Pelorus
      int32_t sec = 0;
      uint32_t nsec = 0;
      stamp_to_sec_nsec(frame->stamp, sec, nsec);

      const size_t n = frame->points.size();
      std::vector<float> xyzit;
      xyzit.resize(n * 5);

      for (size_t i = 0; i < n; i++)
      {
        const Eigen::Vector4d &p = frame->points[i];
        const float intensity = (i < frame->intensities.size()) ? static_cast<float>(frame->intensities[i]) : 0.0f;

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

      uint64_t scan_id = pelorus_->add_points_xyzit(xyzit.data(), n, sec, nsec, "");

      if (scan_id == 0)
      {
        logger->error("[odometry_estimation_pelorus] add_points_xyzit returned an error.");
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
        cloud->add_intensities(frame->intensities.data(), static_cast<int>(frame->intensities.size()));
      if (!frame->times.empty())
        cloud->add_times(frame->times);
      out->frame = cloud;

      {
        std::lock_guard<std::mutex> lock(data_mutex_);

        // Check if the result arrived extremely fast and is already cached
        auto cache_it = cached_results_.find(scan_id);
        if (cache_it != cached_results_.end())
        {
          // HIT: Apply cached result immediately
          apply_result_to_frame(out, cache_it->second);

          cached_results_.erase(cache_it);
          // Cleanup cache order tracker
          for (auto it = cache_order_.begin(); it != cache_order_.end(); ++it)
          {
            if (*it == scan_id)
            {
              cache_order_.erase(it);
              break;
            }
          }

          // Notify listeners immediately.
          OdometryEstimationCallbacks::on_new_frame(out);
          return out;
        }
        else
        {
          // MISS: The callback hasn't arrived yet (normal case). Register as pending.
          pending_frames_[scan_id] = out;
        }
      }

      // Return nullptr: GLIM's AsyncOdometryEstimation will skip null returns (see async_odometry_estimation.cpp:132).
      // We call OdometryEstimationCallbacks::on_new_frame() explicitly when the result is ready in on_odometry_callback().
      return EstimationFrame::ConstPtr();
    }

  private:
    struct CachedResult
    {
      Eigen::Isometry3d T_world_lidar;
      Eigen::Vector3d velocity;
      std::shared_ptr<gtsam_points::PointCloudCPU> cloud;
    };

    // Triggered from background Pelorus thread (C context)
    void on_odometry_callback(const pelorus_odometry *odom_c)
    {
      if (!odom_c)
        return;

      uint64_t scan_id = odom_c->scan_id;
      CachedResult result;

      // Reconstruct Pose from Pelorus
      // NOTE: When lidar_as_map_tf_base=true (default), C library outputs in Lidar frame.
      //       When lidar_as_map_tf_base=false, C library outputs in IMU frame.
      //       Since we set lidar_as_map_tf_base=true, we get Lidar frame directly.
      const double *p = odom_c->position;
      const double *q = odom_c->orientation; // w, x, y, z

      Eigen::Vector3d trans(p[0], p[1], p[2]);
      Eigen::Quaterniond quat(q[0], q[1], q[2], q[3]);
      result.T_world_lidar = Eigen::Isometry3d::Identity();
      result.T_world_lidar.linear() = quat.toRotationMatrix();
      result.T_world_lidar.translation() = trans;

      // result.T_world_lidar = result.T_world_lidar * T_lidar_imu_;

      const double *v = odom_c->linear_velocity;
      result.velocity = Eigen::Vector3d(v[0], v[1], v[2]);

      result.cloud = od_to_cloud_raw(odom_c);

      // Prepare holders for notifications we will perform outside the lock
      EstimationFrame::Ptr frame_to_publish = nullptr;
      std::vector<EstimationFrame::Ptr> frames_to_publish;
      PreprocessedFrame::ConstPtr init_update_frame;
      Eigen::Isometry3d init_update_pose = Eigen::Isometry3d::Identity();
      EstimationFrame::Ptr init_finished_frame;

      {
        std::lock_guard<std::mutex> lock(data_mutex_);

        last_T_world_lidar_ = result.T_world_lidar;

        // If Pelorus hasn't finished its own init yet, it tends to return scan_id == 1 for the pre-init scan.
        // We use that to suppress publishing until Pelorus reports the first non-1 scan_id.
        if (!pelorus_initialized_)
        {
          if (scan_id == 1)
          {
            // Still initializing
            auto pending_it = pending_frames_.find(scan_id);
            if (pending_it != pending_frames_.end() && pending_it->second && pending_it->second->raw_frame)
            {
              init_update_frame = pending_it->second->raw_frame;
              init_update_pose = result.T_world_lidar;
              // Drop the pending frame so it won't be published later
              pending_frames_.erase(pending_it);
            }

            // Ensure we don't keep cached init-phase results either
            if (cached_results_.find(scan_id) != cached_results_.end())
            {
              cached_results_.erase(scan_id);
              // remove from cache_order_ if present
              for (auto it = cache_order_.begin(); it != cache_order_.end(); ++it)
              {
                if (*it == scan_id)
                {
                  cache_order_.erase(it);
                  break;
                }
              }
            }

            // Do not publish while Pelorus is initializing and do not store init-phase frames.
            return;
          }

          // First non-1 scan id => consider Pelorus initialized now.
          pelorus_initialized_ = true;

          // Compose an estimated frame to signal 'finished' to GLIM
          EstimationFrame::Ptr estimated(new EstimationFrame);
          // Prefer using the pending frame's stamp if available
          if (auto pit = pending_frames_.find(scan_id); pit != pending_frames_.end() && pit->second)
          {
            estimated->stamp = pit->second->stamp;
            estimated->raw_frame = pit->second->raw_frame;
          }
          else
          {
            const double stamp = static_cast<double>(odom_c->timestamp_sec) + static_cast<double>(odom_c->timestamp_nanosec) * 1e-9;
            estimated->stamp = stamp;
          }

          estimated->id = -1;
          estimated->T_lidar_imu = T_lidar_imu_;
          estimated->T_world_lidar = result.T_world_lidar;
          estimated->T_world_imu = result.T_world_lidar * T_lidar_imu_;
          estimated->v_world_imu = result.velocity;
          estimated->imu_bias.setZero();
          estimated->frame_id = FrameID::LIDAR;

          init_finished_frame = estimated;

          logger->info("[odometry_estimation_pelorus] Pelorus reported initialization (scan_id={}). Draining cached results.", scan_id);

          // Drain cached results: for any cached result that has a pending frame, apply and publish
          for (const uint64_t id : cache_order_)
          {
            auto cache_it = cached_results_.find(id);
            if (cache_it == cached_results_.end())
              continue;

            auto pending_it2 = pending_frames_.find(id);
            if (pending_it2 != pending_frames_.end())
            {
              EstimationFrame::Ptr f = pending_it2->second;
              apply_result_to_frame(f, cache_it->second);
              frames_to_publish.push_back(f);
              pending_frames_.erase(pending_it2);
              // We'll erase cached results below for processed entries.
            }
          }

          // Remove processed cached entries (those corresponding to frames we published)
          for (auto it = cache_order_.begin(); it != cache_order_.end();)
          {
            if (cached_results_.find(*it) == cached_results_.end())
            {
              it = cache_order_.erase(it);
            }
            else
            {
              // If this id was published above, remove it
              bool published = false;
              for (const auto &pf : frames_to_publish)
              {
                if (pf->raw_frame && pf->raw_frame == pf->raw_frame)
                { /* no-op, keep logic clear */
                }
              }
              // Erase cached entry only if we published a frame for it
              if (std::any_of(frames_to_publish.begin(), frames_to_publish.end(), [&](const EstimationFrame::Ptr &pf)
                              { return pf->raw_frame && cached_results_.count(*it) && (cached_results_[*it].cloud == pf->frame); }))
              {
                cached_results_.erase(*it);
                it = cache_order_.erase(it);
              }
              else
              {
                ++it;
              }
            }
          }

          // Handle current scan's pending frame if present (ensure it's published)
          auto pending_it_curr = pending_frames_.find(scan_id);
          if (pending_it_curr != pending_frames_.end())
          {
            apply_result_to_frame(pending_it_curr->second, result);
            frames_to_publish.push_back(pending_it_curr->second);
            pending_frames_.erase(pending_it_curr);
          }

          // Update last pose again
          last_T_world_lidar_ = result.T_world_lidar;
        }
        else
        {
          // Normal behavior after initialization
          auto pending_it = pending_frames_.find(scan_id);
          if (pending_it != pending_frames_.end())
          {
            frame_to_publish = pending_it->second;
            apply_result_to_frame(frame_to_publish, result);
            pending_frames_.erase(pending_it);
          }
          else
          {
            // No frame waiting yet (unlikely if loop is fast, but possible), or we are ahead.
            // Store in cache for when insert_frame is called.
            cached_results_[scan_id] = result;
            cache_order_.push_back(scan_id);

            // Simple cache eviction
            if (cache_order_.size() > kCacheMax)
            {
              uint64_t old_id = cache_order_.front();
              cache_order_.pop_front();
              cached_results_.erase(old_id);
            }
          }
        }
      }

      // Perform notifications outside the lock
      if (init_update_frame)
      {
        IMUStateInitializationCallbacks::on_updated(init_update_frame, init_update_pose);
      }

      if (init_finished_frame)
      {
        IMUStateInitializationCallbacks::on_finished(init_finished_frame);
      }

      if (frame_to_publish)
      {
        OdometryEstimationCallbacks::on_new_frame(frame_to_publish);
      }

      for (const auto &f : frames_to_publish)
      {
        OdometryEstimationCallbacks::on_new_frame(f);
      }
    }

    // Helper: Apply result struct to frame object
    void apply_result_to_frame(EstimationFrame::Ptr &frame, const CachedResult &res)
    {
      frame->T_world_lidar = res.T_world_lidar;
      frame->T_world_imu = res.T_world_lidar * T_lidar_imu_;
      frame->v_world_imu = res.velocity;
      frame->imu_bias.setZero();

      // If Pelorus returned a registered/deskewed cloud, attach it
      if (res.cloud)
      {
        frame->frame = res.cloud;
      }
    }

    // Helper: Manual conversion from raw C struct to GTSAM cloud
    std::shared_ptr<gtsam_points::PointCloudCPU> od_to_cloud_raw(const pelorus_odometry *odom)
    {
      if (!odom || odom->num_points == 0)
        return nullptr;

      std::vector<Eigen::Vector4d> pts;
      pts.reserve(odom->num_points);
      std::vector<float> ints;
      ints.reserve(odom->num_points);

      const pelorus_point_xyzi *pts_ptr = odom->points;
      for (size_t i = 0; i < odom->num_points; ++i)
      {
        const auto &p = pts_ptr[i];
        pts.emplace_back(p.x, p.y, p.z, 0.0);
        ints.push_back(p.intensity);
      }

      auto pc = std::make_shared<gtsam_points::PointCloudCPU>(pts);
      pc->add_intensities(ints.data(), static_cast<int>(ints.size()));
      return pc;
    }

    static void stamp_to_sec_nsec(const double stamp, int32_t &sec, uint32_t &nsec)
    {
      const double s_floor = std::floor(stamp);
      sec = static_cast<int32_t>(s_floor);
      const double frac = stamp - s_floor;
      const double ns = frac * 1e9;
      nsec = (ns < 0) ? 0 : (ns >= 4294967295.0 ? 4294967295u : static_cast<uint32_t>(ns));
    }

  private:
    std::unique_ptr<pelorus::Pelorus> pelorus_;
    std::string generated_rl_path_;

    long next_id_{0};
    Eigen::Isometry3d T_lidar_imu_{Eigen::Isometry3d::Identity()};
    Eigen::Isometry3d last_T_world_lidar_{Eigen::Isometry3d::Identity()};
    // Set to true once Pelorus returns a scan_id != 1 (i.e., its internal initialization is complete)
    bool pelorus_initialized_{false};

    // Shared State Data
    std::mutex data_mutex_;

    // Frames submitted to Pelorus but not yet returned
    std::unordered_map<uint64_t, EstimationFrame::Ptr> pending_frames_;

    // Results returned by Pelorus but not yet picked up by insert_frame
    std::unordered_map<uint64_t, CachedResult> cached_results_;

    // Cache eviction management
    std::deque<uint64_t> cache_order_;
    static constexpr size_t kCacheMax = 64;
  };

  extern "C" OdometryEstimationBase *create_odometry_estimation_module()
  {
    return new OdometryEstimationPelorus();
  }

} // namespace glim
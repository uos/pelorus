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
#include <thread>
#include <vector>
#include <unordered_map>

#include <spdlog/spdlog.h>

#include "pelorus.hpp"
#include "pelorus_eigen.hpp"
#include <deque>

namespace glim
{

  class OdometryEstimationPelorus final : public OdometryEstimationBase
  {
  public:
    OdometryEstimationPelorus()
    {
      // Reuse GLIM's normal config lookup; user should point global.config_odometry to a file
      // that contains these pelorus-specific params.
      Config cfg(GlobalConfig::get_config_path("config_odometry"));

      // Fallback noise parameters (used if Pelorus covariances are missing or invalid)
      const std::string rl_path = cfg.param<std::string>("odometry_estimation", "pelorus_rl_path", "");
      const bool use_glim_config = cfg.param<bool>("odometry_estimation", "use_glim_config", true);

      // Load sensor extrinsics (used to fill EstimationFrame fields)
      {
        Config sensors_cfg(GlobalConfig::get_config_path("config_sensors"));
        T_lidar_imu_ = sensors_cfg.param<Eigen::Isometry3d>("sensors", "T_lidar_imu", Eigen::Isometry3d::Identity());
      }

      // If requested, generate a minimal Pelorus config override using GLIM configs.
      // This only applies when the user did not provide an explicit pelorus_rl_path.
      std::string effective_rl_path = rl_path;
      if (use_glim_config && effective_rl_path.empty())
      {
        try
        {
          const Eigen::Isometry3d T_imu_lidar = T_lidar_imu_.inverse();
          const Eigen::Vector3d t = T_imu_lidar.translation();
          const Eigen::Matrix3d R = T_imu_lidar.linear();

          // Read downsample resolution
          double preprocess_downsample_resolution_m = 0.25;
          try
          {
            Config preprocess_cfg(GlobalConfig::get_config_path("config_preprocess"));
            preprocess_downsample_resolution_m = preprocess_cfg.param<double>("preprocess", "downsample_resolution", 0.25);
          }
          catch (const std::exception &e)
          {
            logger->warn("[odometry_estimation_pelorus] Failed to read config_preprocess: {}", e.what());
          }

          // Try to get map resolution from pelorus config first (if present)
          double map_resolution_m = 0.0;
          try
          {
            Config pelorus_cfg(GlobalConfig::get_config_path("config_odometry"));
            map_resolution_m = pelorus_cfg.param<double>("odometry_estimation", "map_resolution", 0.0);
          }
          catch (const std::exception &e)
          {
            logger->warn("[odometry_estimation_pelorus] No custom map_resolution in pelorus config: {}", e.what());
          }

          // If not set, try global_mapping/sub_mapping
          if (map_resolution_m <= 0.0)
          {
            try
            {
              Config global_mapping_cfg(GlobalConfig::get_config_path("config_global_mapping"));
              map_resolution_m = global_mapping_cfg.param<double>("global_mapping", "submap_voxel_resolution", 0.0);
            }
            catch (const std::exception &e)
            {
              logger->warn("[odometry_estimation_pelorus] Failed to read config_global_mapping: {}", e.what());
            }
          }
          if (map_resolution_m <= 0.0)
          {
            try
            {
              Config sub_mapping_cfg(GlobalConfig::get_config_path("config_sub_mapping"));
              map_resolution_m = sub_mapping_cfg.param<double>("sub_mapping", "submap_voxel_resolution", 0.0);
            }
            catch (const std::exception &e)
            {
              logger->warn("[odometry_estimation_pelorus] Failed to read config_sub_mapping: {}", e.what());
            }
          }
          // If still not set, use downsample resolution
          if (map_resolution_m <= 0.0)
          {
            map_resolution_m = preprocess_downsample_resolution_m;
          }

          const auto tmp_dir = std::filesystem::temp_directory_path();
          std::ostringstream name;
          name << "pelorus_glim_odometry_glim_config_" << std::hex << reinterpret_cast<std::uintptr_t>(this) << ".rl";
          generated_rl_path_ = (tmp_dir / name.str()).string();

          std::ofstream ofs(generated_rl_path_, std::ios::out | std::ios::trunc);
          if (!ofs)
          {
            logger->warn("[odometry_estimation_pelorus] Failed to open temp config for writing: {}", generated_rl_path_);
          }
          else
          {
            ofs << std::fixed << std::setprecision(16);
            ofs << "# Auto-generated by odometry_estimation_pelorus (use_glim_config=true)\n";
            ofs << "# This file overrides Pelorus defaults with values sourced from GLIM configs.\n";
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
            {
              ofs << "preprocessing.downsample.resolution = " << preprocess_downsample_resolution_m << "m\n";
            }
            if (map_resolution_m > 0.0)
            {
              ofs << "registration.map.resolution = " << map_resolution_m << "m\n";
            }

            effective_rl_path = generated_rl_path_;
            logger->info("[odometry_estimation_pelorus] Using GLIM config.");
          }
        }
        catch (const std::exception &e)
        {
          logger->error("[odometry_estimation_pelorus] Failed to generate GLIM-config Pelorus config: {}", e.what());
        }
      }

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
      }
      catch (const pelorus::Exception &e)
      {
        logger->error("[odometry_estimation_pelorus] Pelorus init failed: {}", e.what());
      }

      last_T_world_lidar_.setIdentity();

      // Start background odometry worker (matches pending frames to incoming odom results)
      start_odom_worker();
    }

    ~OdometryEstimationPelorus() override
    {
      shutting_down_.store(true, std::memory_order_release);
      // Join background odometry worker if running
      if (odom_worker_.joinable())
      {
        odom_worker_.join();
      }
    }

    bool requires_imu() const override { return true; }

    // Start background worker to process incoming odometry and match pending frames
    void start_odom_worker()
    {
      if (!pelorus_)
        return;

      odom_worker_ = std::thread([this]()
                                 {
        while (!shutting_down_.load(std::memory_order_acquire))
        {
          // Use non-blocking try_recv with a short sleep to keep shutdown simple and safe
          pelorus::Odometry od = pelorus_->try_recv();
          if (!od.valid())
          {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
          }

          // Try to complete pending frames first; if none, cache the odometry
          {
            std::lock_guard<std::mutex> lock1(pending_mutex_);
            if (complete_pending_or_cache_locked(od))
            {
              continue;
            }
          }

          // If not handled as pending, add to cache under odom lock
          {
            std::lock_guard<std::mutex> lock(odom_mutex_);
            add_odometry_to_cache_locked(od);
          }
        } });
    }

    void insert_imu(const double stamp, const Eigen::Vector3d &linear_acc, const Eigen::Vector3d &angular_vel) override
    {
      OdometryEstimationBase::insert_imu(stamp, linear_acc, angular_vel);

      if (!pelorus_)
      {
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

    EstimationFrame::ConstPtr insert_frame(const PreprocessedFrame::Ptr &frame, std::vector<EstimationFrame::ConstPtr> &marginalized_states) override
    {
      // Keep GLIM callback semantics (extensions can listen to these regardless of backend).
      OdometryEstimationBase::insert_frame(frame, marginalized_states);

      logger->warn("[odometry_estimation_pelorus] want to insert frame");

      // Drain any currently available odometry into local cache so we can match immediately
      if (pelorus_)
      {
        while (true)
        {
          pelorus::Odometry od = pelorus_->try_recv();
          if (!od.valid())
            break;

          std::lock_guard<std::mutex> lock(odom_mutex_);
          add_odometry_to_cache_locked(od);
        }
      }

      // Feed points into Pelorus (if available). We use the raw preprocessed LiDAR frame.
      uint64_t scan_id = 0;
      if (pelorus_ && frame)
      {
        int32_t sec = 0;
        uint32_t nsec = 0;
        stamp_to_sec_nsec(frame->stamp, sec, nsec);

        const size_t n = static_cast<size_t>(frame->points.size());
        std::vector<float> xyzit;
        xyzit.resize(n * 5);

        for (size_t i = 0; i < n; i++)
        {
          const Eigen::Vector4d &p = frame->points[i];
          const float intensity = (i < frame->intensities.size()) ? static_cast<float>(frame->intensities[i]) : 0.0f;

          double t_sec = (i < frame->times.size()) ? frame->times[i] : 0.0;
          if (t_sec < 0.0)
          {
            t_sec = 0.0;
          }
          const double t_ns = std::min(4294967295.0, t_sec * 1e9);

          xyzit[i * 5 + 0] = static_cast<float>(p.x());
          xyzit[i * 5 + 1] = static_cast<float>(p.y());
          xyzit[i * 5 + 2] = static_cast<float>(p.z());
          xyzit[i * 5 + 3] = intensity;
          xyzit[i * 5 + 4] = static_cast<float>(t_ns);
        }

        scan_id = pelorus_->add_points_xyzit(xyzit.data(), n, sec, nsec, "");
        if (scan_id == 0)
        {
          logger->error("[odometry_estimation_pelorus] pelorus add_points_xyzit returned scan_id=0 (error)");
        }
      }

      // Prepare output frame early so we can attach Pelorus-registered clouds during blocking recv.
      EstimationFrame::Ptr out(new EstimationFrame);
      out->id = next_id_++;
      out->stamp = frame ? frame->stamp : 0.0;
      out->T_lidar_imu = T_lidar_imu_;

      // Get Pelorus odometry result for this scan (blocking semantics via recv/try_recv loops).
      bool got = false;
      Eigen::Vector3d chosen_v = Eigen::Vector3d::Zero();

      if (pelorus_)
      {
        if (scan_id != 0)
        {
          // Check cache first
          {
            std::lock_guard<std::mutex> lock(odom_mutex_);
            auto it = odom_by_scan_.find(scan_id);
            if (it != odom_by_scan_.end())
            {
              last_T_world_lidar_ = it->second;
              auto vit = v_by_scan_.find(scan_id);
              if (vit != v_by_scan_.end())
              {
                const auto &v = vit->second;
                chosen_v << v.x(), v.y(), v.z();
              }
              auto cit = cloud_by_scan_.find(scan_id);
              if (cit != cloud_by_scan_.end())
              {
                out->frame = cit->second;
              }

              // Remove used cache entries
              odom_by_scan_.erase(scan_id);
              v_by_scan_.erase(scan_id);
              cloud_by_scan_.erase(scan_id);
              got = true;
            }
          }

          if (!got)
          {
            // No immediate odometry for this scan: register pending frame and return None now.
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_frames_.emplace(scan_id, out);
          }
        }
        else
        {
          // No scan id: block on blocking recv() until an odometry is available or shutdown
          pelorus::Odometry od = pelorus_->recv();

          if (od.valid())
          {
            const pelorus::eigen::OdometryState state = pelorus::eigen::OdometryState::from_pelorus(od);
            last_T_world_lidar_ = state.transform;
            const auto v = od.linear_velocity();
            chosen_v << v[0], v[1], v[2];
            got = true;

            // If odom includes a registered cloud, use it directly as the output
            if (od.num_points())
            {
              std::vector<Eigen::Vector4d> pts;
              pts.reserve(od.num_points());
              std::vector<float> ints;
              ints.reserve(od.num_points());
              const pelorus_point_xyzi *pts_ptr = od.get() ? od.get()->points : nullptr;
              if (pts_ptr)
              {
                for (size_t i = 0; i < od.num_points(); ++i)
                {
                  const auto &p = pts_ptr[i];
                  pts.emplace_back(static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z), 0.0);
                  ints.push_back(p.intensity);
                }
                auto pc = std::make_shared<gtsam_points::PointCloudCPU>(pts);
                pc->add_intensities(ints.data(), static_cast<int>(ints.size()));
                out->frame = pc;
              }
            }
          }
          else
          {
            logger->error("[odometry_estimation_pelorus] pelorus recv() returned no odometry (error or shutdown)");
          }
        }
      }

      // Fill point data: GLIM mapping expects a non-null `frame` for keyframing/overlap.
      // We use the incoming frame points (LiDAR frame) as the deskewed cloud baseline.
      if (frame)
      {
        out->raw_frame = frame;
        out->frame_id = FrameID::LIDAR;
        auto cloud = std::make_shared<gtsam_points::PointCloudCPU>(frame->points);

        // Intensity channel is optional (and PointCloudCPU requires pointer+count).
        if (!frame->intensities.empty() && frame->intensities.size() == frame->points.size())
        {
          cloud->add_intensities(frame->intensities.data(), static_cast<int>(frame->intensities.size()));
        }

        // Preserve point times so downstream modules (deskewing/visualization) can use them
        if (!frame->times.empty())
        {
          cloud->add_times(frame->times);
        }

        out->frame = cloud;

        // If pelorus produced a registered/undistorted cloud earlier via recv()/try_recv(),
        // it may already have been set into out->frame above. The recv() branch sets out->frame directly.
      }
      else
      {
        out->frame_id = FrameID::LIDAR;
      }

      // If we didn't receive any odometry in this call, avoid emitting a zeroed-out frame immediately.
      // Instead, if the frame was registered as pending, we return a null pointer so that the
      // async pipeline does not process a spurious placeholder. The background worker will
      // call `on_new_frame` when the actual odometry arrives and completes the pending frame.
      if (!got)
      {
        // If the frame was registered as pending (present in pending_frames_), do NOT emit it now.
        bool was_pending = false;
        if (scan_id != 0)
        {
          int32_t sec = 0; // not used here; keep for clarity
          uint32_t nsec = 0;
          // We can reconstruct the header timestamp used above if needed, but presence in the map
          // is sufficient to know it's pending.
          std::lock_guard<std::mutex> lock(pending_mutex_);
          // Find by scan id presence in pending_frames_ (older code used scan_id key)
          auto pit = pending_frames_.find(scan_id);
          if (pit != pending_frames_.end())
          {
            was_pending = true;
          }
        }

        if (was_pending)
        {
          // Do not produce a frame now; the worker will call on_new_frame when odometry arrives.
          return EstimationFrame::ConstPtr();
        }

        // Fallback: use last known pose so callers get something (for scan_id == 0 or non-pending)
        out->T_world_lidar = last_T_world_lidar_;
        out->T_world_imu = last_T_world_lidar_ * T_lidar_imu_;
        out->v_world_imu << 0.0, 0.0, 0.0;
        out->imu_bias.setZero();
      }
      else
      {
        out->T_world_lidar = last_T_world_lidar_;
        out->T_world_imu = last_T_world_lidar_ * T_lidar_imu_;
        out->v_world_imu << chosen_v.x(), chosen_v.y(), chosen_v.z();
        out->imu_bias.setZero();

        // Notify listeners (viewer, mapping, etc.) that a new frame is available
        OdometryEstimationCallbacks::on_new_frame(out);
      }

      return out;
    }

  private:
    std::string generated_rl_path_;

    static void stamp_to_sec_nsec(const double stamp, int32_t &sec, uint32_t &nsec)
    {
      const double s_floor = std::floor(stamp);
      sec = static_cast<int32_t>(s_floor);
      const double frac = stamp - s_floor;
      const double ns = frac * 1e9;
      if (ns <= 0.0)
      {
        nsec = 0;
      }
      else if (ns >= 4294967295.0)
      {
        nsec = 4294967295u;
      }
      else
      {
        nsec = static_cast<uint32_t>(ns);
      }
    }

    // Helper: Convert pelorus odometry cloud to PointCloudCPU (returns nullptr if no points)
    std::shared_ptr<gtsam_points::PointCloudCPU> od_to_cloud(const pelorus::Odometry &od)
    {
      if (!od.num_points())
        return nullptr;
      std::vector<Eigen::Vector4d> pts;
      pts.reserve(od.num_points());
      std::vector<float> ints;
      ints.reserve(od.num_points());
      const pelorus_point_xyzi *pts_ptr = od.get() ? od.get()->points : nullptr;
      if (!pts_ptr)
        return nullptr;
      for (size_t i = 0; i < od.num_points(); ++i)
      {
        const auto &p = pts_ptr[i];
        pts.emplace_back(static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z), 0.0);
        ints.push_back(p.intensity);
      }
      auto pc = std::make_shared<gtsam_points::PointCloudCPU>(pts);
      pc->add_intensities(ints.data(), static_cast<int>(ints.size()));
      return pc;
    }

    // Cache helpers (caller must hold odom_mutex_ for _locked variants)
    static constexpr size_t kCacheMax = 64;
    std::deque<uint64_t> cache_order_;

    void trim_cache_locked()
    {
      while (cache_order_.size() > kCacheMax)
      {
        uint64_t oldest = cache_order_.front();
        cache_order_.pop_front();
        odom_by_scan_.erase(oldest);
        v_by_scan_.erase(oldest);
        cloud_by_scan_.erase(oldest);
      }
    }

    void add_odometry_to_cache_locked(const pelorus::Odometry &od)
    {
      const uint64_t sid = od.scan_id();
      const pelorus::eigen::OdometryState state = pelorus::eigen::OdometryState::from_pelorus(od);
      odom_by_scan_[sid] = state.transform;
      const auto v = od.linear_velocity();
      v_by_scan_[sid] = Eigen::Vector3d(v[0], v[1], v[2]);

      auto pc = od_to_cloud(od);
      if (pc)
      {
        cloud_by_scan_[sid] = pc;
      }

      // Maintain insertion order for eviction
      // Remove any existing entry for sid in cache_order_
      for (auto it = cache_order_.begin(); it != cache_order_.end(); ++it)
      {
        if (*it == sid)
        {
          cache_order_.erase(it);
          break;
        }
      }
      cache_order_.push_back(sid);
      trim_cache_locked();
    }

    // Called when we get an odom from the worker: if there's a pending frame, complete it and return true.
    bool complete_pending_or_cache_locked(const pelorus::Odometry &od)
    {
      const uint64_t sid = od.scan_id();

      // We may need to emit a pending frame. Find it while holding the pending mutex,
      // but do the actual callback outside locks to avoid reentrancy and races.
      EstimationFrame::Ptr to_emit;
      {
        std::lock_guard<std::mutex> pend_lock(pending_mutex_);
        auto pit = pending_frames_.find(sid);
        if (pit != pending_frames_.end())
        {
          to_emit = pit->second;

          const pelorus::eigen::OdometryState state = pelorus::eigen::OdometryState::from_pelorus(od);
          to_emit->T_world_lidar = state.transform;
          to_emit->T_world_imu = state.transform * T_lidar_imu_;
          const auto v = od.linear_velocity();
          to_emit->v_world_imu << v[0], v[1], v[2];

          auto pc = od_to_cloud(od);
          if (pc)
          {
            to_emit->frame = pc;
            logger->info("[odometry_estimation_pelorus] Using Pelorus registered cloud for scan={} (worker)", sid);
          }

          pending_frames_.erase(pit);
        }
      }

      if (to_emit)
      {
        // Call callbacks outside of any internal locks to avoid reentrant issues.
        OdometryEstimationCallbacks::on_new_frame(to_emit);
        return true;
      }

      // No pending frame — add to cache
      add_odometry_to_cache_locked(od);
      return false;
    }

  private:
    std::unique_ptr<pelorus::Pelorus> pelorus_;

    long next_id_{0};

    Eigen::Isometry3d T_lidar_imu_{Eigen::Isometry3d::Identity()};
    Eigen::Isometry3d last_T_world_lidar_{Eigen::Isometry3d::Identity()};

    // Async odometry storage (updated from Pelorus callback)
    std::mutex odom_mutex_;
    Eigen::Vector3d last_v_world_{Eigen::Vector3d::Zero()};
    std::atomic<bool> have_odom_{false};

    // If true, we're shutting down and blocking waits should abort
    std::atomic<bool> shutting_down_{false};

    // Background worker for blocking odometry recv and pending-frame matching
    std::thread odom_worker_;

    // Map scan_id -> odometry state (pose + velocity)
    std::unordered_map<uint64_t, Eigen::Isometry3d> odom_by_scan_;
    std::unordered_map<uint64_t, Eigen::Vector3d> v_by_scan_;
    std::unordered_map<uint64_t, gtsam_points::PointCloudCPU::Ptr> cloud_by_scan_;

    // Pending frames waiting for their odometry result
    std::mutex pending_mutex_;
    std::unordered_map<uint64_t, EstimationFrame::Ptr> pending_frames_;
  };

  extern "C" OdometryEstimationBase *create_odometry_estimation_module()
  {
    return new OdometryEstimationPelorus();
  }

} // namespace glim

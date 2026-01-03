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
#include "pelorus_eigen.hpp"

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
          const Eigen::Isometry3d T_imu_lidar = T_lidar_imu_.inverse();
          const Eigen::Vector3d t = T_imu_lidar.translation();
          const Eigen::Matrix3d R = T_imu_lidar.linear();

          double preprocess_downsample_resolution_m = 0.25;
          try
          {
            Config preprocess_cfg(GlobalConfig::get_config_path("config_preprocess"));
            preprocess_downsample_resolution_m = preprocess_cfg.param<double>("preprocess", "downsample_resolution", 0.25);
          }
          catch (...)
          {
          }

          double map_resolution_m = 0.0;
          try
          {
            Config pelorus_cfg(GlobalConfig::get_config_path("config_odometry"));
            map_resolution_m = pelorus_cfg.param<double>("odometry_estimation", "map_resolution", 0.0);
          }
          catch (...)
          {
          }

          if (map_resolution_m <= 0.0)
          {
            try
            {
              Config global_mapping_cfg(GlobalConfig::get_config_path("config_global_mapping"));
              map_resolution_m = global_mapping_cfg.param<double>("global_mapping", "submap_voxel_resolution", 0.0);
            }
            catch (...)
            {
            }
          }

          if (map_resolution_m <= 0.0)
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
            logger->info("[odometry_estimation_pelorus] Using GLIM config override at: {}", generated_rl_path_);
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
        pelorus_->set_callback([this](const pelorus_odometry *odom_c)
                               {
          if (odom_c) {
            this->on_odometry_callback(odom_c);
          } });
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
        return create_fallback_frame(frame);
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

      // 2. Send Points to Pelorus -> Get Unique Scan ID
      // This call is non-blocking regarding the odometry calculation, but it accepts the data.
      uint64_t scan_id = pelorus_->add_points_xyzit(xyzit.data(), n, sec, nsec, "");

      if (scan_id == 0)
      {
        logger->error("[odometry_estimation_pelorus] add_points_xyzit returned 0");
        return create_fallback_frame(frame);
      }

      // 3. Prepare the Output Frame Wrapper
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

      // 4. Critical Section: Check Cache OR Register Pending
      // The callback could potentially fire immediately after add_points_xyzit.
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

      // Return nullptr: This signals to the GLIM pipeline that we are working asynchronously.
      // The actual frame will be pushed via on_new_frame() inside the callback later.
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

      // Reconstruct Pose
      const double *p = odom_c->position;
      const double *q = odom_c->orientation; // w, x, y, z

      Eigen::Vector3d trans(p[0], p[1], p[2]);
      Eigen::Quaterniond quat(q[0], q[1], q[2], q[3]);
      result.T_world_lidar = Eigen::Isometry3d::Identity();
      result.T_world_lidar.linear() = quat.toRotationMatrix();
      result.T_world_lidar.translation() = trans;

      const double *v = odom_c->linear_velocity;
      result.velocity = Eigen::Vector3d(v[0], v[1], v[2]);

      // Extract Cloud manually
      result.cloud = od_to_cloud_raw(odom_c);

      EstimationFrame::Ptr frame_to_publish = nullptr;

      // 2. Critical Section
      {
        std::lock_guard<std::mutex> lock(data_mutex_);

        // Update last known pose (useful for fallbacks)
        last_T_world_lidar_ = result.T_world_lidar;

        auto pending_it = pending_frames_.find(scan_id);
        if (pending_it != pending_frames_.end())
        {
          // Found a waiting frame
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

      // 3. Notify Listener
      // We do this outside the lock to avoid deadlocks if the callback re-enters anything.
      if (frame_to_publish)
      {
        OdometryEstimationCallbacks::on_new_frame(frame_to_publish);
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

    EstimationFrame::ConstPtr create_fallback_frame(const PreprocessedFrame::Ptr &frame)
    {
      EstimationFrame::Ptr out(new EstimationFrame);
      out->id = next_id_++;
      out->stamp = frame ? frame->stamp : 0.0;
      out->T_lidar_imu = T_lidar_imu_;

      out->T_world_lidar = last_T_world_lidar_;
      out->T_world_imu = last_T_world_lidar_ * T_lidar_imu_;
      out->v_world_imu.setZero();
      out->imu_bias.setZero();
      out->frame_id = FrameID::LIDAR;

      if (frame)
      {
        out->raw_frame = frame;
        auto cloud = std::make_shared<gtsam_points::PointCloudCPU>(frame->points);
        if (!frame->intensities.empty())
          cloud->add_intensities(frame->intensities.data(), static_cast<int>(frame->intensities.size()));
        out->frame = cloud;
      }

      OdometryEstimationCallbacks::on_new_frame(out);
      return out;
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
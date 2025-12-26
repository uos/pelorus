#include <glim/odometry/odometry_estimation_base.hpp>

#include <glim/odometry/estimation_frame.hpp>
#include <glim/util/config.hpp>

#include <gtsam_points/types/point_cloud_cpu.hpp>

#include <Eigen/Geometry>
#include <Eigen/Cholesky>

#include <atomic>
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
      // Reuse GLIM's normal config lookup; user should point global.config_odometry to a file
      // that contains these pelorus-specific params.
      Config cfg(GlobalConfig::get_config_path("config_odometry"));

      max_wait_ms_ = cfg.param<int>("odometry_estimation", "max_wait_ms", 200);
      // Fallback noise parameters (used if Pelorus covariances are missing or invalid)
      between_sigma_trans_ = cfg.param<double>("odometry_estimation", "between_sigma_trans", 0.2);
      between_sigma_rot_rad_ = cfg.param<double>("odometry_estimation", "between_sigma_rot_rad", 0.15);
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
          // GLIM's T_lidar_imu is IMU->LiDAR (see config_sensors.json docs). Pelorus expects LiDAR->IMU.
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
            spdlog::debug("[odometry_estimation_pelorus] Failed to read config_preprocess: {}", e.what());
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
            spdlog::debug("[odometry_estimation_pelorus] No custom map_resolution in pelorus config: {}", e.what());
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
              spdlog::debug("[odometry_estimation_pelorus] Failed to read config_global_mapping: {}", e.what());
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
              spdlog::debug("[odometry_estimation_pelorus] Failed to read config_sub_mapping: {}", e.what());
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
            spdlog::warn("[odometry_estimation_pelorus] Failed to open temp config for writing: {}", generated_rl_path_);
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

            if (preprocess_downsample_resolution_m > 0.0)
            {
              ofs << "preprocessing.downsample.resolution = " << preprocess_downsample_resolution_m << "m\n";
            }
            if (map_resolution_m > 0.0)
            {
              ofs << "registration.map.resolution = " << map_resolution_m << "m\n";
            }

            effective_rl_path = generated_rl_path_;
            spdlog::info("[odometry_estimation_pelorus] Using GLIM config.");
          }
        }
        catch (const std::exception &e)
        {
          spdlog::warn("[odometry_estimation_pelorus] Failed to generate GLIM-config Pelorus config: {}", e.what());
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
        spdlog::error("[odometry_estimation_pelorus] Pelorus init failed: {}", e.what());
      }

      last_T_world_lidar_.setIdentity();
    }

    ~OdometryEstimationPelorus() override = default;

    bool requires_imu() const override { return true; }

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

      // Feed points into Pelorus (if available). We use the raw preprocessed LiDAR frame.
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

        pelorus_->add_points_xyzit(xyzit.data(), n, sec, nsec, "");
      }

      // Get Pelorus odometry result for this scan.
      bool got = false;
      // We'll use a blocking recv() like the frontend, then drain remaining outputs via try_recv().
      if (pelorus_)
      {
        pelorus::Odometry first = pelorus_->recv();
        if (first.valid())
        {
          got = true;
          // Process first result and drain remaining outputs
          // We'll keep the most recent valid odometry (last seen) as the output for this frame.
          auto process_odom = [&](const pelorus::Odometry &odom_in, Eigen::Isometry3d &out_T, Eigen::Vector3d &out_v)
          {
            const pelorus::eigen::OdometryState state = pelorus::eigen::OdometryState::from_pelorus(odom_in);
            out_T = state.transform;
            const auto v = odom_in.linear_velocity();
            out_v << v[0], v[1], v[2];
          };

          // Use the latest processed odometry as the output
          Eigen::Isometry3d latest_T = last_T_world_lidar_;
          Eigen::Vector3d latest_v = Eigen::Vector3d::Zero();
          process_odom(first, latest_T, latest_v);

          while (true)
          {
            pelorus::Odometry od = pelorus_->try_recv();
            if (!od.valid())
              break;
            process_odom(od, latest_T, latest_v);
          }

          // Set the chosen values as the output
          last_T_world_lidar_ = latest_T;
          out->v_world_imu << latest_v.x(), latest_v.y(), latest_v.z();
        }
      }

      EstimationFrame::Ptr out(new EstimationFrame);
      out->id = next_id_++;
      out->stamp = frame ? frame->stamp : 0.0;
      out->T_lidar_imu = T_lidar_imu_;

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

        out->frame = cloud;
      }
      else
      {
        out->frame_id = FrameID::LIDAR;
      }

      // If we didn't receive any odometry in this call, use last known transforms/velocity.
      if (!got)
      {
        out->T_world_lidar = last_T_world_lidar_;
        out->T_world_imu = last_T_world_lidar_ * T_lidar_imu_;
        out->v_world_imu << 0.0, 0.0, 0.0;
        out->imu_bias.setZero();
      }
      else
      {
        out->T_world_lidar = last_T_world_lidar_;
        out->T_world_imu = last_T_world_lidar_ * T_lidar_imu_;
        // velocity already set above into out->v_world_imu when processing odometry
        out->imu_bias.setZero();
      }

      // Note: EstimationFrame currently has no explicit covariance field; we keep the latest Pelorus
      // pose covariance internally for use by consumers that query this module or for debugging.

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

  private:
    std::unique_ptr<pelorus::Pelorus> pelorus_;

    int max_wait_ms_{200};
    long next_id_{0};

    Eigen::Isometry3d T_lidar_imu_{Eigen::Isometry3d::Identity()};
    Eigen::Isometry3d last_T_world_lidar_{Eigen::Isometry3d::Identity()};
    // Fallback noise parameters (read from config)
    double between_sigma_trans_{0.2};
    double between_sigma_rot_rad_{0.15};
  };

  extern "C" OdometryEstimationBase *create_odometry_estimation_module()
  {
    return new OdometryEstimationPelorus();
  }

} // namespace glim

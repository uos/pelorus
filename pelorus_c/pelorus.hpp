/**
 * @file pelorus.hpp
 * @brief C++ convenience wrapper for Pelorus C library
 *
 * Provides modern C++ wrapper classes and idioms for easier integration
 * with C++ projects.
 */

#ifndef PELORUS_HPP
#define PELORUS_HPP

#include "pelorus.h"
#include <memory>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>

namespace pelorus
{

    /**
     * @brief Exception thrown when Pelorus operations fail
     */
    class Exception : public std::runtime_error
    {
    public:
        explicit Exception(const std::string &message)
            : std::runtime_error(message) {}
    };

    /**
     * @brief Wrapper for pelorus_imu_msg
     */
    struct ImuMsg
    {
        int32_t timestamp_sec;
        uint32_t timestamp_nanosec;
        std::string frame_id;
        double orientation[4] = {1.0, 0.0, 0.0, 0.0}; // w, x, y, z
        double orientation_covariance[9] = {0};
        double angular_velocity[3] = {0};
        double angular_velocity_covariance[9] = {0};
        double linear_acceleration[3] = {0};
        double linear_acceleration_covariance[9] = {0};

        /**
         * @brief Convert to C struct for API calls
         */
        pelorus_imu_msg to_c() const
        {
            pelorus_imu_msg msg;
            msg.timestamp_sec = timestamp_sec;
            msg.timestamp_nanosec = timestamp_nanosec;
            msg.frame_id = frame_id.c_str();
            std::memcpy(msg.orientation, orientation, sizeof(orientation));
            std::memcpy(msg.orientation_covariance, orientation_covariance, sizeof(orientation_covariance));
            std::memcpy(msg.angular_velocity, angular_velocity, sizeof(angular_velocity));
            std::memcpy(msg.angular_velocity_covariance, angular_velocity_covariance, sizeof(angular_velocity_covariance));
            std::memcpy(msg.linear_acceleration, linear_acceleration, sizeof(linear_acceleration));
            std::memcpy(msg.linear_acceleration_covariance, linear_acceleration_covariance, sizeof(linear_acceleration_covariance));
            return msg;
        }
    };

    /**
     * @brief Wrapper for pelorus_odom_msg
     */
    struct OdomMsg
    {
        int32_t timestamp_sec;
        uint32_t timestamp_nanosec;
        double twist_linear[3] = {0};  // Linear velocity [vx, vy, vz]
        double twist_angular[3] = {0}; // Angular velocity [wx, wy, wz]
        double twist_covariance[36] = {0};

        /**
         * @brief Convert to C struct for API calls
         */
        pelorus_odom_msg to_c() const
        {
            pelorus_odom_msg msg;
            msg.timestamp_sec = timestamp_sec;
            msg.timestamp_nanosec = timestamp_nanosec;
            std::memcpy(msg.twist_linear, twist_linear, sizeof(twist_linear));
            std::memcpy(msg.twist_angular, twist_angular, sizeof(twist_angular));
            std::memcpy(msg.twist_covariance, twist_covariance, sizeof(twist_covariance));
            return msg;
        }
    };

    /**
     * @brief Wrapper for pelorus_odometry result
     */
    class Odometry
    {
    public:
        Odometry() : valid_(false)
        {
            std::memset(&odom_, 0, sizeof(odom_));
        }

        explicit Odometry(const pelorus_odometry &odom) : odom_(odom), valid_(true) {}

        ~Odometry()
        {
            if (valid_)
            {
                pelorus_free_odometry(&odom_);
            }
        }

        // Move semantics
        Odometry(Odometry &&other) noexcept : odom_(other.odom_), valid_(other.valid_)
        {
            other.valid_ = false;
            std::memset(&other.odom_, 0, sizeof(other.odom_));
        }

        Odometry &operator=(Odometry &&other) noexcept
        {
            if (this != &other)
            {
                if (valid_)
                {
                    pelorus_free_odometry(&odom_);
                }
                odom_ = other.odom_;
                valid_ = other.valid_;
                other.valid_ = false;
                std::memset(&other.odom_, 0, sizeof(other.odom_));
            }
            return *this;
        }

        // Disable copy
        Odometry(const Odometry &) = delete;
        Odometry &operator=(const Odometry &) = delete;

        /**
         * @brief Check if odometry data is valid
         */
        bool valid() const { return valid_; }

        /**
         * @brief Get raw pointer (use with caution)
         */
        const pelorus_odometry *get() const { return valid_ ? &odom_ : nullptr; }

        /**
         * @brief Access timestamp (seconds)
         */
        int32_t timestamp_sec() const { return valid_ ? odom_.timestamp_sec : 0; }

        /**
         * @brief Access timestamp (nanoseconds)
         */
        uint32_t timestamp_nanosec() const { return valid_ ? odom_.timestamp_nanosec : 0; }

        /**
         * @brief Access position
         */
        const double *position() const { return valid_ ? odom_.position : nullptr; }

        /**
         * @brief Access orientation (quaternion: w, x, y, z)
         */
        const double *orientation() const { return valid_ ? odom_.orientation : nullptr; }

        /**
         * @brief Access linear velocity
         */
        const double *linear_velocity() const { return valid_ ? odom_.linear_velocity : nullptr; }

        /**
         * @brief Access angular velocity
         */
        const double *angular_velocity() const { return valid_ ? odom_.angular_velocity : nullptr; }

        /**
         * @brief Get number of points in the cloud
         */
        size_t num_points() const { return valid_ ? odom_.num_points : 0; }

        /**
         * @brief Access point cloud
         */
        const pelorus_point_xyzi *points() const { return valid_ ? odom_.points : nullptr; }

        /**
         * @brief Get point cloud as PCL format (requires pelorus_pcl.hpp)
         * This is a convenience method that extracts just the point cloud.
         * For full PCL integration, include pelorus_pcl.hpp after pelorus.hpp.
         *
         * Usage:
         *   auto cloud = odom.get_cloud_pcl<pcl::PointCloud<pcl::PointXYZI>::Ptr>();
         */
        template <typename PointCloudPtr>
        PointCloudPtr get_cloud_pcl() const;

        // Note: Eigen3 transformation methods are available as free functions in pelorus_eigen.hpp
        // Usage:
        //   #include <pelorus_eigen.hpp>
        //   auto transform = pelorus::eigen::to_transform(odom);
        //   auto state = pelorus::eigen::OdometryState::from_pelorus(odom);

    private:
        pelorus_odometry odom_;
        bool valid_;
    };

    /**
     * @brief Modern C++ wrapper for PelorusHandle
     */
    class Pelorus
    {
    public:
        /**
         * @brief Callback function type for asynchronous odometry
         */
        using OdometryCallback = std::function<void(const pelorus_odometry *)>;

        /**
         * @brief Create Pelorus instance with default configuration
         */
        Pelorus() : Pelorus(static_cast<const char *>(nullptr)) {}

        /**
         * @brief Create Pelorus instance with custom configuration file
         * @param config_path Path to .rl configuration file (nullptr for default)
         */
        explicit Pelorus(const char *config_path)
            : handle_(pelorus_create(config_path), pelorus_destroy), callback_wrapper_(nullptr)
        {
            if (!handle_)
            {
                throw Exception("Failed to create Pelorus handle");
            }
        }

        /**
         * @brief Create Pelorus instance with options
         * @param opts Creation options
         */
        explicit Pelorus(const pelorus_create_options *opts)
            : handle_(pelorus_create_with_options(opts), pelorus_destroy), callback_wrapper_(nullptr)
        {
            if (!handle_)
            {
                throw Exception("Failed to create Pelorus handle with options");
            }
        }

        /**
         * @brief Get library version
         */
        static std::string version()
        {
            return pelorus_version();
        }

        /**
         * @brief Add IMU measurement
         */
        void add_imu(const ImuMsg &imu)
        {
            auto c_msg = imu.to_c();
            pelorus_add_imu(handle_.get(), &c_msg);
        }

        /**
         * @brief Add odometry measurement
         */
        void add_odom(const OdomMsg &odom)
        {
            auto c_msg = odom.to_c();
            pelorus_add_odom(handle_.get(), &c_msg);
        }

        /**
         * @brief Add point cloud (PointCloud2 format)
         */
        uint64_t add_pointcloud(const pelorus_pointcloud2 &cloud)
        {
            return pelorus_add_pointcloud(handle_.get(), &cloud);
        }

        /**
         * @brief Add raw XYZT points (N x 4 float array)
         * @param points Array of [x, y, z, t] values
         * @param num_points Number of points
         * @param timestamp_sec Header timestamp seconds
         * @param timestamp_nanosec Header timestamp nanoseconds
         * @param frame_id Frame ID string
         */
        uint64_t add_points_xyzt(const float *points, size_t num_points,
                                 int32_t timestamp_sec, uint32_t timestamp_nanosec,
                                 const char *frame_id)
        {
            return pelorus_add_points_xyzt_f32(handle_.get(), timestamp_sec, timestamp_nanosec,
                                               frame_id, points, num_points);
        }

        /**
         * @brief Add raw XYZIT points (N x 5 float array)
         * @param points Array of [x, y, z, intensity, t] values
         * @param num_points Number of points
         * @param timestamp_sec Header timestamp seconds
         * @param timestamp_nanosec Header timestamp nanoseconds
         * @param frame_id Frame ID string
         */
        uint64_t add_points_xyzit(const float *points, size_t num_points,
                                  int32_t timestamp_sec, uint32_t timestamp_nanosec,
                                  const char *frame_id)
        {
            return pelorus_add_points_xyzit_f32(handle_.get(), timestamp_sec, timestamp_nanosec,
                                                frame_id, points, num_points);
        }

        /**
         * @brief Try to receive odometry (non-blocking)
         * @return Odometry object (check .valid() for availability)
         */
        Odometry try_recv()
        {
            pelorus_odometry odom;
            std::memset(&odom, 0, sizeof(odom));
            int status = pelorus_try_recv(handle_.get(), &odom);
            if (status == 1)
            {
                return Odometry(odom);
            }
            return Odometry();
        }

        /**
         * @brief Receive odometry (blocking)
         * @return Odometry object (check .valid() for success)
         */
        Odometry recv()
        {
            pelorus_odometry odom;
            std::memset(&odom, 0, sizeof(odom));
            int status = pelorus_recv(handle_.get(), &odom);
            if (status == 1)
            {
                return Odometry(odom);
            }
            return Odometry();
        }

        /**
         * @brief Set callback for asynchronous odometry
         * @param callback Function to call when odometry is available (nullptr to disable)
         */
        void set_callback(OdometryCallback callback)
        {
            if (callback)
            {
                // Store the callback
                user_callback_ = callback;

                // Create a wrapper that bridges C and C++
                callback_wrapper_ = [](const pelorus_odometry *odom, void *user_data)
                {
                    auto *self = static_cast<Pelorus *>(user_data);
                    if (self->user_callback_)
                    {
                        self->user_callback_(odom);
                    }
                };

                pelorus_set_callback(handle_.get(), callback_wrapper_, this);
            }
            else
            {
                // Disable callback
                user_callback_ = nullptr;
                callback_wrapper_ = nullptr;
                pelorus_set_callback(handle_.get(), nullptr, nullptr);
            }
        }

    private:
        std::unique_ptr<PelorusHandle, decltype(&pelorus_destroy)> handle_;
        OdometryCallback user_callback_;
        void (*callback_wrapper_)(const pelorus_odometry *, void *);
    };

} // namespace pelorus

#endif // PELORUS_HPP

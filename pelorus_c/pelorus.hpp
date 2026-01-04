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

// Feature detection: enable optional helpers when dependencies are available
#if defined(__has_include)
#if __has_include(<Eigen/Core>)
#define PELORUS_HAVE_EIGEN 1
#endif
#if __has_include(<pcl/point_cloud.h>) && __has_include(<pcl/point_types.h>)
#define PELORUS_HAVE_PCL 1
#endif
#endif

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
         * @brief Access scan id corresponding to the input pointcloud
         */
        uint64_t scan_id() const { return valid_ ? odom_.scan_id : 0; }

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
         * @brief Status codes mirror the C `pelorus_status` enum.
         */
        enum PelorusStatus
        {
            STATUS_OK = 0,
            STATUS_NO_DATA = 1,
            STATUS_ERROR = -1,
            STATUS_CALLBACK_SET = -2,
            STATUS_DISCONNECTED = -3,
        };

        /**
         * @brief Result for recv/try_recv containing a status and (optional) Odometry.
         */
        struct RecvResult
        {
            int status;
            Odometry odom;
        };

        /**
         * @brief Try to receive odometry (non-blocking)
         * @return RecvResult: check `.status` and use `.odom` when `.status == STATUS_OK`.
         */
        RecvResult try_recv()
        {
            pelorus_odometry odom_raw;
            std::memset(&odom_raw, 0, sizeof(odom_raw));
            int status = pelorus_try_recv(handle_.get(), &odom_raw);
            if (status == STATUS_OK)
            {
                return RecvResult{status, Odometry(odom_raw)};
            }
            return RecvResult{status, Odometry()};
        }

        /**
         * @brief Receive odometry (blocking)
         * @return RecvResult: check `.status` and use `.odom` when `.status == STATUS_OK`.
         */
        RecvResult recv()
        {
            pelorus_odometry odom_raw;
            std::memset(&odom_raw, 0, sizeof(odom_raw));
            int status = pelorus_recv(handle_.get(), &odom_raw);
            if (status == STATUS_OK)
            {
                return RecvResult{status, Odometry(odom_raw)};
            }
            return RecvResult{status, Odometry()};
        }

        /**
         * @brief Receive odometry for a specific scan id (blocking)
         * @param scan_id The scan id to wait for
         * @return Odometry object (check .valid() for success)
         */
        /**
         * @brief Set callback for asynchronous odometry
         * @param callback Function to call when odometry is available (nullptr to disable)
         */
        /**
         * @brief Set callback for asynchronous odometry
         * @param callback Function to call when odometry is available (nullptr to disable)
         * @return status code (0 on success, negative on error)
         */
        int set_callback(OdometryCallback callback)
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

                return pelorus_set_callback(handle_.get(), callback_wrapper_, this);
            }
            else
            {
                // Disable callback
                user_callback_ = nullptr;
                callback_wrapper_ = nullptr;
                return pelorus_set_callback(handle_.get(), nullptr, nullptr);
            }
        }

    private:
        std::unique_ptr<PelorusHandle, decltype(&pelorus_destroy)> handle_;
        OdometryCallback user_callback_;
        void (*callback_wrapper_)(const pelorus_odometry *, void *);
    };

    // Optional helpers (Eigen / PCL) moved below so includes are at global scope
    // This avoids including external library headers while inside `namespace pelorus` which
    // can cause accidental name lookup collisions with third-party headers.

} // namespace pelorus

// Optional helpers: enable only when dependencies are present
#if defined(PELORUS_HAVE_EIGEN) && PELORUS_HAVE_EIGEN
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace pelorus
{
    namespace eigen
    {

        inline Eigen::Quaterniond to_quaternion(const double *quat)
        {
            return Eigen::Quaterniond(quat[0], quat[1], quat[2], quat[3]);
        }

        inline Eigen::Vector3d to_position(const double *pos)
        {
            return Eigen::Vector3d(pos[0], pos[1], pos[2]);
        }

        inline Eigen::Vector3d to_linear_velocity(const double *vel)
        {
            return Eigen::Vector3d(vel[0], vel[1], vel[2]);
        }

        inline Eigen::Vector3d to_angular_velocity(const double *vel)
        {
            return Eigen::Vector3d(vel[0], vel[1], vel[2]);
        }

        inline Eigen::Isometry3d to_transform(const Odometry &odom)
        {
            Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
            if (!odom.valid())
            {
                return transform;
            }
            transform.translation() = to_position(odom.position());
            transform.linear() = to_quaternion(odom.orientation()).toRotationMatrix();
            return transform;
        }

        inline Eigen::Affine3d to_affine(const Odometry &odom)
        {
            Eigen::Affine3d transform = Eigen::Affine3d::Identity();
            if (!odom.valid())
            {
                return transform;
            }
            transform.translation() = to_position(odom.position());
            transform.linear() = to_quaternion(odom.orientation()).toRotationMatrix();
            return transform;
        }

        inline Eigen::Matrix4d to_matrix(const Odometry &odom)
        {
            return to_transform(odom).matrix();
        }

        inline Eigen::Matrix<double, 6, 6> get_pose_covariance(const pelorus_odometry &odom)
        {
            Eigen::Matrix<double, 6, 6> cov;
            for (int i = 0; i < 6; i++)
            {
                for (int j = 0; j < 6; j++)
                {
                    cov(i, j) = odom.pose_covariance[i * 6 + j];
                }
            }
            return cov;
        }

        inline Eigen::Matrix<double, 6, 6> get_twist_covariance(const pelorus_odometry &odom)
        {
            Eigen::Matrix<double, 6, 6> cov;
            for (int i = 0; i < 6; i++)
            {
                for (int j = 0; j < 6; j++)
                {
                    cov(i, j) = odom.twist_covariance[i * 6 + j];
                }
            }
            return cov;
        }

        struct OdometryState
        {
            Eigen::Isometry3d transform;           ///< Pose as transformation matrix
            Eigen::Vector3d linear_velocity;       ///< Linear velocity
            Eigen::Vector3d angular_velocity;      ///< Angular velocity
            Eigen::Matrix<double, 6, 6> pose_cov;  ///< Pose covariance
            Eigen::Matrix<double, 6, 6> twist_cov; ///< Twist covariance
            int32_t timestamp_sec;                 ///< Timestamp seconds
            uint32_t timestamp_nanosec;            ///< Timestamp nanoseconds

            static OdometryState from_pelorus(const Odometry &odom)
            {
                OdometryState state;
                if (!odom.valid())
                {
                    state.transform = Eigen::Isometry3d::Identity();
                    state.linear_velocity = Eigen::Vector3d::Zero();
                    state.angular_velocity = Eigen::Vector3d::Zero();
                    state.pose_cov = Eigen::Matrix<double, 6, 6>::Zero();
                    state.twist_cov = Eigen::Matrix<double, 6, 6>::Zero();
                    state.timestamp_sec = 0;
                    state.timestamp_nanosec = 0;
                    return state;
                }
                state.transform = to_transform(odom);
                state.linear_velocity = to_linear_velocity(odom.linear_velocity());
                state.angular_velocity = to_angular_velocity(odom.angular_velocity());
                state.pose_cov = get_pose_covariance(*odom.get());
                state.twist_cov = get_twist_covariance(*odom.get());
                state.timestamp_sec = odom.timestamp_sec();
                state.timestamp_nanosec = odom.timestamp_nanosec();
                return state;
            }
        };

    } // namespace eigen
} // namespace pelorus
#endif // PELORUS_HAVE_EIGEN

#if defined(PELORUS_HAVE_PCL) && PELORUS_HAVE_PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcl
{
    struct EIGEN_ALIGN16 PointXYZIT
    {
        PCL_ADD_POINT4D;
        float intensity;
        uint32_t t;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}

POINT_CLOUD_REGISTER_POINT_STRUCT(pcl::PointXYZIT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(uint32_t, t, t))

namespace pelorus
{
    namespace pcl_helpers
    {
        inline float timestamp_to_float(uint32_t t_ns)
        {
            return static_cast<float>(t_ns);
        }

        inline ::pcl::PointCloud<::pcl::PointXYZI>::Ptr to_pcl_cloud(
            const pelorus_point_xyzi *points, size_t num_points)
        {
            auto cloud = ::pcl::PointCloud<::pcl::PointXYZI>::Ptr(
                new ::pcl::PointCloud<::pcl::PointXYZI>());

            if (!points || num_points == 0)
            {
                return cloud;
            }

            cloud->points.resize(num_points);
            cloud->width = num_points;
            cloud->height = 1;
            cloud->is_dense = false;

            for (size_t i = 0; i < num_points; i++)
            {
                cloud->points[i].x = points[i].x;
                cloud->points[i].y = points[i].y;
                cloud->points[i].z = points[i].z;
                cloud->points[i].intensity = points[i].intensity;
            }

            return cloud;
        }

        inline void add_pointcloud(Pelorus &pelorus,
                                   const ::pcl::PointCloud<::pcl::PointXYZIT>::Ptr &cloud,
                                   int32_t timestamp_sec, uint32_t timestamp_nanosec,
                                   const char *frame_id)
        {
            std::vector<float> points;
            points.reserve(cloud->size() * 5);
            for (size_t i = 0; i < cloud->size(); i++)
            {
                const auto &pt = cloud->points[i];
                points.push_back(pt.x);
                points.push_back(pt.y);
                points.push_back(pt.z);
                points.push_back(pt.intensity);
                points.push_back(timestamp_to_float(pt.t));
            }

            pelorus.add_points_xyzit(points.data(), cloud->size(),
                                     timestamp_sec, timestamp_nanosec, frame_id);
        }

    } // namespace pcl_helpers

    // Expose a friendly `pelorus::pcl` namespace with the types and helper functions
    namespace pcl
    {
        using PointXYZIT = ::pcl::PointXYZIT;

        inline void add_pointcloud(Pelorus &pelorus, const ::pcl::PointCloud<::pcl::PointXYZIT>::Ptr &cloud,
                                   int32_t timestamp_sec, uint32_t timestamp_nanosec, const char *frame_id)
        {
            pcl_helpers::add_pointcloud(pelorus, cloud, timestamp_sec, timestamp_nanosec, frame_id);
        }

    } // namespace pcl

    template <typename PointCloudPtr>
    PointCloudPtr Odometry::get_cloud_pcl() const
    {
        if (!valid_)
        {
            return PointCloudPtr(new typename PointCloudPtr::element_type());
        }
        return pcl_helpers::to_pcl_cloud(odom_.points, odom_.num_points);
    }

} // namespace pelorus
#endif // PELORUS_HAVE_PCL

#endif // PELORUS_HPP

#pragma once

#include "common.h"
#include "vectorn.h"
#include "vector3.h"
#include "matrix3x3.h"

namespace rmagine
{

    template <typename T>
    T degree_to_radian(T degree)
    {
        return degree * T(M_PI) / T(180);
    }

    template <typename T>
    T radian_to_degree(T radian)
    {
        return radian * T(180) / T(M_PI);
    }

    template <typename T>
    class Quaternion // TODO can currently not be reinterpret casted
    {
    public:
        T w, x, y, z;

        RMAGINE_INLINE_FUNCTION
        Quaternion() : w(1.0), x(0.0), y(0.0), z(0.0) {}

        RMAGINE_INLINE_FUNCTION
        Quaternion(T w_, T x_, T y_, T z_) : w(w_), x(x_), y(y_), z(z_) {}

        RMAGINE_INLINE_FUNCTION
        Quaternion<T> operator*(const Quaternion<T> &other) const
        {
            return Quaternion<T>(
                w * other.y - x * other.z + y * other.w + z * other.x,
                w * other.z + x * other.y - y * other.x + z * other.w,
                w * other.w - x * other.x - y * other.y - z * other.z,
                w * other.x + x * other.w + y * other.z - z * other.y);
        }

        RMAGINE_INLINE_FUNCTION
        Vector3<T> operator*(const Vector3<T> &v) const
        {
            Vector3<T> qv(x, y, z);
            Vector3<T> t = 2 * qv.cross(v);
            return v + w * t + qv.cross(t);
        }

        RMAGINE_INLINE_FUNCTION
        Quaternion<T> conjugate() const
        {
            return Quaternion<T>(w, -x, -y, -z);
        }

        RMAGINE_INLINE_FUNCTION
        double norm() const
        {
            return std::sqrt(w * w + x * x + y * y + z * z);
        }

        RMAGINE_INLINE_FUNCTION
        Quaternion normalized() const
        {
            double n = 1.0 / this->norm();
            return Quaternion<T>(w * n, x * n, y * n, z * n);
        }

        RMAGINE_INLINE_FUNCTION
        Quaternion inverse() const
        {
            double n = this->norm();
            n = n * n;
            return Quaternion<T>(w / n, -x / n, -y / n, -z / n);
        }

        RMAGINE_INLINE_FUNCTION
        void normalize()
        {
            double n = 1.0 / this->norm();
            w *= n;
            x *= n;
            y *= n;
            z *= n;
        }

        RMAGINE_INLINE_FUNCTION
        Matrix3x3d toRotationMatrix() const
        {
            Matrix3x3d rotation_matrix;
            rotation_matrix(0, 0) = w * w + x * x - y * y - z * z;
            rotation_matrix(0, 1) = 2 * x * y - 2 * w * z;
            rotation_matrix(0, 2) = 2 * x * z + 2 * w * y;

            rotation_matrix(1, 0) = 2 * x * y + 2 * w * z;
            rotation_matrix(1, 1) = w * w - x * x + y * y - z * z;
            rotation_matrix(1, 2) = 2 * y * z - 2 * w * x;

            rotation_matrix(2, 0) = 2 * x * z - 2 * w * y;
            rotation_matrix(2, 1) = 2 * y * z + 2 * w * x;
            rotation_matrix(2, 2) = w * w - x * x - y * y + z * z;

            return rotation_matrix;
        }

        static Quaternion<T> fromAxisAngle(const Vector3<T> &axis, T angle)
        {
            Vector3<T> normalized_axis = axis.normalized();

            T half_angle = angle / T(2);
            T sin_half_angle = std::sin(half_angle);

            return Quaternion<T>(
                std::cos(half_angle),               // w
                normalized_axis.x * sin_half_angle, // x
                normalized_axis.y * sin_half_angle, // y
                normalized_axis.z * sin_half_angle  // z
            );
        }

        static Quaternion<T> fromEuler(const Vector3<T> &euler)
        {
            T half_roll = euler.x * T(0.5);
            T half_pitch = euler.y * T(0.5);
            T half_yaw = euler.z * T(0.5);

            T cos_roll = std::cos(half_roll);
            T sin_roll = std::sin(half_roll);
            T cos_pitch = std::cos(half_pitch);
            T sin_pitch = std::sin(half_pitch);
            T cos_yaw = std::cos(half_yaw);
            T sin_yaw = std::sin(half_yaw);

            return Quaternion<T>(
                cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw, // w
                sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw, // x
                cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw, // y
                cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw  // z
            );
        }
    };

    using Quaterniond = Quaternion<double>;
    using Quaternionf = Quaternion<float>;

} // namespace rmagine
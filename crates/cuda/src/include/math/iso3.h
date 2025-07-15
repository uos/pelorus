#pragma once

#include "common.h"
#include "vector3.h"
#include "quaternion.h"

namespace rmagine
{
    template <typename T>
    class Isometry3
    {
    public:
        Quaternion<T> rotation;
        Vector3<T> translation;

        RMAGINE_INLINE_FUNCTION
        Isometry3<T>() : rotation(Quaternion<T>()), translation(Vector3<T>()) {}

        RMAGINE_INLINE_FUNCTION
        Isometry3<T>(const Quaternion<T> &rot, const Vector3<T> &trans) : rotation(rot), translation(trans) {}

        RMAGINE_INLINE_FUNCTION
        Isometry3<T>(const Isometry3<T> &other) : Isometry3<T>(other.rotation, other.translation) {}

        RMAGINE_INLINE_FUNCTION
        Isometry3<T> operator*(const Isometry3<T> &other) const
        {
            Vector3<T> nt = (rotation * other.translation) + translation;
            Quaternion<T> nr = rotation * other.rotation;
            nr.normalize();
            return Isometry3<T>(nr, nt);
        }

        RMAGINE_INLINE_FUNCTION
        Vector3<T> operator*(const Vector3<T> &point) const
        {
            return rotation * point + translation;
        }

        RMAGINE_INLINE_FUNCTION
        Isometry3<T> inverse() const
        {
            Quaternion<T> inv_rot = rotation.inverse();
            inv_rot.normalize();
            Vector3<T> inv_trans = (inv_rot * translation) * -1;
            return Isometry3<T>(inv_rot, inv_trans);
        }

        /**
         * Local transformation in the perspective of this global Isometry3 to the other global Isometry3
         */
        RMAGINE_INLINE_FUNCTION
        Isometry3<T> diff_to(const Isometry3<T> &other) const
        {
            // other.inverse() * (*this);
            //  Inverse of the current rotation
            Quaternion<T> inv_rotation = this->rotation.inverse();

            // Transform the other translation into the local coordinate frame of `this`
            Vector3<T> local_translation = inv_rotation * (other.translation - this->translation);

            // Transform the other rotation into the local coordinate frame of `this`
            Quaternion<T> local_rotation = inv_rotation * other.rotation;

            // Return the new Isometry3 representing the local transformation
            return Isometry3<T>{local_rotation, local_translation};
        }

        /*
                    auto me_conj = rotation.conjugate();
            auto local_rot = me_conj * other.rotation;
            auto local_trans = me_conj * (other.translation - translation);
            return Isometry3<T>(local_rot, local_trans);
            */

        RMAGINE_INLINE_FUNCTION
        static const Isometry3<T> Identity()
        {
            return Isometry3<T>();
        }
    };

    using Isometry3d = Isometry3<double>;
    using Isometry3f = Isometry3<float>;

} // namespace rmagine
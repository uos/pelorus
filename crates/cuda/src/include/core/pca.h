#pragma once

#include <core/math/math.h>

RMAGINE_INLINE_FUNCTION
rmagine::Vector3d calc_centroid(const rmagine::Vector3d *cluster, size_t cluster_size)
{
    rmagine::Vector3d sum(0, 0, 0);
    for (size_t i = 0; i < cluster_size; i++)
    {
        sum.addInplace(cluster[i]);
    }

    return sum * (1.0 / static_cast<double>(cluster_size));
}

RMAGINE_INLINE_FUNCTION
const rmagine::Vector3d calc_centroid(const rmagine::Vector3d &n1,
                                      const rmagine::Vector3d &n2,
                                      const rmagine::Vector3d &n3,
                                      const rmagine::Vector3d &n4,
                                      const rmagine::Vector3d &n5)
{
    return (n1 + n2 + n3 + n4 + n5) * 0.2;
}

RMAGINE_INLINE_FUNCTION
rmagine::Matrix3x3d calc_cov(const rmagine::Vector3d &n1,
                             const rmagine::Vector3d &n2,
                             const rmagine::Vector3d &n3,
                             const rmagine::Vector3d &n4,
                             const rmagine::Vector3d &n5, const rmagine::Vector3d &centroid)
{
    rmagine::Matrix3x3d cov;
    rmagine::Vector3d r = n1 - centroid;
    double xx = r.x * r.x;
    double xy = r.x * r.y;
    double xz = r.x * r.z;
    double yy = r.y * r.y;
    double yz = r.y * r.z;
    double zz = r.z * r.z;

    r = n2 - centroid;
    xx += r.x * r.x;
    xy += r.x * r.y;
    xz += r.x * r.z;
    yy += r.y * r.y;
    yz += r.y * r.z;
    zz += r.z * r.z;

    r = n3 - centroid;
    xx += r.x * r.x;
    xy += r.x * r.y;
    xz += r.x * r.z;
    yy += r.y * r.y;
    yz += r.y * r.z;
    zz += r.z * r.z;

    r = n4 - centroid;
    xx += r.x * r.x;
    xy += r.x * r.y;
    xz += r.x * r.z;
    yy += r.y * r.y;
    yz += r.y * r.z;
    zz += r.z * r.z;

    r = n5 - centroid;
    xx += r.x * r.x;
    xy += r.x * r.y;
    xz += r.x * r.z;
    yy += r.y * r.y;
    yz += r.y * r.z;
    zz += r.z * r.z;

    xx *= 0.2;
    xy *= 0.2;
    xz *= 0.2;
    yy *= 0.2;
    yz *= 0.2;
    zz *= 0.2;

    cov(0, 0) = xx;
    cov(0, 1) = xy;
    cov(0, 2) = xz;
    cov(1, 0) = xy;
    cov(1, 1) = yy;
    cov(1, 2) = yz;
    cov(2, 0) = xz;
    cov(2, 1) = yz;
    cov(2, 2) = zz;

    return std::move(cov);
}

RMAGINE_INLINE_FUNCTION
rmagine::Matrix3x3d calc_cov(const rmagine::Vector3d *cluster, size_t cluster_size, const rmagine::Vector3d &centroid)
{
    double xx = 0.0, xy = 0.0, xz = 0.0, yy = 0.0, yz = 0.0, zz = 0.0;

    for (size_t i = 0; i < cluster_size; i++)
    {
        rmagine::Vector3d r = cluster[i] - centroid;
        xx += r.x * r.x;
        xy += r.x * r.y;
        xz += r.x * r.z;
        yy += r.y * r.y;
        yz += r.y * r.z;
        zz += r.z * r.z;
    }

    xx /= static_cast<double>(cluster_size);
    xy /= static_cast<double>(cluster_size);
    xz /= static_cast<double>(cluster_size);
    yy /= static_cast<double>(cluster_size);
    yz /= static_cast<double>(cluster_size);
    zz /= static_cast<double>(cluster_size);

    rmagine::Matrix3x3d cov;
    cov(0, 0) = xx;
    cov(0, 1) = xy;
    cov(0, 2) = xz;
    cov(1, 0) = xy;
    cov(1, 1) = yy;
    cov(1, 2) = yz;
    cov(2, 0) = xz;
    cov(2, 1) = yz;
    cov(2, 2) = zz;

    return std::move(cov);
}

/*RMAGINE_INLINE_FUNCTION
bool pca_constant(const rmagine::Vector3d &neighbor1,
                  const rmagine::Vector3d &neighbor2,
                  const rmagine::Vector3d &neighbor3,
                  const rmagine::Vector3d &neighbor4,
                  const rmagine::Vector3d &neighbor5, float normal_threshold, rmagine::VectorN<4, double> &plane_equation)
{
    rmagine::Vector3d centroid = calc_centroid(neighbor1, neighbor2, neighbor3, neighbor4, neighbor5);
    rmagine::Matrix3x3d cov = calc_cov(neighbor1, neighbor2, neighbor3, neighbor4, neighbor5, centroid);

    double xx = cov(0, 0);
    double xy = cov(0, 1);
    double xz = cov(0, 2);
    double yy = cov(1, 1);
    double yz = cov(1, 2);
    double zz = cov(2, 2);

    rmagine::Vector3d weighted_dir(0, 0, 0);
    rmagine::Vector3d axis_dir;
    float weight;
    float det_x, det_y, det_z;

    bool n1_invalid = false;
    bool n2_invalid = false;
    bool n3_invalid = false;
    bool n4_invalid = false;
    bool n5_invalid = false;

    det_x = yy * zz - yz * yz;
    axis_dir = {det_x, xz * yz - xy * zz, xy * yz - xz * yy};
    weight = det_x * det_x;
    if (weighted_dir.dot(axis_dir) < 0.0)
    {
        weight = -weight;
    }

    weighted_dir += axis_dir * weight;

    det_y = xx * zz - xz * xz;
    axis_dir = {xz * yz - xy * zz, det_y, xy * xz - yz * xx};
    weight = det_y * det_y;
    if (weighted_dir.dot(axis_dir) < 0.0)
    {
        weight = -weight;
    }

    weighted_dir += axis_dir * weight;

    det_z = xx * yy - xy * xy;
    axis_dir = {xy * yz - xz * yy, xy * xz - yz * xx, det_z};
    weight = det_z * det_z;
    if (weighted_dir.dot(axis_dir) < 0.0)
    {
        weight = -weight;
    }

    weighted_dir += axis_dir * weight;

    // auto len = weighted_dir.l2norm();
    weighted_dir.normalize();
    // rm::Vector3d normalized_rm = weighted_dir.normalized();

    double bias = -(weighted_dir.x * centroid.x + weighted_dir.y * centroid.y + weighted_dir.z * centroid.z);

    plane_equation(0) = weighted_dir.x;
    plane_equation(1) = weighted_dir.y;
    plane_equation(2) = weighted_dir.z;
    plane_equation(3) = bias;

    // if any of the points is too far from the plane, the result is invalid
    n1_invalid = (plane_equation(0) * neighbor1.x + plane_equation(1) * neighbor1.y + plane_equation(2) * neighbor1.z + plane_equation(3)) > normal_threshold;
    n2_invalid = (plane_equation(0) * neighbor2.x + plane_equation(1) * neighbor2.y + plane_equation(2) * neighbor2.z + plane_equation(3)) > normal_threshold;
    n3_invalid = (plane_equation(0) * neighbor3.x + plane_equation(1) * neighbor3.y + plane_equation(2) * neighbor3.z + plane_equation(3)) > normal_threshold;
    n4_invalid = (plane_equation(0) * neighbor4.x + plane_equation(1) * neighbor4.y + plane_equation(2) * neighbor4.z + plane_equation(3)) > normal_threshold;
    n5_invalid = (plane_equation(0) * neighbor5.x + plane_equation(1) * neighbor5.y + plane_equation(2) * neighbor5.z + plane_equation(3)) > normal_threshold;

    return !(n1_invalid || n2_invalid || n3_invalid || n4_invalid || n5_invalid);
}*/

RMAGINE_INLINE_FUNCTION double plane_normal_point_distance(const rmagine::VectorN<4, double> &plane, const rmagine::Vector3d &point)
{
    return plane(0) * point.x + plane(1) * point.y + plane(2) * point.z + plane(3);
}

RMAGINE_INLINE_FUNCTION rmagine::VectorN<4, double> plane_from_point_and_normal(const rmagine::Vector3d &point, const rmagine::Vector3d &normal)
{
    return {normal.x, normal.y, normal.z, -(normal.x * point.x + normal.y * point.y + normal.z * point.z)};
}

// https://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html
RMAGINE_INLINE_FUNCTION
bool pca_constant(const rmagine::Vector3d *cluster, size_t n, float normal_threshold, rmagine::VectorN<4, double> &plane_equation)
{
    if (n < 3)
    {
        return false;
    }

    rmagine::Vector3d centroid = calc_centroid(cluster, n);
    rmagine::Matrix3x3d cov = calc_cov(cluster, n, centroid);

    double xx = cov(0, 0);
    double xy = cov(0, 1);
    double xz = cov(0, 2);
    double yy = cov(1, 1);
    double yz = cov(1, 2);
    double zz = cov(2, 2);

    rmagine::Vector3d weighted_dir(0, 0, 0);
    rmagine::Vector3d axis_dir;
    float weight;
    float det_x, det_y, det_z;

    det_x = yy * zz - yz * yz;
    axis_dir = {det_x, xz * yz - xy * zz, xy * yz - xz * yy};
    weight = det_x * det_x;
    if (weighted_dir.dot(axis_dir) < 0.0)
    {
        weight = -weight;
    }

    weighted_dir += axis_dir * weight;

    det_y = xx * zz - xz * xz;
    axis_dir = {xz * yz - xy * zz, det_y, xy * xz - yz * xx};
    weight = det_y * det_y;
    if (weighted_dir.dot(axis_dir) < 0.0)
    {
        weight = -weight;
    }

    weighted_dir += axis_dir * weight;

    det_z = xx * yy - xy * xy;
    axis_dir = {xy * yz - xz * yy, xy * xz - yz * xx, det_z};
    weight = det_z * det_z;
    if (weighted_dir.dot(axis_dir) < 0.0)
    {
        weight = -weight;
    }

    weighted_dir += axis_dir * weight;

    weighted_dir.normalize();
    plane_equation = plane_from_point_and_normal(centroid, weighted_dir);

    // if any of the points is too far from the plane, return false
    for (size_t i = 0; i < n; i++)
    {
        if (fabs(plane_normal_point_distance(plane_equation, cluster[i])) > normal_threshold)
        {
            return false;
        }
    }

    return true;
}

/*RMAGINE_INLINE_FUNCTION
bool pca_custom_iterative(
    const rmagine::Vector3d &neighbor1,
    const rmagine::Vector3d &neighbor2,
    const rmagine::Vector3d &neighbor3,
    const rmagine::Vector3d &neighbor4,
    const rmagine::Vector3d &neighbor5, float normal_threshold, rmagine::Vector3d &normal, rmagine::Quaterniond &plane_equation)
{
    rmagine::Vector3d cog = calc_centroid(neighbor1, neighbor2, neighbor3, neighbor4, neighbor5);
    rmagine::Matrix3x3d K = calc_cov(neighbor1, neighbor2, neighbor3, neighbor4, neighbor5, cog);

    double A[3][3] = {{K(0, 0), K(0, 1), K(0, 2)}, {K(1, 0), K(1, 1), K(1, 2)}, {K(2, 0), K(2, 1), K(2, 2)}};

    // jacobi eigen solver
    // init eigen vector matrix as identity
    double eigenvectors[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    double eigenvalues[3] = {0, 0, 0};

    int n = 3;

    // Compute the off-diagonal norm of A
    double offdiag_norm = 0.0;
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            offdiag_norm += A[i][j] * A[i][j];
        }
    }

    // Compute the eigenvalues and eigenvectors
    int max_iter = 1000;
    for (int iter = 0; iter < max_iter; iter++)
    {
        // Find the largest off-diagonal element
        int p = 0;
        int q = 1;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (fabs(A[i][j]) > fabs(A[p][q]))
                {
                    p = i;
                    q = j;
                }
            }
        }

        // Check for convergence
        double threshold = 1e-8;
        if (offdiag_norm < threshold)
        {
            break;
        }

        // Compute the Jacobi rotation angle
        double theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        double t = 1.0 / (fabs(theta) + sqrt(theta * theta + 1.0));
        if (theta < 0)
        {
            t = -t;
        }

        // Compute the Jacobi rotation matrix
        double c = 1.0 / sqrt(t * t + 1.0);
        double s = t * c;

        // Apply the Jacobi rotation
        double A_pq = A[p][q];
        A[p][q] = 0.0;
        A[q][p] = 0.0;
        A[p][p] -= t * A_pq;
        A[q][q] += t * A_pq;
        for (int r = 0; r < n; r++)
        {
            if (r != p && r != q)
            {
                double A_pr = A[p][r];
                double A_qr = A[q][r];
                A[p][r] = c * A_pr - s * A_qr;
                A[r][p] = A[p][r];
                A[q][r] = c * A_qr + s * A_pr;
                A[r][q] = A[q][r];
            }

            // Update the eigenvectors
            double eigenvector_pr = eigenvectors[p][r];
            double eigenvector_qr = eigenvectors[q][r];
            eigenvectors[p][r] = c * eigenvector_pr - s * eigenvector_qr;
            eigenvectors[q][r] = c * eigenvector_qr + s * eigenvector_pr;
        }

        // Update the off-diagonal norm
        offdiag_norm -= A_pq * A_pq;
    }

    // Sort the eigenvalues and eigenvectors
    for (int i = 0; i < n; i++)
    {
        eigenvalues[i] = A[i][i];
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            if (eigenvalues[i] > eigenvalues[j])
            {
                double tmp = eigenvalues[i];
                eigenvalues[i] = eigenvalues[j];
                eigenvalues[j] = tmp;
                for (int k = 0; k < n; k++)
                {
                    double tmp = eigenvectors[k][i];
                    eigenvectors[k][i] = eigenvectors[k][j];
                    eigenvectors[k][j] = tmp;
                }
            }
        }
    }

    // eigen vector with smallest eigen value is the normal
    double minEigenV[3] = {eigenvectors[0][0], eigenvectors[1][0], eigenvectors[2][0]};

    // normalize
    double norm = sqrt(minEigenV[0] * minEigenV[0] + minEigenV[1] * minEigenV[1] + minEigenV[2] * minEigenV[2]);
    minEigenV[0] /= norm;
    minEigenV[1] /= norm;
    minEigenV[2] /= norm;

    double d = -(minEigenV[0] * cog.x + minEigenV[1] * cog.y + minEigenV[2] * cog.z);

    plane_equation.x = minEigenV[0];
    plane_equation.y = minEigenV[1];
    plane_equation.z = minEigenV[2];
    plane_equation.w = d;

    normal.x = (float)minEigenV[0];
    normal.y = (float)minEigenV[1];
    normal.z = (float)minEigenV[2];

    rm::Vector3d cluster[] = {neighbor1, neighbor2, neighbor3, neighbor4, neighbor5};

    for (size_t i = 0; i < 5; i++)
    {
        double x = minEigenV[0] * cluster[i].x;
        double y = minEigenV[1] * cluster[i].y;
        double z = minEigenV[2] * cluster[i].z;

        double absolute = fabs(x + y + z);
        if (absolute > normal_threshold)
        {
            return false;
        }
    }

    return true;
}*/

/**
 * Plane Normal vector estimation using PCA
 *
 */
RMAGINE_INLINE_FUNCTION
bool pca_custom_iterative(const rmagine::Vector3d *cluster, size_t cluster_size, float normal_threshold, rmagine::VectorN<4, double> &plane_equation)
{
    if (cluster_size < 3)
    {
        return false;
    }

    rmagine::Vector3d cog = calc_centroid(cluster, cluster_size);
    rmagine::Matrix3x3d K = calc_cov(cluster, cluster_size, cog);

    double A[3][3] = {{K(0, 0), K(0, 1), K(0, 2)}, {K(1, 0), K(1, 1), K(1, 2)}, {K(2, 0), K(2, 1), K(2, 2)}};

    // jacobi eigen solver
    // init eigen vector matrix as identity
    double eigenvectors[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    double eigenvalues[3] = {0, 0, 0};

    int n = 3;

    // Compute the off-diagonal norm of A
    double offdiag_norm = 0.0;
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            offdiag_norm += A[i][j] * A[i][j];
        }
    }

    // Compute the eigenvalues and eigenvectors
    int max_iter = 1000;
    for (int iter = 0; iter < max_iter; iter++)
    {
        // Find the largest off-diagonal element
        int p = 0;
        int q = 1;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (fabs(A[i][j]) > fabs(A[p][q]))
                {
                    p = i;
                    q = j;
                }
            }
        }

        // Check for convergence
        double threshold = 1e-8;
        if (offdiag_norm < threshold)
        {
            break;
        }

        // Compute the Jacobi rotation angle
        double theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        double t = 1.0 / (fabs(theta) + sqrt(theta * theta + 1.0));
        if (theta < 0)
        {
            t = -t;
        }

        // Compute the Jacobi rotation matrix
        double c = 1.0 / sqrt(t * t + 1.0);
        double s = t * c;

        // Apply the Jacobi rotation
        double A_pq = A[p][q];
        A[p][q] = 0.0;
        A[q][p] = 0.0;
        A[p][p] -= t * A_pq;
        A[q][q] += t * A_pq;
        for (int r = 0; r < n; r++)
        {
            if (r != p && r != q)
            {
                double A_pr = A[p][r];
                double A_qr = A[q][r];
                A[p][r] = c * A_pr - s * A_qr;
                A[r][p] = A[p][r];
                A[q][r] = c * A_qr + s * A_pr;
                A[r][q] = A[q][r];
            }

            // Update the eigenvectors
            double eigenvector_pr = eigenvectors[p][r];
            double eigenvector_qr = eigenvectors[q][r];
            eigenvectors[p][r] = c * eigenvector_pr - s * eigenvector_qr;
            eigenvectors[q][r] = c * eigenvector_qr + s * eigenvector_pr;
        }

        // Update the off-diagonal norm
        offdiag_norm -= A_pq * A_pq;
    }

    // Sort the eigenvalues and eigenvectors
    for (int i = 0; i < n; i++)
    {
        eigenvalues[i] = A[i][i];
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            if (eigenvalues[i] > eigenvalues[j])
            {
                double tmp = eigenvalues[i];
                eigenvalues[i] = eigenvalues[j];
                eigenvalues[j] = tmp;
                for (int k = 0; k < n; k++)
                {
                    double tmp = eigenvectors[k][i];
                    eigenvectors[k][i] = eigenvectors[k][j];
                    eigenvectors[k][j] = tmp;
                }
            }
        }
    }

    // eigen vector with smallest eigen value is the normal
    double minEigenV[3] = {eigenvectors[0][0], eigenvectors[1][0], eigenvectors[2][0]};
    rm::Vector3d eigen_v(minEigenV[0], minEigenV[1], minEigenV[2]);

    auto len = eigen_v.l2norm();
    plane_equation = plane_from_point_and_normal(cog, eigen_v.normalized());

    for (size_t i = 0; i < n; i++)
    {
        if (fabs(plane_normal_point_distance(plane_equation, cluster[i])) > normal_threshold)
        {
            return false;
        }
    }

    // plane_equation(3) = 1.0 / len;

    return true;
}

// TODO [corr] y1 test fails, normal exists but is quite off. maybe because of missing iterations?
RMAGINE_INLINE_FUNCTION
bool pca_supervoxel(const rmagine::Vector3d *cluster, size_t cluster_size, float normal_threshold, rmagine::VectorN<4, double> &plane_equation)
{
    double cx = 0, cy = 0, cz = 0;
    for (int i = 0; i < cluster_size; i++)
    {
        cx += cluster[i].x;
        cy += cluster[i].y;
        cz += cluster[i].z;
    }
    cx /= cluster_size;
    cy /= cluster_size;
    cz /= cluster_size;

    double a00 = 0.0, a01 = 0.0, a02 = 0.0, a11 = 0.0, a12 = 0.0, a22 = 0.0;
    double sum = 0.0;
    for (int i = 0; i < cluster_size; i++)
    {
        double x = cluster[i].x - cx;
        double y = cluster[i].y - cy;
        double z = cluster[i].z - cz;
        const double w = 1.0;

        a00 += w * x * x;
        a01 += w * x * y;
        a02 += w * x * z;
        a11 += w * y * y;
        a12 += w * y * z;
        a22 += w * z * z;

        sum += w;
    }

    double t = 1.0 / sum;
    a00 = a00 * t;
    a01 = a01 * t;
    a02 = a02 * t;
    a11 = a11 * t;
    a12 = a12 * t;
    a22 = a22 * t;

    double q = (a00 + a11 + a22) / 3.0;
    double pq = (a00 - q) * (a00 - q) + (a11 - q) * (a11 - q) +
                (a22 - q) * (a22 - q) +
                2.0 * (a01 * a01 + a02 * a02 + a12 * a12);
    pq = sqrt(pq / 6.0);
    double mpq = 1.0 / (pq * pq * pq); // std::pow(1.0 / pq, 3.0);
    double det_b = mpq * ((a00 - q) * ((a11 - q) * (a22 - q) - a12 * a12) -
                          a01 * (a01 * (a22 - q) - a12 * a02) +
                          a02 * (a01 * a12 - (a11 - q) * a02));
    double r = 0.5 * det_b;
    double phi = 0.0;
    if (r <= -1.0)
        phi = 3.1415926 / 3.0;
    else if (r >= 1.0)
        phi = 0.0;
    else
        phi = acos(r) / 3.0;
    double eig = q + 2.0 * pq * cos(phi + 3.1415926 * (2.0 / 3.0));

    double nx = a01 * a12 - a02 * (a11 - eig);
    double ny = a01 * a02 - a12 * (a00 - eig);
    double nz = (a00 - eig) * (a11 - eig) - a01 * a01;
    double len = sqrt(nx * nx + ny * ny + nz * nz);
    nx /= len;
    ny /= len;
    nz /= len;

    // double d = -(nx * cx + ny * cy + nz * cz);

    rm::Vector3d eigen_v(nx, ny, nz);
    rm::Vector3d cog(cx, cy, cz);

    plane_equation = plane_from_point_and_normal(cog, eigen_v.normalized());

    for (size_t i = 0; i < cluster_size; i++)
    {
        if (fabs(plane_normal_point_distance(plane_equation, cluster[i])) > normal_threshold)
        {
            return false;
        }
    }

    plane_equation(3) = 1.0 / len;

    return true;
}
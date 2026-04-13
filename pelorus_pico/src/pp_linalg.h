#ifndef PP_LINALG_H
#define PP_LINALG_H

/* =========================================================
 * pp_linalg.h — 3×3 float, 24×24 double, SO(3) operations
 *
 * All large temporaries (24×24) must live in static storage
 * (g_ctx), never on the stack.
 * ========================================================= */

/* ---- 3×3 float matrix (row-major) ---- */
typedef float  mat3f[3][3];
typedef double mat3d[3][3];

/* ---- 24×24 double matrix (row-major) ---- */
typedef double mat24[24][24];

/* ---- 3-vector helpers ---- */
void   vec3f_zero(float  v[3]);
void   vec3d_zero(double v[3]);
void   vec3f_copy(float  dst[3], const float  src[3]);
void   vec3d_copy(double dst[3], const double src[3]);
void   vec3f_add(float  dst[3], const float  a[3], const float  b[3]);
void   vec3d_add(double dst[3], const double a[3], const double b[3]);
void   vec3f_sub(float  dst[3], const float  a[3], const float  b[3]);
void   vec3d_sub(double dst[3], const double a[3], const double b[3]);
void   vec3f_scale(float  dst[3], const float  v[3], float  s);
void   vec3d_scale(double dst[3], const double v[3], double s);
float  vec3f_dot(const float  a[3], const float  b[3]);
double vec3d_dot(const double a[3], const double b[3]);
float  vec3f_norm(const float  v[3]);
double vec3d_norm(const double v[3]);
void   vec3f_cross(float  dst[3], const float  a[3], const float  b[3]);
void   vec3d_cross(double dst[3], const double a[3], const double b[3]);
/* dst = R * v  (3×3 × 3×1) */
void   mat3f_mulv(float  dst[3], const mat3f R, const float  v[3]);
void   mat3d_mulv(double dst[3], const mat3d R, const double v[3]);

/* ---- 3×3 float matrix ops ---- */
void  mat3f_zero(mat3f m);
void  mat3f_identity(mat3f m);
void  mat3f_copy(mat3f dst, const mat3f src);
void  mat3f_transpose(mat3f dst, const mat3f src);
void  mat3f_mul(mat3f dst, const mat3f a, const mat3f b);
void  mat3f_add(mat3f dst, const mat3f a, const mat3f b);
void  mat3f_sub(mat3f dst, const mat3f a, const mat3f b);
void  mat3f_scale(mat3f dst, const mat3f m, float  s);
/* 3×3 inverse via cofactors (exact, no decomposition); returns 0 if singular */
int   mat3f_inv(mat3f inv, const mat3f m);
/* Cholesky L factor of a 3×3 symmetric positive-definite matrix.
 * Returns 0 on success, -1 if not PD (sets L to identity). */
int   mat3f_cholesky_l(mat3f L, const mat3f A);

/* ---- 3×3 symmetric eigenvalues (float) ----
 * Fills eigenvalues[3] using the analytic characteristic polynomial.
 * Smallest eigenvalue is eigenvalues[0] after the call sorts them. */
void mat3f_eigenvalues(const mat3f A, float eigenvalues[3]);

/* ---- 3×3 eigenvector for the SMALLEST eigenvalue ----
 * Uses null-space via cross-product of rows of (A - lambda*I).
 * Result written to normal[3] (unit vector).
 * Returns 0 on success, -1 if degenerate. */
int mat3f_eigenvec_smallest(const mat3f A, float normal[3]);

/* ---- Skew-symmetric matrix from 3-vector ---- */
/* K[i][j] such that K*v = w × v */
void skew3f(mat3f K, const float  w[3]);
void skew3d(mat3d K, const double w[3]);

/* ---- SO(3) operations (double) ---- */
/* Rodrigues formula: exp(v) -> R  (v is axis-angle vector) */
void so3_exp(mat3d R, const double v[3]);
/* Inverse Rodrigues: log(R) -> v */
void so3_log(double v[3], const mat3d R);
/* R_dst = R_a * R_b */
void so3_mul(mat3d dst, const mat3d a, const mat3d b);
/* dst = src^T  (= inverse for rotation matrices) */
void so3_transpose(mat3d dst, const mat3d src);
/* Apply rotation: dst = R * v */
void so3_apply(double dst[3], const mat3d R, const double v[3]);
/* Quaternion to rotation matrix [w,x,y,z] -> R */
void quat_to_so3(mat3d R, const double q[4]);
/* Rotation matrix to quaternion -> q[w,x,y,z] */
void so3_to_quat(double q[4], const mat3d R);

/* ---- 24×24 double matrix ops ---- */
void mat24_zero(mat24 m);
void mat24_identity(mat24 m);
void mat24_copy(mat24 dst, const mat24 src);
void mat24_transpose(mat24 dst, const mat24 src);
/* dst = a + b */
void mat24_add(mat24 dst, const mat24 a, const mat24 b);
/* dst = a - b */
void mat24_sub(mat24 dst, const mat24 a, const mat24 b);
/* dst = a * b */
void mat24_mul(mat24 dst, const mat24 a, const mat24 b);
/* dst = s * m */
void mat24_scale(mat24 dst, const mat24 m, double s);

/* ---- LUP decomposition and 24×24 inverse ----
 * Returns 0 on success, -1 if singular. */
int mat24_lup_inverse(mat24 inv, const mat24 A);

/* ---- Mixed-size helpers (for IEKF) ---- */
/* dst[24][24] = a[24][12] * b[12][24] */
void mat_mul_24x12_12x24(mat24 dst, const double a[24][12], const double b[12][24]);
/* dst[24][12] = m[24][24] * b[24][12] */
void mat_mul_24x24_24x12(double dst[24][12], const mat24 m, const double b[24][12]);
/* dst[12][12] = a[12][24] * b[24][12]   (only uses first 12 cols of b) */
void mat_mul_12x24_24x12(double dst[12][12], const double a[12][24], const double b[24][12]);
/* LUP inverse of a 12×12 matrix */
int mat12_lup_inverse(double inv[12][12], const double A[12][12]);
/* dst[12][12] = s * m[12][12] */
void mat12_scale(double dst[12][12], const double m[12][12], double s);
/* dst[12][12] = a[12][12] + b[12][12] */
void mat12_add(double dst[12][12], const double a[12][12], const double b[12][12]);

#endif /* PP_LINALG_H */

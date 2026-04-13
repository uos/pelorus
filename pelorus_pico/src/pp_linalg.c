/* pp_linalg.c — 3×3 float, 24×24 double, SO(3) math
 * Freestanding: only __builtin_sqrt/sin/cos/acos needed.
 * On a hosted environment these resolve to libm; on bare-metal
 * the compiler emits hardware FPU or soft-float calls.
 */
#include "pp_linalg.h"

/* Use compiler built-ins so no libm.h needed in freestanding mode. */
#define PP_SQRT(x)  __builtin_sqrt(x)
#define PP_SQRTF(x) __builtin_sqrtf(x)
#define PP_SIN(x)   __builtin_sin(x)
#define PP_COS(x)   __builtin_cos(x)
#define PP_ACOS(x)  __builtin_acos(x)
#define PP_FABS(x)  __builtin_fabs(x)
#define PP_FABSF(x) __builtin_fabsf(x)

/* ================================================================
 * 3-vector helpers
 * ================================================================ */
void vec3f_zero(float v[3])  { v[0]=v[1]=v[2]=0.0f; }
void vec3d_zero(double v[3]) { v[0]=v[1]=v[2]=0.0; }

void vec3f_copy(float dst[3], const float src[3])   { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; }
void vec3d_copy(double dst[3], const double src[3]) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; }

void vec3f_add(float dst[3], const float a[3], const float b[3]) {
    dst[0]=a[0]+b[0]; dst[1]=a[1]+b[1]; dst[2]=a[2]+b[2];
}
void vec3d_add(double dst[3], const double a[3], const double b[3]) {
    dst[0]=a[0]+b[0]; dst[1]=a[1]+b[1]; dst[2]=a[2]+b[2];
}
void vec3f_sub(float dst[3], const float a[3], const float b[3]) {
    dst[0]=a[0]-b[0]; dst[1]=a[1]-b[1]; dst[2]=a[2]-b[2];
}
void vec3d_sub(double dst[3], const double a[3], const double b[3]) {
    dst[0]=a[0]-b[0]; dst[1]=a[1]-b[1]; dst[2]=a[2]-b[2];
}
void vec3f_scale(float dst[3], const float v[3], float s)   { dst[0]=v[0]*s; dst[1]=v[1]*s; dst[2]=v[2]*s; }
void vec3d_scale(double dst[3], const double v[3], double s){ dst[0]=v[0]*s; dst[1]=v[1]*s; dst[2]=v[2]*s; }

float  vec3f_dot(const float a[3], const float b[3])   { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
double vec3d_dot(const double a[3], const double b[3]) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

float  vec3f_norm(const float v[3])   { return PP_SQRTF(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
double vec3d_norm(const double v[3])  { return PP_SQRT(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }

void vec3f_cross(float dst[3], const float a[3], const float b[3]) {
    dst[0] = a[1]*b[2] - a[2]*b[1];
    dst[1] = a[2]*b[0] - a[0]*b[2];
    dst[2] = a[0]*b[1] - a[1]*b[0];
}
void vec3d_cross(double dst[3], const double a[3], const double b[3]) {
    dst[0] = a[1]*b[2] - a[2]*b[1];
    dst[1] = a[2]*b[0] - a[0]*b[2];
    dst[2] = a[0]*b[1] - a[1]*b[0];
}

void mat3f_mulv(float dst[3], const mat3f R, const float v[3]) {
    for (int i = 0; i < 3; i++) {
        dst[i] = R[i][0]*v[0] + R[i][1]*v[1] + R[i][2]*v[2];
    }
}
void mat3d_mulv(double dst[3], const mat3d R, const double v[3]) {
    for (int i = 0; i < 3; i++) {
        dst[i] = R[i][0]*v[0] + R[i][1]*v[1] + R[i][2]*v[2];
    }
}

/* ================================================================
 * 3×3 float matrix ops
 * ================================================================ */
void mat3f_zero(mat3f m) {
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) m[i][j]=0.0f;
}
void mat3f_identity(mat3f m) {
    mat3f_zero(m); m[0][0]=m[1][1]=m[2][2]=1.0f;
}
void mat3f_copy(mat3f dst, const mat3f src) {
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) dst[i][j]=src[i][j];
}
void mat3f_transpose(mat3f dst, const mat3f src) {
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) dst[i][j]=src[j][i];
}
void mat3f_mul(mat3f dst, const mat3f a, const mat3f b) {
    mat3f tmp; mat3f_zero(tmp);
    for (int i=0;i<3;i++)
        for (int k=0;k<3;k++) if (a[i][k]!=0.0f)
            for (int j=0;j<3;j++) tmp[i][j] += a[i][k]*b[k][j];
    mat3f_copy(dst, tmp);
}
void mat3f_add(mat3f dst, const mat3f a, const mat3f b) {
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) dst[i][j]=a[i][j]+b[i][j];
}
void mat3f_sub(mat3f dst, const mat3f a, const mat3f b) {
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) dst[i][j]=a[i][j]-b[i][j];
}
void mat3f_scale(mat3f dst, const mat3f m, float s) {
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) dst[i][j]=m[i][j]*s;
}

int mat3f_inv(mat3f inv, const mat3f m) {
    /* cofactor / Cramer */
    float c00 = m[1][1]*m[2][2] - m[1][2]*m[2][1];
    float c01 = m[1][2]*m[2][0] - m[1][0]*m[2][2];
    float c02 = m[1][0]*m[2][1] - m[1][1]*m[2][0];
    float det = m[0][0]*c00 + m[0][1]*c01 + m[0][2]*c02;
    if (PP_FABSF(det) < 1e-12f) return -1;
    float inv_det = 1.0f / det;
    inv[0][0] =  c00 * inv_det;
    inv[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * inv_det;
    inv[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * inv_det;
    inv[1][0] =  c01 * inv_det;
    inv[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * inv_det;
    inv[1][2] = (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * inv_det;
    inv[2][0] =  c02 * inv_det;
    inv[2][1] = (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * inv_det;
    inv[2][2] = (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * inv_det;
    return 0;
}

int mat3f_cholesky_l(mat3f L, const mat3f A) {
    /* Cholesky-Banachiewicz for 3×3 SPD matrix. */
    mat3f_zero(L);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j <= i; j++) {
            float s = A[i][j];
            for (int k = 0; k < j; k++) s -= L[i][k]*L[j][k];
            if (i == j) {
                if (s <= 0.0f) { mat3f_identity(L); return -1; }
                L[i][j] = PP_SQRTF(s);
            } else {
                L[i][j] = s / L[j][j];
            }
        }
    }
    return 0;
}

/* ---- eigenvalues of 3×3 symmetric matrix (analytic cubic) ---- */
void mat3f_eigenvalues(const mat3f A, float ev[3]) {
    /* Characteristic polynomial: λ³ - tr λ² + c2 λ - det = 0 */
    float tr = A[0][0]+A[1][1]+A[2][2];
    float c2 = (A[0][0]*A[1][1]+A[0][0]*A[2][2]+A[1][1]*A[2][2])
             - (A[0][1]*A[0][1]+A[0][2]*A[0][2]+A[1][2]*A[1][2]);
    float det = A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
              - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
              + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);

    /* Depressed cubic: t³ + p*t + q = 0  (t = λ - tr/3) */
    float p = c2 - tr*tr/3.0f;
    float q = (2.0f*tr*tr*tr/27.0f) - (tr*c2/3.0f) + det;

    float angle;
    float two_sqrt;
    if (PP_FABSF(p) < 1e-7f) {
        angle = 0.0f;
        two_sqrt = 0.0f;
    } else {
        float disc = (-p*p*p/27.0f);
        float sqrt_disc = PP_SQRTF(disc < 0.0f ? 0.0f : disc);
        float acos_arg = (-q/2.0f) / sqrt_disc;
        if (acos_arg >  1.0f) acos_arg =  1.0f;
        if (acos_arg < -1.0f) acos_arg = -1.0f;
        angle = PP_ACOS((double)acos_arg);
        two_sqrt = 2.0f * PP_SQRTF(-p/3.0f);
    }
    float shift = tr/3.0f;
    float pi = 3.14159265358979323846f;
    ev[0] = two_sqrt * (float)PP_COS((double)(angle/3.0f))             + shift;
    ev[1] = two_sqrt * (float)PP_COS((double)((angle+2.0f*pi)/3.0f))   + shift;
    ev[2] = two_sqrt * (float)PP_COS((double)((angle+4.0f*pi)/3.0f))   + shift;

    /* Sort ascending */
    float tmp;
    if (ev[0] > ev[1]) { tmp=ev[0]; ev[0]=ev[1]; ev[1]=tmp; }
    if (ev[1] > ev[2]) { tmp=ev[1]; ev[1]=ev[2]; ev[2]=tmp; }
    if (ev[0] > ev[1]) { tmp=ev[0]; ev[0]=ev[1]; ev[1]=tmp; }
}

int mat3f_eigenvec_smallest(const mat3f A, float normal[3]) {
    float ev[3];
    mat3f_eigenvalues(A, ev);
    float lambda = ev[0]; /* smallest */

    /* Shift: M = A - lambda*I */
    mat3f M;
    mat3f_copy(M, A);
    M[0][0] -= lambda; M[1][1] -= lambda; M[2][2] -= lambda;

    /* Plane normal = null space of M: cross products of pairs of rows */
    float row0[3] = { M[0][0], M[0][1], M[0][2] };
    float row1[3] = { M[1][0], M[1][1], M[1][2] };
    float row2[3] = { M[2][0], M[2][1], M[2][2] };

    float v1[3], v2[3], v3[3];
    vec3f_cross(v1, row0, row1);
    vec3f_cross(v2, row1, row2);
    vec3f_cross(v3, row2, row0);

    float n1 = vec3f_dot(v1,v1);
    float n2 = vec3f_dot(v2,v2);
    float n3 = vec3f_dot(v3,v3);

    float *best = (n1>=n2 && n1>=n3) ? v1 : (n2>=n3 ? v2 : v3);
    float len = vec3f_norm(best);
    if (len < 1e-12f) return -1;
    normal[0] = best[0]/len;
    normal[1] = best[1]/len;
    normal[2] = best[2]/len;
    return 0;
}

/* ================================================================
 * Skew-symmetric
 * ================================================================ */
void skew3f(mat3f K, const float w[3]) {
    K[0][0]=0;     K[0][1]=-w[2]; K[0][2]= w[1];
    K[1][0]= w[2]; K[1][1]=0;     K[1][2]=-w[0];
    K[2][0]=-w[1]; K[2][1]= w[0]; K[2][2]=0;
}
void skew3d(mat3d K, const double w[3]) {
    K[0][0]=0;     K[0][1]=-w[2]; K[0][2]= w[1];
    K[1][0]= w[2]; K[1][1]=0;     K[1][2]=-w[0];
    K[2][0]=-w[1]; K[2][1]= w[0]; K[2][2]=0;
}

/* ================================================================
 * SO(3) — Rodrigues formula (double)
 * ================================================================ */
void so3_exp(mat3d R, const double v[3]) {
    double theta = PP_SQRT(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    if (theta < 1e-10) {
        /* R ≈ I + skew(v) for tiny angles */
        R[0][0]=1; R[0][1]=-v[2]; R[0][2]= v[1];
        R[1][0]= v[2]; R[1][1]=1; R[1][2]=-v[0];
        R[2][0]=-v[1]; R[2][1]= v[0]; R[2][2]=1;
        return;
    }
    double inv_th = 1.0/theta;
    double ax = v[0]*inv_th, ay = v[1]*inv_th, az = v[2]*inv_th;
    double s = PP_SIN(theta), c = PP_COS(theta), t = 1.0-c;
    R[0][0] = t*ax*ax + c;
    R[0][1] = t*ax*ay - s*az;
    R[0][2] = t*ax*az + s*ay;
    R[1][0] = t*ax*ay + s*az;
    R[1][1] = t*ay*ay + c;
    R[1][2] = t*ay*az - s*ax;
    R[2][0] = t*ax*az - s*ay;
    R[2][1] = t*ay*az + s*ax;
    R[2][2] = t*az*az + c;
}

void so3_log(double v[3], const mat3d R) {
    double trace = R[0][0]+R[1][1]+R[2][2];
    double cos_theta = (trace - 1.0) * 0.5;
    if (cos_theta > 1.0)  cos_theta = 1.0;
    if (cos_theta < -1.0) cos_theta = -1.0;
    double theta = PP_ACOS(cos_theta);
    if (PP_FABS(theta) < 1e-10) {
        v[0] = 0.5*(R[2][1]-R[1][2]);
        v[1] = 0.5*(R[0][2]-R[2][0]);
        v[2] = 0.5*(R[1][0]-R[0][1]);
        return;
    }
    double s = theta / (2.0 * PP_SIN(theta));
    v[0] = s*(R[2][1]-R[1][2]);
    v[1] = s*(R[0][2]-R[2][0]);
    v[2] = s*(R[1][0]-R[0][1]);
}

void so3_mul(mat3d dst, const mat3d a, const mat3d b) {
    mat3d tmp;
    for (int i=0;i<3;i++)
        for (int j=0;j<3;j++) {
            tmp[i][j] = 0.0;
            for (int k=0;k<3;k++) tmp[i][j] += a[i][k]*b[k][j];
        }
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) dst[i][j]=tmp[i][j];
}

void so3_transpose(mat3d dst, const mat3d src) {
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) dst[i][j]=src[j][i];
}

void so3_apply(double dst[3], const mat3d R, const double v[3]) {
    mat3d_mulv(dst, R, v);
}

void quat_to_so3(mat3d R, const double q[4]) {
    /* q = [w, x, y, z] */
    double w=q[0],x=q[1],y=q[2],z=q[3];
    R[0][0]=1-2*(y*y+z*z); R[0][1]=2*(x*y-w*z);   R[0][2]=2*(x*z+w*y);
    R[1][0]=2*(x*y+w*z);   R[1][1]=1-2*(x*x+z*z); R[1][2]=2*(y*z-w*x);
    R[2][0]=2*(x*z-w*y);   R[2][1]=2*(y*z+w*x);   R[2][2]=1-2*(x*x+y*y);
}

void so3_to_quat(double q[4], const mat3d R) {
    double trace = R[0][0]+R[1][1]+R[2][2];
    if (trace > 0.0) {
        double s = 0.5 / PP_SQRT(trace+1.0);
        q[0] = 0.25/s;
        q[1] = (R[2][1]-R[1][2])*s;
        q[2] = (R[0][2]-R[2][0])*s;
        q[3] = (R[1][0]-R[0][1])*s;
    } else if (R[0][0]>R[1][1] && R[0][0]>R[2][2]) {
        double s = 2.0*PP_SQRT(1.0+R[0][0]-R[1][1]-R[2][2]);
        q[0]=(R[2][1]-R[1][2])/s; q[1]=0.25*s;
        q[2]=(R[0][1]+R[1][0])/s; q[3]=(R[0][2]+R[2][0])/s;
    } else if (R[1][1]>R[2][2]) {
        double s = 2.0*PP_SQRT(1.0+R[1][1]-R[0][0]-R[2][2]);
        q[0]=(R[0][2]-R[2][0])/s; q[1]=(R[0][1]+R[1][0])/s;
        q[2]=0.25*s; q[3]=(R[1][2]+R[2][1])/s;
    } else {
        double s = 2.0*PP_SQRT(1.0+R[2][2]-R[0][0]-R[1][1]);
        q[0]=(R[1][0]-R[0][1])/s; q[1]=(R[0][2]+R[2][0])/s;
        q[2]=(R[1][2]+R[2][1])/s; q[3]=0.25*s;
    }
}

/* ================================================================
 * 24×24 double matrix ops
 * ================================================================ */
void mat24_zero(mat24 m) {
    for (int i=0;i<24;i++) for (int j=0;j<24;j++) m[i][j]=0.0;
}
void mat24_identity(mat24 m) {
    mat24_zero(m); for (int i=0;i<24;i++) m[i][i]=1.0;
}
void mat24_copy(mat24 dst, const mat24 src) {
    for (int i=0;i<24;i++) for (int j=0;j<24;j++) dst[i][j]=src[i][j];
}
void mat24_transpose(mat24 dst, const mat24 src) {
    /* safe even if dst==src via two-pass */
    double tmp[24][24];
    for (int i=0;i<24;i++) for (int j=0;j<24;j++) tmp[i][j]=src[j][i];
    for (int i=0;i<24;i++) for (int j=0;j<24;j++) dst[i][j]=tmp[i][j];
}
void mat24_add(mat24 dst, const mat24 a, const mat24 b) {
    for (int i=0;i<24;i++) for (int j=0;j<24;j++) dst[i][j]=a[i][j]+b[i][j];
}
void mat24_sub(mat24 dst, const mat24 a, const mat24 b) {
    for (int i=0;i<24;i++) for (int j=0;j<24;j++) dst[i][j]=a[i][j]-b[i][j];
}
void mat24_mul(mat24 dst, const mat24 a, const mat24 b) {
    double tmp[24][24];
    for (int i=0;i<24;i++) {
        for (int j=0;j<24;j++) {
            double s = 0.0;
            for (int k=0;k<24;k++) s += a[i][k]*b[k][j];
            tmp[i][j] = s;
        }
    }
    for (int i=0;i<24;i++) for (int j=0;j<24;j++) dst[i][j]=tmp[i][j];
}
void mat24_scale(mat24 dst, const mat24 m, double s) {
    for (int i=0;i<24;i++) for (int j=0;j<24;j++) dst[i][j]=m[i][j]*s;
}

/* ---- LUP decomposition helpers (used by both 12×12 and 24×24 inverse) ---- */

/* Generic NxN LUP inverse via Doolittle.
 * n <= 24, uses stack scratch of n*n doubles. */
static int lup_inverse_n(int n, double *inv /*n×n*/, const double *A /*n×n*/) {
    /* We need scratch space for L+U in-place and pivot array.
     * Maximum n=24: 24×24 = 576 doubles = 4608 bytes — small enough on A53 stack. */
    double LU[24][24];
    int    piv[24];
    int    i, j, k;

    /* Copy A into LU */
    for (i=0;i<n;i++) for (j=0;j<n;j++) LU[i][j] = A[i*n+j];
    for (i=0;i<n;i++) piv[i] = i;

    /* Doolittle LU with partial pivoting */
    for (k=0;k<n;k++) {
        /* Find pivot */
        double max_val = PP_FABS(LU[k][k]);
        int max_row = k;
        for (i=k+1;i<n;i++) {
            double v = PP_FABS(LU[i][k]);
            if (v > max_val) { max_val = v; max_row = i; }
        }
        if (max_val < 1e-14) return -1; /* singular */

        if (max_row != k) {
            /* Swap rows in LU */
            for (j=0;j<n;j++) { double t=LU[k][j]; LU[k][j]=LU[max_row][j]; LU[max_row][j]=t; }
            int tp = piv[k]; piv[k] = piv[max_row]; piv[max_row] = tp;
        }

        double inv_k = 1.0/LU[k][k];
        for (i=k+1;i<n;i++) {
            LU[i][k] *= inv_k;
            for (j=k+1;j<n;j++) LU[i][j] -= LU[i][k]*LU[k][j];
        }
    }

    /* Solve n systems LU * x = e_j to get columns of inverse */
    for (j=0;j<n;j++) {
        /* Build permuted e_j */
        double b[24] = {0};
        b[piv[j]] = 1.0; /* actually we need to find where piv maps j */
        /* Redo: build rhs as e_piv^{-1}[j] — find which original row is at position j */
        for (i=0;i<n;i++) b[i]=0.0;
        /* Find i such that piv[i]==j to locate original row of e_j */
        for (i=0;i<n;i++) if (piv[i]==j) { b[i]=1.0; break; }

        /* Forward substitution: L * y = b */
        double y[24];
        for (i=0;i<n;i++) {
            y[i] = b[i];
            for (k=0;k<i;k++) y[i] -= LU[i][k]*y[k];
        }
        /* Back substitution: U * x = y */
        double x[24];
        for (i=n-1;i>=0;i--) {
            x[i] = y[i];
            for (k=i+1;k<n;k++) x[i] -= LU[i][k]*x[k];
            x[i] /= LU[i][i];
        }
        /* Store column j of inverse */
        for (i=0;i<n;i++) inv[i*n+j] = x[i];
    }
    return 0;
}

int mat24_lup_inverse(mat24 inv, const mat24 A) {
    return lup_inverse_n(24, &inv[0][0], &A[0][0]);
}

int mat12_lup_inverse(double inv[12][12], const double A[12][12]) {
    return lup_inverse_n(12, &inv[0][0], &A[0][0]);
}

void mat12_scale(double dst[12][12], const double m[12][12], double s) {
    for (int i=0;i<12;i++) for (int j=0;j<12;j++) dst[i][j]=m[i][j]*s;
}
void mat12_add(double dst[12][12], const double a[12][12], const double b[12][12]) {
    for (int i=0;i<12;i++) for (int j=0;j<12;j++) dst[i][j]=a[i][j]+b[i][j];
}

/* ---- Mixed-size helpers ---- */
void mat_mul_24x12_12x24(mat24 dst, const double a[24][12], const double b[12][24]) {
    for (int i=0;i<24;i++) for (int j=0;j<24;j++) {
        double s=0.0;
        for (int k=0;k<12;k++) s += a[i][k]*b[k][j];
        dst[i][j] = s;
    }
}

void mat_mul_24x24_24x12(double dst[24][12], const mat24 m, const double b[24][12]) {
    for (int i=0;i<24;i++) for (int j=0;j<12;j++) {
        double s=0.0;
        for (int k=0;k<24;k++) s += m[i][k]*b[k][j];
        dst[i][j] = s;
    }
}

void mat_mul_12x24_24x12(double dst[12][12], const double a[12][24], const double b[24][12]) {
    for (int i=0;i<12;i++) for (int j=0;j<12;j++) {
        double s=0.0;
        for (int k=0;k<24;k++) s += a[i][k]*b[k][j];
        dst[i][j] = s;
    }
}

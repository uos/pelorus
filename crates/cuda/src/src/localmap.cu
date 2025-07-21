#include <cassert>
#include <cfloat>
#include <cmath>
#include <core/common.h>
#include <core/cuda_stream.h>
#include <core/memory.h>
#include <core/memory_cuda.h>
#include <cstdint>
#include <hashmap.h>
#include <lib.h>
#include <localmap.h>
#include <math/matrix3x3.h>
#include <math/vector3.h>
#include <sys/types.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace rm = rmagine;

#define INVALID_INDEX UINT32_MAX

/**
 * @brief Performs Cholesky decomposition for a 3x3 symmetric positive-definite
 * matrix.
 * @param A The input matrix (must be symmetric positive-definite).
 * @param L The output lower-triangular matrix such that A = L * L.T.
 * @return True if decomposition is successful, false otherwise.
 */
__device__ bool cholesky_3x3(const rm::Matrix3x3f &A, rm::Matrix3x3f &L) {
  L.setZeros();

  // L(0, 0)
  float l00_sq = A.at(0, 0);
  if (l00_sq <= 0.0f)
    return false;
  float l00 = sqrtf(l00_sq);
  L.at(0, 0) = l00;

  // First column of L
  L.at(1, 0) = A.at(1, 0) / l00;
  L.at(2, 0) = A.at(2, 0) / l00;

  // L(1, 1)
  float l11_sq = A.at(1, 1) - L.at(1, 0) * L.at(1, 0);
  if (l11_sq <= 0.0f)
    return false;
  float l11 = sqrtf(l11_sq);
  L.at(1, 1) = l11;

  // Second column of L (below diagonal)
  L.at(2, 1) = (A.at(2, 1) - L.at(2, 0) * L.at(1, 0)) / l11;

  // L(2, 2)
  float l22_sq =
      A.at(2, 2) - (L.at(2, 0) * L.at(2, 0) + L.at(2, 1) * L.at(2, 1));
  if (l22_sq <= 0.0f)
    return false;
  L.at(2, 2) = sqrtf(l22_sq);

  return true;
}

/**
 * @brief Solves the linear system Ax=b using the Cholesky factor L.
 * @param L The lower-triangular Cholesky factor of A.
 * @param b The vector 'b' in Ax=b.
 * @return The solution vector 'x'.
 */
__device__ rm::Vector3f cholesky_solve_3x3(const rm::Matrix3x3f &L,
                                           const rm::Vector3f &b) {
  // --- Step 1: Forward substitution to solve Ly = b ---
  rm::Vector3f y;
  y.x = b.x / L.at(0, 0);
  y.y = (b.y - L.at(1, 0) * y.x) / L.at(1, 1);
  y.z = (b.z - L.at(2, 0) * y.x - L.at(2, 1) * y.y) / L.at(2, 2);

  // --- Step 2: Backward substitution to solve L'x = y ---
  rm::Vector3f x;
  x.z = y.z / L.at(2, 2);
  x.y = (y.y - L.at(2, 1) * x.z) / L.at(1, 1);
  x.x = (y.x - L.at(1, 0) * x.y - L.at(2, 0) * x.z) / L.at(0, 0);

  return x;
}

/**
 * @brief Computes the eigenvalues of a 3x3 symmetric matrix.
 * @param A The 3x3 symmetric matrix.
 * @return A Vector3f containing the three real eigenvalues.
 */
__device__ rm::Vector3f symmetric_eigen_3x3(const rm::Matrix3x3f &A) {
  // The characteristic equation is λ³ - tr(A)λ² + c₂λ - det(A) = 0
  // where c₂ = 0.5 * (tr(A)² - tr(A²))

  const float a00 = A.at(0, 0), a01 = A.at(0, 1), a02 = A.at(0, 2);
  const float a11 = A.at(1, 1), a12 = A.at(1, 2);
  const float a22 = A.at(2, 2);

  const float traceA = a00 + a11 + a22;

  const float c2 =
      (a00 * a11 + a00 * a22 + a11 * a22) - (a01 * a01 + a02 * a02 + a12 * a12);

  const float detA = a00 * (a11 * a22 - a12 * a12) -
                     a01 * (a01 * a22 - a12 * a02) +
                     a02 * (a01 * a12 - a11 * a02);

  // Solve the cubic equation: x³ + p*x² + q*x + r = 0
  const float p = -traceA;
  const float q = c2;
  const float r = -detA;

  // Convert to depressed cubic: t³ + a*t + b = 0
  const float a = (3.0f * q - p * p) / 3.0f;
  const float b = (2.0f * p * p * p - 9.0f * p * q + 27.0f * r) / 27.0f;

  // For symmetric matrices, eigenvalues are real, so the discriminant is <= 0.
  // We use the trigonometric solution for three real roots.
  float angle = 0.0f;
  if (a != 0.0f) {
    // Clamp the argument to acos to handle floating point inaccuracies
    float acos_arg = (-b / 2.0f) / sqrtf(-a * a * a / 27.0f);
    acos_arg = fmaxf(-1.0f, fminf(1.0f, acos_arg));
    angle = acosf(acos_arg);
  }

  const float two_sqrt_neg_a_div_3 = 2.0f * sqrtf(-a / 3.0f);
  const float p_div_3 = p / 3.0f;

  rm::Vector3f eigenvalues;

  // Root 1
  eigenvalues.x = two_sqrt_neg_a_div_3 * cosf(angle / 3.0f) - p_div_3;
  // Root 2
  eigenvalues.y =
      two_sqrt_neg_a_div_3 * cosf((angle + 2.0f * M_PI) / 3.0f) - p_div_3;
  // Root 3
  eigenvalues.z =
      two_sqrt_neg_a_div_3 * cosf((angle + 4.0f * M_PI) / 3.0f) - p_div_3;

  return eigenvalues;
}

// Weighted update of the distribution in a voxel. For more information see the
// CPU variant in Rust. This function is just a port.
__device__ void VoxelDistribution::update(const rm::Vector3f &new_point,
                                          float sensitivity, float min_dist,
                                          size_t current_scan_number) {
  this->last_updated_at_scan = current_scan_number;

  if (this->point_count < 2) {
    if (this->point_count == 1) {
      if ((this->mean - new_point).l2normSquared() > min_dist * min_dist) {
        return;
      }
    }

    this->point_count++;
    const float n = static_cast<float>(this->point_count);
    const rm::Vector3f old_mean = this->mean;
    const rm::Vector3f diff_from_old_mean = new_point - old_mean;
    this->mean = old_mean + diff_from_old_mean / n;

    if (this->point_count > 1) {
      const rm::Vector3f diff_from_new_mean = new_point - this->mean;

      rm::Matrix3x3f scatter_update;
      const float u[3] = {diff_from_old_mean.x, diff_from_old_mean.y,
                          diff_from_old_mean.z};
      const float v[3] = {diff_from_new_mean.x, diff_from_new_mean.y,
                          diff_from_new_mean.z};

      for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
          scatter_update.at(i, j) = u[i] * v[j];
        }
      }

      this->covariance = this->covariance * ((n - 2.0f) / (n - 1.0f)) +
                         scatter_update * (1.0f / (n - 1.0f));
    }
    if (!cholesky_3x3(this->covariance, this->transform)) {
      rm::Vector3f eigenvalues = symmetric_eigen_3x3(this->covariance);
      this->transform = rm::Matrix3x3f{};
      this->transform(0, 0) = sqrtf(fmaxf(0.0f, eigenvalues.x));
      this->transform(1, 1) = sqrtf(fmaxf(0.0f, eigenvalues.y));
      this->transform(2, 2) = sqrtf(fmaxf(0.0f, eigenvalues.z));
    }
    return;
  }

  rm::Matrix3x3f id;
  id.setIdentity();

  const float maturity_decay =
      fmaxf(1.0f, logf(static_cast<float>(this->point_count) - 1.0f));
  const float effective_sensitivity = sensitivity / maturity_decay;
  const rm::Vector3f diff_from_old_mean = new_point - this->mean;

  const float regularization_factor = 1e-6f;
  rm::Matrix3x3f regularized_cov =
      this->covariance + id * regularization_factor;

  float distance_sq;
  if (cholesky_3x3(regularized_cov, this->transform)) {
    rm::Vector3f sol = cholesky_solve_3x3(this->transform, diff_from_old_mean);
    distance_sq = diff_from_old_mean.dot(sol);
  } else {
    rm::Vector3f eigenvalues = symmetric_eigen_3x3(this->covariance);
    this->transform = rm::Matrix3x3f{};
    this->transform(0, 0) = sqrtf(fmaxf(0.0f, eigenvalues.x));
    this->transform(1, 1) = sqrtf(fmaxf(0.0f, eigenvalues.y));
    this->transform(2, 2) = sqrtf(fmaxf(0.0f, eigenvalues.z));
    distance_sq = diff_from_old_mean.l2normSquared();
  }

  const float alpha = expf(-distance_sq / (2.0f * effective_sensitivity));
  const rm::Vector3f new_cov_sample = (1.0f - alpha) * diff_from_old_mean;
  rm::Matrix3x3f new_cov_mt;
  const float u[3] = {new_cov_sample.x, new_cov_sample.y, diff_from_old_mean.z};
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      new_cov_mt.at(i, j) = u[i] * u[j];
    }
  }
  this->covariance = this->covariance * (1.0 - alpha) + new_cov_mt * alpha;
  this->mean = (1.0f - alpha) * this->mean + alpha * new_point;
  this->point_count++;
}

// -- Localmap --

Localmap::Localmap(size_t capacity, float sensitivity, float min_dist,
                   float resolution, rm::CudaStream &stream)
    : capacity(capacity), sensitivity(sensitivity), min_dist(min_dist),
      add_cursor(0), lru_cursor(0), buffer(capacity), lru_mapping(capacity),
      resolution(resolution) {
  thrust::fill(thrust::device.on(stream.handle()), lru_mapping.begin(),
               lru_mapping.end(), INVALID_INDEX);
  // TODO one larger power of two
  // syncs the stream after initial copy
  lookup = create_map(capacity, stream);
}

Localmap::~Localmap() { destroy_map(this->lookup); }

void Localmap::clear() {
  add_cursor = 0;
  lru_cursor = 0;
  clear_map(lookup);
  thrust::fill(lru_mapping.begin(), lru_mapping.end(), INVALID_INDEX);
}

__device__ rm::Vector3<int32_t> pos2grid(float resolution,
                                         const rm::Pointf &point) {
  float inverse_res = 1.0 / resolution;
  auto inv = point * inverse_res;

  float x = roundf(inv.x);
  float y = roundf(inv.y);
  float z = roundf(inv.z);
  return rm::Vector3<int32_t>(x, y, z);
}

__device__ int32_t hash_grid(const rm::Vector3<int32_t> &grid_key) {
  constexpr int32_t PRIME_1 = 73856093;
  constexpr int32_t PRIME_2 = 19349663;
  constexpr int32_t PRIME_3 = 83492791;

  int32_t hash = grid_key.x * PRIME_1;
  hash = hash + grid_key.y * PRIME_2;
  hash = hash + grid_key.z * PRIME_3;

  return hash;
}

__global__ void point_insert(Point *points, size_t point_size,
                             size_t scan_number, Value *buffer,
                             GpuHashMap *lookup, float resolution,
                             float sensitivity, float min_dist,
                             uint32_t *lru_cursor, uint32_t *add_cursor,
                             uint32_t *lru_mapping) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx > point_size) {
    return;
  }

  Point point_raw = points[idx];
  rm::Pointf point = rm::Pointf(point_raw.x, point_raw.y, point_raw.z);

  auto cell = pos2grid(resolution, point);
  auto key = hash_grid(cell);
  bool found = true;
  auto buffer_idx = lookup_item(lookup->d_table, lookup->capacity, key, found);
  if (found) {
    buffer[buffer_idx].second.update(point, sensitivity, min_dist, scan_number);
  } else {
    // not found in map, must create new dist
    thrust::pair<uint32_t, VoxelDistribution> val =
        thrust::make_pair(key, VoxelDistribution(point));
    // NOTE diff to CPU version: since point inserting is fully parallel, the
    // lru mapping can not be triggered inside a check of a single point.
    // Therefore the lru_mapping is built at every step, regardless of not
    // reaching capacity. So it is potentially unused but makes the per-point
    // insertion easier.
    // auto lru_cursor_local = *lru_cursor;
    if (*add_cursor > lookup->capacity) {
      auto lru_cursor_local = atomicAdd(lru_cursor, 1);
      assert((lru_cursor_local < lookup->capacity));
      auto must_be_deleted = lru_mapping[lru_cursor_local];
      auto del = &buffer[must_be_deleted];
      assert((remove_item(lookup->d_table, lookup->capacity, del->first)));
      del->swap(val);
      insert_item(lookup->d_table, lookup->capacity, key, must_be_deleted);
    } else {
      auto add_cursor_local = atomicAdd(add_cursor, 1);
      buffer[add_cursor_local].swap(val);
      lru_mapping[add_cursor_local] = add_cursor_local;
      insert_item(lookup->d_table, lookup->capacity, key, add_cursor_local);
    }
  }
}

// assumes points is already on the gpu
void Localmap::add(Point *points, size_t point_size, size_t scan_number,
                   rm::CudaStream &stream) {

  rm::Memory<uint32_t, rm::RAM> lru_cursor_host(1);
  *lru_cursor_host = this->lru_cursor;
  rm::Memory<uint32_t, rm::VRAM_CUDA> lru_cursor_device = lru_cursor_host;

  rm::Memory<uint32_t, rm::RAM> add_cursor_host(1);
  *add_cursor_host = this->add_cursor;
  rm::Memory<uint32_t, rm::VRAM_CUDA> add_cursor_device = add_cursor_host;

  Value *buffer = thrust::raw_pointer_cast(this->buffer.data());
  uint32_t *lru_mapping = thrust::raw_pointer_cast(this->lru_mapping.data());
  int threads_per_block = 256;
  int blocks_per_grid = (capacity + threads_per_block - 1) / threads_per_block;
  point_insert<<<blocks_per_grid, threads_per_block, 0, stream.handle()>>>(
      points, point_size, scan_number, buffer, this->lookup.get(),
      this->resolution, this->sensitivity, this->min_dist,
      lru_cursor_device.raw(), add_cursor_device.raw(), lru_mapping);

  checkLastCudaError(__FILE__, __LINE__);

  this->build(scan_number, stream);

  // rm::Memory could go out of scope before point_insert is accessing it
  stream.synchronize();
}

struct ScoreComparator {
  const thrust::pair<uint32_t, VoxelDistribution> *buffer_ptr;
  size_t current_scan_number;

  __host__ __device__ double calculate_score(size_t last_updated_at_scan,
                                             uint64_t point_count) const {
    if (point_count == 0) {
      return DBL_MAX;
    }
    size_t temporal_dist = current_scan_number > last_updated_at_scan
                               ? current_scan_number - last_updated_at_scan
                               : 0;
    return static_cast<double>(temporal_dist) /
           static_cast<double>(point_count);
  }

  __host__ __device__ bool operator()(uint32_t l_idx, uint32_t r_idx) const {
    if (l_idx == INVALID_INDEX && r_idx == INVALID_INDEX)
      return false;
    if (l_idx == INVALID_INDEX)
      return true;
    if (r_idx == INVALID_INDEX)
      return false;

    const VoxelDistribution &l = buffer_ptr[l_idx].second;
    const VoxelDistribution &r = buffer_ptr[r_idx].second;

    double score_l = calculate_score(l.last_updated_at_scan, l.point_count);
    double score_r = calculate_score(r.last_updated_at_scan, r.point_count);

    // higher scores first
    return score_r < score_l;
  }
};

void Localmap::build(size_t scan_number, rm::CudaStream &stream) {
  if (this->is_empty()) {
    return;
  }

  this->lru_cursor = 0;

  ScoreComparator comparator{thrust::raw_pointer_cast(buffer.data()),
                             scan_number};

  thrust::sort(thrust::device.on(stream.handle()), lru_mapping.begin(),
               lru_mapping.begin() + add_cursor, comparator);
}

struct GetMean {
  __host__ __device__ Point
  operator()(const thrust::pair<uint32_t, VoxelDistribution> &val) const {
    return Point{val.second.mean.x, val.second.mean.y, val.second.mean.z};
  }
};

std::vector<Point> Localmap::get_map() const {
  if (this->is_empty()) {
    return {};
  }

  std::vector<Point> host_means(add_cursor);
  thrust::device_vector<Point> device_means(add_cursor);

  thrust::transform(buffer.begin(), buffer.begin() + add_cursor,
                    device_means.begin(), GetMean());

  thrust::copy(device_means.begin(), device_means.end(), host_means.begin());

  return host_means;
}

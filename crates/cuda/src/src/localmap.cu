#include <cassert>
#include <cfloat>
#include <cmath>
#include <core/common.cuh>
#include <core/cuda_stream.cuh>
#include <core/memory.cuh>
#include <core/memory_cuda.cuh>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <hashmap.cuh>
#include <lib.h>
#include <localmap.cuh>
#include <math/iso3.cuh>
#include <math/matrix3x3.cuh>
#include <math/quaternion.cuh>
#include <math/vector3.cuh>
#include <sys/types.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
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

__device__ void VoxelDistribution::lock() {
  // spin until we successfully acquire the lock by swapping 0 to 1
  while (atomicCAS(&this->lock_flag, 0, 1) != 0) {
    // add __nanosleep or other yield instructions here if contention is high
  }
}

__device__ void VoxelDistribution::unlock() {
  // ensure all memory writes are globally visible before releasing the lock
  __threadfence();
  // Release the lock by setting it back to 0
  atomicExch(&this->lock_flag, 0);
}

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

Localmap::Localmap(size_t capacity, float sensitivity, float min_dist,
                   float resolution, NearbyType nbtype, rm::CudaStream &stream)
    : capacity(capacity), sensitivity(sensitivity), min_dist(min_dist),
      add_cursor(0), lru_cursor(0), buffer(capacity), lru_mapping(capacity),
      resolution(resolution), nbtype(nbtype) {

  assert((stream.handle() != NULL));

  thrust::fill(thrust::device.on(stream.handle()), lru_mapping.begin(),
               lru_mapping.end(), INVALID_INDEX);
  // syncs the stream after initial copy
  lookup = create_map(capacity);
  stream.synchronize();
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
  float x = floorf(point.x * inverse_res);
  float y = floorf(point.y * inverse_res);
  float z = floorf(point.z * inverse_res);
  return rm::Vector3<int32_t>(static_cast<int32_t>(x), static_cast<int32_t>(y),
                              static_cast<int32_t>(z));
}

__device__ int32_t hash_grid(const rm::Vector3<int32_t> &grid_key) {
  return static_cast<int32_t>(static_cast<size_t>(((grid_key.x) * 73856093) ^
                                                  ((grid_key.y) * 471943) ^
                                                  ((grid_key.z) * 83492791)));
}

__global__ void calculate_keys_kernel(const Point *points, uint32_t point_size,
                                      Pose global_transform, float resolution,
                                      int32_t *d_keys, uint32_t *d_indices) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= point_size) {
    return;
  }

  Point point_raw = points[idx];
  rm::Pointf point_local = rm::Pointf(point_raw.x, point_raw.y, point_raw.z);
  rm::Quaternionf ekf_rot(
      global_transform.orientation.w, global_transform.orientation.x,
      global_transform.orientation.y, global_transform.orientation.z);
  rm::Vector3f ekf_trans(global_transform.position.x,
                         global_transform.position.y,
                         global_transform.position.z);
  rm::Isometry3f global_pose{ekf_rot, ekf_trans};
  rm::Pointf point = global_pose * point_local;
  auto cell = pos2grid(resolution, point);

  d_keys[idx] = hash_grid(cell);
  d_indices[idx] = idx;
}

__global__ void
voxel_krnl(const int32_t *unique_keys,
           const uint32_t *run_starts, // start index for each key points
           const uint32_t *run_ends,   // end index for each key points
           const Point *original_points, const uint32_t *sorted_indices,
           uint32_t unique_key_count, uint32_t scan_number,
           KeyValuePair *lookup, Value *buffer, uint32_t capacity,
           float resolution, float sensitivity, float min_dist,
           uint32_t *lru_cursor, uint32_t *add_cursor, uint32_t *lru_mapping,
           Pose global_transform) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= unique_key_count) {
    return;
  }

  int32_t key = unique_keys[idx];
  uint32_t start_offset = run_starts[idx];
  uint32_t end_offset = run_ends[idx];

  // inlined search in the hashmap
  uint32_t buffer_idx;
  bool found_existing = false;
  uint32_t slot = hash_function(key) & ((capacity * 2) - 1);
  while (true) {
    if (lookup[slot].key == key) {
      buffer_idx = lookup[slot].value;
      found_existing = true;
      break;
    }
    if (lookup[slot].key == EMPTY_KEY) {
      break;
    }
    slot = (slot + 1) & ((capacity * 2) - 1); // Linear probe
  }

  rm::Quaternionf ekf_rot(
      global_transform.orientation.w, global_transform.orientation.x,
      global_transform.orientation.y, global_transform.orientation.z);
  rm::Vector3f ekf_trans(global_transform.position.x,
                         global_transform.position.y,
                         global_transform.position.z);
  rm::Isometry3f global_pose{ekf_rot, ekf_trans};

  if (found_existing) {
    VoxelDistribution &voxel = buffer[buffer_idx].second;
    voxel.lock();

    for (uint32_t i = start_offset; i < end_offset; ++i) {
      uint32_t point_original_idx = sorted_indices[i];
      Point point_raw = original_points[point_original_idx];
      rm::Pointf point_local =
          rm::Pointf(point_raw.x, point_raw.y, point_raw.z);
      rm::Pointf point_global = global_pose * point_local;
      voxel.update(point_global, sensitivity, min_dist, scan_number);
    }
    voxel.unlock();

  } else {
    uint32_t claimed_idx = atomicAdd(add_cursor, 1);
    if (claimed_idx < capacity) {
      buffer_idx = claimed_idx;
      lru_mapping[buffer_idx] = buffer_idx;
    } else {
      // Buffer is full, get a victim's slot to overwrite from the LRU list.
      uint32_t lru_idx_local = atomicAdd(lru_cursor, 1);
      buffer_idx = lru_mapping[lru_idx_local & (capacity - 1)];
    }

    uint32_t first_point_original_idx = sorted_indices[start_offset];
    Point first_point_raw = original_points[first_point_original_idx];
    rm::Pointf first_point_local =
        rm::Pointf(first_point_raw.x, first_point_raw.y, first_point_raw.z);
    rm::Pointf first_point_global = global_pose * first_point_local;

    buffer[buffer_idx] =
        thrust::make_pair(key, VoxelDistribution(first_point_global));

    for (uint32_t i = start_offset + 1; i < end_offset; ++i) {
      uint32_t point_original_idx = sorted_indices[i];
      Point point_raw = original_points[point_original_idx];
      rm::Pointf point_local =
          rm::Pointf(point_raw.x, point_raw.y, point_raw.z);
      rm::Pointf point_global = global_pose * point_local;
      buffer[buffer_idx].second.update(point_global, sensitivity, min_dist,
                                       scan_number);
    }
  }
}

// assumes points is already on the gpu
void Localmap::add(Point *points, uint32_t point_size, uint32_t scan_number,
                   Pose registered_pose, rm::CudaStream &stream) {

  constexpr float BUFFER_ADDITIONAL = 0.2;
  uint32_t point_buffer_with_additional = static_cast<uint32_t>(
      static_cast<float>(point_size) * (1.0 + BUFFER_ADDITIONAL));
  this->d_unique_keys.resize(point_buffer_with_additional);
  this->d_run_counts.resize(point_buffer_with_additional);
  this->d_keys.resize(point_buffer_with_additional);
  this->d_indices.resize(point_buffer_with_additional);

  rm::Memory<uint32_t, rm::RAM> lru_cursor_host(1);
  *lru_cursor_host = this->lru_cursor;
  rm::Memory<uint32_t, rm::VRAM_CUDA> lru_cursor_device = lru_cursor_host;

  rm::Memory<uint32_t, rm::RAM> add_cursor_host(1);
  *add_cursor_host = this->add_cursor;
  rm::Memory<uint32_t, rm::VRAM_CUDA> add_cursor_device = add_cursor_host;

  // keys for all points
  int threads_per_block = 256;
  int blocks_per_grid =
      (point_size + threads_per_block - 1) / threads_per_block;
  calculate_keys_kernel<<<blocks_per_grid, threads_per_block, 0,
                          stream.handle()>>>(
      points, point_size, registered_pose, this->resolution,
      thrust::raw_pointer_cast(d_keys.data()),
      thrust::raw_pointer_cast(d_indices.data()));

  // sort by key to group points belonging to the same voxel
  thrust::sort_by_key(thrust::device.on(stream.handle()), d_keys.begin(),
                      d_keys.end(), d_indices.begin());

  // unique keys and the number of points for each
  auto new_end = thrust::reduce_by_key(
      thrust::device.on(stream.handle()), d_keys.begin(), d_keys.end(),
      thrust::make_constant_iterator(1), // Count 1 for each key
      this->d_unique_keys.begin(), this->d_run_counts.begin());

  uint32_t unique_key_count = new_end.first - d_unique_keys.begin();
  uint32_t unique_keys_with_additional = static_cast<uint32_t>(
      static_cast<float>(unique_key_count) * (1.0 + BUFFER_ADDITIONAL));
  this->d_unique_keys.resize(unique_keys_with_additional);
  this->d_run_counts.resize(unique_keys_with_additional);
  this->d_run_starts.resize(unique_keys_with_additional);
  this->d_run_ends.resize(unique_keys_with_additional);

  // run start/end indices for each unique key
  thrust::inclusive_scan(thrust::device.on(stream.handle()),
                         this->d_run_counts.begin(), this->d_run_counts.end(),
                         this->d_run_ends.begin());

  this->d_run_starts[0] = 0;
  thrust::transform(thrust::device.on(stream.handle()),
                    this->d_run_ends.begin(), this->d_run_ends.end() - 1,
                    this->d_run_starts.begin() + 1,
                    thrust::identity<uint32_t>());

  blocks_per_grid =
      (unique_key_count + threads_per_block - 1) / threads_per_block;
  voxel_krnl<<<blocks_per_grid, threads_per_block, 0, stream.handle()>>>(
      thrust::raw_pointer_cast(this->d_unique_keys.data()),
      thrust::raw_pointer_cast(this->d_run_starts.data()),
      thrust::raw_pointer_cast(this->d_run_ends.data()), points,
      thrust::raw_pointer_cast(d_indices.data()), unique_key_count, scan_number,
      thrust::raw_pointer_cast(this->lookup.get()->d_table.data()),
      thrust::raw_pointer_cast(this->buffer.data()), this->capacity,
      this->resolution, this->sensitivity, this->min_dist,
      lru_cursor_device.raw(), add_cursor_device.raw(),
      thrust::raw_pointer_cast(this->lru_mapping.data()), registered_pose);

  stream.synchronize();
  checkLastCudaError(__FILE__, __LINE__);

  add_cursor_host = add_cursor_device;
  this->add_cursor = *add_cursor_host;
  this->build(scan_number, stream);

  // rm::Memory could go out of scope before point_insert is accessing it
  stream.synchronize();
  checkLastCudaError(__FILE__, __LINE__);
}

__global__ void rebuild_hashmap_kernel(KeyValuePair *table, uint32_t capacity,
                                       const Value *buffer,
                                       uint32_t valid_size) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= valid_size) {
    return;
  }

  int32_t key = buffer[idx].first;
  uint32_t value_idx = idx;

  uint32_t slot = hash_function(key) & (capacity - 1);
  while (true) {
    // handle the rare case of hash collisions
    if (atomicCAS(&table[slot].key, EMPTY_KEY, key) == EMPTY_KEY) {
      table[slot].value = value_idx;
      return;
    }
    slot = (slot + 1) & (capacity - 1);
  }
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

    // First, compare by score. A higher score means it's "less than" in a
    // min-heap sense, so it comes first in a descending sort.
    if (score_l > score_r)
      return true;
    if (score_l < score_r)
      return false;

    // Tie-breaker: If scores are identical, use the unique index
    // to create a stable, deterministic order.
    return l_idx < r_idx;
  }
};
__global__ void initialize_kernel_a(KeyValuePair *table, uint32_t capacity) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < capacity) {
    table[idx].key = EMPTY_KEY;
    table[idx].value = 0;
  }
}

void Localmap::build(size_t scan_number, rm::CudaStream &stream) {
  if (this->is_empty()) {
    return;
  }

  this->lru_cursor = 0;

  int threads_per_block = 256;

  int blocks_per_grid =
      (this->lookup->capacity + threads_per_block - 1) / threads_per_block;
  initialize_kernel_a<<<blocks_per_grid, threads_per_block, 0,
                        stream.handle()>>>(
      thrust::raw_pointer_cast(this->lookup->d_table.data()),
      this->lookup->capacity);

  blocks_per_grid =
      (std::min(this->add_cursor, this->capacity) + threads_per_block - 1) /
      threads_per_block;

  rebuild_hashmap_kernel<<<blocks_per_grid, threads_per_block, 0,
                           stream.handle()>>>(
      thrust::raw_pointer_cast(this->lookup.get()->d_table.data()),
      this->lookup.get()->capacity,
      thrust::raw_pointer_cast(this->buffer.data()),
      this->add_cursor // Use the correct, updated size
  );

  ScoreComparator comparator{thrust::raw_pointer_cast(this->buffer.data()),
                             scan_number};

  thrust::sort(thrust::device.on(stream.handle()), this->lru_mapping.begin(),
               this->lru_mapping.begin() +
                   std::min(this->add_cursor, this->capacity),
               comparator);
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

  auto capped = std::min(this->add_cursor, this->capacity);
  std::vector<Point> host_means(capped);
  thrust::device_vector<Point> device_means(capped);

  thrust::transform(buffer.begin(), buffer.begin() + capped,
                    device_means.begin(), GetMean());

  thrust::copy(device_means.begin(), device_means.end(), host_means.begin());

  return host_means;
}

// A struct to hold 7 neighbors for nearby_6
struct Neighbors7 {
  rm::Vector3i data[7];
};

// A struct to hold 19 neighbors for nearby_18
struct Neighbors19 {
  rm::Vector3i data[19];
};

// A struct to hold 27 neighbors for nearby_26
struct Neighbors27 {
  rm::Vector3i data[27];
};

/**
 * TODO Von Neumann Neighborhood?
 * @brief Returns the 7 grid coordinates for the center and face neighbors.
 * @param grid_key The center grid coordinate (gx, gy, gz).
 * @return A struct containing a fixed-size array of 7 coordinates.
 */
__device__ inline Neighbors7 nearby_6(const rm::Vector3i &grid_key) {
  const int gx = grid_key.x;
  const int gy = grid_key.y;
  const int gz = grid_key.z;
  const Neighbors7 result = {{
      {gx, gy, gz},
      {gx - 1, gy, gz},
      {gx + 1, gy, gz},
      {gx, gy + 1, gz},
      {gx, gy - 1, gz},
      {gx, gy, gz - 1},
      {gx, gy, gz + 1},
  }};
  return result;
}

/**
 * TODO Edge Neighbors ?
 * @brief Returns the 19 grid coordinates for the center, face, and edge
 * neighbors.
 * @param grid_key The center grid coordinate (gx, gy, gz).
 * @return A struct containing a fixed-size array of 19 coordinates.
 */
__device__ inline Neighbors19 nearby_18(const rm::Vector3i &grid_key) {
  const int gx = grid_key.x;
  const int gy = grid_key.y;
  const int gz = grid_key.z;
  const Neighbors19 result = {{
      {gx, gy, gz},         {gx - 1, gy, gz},     {gx + 1, gy, gz},
      {gx, gy + 1, gz},     {gx, gy - 1, gz},     {gx, gy, gz - 1},
      {gx, gy, gz + 1},     {gx + 1, gy + 1, gz}, {gx - 1, gy + 1, gz},
      {gx + 1, gy - 1, gz}, {gx - 1, gy - 1, gz}, {gx + 1, gy, gz + 1},
      {gx - 1, gy, gz + 1}, {gx + 1, gy, gz - 1}, {gx - 1, gy, gz - 1},
      {gx, gy + 1, gz + 1}, {gx, gy - 1, gz + 1}, {gx, gy + 1, gz - 1},
      {gx, gy - 1, gz - 1},
  }};
  return result;
}

/**
 * TODO Moore Neighborhood sharing face, edge or corner ?
 * @brief Returns the 27 grid coordinates for a full 3x3x3 neighborhood.
 * @param grid_key The center grid coordinate (gx, gy, gz).
 * @return A struct containing a fixed-size array of 27 coordinates.
 */
__device__ inline Neighbors27 nearby_26(const rm::Vector3i &grid_key) {
  const int gx = grid_key.x;
  const int gy = grid_key.y;
  const int gz = grid_key.z;
  const Neighbors27 result = {{
      {gx, gy, gz},
      {gx - 1, gy, gz},
      {gx + 1, gy, gz},
      {gx, gy + 1, gz},
      {gx, gy - 1, gz},
      {gx, gy, gz - 1},
      {gx, gy, gz + 1},
      {gx + 1, gy + 1, gz},
      {gx - 1, gy + 1, gz},
      {gx + 1, gy - 1, gz},
      {gx - 1, gy - 1, gz},
      {gx + 1, gy, gz + 1},
      {gx - 1, gy, gz + 1},
      {gx + 1, gy, gz - 1},
      {gx - 1, gy, gz - 1},
      {gx, gy + 1, gz + 1},
      {gx, gy - 1, gz + 1},
      {gx, gy + 1, gz - 1},
      {gx, gy - 1, gz - 1},
      {gx + 1, gy + 1, gz + 1},
      {gx - 1, gy + 1, gz + 1},
      {gx + 1, gy - 1, gz + 1},
      {gx + 1, gy + 1, gz - 1},
      {gx - 1, gy - 1, gz + 1},
      {gx - 1, gy + 1, gz - 1},
      {gx + 1, gy - 1, gz - 1},
      {gx - 1, gy - 1, gz - 1},
  }};
  return result;
}

__device__ void
cache_and_sum_density(rm::Vector3i *nb, uint32_t size, KeyValuePair *map,
                      uint32_t capacity, uint32_t &total_density,
                      uint32_t &num_cached,
                      thrust::pair<uint32_t, VoxelDistribution> *buffer_ptr,
                      VoxelDistribution *out) {
  for (uint32_t i = 0; i < size; i++) {
    auto cnb = nb[i];
    auto k = hash_grid(cnb);
    auto found = false;
    auto nb_buffer_idx = lookup_item(map, capacity, k, found);

    if (found) {
      auto dist = buffer_ptr[nb_buffer_idx].second;
      if (dist.point_count > 0 && num_cached < MAX_NEIGHBORS) {
        total_density += dist.point_count;
        num_cached += 1;
        out[num_cached] = dist;
      }
    }
  }
}

__device__ void sample_from_distribution(const VoxelDistribution &dist,
                                         uint32_t num_samples,
                                         curandState_t &rng,
                                         rm::Pointf *result) {
  assert((num_samples < MAX_NEIGHBORS));

  for (uint32_t i = 0; i < num_samples; i++) {
    rm::Pointf random_point(curand_normal(&rng), curand_normal(&rng),
                            curand_normal(&rng));

    result[i] = dist.mean + dist.transform * random_point;
  }
}

__device__ bool
k_nearest_neighbor_query(const rm::Pointf &query, uint32_t k, NearbyType nbtype,
                         float max_range, KeyValuePair *map, uint32_t capacity,
                         thrust::pair<uint32_t, VoxelDistribution> *buffer,
                         float resolution, rm::Pointf *result,
                         uint32_t &out_size, time_t time, int thread_id) {
  assert((k != 0 && result != NULL));

  VoxelDistribution cached_neighbors[MAX_NEIGHBORS];
  uint32_t num_cached = 0;
  uint32_t total_density = 0;

  auto grid = pos2grid(resolution, query);
  switch (nbtype) {
  case NearbyType::Center: {
    cache_and_sum_density(&grid, 1, map, capacity, total_density, num_cached,
                          buffer, cached_neighbors);
    break;
  }
  case NearbyType::Nearby6: {
    auto nb = nearby_6(grid);
    cache_and_sum_density(nb.data, 7, map, capacity, total_density, num_cached,
                          buffer, cached_neighbors);
    break;
  }
  case NearbyType::Nearby18: {
    auto nb = nearby_18(grid);
    cache_and_sum_density(nb.data, 19, map, capacity, total_density, num_cached,
                          buffer, cached_neighbors);
    break;
  }
  case NearbyType::Nearby26: {
    auto nb = nearby_26(grid);
    cache_and_sum_density(nb.data, 27, map, capacity, total_density, num_cached,
                          buffer, cached_neighbors);
    break;
  }
  }

  if (total_density == 0) {
    return false;
  }

  // Proportional Sampling
  uint32_t current_found_count = 0;
  curandState_t rng;
  curand_init(time, thread_id, 0, &rng);

  rm::Pointf sampled_out[MAX_NEIGHBORS];
  auto sq_max_range = max_range * max_range;
  for (uint32_t i = 0; i < num_cached; i++) {
    if (current_found_count >= k) {
      break;
    }

    const VoxelDistribution *dist = &cached_neighbors[i];
    auto relative_density = static_cast<double>(dist->point_count) /
                            static_cast<double>(total_density);
    auto sample_count =
        static_cast<uint32_t>(round(relative_density * static_cast<double>(k)));

    uint32_t samples_needed = k - current_found_count;
    sample_count = min(sample_count, samples_needed);

    if (sample_count == 0) {
      continue;
    }

    sample_from_distribution(*dist, sample_count, rng, sampled_out);

    for (uint32_t i = 0; i < sample_count; i++) {
      auto p = sampled_out[i];
      auto dist = (query - p).l2normSquared();
      if (dist < sq_max_range) {
        result[current_found_count++] = p;
        if (current_found_count >= k) {
          break;
        }
      }
    }
  }

  out_size = current_found_count;
  return out_size > 0;
}

__device__ inline double
plane_normal_point_distance(const double4 &plane, const rm::Vector3d &point) {
  return plane.x * point.x + plane.y * point.y + plane.z * point.z + plane.w;
}

__device__ inline double4
plane_from_point_and_normal(const rm::Vector3d &point,
                            const rm::Vector3d &normal) {
  return {normal.x, normal.y, normal.z,
          -(normal.x * point.x + normal.y * point.y + normal.z * point.z)};
}

__device__ inline float plane_normal_point_distance(const float4 &plane,
                                                    const rm::Vector3f &point) {
  return plane.x * point.x + plane.y * point.y + plane.z * point.z + plane.w;
}

__device__ inline float4
plane_from_point_and_normal(const rm::Vector3f &point,
                            const rm::Vector3f &normal) {
  return {normal.x, normal.y, normal.z,
          -(normal.x * point.x + normal.y * point.y + normal.z * point.z)};
}

__device__ inline rm::Vector3d calc_centroid(const rm::Vector3f *cluster,
                                             size_t cluster_size) {
  rm::Vector3d sum(0, 0, 0);
  for (size_t i = 0; i < cluster_size; i++) {
    sum.addInplace(rm::Vector3d(cluster[i].x, cluster[i].y, cluster[i].z));
  }

  return sum * (1.0 / static_cast<double>(cluster_size));
}

__device__ inline rm::Vector3f calc_centroid_f(const rm::Vector3f *cluster,
                                               size_t cluster_size) {
  rm::Vector3f sum(0, 0, 0);
  for (size_t i = 0; i < cluster_size; i++) {
    sum.addInplace(cluster[i]);
  }

  return sum * (1.0 / static_cast<double>(cluster_size));
}

__device__ inline rm::Matrix3x3f calc_cov(const rm::Vector3f *cluster,
                                          size_t cluster_size,
                                          const rm::Vector3f &centroid) {
  float xx = 0.0, xy = 0.0, xz = 0.0, yy = 0.0, yz = 0.0, zz = 0.0;

  for (size_t i = 0; i < cluster_size; i++) {
    rm::Vector3f r = cluster[i] - centroid;
    xx += r.x * r.x;
    xy += r.x * r.y;
    xz += r.x * r.z;
    yy += r.y * r.y;
    yz += r.y * r.z;
    zz += r.z * r.z;
  }

  xx /= static_cast<float>(cluster_size);
  xy /= static_cast<float>(cluster_size);
  xz /= static_cast<float>(cluster_size);
  yy /= static_cast<float>(cluster_size);
  yz /= static_cast<float>(cluster_size);
  zz /= static_cast<float>(cluster_size);

  rm::Matrix3x3f cov;
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

__device__ inline rm::Matrix3x3d calc_cov(const rm::Vector3f *cluster,
                                          size_t cluster_size,
                                          const rm::Vector3d &centroid) {
  double xx = 0.0, xy = 0.0, xz = 0.0, yy = 0.0, yz = 0.0, zz = 0.0;

  for (size_t i = 0; i < cluster_size; i++) {
    rm::Vector3d r =
        rm::Vector3d(cluster[i].x, cluster[i].y, cluster[i].z) - centroid;
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

  rm::Matrix3x3d cov;
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

__device__ void normalize_safe(rm::Vector3f &v) {
  float mag = sqrtf(v.dot(v));
  // Add the safety check
  if (mag > 1e-9f) { // Use a small epsilon
    v = v * (1.0f / mag);
  } else {
    v.x = 0.0f;
    v.y = 0.0f;
    v.z = 0.0f;
  }
}

/**
 * @brief Fits a plane to a point cloud using Principal Component Analysis (PCA)
 * via robust eigen-decomposition.
 *
 * This method calculates the covariance matrix of the points, finds its
 * eigenvalues and eigenvectors. The eigenvector corresponding to the smallest
 * eigenvalue is the normal of the best-fit plane.
 *
 * @param cluster Pointer to the array of 3D points.
 * @param n Number of points in the cluster.
 * @param normal_threshold The maximum distance a point can be from the
 * fitted plane for the fit to be considered successful.
 * @param plane_equation Output reference to a float4 representing the plane.
 * @return True if a stable plane was found, false otherwise.
 */
__device__ bool pca_eigen(const rm::Vector3f *cluster, uint32_t n,
                          float normal_threshold, float4 &plane_equation) {
  if (n < 3) {
    return false;
  }

  rm::Vector3f centroid = calc_centroid_f(cluster, n);
  rm::Matrix3x3f cov = calc_cov(cluster, n, centroid);

  rm::Vector3f eigenvalues = symmetric_eigen_3x3(cov);

  // smallest eigenvalue
  float min_eigenvalue = eigenvalues.x;
  if (eigenvalues.y < min_eigenvalue) {
    min_eigenvalue = eigenvalues.y;
  }
  if (eigenvalues.z < min_eigenvalue) {
    min_eigenvalue = eigenvalues.z;
  }

  // Find the eigenvector for the smallest eigenvalue.
  // We solve (cov - min_eigenvalue * I) * v = 0.
  // The eigenvector 'v' is in the null space of the matrix M = cov - λI.
  // A robust way to find a vector in the null space is to take the cross
  // product of two non-parallel rows of M. We compute all three cross
  // products and pick the one with the largest magnitude for best stability.
  rm::Matrix3x3f M = cov;
  M.at(0, 0) -= min_eigenvalue;
  M.at(1, 1) -= min_eigenvalue;
  M.at(2, 2) -= min_eigenvalue;

  rm::Vector3f row0(M.at(0, 0), M.at(0, 1), M.at(0, 2));
  rm::Vector3f row1(M.at(1, 0), M.at(1, 1), M.at(1, 2));
  rm::Vector3f row2(M.at(2, 0), M.at(2, 1), M.at(2, 2));

  rm::Vector3f v1 = row0.cross(row1);
  rm::Vector3f v2 = row1.cross(row2);
  rm::Vector3f v3 = row2.cross(row0);

  float mag1_sq = v1.dot(v1);
  float mag2_sq = v2.dot(v2);
  float mag3_sq = v3.dot(v3);

  rm::Vector3f normal_vec;
  if (mag1_sq > mag2_sq && mag1_sq > mag3_sq) {
    normal_vec = v1;
  } else if (mag2_sq > mag3_sq) {
    normal_vec = v2;
  } else {
    normal_vec = v3;
  }

  // Normalize the resulting normal vector (with a safety check)
  // float normal_mag_sq = normal_vec.dot(normal_vec);
  // if (normal_mag_sq < 1e-12f) {
  //   // This can happen with degenerate point configurations (e.g., spherical)
  //   // where eigenvalues are repeated, or due to numerical instability.
  //   return false;
  // }
  normalize_safe(normal_vec);
  // normal_vec.normalize();

  plane_equation = plane_from_point_and_normal(centroid, normal_vec);

  for (uint32_t i = 0; i < n; i++) {
    if (fabs(plane_normal_point_distance(plane_equation, cluster[i])) >
        normal_threshold) {
      return false;
    }
  }

  return true;
}

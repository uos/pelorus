#define THRUST_CUB_WRAPPED_NAMESPACE pelorus_multires_cccl
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <localmap.cuh>
#include <thrust/copy.h>
#include <thrust/merge.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <vector>

namespace thrust = pelorus_multires_cccl::thrust;

#define MR_CHECK(expr)                                                         \
  do {                                                                         \
    auto e = (expr);                                                           \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA multiresolution %s:%d: %s\n", __FILE__, __LINE__,  \
              cudaGetErrorString(e));                                          \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

namespace mr {
template <class T> struct Buffer {
  T *p = nullptr;
  size_t capacity = 0;
  Buffer() = default;
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  ~Buffer() {
    if (p)
      cudaFree(p);
  }
  void reserve(size_t n) {
    if (n <= capacity)
      return;
    if (p)
      MR_CHECK(cudaFree(p));
    capacity = n + n / 2 + 1;
    MR_CHECK(cudaMalloc(&p, capacity * sizeof(T)));
  }
  void swap(Buffer &other) {
    std::swap(p, other.p);
    std::swap(capacity, other.capacity);
  }
};
struct Dist {
  uint64_t count;
  float mean[3], cov[9];
  uint32_t scan;
};
struct Cell {
  int3 key;
  Dist d;
};
struct Sample {
  int3 key;
  float p[3];
  bool valid;
};
struct Fit {
  // 0: insufficient, 1: conflicting, 2: plane
  int status;
  float mean[3], normal[3], tangent[2][3], extent[2], d;
};
// Fit of a cell that is absent from the map, cached for one map version.
// state = version << 1 | ready; a slot of an older version counts as empty, so
// the cache never has to be cleared.
struct MissSlot {
  uint32_t state;
  int3 key;
  Fit fit;
};
struct MissWork {
  uint32_t level, slot;
  int3 key;
};
struct Level {
  Cell *cells;
  Fit *fits;
  int *table;
  uint32_t n, table_n;
  float resolution;
  MissSlot *miss;
  uint32_t miss_n;
};
struct Storage {
  // *_next hold the level while a scan is merged, so unchanged fits can be
  // carried over from the previous version instead of being recomputed.
  Buffer<Cell> cells, cells_next;
  Buffer<Fit> fits, fits_next;
  Buffer<int> table, table_next;
  Buffer<unsigned char> dirty;
  Buffer<MissSlot> miss;
  uint32_t n = 0, budget = 0, table_n = 0, miss_n = 0;
};
struct EvictKey;
struct State {
  Storage level[4];
  float base, sensitivity;
  uint32_t capacity;
  // Map version for the miss caches, advanced after every map update.
  uint32_t version = 1;
  Buffer<Sample> samples, valid_samples;
  Buffer<Cell> batches, parents, scratch, merged;
  Buffer<PointNormal> planes;
  Buffer<unsigned char> flags;
  Buffer<MissWork> work;
  Buffer<uint32_t> counters;
  Buffer<EvictKey> evict;
  Buffer<unsigned char> evicted;
  Buffer<Cell> evicted_cells;
};
static State *state = nullptr;
__host__ __device__ bool less(int3 a, int3 b) {
  return a.x != b.x ? a.x < b.x : a.y != b.y ? a.y < b.y : a.z < b.z;
}
__host__ __device__ bool equal(int3 a, int3 b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
struct CellLess {
  __host__ __device__ bool operator()(const Cell &a, const Cell &b) const {
    return less(a.key, b.key);
  }
};
struct SampleLess {
  __host__ __device__ bool operator()(const Sample &a, const Sample &b) const {
    return less(a.key, b.key);
  }
};
struct ValidSample {
  __host__ __device__ bool operator()(const Sample &a) const { return a.valid; }
};
struct ValidCell {
  __host__ __device__ bool operator()(const Cell &a) const {
    return a.d.count != 0;
  }
};
struct Flag {
  __host__ __device__ bool operator()(unsigned char a) const { return a != 0; }
};
__device__ float dot(const float *a, const float *b) {
  return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
}
__device__ int coordinate(float p, float r) {
  float v = floorf(p / r);
  if (v >= float(INT_MAX))
    return INT_MAX;
  if (v <= float(INT_MIN))
    return INT_MIN;
  return int(v);
}
__device__ int3 grid(const float *p, float r) {
  return make_int3(coordinate(p[0], r), coordinate(p[1], r),
                   coordinate(p[2], r));
}
__device__ int parent(int x) { return x / 2 - (x < 0 && x % 2 != 0); }
__device__ int neighbor(int x, int d) {
  return d > 0 && x == INT_MAX   ? INT_MAX
         : d < 0 && x == INT_MIN ? INT_MIN
                                 : x + d;
}
__device__ uint32_t hash(int3 k) {
  uint32_t h = uint32_t(k.x) * 73856093u + uint32_t(k.y) * 471943u +
               uint32_t(k.z) * 83492791u;
  h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  h *= 0xc2b2ae35u;
  return h ^ (h >> 16);
}
__device__ int lookup(Level l, int3 key) {
  uint32_t slot = hash(key) & (l.table_n - 1);
  for (uint32_t i = 0; i < l.table_n; ++i) {
    int index = l.table[slot];
    if (index < 0)
      return -1;
    if (equal(l.cells[index].key, key))
      return index;
    slot = (slot + 1) & (l.table_n - 1);
  }
  return -1;
}
__global__ void index_cells(Level l) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= l.n)
    return;
  uint32_t slot = hash(l.cells[i].key) & (l.table_n - 1);
  while (atomicCAS(l.table + slot, -1, int(i)) != -1)
    slot = (slot + 1) & (l.table_n - 1);
}
__device__ void pool(Dist &a, const Dist &b) {
  if (!b.count)
    return;
  if (!a.count) {
    a = b;
    return;
  }
  float n1 = float(a.count), n2 = float(b.count), n = n1 + n2, delta[3];
  for (int i = 0; i < 3; ++i)
    delta[i] = a.mean[i] - b.mean[i];
  for (int i = 0; i < 3; ++i)
    a.mean[i] = (n1 * a.mean[i] + n2 * b.mean[i]) / n;
  float w = n1 * n2 / n;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      a.cov[3 * r + c] = (n1 * a.cov[3 * r + c] + n2 * b.cov[3 * r + c] +
                          delta[r] * delta[c] * w) /
                         n;
  a.count = uint64_t(n);
}
__device__ void update(Dist &a, const Dist &b, float sensitivity) {
  a.scan = b.scan;
  float delta[3];
  for (int i = 0; i < 3; ++i)
    delta[i] = a.mean[i] - b.mean[i];
  float l00 = sqrtf(a.cov[0] + 1e-6f), l10 = a.cov[3] / l00,
        l20 = a.cov[6] / l00;
  float l11 = sqrtf(a.cov[4] + 1e-6f - l10 * l10),
        l21 = (a.cov[7] - l20 * l10) / l11;
  float l22 = sqrtf(a.cov[8] + 1e-6f - (l20 * l20 + l21 * l21));
  float m = dot(delta, delta);
  if (isfinite(l00) && isfinite(l11) && isfinite(l22) && l00 > 0 && l11 > 0 &&
      l22 > 0) {
    float y0 = delta[0] / l00, y1 = (delta[1] - l10 * y0) / l11,
          y2 = (delta[2] - (l20 * y0 + l21 * y1)) / l22;
    float x2 = y2 / l22, x1 = (y1 - l21 * x2) / l11,
          x0 = (y0 - (l10 * x1 + l20 * x2)) / l00;
    m = (delta[0] * x0 + delta[1] * x1) + delta[2] * x2;
  }
  float alpha =
      expf(-m / (2 * (sensitivity / fmaxf(logf(float(a.count)), 1.0f))));
  if (alpha < 1e-5f)
    return;
  float oma = 1 - alpha;
  for (int i = 0; i < 3; ++i)
    a.mean[i] = oma * a.mean[i] + alpha * b.mean[i];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      a.cov[3 * r + c] = oma * a.cov[3 * r + c] + alpha * b.cov[3 * r + c] +
                         alpha * oma * (delta[r] * delta[c]);
  a.count += b.count;
}
__global__ void transform_samples(const Point *points, uint32_t n, Pose pose,
                                  float resolution, Sample *out) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  double x = points[i].x, y = points[i].y, z = points[i].z;
  auto q = pose.orientation;
  double tx = 2 * (q.y * z - q.z * y), ty = 2 * (q.z * x - q.x * z),
         tz = 2 * (q.x * y - q.y * x);
  Sample s;
  s.p[0] = float(x + q.w * tx + (q.y * tz - q.z * ty) + pose.position.x);
  s.p[1] = float(y + q.w * ty + (q.z * tx - q.x * tz) + pose.position.y);
  s.p[2] = float(z + q.w * tz + (q.x * ty - q.y * tx) + pose.position.z);
  s.valid = isfinite(s.p[0]) && isfinite(s.p[1]) && isfinite(s.p[2]);
  s.key = s.valid ? grid(s.p, resolution) : make_int3(0, 0, 0);
  out[i] = s;
}
__global__ void scan_cells(const Sample *s, uint32_t n, uint32_t scan,
                           Cell *out) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  Cell cell{};
  cell.key = s[i].key;
  if (i && equal(s[i - 1].key, s[i].key)) {
    out[i] = cell;
    return;
  }
  uint32_t end = i + 1;
  while (end < n && equal(s[i].key, s[end].key))
    ++end;
  cell.d.count = end - i;
  cell.d.scan = scan;
  float count = float(end - i);
  for (uint32_t j = i; j < end; ++j)
    for (int a = 0; a < 3; ++a)
      cell.d.mean[a] += s[j].p[a];
  for (int a = 0; a < 3; ++a)
    cell.d.mean[a] /= count;
  for (uint32_t j = i; j < end; ++j) {
    float delta[3];
    for (int a = 0; a < 3; ++a)
      delta[a] = s[j].p[a] - cell.d.mean[a];
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        cell.d.cov[3 * a + b] += delta[a] * delta[b];
  }
  for (int a = 0; a < 9; ++a)
    cell.d.cov[a] /= count;
  out[i] = cell;
}
__global__ void parent_keys(Cell *cells, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int3 k = cells[i].key;
  cells[i].key = make_int3(parent(k.x), parent(k.y), parent(k.z));
}
__global__ void combine(const Cell *in, uint32_t n, Cell *out, bool temporal,
                        float sensitivity) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  Cell c{};
  c.key = in[i].key;
  if (i && equal(in[i - 1].key, c.key)) {
    out[i] = c;
    return;
  }
  c = in[i];
  for (uint32_t j = i + 1; j < n && equal(c.key, in[j].key); ++j) {
    if (temporal)
      update(c.d, in[j].d, sensitivity);
    else
      pool(c.d, in[j].d);
  }
  out[i] = c;
}
// The fields eviction ranks by, so it sorts 32 instead of 80 bytes per cell.
struct EvictKey {
  uint64_t count;
  uint32_t age;
  int3 key;
  uint32_t index;
};
// Exact 128-bit age/support products, including a coordinate tie break.
struct EvictKeyLess {
  __host__ __device__ bool operator()(const EvictKey &a,
                                      const EvictKey &b) const {
    uint64_t age_a = a.age, age_b = b.age;
    uint64_t lo_a = age_a * b.count, lo_b = age_b * a.count, hi_a, hi_b;
#ifdef __CUDA_ARCH__
    hi_a = __umul64hi(age_a, b.count);
    hi_b = __umul64hi(age_b, a.count);
#else
    hi_a = uint64_t((__uint128_t(age_a) * b.count) >> 64);
    hi_b = uint64_t((__uint128_t(age_b) * a.count) >> 64);
#endif
    if (hi_a != hi_b)
      return hi_a > hi_b;
    return lo_a != lo_b ? lo_a > lo_b : less(a.key, b.key);
  }
};
__global__ void evict_keys(const Cell *cells, uint32_t n, uint32_t scan,
                           EvictKey *out) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const Cell &c = cells[i];
  out[i] = EvictKey{c.d.count, scan >= c.d.scan ? scan - c.d.scan : 0u, c.key,
                    i};
}
// Flags the evicted cells and lists them for mark_dirty.
__global__ void flag_evicted(const EvictKey *keys, uint32_t excess,
                             const Cell *cells, unsigned char *evicted,
                             Cell *evicted_cells) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= excess)
    return;
  evicted[keys[i].index] = 1;
  evicted_cells[i] = cells[keys[i].index];
}
struct NotFlagged {
  __host__ __device__ bool operator()(unsigned char a) const { return a == 0; }
};


__device__ Fit fit(Dist d, float base) {
  Fit f{};
  if (d.count < 6)
    return f;
  f.status = 1;
  double a[3][3], v[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) {
      if (!isfinite(d.cov[3 * r + c]))
        return f;
      a[r][c] = d.cov[3 * r + c];
    }
  for (int it = 0; it < 24; ++it) {
    int p = 0, q = 1;
    if (fabs(a[0][2]) > fabs(a[p][q])) {
      p = 0;
      q = 2;
    }
    if (fabs(a[1][2]) > fabs(a[p][q])) {
      p = 1;
      q = 2;
    }
    if (fabs(a[p][q]) <
        1e-14 * (fabs(a[0][0]) + fabs(a[1][1]) + fabs(a[2][2]) + 1e-20))
      break;
    double tau = (a[q][q] - a[p][p]) / (2 * a[p][q]);
    double t = copysign(1.0, tau) / (fabs(tau) + sqrt(1 + tau * tau)),
           c = 1 / sqrt(1 + t * t), s = t * c;
    double app = a[p][p], aqq = a[q][q], apq = a[p][q];
    a[p][p] = app - t * apq;
    a[q][q] = aqq + t * apq;
    a[p][q] = a[q][p] = 0;
    for (int k = 0; k < 3; ++k) {
      if (k != p && k != q) {
        double kp = a[k][p], kq = a[k][q];
        a[k][p] = a[p][k] = c * kp - s * kq;
        a[k][q] = a[q][k] = s * kp + c * kq;
      }
      double vp = v[k][p], vq = v[k][q];
      v[k][p] = c * vp - s * vq;
      v[k][q] = s * vp + c * vq;
    }
  }
  int idx[3] = {0, 1, 2};
  for (int i = 0; i < 3; ++i)
    for (int j = i + 1; j < 3; ++j)
      if (a[idx[j]][idx[j]] < a[idx[i]][idx[i]]) {
        int t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
      }
  float small = fmaxf(float(a[idx[0]][idx[0]]), 0),
        middle = float(a[idx[1]][idx[1]]), large = float(a[idx[2]][idx[2]]);
  if (!isfinite(middle) || !isfinite(large))
    return f;
  if (middle < 1e-4f) {
    f.status = 0;
    return f;
  }
  float thickness = base * .15f;
  if (small > thickness * thickness || small > .1f * middle)
    return f;
  f.status = 2;
  for (int k = 0; k < 3; ++k) {
    f.mean[k] = d.mean[k];
    f.normal[k] = float(v[k][idx[0]]);
    f.tangent[0][k] = float(v[k][idx[1]]);
    f.tangent[1][k] = float(v[k][idx[2]]);
  }
  f.extent[0] = 3 * sqrtf(fmaxf(middle, 0)) + base * .5f;
  f.extent[1] = 3 * sqrtf(fmaxf(large, 0)) + base * .5f;
  f.d = -dot(f.normal, f.mean);
  return f;
}
__device__ Fit neighborhood(Level l, int3 key, float base) {
  Dist d{};
  for (int x = -1; x <= 1; ++x)
    for (int y = -1; y <= 1; ++y)
      for (int z = -1; z <= 1; ++z) {
        int i = lookup(l, make_int3(neighbor(key.x, x), neighbor(key.y, y),
                                    neighbor(key.z, z)));
        if (i >= 0)
          pool(d, l.cells[i].d);
      }
  return fit(d, base);
}
// A fit only depends on the 27 cells around its key. Marks every cell of the
// new level version next to a cell that was updated, inserted or evicted.
__global__ void mark_dirty(const Cell *changed, uint32_t n, Level next,
                           unsigned char *dirty) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int3 key = changed[i].key;
  for (int x = -1; x <= 1; ++x)
    for (int y = -1; y <= 1; ++y)
      for (int z = -1; z <= 1; ++z) {
        int j = lookup(next, make_int3(neighbor(key.x, x), neighbor(key.y, y),
                                       neighbor(key.z, z)));
        if (j >= 0)
          dirty[j] = 1;
      }
}
// Carries unchanged fits over from the previous level version and refits the
// rest. Gives the same fits as refitting every cell of the new version.
__global__ void refresh_fits(Level prev, Level next, const unsigned char *dirty,
                             float base) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= next.n)
    return;
  int3 key = next.cells[i].key;
  if (!dirty[i]) {
    int j = lookup(prev, key);
    if (j >= 0) {
      next.fits[i] = prev.fits[j];
      return;
    }
  }
  next.fits[i] = neighborhood(next, key, base);
}
// Neighborhood fits of absent cells do not change while the map is unchanged,
// but the iterated correction queries them up to max_iter + 1 times per scan.
constexpr int MISS_PROBES = 32;
__device__ const Fit *cached_miss(Level l, int3 key, uint32_t version) {
  uint32_t slot = hash(key) & (l.miss_n - 1);
  for (int probe = 0; probe < MISS_PROBES; ++probe) {
    uint32_t st = l.miss[slot].state;
    if ((st >> 1) != version)
      return nullptr;
    if ((st & 1) && equal(l.miss[slot].key, key))
      return &l.miss[slot].fit;
    slot = (slot + 1) & (l.miss_n - 1);
  }
  return nullptr;
}
__device__ void transform_f32(Pose pose, const float *p, float *out) {
  float x = float(pose.orientation.x), y = float(pose.orientation.y),
        z = float(pose.orientation.z), w = float(pose.orientation.w);
  float tx = (y * p[2] - z * p[1]) * 2, ty = (z * p[0] - x * p[2]) * 2,
        tz = (x * p[1] - y * p[0]) * 2;
  out[0] = p[0] + w * tx + (y * tz - z * ty) + float(pose.position.x);
  out[1] = p[1] + w * ty + (z * tx - x * tz) + float(pose.position.y);
  out[2] = p[2] + w * tz + (x * ty - y * tx) + float(pose.position.z);
}
// Pass 1: claims a miss-cache slot for every absent cell a point could query on
// any level and lists it once, so pass 2 computes each fit a single time in a
// kernel where all threads take the same path.
__global__ void collect_misses(const Point *cloud, uint32_t n, Pose ext,
                               Pose pose, Level l0, Level l1, Level l2,
                               Level l3, uint32_t version, MissWork *work,
                               uint32_t work_cap, uint32_t *work_count) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float local[3] = {cloud[i].x, cloud[i].y, cloud[i].z}, imu[3], world[3];
  transform_f32(ext, local, imu);
  transform_f32(pose, imu, world);
  if (!isfinite(world[0]) || !isfinite(world[1]) || !isfinite(world[2]))
    return;
  Level levels[4] = {l0, l1, l2, l3};
  for (int level = 0; level < 4; ++level) {
    Level l = levels[level];
    int3 k = grid(world, l.resolution);
    if (lookup(l, k) >= 0)
      continue;
    uint32_t slot = hash(k) & (l.miss_n - 1);
    for (int probe = 0; probe < MISS_PROBES; ++probe) {
      volatile MissSlot *s = l.miss + slot;
      uint32_t st = s->state;
      if ((st >> 1) != version) {
        // Empty for this map version: claim it, still not ready.
        if (atomicCAS(&l.miss[slot].state, st, version << 1) == st) {
          l.miss[slot].key = k;
          uint32_t w = atomicAdd(work_count, 1u);
          // A full list leaves the slot unready; pass 3 then fits inline.
          if (w < work_cap)
            work[w] = MissWork{uint32_t(level), slot, k};
          break;
        }
        continue; // lost the race for this slot, look at it again
      }
      // Claimed in this version. A slot claimed in this launch may still show
      // an older key, which at worst lists a key twice or leaves it to pass 3.
      if (s->key.x == k.x && s->key.y == k.y && s->key.z == k.z)
        break;
      slot = (slot + 1) & (l.miss_n - 1);
    }
  }
}
// Pass 2: one fit per listed absent cell.
__global__ void compute_misses(const MissWork *work, const uint32_t *work_count,
                               uint32_t work_cap, Level l0, Level l1, Level l2,
                               Level l3, float base, uint32_t version) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= min(*work_count, work_cap))
    return;
  MissWork w = work[i];
  Level levels[4] = {l0, l1, l2, l3};
  Level l = levels[w.level];
  l.miss[w.slot].fit = neighborhood(l, w.key, base);
  l.miss[w.slot].state = (version << 1) | 1;
}
// Pass 3: plane selection, reading absent-cell fits from the cache.
__global__ void query(const Point *cloud, uint32_t n, Pose ext, Pose pose,
                      Level l0, Level l1, Level l2, Level l3, float base,
                      uint32_t version, PointNormal *out,
                      unsigned char *flags) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  flags[i] = 0;
  float local[3] = {cloud[i].x, cloud[i].y, cloud[i].z}, imu[3], world[3];
  transform_f32(ext, local, imu);
  transform_f32(pose, imu, world);
  if (!isfinite(world[0]) || !isfinite(world[1]) || !isfinite(world[2]))
    return;
  Level levels[4] = {l0, l1, l2, l3};
  // The finest level with a valid plane supplies the residual. Coarser levels
  // only step in where the finer ones are sparse or degenerate.
  Fit selected{};
  for (int level = 0; level < 4; ++level) {
    Level l = levels[level];
    int3 k = grid(world, l.resolution);
    int cell = lookup(l, k);
    Fit f;
    if (cell >= 0) {
      f = l.fits[cell];
    } else {
      const Fit *cached = cached_miss(l, k, version);
      f = cached ? *cached : neighborhood(l, k, base);
    }
    if (f.status != 2)
      continue;
    float delta[3];
    for (int k = 0; k < 3; ++k)
      delta[k] = world[k] - f.mean[k];
    if (fabsf(dot(delta, f.tangent[0])) > f.extent[0] ||
        fabsf(dot(delta, f.tangent[1])) > f.extent[1])
      continue;
    selected = f;
    break;
  }
  if (selected.status != 2)
    return;
  float distance = dot(selected.normal, world) + selected.d;
  float s =
      1 - .9f * fabsf(distance) / sqrtf(sqrtf(fmaxf(dot(local, local), 1e-9f)));
  if (s <= .9f)
    return;
  PointNormal p{};
  p.x = local[0];
  p.y = local[1];
  p.z = local[2];
  p.intensity = selected.d;
  p.normal_x = selected.normal[0];
  p.normal_y = selected.normal[1];
  p.normal_z = selected.normal[2];
  out[i] = p;
  flags[i] = 1;
}
static int blocks(uint32_t n) { return (n + 255) / 256; }
static Level view(int i) {
  auto &s = *state;
  auto &l = s.level[i];
  return {l.cells.p, l.fits.p, l.table.p, l.n, l.table_n,
          s.base * (.5f * float(1 << i)), l.miss.p, l.miss_n};
}
// The level version being built while a scan is merged.
static Level view_next(int i, uint32_t n) {
  auto &s = *state;
  auto &l = s.level[i];
  return {l.cells_next.p, l.fits_next.p, l.table_next.p, n, l.table_n,
          s.base * (.5f * float(1 << i)), l.miss.p, l.miss_n};
}
} // namespace mr

bool multiresolution_active_impl() { return mr::state != nullptr; }
void init_multiresolution_impl(uint32_t capacity, float base,
                               float sensitivity) {
  assert(!mr::state && capacity >= 4 && std::isfinite(base) && base > 0);
  mr::state = new mr::State();
  auto &s = *mr::state;
  s.capacity = capacity;
  s.base = base;
  s.sensitivity = sensitivity;
  s.counters.reserve(1);
  for (int i = 0; i < 4; ++i) {
    auto &l = s.level[i];
    l.budget = capacity / 4 + uint32_t(i < int(capacity % 4));
    l.cells.reserve(l.budget);
    l.cells_next.reserve(l.budget);
    l.fits.reserve(l.budget);
    l.fits_next.reserve(l.budget);
    l.dirty.reserve(l.budget);
    // Power-of-two tables avoid integer division on every neighborhood probe.
    l.table_n = 1;
    while (l.table_n < 2 * l.budget)
      l.table_n *= 2;
    l.table.reserve(l.table_n);
    l.table_next.reserve(l.table_n);
    MR_CHECK(cudaMemset(l.table.p, 255, size_t(l.table_n) * sizeof(int)));
    // State 0 belongs to version 0, which is never current: all slots empty.
    l.miss_n = l.table_n;
    l.miss.reserve(l.miss_n);
    MR_CHECK(cudaMemset(l.miss.p, 0, size_t(l.miss_n) * sizeof(mr::MissSlot)));
  }
}
void add_multiresolution_impl(cudaStream_t stream, Point *points, uint32_t n,
                              Pose pose, uint32_t scan) {
  if (!n)
    return;
  using namespace mr;
  auto &s = *state;

  auto policy = thrust::cuda::par_nosync.on(stream);
  s.samples.reserve(n);
  s.valid_samples.reserve(n);
  s.scratch.reserve(n);
  s.batches.reserve(n);
  s.parents.reserve(n);
  transform_samples<<<blocks(n), 256, 0, stream>>>(points, n, pose,
                                                   s.base * .5f, s.samples.p);
  n = thrust::copy_if(policy, s.samples.p, s.samples.p + n, s.valid_samples.p,
                      ValidSample{}) -
      s.valid_samples.p;
  if (!n)
    return;
  thrust::stable_sort(policy, s.valid_samples.p, s.valid_samples.p + n,
                      SampleLess{});
  scan_cells<<<blocks(n), 256, 0, stream>>>(s.valid_samples.p, n, scan,
                                            s.scratch.p);
  uint32_t batches = thrust::copy_if(policy, s.scratch.p, s.scratch.p + n,
                                     s.batches.p, ValidCell{}) -
                     s.batches.p;
  for (int i = 0; i < 4; ++i) {
    if (i) {
      parent_keys<<<blocks(batches), 256, 0, stream>>>(s.batches.p, batches);
      thrust::stable_sort(policy, s.batches.p, s.batches.p + batches,
                          CellLess{});
      combine<<<blocks(batches), 256, 0, stream>>>(
          s.batches.p, batches, s.parents.p, false, s.sensitivity);
      batches = thrust::copy_if(policy, s.parents.p, s.parents.p + batches,
                                s.batches.p, ValidCell{}) -
                s.batches.p;
    }
    auto &l = s.level[i];
    uint32_t total = l.n + batches;
    s.scratch.reserve(total);
    s.merged.reserve(total);

    // Both inputs are already key-sorted. Stable merge keeps the old map
    // before this scan at equal keys.
    thrust::merge(policy, l.cells.p, l.cells.p + l.n, s.batches.p,
                  s.batches.p + batches, s.scratch.p, CellLess{});
    combine<<<blocks(total), 256, 0, stream>>>(s.scratch.p, total, s.merged.p,
                                               true, s.sensitivity);
    uint32_t count = thrust::copy_if(policy, s.merged.p, s.merged.p + total,
                                     s.scratch.p, ValidCell{}) -
                     s.scratch.p;
    uint32_t excess = count > l.budget ? count - l.budget : 0;
    uint32_t n_next = count - excess;
    if (excess) {
      // Rank compact keys instead of whole cells, then drop the evicted cells
      // with an order-preserving compaction: the merge is already key-sorted,
      // so no second sort is needed.
      s.evict.reserve(count);
      s.evicted.reserve(count);
      s.evicted_cells.reserve(excess);
      evict_keys<<<blocks(count), 256, 0, stream>>>(s.scratch.p, count, scan,
                                                    s.evict.p);
      thrust::sort(policy, s.evict.p, s.evict.p + count, EvictKeyLess{});
      MR_CHECK(cudaMemsetAsync(s.evicted.p, 0, count, stream));
      flag_evicted<<<blocks(excess), 256, 0, stream>>>(
          s.evict.p, excess, s.scratch.p, s.evicted.p, s.evicted_cells.p);
      uint32_t kept = thrust::copy_if(policy, s.scratch.p, s.scratch.p + count,
                                      s.evicted.p, l.cells_next.p,
                                      NotFlagged{}) -
                      l.cells_next.p;
      assert(kept == n_next);
      (void)kept;
    } else {
      MR_CHECK(cudaMemcpyAsync(l.cells_next.p, s.scratch.p,
                               n_next * sizeof(Cell), cudaMemcpyDeviceToDevice,
                               stream));
    }
    Level prev = view(i), next = view_next(i, n_next);
    MR_CHECK(cudaMemsetAsync(next.table, 255, next.table_n * sizeof(int),
                             stream));
    if (n_next)
      index_cells<<<blocks(n_next), 256, 0, stream>>>(next);
    // Only fits next to this scan's updates, insertions and evictions change.
    MR_CHECK(cudaMemsetAsync(l.dirty.p, 0, n_next, stream));
    if (batches && n_next)
      mark_dirty<<<blocks(batches), 256, 0, stream>>>(s.batches.p, batches,
                                                      next, l.dirty.p);
    if (excess && n_next)
      mark_dirty<<<blocks(excess), 256, 0, stream>>>(s.evicted_cells.p, excess, next,
                                                     l.dirty.p);
    if (n_next)
      refresh_fits<<<blocks(n_next), 256, 0, stream>>>(prev, next, l.dirty.p,
                                                       s.base);
    l.cells.swap(l.cells_next);
    l.fits.swap(l.fits_next);
    l.table.swap(l.table_next);
    l.n = n_next;
  }
  MR_CHECK(cudaGetLastError());
  MR_CHECK(cudaStreamSynchronize(stream));
  // The map changed, so cached fits of absent cells are stale.
  ++s.version;
}
uint32_t planes_multiresolution_impl(cudaStream_t stream, Point *cloud,
                                     uint32_t n, Pose ext, Pose pose,
                                     PointNormal *out) {
  if (!n)
    return 0;
  using namespace mr;
  auto &s = *state;
  s.planes.reserve(n);
  s.flags.reserve(n);
  uint32_t work_cap = 4 * n;
  s.work.reserve(work_cap);
  MR_CHECK(cudaMemsetAsync(s.counters.p, 0, sizeof(uint32_t), stream));
  collect_misses<<<blocks(n), 256, 0, stream>>>(
      cloud, n, ext, pose, view(0), view(1), view(2), view(3), s.version,
      s.work.p, work_cap, s.counters.p);
  compute_misses<<<blocks(work_cap), 256, 0, stream>>>(
      s.work.p, s.counters.p, work_cap, view(0), view(1), view(2), view(3),
      s.base, s.version);
  query<<<blocks(n), 256, 0, stream>>>(cloud, n, ext, pose, view(0), view(1),
                                       view(2), view(3), s.base, s.version,
                                       s.planes.p, s.flags.p);
  uint32_t count =
      thrust::copy_if(thrust::cuda::par_nosync.on(stream), s.planes.p,
                      s.planes.p + n, s.flags.p, out, Flag{}) -
      out;
  MR_CHECK(cudaGetLastError());
  return count;
}
uint32_t multiresolution_capacity_impl() {
  return mr::state ? mr::state->capacity : 0;
}
uint32_t multiresolution_size_impl() {
  if (!mr::state)
    return 0;
  uint32_t n = 0;
  for (auto &l : mr::state->level)
    n += l.n;
  return n;
}
void get_multiresolution_impl(Point *points, uint32_t *size) {
  uint32_t total = 0;
  for (int level = 0; level < 4; ++level) {
    auto &l = mr::state->level[level];
    if (!l.n)
      continue;
    std::vector<mr::Cell> cells(l.n);
    MR_CHECK(cudaMemcpy(cells.data(), l.cells.p, l.n * sizeof(mr::Cell),
                        cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < l.n; ++i) {
      Point p{};
      p.x = cells[i].d.mean[0];
      p.y = cells[i].d.mean[1];
      p.z = cells[i].d.mean[2];
      // Resolution level this cell lives at, and the scan it was last
      // touched on, stashed in the otherwise-unused intensity/time slots so
      // callers that only want geometry can ignore them.
      p.i = float(level);
      p.t = float(cells[i].d.scan);
      points[total++] = p;
    }
  }
  *size = total;
}
void cleanup_multiresolution_impl() {
  delete mr::state;
  mr::state = nullptr;
}

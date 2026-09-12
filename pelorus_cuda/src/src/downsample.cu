#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <core/cuda_stream.cuh>
#include <lib.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_e));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

namespace rm = rmagine;

namespace {

constexpr uint32_t THREADS = 256;

enum class DownsampleStage : size_t {
    Upload = 0,
    KeyBuild,
    KeyedSort,
    RunLengthEncode,
    RunStartsScan,
    MedoidReduce,
    TimestampKeyBuild,
    TimestampSort,
    Count
};

void destroy_event(cudaEvent_t &event) {
    if (event == nullptr) {
        return;
    }
    const cudaError_t err = cudaEventDestroy(event);
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
        fprintf(stderr, "CUDA event destroy failed: %s\n", cudaGetErrorString(err));
    }
    event = nullptr;
}

struct StageEventRecorder {
    static constexpr size_t kStageCount = static_cast<size_t>(DownsampleStage::Count);

    std::array<cudaEvent_t, kStageCount> starts{};
    std::array<cudaEvent_t, kStageCount> stops{};
    std::array<bool, kStageCount> recorded{};

    StageEventRecorder() {
        for (size_t i = 0; i < kStageCount; ++i) {
            CUDA_CHECK(cudaEventCreateWithFlags(&starts[i], cudaEventDefault));
            CUDA_CHECK(cudaEventCreateWithFlags(&stops[i], cudaEventDefault));
        }
    }

    ~StageEventRecorder() {
        for (size_t i = 0; i < kStageCount; ++i) {
            destroy_event(starts[i]);
            destroy_event(stops[i]);
        }
    }

    static constexpr size_t index(DownsampleStage stage) {
        return static_cast<size_t>(stage);
    }

    template <typename Fn>
    void measure(DownsampleStage stage, cudaStream_t stream, Fn &&enqueue) {
        const size_t i = index(stage);
        recorded[i] = true;
        CUDA_CHECK(cudaEventRecord(starts[i], stream));
        enqueue();
        CUDA_CHECK(cudaEventRecord(stops[i], stream));
    }

    double elapsed_ms(DownsampleStage stage) const {
        const size_t i = index(stage);
        if (!recorded[i]) {
            return 0.0;
        }

        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, starts[i], stops[i]));
        return static_cast<double>(elapsed);
    }
};

struct DownsampleScratch {
    Point *input = nullptr;
    Point *selected = nullptr;
    Point *selected_alt = nullptr;
    Point *medoids = nullptr;
    Point *medoids_alt = nullptr;
    int32_t *keys = nullptr;
    int32_t *keys_alt = nullptr;
    int32_t *unique_keys = nullptr;
    uint32_t *run_counts = nullptr;
    uint32_t *run_starts = nullptr;
    uint32_t *num_unique = nullptr;
    uint32_t *timestamp_keys = nullptr;
    uint32_t *timestamp_keys_alt = nullptr;
    Point *host_input = nullptr;
    Point *host_output = nullptr;
    void *cub_temp = nullptr;
    size_t cub_temp_bytes = 0;
    size_t capacity = 0;
    size_t host_capacity = 0;
};

DownsampleScratch g_scratch;
std::mutex g_scratch_mutex;

template <typename T>
void free_device(T *&ptr) {
    if (ptr != nullptr) {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
            fprintf(stderr, "CUDA free failed: %s\n", cudaGetErrorString(err));
        }
        ptr = nullptr;
    }
}

template <typename T>
void alloc_device(T *&ptr, size_t n) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T)));
}

template <typename T>
void free_host(T *&ptr) {
    if (ptr != nullptr) {
        cudaError_t err = cudaFreeHost(ptr);
        if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
            fprintf(stderr, "CUDA host free failed: %s\n", cudaGetErrorString(err));
        }
        ptr = nullptr;
    }
}

template <typename T>
void alloc_host(T *&ptr, size_t n) {
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&ptr), n * sizeof(T)));
}

void free_scratch() {
    free_device(g_scratch.input);
    free_device(g_scratch.selected);
    free_device(g_scratch.selected_alt);
    free_device(g_scratch.medoids);
    free_device(g_scratch.medoids_alt);
    free_device(g_scratch.keys);
    free_device(g_scratch.keys_alt);
    free_device(g_scratch.unique_keys);
    free_device(g_scratch.run_counts);
    free_device(g_scratch.run_starts);
    free_device(g_scratch.num_unique);
    free_device(g_scratch.timestamp_keys);
    free_device(g_scratch.timestamp_keys_alt);
    free_host(g_scratch.host_input);
    free_host(g_scratch.host_output);
    free_device(g_scratch.cub_temp);
    g_scratch.cub_temp_bytes = 0;
    g_scratch.capacity = 0;
    g_scratch.host_capacity = 0;
}

void ensure_capacity(size_t n) {
    if (g_scratch.capacity >= n && g_scratch.host_capacity >= n) {
        return;
    }

    const size_t new_capacity = std::max(n, g_scratch.capacity + g_scratch.capacity / 2 + 1);
    free_scratch();

    alloc_device(g_scratch.input, new_capacity);
    alloc_device(g_scratch.selected, new_capacity);
    alloc_device(g_scratch.selected_alt, new_capacity);
    alloc_device(g_scratch.medoids, new_capacity);
    alloc_device(g_scratch.medoids_alt, new_capacity);
    alloc_device(g_scratch.keys, new_capacity);
    alloc_device(g_scratch.keys_alt, new_capacity);
    alloc_device(g_scratch.unique_keys, new_capacity);
    alloc_device(g_scratch.run_counts, new_capacity);
    alloc_device(g_scratch.run_starts, new_capacity);
    alloc_device(g_scratch.timestamp_keys, new_capacity);
    alloc_device(g_scratch.timestamp_keys_alt, new_capacity);
    alloc_device(g_scratch.num_unique, 1);
    alloc_host(g_scratch.host_input, new_capacity);
    alloc_host(g_scratch.host_output, new_capacity);
    g_scratch.capacity = new_capacity;
    g_scratch.host_capacity = new_capacity;
}

void ensure_temp_storage(uint32_t n) {
    size_t temp_bytes = 0;
    size_t max_temp_bytes = 0;

    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, g_scratch.keys,
                                    g_scratch.keys_alt, g_scratch.selected,
                                    g_scratch.selected_alt, n);
    max_temp_bytes = std::max(max_temp_bytes, temp_bytes);

    temp_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, temp_bytes, g_scratch.keys_alt,
                                       g_scratch.unique_keys,
                                       g_scratch.run_counts,
                                       g_scratch.num_unique, n);
    max_temp_bytes = std::max(max_temp_bytes, temp_bytes);

    temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, g_scratch.run_counts,
                                  g_scratch.run_starts, n);
    max_temp_bytes = std::max(max_temp_bytes, temp_bytes);

    temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
                                    g_scratch.timestamp_keys,
                                    g_scratch.timestamp_keys_alt,
                                    g_scratch.medoids, g_scratch.medoids_alt,
                                    n);
    max_temp_bytes = std::max(max_temp_bytes, temp_bytes);

    if (g_scratch.cub_temp_bytes < max_temp_bytes) {
        free_device(g_scratch.cub_temp);
        CUDA_CHECK(cudaMalloc(&g_scratch.cub_temp, max_temp_bytes));
        g_scratch.cub_temp_bytes = max_temp_bytes;
    }
}

__host__ __device__ __forceinline__ int32_t downsample_hash(const Point &p,
                                                            float voxel_size) {
    const float inv = 1.0f / voxel_size;
    const int32_t x = static_cast<int32_t>(floorf(p.x * inv));
    const int32_t y = static_cast<int32_t>(floorf(p.y * inv));
    const int32_t z = static_cast<int32_t>(floorf(p.z * inv));
    uint32_t h = static_cast<uint32_t>(x) * 73856093u;
    h += static_cast<uint32_t>(y) * 471943u;
    h += static_cast<uint32_t>(z) * 83492791u;
    return static_cast<int32_t>(h);
}

__device__ __forceinline__ uint32_t ordered_float_key(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}

enum class PointComponent : uint32_t {
    Timestamp = 0,
    X,
    Y,
    Z,
    Intensity,
};

__device__ __forceinline__ float point_component(const Point &point,
                                                 PointComponent component) {
    switch (component) {
        case PointComponent::Timestamp:
            return point.t;
        case PointComponent::X:
            return point.x;
        case PointComponent::Y:
            return point.y;
        case PointComponent::Z:
            return point.z;
        case PointComponent::Intensity:
            return point.i;
    }
    return 0.0f;
}

__global__ void select_downsample_points_kernel(
    const Point *__restrict__ input, uint32_t cloud_size, uint32_t point_skip,
    Point *__restrict__ selected, uint32_t selected_size) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= selected_size) {
        return;
    }

    const uint32_t src_idx = tid * point_skip;
    if (src_idx >= cloud_size) {
        return;
    }

    selected[tid] = input[src_idx];
}

__global__ void build_voxel_keys_kernel(const Point *__restrict__ points,
                                        uint32_t n, float voxel_size,
                                        int32_t *__restrict__ keys) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    keys[tid] = downsample_hash(points[tid], voxel_size);
}

__global__ void build_component_keys_kernel(
    const Point *__restrict__ points, uint32_t n, PointComponent component,
    bool numeric_order, uint32_t *__restrict__ keys) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }

    float value = point_component(points[tid], component);
    if (numeric_order) {
        // Rust's partial_cmp treats -0.0 and +0.0 as equal.
        if (value == 0.0f) {
            value = 0.0f;
        }
        keys[tid] = ordered_float_key(value);
    } else {
        // Match pelorus_sync::keyed_point_sort_key, which orders secondary fields by
        // their raw f32 bit patterns.
        keys[tid] = __float_as_uint(value);
    }
}

__device__ __forceinline__ bool point_lexicographically_less(
    const Point &lhs, const Point &rhs) {
    if (lhs.t != rhs.t) return lhs.t < rhs.t;
    if (lhs.x != rhs.x) return lhs.x < rhs.x;
    if (lhs.y != rhs.y) return lhs.y < rhs.y;
    if (lhs.z != rhs.z) return lhs.z < rhs.z;
    return lhs.i < rhs.i;
}

// Rust evaluates the CPU medoid distance as separate f32 multiplies and adds.
// NVCC contracts the same expression into FMAs at -O3, which breaks exact
// distance ties (especially in two-point voxels) and selects a different
// timestamp. Explicit round-to-nearest intrinsics preserve the CPU operation
// boundaries and ordering.
__device__ __forceinline__ float squared_distance_cpu_order(
    const Point &point, float cx, float cy, float cz) {
    const float dx = __fsub_rn(point.x, cx);
    const float dy = __fsub_rn(point.y, cy);
    const float dz = __fsub_rn(point.z, cz);
    const float dx2 = __fmul_rn(dx, dx);
    const float dy2 = __fmul_rn(dy, dy);
    const float dz2 = __fmul_rn(dz, dz);
    return __fadd_rn(__fadd_rn(dx2, dy2), dz2);
}

__global__ void medoid_runs_kernel(const Point *__restrict__ sorted_points,
                                   const uint32_t *__restrict__ run_starts,
                                   const uint32_t *__restrict__ run_counts,
                                   const uint32_t *__restrict__ num_unique,
                                   Point *__restrict__ medoids) {
    const uint32_t run = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t runs = *num_unique;
    if (run >= runs) {
        return;
    }

    const uint32_t start = run_starts[run];
    const uint32_t count = run_counts[run];
    if (count == 1u) {
        medoids[run] = sorted_points[start];
        return;
    }

    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_z = 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
        const Point p = sorted_points[start + i];
        sum_x = __fadd_rn(sum_x, p.x);
        sum_y = __fadd_rn(sum_y, p.y);
        sum_z = __fadd_rn(sum_z, p.z);
    }

    const float fn = static_cast<float>(count);
    const float cx = __fdiv_rn(sum_x, fn);
    const float cy = __fdiv_rn(sum_y, fn);
    const float cz = __fdiv_rn(sum_z, fn);

    Point best = sorted_points[start];
    float best_dist = squared_distance_cpu_order(best, cx, cy, cz);

    for (uint32_t i = 1; i < count; ++i) {
        const Point p = sorted_points[start + i];
        const float dist = squared_distance_cpu_order(p, cx, cy, cz);
        if (dist < best_dist ||
            (dist == best_dist && point_lexicographically_less(p, best))) {
            best_dist = dist;
            best = p;
        }
    }

    medoids[run] = best;
}

uint32_t grid_dim(uint32_t n) {
    return (n + THREADS - 1u) / THREADS;
}

enum class DownsampleOutputMode {
    Host,
    Device,
};

uint32_t downsample_points_impl(void *stream, const Point *points_cpu,
                                uint32_t cloud_size, uint32_t point_skip,
                                float voxel_size, Point *out_points,
                                uint32_t out_capacity,
                                CudaDownsampleTimings *timings_out,
                                DownsampleOutputMode output_mode) {
    CudaDownsampleTimings timings{};
    if (timings_out != nullptr) {
        *timings_out = timings;
    }

    assert(stream != nullptr);
    if (points_cpu == nullptr || out_points == nullptr || cloud_size == 0 ||
        out_capacity == 0 || voxel_size <= 0.0f) {
        return 0;
    }

    point_skip = std::max(point_skip, 1u);
    const uint32_t selected_size =
        (cloud_size + point_skip - 1u) / point_skip;
    const uint32_t max_out = std::min(selected_size, out_capacity);
    if (max_out == 0) {
        return 0;
    }

    rm::CudaStream *stream_ = reinterpret_cast<rm::CudaStream *>(stream);
    stream_->context().get()->use();
    cudaStream_t cuda_stream = stream_->handle();

    std::lock_guard<std::mutex> lock(g_scratch_mutex);
    ensure_capacity(std::max(cloud_size, selected_size));
    ensure_temp_storage(selected_size);
    StageEventRecorder stage_timers;

    auto finalize_stage_timings = [&]() {
        timings.upload_ms = stage_timers.elapsed_ms(DownsampleStage::Upload);
        timings.key_build_ms = stage_timers.elapsed_ms(DownsampleStage::KeyBuild);
        timings.keyed_sort_ms = stage_timers.elapsed_ms(DownsampleStage::KeyedSort);
        timings.run_length_encode_ms =
            stage_timers.elapsed_ms(DownsampleStage::RunLengthEncode);
        timings.run_starts_scan_ms =
            stage_timers.elapsed_ms(DownsampleStage::RunStartsScan);
        timings.medoid_reduce_ms = stage_timers.elapsed_ms(DownsampleStage::MedoidReduce);
        timings.timestamp_key_build_ms =
            stage_timers.elapsed_ms(DownsampleStage::TimestampKeyBuild);
        timings.timestamp_sort_ms = stage_timers.elapsed_ms(DownsampleStage::TimestampSort);
        timings.total_ms = timings.upload_ms + timings.key_build_ms +
                           timings.keyed_sort_ms + timings.run_length_encode_ms +
                           timings.run_starts_scan_ms + timings.unique_count_sync_ms +
                           timings.medoid_reduce_ms +
                           timings.timestamp_key_build_ms +
                           timings.timestamp_sort_ms + timings.download_ms;
        if (timings_out != nullptr) {
            *timings_out = timings;
        }
    };

    std::memcpy(g_scratch.host_input, points_cpu, cloud_size * sizeof(Point));
    stage_timers.measure(DownsampleStage::Upload, cuda_stream, [&]() {
        CUDA_CHECK(cudaMemcpyAsync(g_scratch.input, g_scratch.host_input,
                                   cloud_size * sizeof(Point),
                                   cudaMemcpyHostToDevice, cuda_stream));
    });

    stage_timers.measure(DownsampleStage::KeyBuild, cuda_stream, [&]() {
        select_downsample_points_kernel<<<grid_dim(selected_size), THREADS, 0,
                                          cuda_stream>>>(
            g_scratch.input, cloud_size, point_skip, g_scratch.selected,
            selected_size);
    });

    size_t temp_bytes = g_scratch.cub_temp_bytes;
    Point *sorted_points = nullptr;
    stage_timers.measure(DownsampleStage::KeyedSort, cuda_stream, [&]() {
        // CPU preprocessing uses a full deterministic tuple key:
        // (voxel_hash, timestamp_bits, x_bits, y_bits, z_bits,
        // intensity_bits). CUB radix sort is stable, so sorting the secondary
        // fields from least to most significant before the voxel hash exactly
        // reproduces that lexicographic order.
        constexpr PointComponent components[] = {
            PointComponent::Intensity, PointComponent::Z, PointComponent::Y,
            PointComponent::X, PointComponent::Timestamp};
        Point *src = g_scratch.selected;
        Point *dst = g_scratch.selected_alt;
        for (PointComponent component : components) {
            build_component_keys_kernel<<<grid_dim(selected_size), THREADS, 0,
                                          cuda_stream>>>(
                src, selected_size, component, false,
                g_scratch.timestamp_keys);
            temp_bytes = g_scratch.cub_temp_bytes;
            cub::DeviceRadixSort::SortPairs(
                g_scratch.cub_temp, temp_bytes, g_scratch.timestamp_keys,
                g_scratch.timestamp_keys_alt, src, dst, selected_size, 0, 32,
                cuda_stream);
            std::swap(src, dst);
        }

        build_voxel_keys_kernel<<<grid_dim(selected_size), THREADS, 0,
                                  cuda_stream>>>(
            src, selected_size, voxel_size, g_scratch.keys);
        temp_bytes = g_scratch.cub_temp_bytes;
        cub::DeviceRadixSort::SortPairs(
            g_scratch.cub_temp, temp_bytes, g_scratch.keys, g_scratch.keys_alt,
            src, dst, selected_size, 0, 32, cuda_stream);
        sorted_points = dst;
    });

    stage_timers.measure(DownsampleStage::RunLengthEncode, cuda_stream, [&]() {
        CUDA_CHECK(cudaMemsetAsync(g_scratch.run_counts, 0,
                                   selected_size * sizeof(uint32_t),
                                   cuda_stream));
        temp_bytes = g_scratch.cub_temp_bytes;
        cub::DeviceRunLengthEncode::Encode(
            g_scratch.cub_temp, temp_bytes, g_scratch.keys_alt,
            g_scratch.unique_keys, g_scratch.run_counts, g_scratch.num_unique,
            selected_size, cuda_stream);
    });

    stage_timers.measure(DownsampleStage::RunStartsScan, cuda_stream, [&]() {
        temp_bytes = g_scratch.cub_temp_bytes;
        cub::DeviceScan::ExclusiveSum(g_scratch.cub_temp, temp_bytes,
                                      g_scratch.run_counts,
                                      g_scratch.run_starts, selected_size,
                                      cuda_stream);
    });

    uint32_t unique_count = 0;
    const auto unique_sync_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(&unique_count, g_scratch.num_unique,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost,
                               cuda_stream));
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    timings.unique_count_sync_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                  unique_sync_start)
            .count();
    unique_count = std::min(unique_count, max_out);
    if (unique_count == 0) {
        finalize_stage_timings();
        return 0;
    }

    stage_timers.measure(DownsampleStage::MedoidReduce, cuda_stream, [&]() {
        medoid_runs_kernel<<<grid_dim(unique_count), THREADS, 0, cuda_stream>>>(
            sorted_points, g_scratch.run_starts, g_scratch.run_counts,
            g_scratch.num_unique, g_scratch.medoids);
    });

    stage_timers.measure(DownsampleStage::TimestampKeyBuild, cuda_stream, [&]() {
        // The actual key construction is included in the stable multi-pass
        // output sort below.
    });

    Point *ordered_medoids = nullptr;
    stage_timers.measure(DownsampleStage::TimestampSort, cuda_stream, [&]() {
        // Match compare_point_timestamp:
        // (timestamp, x, y, z, intensity), all in numeric f32 order.
        constexpr PointComponent components[] = {
            PointComponent::Intensity, PointComponent::Z, PointComponent::Y,
            PointComponent::X, PointComponent::Timestamp};
        Point *src = g_scratch.medoids;
        Point *dst = g_scratch.medoids_alt;
        for (PointComponent component : components) {
            build_component_keys_kernel<<<grid_dim(unique_count), THREADS, 0,
                                          cuda_stream>>>(
                src, unique_count, component, true,
                g_scratch.timestamp_keys);
            temp_bytes = g_scratch.cub_temp_bytes;
            cub::DeviceRadixSort::SortPairs(
                g_scratch.cub_temp, temp_bytes, g_scratch.timestamp_keys,
                g_scratch.timestamp_keys_alt, src, dst, unique_count, 0, 32,
                cuda_stream);
            std::swap(src, dst);
        }
        ordered_medoids = src;
    });

    if (output_mode == DownsampleOutputMode::Host) {
        const auto download_start = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(g_scratch.host_output, ordered_medoids,
                                   unique_count * sizeof(Point),
                                   cudaMemcpyDeviceToHost, cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
        std::memcpy(out_points, g_scratch.host_output, unique_count * sizeof(Point));
        timings.download_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - download_start)
                                  .count();
    } else {
        if (out_points != ordered_medoids) {
            CUDA_CHECK(cudaMemcpyAsync(out_points, ordered_medoids,
                                       unique_count * sizeof(Point),
                                       cudaMemcpyDeviceToDevice, cuda_stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
        timings.download_ms = 0.0;
    }

    finalize_stage_timings();
    return unique_count;
}

} // namespace

extern "C" uint32_t downsample_points(void *stream, const Point *points_cpu,
                                      uint32_t cloud_size,
                                      uint32_t point_skip,
                                      float voxel_size, Point *out_cpu,
                                      uint32_t out_capacity,
                                      CudaDownsampleTimings *timings_out) {
    try {
        return downsample_points_impl(stream, points_cpu, cloud_size, point_skip,
                                      voxel_size, out_cpu, out_capacity,
                                      timings_out, DownsampleOutputMode::Host);
    } catch (const std::exception &e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        std::exit(1);
    } catch (...) {
        std::cerr << "Caught unknown exception." << std::endl;
        std::exit(1);
    }
}

extern "C" uint32_t downsample_points_device(
    void *stream, const Point *points_cpu, uint32_t cloud_size,
    uint32_t point_skip, float voxel_size, Point *out_gpu,
    uint32_t out_capacity, CudaDownsampleTimings *timings_out) {
    try {
        return downsample_points_impl(stream, points_cpu, cloud_size, point_skip,
                                      voxel_size, out_gpu, out_capacity,
                                      timings_out, DownsampleOutputMode::Device);
    } catch (const std::exception &e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        std::exit(1);
    } catch (...) {
        std::cerr << "Caught unknown exception." << std::endl;
        std::exit(1);
    }
}

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

__global__ void build_downsample_keys_kernel(const Point *__restrict__ input,
                                             uint32_t cloud_size,
                                             uint32_t point_skip,
                                             float voxel_size,
                                             int32_t *__restrict__ keys,
                                             Point *__restrict__ selected,
                                             uint32_t selected_size) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= selected_size) {
        return;
    }

    const uint32_t src_idx = tid * point_skip;
    if (src_idx >= cloud_size) {
        return;
    }

    const Point point = input[src_idx];
    keys[tid] = downsample_hash(point, voxel_size);
    selected[tid] = point;
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
        sum_x += p.x;
        sum_y += p.y;
        sum_z += p.z;
    }

    const float inv_count = 1.0f / static_cast<float>(count);
    const float cx = sum_x * inv_count;
    const float cy = sum_y * inv_count;
    const float cz = sum_z * inv_count;

    Point best = sorted_points[start];
    float best_dist = (best.x - cx) * (best.x - cx)
                    + (best.y - cy) * (best.y - cy)
                    + (best.z - cz) * (best.z - cz);

    for (uint32_t i = 1; i < count; ++i) {
        const Point p = sorted_points[start + i];
        const float dist = (p.x - cx) * (p.x - cx)
                         + (p.y - cy) * (p.y - cy)
                         + (p.z - cz) * (p.z - cz);
        if (dist < best_dist) {
            best_dist = dist;
            best = p;
        }
    }

    medoids[run] = best;
}

__global__ void build_timestamp_keys_kernel(const Point *__restrict__ points,
                                            uint32_t n,
                                            uint32_t *__restrict__ keys) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    keys[tid] = ordered_float_key(points[tid].t);
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
        build_downsample_keys_kernel<<<grid_dim(selected_size), THREADS, 0,
                                       cuda_stream>>>(
            g_scratch.input, cloud_size, point_skip, voxel_size, g_scratch.keys,
            g_scratch.selected, selected_size);
    });

    size_t temp_bytes = g_scratch.cub_temp_bytes;
    stage_timers.measure(DownsampleStage::KeyedSort, cuda_stream, [&]() {
        cub::DeviceRadixSort::SortPairs(
            g_scratch.cub_temp, temp_bytes, g_scratch.keys, g_scratch.keys_alt,
            g_scratch.selected, g_scratch.selected_alt, selected_size, 0, 32,
            cuda_stream);
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
            g_scratch.selected_alt, g_scratch.run_starts, g_scratch.run_counts,
            g_scratch.num_unique, g_scratch.medoids);
    });

    stage_timers.measure(DownsampleStage::TimestampKeyBuild, cuda_stream, [&]() {
        build_timestamp_keys_kernel<<<grid_dim(unique_count), THREADS, 0,
                                      cuda_stream>>>(
            g_scratch.medoids, unique_count, g_scratch.timestamp_keys);
    });

    stage_timers.measure(DownsampleStage::TimestampSort, cuda_stream, [&]() {
        temp_bytes = g_scratch.cub_temp_bytes;
        cub::DeviceRadixSort::SortPairs(
            g_scratch.cub_temp, temp_bytes, g_scratch.timestamp_keys,
            g_scratch.timestamp_keys_alt, g_scratch.medoids,
            g_scratch.medoids_alt, unique_count, 0, 32, cuda_stream);
    });

    if (output_mode == DownsampleOutputMode::Host) {
        const auto download_start = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(g_scratch.host_output, g_scratch.medoids_alt,
                                   unique_count * sizeof(Point),
                                   cudaMemcpyDeviceToHost, cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
        std::memcpy(out_points, g_scratch.host_output, unique_count * sizeof(Point));
        timings.download_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - download_start)
                                  .count();
    } else {
        if (out_points != g_scratch.medoids_alt) {
            CUDA_CHECK(cudaMemcpyAsync(out_points, g_scratch.medoids_alt,
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

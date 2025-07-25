#include <core/common.cuh>

#include <core/cuda_debug.cuh>
#include <iostream>

void check(cudaError_t err, const char *const func, const char *const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    // We don't exit when we encounter CUDA errors in this example.
    std::exit(EXIT_FAILURE);
  }
}

void checkLastCudaError(const char *const file, const int line) {
  cudaError_t err{cudaGetLastError()};
  if (err != cudaSuccess) {
    // std::cerr << "CUDA Runtime Error at: " << file << ":" << line <<
    // std::endl; std::cerr << cudaGetErrorString(err) << std::endl;
    //  We don't exit when we encounter CUDA errors in this example.
    //  crash but so we catch in debugger
    // std::exit(EXIT_FAILURE);
    throw rmagine::CudaException(cudaGetErrorString(err), file, "", line);
  }
}

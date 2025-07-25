#include <core/cuda_debug.cuh>

#include <cuda_runtime.h>
#include <sstream>

void cudaAssert(cudaError_t code, const char *file, const char *func,
                int line) {
  if (code != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA Error! Name: " << cudaGetErrorName(code)
       << ", Message: " << cudaGetErrorString(code) << "\n";
    throw rmagine::CudaException(ss.str(), file, func, line);
  }
}

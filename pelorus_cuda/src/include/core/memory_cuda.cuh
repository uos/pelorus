/**
 * Copyright (c) 2021, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * MemoryCuda.hpp
 *
 *  Created on: Jul 17, 2021
 *      Author: Alexander Mock
 */

#ifndef RMAGINE_MEMORY_CUDA_HPP
#define RMAGINE_MEMORY_CUDA_HPP

#include <core/cuda_definitions.cuh>
#include <core/memory.cuh>

namespace rmagine {

// CUDA HELPER
namespace cuda {

void *memcpyHostToDevice(void *dest, const void *src, size_t count);
void *memcpyHostToDevice(void *dest, const void *src, size_t count,
                         CudaStreamPtr stream);
void *memcpyDeviceToHost(void *dest, const void *src, size_t count);
void *memcpyDeviceToHost(void *dest, const void *src, size_t count,
                         CudaStreamPtr stream);
void *memcpyDeviceToDevice(void *dest, const void *src, size_t count);
void *memcpyDeviceToDevice(void *dest, const void *src, size_t count,
                           CudaStreamPtr stream);
void *memcpyHostToHost(void *dest, const void *src, size_t count);
void *memcpyHostToHost(void *dest, const void *src, size_t count,
                       CudaStreamPtr stream);

void **malloc(void **ptr, size_t count);
void **mallocHost(void **ptr, size_t count);
void **mallocManaged(void **ptr, size_t count);

void *free(void *ptr);
void *freeHost(void *ptr);

} // namespace cuda

struct VRAM_CUDA {

  template <typename DataT> static DataT *alloc(size_t N);

  template <typename DataT>
  static DataT *realloc(DataT *mem, size_t Nold, size_t Nnew);

  template <typename DataT> static void free(DataT *mem, size_t N);
};

struct RAM_CUDA {
  template <typename DataT> static DataT *alloc(size_t N);

  template <typename DataT>
  static DataT *realloc(DataT *mem, size_t Nold, size_t Nnew);

  template <typename DataT> static void free(DataT *mem, size_t N);
};

struct UNIFIED_CUDA {
  template <typename DataT> static DataT *alloc(size_t N);

  template <typename DataT>
  static DataT *realloc(DataT *mem, size_t Nold, size_t Nnew);

  template <typename DataT> static void free(DataT *mem, size_t N);
};

// Copy Functions

//////////////////////////
///   HOST TO DEVICE   ///
//////////////////////////
template <typename DataT>
void copy(const MemoryView<DataT, RAM> &from,
          MemoryView<DataT, VRAM_CUDA> &to) {
  cuda::memcpyHostToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template <typename DataT>
void copy(const MemoryView<DataT, RAM> &from, MemoryView<DataT, VRAM_CUDA> &to,
          CudaStreamPtr stream) {
  cuda::memcpyHostToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size(),
                           stream);
}

template <typename DataT>
void copy(const MemoryView<DataT, RAM_CUDA> &from,
          MemoryView<DataT, VRAM_CUDA> &to) {
  cuda::memcpyHostToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template <typename DataT>
void copy(const MemoryView<DataT, RAM_CUDA> &from,
          MemoryView<DataT, VRAM_CUDA> &to, CudaStreamPtr stream) {
  cuda::memcpyHostToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size(),
                           stream);
}

template <typename DataT>
void copy(const MemoryView<DataT, RAM> &from,
          MemoryView<DataT, UNIFIED_CUDA> &to) {
  cuda::memcpyHostToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template <typename DataT>
void copy(const MemoryView<DataT, RAM> &from,
          MemoryView<DataT, UNIFIED_CUDA> &to, CudaStreamPtr stream) {
  cuda::memcpyHostToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size(),
                           stream);
}

template <typename DataT>
void copy(const MemoryView<DataT, RAM_CUDA> &from,
          MemoryView<DataT, UNIFIED_CUDA> &to) {
  cuda::memcpyHostToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template <typename DataT>
void copy(const MemoryView<DataT, RAM_CUDA> &from,
          MemoryView<DataT, UNIFIED_CUDA> &to, CudaStreamPtr stream) {
  cuda::memcpyHostToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size(),
                           stream);
}

//////////////////////////
///   DEVICE TO HOST   ///
//////////////////////////
template <typename DataT>
void copy(const MemoryView<DataT, VRAM_CUDA> &from,
          MemoryView<DataT, RAM> &to) {
  cuda::memcpyDeviceToHost(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template <typename DataT>
void copy(const MemoryView<DataT, VRAM_CUDA> &from, MemoryView<DataT, RAM> &to,
          CudaStreamPtr stream) {
  cuda::memcpyDeviceToHost(to.raw(), from.raw(), sizeof(DataT) * from.size(),
                           stream);
}

template <typename DataT>
void copy(const MemoryView<DataT, VRAM_CUDA> &from,
          MemoryView<DataT, RAM_CUDA> &to) {
  cuda::memcpyDeviceToHost(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template <typename DataT>
void copy(const MemoryView<DataT, VRAM_CUDA> &from,
          MemoryView<DataT, RAM_CUDA> &to, CudaStreamPtr stream) {
  cuda::memcpyDeviceToHost(to.raw(), from.raw(), sizeof(DataT) * from.size(),
                           stream);
}

template <typename DataT>
void copy(const MemoryView<DataT, UNIFIED_CUDA> &from,
          MemoryView<DataT, RAM> &to) {
  cuda::memcpyDeviceToHost(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template <typename DataT>
void copy(const MemoryView<DataT, UNIFIED_CUDA> &from,
          MemoryView<DataT, RAM> &to, CudaStreamPtr stream) {
  cuda::memcpyDeviceToHost(to.raw(), from.raw(), sizeof(DataT) * from.size(),
                           stream);
}

template <typename DataT>
void copy(const MemoryView<DataT, UNIFIED_CUDA> &from,
          MemoryView<DataT, RAM_CUDA> &to) {
  cuda::memcpyDeviceToHost(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template <typename DataT>
void copy(const MemoryView<DataT, UNIFIED_CUDA> &from,
          MemoryView<DataT, RAM_CUDA> &to, CudaStreamPtr stream) {
  cuda::memcpyDeviceToHost(to.raw(), from.raw(), sizeof(DataT) * from.size(),
                           stream);
}

////////////////////////////
///   DEVICE TO DEVICE   ///
////////////////////////////
template <typename DataT>
void copy(const MemoryView<DataT, VRAM_CUDA> &from,
          MemoryView<DataT, VRAM_CUDA> &to) {
  cuda::memcpyDeviceToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template <typename DataT>
void copy(const MemoryView<DataT, VRAM_CUDA> &from,
          MemoryView<DataT, VRAM_CUDA> &to, CudaStreamPtr stream) {
  cuda::memcpyDeviceToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size(),
                             stream);
}

template <typename DataT>
void copy(const MemoryView<DataT, UNIFIED_CUDA> &from,
          MemoryView<DataT, UNIFIED_CUDA> &to) {
  cuda::memcpyDeviceToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template <typename DataT>
void copy(const MemoryView<DataT, UNIFIED_CUDA> &from,
          MemoryView<DataT, UNIFIED_CUDA> &to, CudaStreamPtr stream) {
  cuda::memcpyDeviceToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size(),
                             stream);
}

template <typename DataT>
void copy(const MemoryView<DataT, UNIFIED_CUDA> &from,
          MemoryView<DataT, VRAM_CUDA> &to) {
  cuda::memcpyDeviceToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template <typename DataT>
void copy(const MemoryView<DataT, UNIFIED_CUDA> &from,
          MemoryView<DataT, VRAM_CUDA> &to, CudaStreamPtr stream) {
  cuda::memcpyDeviceToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size(),
                             stream);
}

template <typename DataT>
void copy(const MemoryView<DataT, VRAM_CUDA> &from,
          MemoryView<DataT, UNIFIED_CUDA> &to) {
  cuda::memcpyDeviceToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template <typename DataT>
void copy(const MemoryView<DataT, VRAM_CUDA> &from,
          MemoryView<DataT, UNIFIED_CUDA> &to, CudaStreamPtr stream) {
  cuda::memcpyDeviceToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size(),
                             stream);
}

////////////////////////
///   HOST TO HOST   ///
////////////////////////
template <typename DataT>
void copy(const MemoryView<DataT, RAM_CUDA> &from,
          MemoryView<DataT, RAM_CUDA> &to) {
  cuda::memcpyHostToHost(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template <typename DataT>
void copy(const MemoryView<DataT, RAM_CUDA> &from,
          MemoryView<DataT, RAM_CUDA> &to, CudaStreamPtr stream) {
  cuda::memcpyHostToHost(to.raw(), from.raw(), sizeof(DataT) * from.size(),
                         stream);
}

template <typename DataT>
void copy(const MemoryView<DataT, RAM_CUDA> &from, MemoryView<DataT, RAM> &to) {
  cuda::memcpyHostToHost(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template <typename DataT>
void copy(const MemoryView<DataT, RAM_CUDA> &from, MemoryView<DataT, RAM> &to,
          CudaStreamPtr stream) {
  cuda::memcpyHostToHost(to.raw(), from.raw(), sizeof(DataT) * from.size(),
                         stream);
}

template <typename DataT>
void copy(const MemoryView<DataT, RAM> &from, MemoryView<DataT, RAM_CUDA> &to) {
  cuda::memcpyHostToHost(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template <typename DataT>
void copy(const MemoryView<DataT, RAM> &from, MemoryView<DataT, RAM_CUDA> &to,
          CudaStreamPtr stream) {
  cuda::memcpyHostToHost(to.raw(), from.raw(), sizeof(DataT) * from.size(),
                         stream);
}

} // namespace rmagine

// -- IMPLEMENTATION --

namespace rmagine {

/// VRAM_CUDA
template <typename DataT> DataT *VRAM_CUDA::alloc(size_t N) {
  DataT *ret;
  cuda::malloc((void **)&ret, N * sizeof(DataT));
  return ret;
}

template <typename DataT>
DataT *VRAM_CUDA::realloc(DataT *mem, size_t Nold, size_t Nnew) {
  DataT *ret;
  cuda::malloc((void **)&ret, sizeof(DataT) * Nnew);
  cuda::free(mem);
  return ret;
}

template <typename DataT> void VRAM_CUDA::free(DataT *mem, size_t N) {
  cuda::free(mem);
}

/// RAM_CUDA
template <typename DataT> DataT *RAM_CUDA::alloc(size_t N) {
  DataT *ret;
  cuda::mallocHost((void **)&ret, N * sizeof(DataT));
  return ret;
}

template <typename DataT>
DataT *RAM_CUDA::realloc(DataT *mem, size_t Nold, size_t Nnew) {
  DataT *ret;
  cuda::mallocHost((void **)&ret, Nnew * sizeof(DataT));
  cuda::freeHost(mem);
  return ret;
}

template <typename DataT> void RAM_CUDA::free(DataT *mem, size_t N) {
  cuda::freeHost(mem);
}

/// VRAM_CUDA
template <typename DataT> DataT *UNIFIED_CUDA::alloc(size_t N) {
  DataT *ret;
  cuda::mallocManaged((void **)&ret, sizeof(DataT) * N);
  return ret;
}

template <typename DataT>
DataT *UNIFIED_CUDA::realloc(DataT *mem, size_t Nold, size_t Nnew) {
  DataT *ret;
  cuda::mallocManaged((void **)&ret, sizeof(DataT) * Nnew);
  cuda::free(mem);
  return ret;
}

template <typename DataT> void UNIFIED_CUDA::free(DataT *mem, size_t N) {
  cuda::free(mem);
}

} // namespace rmagine

#endif

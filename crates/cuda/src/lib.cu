#include "lib.h"
#include <cuda_runtime.h>
#include <stdio.h>

#include <core/cuda_stream.h>

// Implementation of the function declared in lib.h
extern "C" int allocate_memory_on_gpu(int size) {
    int* d_array = NULL; // Pointer to device memory

    // Calculate the total size in bytes
    size_t byte_size = size * sizeof(int);

    // Allocate memory on the GPU device
    // cudaMalloc takes a pointer to a pointer and the size in bytes.
    cudaError_t malloc_status = cudaMalloc((void**)&d_array, byte_size);

    // Check for errors during memory allocation
    if (malloc_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(malloc_status));
        return -1; // Return an error code
    }

    printf("Successfully allocated %zu bytes on the GPU.\n", byte_size);

    // In a real application, you would now launch a kernel to use this memory.
    // For this example, we'll just free it immediately.

    // Free the allocated memory on the device
    cudaError_t free_status = cudaFree(d_array);

    // Check for errors during memory deallocation
    if (free_status != cudaSuccess) {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(free_status));
        return -2; // Return a different error code
    }
    
    printf("Successfully freed memory on the GPU.\n");

    return 0; // Success
}

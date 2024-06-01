#ifndef COMMON_CUDA_H
#define COMMON_CUDA_H

#include "matrix.h"

#include <cuda_runtime.h>
#include <curand.h>
#include <assert.h>

// NVIDIA A30 L2 cache size in bytes
#define L2_CACHE_SIZE 25165824

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
cudaError_t checkCuda(cudaError_t result);

// Functions to fill device matrix with random float32 values
void gpu_fill_rand(matrix mat, const int SIZE);
__global__ void generate_in_a_b(matrix mat, const float A, const float B, const int SIZE, const int BLK_SIZE, const int TOT_SIZE);

#endif



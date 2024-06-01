#ifndef TRANSPOSE_GPU_H
#define TRANSPOSE_GPU_H

#include "matrix_gpu.h"
#include "common_cuda.h"

// Transpose on GPU using shared memory
__global__ void transpose_blocks_gpu_coalesced(matrix mat, const int SIZE, const int BLK_SIZE, const int TOT_SIZE);
// Transpose on GPU using global memory only
__global__ void transpose_blocks_gpu(matrix mat, const int SIZE, const int BLK_SIZE, const int TOT_SIZE);
// Kernel which performs some random oparation to call before other kernels
__global__ void warm_up_gpu();

#endif
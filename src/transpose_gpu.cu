extern "C" { 
    #include "headers/matrix.h"
}
#include "headers/transpose_gpu.h"
#include "headers/common_cuda.h"

__global__ void transpose_blocks_gpu_coalesced(matrix mat, const int SIZE, const int BLK_SIZE, const int TOT_SIZE){
    
    // define a shared memory buffer which contains 2*BLK_SIZE*BLK_SIZE elements
    extern __shared__ matrix_element buffer[];
    // respectively : 
    //      block_A : stores SUPERIOR block (w.r.t. main diagonal)
    //      block_B : stores INFERIOR block (w.r.t. main diagonal)
    matrix_element *block_A = &buffer[0];
    matrix_element *block_B = &buffer[BLK_SIZE*BLK_SIZE];

    // block A - base thread offsets w.r.t. input matrix
    const unsigned int xA = blockIdx.x * BLK_SIZE + threadIdx.x;
    const unsigned int yA = blockIdx.y * BLK_SIZE + threadIdx.y;
    
    // variable declarations
    unsigned int i, j, A_idx, B_idx;

    // handle the general case, otherwise specificly handle thread blocks in the diagonal
    if (blockIdx.x > blockIdx.y){
        // block B - base thread offsets w.r.t. input matrix
        const int xB = blockIdx.y * BLK_SIZE + threadIdx.x;
        const int yB = blockIdx.x * BLK_SIZE + threadIdx.y;

        // copy from global memory to shared in both blocks
        for (j = 0; j < BLK_SIZE; j += blockDim.y){
            for (i = 0; i < BLK_SIZE; i += blockDim.x){
                A_idx = (yA+j)*SIZE + xA+i;
                B_idx = (yB+j)*SIZE + xB+i;

                // ensure safe in-bounds access to input matrix
                if (B_idx < TOT_SIZE && A_idx < TOT_SIZE){
                    block_A[(threadIdx.y+j)*BLK_SIZE + threadIdx.x+i] = mat[A_idx];
                    block_B[(threadIdx.y+j)*BLK_SIZE + threadIdx.x+i] = mat[B_idx];
                }
            }
        }

        // wait for all threads in the block to finish
        __syncthreads();

        // Write the 2 transposed blocks back on global memory
        // notice that block_A and block_B are swapped
        for (j = 0; j < BLK_SIZE; j += blockDim.y){
            for (i = 0; i < BLK_SIZE; i += blockDim.x){
                A_idx = (yA+j)*SIZE + xA+i;
                B_idx = (yB+j)*SIZE + xB+i;

                if (B_idx < TOT_SIZE && A_idx < TOT_SIZE){
                    mat[A_idx] = block_B[(threadIdx.x+i)*BLK_SIZE + threadIdx.y+j];
                    mat[B_idx] = block_A[(threadIdx.x+i)*BLK_SIZE + threadIdx.y+j];
                }
            }
        }
    }
    else if (blockIdx.x == blockIdx.y){
        for (j = 0; j < BLK_SIZE; j += blockDim.y){
            for (i = 0; i < BLK_SIZE; i += blockDim.x){
                A_idx = (yA+j)*SIZE + xA+i;

                if (A_idx < TOT_SIZE){
                    block_A[(threadIdx.y+j)*BLK_SIZE + threadIdx.x+i] = mat[A_idx];
                }
            }
        }

        __syncthreads();

        for (j = 0; j < BLK_SIZE; j += blockDim.y){
            for (i = 0; i < BLK_SIZE; i += blockDim.x){
                A_idx = (yA+j)*SIZE + xA+i;

                if (A_idx < TOT_SIZE){
                    mat[A_idx] = block_A[(threadIdx.x+i)*BLK_SIZE + threadIdx.y+j];
                }
            }
        }
    }
}


// perform matrix transposition witohut using shared memory
__global__ void transpose_blocks_gpu(matrix mat, const int SIZE, const int BLK_SIZE, const int TOT_SIZE){
    
    const unsigned int x = blockIdx.x * BLK_SIZE + threadIdx.x;
    const unsigned int y = blockIdx.y * BLK_SIZE + threadIdx.y;
    
    matrix_element tmp;
    unsigned int i, j, A_idx, B_idx;

    if (blockIdx.x > blockIdx.y){
        for (j = 0; j < BLK_SIZE; j += blockDim.y){
            for (i = 0; i < BLK_SIZE; i += blockDim.x){
                A_idx = (y+j)*SIZE + x+i;
                B_idx = (x+i)*SIZE + y+j;
                
                if (B_idx < TOT_SIZE && A_idx < TOT_SIZE){
                    tmp = mat[A_idx];
                    mat[A_idx] = mat[B_idx];
                    mat[B_idx] = tmp;
                }
            }
        }
    }
    else if (blockIdx.x == blockIdx.y){
        for (j = 0; j < BLK_SIZE; j += blockDim.y){
            for (i = 0; i < BLK_SIZE; i += blockDim.x){
                A_idx = (y+j)*SIZE + x+i;
                B_idx = (x+i)*SIZE + y+j;

                if (A_idx < B_idx && B_idx < TOT_SIZE && A_idx < TOT_SIZE){
                    tmp = mat[A_idx];
                    mat[A_idx] = mat[B_idx];
                    mat[B_idx] = tmp;
                }
            }
        }
    }
}


__global__ void warm_up_gpu(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid; 
}
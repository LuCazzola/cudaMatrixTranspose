#include "headers/transpose.h"
#include "headers/common.h"

// Naive matrix transposition
void transpose_naive(matrix mat, const int SIZE){
    matrix_element tmp;
    
    for (unsigned int i = 0; i < SIZE; i++){
        for (unsigned int j = i + 1; j < SIZE; j++){
            tmp = mat[i*SIZE + j];
            mat[i*SIZE + j] = mat[j*SIZE + i];
            mat[j*SIZE + i] = tmp;
        }
    }
}

// Cache friendly matrix transposition
void transpose_blocks(matrix mat, const int SIZE, const int BLK_SIZE){
    matrix_element tmp;
    
    int row_index, column_index, blocks_per_row = 1;
    if (BLK_SIZE < SIZE)
        blocks_per_row = (int)(SIZE / BLK_SIZE);

    // Parse separatelly blocks on the main diagonal
    for (unsigned int diagonal = 0; diagonal < blocks_per_row; diagonal++){
        
        // Parse specific block
        for (unsigned int i = 0; i < BLK_SIZE; i++){
            row_index = i + (diagonal * BLK_SIZE);        // get row value inside the block

            for (unsigned int j = i + 1; j < BLK_SIZE; j++){
                column_index = j + (diagonal * BLK_SIZE); // get column value inside the block

                if(row_index < SIZE && column_index < SIZE){
                    tmp = mat[row_index*SIZE + column_index];
                    mat[row_index*SIZE + column_index] = mat[column_index*SIZE + row_index];
                    mat[column_index*SIZE + row_index] = tmp;
                }
            }
        }
    }

   // Parse the rest of the matrix
   // access blocks in row-column order
    for (unsigned int block_row_id = 0; block_row_id < blocks_per_row; block_row_id++) {
        for (unsigned int block_column_id = block_row_id + 1; block_column_id < blocks_per_row; block_column_id++) {
            
            // Process a specific block
            for (unsigned int i = 0; i < BLK_SIZE; i++) {
                row_index = i + (block_row_id * BLK_SIZE);

                for (unsigned int j = 0; j < BLK_SIZE; j++) {
                    column_index = j + (block_column_id * BLK_SIZE);

                    if(row_index < SIZE && column_index < SIZE){
                        tmp = mat[row_index*SIZE + column_index];
                        mat[row_index*SIZE + column_index] = mat[column_index*SIZE + row_index];
                        mat[column_index*SIZE + row_index] = tmp;

                    }
                }
            }
        }
    }
}
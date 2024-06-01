#include "headers/transpose.h"
#include "headers/common.h"
#include "headers/opt_parser.h"

// flag to enable printing of before & after transpose is applied
#define PRINT_MATRICES false
// flag to enable printing of the error before & after the transpose is applied
#define PRINT_TRANSPOSE_ERROR false

/*
    PROBLEM STRUCTURE

    Compute the transpose of a matrix provided that :
    - matrix is [2^N, 2^N] (N provided as input)
    - matrix is NON symmetrical 
    - matrix is randomly populated with float / int values
    - ...
*/

void print_metrics (double exec_time, const int SIZE){
    // metrics evaluation
    printf("\n========================== METRICS ==========================\n");

    // each element in the matrix (except the diagonal) is
    // subject to one read and one write operation
    // total reads + writes = 2 * size^2 (expressed in bytes)
    double Br_Bw = sizeof(matrix_element) * (SIZE*SIZE - SIZE) * 2;

    // effective bandwidth (expressed in GB/s)
    double effective_bandwidth = ( Br_Bw / pow(10,9) ) / exec_time;

    // print out values
    printf("\nExecution time :       %f s\n", exec_time);
    printf("\nEffective Bandwidth :  %f GB/s\n\n", effective_bandwidth);
}

int main(int argc, char* argv[]){
    
    // options to set in "run_transpose.sh" file
    char method [8];
    int N;
    int block_size;
    
    // process CL parameters
    process_transpose_options(argc, argv, method, &N, &block_size);

    const int ROWS = (int) pow(2, N), COLS = ROWS, SIZE = ROWS;

    // matrices initialization
    matrix mat = (matrix) malloc(ROWS * COLS * sizeof(matrix_element));
    fill_matrix(mat, ROWS, COLS);
    
    if (PRINT_MATRICES){
        print_matrix(mat, ROWS, COLS);
    }

    matrix mat_original;
    if (PRINT_TRANSPOSE_ERROR){
        mat_original = matrix_copy(mat, ROWS, COLS);
    }

    // Matrix transposition operation
    // choose between "naive" and "block" version by changing "run_transpose.sh"
    TIMER_DEF;

    printf("\nMatrix size :             2^%d x 2^%d\n", N, N);
    printf("Matrix elemets datatype : %s\n", VALUE(MATRIX_ELEM_DTYPE));
    printf("Optimization flag :       %s\n", VALUE(ADDITIONAL_OPTIM_FALG));
   
    if (strcmp(method,"naive") == 0){
        printf("\nRunning transpose_naive()...\n");
       
        TIMER_START;
        transpose_naive(mat, ROWS);
        TIMER_STOP;

    } else if (strcmp(method,"blocks") == 0){
        const int BLK_SIZE = pow(2, block_size);
        printf("\nRunning transpose_blocks()...\n");
        printf("Block size  : 2^%d x 2^%d\n", block_size, block_size);

        TIMER_START;
        transpose_blocks(mat, ROWS, BLK_SIZE);
        TIMER_STOP;
    }

    // transposition result and error
    if (PRINT_MATRICES){
        print_matrix(mat, ROWS, COLS);
    }
    if (PRINT_TRANSPOSE_ERROR){
        print_transpose_error(mat_original, mat, ROWS);
        free(mat_original);
    }
    print_metrics(TIMER_ELAPSED, SIZE);

    // deallocate memory
    free(mat);

    return(0);
}

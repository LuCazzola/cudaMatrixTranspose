#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include "matrix.h"

// Naive matrix transposition method
void transpose_naive(matrix mat, const int SIZE);
// Cache friendly block-based matrix transposition
void transpose_blocks(matrix mat, const int SIZE, const int BLK_SIZE);

#endif
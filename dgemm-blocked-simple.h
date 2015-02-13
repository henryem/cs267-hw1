#ifndef _dgemm_blocked_simple_h
#define _dgemm_blocked_simple_h

void square_dgemm_with_block_size(int matrix_size, double* A, double* B, double* C, int block_size);

void square_dgemm (int matrix_size, double* A, double* B, double* C);

#endif
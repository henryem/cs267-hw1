#ifndef _dgemm_blocked_h
#define _dgemm_blocked_h

/* Describes the strategy for a recursive block DGEMM algorithm. */
typedef struct dgemm_square_block_strategy {
  // The number of levels of recursion.  0 means that the entire DGEMM is
  // done on a single block.
  int num_levels;
  // On recursion level i, the matrix gets decomposed into block_counts[i]
  // blocks per dimension, or (block_counts[i])^2 total blocks.  This is not
  // the same as the **size** of the resulting blocks, which is (up to
  // rounding) the size of the block before recursion level i, divided by
  // block_counts[i] (per dimension).
  int *block_counts;
} dgemm_square_block_strategy;

dgemm_square_block_strategy* dgemm_square_block_strategy_new(int num_levels, const int *block_counts);

void dgemm_square_block_strategy_free(dgemm_square_block_strategy* s);

/* The size of the smallest blocks accessed by @strategy when it is used on a
 * matrix of size @matrix_size.  It is assumed that the product of the block
 * counts divides @matrix_size.
 */
int smallest_block_size(dgemm_square_block_strategy* strategy, int matrix_size);

void square_dgemm_with_strategy(int matrix_size, double* A, double* B, double* C, dgemm_square_block_strategy* dgemm_strategy);

void square_dgemm (int matrix_size, double* A, double* B, double* C);

#endif
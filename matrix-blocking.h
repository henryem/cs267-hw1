#ifndef _matrix_blocking_h
#define _matrix_blocking_h

#include <stdlib.h>

#include "matrix-storage.h"

/* Describes a square block located somewhere inside a larger square matrix.
 * For example, the trivial block has matrix_size = block_size and
 * block_row_idx = block_col_idx = 0.
 *  
 * For efficiency reasons, this contains only metadata about the block.  The
 * data itself are stored elsewhere, and this struct only provides methods to
 * calculate indices in the data matrix.
 */
typedef struct square_mat_square_block {
  // The edge size of each block.
  int block_size;
  // The indices of the block.  For example, if matrix_size is 10, block_size is 3,
  // block_row_idx is 1, and block_col_idx is 2, then the top-left corner of
  // this block is element M[3,6] of the original matrix M, and the bottom-left
  // corner is element M[5,8].
  int block_row_idx, block_col_idx;
  square_matrix_storage_format *format;
} square_mat_square_block;

square_mat_square_block* square_mat_square_block_new(int block_size, int block_row_idx, int block_col_idx, square_matrix_storage_format* format);

square_mat_square_block* trivial_block(square_matrix_storage_format* format);

/* Get a subblock of some existing block.  It is assumed that we are dividing
 * b into subblock_count^2 blocks, and we want subblock (subblock_row_idx, subblock_col_idx)
 * in that collection of subblocks.
 * @param subblock_count must be a divisor of b->block_size.
 */
square_mat_square_block* get_subblock(square_mat_square_block* b, int subblock_count, int subblock_row_idx, int subblock_col_idx);

int get_top_left_index(square_mat_square_block* b);

/* Get the index in the full matrix of element
 * (row_in_block_idx, col_in_block_idx) of block b.
 */
int get_index_in_matrix(square_mat_square_block* b, int row_in_block_idx, int col_in_block_idx);

#endif
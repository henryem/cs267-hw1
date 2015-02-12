#include <assert.h>

#include "matrix-blocking.h"
#include "matrix-storage.h"

square_mat_square_block* square_mat_square_block_new(int block_size, int block_row_idx, int block_col_idx, square_matrix_storage_format* format) {
  assert(format->strategy == BLOCK);
  assert(block_size > 0);
  assert(block_row_idx >= 0);
  assert(block_row_idx < block_count(format));
  assert(block_col_idx >= 0);
  assert(block_col_idx < block_count(format));
  square_mat_square_block* b = malloc(sizeof(square_mat_square_block));
  b->block_size = block_size;
  b->block_row_idx = block_row_idx;
  b->block_col_idx = block_col_idx;
  b->format = format;
  return b;
}

square_mat_square_block* trivial_block(square_matrix_storage_format* format) {
  return square_mat_square_block_new(format->matrix_size, 0, 0, format);
}

/* Get a subblock of some existing block.  It is assumed that we are dividing
 * b into subblock_count^2 blocks, and we want subblock (subblock_row_idx, subblock_col_idx)
 * in that collection of subblocks.
 * @param subblock_count must be a divisor of b->block_size. */
square_mat_square_block* get_subblock(square_mat_square_block* b, int subblock_count, int subblock_row_idx, int subblock_col_idx) {
  int subblock_size = b->block_size / subblock_count;
  int current_block_topleft_row_idx = b->block_row_idx * subblock_count;
  int current_block_topleft_col_idx = b->block_col_idx * subblock_count;
  int overall_row_idx = current_block_topleft_row_idx + subblock_row_idx;
  int overall_col_idx = current_block_topleft_col_idx + subblock_col_idx;
  square_mat_square_block* subblock = square_mat_square_block_new(subblock_size, overall_row_idx, overall_col_idx, b->format);
  return subblock;
}

int get_top_left_index(square_mat_square_block* b) {
  int overall_row_idx = b->block_row_idx * b->block_size;
  int overall_col_idx = b->block_col_idx * b->block_size;
  return get_index(b->format, overall_row_idx, overall_col_idx);
}

int get_index_in_matrix(square_mat_square_block* b, int row_in_block_idx, int col_in_block_idx) {
  int overall_row_idx = b->block_row_idx * b->block_size + row_in_block_idx;
  int overall_col_idx = b->block_col_idx * b->block_size + col_in_block_idx;
  return get_index(b->format, overall_row_idx, overall_col_idx);
}
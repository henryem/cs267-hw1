#include <assert.h>

#include "matrix-blocking.h"
#include "matrix-storage.h"

square_mat_square_block* square_mat_square_block_new(int block_size, int block_row_idx, int block_col_idx, square_matrix_storage_format* format) {
  assert(format->strategy == BLOCK_CM);
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

inline void set_subblock(square_mat_square_block* next_b, square_mat_square_block* b, int subblock_count, int subblock_row_idx, int subblock_col_idx) {
  int current_block_topleft_row_idx = b->block_row_idx * subblock_count;
  int current_block_topleft_col_idx = b->block_col_idx * subblock_count;
  int overall_row_idx = current_block_topleft_row_idx + subblock_row_idx;
  int overall_col_idx = current_block_topleft_col_idx + subblock_col_idx;
  next_b->block_row_idx = overall_row_idx;
  next_b->block_col_idx = overall_col_idx;
}

square_mat_square_block* get_subblock(square_mat_square_block* b, int subblock_count, int subblock_row_idx, int subblock_col_idx) { 
  int subblock_size = b->block_size / subblock_count;
  square_mat_square_block* subblock = malloc(sizeof(square_mat_square_block));
  subblock->block_size = subblock_size;
  subblock->format = b->format;
  set_subblock(subblock, b, subblock_count, subblock_row_idx, subblock_col_idx);
  return subblock;
}

inline int get_top_left_index(square_mat_square_block* b) {
  int overall_row_idx = b->block_row_idx * b->block_size;
  int overall_col_idx = b->block_col_idx * b->block_size;
  return get_index(b->format, overall_row_idx, overall_col_idx);
}

inline int get_index_in_matrix(square_mat_square_block* b, int row_in_block_idx, int col_in_block_idx) {
  int overall_row_idx = b->block_row_idx * b->block_size + row_in_block_idx;
  int overall_col_idx = b->block_col_idx * b->block_size + col_in_block_idx;
  return get_index(b->format, overall_row_idx, overall_col_idx);
}
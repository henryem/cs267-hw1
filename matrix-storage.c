#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#include "unit-test-framework.h"
#include "matrix-storage.h"

#define div_round_up(a, b) ((a+b-1) / b)
#define xor(a, b) ((a && !b) || (!a && b))

square_matrix_storage_format* square_matrix_storage_format_new(int matrix_size, matrix_storage_strategy strategy, int block_size) {
  assert(matrix_size > 0);
  assert(xor(strategy == BLOCK_CM || strategy == BLOCK_RM, block_size == 0));
  
  square_matrix_storage_format* f = malloc(sizeof(square_matrix_storage_format));
  f->matrix_size = matrix_size;
  f->strategy = strategy;
  f->block_size = block_size;
  return f;
}

inline int num_elements(square_matrix_storage_format* format) {
  return format->matrix_size * format->matrix_size;
}

inline int block_count(square_matrix_storage_format* format) {
  matrix_storage_strategy s = format->strategy;
  if (s == COLUMN_MAJOR) {
    return 0;
  }
  return div_round_up(format->matrix_size, format->block_size);
}

inline bool is_aligned(square_matrix_storage_format* format) {
  return aligned_matrix_size(format) == format->matrix_size;
}

inline int aligned_matrix_size(square_matrix_storage_format* format) {
  return block_count(format) * format->block_size;
}

inline int edge_block_size(square_matrix_storage_format* format) {
  matrix_storage_strategy s = format->strategy;
  if (s == COLUMN_MAJOR) {
    return 0;
  }
  int aligned_size = aligned_matrix_size(format);
  if (aligned_size == format->matrix_size) {
    return format->block_size;
  } else {
    // The edge blocks are missing (aligned_size - matrix_size) elements.
    return format->block_size - (aligned_size - format->matrix_size);
  }
}

double* make_arbitrary_matrix(square_matrix_storage_format* format) {
  switch (format->strategy) {
    case COLUMN_MAJOR: return malloc(format->matrix_size*format->matrix_size * sizeof(double));
    case BLOCK_RM: return malloc(format->matrix_size*format->matrix_size * sizeof(double));
    case BLOCK_CM: return malloc(format->matrix_size*format->matrix_size * sizeof(double));
  }
}

double* copy(double* original, square_matrix_storage_format* format) {
  //FIXME: Very unoptimized.
  return to_format(original, format, format);
}

double* to_format(double* original, square_matrix_storage_format* original_format, square_matrix_storage_format* new_format) {
  //TODO: Implemented simply but inefficiently for now.  Could be more
  // locality-aware.  Optimize if this is a bottleneck.
  int matrix_size = original_format->matrix_size;
  int num_elements = matrix_size * matrix_size;
  //FIXME: We assume that every format uses the same physical storage,
  //   num_elements*sizeof(double).
  // That would be wrong for padded or sparse formats, for example.
  double* copy = malloc(sizeof(double) * num_elements);
  copy_to_format(original, original_format, copy, new_format);
  return copy;
}

void copy_to_format(double* from, square_matrix_storage_format* source_format, double* to, square_matrix_storage_format* target_format) {
  //TODO: Implemented simply but inefficiently for now.  Could be more
  // locality-aware.  Optimize if this is a bottleneck.
  // printf("copy_to_format source_format->matrix_size=%d, source_format->strategy=%d, source_format->block_size=%d, target_format->matrix_size=%d, target_format->strategy=%d, target_format->block_size=%d\n", source_format->matrix_size, source_format->strategy, source_format->block_size, target_format->matrix_size, target_format->strategy, target_format->block_size);
  int matrix_size = source_format->matrix_size;
  for (int col_idx = 0; col_idx < matrix_size; col_idx++) {
    for (int row_idx = 0; row_idx < matrix_size; row_idx++) {
      int to_idx = get_index(target_format, row_idx, col_idx);
      int from_idx = get_index(source_format, row_idx, col_idx);
      to[to_idx] = from[from_idx];
    }
  }
}

square_matrix_storage_format* padded_format(square_matrix_storage_format* format, matrix_storage_strategy new_strategy, int block_size) {
  assert(new_strategy == BLOCK_CM || new_strategy == BLOCK_RM);
  assert(block_size > 0);
  int new_matrix_size = block_size * div_round_up(format->matrix_size, block_size);
  return square_matrix_storage_format_new(new_matrix_size, new_strategy, block_size);
}

square_matrix_storage_format* padded_block_format(square_matrix_storage_format* format) {
  return padded_format(format, format->strategy, format->block_size);
}

double* pad_to_format(square_matrix_storage_format* padded_format, double* original, square_matrix_storage_format* original_format) {
  assert(is_aligned(padded_format));
  
  int new_matrix_size = padded_format->matrix_size;
  int original_matrix_size = original_format->matrix_size;
  double* new_matrix = make_arbitrary_matrix(padded_format);
  //FIXME: Iteration order is optimized for @original_format having COLUMN_MAJOR
  // format, not others.
  for (int row_idx = 0; row_idx < new_matrix_size; row_idx++) {
    for (int col_idx = 0; col_idx < new_matrix_size; col_idx++) {
      double value;
      if (row_idx >= original_matrix_size || col_idx >= original_matrix_size) {
        value = 0.0;
      } else {
        value = original[get_index(original_format, row_idx, col_idx)];
      }
      new_matrix[get_index(padded_format, row_idx, col_idx)] = value;
    }
  }
  return new_matrix;
}

double* unpad_to_format(square_matrix_storage_format* unpadded_format, double* padded, square_matrix_storage_format* padded_format) {
  int new_matrix_size = unpadded_format->matrix_size;
  double* new_matrix = make_arbitrary_matrix(unpadded_format);
  for (int row_idx = 0; row_idx < new_matrix_size; row_idx++) {
    for (int col_idx = 0; col_idx < new_matrix_size; col_idx++) {
      new_matrix[get_index(unpadded_format, row_idx, col_idx)] = padded[get_index(padded_format, row_idx, col_idx)];
    }
  }
  return new_matrix;
}

inline int get_index(square_matrix_storage_format* f, int row_idx, int col_idx) {
  // Poor man's polymorphism.
  if (f->strategy == COLUMN_MAJOR) {
    return row_idx + col_idx*f->matrix_size;
  } else { //BLOCK_CM || BLOCK_RM
    // Blocks are stored in column-major format, and the elements of blocks
    // are also stored in column-major format.
    int block_size = f->block_size;
    int block_count_tmp = block_count(f);
    int num_elements_in_block = block_size * block_size;
    // The block indices of the block in which our element lives.
    int block_col_idx = col_idx / block_size; // (Rounding down.)
    int block_row_idx = row_idx / block_size;
    // The indices of our element within its block.
    int col_in_block_idx = col_idx % block_size;
    int row_in_block_idx = row_idx % block_size;
    matrix_storage_strategy s = f->strategy;
    
    // First, we count the number of previous complete and incomplete blocks
    // so we can find the offset of the top-left element of this block.
    int num_previous_complete_blocks;
    int num_previous_incomplete_blocks;
    if (s == BLOCK_CM) {
      if (block_col_idx == (block_count_tmp-1)) {
        // We are on the right edge.  The blocks in our column are all
        // incomplete.
        num_previous_complete_blocks = (block_count_tmp-1)*block_col_idx;
        num_previous_incomplete_blocks = block_col_idx + block_row_idx;
      } else {
        // We are not on the right edge.
        num_previous_complete_blocks = block_row_idx + (block_count_tmp-1)*block_col_idx;
        num_previous_incomplete_blocks = block_col_idx;
      }
    } else {
      // s == BLOCK_RM
      if (block_row_idx == (block_count_tmp-1)) {
        // We are on the bottom edge.  The blocks in our row are all
        // incomplete.
        num_previous_complete_blocks = (block_count_tmp-1)*block_row_idx;
        num_previous_incomplete_blocks = block_row_idx + block_col_idx;
      } else {
        // We are not on the bottom edge.
        num_previous_complete_blocks = block_col_idx + (block_count_tmp-1)*block_row_idx;
        num_previous_incomplete_blocks = block_row_idx;
      }
    }
    
    // Not counting the bottom-right-hand incomplete block, each incomplete
    // block in a square matrix has the same edge size on its incomplete edge.
    int incomplete_block_size = edge_block_size(f);
    int num_elements_in_incomplete_block = incomplete_block_size * block_size;
    int block_start_idx = num_elements_in_block*num_previous_complete_blocks + num_elements_in_incomplete_block*num_previous_incomplete_blocks;
    
    int offset_in_block;
    if (s == BLOCK_CM) {
      if (block_row_idx == (block_count_tmp-1)) {
        // We are on the bottom edge.  Each column is incomplete.
        offset_in_block = row_in_block_idx + col_in_block_idx*incomplete_block_size;
      } else {
        // We are not on the bottom edge.
        offset_in_block = row_in_block_idx + col_in_block_idx*block_size;
      }
    } else {
      // s == BLOCK_RM
      if (block_col_idx == (block_count_tmp-1)) {
        // We are on the right edge.  Each row is incomplete.
        offset_in_block = col_in_block_idx + row_in_block_idx*incomplete_block_size;
      } else {
        // We are not on the right edge.
        offset_in_block = col_in_block_idx + row_in_block_idx*block_size;
      }
    }
    
    return block_start_idx + offset_in_block;
  }
}

bool equals(double* a, square_matrix_storage_format* a_format, double* b, square_matrix_storage_format* b_format) {
  return approx_equals(a, a_format, b, b_format, 0.0);
}

bool approx_equals(double* a, square_matrix_storage_format* a_format, double* b, square_matrix_storage_format* b_format, double epsilon) {
  //TODO: Optimized for column major storage.
  //TODO: Check that both are of the same size.
  if (a_format->matrix_size != b_format->matrix_size) {
    return false;
  }
  for (int col_idx = 0; col_idx < a_format->matrix_size; col_idx++) {
    for (int row_idx = 0; row_idx < a_format->matrix_size; row_idx++) {
      if (fabs(a[get_index(a_format, row_idx, col_idx)] - b[get_index(b_format, row_idx, col_idx)]) > epsilon) {
        return false;
      }
    }
  }
  return true;
}

double* scale(double scale, double* a, square_matrix_storage_format* format) {
  double* result = copy(a, format);
  //FIXME: Supports only BLOCK_CM, BLOCK_RM, and COLUMN_MAJOR.  And is inefficient.
  for (int raw_idx = 0; raw_idx < num_elements(format); raw_idx++) {
    result[raw_idx] = scale * result[raw_idx];
  }
  return result;
}

double* add(double* a, square_matrix_storage_format* a_format, double* b, square_matrix_storage_format* b_format) {
  double* result = copy(a, a_format);
  //FIXME: Supports only BLOCK_CM, BLOCK_RM, and COLUMN_MAJOR.  And assumes both formats
  // are the same.
  for (int raw_idx = 0; raw_idx < num_elements(a_format); raw_idx++) {
    result[raw_idx] = a[raw_idx] + b[raw_idx];
  }
  return result;
}

double max_norm(double* a, square_matrix_storage_format* format) {
  double result = -INFINITY;
  //FIXME: Supports only BLOCK_CM, BLOCK_RM, and COLUMN_MAJOR.
  for (int raw_idx = 0; raw_idx < num_elements(format); raw_idx++) {
    result = fmax(result, fabs(a[raw_idx]));
  }
  return result;
}

void print_matrix(double* a, square_matrix_storage_format* format) {
  for (int row_idx = 0; row_idx < format->matrix_size; row_idx++) {
    for (int col_idx = 0; col_idx < format->matrix_size; col_idx++) {
      printf("|%+4.4f", a[get_index(format, row_idx, col_idx)]);
    }
    printf("|\n");
  }
  printf("\n");
}

double* make_rand(square_matrix_storage_format* format) {
  double* p = make_arbitrary_matrix(format);
  for (int i = 0; i < num_elements(format); i++) {
    p[i] = 2 * drand48() - 1; // Uniformly distributed over [-1, 1]
  }
  return p;
}

double* make_identity(square_matrix_storage_format* format) {
  double* p = make_arbitrary_matrix(format);
  for (int i = 0; i < num_elements(format); i++) {
    p[i] = 0.0;
  }
  for (int i = 0; i < format->matrix_size; i++) {
    p[i + format->matrix_size*i] = 1.0;
  }
  return p;
}

double* make_filled(square_matrix_storage_format* format, double fill_value) {
  double* p = make_arbitrary_matrix(format);
  for (int i = 0; i < num_elements(format); i++) {
    p[i] = fill_value;
  }
  return p;
}

double* to_absolute_value (double *p, square_matrix_storage_format* format) {
  double* copy = make_arbitrary_matrix(format);
  for (int i = 0; i < num_elements(format); i++) {
    copy[i] = fabs(p[i]);
  }
  return copy;
}

char* assert_matrix_approx_equals(double* a, square_matrix_storage_format* a_format, double* b, square_matrix_storage_format* b_format, double epsilon) {
  char* assertion_message;
  asprintf(&assertion_message, "matrix a is at most %f away in max-norm from matrix b", epsilon);
  return assert_that(approx_equals(a, a_format, b, b_format, epsilon), assertion_message);
}

char* assert_matrix_equals(double* a, square_matrix_storage_format* a_format, double* b, square_matrix_storage_format* b_format) {
  return assert_matrix_approx_equals(a, a_format, b, b_format, 0.0);
}
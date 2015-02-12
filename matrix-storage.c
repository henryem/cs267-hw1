#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#include "matrix-storage.h"

#define div_round_up(a, b) (a+b-1) / b

square_matrix_storage_format* square_matrix_storage_format_new(int matrix_size, matrix_storage_strategy strategy, int block_size) {
  assert(matrix_size > 0);
  assert(strategy != BLOCK || block_size > 0);
  square_matrix_storage_format* f = malloc(sizeof(square_matrix_storage_format));
  f->matrix_size = matrix_size;
  f->strategy = strategy;
  f->block_size = block_size;
  return f;
}

int num_elements(square_matrix_storage_format* format) {
  return format->matrix_size * format->matrix_size;
}

int block_count(square_matrix_storage_format* format) {
  if (format->strategy != BLOCK) {
    return 0;
  }
  return div_round_up(format->matrix_size, format->block_size);
}

double* make_arbitrary_matrix(square_matrix_storage_format* format) {
  switch (format->strategy) {
    case COLUMN_MAJOR: return malloc(format->matrix_size*format->matrix_size * sizeof(double));
    case BLOCK: return malloc(format->matrix_size*format->matrix_size * sizeof(double));
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

/* Copy entries from @from to @to, respecting the formats of both.  It is
 * assumed that @from_format and @to_format are of the same size.
 */
void copy_to_format(double* from, square_matrix_storage_format* from_format, double* to, square_matrix_storage_format* to_format) {
  //TODO: Implemented simply but inefficiently for now.  Could be more
  // locality-aware.  Optimize if this is a bottleneck.
  // printf("copy_to_format from_format->matrix_size=%d, from_format->strategy=%d, from_format->block_size=%d, to_format->matrix_size=%d, to_format->strategy=%d, to_format->block_size=%d\n", from_format->matrix_size, from_format->strategy, from_format->block_size, to_format->matrix_size, to_format->strategy, to_format->block_size);
  int matrix_size = from_format->matrix_size;
  for (int col_idx = 0; col_idx < matrix_size; col_idx++) {
    for (int row_idx = 0; row_idx < matrix_size; row_idx++) {
      int to_idx = get_index(to_format, row_idx, col_idx);
      int from_idx = get_index(from_format, row_idx, col_idx);
      to[to_idx] = from[from_idx];
    }
  }
}

inline int get_index(square_matrix_storage_format* f, int row_idx, int col_idx) {
  // Poor man's polymorphism.
  switch (f->strategy) {
    case COLUMN_MAJOR: {
      return row_idx + col_idx*f->matrix_size;
    }
    case BLOCK: {
      // Blocks are stored in column-major format, and the elements of blocks
      // are also stored in column-major format.
      int block_size = f->block_size;
      int num_elements_in_block = block_size * block_size;
      // The number of blocks on an edge of the matrix (so the total number of
      // blocks is block_count^2).  It is assumed that the block size evenly
      // divides the matrix size.
      int block_count = f->matrix_size / block_size;
      // The block indices of the block in which our element lives.
      int block_col_idx = col_idx / block_size;
      int block_row_idx = row_idx / block_size;
      // The indices of our element within its block.
      int col_in_block_idx = col_idx % block_size;
      int row_in_block_idx = row_idx % block_size;
      // The vectorized index of the first element in the block.
      int block_start_idx = num_elements_in_block*(block_row_idx + block_col_idx*block_count);
      // Now we can find our element in this block.  Again, the elements within
      // a block are stored in column-major order.
      return block_start_idx + row_in_block_idx + col_in_block_idx*block_size;
    }
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
  //FIXME: Supports only BLOCK and COLUMN_MAJOR.  And is inefficient.
  for (int raw_idx = 0; raw_idx < num_elements(format); raw_idx++) {
    result[raw_idx] = scale * result[raw_idx];
  }
  return result;
}

double* add(double* a, square_matrix_storage_format* a_format, double* b, square_matrix_storage_format* b_format) {
  double* result = copy(a, a_format);
  //FIXME: Supports only BLOCK and COLUMN_MAJOR.  And assumes both formats
  // are the same.
  for (int raw_idx = 0; raw_idx < num_elements(a_format); raw_idx++) {
    result[raw_idx] = a[raw_idx] + b[raw_idx];
  }
  return result;
}

double max_norm(double* a, square_matrix_storage_format* format) {
  double result = -INFINITY;
  //FIXME: Supports only BLOCK and COLUMN_MAJOR.
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
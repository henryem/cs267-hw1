#include <assert.h>
#include <pmmintrin.h>
#include <emmintrin.h>

#include "matrix-storage.h"

const char* dgemm_desc = "One-level blocked dgemm with optimizations.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#if !defined(SIMD_VECTOR_SIZE)
#define SIMD_VECTOR_SIZE 2
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  c_block := c_block + a_block * b_block
 * 
 * It is assumed that a_block is stored in row-major format, b_block in column-
 * major, and c_block in column-major, all with block size block_size.
 */
static void do_block(int block_size, double* a_block, double* b_block, double* c_block) {
  /* For each column of b_block */
  for (int b_col_idx = 0; b_col_idx < block_size; b_col_idx++) {
    double* b_col = b_block + b_col_idx*block_size;
    double* c_col = c_block + b_col_idx*block_size;
    /* For each row of a_block */
    for (int a_row_idx = 0; a_row_idx < block_size; a_row_idx++) {
      /* Compute the inner product and increment c_block[a_row_idx,b_col_idx] */
      double* a_row = a_block + a_row_idx*block_size;
      double running_sum = 0.0;
      for (int inner_idx = 0; inner_idx < block_size; inner_idx++) {
	      running_sum += a_row[inner_idx] * b_col[inner_idx];
      }
      c_col[a_row_idx] += running_sum;
    }
  }
}

/* As do_block, but with performance optimizations.  The block size must divide
 * the size of the inner loop.  For now there is only 1 inner loop size:
 * SIMD_VECTOR_SIZE.
 */
static void do_block_with_simd(int block_size, double* a_block, double* b_block, double* c_block) {
  assert(block_size % SIMD_VECTOR_SIZE == 0);
  int num_subvectors = block_size / SIMD_VECTOR_SIZE;
  /* For each column of b_block */ 
  for (int b_col_idx = 0; b_col_idx < block_size; b_col_idx++) {
    double *restrict b_col = b_block + b_col_idx*block_size;
    double *restrict c_col = c_block + b_col_idx*block_size;
    /* For each row of a_block */
    for (int a_row_idx = 0; a_row_idx < block_size; a_row_idx++) {
      /* Compute the inner product and increment c_block[a_row_idx,b_col_idx].
       * We divide the inner product into a sum of inner products of vectors of
       * size SIMD_VECTOR_SIZE, to make the problem amenable to SIMD.
       */
      double *restrict a_row = a_block + a_row_idx*block_size;
      // running_sum[0] stores the current value in c, plus the dot product of
      // half the elements.  running_sum[1] stores the dot product of the other
      // half.
      register __m128d running_sum = _mm_load_sd(c_col + a_row_idx);
      for (int subvector_idx = 0; subvector_idx < num_subvectors; subvector_idx++) {
        // Load elements of a_row and b_col:
        __m128d a_elts = _mm_load_pd(a_row + subvector_idx*SIMD_VECTOR_SIZE);
        __m128d b_elts = _mm_load_pd(b_col + subvector_idx*SIMD_VECTOR_SIZE);
        running_sum = _mm_add_pd(running_sum, _mm_mul_pd(a_elts, b_elts));
      }
      _mm_store_sd(c_col + a_row_idx, _mm_hadd_pd(running_sum, running_sum)); //FIXME: Not sure this is the right operation.
    }
  }
}

static void do_full_dgemm(double* A, square_matrix_storage_format* a_format, double* B, square_matrix_storage_format* b_format, double* C, square_matrix_storage_format* c_format) {
  assert(a_format->matrix_size == b_format->matrix_size);
  assert(b_format->matrix_size == c_format->matrix_size);
  assert(a_format->strategy == BLOCK_RM);
  assert(b_format->strategy == BLOCK_CM);
  assert(c_format->strategy == BLOCK_CM);
  assert(a_format->block_size > 0);
  assert(a_format->block_size == b_format->block_size);
  assert(b_format->block_size == c_format->block_size);
  assert(a_format->matrix_size % a_format->block_size == 0);
  
  int block_size = a_format->block_size;
  int block_count = a_format->matrix_size / block_size;
  int num_elements_in_block = block_size*block_size;
  /* For each column of blocks in B */ 
  for (int b_col_idx = 0; b_col_idx < block_count; b_col_idx++) {
    /* For each row of blocks in A */
    for (int a_row_idx = 0; a_row_idx < block_count; a_row_idx++) {
      /* Compute product of the row of A and the column of B, and store it in
       * the corresponding block of C. */
      // See below for an explanation of the pointer calculations here.
      int c_block_raw_idx = a_row_idx + b_col_idx*block_count;
      int c_block_first_element_raw_idx = num_elements_in_block * c_block_raw_idx;
      double* block_pointer_in_c = C + c_block_first_element_raw_idx;
      for (int inner_idx = 0; inner_idx < block_count; inner_idx++) {
        /* The matrices are stored in a format suitable for block-multiply.
         * Each block is a contiguous chunk of memory, so do_block only needs
         * to see an appropriately-incremented pointer in the matrices.
         * Further, each row of blocks of A, and each column of blocks of B and
         * C, is contiguous.  That means that higher-level cache may be able to
         * store an entire row (or column) of blocks.  We take advantage of
         * this here by iterating through the blocks themselves in a suitable
         * order.  Note that rows of A are accessed block_count times, which is
         * unavoidable in this scheme.
         */
        // Thinking of A as a collection of blocks in row-major format, this is
        // the index of the block of A we need:
        int a_block_raw_idx = inner_idx + a_row_idx*block_count;
        // Its first actual element is this:
        int a_block_first_element_raw_idx = num_elements_in_block * a_block_raw_idx;
        // Therefore a pointer to the block in A is:
        double* block_pointer_in_a = A + a_block_first_element_raw_idx;
        // Similarly for B:
        int b_block_raw_idx = inner_idx + b_col_idx*block_count;
        int b_block_first_element_raw_idx = num_elements_in_block * b_block_raw_idx;
        double* block_pointer_in_b = B + b_block_first_element_raw_idx;
        // We already calculated the pointer for C, since the target block in C
        // doesn't change through this inner loop.
        // Now we multiply the two blocks.  We use SIMD-optimized code if
        // possible.
        if (block_size % SIMD_VECTOR_SIZE == 0) {
          do_block_with_simd(block_size, block_pointer_in_a, block_pointer_in_b, block_pointer_in_c);
        } else {
          do_block(block_size, block_pointer_in_a, block_pointer_in_b, block_pointer_in_c);
        }
      }
    }
  }
}

static int get_default_block_size(int matrix_size) {
  //FIXME
  return BLOCK_SIZE;
}

void square_dgemm_with_block_size(int matrix_size, double* A, double* B, double* C, int block_size) {
  // Copy each matrix into block format.  We cheat and break abstractions by
  // assuming that methods like to_format use the obvious internal
  // representation for block matrices.  This is for the sake of performance.
  assert(matrix_size > 0);
  assert(block_size > 0);
  square_matrix_storage_format* original_format = square_matrix_storage_format_new(matrix_size, COLUMN_MAJOR, 0);
  square_matrix_storage_format* new_a_format = padded_format(original_format, BLOCK_RM, block_size);
  square_matrix_storage_format* new_b_format = padded_format(original_format, BLOCK_CM, block_size);
  square_matrix_storage_format* new_c_format = padded_format(original_format, BLOCK_CM, block_size);
  double* formatted_a = pad_to_format(new_a_format, A, original_format);
  double* formatted_b = pad_to_format(new_b_format, B, original_format);
  double* formatted_c = pad_to_format(new_c_format, C, original_format);
  
  // Do the matrix multiply, storing the result in formatted_c.
  do_full_dgemm(formatted_a, new_a_format, formatted_b, new_b_format, formatted_c, new_c_format);
  // Uncomment (and comment other stuff) to test the (incorrect) algorithm
  // without any copying.
  // do_full_dgemm(A, new_a_format, B, new_b_format, C, new_c_format);
  
  // Copy the result back to C.
  double* unformatted_c = unpad_to_format(original_format, formatted_c, new_c_format);
  //FIXME: Extra copy here.
  copy_to_format(unformatted_c, original_format, C, original_format);
  
  free(unformatted_c);
  free(original_format);
  free(new_a_format);
  free(new_b_format);
  free(new_c_format);
  free(formatted_a);
  free(formatted_b);
  free(formatted_c);
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int matrix_size, double* A, double* B, double* C) {
  square_dgemm_with_block_size(matrix_size, A, B, C, get_default_block_size(matrix_size));
}

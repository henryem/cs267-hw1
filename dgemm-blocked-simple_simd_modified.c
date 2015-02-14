#include <assert.h>

#include "matrix-storage.h"

const char* dgemm_desc = "One-level blocked dgemm with optimizations.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 8
#endif

#if !defined(SIMD_VECTOR_SIZE)
#define SIMD_VECTOR_SIZE 8
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

/* As do_block, but with performance optimizations.  Since we want the compiler
 * to optimize an inner loop here, the block size must divide the size of the
 * inner loop.  For now there is only 1 inner loop size: SIMD_VECTOR_SIZE.
 */
static void do_block_with_autovectorization(int block_size, double* a_block, double* b_block, double* c_block) {
  assert(block_size % SIMD_VECTOR_SIZE == 0);
  int num_subvectors = block_size / SIMD_VECTOR_SIZE;
  /* For each column of b_block */ 
  for (int b_col_idx = 0; b_col_idx < block_size; b_col_idx++) {
    double* b_col = b_block + b_col_idx*block_size;
    double* c_col = c_block + b_col_idx*block_size;
    /* For each row of a_block */
    for (int a_row_idx = 0; a_row_idx < block_size; a_row_idx++) {
      /* Compute the inner product and increment c_block[a_row_idx,b_col_idx].
       * To get autovectorization to work, we divide the inner product into
       * a sum of inner products of vectors of size SIMD_VECTOR_SIZE.
       */
      double* a_row = a_block + a_row_idx*block_size;
      double running_sum = 0.0;
      for (int subvector_idx = 0; subvector_idx < num_subvectors; subvector_idx++) {
        // The following loop is supposed to be autovectorized:
        for (int inner_idx = subvector_idx*SIMD_VECTOR_SIZE; inner_idx < (subvector_idx+1)*SIMD_VECTOR_SIZE; inner_idx++) {
          running_sum += a_row[inner_idx] * b_col[inner_idx];
        }
      }
      c_col[a_row_idx] += running_sum;
    }
  }
}

static void do_block_with_simd(int n, double* a_block, double* b_block, double* c_block) {
  /*
  void *d;
  posix_memalign(&d, 16, 16*n);
  */

  int i, i2, j, j2, k, k2;
  double *restrict rC;
  double *restrict rA;
  double *restrict rB;

  int SM;

  if (n%8 == 0 || n%8 == 1 || n%8 == 3 || n%8 == 5 || n%8 == 7) SM=8;  
  else if (n%6 == 0 || n%6 == 1 || n%6 == 3 || n%6 == 5) SM=6;
  else if (n%4 == 0 || n%4 == 1 || n%4 == 3) SM=4;
  else SM=2;




  // Within the Block

  for (i = 0; i <= n-SM; i+= SM){
    for (j=0; j<= n-SM; j+=SM){
      for (k=0; k<=n-SM; k+=SM){

        
        for (i2 = 0, rC = &C[i+j*n], rB = &B[k+j*n]; i2 < SM;
        ++i2, rC += n, rB += n)
        {
          //_mm_prefetch (&rA[SM], _MM_HINT_NTA);
          for (k2 = 0, rA = &A[k+i*n]; k2 < SM; ++k2, rA += n)
          {
            __m128d m1d = _mm_load_sd (&rB[k2]);
            m1d = _mm_unpacklo_pd (m1d, m1d);
            for (j2 = 0; j2 < SM; j2 += 2)
            {
              __m128d m2 = _mm_loadu_pd (&rA[j2]); 
              __m128d r2 = _mm_loadu_pd (&rC[j2]);
              _mm_storeu_pd (&rC[j2], _mm_add_pd (_mm_mul_pd (m2, m1d), r2));
            }

          }
        }
      }
    }
  }


  // Within the block: Correction by adding the matrixs outside blocks
  if (n%SM==7){
    for (i=0;i<n-7;++i)
      for (j=0;j<n-7;++j)
        C[i+j*n] = C[i+j*n]
                  + A[(n-7)+i*n]* B[n-7+j*n]
                  + A[(n-6)+i*n]* B[n-6+j*n]
                  + A[(n-5)+i*n]* B[n-5+j*n]
                  + A[(n-4)+i*n]* B[n-4+j*n]
                  + A[(n-3)+i*n]* B[n-3+j*n]  //C[i][j] += A[i][n-1]+ B[n-1][j]
                  + A[(n-2)+i*n]* B[n-2+j*n]
                  + A[(n-1)+i*n]* B[n-1+j*n];
  }


  if (n%SM==5){
    for (i=0;i<n-5;++i)
      for (j=0;j<n-5;++j)
        C[i+j*n] = C[i+j*n]
                  + A[(n-5)+i*n]* B[n-5+j*n]
                  + A[(n-4)+i*n]* B[n-4+j*n]
                  + A[(n-3)+i*n]* B[n-3+j*n]  //C[i][j] += A[i][n-1]+ B[n-1][j]
                  + A[(n-2)+i*n]* B[n-2+j*n]
                  + A[(n-1)+i*n]* B[n-1+j*n];
  }

  else if (n%SM==3){
    for (i=0;i<n-3;++i)
      for (j=0;j<n-3;++j)
        C[i+j*n] = C[i+j*n]
                  + A[(n-3)+i*n]* B[n-3+j*n]  //C[i][j] += A[i][n-1]+ B[n-1][j]
                  + A[(n-2)+i*n]* B[n-2+j*n]
                  + A[(n-1)+i*n]* B[n-1+j*n];
  }
  else if (n%SM==1){
    for (i=0;i<n-1;++i)
      for (j=0;j<n-1;++j)
        C[i+j*n] += A[(n-1)+i*n]* B[n-1+j*n];  //C[i][j] += A[i][n-1]+ B[n-1][j]

  }



  // Calculation Outside the block

  int current_index = i;

  for (i=current_index; i < n; ++i)
    for (j=0;j<n; ++j){
      double cij = C[i+j*n];
      for (k=0;k<n;++k)
        cij += A[k+i*n] * B[k+j*n];
      C[i+j*n] = cij;
    }

  for (j = current_index; j < n; ++j)
    for (i=0; i< n-(n%SM); ++i){
      double cij = C[i+j*n];
      for (k=0; k<n; ++k)
        cij += A[k+i*n] * B[k+j*n];
      C[i+j*n] = cij;
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
        if (block_size % SIMD_VECTOR_SIZE == 0) {
          do_block_with_autovectorization(block_size, block_pointer_in_a, block_pointer_in_b, block_pointer_in_c);
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

#include <stdlib.h>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE = 1 // for asprintf
#endif
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include <float.h>  // For: DBL_EPSILON
#include <math.h>   // For: fabs

#include "unit-test-framework.h"
#include "matrix-storage.h"
#include "dgemm-blocked-simple.h"

/* reference_dgemm wraps a call to the BLAS-3 routine DGEMM, via the standard FORTRAN interface - hence the reference semantics. */ 
#define DGEMM dgemm_
extern void DGEMM (char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*); 
void reference_dgemm (int N, double ALPHA, double* A, double* B, double* C)
{
  char TRANSA = 'N';
  char TRANSB = 'N';
  int M = N;
  int K = N;
  double BETA = 1.;
  int LDA = N;
  int LDB = N;
  int LDC = N;
  DGEMM(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}   

extern const char* dgemm_desc;
extern void square_dgemm (int, double*, double*, double*);

/* Assert that square_dgemm gives approximately the same value as the reference
 * implementation for a * b.
 */
char* assert_dgemm_works(int matrix_size, double* a, double* b) {
  square_matrix_storage_format* format = square_matrix_storage_format_new(matrix_size, COLUMN_MAJOR, 0);
  double* reference_result = make_arbitrary_matrix(format);
  double* actual_result = make_arbitrary_matrix(format);
  reference_dgemm(matrix_size, 1, a, b, reference_result);
  square_dgemm(matrix_size, a, b, actual_result);
  return assert_matrix_approx_equals(reference_result, format, actual_result, format, 3.0*DBL_EPSILON*matrix_size);
}

/* Assert that square_dgemm_with_block_size gives approximately the same value as
 * the reference implementation for a * b.
 */
char* assert_dgemm_with_block_size_works(int matrix_size, double* a, double* b, int block_size) {
  square_matrix_storage_format* format = square_matrix_storage_format_new(matrix_size, COLUMN_MAJOR, 0);
  double* reference_result = make_arbitrary_matrix(format);
  double* actual_result = copy(reference_result, format);
  reference_dgemm(matrix_size, 1, a, b, reference_result);
  square_dgemm_with_block_size(matrix_size, a, b, actual_result, block_size);
  return assert_matrix_approx_equals(reference_result, format, actual_result, format, 3.0*DBL_EPSILON*matrix_size);
}

///////////////////////////////////////////////////////////////////////////////
// Begin tests
///////////////////////////////////////////////////////////////////////////////

static char* test_dgemm_args_not_modified() {
  square_matrix_storage_format* format = square_matrix_storage_format_new(10, COLUMN_MAJOR, 0);
  double* a = make_rand(format);
  double* a_copy = malloc(num_elements(format) * sizeof(double));
  memcpy(a_copy, a, num_elements(format) * sizeof(double));
  double* b = make_rand(format);
  double* b_copy = malloc(num_elements(format) * sizeof(double));
  memcpy(b_copy, b, num_elements(format) * sizeof(double));
  double* c = make_rand(format);
  square_dgemm_with_block_size(format->matrix_size, a, b, c, 2);
  return both(
    assert_matrix_equals(a, format, a_copy, format),
    assert_matrix_equals(b, format, b_copy, format));
}

static char* test_dgemm_left_identity() {
  int matrix_size = 10;
  square_matrix_storage_format* format = square_matrix_storage_format_new(matrix_size, COLUMN_MAJOR, 0);
  double* i = make_identity(format);
  double* m = make_rand(format);
  return assert_dgemm_with_block_size_works(matrix_size, i, m, 2);
}

static char* test_dgemm_right_identity() {
  int matrix_size = 6;
  square_matrix_storage_format* format = square_matrix_storage_format_new(matrix_size, COLUMN_MAJOR, 0);
  double* i = make_identity(format);
  double* m = make_rand(format);
  return assert_dgemm_with_block_size_works(matrix_size, m, i, 2);
}

static char* test_dgemm_random() {
  int matrix_size = 10;
  square_matrix_storage_format* format = square_matrix_storage_format_new(matrix_size, COLUMN_MAJOR, 0);
  double* a = make_rand(format);
  double* b = make_rand(format);
  return assert_dgemm_with_block_size_works(matrix_size, a, b, 2);
}

static char* test_dgemm_small_block() {
  int matrix_size = 10;
  square_matrix_storage_format* format = square_matrix_storage_format_new(matrix_size, COLUMN_MAJOR, 0);
  double* a = make_rand(format);
  double* b = make_rand(format);
  return assert_dgemm_with_block_size_works(matrix_size, a, b, 1);
}

static char* test_dgemm_large_block() {
  int matrix_size = 10;
  square_matrix_storage_format* format = square_matrix_storage_format_new(matrix_size, COLUMN_MAJOR, 0);
  double* a = make_rand(format);
  double* b = make_rand(format);
  return assert_dgemm_with_block_size_works(matrix_size, a, b, 10);
}

static char* test_dgemm_unaligned() {
  int matrix_size = 10;
  square_matrix_storage_format* format = square_matrix_storage_format_new(matrix_size, COLUMN_MAJOR, 0);
  double* a = make_rand(format);
  double* b = make_rand(format);
  return assert_dgemm_with_block_size_works(matrix_size, a, b, 3);
}

///////////////////////////////////////////////////////////////////////////////
// End tests
///////////////////////////////////////////////////////////////////////////////

int main (int argc, char **argv) {
  printf ("Running tests for algorithm with description: %s\n", dgemm_desc);
  test_definition* tests[] = {
    test("multiplying without changing input arguments", test_dgemm_args_not_modified),
    test("multiplying on the left by the identity", test_dgemm_left_identity),
    test("multiplying on the right by the identity", test_dgemm_right_identity),
    test("multiplying random matrices", test_dgemm_random),
    test("multiplying random matrices with a tiny block size", test_dgemm_small_block),
    test("multiplying random matrices with a large block size", test_dgemm_large_block),
    test("multiplying random matrices with a matrix size that does not align to the block size", test_dgemm_unaligned),
    NULL
  };
  run_tests(true, tests);
  return 0;
}
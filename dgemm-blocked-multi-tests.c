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
#include "matrix-blocking.h"
#include "matrix-storage.h"
#include "dgemm-blocked-multi.h"

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

/* Assert that square_dgemm_with_strategy gives approximately the same value as
 * the reference implementation for a * b.
 */
char* assert_dgemm_with_strategy_works(int matrix_size, double* a, double* b, dgemm_square_block_strategy* s) {
  square_matrix_storage_format* format = square_matrix_storage_format_new(matrix_size, COLUMN_MAJOR, 0);
  double* reference_result = make_arbitrary_matrix(format);
  double* actual_result = copy(reference_result, format);
  reference_dgemm(matrix_size, 1, a, b, reference_result);
  square_dgemm_with_strategy(matrix_size, a, b, actual_result, s);
  dgemm_square_block_strategy_free(s);
  return assert_matrix_approx_equals(reference_result, format, actual_result, format, 3.0*DBL_EPSILON*matrix_size);
}

///////////////////////////////////////////////////////////////////////////////
// Begin tests
///////////////////////////////////////////////////////////////////////////////

static char* test_to_format_cm_to_cm() {
  square_matrix_storage_format* format = square_matrix_storage_format_new(10, COLUMN_MAJOR, 0);
  double* m = make_rand(format);
  double* copy = to_format(m, format, format);
  char* result = assert_matrix_approx_equals(m, format, copy, format, 0.0);
  free(copy);
  free(m);
  free(format);
  return result;
}

static char* test_to_format_cm_to_block() {
  square_matrix_storage_format* format = square_matrix_storage_format_new(10, COLUMN_MAJOR, 0);
  double* m = make_rand(format);
  square_matrix_storage_format* block_format = square_matrix_storage_format_new(format->matrix_size, BLOCK_CM, 2);
  double* copy = to_format(m, format, block_format);
  char* result = assert_matrix_approx_equals(m, format, copy, block_format, 0.0);
  free(copy);
  free(block_format);
  free(m);
  free(format);
  return result;
}

static char* test_to_format_block_to_block() {
  square_matrix_storage_format* format = square_matrix_storage_format_new(10, BLOCK_CM, 2);
  double* m = make_rand(format);
  double* copy = to_format(m, format, format);
  char* result = assert_matrix_approx_equals(m, format, copy, format, 0.0);
  
  free(m);
  free(copy);
  free(format);
  return result;
}

static char* test_get_index_block_cm_unaligned() {
  square_matrix_storage_format* format = square_matrix_storage_format_new(5, BLOCK_CM, 2);
  return
    both(assert_int_equals(get_index(format, 0, 0), 0, "index of (0,0)"),
    both(assert_int_equals(get_index(format, 1, 0), 1, "index of (1,0)"),
    both(assert_int_equals(get_index(format, 1, 1), 3, "index of (1,1)"),
    both(assert_int_equals(get_index(format, 2, 1), 6, "index of (2,1)"),
    both(assert_int_equals(get_index(format, 1, 2), 11, "index of (1,2)"),
    both(assert_int_equals(get_index(format, 4, 1), 9, "index of (4,1)"),
    both(assert_int_equals(get_index(format, 1, 4), 21, "index of (1,4)"),
    assert_int_equals(get_index(format, 4, 4), 24, "index of (4,4)"))))))));
}

static char* test_get_index_block_cm_aligned() {
  square_matrix_storage_format* format = square_matrix_storage_format_new(6, BLOCK_CM, 2);
  return
    both(assert_int_equals(get_index(format, 0, 0), 0, "index of (0,0)"),
    both(assert_int_equals(get_index(format, 1, 0), 1, "index of (1,0)"),
    both(assert_int_equals(get_index(format, 1, 1), 3, "index of (1,1)"),
    both(assert_int_equals(get_index(format, 2, 1), 6, "index of (2,1)"),
    both(assert_int_equals(get_index(format, 1, 2), 13, "index of (1,2)"),
    both(assert_int_equals(get_index(format, 5, 1), 11, "index of (5,1)"),
    both(assert_int_equals(get_index(format, 1, 5), 27, "index of (1,5)"),
    assert_int_equals(get_index(format, 5, 5), 35, "index of (5,5)"))))))));
}

static char* test_get_index_block_rm_unaligned() {
  square_matrix_storage_format* format = square_matrix_storage_format_new(5, BLOCK_RM, 2);
  return
    both(assert_int_equals(get_index(format, 0, 0), 0, "index of (0,0)"),
    both(assert_int_equals(get_index(format, 0, 1), 1, "index of (0,1)"),
    both(assert_int_equals(get_index(format, 1, 1), 3, "index of (1,1)"),
    both(assert_int_equals(get_index(format, 1, 2), 6, "index of (1,2)"),
    both(assert_int_equals(get_index(format, 2, 1), 11, "index of (2,1)"),
    both(assert_int_equals(get_index(format, 1, 4), 9, "index of (1,4)"),
    both(assert_int_equals(get_index(format, 4, 1), 21, "index of (4,1)"),
    assert_int_equals(get_index(format, 4, 4), 24, "index of (4,4)"))))))));
}

static char* test_get_index_block_rm_aligned() {
  square_matrix_storage_format* format = square_matrix_storage_format_new(6, BLOCK_RM, 2);
  return
    both(assert_int_equals(get_index(format, 0, 0), 0, "index of (0,0)"),
    both(assert_int_equals(get_index(format, 0, 1), 1, "index of (0,1)"),
    both(assert_int_equals(get_index(format, 1, 1), 3, "index of (1,1)"),
    both(assert_int_equals(get_index(format, 1, 2), 6, "index of (1,2)"),
    both(assert_int_equals(get_index(format, 2, 1), 13, "index of (2,1)"),
    both(assert_int_equals(get_index(format, 1, 5), 11, "index of (1,5)"),
    both(assert_int_equals(get_index(format, 5, 1), 27, "index of (5,1)"),
    assert_int_equals(get_index(format, 5, 5), 35, "index of (5,5)"))))))));
}

static char* test_pad_to_format() {
  square_matrix_storage_format* format = square_matrix_storage_format_new(3, BLOCK_CM, 2);
  double* m = make_filled(format, 1.0);
  square_matrix_storage_format* padded_format_result = padded_block_format(format);
  double* padded = pad_to_format(padded_format_result, m, format);
  
  square_matrix_storage_format* expected_format = square_matrix_storage_format_new(4, BLOCK_CM, 2);
  double* expected = make_filled(expected_format, 1.0);
  expected[get_index(expected_format, 0, 3)] = 0.0;
  expected[get_index(expected_format, 1, 3)] = 0.0;
  expected[get_index(expected_format, 2, 3)] = 0.0;
  expected[get_index(expected_format, 3, 3)] = 0.0;
  expected[get_index(expected_format, 3, 2)] = 0.0;
  expected[get_index(expected_format, 3, 1)] = 0.0;
  expected[get_index(expected_format, 3, 0)] = 0.0;
  
  return both(
    assert_int_equals(padded_format_result->matrix_size, 4, "padded matrix size"),
    assert_matrix_equals(padded, padded_format_result, expected, expected_format));
}

static char* test_unpad_to_format() {
  square_matrix_storage_format* format = square_matrix_storage_format_new(4, BLOCK_CM, 2);
  double* m = make_filled(format, 1.0);
  m[get_index(format, 0, 3)] = 0.0;
  m[get_index(format, 1, 3)] = 0.0;
  m[get_index(format, 2, 3)] = 0.0;
  m[get_index(format, 3, 3)] = 0.0;
  m[get_index(format, 3, 2)] = 0.0;
  m[get_index(format, 3, 1)] = 0.0;
  m[get_index(format, 3, 0)] = 0.0;
  square_matrix_storage_format* unpadded_format = square_matrix_storage_format_new(3, BLOCK_CM, 2);
  double* unpadded = unpad_to_format(unpadded_format, m, format);
  
  double* expected = make_filled(unpadded_format, 1.0);
  
  return assert_matrix_equals(unpadded, unpadded_format, expected, unpadded_format);
}

static char* test_dgemm_args_not_modified() {
  square_matrix_storage_format* format = square_matrix_storage_format_new(10, COLUMN_MAJOR, 0);
  double* a = make_rand(format);
  double* a_copy = malloc(num_elements(format) * sizeof(double));
  memcpy(a_copy, a, num_elements(format) * sizeof(double));
  double* b = make_rand(format);
  double* b_copy = malloc(num_elements(format) * sizeof(double));
  memcpy(b_copy, b, num_elements(format) * sizeof(double));
  double* c = make_rand(format);
  int block_levels[] = {2};
  dgemm_square_block_strategy* s = dgemm_square_block_strategy_new(1, block_levels);
  square_dgemm_with_strategy(format->matrix_size, a, b, c, s);
  dgemm_square_block_strategy_free(s);
  return both(
    assert_matrix_equals(a, format, a_copy, format),
    assert_matrix_equals(b, format, b_copy, format));
}

static char* test_dgemm_left_identity() {
  int matrix_size = 10;
  square_matrix_storage_format* format = square_matrix_storage_format_new(matrix_size, COLUMN_MAJOR, 0);
  double* i = make_identity(format);
  double* m = make_rand(format);
  int block_levels[] = {2};
  dgemm_square_block_strategy* s = dgemm_square_block_strategy_new(1, block_levels);
  return assert_dgemm_with_strategy_works(matrix_size, i, m, s);
}

static char* test_dgemm_right_identity() {
  int matrix_size = 6;
  square_matrix_storage_format* format = square_matrix_storage_format_new(matrix_size, COLUMN_MAJOR, 0);
  double* i = make_identity(format);
  double* m = make_rand(format);
  int block_levels[] = {2};
  dgemm_square_block_strategy* s = dgemm_square_block_strategy_new(1, block_levels);
  return assert_dgemm_with_strategy_works(matrix_size, m, i, s);
}

static char* test_dgemm_random() {
  int matrix_size = 10;
  square_matrix_storage_format* format = square_matrix_storage_format_new(matrix_size, COLUMN_MAJOR, 0);
  double* a = make_rand(format);
  double* b = make_rand(format);
  int block_levels[] = {2};
  dgemm_square_block_strategy* s = dgemm_square_block_strategy_new(1, block_levels);
  return assert_dgemm_with_strategy_works(matrix_size, a, b, s);
}

static char* test_dgemm_multilevel() {
  int matrix_size = 30;
  square_matrix_storage_format* format = square_matrix_storage_format_new(matrix_size, COLUMN_MAJOR, 0);
  double* a = make_rand(format);
  double* b = make_rand(format);
  int block_levels[] = {2, 3};
  dgemm_square_block_strategy* s = dgemm_square_block_strategy_new(2, block_levels);
  return assert_dgemm_with_strategy_works(matrix_size, a, b, s);
}

///////////////////////////////////////////////////////////////////////////////
// End tests
///////////////////////////////////////////////////////////////////////////////

int main (int argc, char **argv) {
  printf ("Running tests for algorithm with description: %s\n", dgemm_desc);
  //TODO: Some of these are really testing matrix-storage.c.  Factor those out
  // into a separate suite.
  test_definition* tests[] = {
    test("formatting a COLUMN_MAJOR matrix to COLUMN_MAJOR", test_to_format_cm_to_cm),
    test("formatting a COLUMN_MAJOR matrix to BLOCK_CM", test_to_format_cm_to_block),
    test("formatting a BLOCK_CM matrix to BLOCK_CM", test_to_format_block_to_block),
    test("getting a raw index into an unaligned BLOCK_CM matrix", test_get_index_block_cm_unaligned),
    test("getting a raw index into an aligned BLOCK_CM matrix", test_get_index_block_cm_aligned),
    test("getting a raw index into an unaligned BLOCK_RM matrix", test_get_index_block_rm_unaligned),
    test("getting a raw index into an aligned BLOCK_RM matrix", test_get_index_block_rm_aligned),
    test("padding a matrix", test_pad_to_format),
    test("unpadding a matrix", test_unpad_to_format),
    test("multiplying without changing input arguments", test_dgemm_args_not_modified),
    test("multiplying on the left by the identity", test_dgemm_left_identity),
    test("multiplying on the right by the identity", test_dgemm_right_identity),
    test("multiplying random matrices", test_dgemm_random),
    test("multiplying with multilevel blocking", test_dgemm_multilevel),
    NULL
  };
  run_tests(true, tests);
  return 0;
}
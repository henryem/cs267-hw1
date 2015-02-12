#include <stdlib.h>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE = 1 # for asprintf
#endif
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include <float.h>  // For: DBL_EPSILON
#include <math.h>   // For: fabs

#ifdef GETTIMEOFDAY
#include <sys/time.h> // For struct timeval, gettimeofday
#else
#include <time.h> // For struct timespec, clock_gettime, CLOCK_MONOTONIC
#endif

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#define MAX_SPEED 8.4  // definning Hopper Max Gflops/s per node

#include "unit-test-framework.h"
// The tests depend on some data structures that are currently used only in
// dgemm-blocked.c .  This is a little weird.
#include "matrix-blocking.h"
#include "matrix-storage.h"
#include "dgemm-blocked.h"

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
  //FIXME: Works only if @format is COLUMN_MAJOR.
  square_matrix_storage_format* copy_format = square_matrix_storage_format_new(format->matrix_size, COLUMN_MAJOR, 0);
  double* copy = make_arbitrary_matrix(copy_format);
  for (int i = 0; i < num_elements(copy_format); i++) {
    copy[i] = fabs(p[i]);
  }
  return copy;
}

char* assert_approx_equals(double* a, square_matrix_storage_format* a_format, double* b, square_matrix_storage_format* b_format, double epsilon) {
  char* assertion_message;
  asprintf(&assertion_message, "matrix a is at most %f away in max-norm from matrix b", epsilon);
  return assert_that(approx_equals(a, a_format, b, b_format, epsilon), assertion_message);
}

char* assert_equals(double* a, square_matrix_storage_format* a_format, double* b, square_matrix_storage_format* b_format) {
  return assert_approx_equals(a, a_format, b, b_format, 0.0);
}

/* Assert that square_dgemm gives approximately the same value as the reference
 * implementation for a * b.
 */
char* assert_dgemm_works(int matrix_size, double* a, double* b) {
  square_matrix_storage_format* format = square_matrix_storage_format_new(matrix_size, COLUMN_MAJOR, 0);
  double* reference_result = make_arbitrary_matrix(format);
  double* actual_result = make_arbitrary_matrix(format);
  reference_dgemm(matrix_size, 1, a, b, reference_result);
  square_dgemm(matrix_size, a, b, actual_result);
  return assert_approx_equals(reference_result, format, actual_result, format, 3.0*DBL_EPSILON*matrix_size);
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
  return assert_approx_equals(reference_result, format, actual_result, format, 3.0*DBL_EPSILON*matrix_size);
}

///////////////////////////////////////////////////////////////////////////////
// Begin tests
///////////////////////////////////////////////////////////////////////////////

static char* test_to_format_cm_to_cm() {
  square_matrix_storage_format* format = square_matrix_storage_format_new(10, COLUMN_MAJOR, 0);
  double* m = make_rand(format);
  double* copy = to_format(m, format, format);
  return assert_approx_equals(m, format, copy, format, 0.0);
}

static char* test_to_format_cm_to_block() {
  square_matrix_storage_format* format = square_matrix_storage_format_new(10, COLUMN_MAJOR, 0);
  double* m = make_rand(format);
  square_matrix_storage_format* block_format = square_matrix_storage_format_new(format->matrix_size, BLOCK, 2);
  double* copy = to_format(m, format, block_format);
  return assert_approx_equals(m, format, copy, block_format, 0.0);
}

static char* test_to_format_block_to_block() {
  square_matrix_storage_format* format = square_matrix_storage_format_new(10, BLOCK, 2);
  double* m = make_rand(format);
  double* copy = to_format(m, format, format);
  return assert_approx_equals(m, format, copy, format, 0.0);
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
    assert_equals(a, format, a_copy, format),
    assert_equals(b, format, b_copy, format));
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
  test_definition* tests[] = {
    test("formatting a COLUMN_MAJOR matrix to COLUMN_MAJOR", test_to_format_cm_to_cm),
    test("formatting a COLUMN_MAJOR matrix to BLOCK", test_to_format_cm_to_block),
    test("formatting a BLOCK matrix to BLOCK", test_to_format_block_to_block),
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
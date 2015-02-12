#ifndef _matrix_storage_h
#define _matrix_storage_h

#include <stdlib.h>
#include <stdbool.h>

//TODO: Add a padded format?
typedef enum matrix_storage_strategy { COLUMN_MAJOR, BLOCK } matrix_storage_strategy;

typedef struct square_matrix_storage_format {
  int matrix_size;
  matrix_storage_strategy strategy;
  // HACK: Ignore block_size for strategies other than BLOCK.  Wish we were
  // working in a language with easier support for OOP.
  int block_size;
} square_matrix_storage_format;

square_matrix_storage_format* square_matrix_storage_format_new(int matrix_size, matrix_storage_strategy strategy, int block_size);

/* The total number of elements in a matrix with format @format. */
int num_elements(square_matrix_storage_format* format);

int block_count(square_matrix_storage_format* format);

/* Make a matrix of format @format, filled with arbitrary (e.g. uninitialized)
 * values.
 */
double* make_arbitrary_matrix(square_matrix_storage_format* format);

double* copy(double* original, square_matrix_storage_format* format);

/* Create a copy of @original, converting from @original_format to @new_format.
 * @new_format must be a valid format (for example, it must have the right
 * size).
 */
double* to_format(double* original, square_matrix_storage_format* original_format, square_matrix_storage_format* new_format);

/* Copy entries from @from to @to, respecting the formats of both.  It is
 * assumed that @from_format and @to_format are of the same size.
 */
void copy_to_format(double* from, square_matrix_storage_format* from_format, double* to, square_matrix_storage_format* to_format);

int get_index(square_matrix_storage_format* f, int row_idx, int col_idx);

bool equals(double* a, square_matrix_storage_format* a_format, double* b, square_matrix_storage_format* b_format);

/* Ensure that ||a - b||_\infty < epsilon. */
bool approx_equals(double* a, square_matrix_storage_format* a_format, double* b, square_matrix_storage_format* b_format, double epsilon);

double* scale(double scale, double* a, square_matrix_storage_format* format);

double* add(double* a, square_matrix_storage_format* a_format, double* b, square_matrix_storage_format* b_format);

double max_norm(double* a, square_matrix_storage_format* format);

void print_matrix(double* a, square_matrix_storage_format* format);

#endif
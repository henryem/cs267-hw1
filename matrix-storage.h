#ifndef _matrix_storage_h
#define _matrix_storage_h

#include <stdlib.h>
#include <stdbool.h>

//TODO: Add a padded format?
typedef enum matrix_storage_strategy { COLUMN_MAJOR, BLOCK_CM, BLOCK_RM } matrix_storage_strategy;

typedef struct square_matrix_storage_format {
  int matrix_size;
  matrix_storage_strategy strategy;
  // HACK: Ignore block_size for strategies other than BLOCK_CM and BLOCK_RM.
  // Wish we were working in a language with easier support for OOP.
  int block_size;
} square_matrix_storage_format;

square_matrix_storage_format* square_matrix_storage_format_new(int matrix_size, matrix_storage_strategy strategy, int block_size);

/* The total number of elements in a matrix with format @format. */
int num_elements(square_matrix_storage_format* format);

int block_count(square_matrix_storage_format* format);

/* True if @format is block-aligned (i.e. padded so that each block is of the
 * same size).
 */
bool is_aligned(square_matrix_storage_format* format);

/* If @format were block-aligned (i.e. padded so that each block were of the
 * same size), it would be of this size.
 */
int aligned_matrix_size(square_matrix_storage_format* format);

/* The size of short edge of edge-blocks of @format.  If the block size does
 * not divide the matrix size, this may differ from the size of normal blocks.
 *
 * 0 if @format is not blocked.
 */
int edge_block_size(square_matrix_storage_format* format);

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

/* When blocked into blocks of size @block_size, a matrix of size
 * @format->matrix_size may have a number of blocks that does not evenly
 * divide its size.  Return a storage format that includes padding with 0s
 * and uses @new_strategy as a storage strategy.
 * 
 * The actual padding of a physical matrix must be done separately, e.g. by
 * pad_to_format().  Example calls when @format is already blocked:
 * 
 * double* padded_matrix = pad_to_format(padded_block_format(original_format), original_matrix, original_format);
 * // Unpad later if desired:
 * double* unpadded_matrix = unpad_to_format(original_format, padded_matrix, padded_block_format(original_format));
 */
square_matrix_storage_format* padded_format(square_matrix_storage_format* format, matrix_storage_strategy new_strategy, int block_size);

/* As padded_format, but @format is used to figure out the new block storage
 * strategy and block size.  @format must be a block format.
 */
square_matrix_storage_format* padded_block_format(square_matrix_storage_format* format);

/* @original may have a number of blocks that does not evenly divide its size.
 * Make a new, larger copy of it that is padded with 0s on the right and bottom
 * edges so that the block size evenly divides the new size.  Note that the new
 * matrix is not merely a new representation of @original; it actually contains
 * the zeros.  Use unpad_to_format() to unpad, and ensure that any operations
 * commute with padding.  That is, ensure that
 *  unpad_to_format(o(pad_to_format(mat))) = o(mat) .
 * Multiplication and addition are two such operations.
 * 
 * The padded result is returned.  A copy is made even if padding is
 * unnecessary.
 */
double* pad_to_format(square_matrix_storage_format* padded_format, double* original, square_matrix_storage_format* original_format);

/* See pad_to_format().  This is the inverse of that operation. */
double* unpad_to_format(square_matrix_storage_format* unpadded_format, double* padded, square_matrix_storage_format* padded_format);

int get_index(square_matrix_storage_format* f, int row_idx, int col_idx);

bool equals(double* a, square_matrix_storage_format* a_format, double* b, square_matrix_storage_format* b_format);

/* Ensure that ||a - b||_\infty < epsilon. */
bool approx_equals(double* a, square_matrix_storage_format* a_format, double* b, square_matrix_storage_format* b_format, double epsilon);

double* scale(double scale, double* a, square_matrix_storage_format* format);

double* add(double* a, square_matrix_storage_format* a_format, double* b, square_matrix_storage_format* b_format);

double max_norm(double* a, square_matrix_storage_format* format);

void print_matrix(double* a, square_matrix_storage_format* format);

double* make_rand(square_matrix_storage_format* format);

double* make_identity(square_matrix_storage_format* format);

double* make_filled(square_matrix_storage_format* format, double fill_value);

double* to_absolute_value (double *p, square_matrix_storage_format* format);

char* assert_matrix_approx_equals(double* a, square_matrix_storage_format* a_format, double* b, square_matrix_storage_format* b_format, double epsilon);

char* assert_matrix_equals(double* a, square_matrix_storage_format* a_format, double* b, square_matrix_storage_format* b_format);

#endif
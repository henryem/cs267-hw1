#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "dgemm-blocked.h"
#include "matrix-storage.h"
#include "matrix-blocking.h"

const char* dgemm_desc = "Blocked dgemm with optimizations.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#define min(a,b) (((a)<(b))?(a):(b))

const int NUM_LEVELS = 2;
//FIXME
const int BLOCK_COUNTS[] = {8, 32};

dgemm_square_block_strategy* dgemm_square_block_strategy_new(int num_levels, const int *block_counts) {
  dgemm_square_block_strategy* s = malloc(sizeof(dgemm_square_block_strategy));
  s->num_levels = num_levels;
  int* block_counts_copy = malloc(num_levels * sizeof(int));
  memcpy(block_counts_copy, block_counts, num_levels * sizeof(int));
  s->block_counts = block_counts_copy;
  return s;
}

void dgemm_square_block_strategy_free(dgemm_square_block_strategy* s) {
  free(s->block_counts);
  free(s);
}

int smallest_block_size(dgemm_square_block_strategy* strategy, int matrix_size) {
  int block_size = matrix_size;
  for (int level = 0; level < strategy->num_levels; level++) {
    block_size /= strategy->block_counts[level];
  }
  return block_size;
}

/* Perform dgemm on a single square block -- the actual multiplication step
 *  C[i,j] += A[i,k] * B[k,j],
 * where X[l,m] refers to the (l,m) block of matrix X.
 * 
 * @param A is the whole matrix, in which @a_block references a single block.
 *   The same is true for @B.  @C is assumed to share a format with @A.
 */
static void do_block (double* A, square_mat_square_block* a_block, double* B, square_mat_square_block* b_block, double* C, square_mat_square_block* c_block) {
  int block_size = a_block->block_size;
  /* For each row of block a_block: */
  for (int a_row_idx = 0; a_row_idx < block_size; a_row_idx++) {
    /* For each column of block b_block: */
    for (int b_col_idx = 0; b_col_idx < block_size; b_col_idx++) {
      /* Compute the contribution to c_block[a_row_idx,b_col_idx] */
      double running_sum = 0;
      /* Compute the dot product <A[a_row_idx,.]^T, B[.,b_col_idx]>. */
      for (int inner_idx = 0; inner_idx < block_size; inner_idx++) {
        //TODO: Could avoid function calls (which should be inlined) and a few
        // add/subtracts (which will not be eliminated by the compiler) here by
        // directly calculating the indices.
        int a_idx = get_index_in_matrix(a_block, a_row_idx, inner_idx);
        int b_idx = get_index_in_matrix(b_block, inner_idx, b_col_idx);
	      running_sum += A[a_idx] * B[b_idx];
      }
      int c_idx = get_index_in_matrix(c_block, a_row_idx, b_col_idx);
      C[c_idx] += running_sum;
    }
  }
}

/* Perform a dgemm operation
 *  C := C + A * B
 * where A, B, and C are matrix_size-by-matrix_size matrices stored in column-major format.
 * The operation is done recursively, according to @strategy.
 */
static void recursive_block_dgemm(
    double* A,
    square_mat_square_block* a_blocks,
    double* B,
    square_mat_square_block* b_blocks,
    double* C,
    square_mat_square_block* c_block,
    dgemm_square_block_strategy* strategy,
    int level) {
  // printf("recursive_block_dgemm at level %d\n", level);
  square_mat_square_block* a_block = a_blocks + level;
  square_mat_square_block* b_block = b_blocks + level;
  if (level == 0) {
    c_block->block_row_idx = a_block->block_row_idx;
    c_block->block_col_idx = b_block->block_col_idx;
    do_block(A, a_block, B, b_block, C, c_block);
  } else {  
    int block_count = strategy->block_counts[level-1];
    square_mat_square_block* next_a_block = a_blocks + (level-1);
    square_mat_square_block* next_b_block = b_blocks + (level-1);
    /* For each block-row of A */ 
    for (int a_row_idx = 0; a_row_idx < block_count; a_row_idx++) {
      /* For each block-column of B */
      for (int b_col_idx = 0; b_col_idx < block_count; b_col_idx++) {
        /* Accumulate block dgemms into block of C */
        for (int inner_idx = 0; inner_idx < block_count; inner_idx++) {
          // Set indices in next_a_block and next_b_block to the indices of the
          // subblocks.  We construct a fixed list of block objects and then
          // modify them in-place here.  This is tricky, but it avoids
          // constructing objects here, which turns out to be too expensive.
          set_subblock(next_a_block, a_block, block_count, a_row_idx, inner_idx);
          set_subblock(next_b_block, b_block, block_count, inner_idx, b_col_idx);
          recursive_block_dgemm(A, a_blocks, B, b_blocks, C, c_block, strategy, level-1);
        }
      }
    }
  }
}

square_mat_square_block* make_blocks_for_strategy(dgemm_square_block_strategy* s, square_matrix_storage_format* f) {
  square_mat_square_block* blocks = malloc((s->num_levels+1)*sizeof(square_mat_square_block));
  int current_block_size = f->matrix_size;
  for (int level = s->num_levels; level >= 0; level--) {
    blocks[level].block_size = current_block_size;
    blocks[level].block_row_idx = 0;
    blocks[level].block_col_idx = 0;
    blocks[level].format = f;
    //FIXME: Assumes aligned blocks.
    if (level > 0) {
      current_block_size /= s->block_counts[level-1];
    }
  }
  return blocks;
}

square_mat_square_block* make_smallest_block_for_strategy(dgemm_square_block_strategy* s, square_matrix_storage_format* f) {
  return square_mat_square_block_new(smallest_block_size(s, f->matrix_size), 0, 0, f);
}

static dgemm_square_block_strategy* get_static_strategy(int matrix_size) {
  return dgemm_square_block_strategy_new(NUM_LEVELS, BLOCK_COUNTS);
}

void square_dgemm_with_strategy(int matrix_size, double* A, double* B, double* C, dgemm_square_block_strategy* dgemm_strategy) {
  //FIXME: Do padding.
  
  // Determine the storage strategy to be used for the matrices.
  int storage_block_size = smallest_block_size(dgemm_strategy, matrix_size);
  square_matrix_storage_format* new_format = square_matrix_storage_format_new(matrix_size, BLOCK_CM, storage_block_size);
  
  // Copy each matrix into block format.
  // printf("Copying matrices to block format.\n");
  square_matrix_storage_format* original_format = square_matrix_storage_format_new(matrix_size, COLUMN_MAJOR, 0);
  double* formatted_a = to_format(A, original_format, new_format);
  double* formatted_b = to_format(B, original_format, new_format);
  double* formatted_c = to_format(C, original_format, new_format);
  
  // Do the matrix multiply, storing the result in formatted_c.
  // printf("Starting recursive_block_dgemm with %d levels\n", dgemm_strategy->num_levels);
  square_mat_square_block* a_blocks = make_blocks_for_strategy(dgemm_strategy, new_format);
  square_mat_square_block* b_blocks = make_blocks_for_strategy(dgemm_strategy, new_format);
  square_mat_square_block* c_block = make_smallest_block_for_strategy(dgemm_strategy, new_format);
  
  recursive_block_dgemm(formatted_a, a_blocks, formatted_b, b_blocks, formatted_c, c_block, dgemm_strategy, dgemm_strategy->num_levels);
  
  // Copy the result back to C.
  copy_to_format(formatted_c, new_format, C, original_format);
  
  free(a_blocks);
  free(b_blocks);
  free(c_block);
  free(original_format);
  free(new_format);
  free(formatted_a);
  free(formatted_b);
  free(formatted_c);
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are matrix_size-by-matrix_size matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int matrix_size, double* A, double* B, double* C) {
  dgemm_square_block_strategy* dgemm_strategy = get_static_strategy(matrix_size);
  square_dgemm_with_strategy(matrix_size, A, B, C, dgemm_strategy);
  dgemm_square_block_strategy_free(dgemm_strategy);
}
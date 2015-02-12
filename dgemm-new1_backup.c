#include <emmintrin.h>
#include <stdio.h>

const char* dgemm_desc = "New Test1 dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 40
#endif

#define min(a,b) (((a)<(b))?(a):(b))
/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */


static void do_simd (int lda, double* A, double* B, double* C)
{
  __m128d c1 = _mm_loadu_pd (C+0);
  __m128d c2 = _mm_loadu_pd (C+lda);

  for (int i = 0; i< 2; i++)
  {
    __m128d a = _mm_loadu_pd(A+i*lda);
    __m128d b1 = _mm_load1_pd (B+0+i);
    __m128d b2 = _mm_load1_pd (B+lda+i);

    c1 = _mm_add_pd(c1, _mm_mul_pd(a,b1));
    c2 = _mm_add_pd(c2, _mm_mul_pd(a,b2));

  }
  
  _mm_storeu_pd( C+0, c1 ); 
  _mm_storeu_pd( C+lda, c2 );
}

static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  int i, j, k;
  double cij;

  for (i = 0; i < M-1; i+=2)
    for (j = 0; j < N-1; j+=2){ 
      for (k = 0; k < K-1; k+=2){
        do_simd(lda, A+ i + k*lda, B+k+j*lda,C+i+j*lda);
      }
    }

/*
  i = M-1;
  for (j = 0; j < N-1; j++){
    cij = C[i+j*lda];
    for (k = 0; k < K; k++)
      cij += A[i+k*lda] * B[k+j*lda];
    C[i+j*lda] = cij;
  }

  j = N-1;

  for (i = 0; i < M-1; i++){
    cij = C[i+j*lda];
    for (k = 0; k < K; k++)
      cij += A[i+k*lda] * B[k+j*lda];
    C[i+j*lda] = cij;
  }

  i = M-1;
  j = N-1;
  cij = C[i+j*lda];
  for (k = 0; k < K; k++)
      cij += A[i+k*lda] * B[k+j*lda];
  C[i+j*lda] = cij;


*/
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE, lda-i);
	int N = min (BLOCK_SIZE, lda-j);
	int K = min (BLOCK_SIZE, lda-k);

	/* Perform individual block dgemm */
	do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
  }
}

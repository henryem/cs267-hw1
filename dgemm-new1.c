#include <stdlib.h> 
#include <stdio.h>
#include <emmintrin.h>
const char* dgemm_desc = "Naive, three-loop dgemm.";

#define SM 4

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    




void square_dgemm (int n, double* A, double* B, double* C)
{

  int i, i2, j, j2, k, k2;
  double *restrict rC;
  double *restrict rA;
  double *restrict rB;



  for (i = 0; i < n; i+= SM){
    for (j=0; j<n; j+=SM){
      for (k=0; k<n; k+=SM){
        for (i2 = 0, rC = &C[i+j*n], rB = &B[k+j*n]; i2 < SM;
        ++i2, rC += n, rB += n)
        {
          //_mm_prefetch (&rB[SM], _MM_HINT_NTA);
          for (k2 = 0, rA = &A[i+k*n]; k2 < SM; ++k2, rA += n)
          {
            __m128d m1d = _mm_load_sd (&rB[k2]);
            m1d = _mm_unpacklo_pd (m1d, m1d);
            for (j2 = 0; j2 < SM; j2 += 2)
            {
              __m128d m2 = _mm_load_pd (&rA[j2]);
              __m128d r2 = _mm_load_pd (&rC[j2]);
              _mm_store_pd (&rC[j2],
              _mm_add_pd (_mm_mul_pd (m2, m1d), r2));
           }
          }
        }
      }
    }
  }




/*
  printf("A=\n%f %f\n%f %f\n",A[0], A[2],A[1],A[3]);
  printf("B=\n%f %f\n%f %f\n",B[0], B[2],B[1],B[3]);
  printf("C=\n%f %f\n%f %f\n",C[0], C[2],C[1],C[3]);
  printf("RealC=\n%f %f\n%f %f\n",
    A[0]*B[0]+A[2]*B[1], 
    A[0]*B[2]+A[2]*B[3],
    A[1]*B[0]+A[3]*B[1],
    A[1]*B[2]+A[3]*B[3]);
    */
/*
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) 
    {
      double cij = C[i+j*n];
      for( int k = 0; k < n; k++ )
	       cij += tmp[k+i*n] * B[k+j*n];
      C[i+j*n] = cij;
    }
    */
}

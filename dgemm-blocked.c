#include <stdlib.h> 
#include <stdio.h>
#include <emmintrin.h>

const char* dgemm_desc = "SIMPLE-SIMD";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    


void square_dgemm (int n, double* A, double* B, double* C)
{

  /*
  void *d;
  posix_memalign(&d, 16, 16*n);
  */

  int i, i2, j, j2, k, k2;
  double *restrict rC;
  double *restrict rA;
  double *restrict rB;

  int SM;


  if (n%16 == 0 || n%16 == 1 || n%16 == 3 || n%16 == 5 || n%16 == 7) SM=16;
  else if (n%14 == 0 || n%14 == 1 || n%14 == 3 || n%14 == 5 || n%14 == 7) SM=14;
  else if (n%12 == 0 || n%12 == 1 || n%12 == 3 || n%12 == 5 || n%12 == 7) SM=12;
  else if (n%10 == 0 || n%10 == 1 || n%10 == 3 || n%10 == 5 || n%10 == 7) SM=10;
  else if (n%8 == 0 || n%8 == 1 || n%8 == 3 || n%8 == 5 || n%8 == 7) SM=8;  
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
          for (k2 = 0, rA = &A[i+k*n]; k2 < SM; ++k2, rA += n)
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
                  + A[i+(n-7)*n]* B[n-7+j*n]
                  + A[i+(n-6)*n]* B[n-6+j*n]
                  + A[i+(n-5)*n]* B[n-5+j*n]
                  + A[i+(n-4)*n]* B[n-4+j*n]
                  + A[i+(n-3)*n]* B[n-3+j*n]  //C[i][j] += A[i][n-1]+ B[n-1][j]
                  + A[i+(n-2)*n]* B[n-2+j*n]
                  + A[i+(n-1)*n]* B[n-1+j*n];
  }


  if (n%SM==5){
    for (i=0;i<n-5;++i)
      for (j=0;j<n-5;++j)
        C[i+j*n] = C[i+j*n]
                  + A[i+(n-5)*n]* B[n-5+j*n]
                  + A[i+(n-4)*n]* B[n-4+j*n]
                  + A[i+(n-3)*n]* B[n-3+j*n]  //C[i][j] += A[i][n-1]+ B[n-1][j]
                  + A[i+(n-2)*n]* B[n-2+j*n]
                  + A[i+(n-1)*n]* B[n-1+j*n];
  }

  else if (n%SM==3){
    for (i=0;i<n-3;++i)
      for (j=0;j<n-3;++j)
        C[i+j*n] = C[i+j*n]
                  + A[i+(n-3)*n]* B[n-3+j*n]  //C[i][j] += A[i][n-1]+ B[n-1][j]
                  + A[i+(n-2)*n]* B[n-2+j*n]
                  + A[i+(n-1)*n]* B[n-1+j*n];
  }
  else if (n%SM==1){
    for (i=0;i<n-1;++i)
      for (j=0;j<n-1;++j)
        C[i+j*n] += A[i+(n-1)*n]* B[n-1+j*n];  //C[i][j] += A[i][n-1]+ B[n-1][j]

  }



  // Calculation Outside the block

  int current_index = i;

  for (i=current_index; i < n; ++i)
    for (j=0;j<n; ++j){
      double cij = C[i+j*n];
      for (k=0;k<n;++k)
        cij += A[i+k*n] * B[k+j*n];
      C[i+j*n] = cij;
    }

  for (j = current_index; j < n; ++j)
    for (i=0; i< n-(n%SM); ++i){
      double cij = C[i+j*n];
      for (k=0; k<n; ++k)
        cij += A[i+k*n] * B[k+j*n];
      C[i+j*n] = cij;
    }

}

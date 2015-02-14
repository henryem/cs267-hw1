#include <stdlib.h> 
#include <stdio.h>
#include <emmintrin.h>

const char* dgemm_desc = "SIMD_new";

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


/*

  // 3X3
  printf("A=\n%f %f %f\n%f %f %f\n%f %f %f\n",A[0], A[3],A[6],A[1],A[4],A[7],A[2],A[5],A[8]);
  printf("B=\n%f %f %f\n%f %f %f\n%f %f %f\n",B[0], B[3],B[6],B[1],B[4],B[7],B[2],B[5],B[8]);
  printf("C=\n%f %f %f\n%f %f %f\n%f %f %f\n",C[0], C[3],C[6],C[1],C[4],C[7],C[2],C[5],C[8]);
  printf("Real_C=\n%f %f %f\n%f %f %f\n%f %f %f\n",
    A[0]*B[0]+A[3]*B[1]+A[6]*B[2], 
    A[0]*B[3]+A[3]*B[4]+A[6]*B[5], 
    A[0]*B[6]+A[3]*B[7]+A[6]*B[8], 
    A[1]*B[0]+A[4]*B[1]+A[7]*B[2], 
    A[1]*B[3]+A[4]*B[4]+A[7]*B[5], 
    A[1]*B[6]+A[4]*B[7]+A[7]*B[8], 
    A[2]*B[0]+A[5]*B[1]+A[8]*B[2], 
    A[2]*B[3]+A[5]*B[4]+A[8]*B[5], 
    A[2]*B[6]+A[5]*B[7]+A[8]*B[8]);

*/

/*
  // 5x5
  printf("C1=\n%f %f %f %f %f\n",C[0], C[5],C[10],C[15],C[20]);
  printf("Real_C1=\n%f %f %f %f %f\n",
    A[0]*B[0]+ A[5]* B[1]+A[10]* B[2]+A[15]* B[3]+A[20]* B[4],
    A[0]*B[5]+ A[5]* B[6]+A[10]* B[7]+A[15]* B[8]+A[20]* B[9],
    A[0]*B[10]+A[5]*B[11]+A[10]*B[12]+A[15]*B[13]+A[20]*B[14],
    A[0]*B[15]+A[5]*B[16]+A[10]*B[17]+A[15]*B[18]+A[20]*B[19],
    A[0]*B[20]+A[5]*B[21]+A[10]*B[22]+A[15]*B[23]+A[20]*B[24]
    );


  printf("C2=\n%f %f %f %f %f\n",C[1], C[6],C[11],C[16],C[21]);
  printf("Real_C2=\n%f %f %f %f %f\n",
    A[1]*B[0]+ A[6]* B[1]+A[11]* B[2]+A[16]* B[3]+A[21]* B[4],
    A[1]*B[5]+ A[6]* B[6]+A[11]* B[7]+A[16]* B[8]+A[21]* B[9],
    A[1]*B[10]+A[6]*B[11]+A[11]*B[12]+A[16]*B[13]+A[21]*B[14],
    A[1]*B[15]+A[6]*B[16]+A[11]*B[17]+A[16]*B[18]+A[21]*B[19],
    A[1]*B[20]+A[6]*B[21]+A[11]*B[22]+A[16]*B[23]+A[21]*B[24]
    );

  printf("C3=\n%f %f %f %f %f\n",C[2], C[7],C[12],C[17],C[22]);
  printf("Real_C3=\n%f %f %f %f %f\n",
    A[2]*B[0]+ A[7]* B[1]+A[12]* B[2]+A[17]* B[3]+A[22]* B[4],
    A[2]*B[5]+ A[7]* B[6]+A[12]* B[7]+A[17]* B[8]+A[22]* B[9],
    A[2]*B[10]+A[7]*B[11]+A[12]*B[12]+A[17]*B[13]+A[22]*B[14],
    A[2]*B[15]+A[7]*B[16]+A[12]*B[17]+A[17]*B[18]+A[22]*B[19],
    A[2]*B[20]+A[7]*B[21]+A[12]*B[22]+A[17]*B[23]+A[22]*B[24]
    );

  printf("C4=\n%f %f %f %f %f\n",C[3], C[8],C[13],C[18],C[23]);
  printf("Real_C4=\n%f %f %f %f %f\n",
    A[3]*B[0]+ A[8]* B[1]+A[13]* B[2]+A[18]* B[3]+A[23]* B[4],
    A[3]*B[5]+ A[8]* B[6]+A[13]* B[7]+A[18]* B[8]+A[23]* B[9],
    A[3]*B[10]+A[8]*B[11]+A[13]*B[12]+A[18]*B[13]+A[23]*B[14],
    A[3]*B[15]+A[8]*B[16]+A[13]*B[17]+A[18]*B[18]+A[23]*B[19],
    A[3]*B[20]+A[8]*B[21]+A[13]*B[22]+A[18]*B[23]+A[23]*B[24]
    );

  printf("C5=\n%f %f %f %f %f\n",C[4], C[9],C[14],C[19],C[24]);
  printf("Real_C5=\n%f %f %f %f %f\n",
    A[4]*B[0]+ A[9]* B[1]+A[14]* B[2]+A[19]* B[3]+A[24]* B[4],
    A[4]*B[5]+ A[9]* B[6]+A[14]* B[7]+A[19]* B[8]+A[24]* B[9],
    A[4]*B[10]+A[9]*B[11]+A[14]*B[12]+A[19]*B[13]+A[24]*B[14],
    A[4]*B[15]+A[9]*B[16]+A[14]*B[17]+A[19]*B[18]+A[24]*B[19],
    A[4]*B[20]+A[9]*B[21]+A[14]*B[22]+A[19]*B[23]+A[24]*B[24]
    );

*/
/*
  // 7x7
  printf("C1=\n%f %f %f %f %f %f %f\n",C[0], C[7],C[14],C[21],C[28],C[35],C[42]);
  printf("Real_C1=\n%f %f %f %f %f %f %f\n",
    A[0]*B[0]+ A[7]* B[1]+A[14]* B[2]+A[21]* B[3]+A[28]* B[4]+A[35]*B[5]+A[42]*B[6],
    A[0]*B[7]+ A[7]* B[8]+A[14]* B[9]+A[21]*B[10]+A[28]* B[11]+A[35]*B[12]+A[42]*B[13],
    A[0]*B[14]+A[7]*B[15]+A[14]*B[16]+A[21]*B[17]+A[28]*B[18]+A[35]*B[19]+A[42]*B[20],
    A[0]*B[21]+A[7]*B[22]+A[14]*B[23]+A[21]*B[24]+A[28]*B[25]+A[35]*B[26]+A[42]*B[27],
    A[0]*B[28]+A[7]*B[29]+A[14]*B[30]+A[21]*B[31]+A[28]*B[32]+A[35]*B[33]+A[42]*B[34],
    A[0]*B[35]+A[7]*B[36]+A[14]*B[37]+A[21]*B[38]+A[28]*B[39]+A[35]*B[40]+A[42]*B[41],
    A[0]*B[42]+A[7]*B[43]+A[14]*B[44]+A[21]*B[45]+A[28]*B[46]+A[35]*B[47]+A[42]*B[48]
    );

*/
/*

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) 
    {
      double cij = C[i+j*n];
      for( int k = 0; k < n; k++ )
	       cij += A[k+i*n] * B[k+j*n];
      C[i+j*n] = cij;
    }
*/
}

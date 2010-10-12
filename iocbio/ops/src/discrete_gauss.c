
#include <stdlib.h>
#include <math.h>
#include "discrete_gauss.h"

int dg_convolve(double *seq, int n, double t)
{
  int k;
  double c;
  double *fseq = (double*)malloc(n*sizeof(double));
  fftw_plan plan = fftw_plan_r2r_1d(n, seq, fseq, FFTW_R2HC, FFTW_ESTIMATE);
  fftw_plan iplan = fftw_plan_r2r_1d(n, fseq, seq, FFTW_HC2R, FFTW_ESTIMATE);
  fftw_execute(plan);
  /* real part: */
  fseq[0] /= n; /* DC component */
  for (k = 1; k < (n+1)/2; ++k)  /* (k < N/2 rounded up) */
    {
      c = exp((cos((2.0*M_PI*k)/n)-1.0)*t)/n;
      fseq[k] *= c; // real part
      fseq[n-k] *= c; // imaginary part
    }
  if (n % 2 == 0) /* N is even */
    fseq[n/2] *= exp((cos((2.0*M_PI*(n/2))/n)-1.0)*t)/n;  /* Nyquist freq. */
  fftw_execute(iplan);
  fftw_destroy_plan(plan);
  fftw_destroy_plan(iplan);
  free(fseq);
  return 0;
}

int dg_high_pass_filter(double *seq, int n, int rows, double t)
{
  int k, m;
  double c;
  double *fseq = (double*)malloc((n*rows)*sizeof(double));
  int n_dims[] = {n};
  fftw_r2r_kind kinds[] = {FFTW_R2HC};
  fftw_r2r_kind ikinds[] = {FFTW_HC2R};
  fftw_plan plan = fftw_plan_many_r2r(1, // rank 
				      n_dims, // n
				      rows, // howmany
				      seq, // in
				      NULL, // inembed
				      1, // istride
				      n, // idist
				      fseq, NULL, 1, n,
				      kinds, 
				      FFTW_ESTIMATE);
  fftw_plan iplan = fftw_plan_many_r2r(1, n_dims, rows, 
				       fseq, NULL, 1, n,
				       seq, NULL, 1, n,
				       ikinds, 
				       FFTW_ESTIMATE);
  fftw_execute(plan);
  for (m=0; m<rows; ++m)
    {
      /* real part: */
      (fseq+m*n)[0] = 0.0; /* DC component */
      for (k = 1; k < (n+1)/2; ++k)  /* (k < N/2 rounded up) */
	{
	  c = (1.0-exp((cos((2.0*M_PI*k)/n)-1.0)*t))/n;
	  (fseq+m*n)[k] *= c; // real part
	  (fseq+m*n)[n-k] *= c; // imaginary part
	}
      if (n % 2 == 0) /* N is even */
	(fseq+m*n)[n/2] *= (1.0-exp((cos((2.0*M_PI*(n/2))/n)-1.0)*t))/n;  /* Nyquist freq. */
    }
  fftw_execute(iplan);
  fftw_destroy_plan(plan);
  fftw_destroy_plan(iplan);
  free(fseq);
  return 0;
}


#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

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

int dg_DGR2HCConfig_init(DGR2HCConfig *config, int rank, int dims[3], int howmany)
{
  int i, sz;
  fftw_r2r_kind *r2hckinds = NULL;
  fftw_r2r_kind *hc2rkinds = NULL;
  config->rank = rank;
  r2hckinds = (fftw_r2r_kind*)malloc(rank*sizeof(fftw_r2r_kind));
  hc2rkinds = (fftw_r2r_kind*)malloc(rank*sizeof(fftw_r2r_kind));
  sz = 1;
  for (i=0; i<rank; ++i)
    {
      sz *= dims[i];
      config->dims[i] = dims[i];
      r2hckinds[i] = FFTW_R2HC;
      hc2rkinds[i] = FFTW_HC2R;
    }
  config->howmany = howmany;
  config->sz = sz;
  config->rdata = (double*)fftw_malloc((sz*howmany)*sizeof(double));
  config->hcdata = (double*)fftw_malloc((sz*howmany)*sizeof(double));
  config->r2hc_plan = fftw_plan_many_r2r(rank, // rank 
					 config->dims, // n
					 howmany, // howmany
					 config->rdata, // in
					 NULL, // inembed
					 1, // istride
					 sz, // idist
					 config->hcdata, NULL, 1, sz,
					 r2hckinds,
					 FFTW_ESTIMATE);
  config->hc2r_plan = fftw_plan_many_r2r(rank, // rank 
					 config->dims, // n
					 howmany, // howmany
					 config->hcdata, // in
					 NULL, // inembed
					 1, // istride
					 sz, // idist
					 config->rdata, NULL, 1, sz,
					 hc2rkinds,
					 FFTW_ESTIMATE);
  free(r2hckinds);
  free(hc2rkinds);
  return 0;
}

int dg_DGR2HCConfig_clean(DGR2HCConfig *config)
{
  if (config->sz==0)
    return 0;
  fftw_destroy_plan(config->r2hc_plan);
  fftw_destroy_plan(config->hc2r_plan);
  if (config->rdata != NULL)
    fftw_free(config->rdata);
  if (config->hcdata != NULL)
    fftw_free(config->hcdata);
  if (config->kernel_real != NULL)
    free(config->kernel_real);
  if (config->kernel_imag != NULL)
    free(config->kernel_imag);
  config->rdata = NULL;
  config->hcdata = NULL;
  config->kernel_real = NULL;
  config->kernel_imag = NULL;
  config->sz = config->rank = config->howmany = 0;
  return 0;
}

int dg_DGR2HCConfig_apply_real_filter(DGR2HCConfig *config)
{
  int m, k;
  int n = config->dims[0];
  assert(config->rank==1);
  for (m=0; m<config->howmany; ++m)
    {
      (config->hcdata+m*n)[0] *= config->kernel_real[0];
      for (k = 1; k < (n+1)/2; ++k)  /* (k < N/2 rounded up) */
	{
	  (config->hcdata+m*n)[k] *= config->kernel_real[k]; // real part
	  (config->hcdata+m*n)[n-k] *= config->kernel_real[k]; // imaginary part
	}
      if (n % 2 == 0) /* N is even */
	(config->hcdata+m*n)[n/2] *= config->kernel_real[n/2];  /* Nyquist freq. */
    }
  return 0;
}

int dg_high_pass_filter_init(DGR2HCConfig *config, int rank, int dims[3], int howmany, double scale_parameter)
{
  int k;
  dg_DGR2HCConfig_init(config, rank, dims, howmany);
  config->kernel_real = (double*)malloc(config->sz*sizeof(double));
  config->kernel_imag = NULL;
  assert(rank==1); // todo: generalize to rank 2 and 3
  for (k=0; k<(dims[0]+1)/2; ++k)
    config->kernel_real[k] = (1.0-exp((cos((2.0*M_PI*k)/dims[0])-1.0)*scale_parameter))/dims[0];
  return 0;
}

int dg_high_pass_filter_clean(DGR2HCConfig *config)
{
  return dg_DGR2HCConfig_clean(config);
}

int dg_high_pass_filter_apply(DGR2HCConfig *config, double *seq)
{
  memcpy(config->rdata, seq, config->sz*config->howmany*sizeof(double));
  fftw_execute(config->r2hc_plan);
  dg_DGR2HCConfig_apply_real_filter(config);
  fftw_execute(config->hc2r_plan);
  memcpy(seq, config->rdata, config->sz*config->howmany*sizeof(double));
  return 0;
}

int dg_high_pass_filter(double *seq, int n, int rows, double t)
{
  DGR2HCConfig config;
  int dims[3] = {n,1,1};
  dg_high_pass_filter_init(&config, 1, dims, rows, t);
  dg_high_pass_filter_apply(&config, seq);
  dg_high_pass_filter_clean(&config);
  return 0;
}

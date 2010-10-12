#ifndef DISCRETE_GAUSS_H_INCLUDE
#define DISCRETE_GAUSS_H_INCLUDE
#include "fftw3.h"

typedef struct 
{
  int rank;
  int dims[3];
  int howmany;
  int sz;              /* prod(dims) */
  double *rdata;
  double *hcdata;      /* fft of rdata */
  double *kernel_real; /* filter kernel: real part */
  double *kernel_imag; /* filter kernel: imaginary part */
  fftw_plan r2hc_plan; 
  fftw_plan hc2r_plan; 
} DGR2HCConfig;


// user interface:
extern int dg_convolve(double *seq, int n, double t);
extern int dg_high_pass_filter(double *seq, int n, int rows, double t);

// user interface with fftw plan caching:
extern int dg_high_pass_filter_init(DGR2HCConfig *config, int rank, int dims[3], int howmany, double scale_parameter);
extern int dg_high_pass_filter_clean(DGR2HCConfig *config);
extern int dg_high_pass_filter_apply(DGR2HCConfig *config, double *seq);

#endif

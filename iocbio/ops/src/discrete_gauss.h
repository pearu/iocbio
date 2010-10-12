#ifndef DISCRETE_GAUSS_H_INCLUDE
#define DISCRETE_GAUSS_H_INCLUDE
#include "fftw3.h"
extern int dg_convolve(double *seq, int n, double t);
extern int dg_high_pass_filter(double *seq, int n, int rows, double t);
#endif

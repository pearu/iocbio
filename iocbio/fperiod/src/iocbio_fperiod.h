/*
  Header file for iocbio_fperiod.c. See the C source file for documentation.

  Author: Pearu Peterson
  Created: October 2011
 */

#ifndef IOCBIO_FPERIOD_H
#define IOCBIO_FPERIOD_H

#ifdef __cplusplus
extern "C" {
#endif

  extern void iocbio_objective(double *y, int k, double *f, int n, int m, int order, int method, double *r);
  extern double iocbio_fperiod(double *f, int n, int m, double initial_period, int detrend, int method);
  extern double iocbio_fperiod_cached(double *f, int n, int m, double initial_period, int detrend, int method, double *cache);
  extern double iocbio_fperiod2_cached(double *f, int n, int m, double min_period, double max_period, int detrend, int method, double *cache);

#ifdef __cplusplus
}
#endif

#endif

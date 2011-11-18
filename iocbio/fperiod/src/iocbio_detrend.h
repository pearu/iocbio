/*
  Header file for dp.c. See dp.c for documentation.

  Author: Pearu Peterson
  Created: October 2011
 */

#ifndef IOCBIO_DETREND_H
#define IOCBIO_DETREND_H

#ifdef __cplusplus
extern "C" {
#endif

extern void iocbio_detrend(double *f, int n, int m, double period, double *r);
extern void iocbio_trend(double *f, int n, int m, double period, double *r);
extern void iocbio_detrend1(double *f, int n, int fstride, double period, double *r, int rstride);
extern void iocbio_compute_trend_spline_data(double *f, int n, double period, 
				      int* nof_extremes, double* extreme_positions, double* extreme_values,
				      int* nof_averages, double* average_positions, double* average_values);

#ifdef __cplusplus
}
#endif

#endif

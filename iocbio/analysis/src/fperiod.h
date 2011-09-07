/*
  Header file for fperiod.c. See fperiod.c for documentation.

  Author: Pearu Peterson
  Created: March 2011
 */

#ifndef FPERIOD_H
#define FPERIOD_H

#ifdef __cplusplus
extern "C" {
#endif
  extern int fperiod_compute_period(double* f, int n, int m, double structure_size, double exp, int method, double* period, double* period2);
  //extern int fperiod_compute_period_cache(double* f, int n, int m, double structure_size, double exp, int method, double *r, double* period, double* period2);
  extern int fperiod_compute_period_cache(double* f, int n, int m, double structure_size, double exp, int method, double *r, double* period, double* period2);
  extern double fperiod_find_cf_maximum(double* f, int n, int m, int lbound, int ubound);
  extern double fperiod_find_cf_d2_minimum(double* f, int n, int m, int lbound, int ubound);
  extern int fperiod_find_cf_argmax_argmin2der(double* f, int n, int m, int lbound, int ubound, double *argmax, double *argmin2der);
  extern double fperiod_cf(double y, double* f, int n, int m);
  extern double fperiod_cf_d1(double y, double* f, int n, int m);
  extern double fperiod_cf_d2(double y, double* f, int n, int m);
  extern void fperiod_subtract_average1(double* f, int n, int fstride, int smoothness, double* r, int rstride);
  extern void fperiod_subtract_average(double* f, int n, int m, int structure_size, double* r);
  extern void fperiod_mark_crests1(double* f, int n, int fstride, int smoothness, double* r, int rstride);
  extern void fperiod_mark_crests(double* f, int n, int m, int structure_size, int q, double* r);
  extern void fperiod_subtract_average_2d(double* f, int n, int m, int smoothness, double* r);

#ifdef __cplusplus
}
#endif

#endif

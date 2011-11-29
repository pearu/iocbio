/*
  Header file for libfperiod.c. See the C source file for documentation.

  Author: Pearu Peterson
  Created: November 2011
 */

#ifndef LIBFPERIOD_H
#define LIBFPERIOD_H

#ifdef __cplusplus
extern "C" {
#endif
extern void iocbio_objective(double *y, int k, double *f, int n, int m, int order, int method, double *r);
extern double iocbio_fperiod(double *f, int n, int m, double initial_period, int detrend, int method);
extern double iocbio_fperiod_cached(double *f, int n, int m, double initial_period, int detrend, int method, double *cache);
extern void iocbio_detrend(double *f, int n, int m, double period, double *r);
extern void iocbio_detrend1(double *f, int n, int fstride, double period, double *r, int rstride);
extern void iocbio_ipwf_e11_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_e11_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_e11_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_e11_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_e11_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_e11_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_e11_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_e11_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_e11_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_e11_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result, double* slope);
extern double iocbio_ipwf_e11_evaluate(double y, double *fm, int n, int m, int order);
extern void iocbio_ipwf_linear_approximation_3_0(double a1_0, double a2_0, double a3_0, double* p0, double* p1);
extern void iocbio_ipwf_linear_approximation_1_1(double a1_0, double a1_1, double* p0, double* p1);
extern void iocbio_ipwf_linear_approximation_1_3(double a1_0, double a1_1, double a1_2, double a1_3, double* p0, double* p1);
extern double iocbio_ipwf_find_real_zero_in_01_2(double a_0, double a_1, double a_2, int direction, double *slope);
#ifdef __cplusplus
}
#endif

#endif

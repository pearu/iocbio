
/* This file is generated using generate_iocbio_ipwf_source.py.

  Author: Pearu Peterson
  Created: Oct 2011
*/
#ifndef IOCBIO_IPWF_H
#define IOCBIO_IPWF_H

#ifdef __cplusplus
extern "C" {
#endif
extern void iocbio_ipwf_e11_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_e11_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result);
extern int iocbio_ipwf_e11_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_e11_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_e11_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result);
extern int iocbio_ipwf_e11_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_e11_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_e11_find_extreme_diff2(int j0, int j1, double *fm, int n, int m, double* result);
extern int iocbio_ipwf_e11_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_e11_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_e11_find_extreme_diff3(int j0, int j1, double *fm, int n, int m, double* result);
extern int iocbio_ipwf_e11_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_e11_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_e11_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result);
extern int iocbio_ipwf_e11_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result, double* slope);
extern double iocbio_ipwf_e11_evaluate(double y, double *fm, int n, int m, int order);
extern double iocbio_ipwf_e11_f1_evaluate(double x, double *f, int n, int order);
extern double iocbio_ipwf_e11_f2_evaluate(double x, double *f, int n, int order);
extern void iocbio_ipwf_ep11_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_ep11_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result);
extern int iocbio_ipwf_ep11_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_ep11_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_ep11_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result);
extern int iocbio_ipwf_ep11_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_ep11_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_ep11_find_extreme_diff2(int j0, int j1, double *fm, int n, int m, double* result);
extern int iocbio_ipwf_ep11_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_ep11_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_ep11_find_extreme_diff3(int j0, int j1, double *fm, int n, int m, double* result);
extern int iocbio_ipwf_ep11_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_ep11_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_ep11_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result);
extern int iocbio_ipwf_ep11_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result, double* slope);
extern double iocbio_ipwf_ep11_evaluate(double y, double *fm, int n, int m, int order);
extern double iocbio_ipwf_ep11_f1_evaluate(double x, double *f, int n, int order);
extern double iocbio_ipwf_ep11_f2_evaluate(double x, double *f, int n, int order);
extern void iocbio_ipwf_a00_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1);
extern int iocbio_ipwf_a00_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result);
extern int iocbio_ipwf_a00_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_a00_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1);
extern int iocbio_ipwf_a00_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result);
extern int iocbio_ipwf_a00_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_a00_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1);
extern int iocbio_ipwf_a00_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result);
extern int iocbio_ipwf_a00_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result, double* slope);
extern double iocbio_ipwf_a00_evaluate(double y, double *fm, int n, int m, int order);
extern double iocbio_ipwf_a00_f1_evaluate(double x, double *f, int n, int order);
extern double iocbio_ipwf_a00_f2_evaluate(double x, double *f, int n, int order);
extern void iocbio_ipwf_a11_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_a11_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result);
extern int iocbio_ipwf_a11_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_a11_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_a11_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result);
extern int iocbio_ipwf_a11_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_a11_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_a11_find_extreme_diff2(int j0, int j1, double *fm, int n, int m, double* result);
extern int iocbio_ipwf_a11_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_a11_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_a11_find_extreme_diff3(int j0, int j1, double *fm, int n, int m, double* result);
extern int iocbio_ipwf_a11_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_a11_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_a11_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result);
extern int iocbio_ipwf_a11_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result, double* slope);
extern double iocbio_ipwf_a11_evaluate(double y, double *fm, int n, int m, int order);
extern double iocbio_ipwf_a11_f1_evaluate(double x, double *f, int n, int order);
extern double iocbio_ipwf_a11_f2_evaluate(double x, double *f, int n, int order);
extern void iocbio_ipwf_ap11_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_ap11_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result);
extern int iocbio_ipwf_ap11_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_ap11_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_ap11_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result);
extern int iocbio_ipwf_ap11_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_ap11_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_ap11_find_extreme_diff2(int j0, int j1, double *fm, int n, int m, double* result);
extern int iocbio_ipwf_ap11_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_ap11_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_ap11_find_extreme_diff3(int j0, int j1, double *fm, int n, int m, double* result);
extern int iocbio_ipwf_ap11_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result, double* slope);
extern void iocbio_ipwf_ap11_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3);
extern int iocbio_ipwf_ap11_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result);
extern int iocbio_ipwf_ap11_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result, double* slope);
extern double iocbio_ipwf_ap11_evaluate(double y, double *fm, int n, int m, int order);
extern double iocbio_ipwf_ap11_f1_evaluate(double x, double *f, int n, int order);
extern double iocbio_ipwf_ap11_f2_evaluate(double x, double *f, int n, int order);
extern void iocbio_ipwf_quadratic_approximation_3_0(double a1_0, double a2_0, double a3_0, double* p0, double* p1, double* p2);
extern void iocbio_ipwf_quadratic_approximation_1_0(double a1_0, double* p0, double* p1, double* p2);
extern void iocbio_ipwf_linear_approximation_3_0(double a1_0, double a2_0, double a3_0, double* p0, double* p1);
extern void iocbio_ipwf_linear_approximation_1_0(double a1_0, double* p0, double* p1);
extern void iocbio_ipwf_quadratic_approximation_3_1(double a1_0, double a1_1, double a2_0, double a2_1, double a3_0, double a3_1, double* p0, double* p1, double* p2);
extern void iocbio_ipwf_quadratic_approximation_1_1(double a1_0, double a1_1, double* p0, double* p1, double* p2);
extern void iocbio_ipwf_linear_approximation_3_1(double a1_0, double a1_1, double a2_0, double a2_1, double a3_0, double a3_1, double* p0, double* p1);
extern void iocbio_ipwf_linear_approximation_1_1(double a1_0, double a1_1, double* p0, double* p1);
extern void iocbio_ipwf_quadratic_approximation_3_2(double a1_0, double a1_1, double a1_2, double a2_0, double a2_1, double a2_2, double a3_0, double a3_1, double a3_2, double* p0, double* p1, double* p2);
extern void iocbio_ipwf_quadratic_approximation_1_2(double a1_0, double a1_1, double a1_2, double* p0, double* p1, double* p2);
extern void iocbio_ipwf_linear_approximation_3_2(double a1_0, double a1_1, double a1_2, double a2_0, double a2_1, double a2_2, double a3_0, double a3_1, double a3_2, double* p0, double* p1);
extern void iocbio_ipwf_linear_approximation_1_2(double a1_0, double a1_1, double a1_2, double* p0, double* p1);
extern void iocbio_ipwf_quadratic_approximation_3_3(double a1_0, double a1_1, double a1_2, double a1_3, double a2_0, double a2_1, double a2_2, double a2_3, double a3_0, double a3_1, double a3_2, double a3_3, double* p0, double* p1, double* p2);
extern void iocbio_ipwf_quadratic_approximation_1_3(double a1_0, double a1_1, double a1_2, double a1_3, double* p0, double* p1, double* p2);
extern void iocbio_ipwf_linear_approximation_3_3(double a1_0, double a1_1, double a1_2, double a1_3, double a2_0, double a2_1, double a2_2, double a2_3, double a3_0, double a3_1, double a3_2, double a3_3, double* p0, double* p1);
extern void iocbio_ipwf_linear_approximation_1_3(double a1_0, double a1_1, double a1_2, double a1_3, double* p0, double* p1);
extern void iocbio_ipwf_quadratic_approximation_3_4(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double* p0, double* p1, double* p2);
extern void iocbio_ipwf_quadratic_approximation_1_4(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double* p0, double* p1, double* p2);
extern void iocbio_ipwf_linear_approximation_3_4(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double* p0, double* p1);
extern void iocbio_ipwf_linear_approximation_1_4(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double* p0, double* p1);
extern void iocbio_ipwf_quadratic_approximation_3_5(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a2_5, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double a3_5, double* p0, double* p1, double* p2);
extern void iocbio_ipwf_quadratic_approximation_1_5(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double* p0, double* p1, double* p2);
extern void iocbio_ipwf_linear_approximation_3_5(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a2_5, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double a3_5, double* p0, double* p1);
extern void iocbio_ipwf_linear_approximation_1_5(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double* p0, double* p1);
extern void iocbio_ipwf_quadratic_approximation_3_6(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a2_5, double a2_6, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double a3_5, double a3_6, double* p0, double* p1, double* p2);
extern void iocbio_ipwf_quadratic_approximation_1_6(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double* p0, double* p1, double* p2);
extern void iocbio_ipwf_linear_approximation_3_6(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a2_5, double a2_6, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double a3_5, double a3_6, double* p0, double* p1);
extern void iocbio_ipwf_linear_approximation_1_6(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double* p0, double* p1);
extern void iocbio_ipwf_quadratic_approximation_3_7(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double a1_7, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a2_5, double a2_6, double a2_7, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double a3_5, double a3_6, double a3_7, double* p0, double* p1, double* p2);
extern void iocbio_ipwf_quadratic_approximation_1_7(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double a1_7, double* p0, double* p1, double* p2);
extern void iocbio_ipwf_linear_approximation_3_7(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double a1_7, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a2_5, double a2_6, double a2_7, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double a3_5, double a3_6, double a3_7, double* p0, double* p1);
extern void iocbio_ipwf_linear_approximation_1_7(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double a1_7, double* p0, double* p1);
extern double iocbio_ipwf_find_real_zero_in_01_2(double a_0, double a_1, double a_2, int direction, double *slope);

#ifdef __cplusplus
}
#endif
#endif

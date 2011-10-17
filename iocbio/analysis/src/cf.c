
/* This file is generated using iocbio/analysis/src/generate_cf_source.py.

  Author: Pearu Peterson
  Created: Oct 2011
*/
#include <math.h>
#include <stdio.h>
#include "cf.h"
    
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a33_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7)
{
  /* int(f1(x)*f2(x+y), x=0..L-y) = sum(a_k*r^k, k=0..7) where y=j+r
     f1(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double b6 = 0.0;
  double b7 = 0.0;
  double f_im1, f_m3mjpn, f_ipj, f_mjpn, f_n, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip3pj, f_ip2, f_ip1, f_ip1pj, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_mjpn = F(-j+n);
      f_n = F(n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_im1 = F(i-1);
        f_ipj = F(i+j);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip3pj = F(i+3+j);
        f_ip2 = F(i+2);
        f_ip1 = F(i+1);
        f_ip1pj = F(i+1+j);
        b0 += -0.01785714285714286*f_ip1pj*f_im1 - 0.01785714285714286*f_ipj*f_ip2 - 0.01785714285714286*f_im1pj*f_ip1 - 0.01785714285714286*f_ip2pj*f_i + 0.001785714285714286*f_ip2*f_im1pj + 0.1827380952380952*f_ipj*f_ip1 + 0.4047619047619048*f_ipj*f_i + 0.002380952380952381*f_im1pj*f_im1 - 0.02797619047619048*f_ip2pj*f_ip1 - 0.02797619047619048*f_i*f_im1pj + 0.1827380952380952*f_i*f_ip1pj + 0.4047619047619048*f_ip1*f_ip1pj - 0.02797619047619048*f_ipj*f_im1 + 0.001785714285714286*f_ip2pj*f_im1 - 0.02797619047619048*f_ip2*f_ip1pj + 0.002380952380952381*f_ip2pj*f_ip2;
        b1 += -0.05*f_ip1pj*f_im1 + 0.05*f_ipj*f_ip2 + 0.05*f_im1pj*f_ip1 - 0.5958333333333333*f_ipj*f_ip1 - 0.004166666666666667*f_ip2*f_im1pj - 0.05*f_ip2pj*f_i + 0.04583333333333333*f_ip2pj*f_ip1 + 0.5*f_ip1*f_ip1pj - 0.5*f_ipj*f_i - 0.04583333333333333*f_i*f_im1pj + 0.5958333333333333*f_i*f_ip1pj + 0.04583333333333333*f_ipj*f_im1 + 0.004166666666666667*f_ip2pj*f_im1 - 0.04583333333333333*f_ip2*f_ip1pj;
        b2 += -0.008333333333333333*f_ip1pj*f_im1 - 0.008333333333333333*f_ipj*f_ip2 - 0.008333333333333333*f_im1pj*f_ip1 + 0.2958333333333333*f_ipj*f_ip1 - 0.004166666666666667*f_ip2*f_im1pj - 0.008333333333333333*f_ip2pj*f_i + 0.2791666666666667*f_ip2pj*f_ip1 - 0.5666666666666667*f_ip1*f_ip1pj - 0.5666666666666667*f_ipj*f_i + 0.2791666666666667*f_i*f_im1pj + 0.2958333333333333*f_i*f_ip1pj - 0.01666666666666667*f_im1pj*f_im1 + 0.02916666666666667*f_ipj*f_im1 - 0.004166666666666667*f_ip2pj*f_im1 + 0.02916666666666667*f_ip2*f_ip1pj - 0.01666666666666667*f_ip2pj*f_ip2;
        b3 += 0.0625*f_ip1pj*f_im1 - 0.0625*f_ipj*f_ip2 - 0.1041666666666667*f_im1pj*f_ip1 + 0.4791666666666667*f_ipj*f_ip1 - 0.1666666666666667*f_ip3pj*f_ip1 + 0.02083333333333333*f_ip2*f_im1pj + 0.2708333333333333*f_ip2pj*f_i + 0.8125*f_ipj*f_i - 0.8125*f_ip1*f_ip1pj + 0.6041666666666667*f_ip2pj*f_ip1 - 0.2708333333333333*f_i*f_im1pj - 0.8125*f_i*f_ip1pj + 0.02083333333333333*f_im1pj*f_im1 - 0.0625*f_ipj*f_im1 - 0.02083333333333333*f_ip2pj*f_im1 + 0.0625*f_ip2*f_ip1pj - 0.02083333333333333*f_ip2pj*f_ip2;
        b4 += 0.125*f_im1pj*f_ip1 + 0.04166666666666667*f_ipj*f_ip2 - 0.5*f_ipj*f_ip1 + 0.125*f_ip3pj*f_ip1 - 0.02083333333333333*f_ip2*f_im1pj + 0.04166666666666667*f_ip2pj*f_i - 0.5*f_ip2pj*f_ip1 + 0.75*f_ip1*f_ip1pj - 0.04166666666666667*f_ipj*f_i + 0.02083333333333333*f_ip2*f_ip3pj + 0.02083333333333333*f_i*f_im1pj - 0.02083333333333333*f_i*f_ip3pj - 0.04166666666666667*f_ip2pj*f_ip2;
        b5 += -0.04166666666666667*f_im1pj*f_ip1 + 0.01666666666666667*f_ipj*f_ip2 - 0.02916666666666667*f_ip2*f_ip3pj + 0.01666666666666667*f_ip2pj*f_i + 0.04166666666666667*f_ip3pj*f_ip1 + 0.004166666666666667*f_ip2*f_im1pj + 0.08333333333333333*f_ipj*f_ip1 - 0.08333333333333333*f_ip2pj*f_ip1 - 0.008333333333333333*f_im1pj*f_im1 - 0.1166666666666667*f_ipj*f_i + 0.04583333333333333*f_i*f_im1pj + 0.075*f_i*f_ip1pj + 0.008333333333333333*f_ip3pj*f_im1 + 0.01666666666666667*f_ipj*f_im1 - 0.02083333333333333*f_i*f_ip3pj - 0.01666666666666667*f_ip2pj*f_im1 - 0.075*f_ip2*f_ip1pj + 0.08333333333333333*f_ip2pj*f_ip2;
        b6 += -0.025*f_ip1pj*f_im1 + 0.0125*f_ip2*f_ip3pj - 0.025*f_ipj*f_ip2 - 0.008333333333333333*f_im1pj*f_ip1 + 0.05833333333333333*f_ipj*f_ip1 - 0.03333333333333333*f_ip3pj*f_ip1 + 0.004166666666666667*f_ip2*f_im1pj - 0.09166666666666667*f_ip2pj*f_i - 0.04166666666666667*f_ipj*f_i - 0.125*f_ip1*f_ip1pj + 0.1083333333333333*f_ip2pj*f_ip1 + 0.004166666666666667*f_i*f_im1pj + 0.1*f_i*f_ip1pj - 0.008333333333333333*f_ip3pj*f_im1 + 0.008333333333333333*f_ipj*f_im1 + 0.02916666666666667*f_i*f_ip3pj + 0.025*f_ip2pj*f_im1 + 0.05*f_ip2*f_ip1pj - 0.04166666666666667*f_ip2pj*f_ip2;
        b7 += 0.01071428571428571*f_ip1pj*f_im1 - 0.001785714285714286*f_ip2*f_ip3pj + 0.007142857142857143*f_ipj*f_ip2 + 0.005357142857142857*f_im1pj*f_ip1 - 0.02142857142857143*f_ipj*f_ip1 + 0.005357142857142857*f_ip3pj*f_ip1 - 0.001785714285714286*f_ip2*f_im1pj + 0.02142857142857143*f_ip2pj*f_i - 0.02142857142857143*f_ip2pj*f_ip1 + 0.03214285714285714*f_ip1*f_ip1pj + 0.02142857142857143*f_ipj*f_i - 0.005357142857142857*f_i*f_im1pj - 0.03214285714285714*f_i*f_ip1pj + 0.001785714285714286*f_im1pj*f_im1 + 0.001785714285714286*f_ip3pj*f_im1 - 0.007142857142857143*f_ipj*f_im1 - 0.005357142857142857*f_i*f_ip3pj - 0.007142857142857143*f_ip2pj*f_im1 - 0.01071428571428571*f_ip2*f_ip1pj + 0.007142857142857143*f_ip2pj*f_ip2;
      }
      b0 += 0.001785714285714286*f_m3mjpn*f_n + 0.002380952380952381*f_n*f_mjpn + 0.1827380952380952*f_m1mjpn*f_m2pn - 0.02797619047619048*f_m2pn*f_m3mjpn - 0.02797619047619048*f_m2mjpn*f_m3pn - 0.02797619047619048*f_m1mjpn*f_n - 0.01785714285714286*f_m1mjpn*f_m3pn - 0.01785714285714286*f_m2pn*f_mjpn - 0.01785714285714286*f_m1pn*f_m3mjpn + 0.1827380952380952*f_m2mjpn*f_m1pn + 0.4047619047619048*f_m2mjpn*f_m2pn + 0.002380952380952381*f_m3pn*f_m3mjpn - 0.02797619047619048*f_m1pn*f_mjpn + 0.001785714285714286*f_m3pn*f_mjpn - 0.01785714285714286*f_m2mjpn*f_n + 0.4047619047619048*f_m1pn*f_m1mjpn;
      b1 += 0.004166666666666667*f_n*f_m3mjpn - 0.05*f_m1pn*f_m3mjpn + 0.05*f_m1mjpn*f_m3pn - 0.05*f_m2mjpn*f_n - 0.04583333333333333*f_m2mjpn*f_m3pn + 0.05*f_m2pn*f_mjpn + 0.04583333333333333*f_m2pn*f_m3mjpn - 0.5958333333333333*f_m1mjpn*f_m2pn - 0.5*f_m2mjpn*f_m2pn - 0.5*f_m1pn*f_m1mjpn - 0.04583333333333333*f_m1pn*f_mjpn - 0.004166666666666667*f_m3pn*f_mjpn + 0.04583333333333333*f_m1mjpn*f_n + 0.5958333333333333*f_m1pn*f_m2mjpn;
      b2 += -0.004166666666666667*f_n*f_m3mjpn - 0.01666666666666667*f_n*f_mjpn - 0.5666666666666667*f_m2mjpn*f_m2pn - 0.008333333333333333*f_m1pn*f_m3mjpn - 0.008333333333333333*f_m1mjpn*f_m3pn - 0.008333333333333333*f_m2mjpn*f_n + 0.02916666666666667*f_m2pn*f_m3mjpn - 0.008333333333333333*f_m2pn*f_mjpn + 0.2791666666666667*f_m2mjpn*f_m3pn - 0.5666666666666667*f_m1pn*f_m1mjpn + 0.04583333333333333*f_m1pn*f_m2mjpn - 0.01666666666666667*f_m3pn*f_m3mjpn + 0.2791666666666667*f_m1pn*f_mjpn - 0.004166666666666667*f_m3pn*f_mjpn + 0.02916666666666667*f_m1mjpn*f_n + 0.5458333333333333*f_m1mjpn*f_m2pn;
      b3 += -0.02083333333333333*f_n*f_m3mjpn + 0.02083333333333333*f_n*f_mjpn + 0.8541666666666667*f_m2mjpn*f_m2pn + 0.2291666666666667*f_m1pn*f_m3mjpn - 0.1041666666666667*f_m1mjpn*f_m3pn + 0.2291666666666667*f_m2mjpn*f_n - 0.0625*f_m2pn*f_m3mjpn - 0.1041666666666667*f_m2pn*f_mjpn - 0.2708333333333333*f_m2mjpn*f_m3pn + 0.8541666666666667*f_m1pn*f_m1mjpn - 1.479166666666667*f_m1pn*f_m2mjpn + 0.02083333333333333*f_m3pn*f_m3mjpn - 0.2708333333333333*f_m1pn*f_mjpn + 0.02083333333333333*f_m3pn*f_mjpn - 0.0625*f_m1mjpn*f_n + 0.1458333333333333*f_m1mjpn*f_m2pn;
      b4 += 0.02083333333333333*f_n*f_m3mjpn - 0.125*f_m1pn*f_m3mjpn + 0.02083333333333333*f_m2mjpn*f_m3pn - 0.125*f_m2mjpn*f_n + 0.125*f_m1mjpn*f_m3pn + 0.125*f_m2pn*f_mjpn - 0.02083333333333333*f_m2pn*f_m3mjpn + 0.4791666666666667*f_m1pn*f_m2mjpn - 0.4791666666666667*f_m2pn*f_m1mjpn + 0.02083333333333333*f_m1pn*f_mjpn - 0.02083333333333333*f_m3pn*f_mjpn - 0.02083333333333333*f_m1mjpn*f_n;
      b5 += 0.004166666666666667*f_n*f_m3mjpn - 0.008333333333333333*f_n*f_mjpn + 0.2041666666666667*f_m1pn*f_m2mjpn - 0.04166666666666667*f_m1pn*f_m3mjpn + 0.04583333333333333*f_m2pn*f_m3mjpn - 0.04166666666666667*f_m2mjpn*f_n + 0.04583333333333333*f_m2mjpn*f_m3pn - 0.04166666666666667*f_m2pn*f_mjpn - 0.04166666666666667*f_m1mjpn*f_m3pn - 0.008333333333333333*f_m3pn*f_m3mjpn - 0.2083333333333333*f_m2pn*f_m2mjpn + 0.2041666666666667*f_m2pn*f_m1mjpn + 0.04583333333333333*f_m1pn*f_mjpn + 0.004166666666666667*f_m3pn*f_mjpn + 0.04583333333333333*f_m1mjpn*f_n - 0.2083333333333333*f_m1pn*f_m1mjpn;
      b6 += -0.004166666666666667*f_n*f_m3mjpn + 0.008333333333333333*f_m1pn*f_m3mjpn + 0.004166666666666667*f_m2mjpn*f_m3pn + 0.008333333333333333*f_m2mjpn*f_n - 0.008333333333333333*f_m1mjpn*f_m3pn - 0.008333333333333333*f_m2pn*f_mjpn - 0.004166666666666667*f_m2pn*f_m3mjpn - 0.0125*f_m1pn*f_m2mjpn + 0.0125*f_m2pn*f_m1mjpn + 0.004166666666666667*f_m1pn*f_mjpn + 0.004166666666666667*f_m3pn*f_mjpn - 0.004166666666666667*f_m1mjpn*f_n;
      b7 += -0.001785714285714286*f_n*f_m3mjpn + 0.001785714285714286*f_n*f_mjpn - 0.01607142857142857*f_m1pn*f_m2mjpn + 0.005357142857142857*f_m1pn*f_m3mjpn - 0.005357142857142857*f_m2pn*f_m3mjpn + 0.005357142857142857*f_m2mjpn*f_n - 0.005357142857142857*f_m2mjpn*f_m3pn + 0.005357142857142857*f_m2pn*f_mjpn + 0.005357142857142857*f_m1mjpn*f_m3pn + 0.001785714285714286*f_m3pn*f_m3mjpn + 0.01607142857142857*f_m2pn*f_m2mjpn - 0.01607142857142857*f_m2pn*f_m1mjpn - 0.005357142857142857*f_m1pn*f_mjpn - 0.001785714285714286*f_m3pn*f_mjpn - 0.005357142857142857*f_m1mjpn*f_n + 0.01607142857142857*f_m1pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
  *a6 = b6;
  *a7 = b7;
}
        
int cf_a33_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a33_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_quadratic_approximation_1_7(a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, a1_6, a1_7, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a33_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a33_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_linear_approximation_1_7(a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, a1_6, a1_7, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a33_find_zero_diff0: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a33_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order=1) = sum(a_k*r^k, k=0..7) where y=j+r
     f1(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double b6 = 0.0;
  double b7 = 0.0;
  double f_im1, f_m3mjpn, f_ipj, f_mjpn, f_n, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip3pj, f_ip2, f_ip1, f_ip1pj, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_mjpn = F(-j+n);
      f_n = F(n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_im1 = F(i-1);
        f_ipj = F(i+j);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip3pj = F(i+3+j);
        f_ip2 = F(i+2);
        f_ip1 = F(i+1);
        f_ip1pj = F(i+1+j);
        b0 += -0.05*f_ip1pj*f_im1 + 0.05*f_ipj*f_ip2 + 0.05*f_im1pj*f_ip1 - 0.5958333333333333*f_ipj*f_ip1 - 0.004166666666666667*f_ip2*f_im1pj - 0.05*f_ip2pj*f_i + 0.04583333333333333*f_ip2pj*f_ip1 + 0.5*f_ip1*f_ip1pj - 0.5*f_ipj*f_i - 0.04583333333333333*f_i*f_im1pj + 0.5958333333333333*f_i*f_ip1pj + 0.04583333333333333*f_ipj*f_im1 + 0.004166666666666667*f_ip2pj*f_im1 - 0.04583333333333333*f_ip2*f_ip1pj;
        b1 += -0.01666666666666667*f_ip1pj*f_im1 - 0.01666666666666667*f_ipj*f_ip2 - 0.01666666666666667*f_im1pj*f_ip1 + 0.5916666666666667*f_ipj*f_ip1 - 0.008333333333333333*f_ip2*f_im1pj - 0.01666666666666667*f_ip2pj*f_i - 1.133333333333333*f_ipj*f_i - 1.133333333333333*f_ip1*f_ip1pj + 0.5583333333333333*f_ip2pj*f_ip1 + 0.5583333333333333*f_i*f_im1pj + 0.5916666666666667*f_i*f_ip1pj - 0.03333333333333333*f_im1pj*f_im1 + 0.05833333333333333*f_ipj*f_im1 - 0.008333333333333333*f_ip2pj*f_im1 + 0.05833333333333333*f_ip2*f_ip1pj - 0.03333333333333333*f_ip2pj*f_ip2;
        b2 += 0.1875*f_ip1pj*f_im1 - 0.1875*f_ipj*f_ip2 - 0.3125*f_im1pj*f_ip1 + 1.4375*f_ipj*f_ip1 - 0.5*f_ip3pj*f_ip1 + 0.0625*f_ip2*f_im1pj + 0.8125*f_ip2pj*f_i + 1.8125*f_ip2pj*f_ip1 - 2.4375*f_ip1*f_ip1pj + 2.4375*f_ipj*f_i - 0.8125*f_i*f_im1pj - 2.4375*f_i*f_ip1pj + 0.0625*f_im1pj*f_im1 - 0.1875*f_ipj*f_im1 - 0.0625*f_ip2pj*f_im1 + 0.1875*f_ip2*f_ip1pj - 0.0625*f_ip2pj*f_ip2;
        b3 += 0.5*f_im1pj*f_ip1 + 0.1666666666666667*f_ipj*f_ip2 - 2.0*f_ipj*f_ip1 + 0.5*f_ip3pj*f_ip1 - 0.08333333333333333*f_ip2*f_im1pj + 0.1666666666666667*f_ip2pj*f_i - 2.0*f_ip2pj*f_ip1 + 3.0*f_ip1*f_ip1pj - 0.1666666666666667*f_ipj*f_i + 0.08333333333333333*f_ip2*f_ip3pj + 0.08333333333333333*f_i*f_im1pj - 0.08333333333333333*f_i*f_ip3pj - 0.1666666666666667*f_ip2pj*f_ip2;
        b4 += -0.2083333333333333*f_im1pj*f_ip1 + 0.08333333333333333*f_ipj*f_ip2 - 0.1458333333333333*f_ip2*f_ip3pj + 0.08333333333333333*f_ip2pj*f_i + 0.2083333333333333*f_ip3pj*f_ip1 + 0.02083333333333333*f_ip2*f_im1pj + 0.4166666666666667*f_ipj*f_ip1 - 0.5833333333333333*f_ipj*f_i - 0.04166666666666667*f_im1pj*f_im1 - 0.4166666666666667*f_ip2pj*f_ip1 + 0.2291666666666667*f_i*f_im1pj + 0.375*f_i*f_ip1pj + 0.04166666666666667*f_ip3pj*f_im1 + 0.08333333333333333*f_ipj*f_im1 - 0.1041666666666667*f_i*f_ip3pj - 0.08333333333333333*f_ip2pj*f_im1 - 0.375*f_ip2*f_ip1pj + 0.4166666666666667*f_ip2pj*f_ip2;
        b5 += -0.15*f_ip1pj*f_im1 + 0.075*f_ip2*f_ip3pj - 0.15*f_ipj*f_ip2 - 0.05*f_im1pj*f_ip1 + 0.35*f_ipj*f_ip1 - 0.2*f_ip3pj*f_ip1 + 0.025*f_ip2*f_im1pj - 0.55*f_ip2pj*f_i + 0.65*f_ip2pj*f_ip1 - 0.75*f_ip1*f_ip1pj - 0.25*f_ipj*f_i + 0.025*f_i*f_im1pj + 0.6*f_i*f_ip1pj - 0.05*f_ip3pj*f_im1 + 0.05*f_ipj*f_im1 + 0.175*f_i*f_ip3pj + 0.15*f_ip2pj*f_im1 + 0.3*f_ip2*f_ip1pj - 0.25*f_ip2pj*f_ip2;
        b6 += 0.075*f_ip1pj*f_im1 - 0.0125*f_ip2*f_ip3pj + 0.05*f_ipj*f_ip2 + 0.0375*f_im1pj*f_ip1 - 0.15*f_ipj*f_ip1 + 0.0375*f_ip3pj*f_ip1 - 0.0125*f_ip2*f_im1pj + 0.15*f_ip2pj*f_i + 0.15*f_ipj*f_i + 0.225*f_ip1*f_ip1pj - 0.15*f_ip2pj*f_ip1 - 0.0375*f_i*f_im1pj - 0.225*f_i*f_ip1pj + 0.0125*f_im1pj*f_im1 + 0.0125*f_ip3pj*f_im1 - 0.05*f_ipj*f_im1 - 0.0375*f_i*f_ip3pj - 0.05*f_ip2pj*f_im1 - 0.075*f_ip2*f_ip1pj + 0.05*f_ip2pj*f_ip2;
      }
      b0 += 0.004166666666666667*f_n*f_m3mjpn - 0.04583333333333333*f_m2mjpn*f_m3pn + 0.05*f_m1mjpn*f_m3pn - 0.05*f_m2mjpn*f_n - 0.05*f_m1pn*f_m3mjpn + 0.05*f_m2pn*f_mjpn + 0.04583333333333333*f_m2pn*f_m3mjpn - 0.5*f_m2mjpn*f_m2pn + 0.5958333333333333*f_m1pn*f_m2mjpn - 0.5958333333333333*f_m1mjpn*f_m2pn - 0.04583333333333333*f_m1pn*f_mjpn - 0.004166666666666667*f_m3pn*f_mjpn + 0.04583333333333333*f_m1mjpn*f_n - 0.5*f_m1pn*f_m1mjpn;
      b1 += -0.008333333333333333*f_n*f_m3mjpn - 0.03333333333333333*f_n*f_mjpn - 1.133333333333333*f_m2mjpn*f_m2pn - 0.01666666666666667*f_m1pn*f_m3mjpn + 0.5583333333333333*f_m2mjpn*f_m3pn - 0.01666666666666667*f_m2mjpn*f_n - 0.01666666666666667*f_m1mjpn*f_m3pn + 0.5583333333333333*f_m1pn*f_mjpn + 0.05833333333333333*f_m2pn*f_m3mjpn + 1.091666666666667*f_m1mjpn*f_m2pn - 1.133333333333333*f_m1pn*f_m1mjpn - 0.03333333333333333*f_m3pn*f_m3mjpn - 0.01666666666666667*f_m2pn*f_mjpn - 0.008333333333333333*f_m3pn*f_mjpn + 0.05833333333333333*f_m1mjpn*f_n + 0.09166666666666667*f_m1pn*f_m2mjpn;
      b2 += -0.0625*f_n*f_m3mjpn + 0.0625*f_n*f_mjpn + 2.5625*f_m2mjpn*f_m2pn + 0.6875*f_m1pn*f_m3mjpn - 0.8125*f_m2mjpn*f_m3pn + 0.6875*f_m2mjpn*f_n - 0.3125*f_m1mjpn*f_m3pn - 0.8125*f_m1pn*f_mjpn - 0.1875*f_m2pn*f_m3mjpn + 0.4375*f_m1mjpn*f_m2pn + 2.5625*f_m1pn*f_m1mjpn + 0.0625*f_m3pn*f_m3mjpn - 0.3125*f_m2pn*f_mjpn + 0.0625*f_m3pn*f_mjpn - 0.1875*f_m1mjpn*f_n - 4.4375*f_m1pn*f_m2mjpn;
      b3 += 0.08333333333333333*f_n*f_m3mjpn - 0.08333333333333333*f_m2pn*f_m3mjpn + 0.5*f_m1mjpn*f_m3pn - 0.5*f_m2mjpn*f_n - 0.5*f_m1pn*f_m3mjpn + 0.5*f_m2pn*f_mjpn + 0.08333333333333333*f_m2mjpn*f_m3pn - 1.916666666666667*f_m2pn*f_m1mjpn + 1.916666666666667*f_m1pn*f_m2mjpn + 0.08333333333333333*f_m1pn*f_mjpn - 0.08333333333333333*f_m3pn*f_mjpn - 0.08333333333333333*f_m1mjpn*f_n;
      b4 += 0.02083333333333333*f_n*f_m3mjpn - 0.04166666666666667*f_n*f_mjpn + 1.020833333333333*f_m1pn*f_m2mjpn - 0.2083333333333333*f_m1mjpn*f_m3pn + 0.2291666666666667*f_m2mjpn*f_m3pn - 0.2083333333333333*f_m2mjpn*f_n - 0.2083333333333333*f_m1pn*f_m3mjpn - 0.2083333333333333*f_m2pn*f_mjpn + 0.2291666666666667*f_m2pn*f_m3mjpn - 0.04166666666666667*f_m3pn*f_m3mjpn + 1.020833333333333*f_m2pn*f_m1mjpn - 1.041666666666667*f_m1pn*f_m1mjpn + 0.2291666666666667*f_m1pn*f_mjpn + 0.02083333333333333*f_m3pn*f_mjpn + 0.2291666666666667*f_m1mjpn*f_n - 1.041666666666667*f_m2pn*f_m2mjpn;
      b5 += -0.025*f_n*f_m3mjpn - 0.025*f_m2pn*f_m3mjpn - 0.05*f_m1mjpn*f_m3pn + 0.05*f_m2mjpn*f_n + 0.05*f_m1pn*f_m3mjpn - 0.05*f_m2pn*f_mjpn + 0.025*f_m2mjpn*f_m3pn + 0.075*f_m2pn*f_m1mjpn - 0.075*f_m1pn*f_m2mjpn + 0.025*f_m1pn*f_mjpn + 0.025*f_m3pn*f_mjpn - 0.025*f_m1mjpn*f_n;
      b6 += -0.0125*f_n*f_m3mjpn + 0.0125*f_n*f_mjpn - 0.1125*f_m1pn*f_m2mjpn + 0.0375*f_m1mjpn*f_m3pn - 0.0375*f_m2mjpn*f_m3pn + 0.0375*f_m2mjpn*f_n + 0.0375*f_m1pn*f_m3mjpn + 0.0375*f_m2pn*f_mjpn - 0.0375*f_m2pn*f_m3mjpn + 0.0125*f_m3pn*f_m3mjpn - 0.1125*f_m2pn*f_m1mjpn + 0.1125*f_m1pn*f_m1mjpn - 0.0375*f_m1pn*f_mjpn - 0.0125*f_m3pn*f_mjpn - 0.0375*f_m1mjpn*f_n + 0.1125*f_m2pn*f_m2mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
  *a6 = b6;
  *a7 = b7;
}
        
int cf_a33_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a33_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_quadratic_approximation_1_6(a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, a1_6, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a33_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a33_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_linear_approximation_1_6(a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, a1_6, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a33_find_zero_diff1: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a33_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order=2) = sum(a_k*r^k, k=0..7) where y=j+r
     f1(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double b6 = 0.0;
  double b7 = 0.0;
  double f_im1, f_m3mjpn, f_ipj, f_m1mjpn, f_n, f_mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip3pj, f_ip2, f_ip1, f_ip1pj, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_m1mjpn = F(-1-j+n);
      f_n = F(n);
      f_mjpn = F(-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_im1 = F(i-1);
        f_ipj = F(i+j);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip3pj = F(i+3+j);
        f_ip2 = F(i+2);
        f_ip1 = F(i+1);
        f_ip1pj = F(i+1+j);
        b0 += -0.01666666666666667*f_ip1pj*f_im1 - 0.01666666666666667*f_ipj*f_ip2 - 0.01666666666666667*f_im1pj*f_ip1 + 0.5916666666666667*f_ipj*f_ip1 - 0.008333333333333333*f_ip2*f_im1pj - 0.01666666666666667*f_ip2pj*f_i + 0.5583333333333333*f_ip2pj*f_ip1 - 1.133333333333333*f_ip1*f_ip1pj - 1.133333333333333*f_ipj*f_i + 0.5583333333333333*f_i*f_im1pj + 0.5916666666666667*f_i*f_ip1pj - 0.03333333333333333*f_im1pj*f_im1 + 0.05833333333333333*f_ipj*f_im1 - 0.008333333333333333*f_ip2pj*f_im1 + 0.05833333333333333*f_ip2*f_ip1pj - 0.03333333333333333*f_ip2pj*f_ip2;
        b1 += 0.375*f_ip1pj*f_im1 - 0.375*f_ipj*f_ip2 - 0.625*f_im1pj*f_ip1 + 2.875*f_ipj*f_ip1 - f_ip3pj*f_ip1 + 0.125*f_ip2*f_im1pj + 1.625*f_ip2pj*f_i + 4.875*f_ipj*f_i - 4.875*f_ip1*f_ip1pj + 3.625*f_ip2pj*f_ip1 - 1.625*f_i*f_im1pj - 4.875*f_i*f_ip1pj + 0.125*f_im1pj*f_im1 - 0.375*f_ipj*f_im1 - 0.125*f_ip2pj*f_im1 + 0.375*f_ip2*f_ip1pj - 0.125*f_ip2pj*f_ip2;
        b2 += 1.5*f_im1pj*f_ip1 + 0.5*f_ipj*f_ip2 - 6.0*f_ipj*f_ip1 + 1.5*f_ip3pj*f_ip1 - 0.25*f_ip2*f_im1pj + 0.5*f_ip2pj*f_i - 6.0*f_ip2pj*f_ip1 + 9.0*f_ip1*f_ip1pj - 0.5*f_ipj*f_i + 0.25*f_ip2*f_ip3pj + 0.25*f_i*f_im1pj - 0.25*f_i*f_ip3pj - 0.5*f_ip2pj*f_ip2;
        b3 += -0.8333333333333333*f_im1pj*f_ip1 + 0.3333333333333333*f_ipj*f_ip2 - 0.5833333333333333*f_ip2*f_ip3pj + 0.3333333333333333*f_ip2pj*f_i + 0.8333333333333333*f_ip3pj*f_ip1 + 0.08333333333333333*f_ip2*f_im1pj + 1.666666666666667*f_ipj*f_ip1 - 1.666666666666667*f_ip2pj*f_ip1 - 0.1666666666666667*f_im1pj*f_im1 - 2.333333333333333*f_ipj*f_i + 0.9166666666666667*f_i*f_im1pj + 1.5*f_i*f_ip1pj + 0.1666666666666667*f_ip3pj*f_im1 + 0.3333333333333333*f_ipj*f_im1 - 0.4166666666666667*f_i*f_ip3pj - 0.3333333333333333*f_ip2pj*f_im1 - 1.5*f_ip2*f_ip1pj + 1.666666666666667*f_ip2pj*f_ip2;
        b4 += -0.75*f_ip1pj*f_im1 + 0.375*f_ip2*f_ip3pj - 0.75*f_ipj*f_ip2 - 0.25*f_im1pj*f_ip1 + 1.75*f_ipj*f_ip1 - f_ip3pj*f_ip1 + 0.125*f_ip2*f_im1pj - 2.75*f_ip2pj*f_i - 1.25*f_ipj*f_i - 3.75*f_ip1*f_ip1pj + 3.25*f_ip2pj*f_ip1 + 0.125*f_i*f_im1pj + 3.0*f_i*f_ip1pj - 0.25*f_ip3pj*f_im1 + 0.25*f_ipj*f_im1 + 0.875*f_i*f_ip3pj + 0.75*f_ip2pj*f_im1 + 1.5*f_ip2*f_ip1pj - 1.25*f_ip2pj*f_ip2;
        b5 += 0.45*f_ip1pj*f_im1 - 0.075*f_ip2*f_ip3pj + 0.3*f_ipj*f_ip2 + 0.225*f_im1pj*f_ip1 - 0.9*f_ipj*f_ip1 + 0.225*f_ip3pj*f_ip1 - 0.075*f_ip2*f_im1pj + 0.9*f_ip2pj*f_i - 0.9*f_ip2pj*f_ip1 + 1.35*f_ip1*f_ip1pj + 0.9*f_ipj*f_i - 0.225*f_i*f_im1pj - 1.35*f_i*f_ip1pj + 0.075*f_im1pj*f_im1 + 0.075*f_ip3pj*f_im1 - 0.3*f_ipj*f_im1 - 0.225*f_i*f_ip3pj - 0.3*f_ip2pj*f_im1 - 0.45*f_ip2*f_ip1pj + 0.3*f_ip2pj*f_ip2;
      }
      b0 += -0.008333333333333333*f_n*f_m3mjpn - 0.03333333333333333*f_n*f_mjpn - 1.133333333333333*f_m2mjpn*f_m2pn - 0.01666666666666667*f_m1pn*f_m3mjpn - 0.01666666666666667*f_m1mjpn*f_m3pn - 0.01666666666666667*f_m2mjpn*f_n + 0.5583333333333333*f_m2mjpn*f_m3pn + 0.5583333333333333*f_m1pn*f_mjpn + 0.05833333333333333*f_m2pn*f_m3mjpn + 0.09166666666666667*f_m1pn*f_m2mjpn + 1.091666666666667*f_m1mjpn*f_m2pn - 0.03333333333333333*f_m3pn*f_m3mjpn - 0.01666666666666667*f_m2pn*f_mjpn - 0.008333333333333333*f_m3pn*f_mjpn + 0.05833333333333333*f_m1mjpn*f_n - 1.133333333333333*f_m1pn*f_m1mjpn;
      b1 += -0.125*f_n*f_m3mjpn + 0.125*f_n*f_mjpn + 5.125*f_m2mjpn*f_m2pn + 1.375*f_m1pn*f_m3mjpn - 0.625*f_m1mjpn*f_m3pn + 1.375*f_m2mjpn*f_n - 1.625*f_m2mjpn*f_m3pn - 1.625*f_m1pn*f_mjpn - 0.375*f_m2pn*f_m3mjpn - 8.875*f_m1pn*f_m2mjpn + 0.875*f_m1mjpn*f_m2pn + 0.125*f_m3pn*f_m3mjpn - 0.625*f_m2pn*f_mjpn + 0.125*f_m3pn*f_mjpn - 0.375*f_m1mjpn*f_n + 5.125*f_m1pn*f_m1mjpn;
      b2 += 0.25*f_n*f_m3mjpn - 1.5*f_m1pn*f_m3mjpn + 0.25*f_m2mjpn*f_m3pn - 1.5*f_m2mjpn*f_n + 1.5*f_m1mjpn*f_m3pn + 1.5*f_m2pn*f_mjpn - 0.25*f_m2pn*f_m3mjpn + 5.75*f_m1pn*f_m2mjpn - 5.75*f_m2pn*f_m1mjpn + 0.25*f_m1pn*f_mjpn - 0.25*f_m3pn*f_mjpn - 0.25*f_m1mjpn*f_n;
      b3 += 0.08333333333333333*f_n*f_m3mjpn - 0.1666666666666667*f_n*f_mjpn + 4.083333333333333*f_m1pn*f_m2mjpn + 0.9166666666666667*f_m2pn*f_m3mjpn + 0.9166666666666667*f_m2mjpn*f_m3pn - 0.8333333333333333*f_m2mjpn*f_n - 0.8333333333333333*f_m1mjpn*f_m3pn + 0.9166666666666667*f_m1pn*f_mjpn - 0.8333333333333333*f_m1pn*f_m3mjpn - 4.166666666666667*f_m2pn*f_m2mjpn - 4.166666666666667*f_m1pn*f_m1mjpn - 0.1666666666666667*f_m3pn*f_m3mjpn - 0.8333333333333333*f_m2pn*f_mjpn + 0.08333333333333333*f_m3pn*f_mjpn + 0.9166666666666667*f_m1mjpn*f_n + 4.083333333333333*f_m2pn*f_m1mjpn;
      b4 += -0.125*f_n*f_m3mjpn + 0.25*f_m1pn*f_m3mjpn + 0.125*f_m2mjpn*f_m3pn + 0.25*f_m2mjpn*f_n - 0.25*f_m1mjpn*f_m3pn - 0.25*f_m2pn*f_mjpn - 0.125*f_m2pn*f_m3mjpn - 0.375*f_m1pn*f_m2mjpn + 0.375*f_m2pn*f_m1mjpn + 0.125*f_m1pn*f_mjpn + 0.125*f_m3pn*f_mjpn - 0.125*f_m1mjpn*f_n;
      b5 += -0.075*f_n*f_m3mjpn + 0.075*f_n*f_mjpn - 0.675*f_m1pn*f_m2mjpn - 0.225*f_m2pn*f_m3mjpn - 0.225*f_m2mjpn*f_m3pn + 0.225*f_m2mjpn*f_n + 0.225*f_m1mjpn*f_m3pn - 0.225*f_m1pn*f_mjpn + 0.225*f_m1pn*f_m3mjpn + 0.675*f_m2pn*f_m2mjpn + 0.675*f_m1pn*f_m1mjpn + 0.075*f_m3pn*f_m3mjpn + 0.225*f_m2pn*f_mjpn - 0.075*f_m3pn*f_mjpn - 0.225*f_m1mjpn*f_n - 0.675*f_m2pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
  *a6 = b6;
  *a7 = b7;
}
        
int cf_a33_find_extreme_diff2(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a33_compute_coeffs_diff2(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_quadratic_approximation_1_5(a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a33_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a33_compute_coeffs_diff2(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_linear_approximation_1_5(a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a33_find_zero_diff2: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a33_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order=3) = sum(a_k*r^k, k=0..7) where y=j+r
     f1(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double b6 = 0.0;
  double b7 = 0.0;
  double f_im1, f_m3mjpn, f_ipj, f_m1mjpn, f_n, f_mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip3pj, f_ip2, f_ip1, f_ip1pj, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_m1mjpn = F(-1-j+n);
      f_n = F(n);
      f_mjpn = F(-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_im1 = F(i-1);
        f_ipj = F(i+j);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip3pj = F(i+3+j);
        f_ip2 = F(i+2);
        f_ip1 = F(i+1);
        f_ip1pj = F(i+1+j);
        b0 += 0.375*f_ip1pj*f_im1 - 0.375*f_ipj*f_ip2 - 0.625*f_im1pj*f_ip1 + 2.875*f_ipj*f_ip1 - f_ip3pj*f_ip1 + 0.125*f_ip2*f_im1pj + 1.625*f_ip2pj*f_i + 3.625*f_ip2pj*f_ip1 - 4.875*f_ip1*f_ip1pj + 4.875*f_ipj*f_i - 1.625*f_i*f_im1pj - 4.875*f_i*f_ip1pj + 0.125*f_im1pj*f_im1 - 0.375*f_ipj*f_im1 - 0.125*f_ip2pj*f_im1 + 0.375*f_ip2*f_ip1pj - 0.125*f_ip2pj*f_ip2;
        b1 += 3.0*f_im1pj*f_ip1 + f_ipj*f_ip2 - 12.0*f_ipj*f_ip1 + 3.0*f_ip3pj*f_ip1 - 0.5*f_ip2*f_im1pj + f_ip2pj*f_i - 12.0*f_ip2pj*f_ip1 + 18.0*f_ip1*f_ip1pj - f_ipj*f_i + 0.5*f_ip2*f_ip3pj + 0.5*f_i*f_im1pj - 0.5*f_i*f_ip3pj - f_ip2pj*f_ip2;
        b2 += -2.5*f_im1pj*f_ip1 + f_ipj*f_ip2 - 1.75*f_ip2*f_ip3pj + f_ip2pj*f_i + 2.5*f_ip3pj*f_ip1 + 0.25*f_ip2*f_im1pj + 5.0*f_ipj*f_ip1 - 7.0*f_ipj*f_i - 0.5*f_im1pj*f_im1 - 5.0*f_ip2pj*f_ip1 + 2.75*f_i*f_im1pj + 4.5*f_i*f_ip1pj + 0.5*f_ip3pj*f_im1 + f_ipj*f_im1 - 1.25*f_i*f_ip3pj - f_ip2pj*f_im1 - 4.5*f_ip2*f_ip1pj + 5.0*f_ip2pj*f_ip2;
        b3 += -3.0*f_ip1pj*f_im1 + 1.5*f_ip2*f_ip3pj - 3.0*f_ipj*f_ip2 - f_im1pj*f_ip1 + 7.0*f_ipj*f_ip1 - 4.0*f_ip3pj*f_ip1 + 0.5*f_ip2*f_im1pj - 11.0*f_ip2pj*f_i + 13.0*f_ip2pj*f_ip1 - 15.0*f_ip1*f_ip1pj - 5.0*f_ipj*f_i + 0.5*f_i*f_im1pj + 12.0*f_i*f_ip1pj - f_ip3pj*f_im1 + f_ipj*f_im1 + 3.5*f_i*f_ip3pj + 3.0*f_ip2pj*f_im1 + 6.0*f_ip2*f_ip1pj - 5.0*f_ip2pj*f_ip2;
        b4 += 2.25*f_ip1pj*f_im1 - 0.375*f_ip2*f_ip3pj + 1.5*f_ipj*f_ip2 + 1.125*f_im1pj*f_ip1 - 4.5*f_ipj*f_ip1 + 1.125*f_ip3pj*f_ip1 - 0.375*f_ip2*f_im1pj + 4.5*f_ip2pj*f_i + 4.5*f_ipj*f_i + 6.75*f_ip1*f_ip1pj - 4.5*f_ip2pj*f_ip1 - 1.125*f_i*f_im1pj - 6.75*f_i*f_ip1pj + 0.375*f_im1pj*f_im1 + 0.375*f_ip3pj*f_im1 - 1.5*f_ipj*f_im1 - 1.125*f_i*f_ip3pj - 1.5*f_ip2pj*f_im1 - 2.25*f_ip2*f_ip1pj + 1.5*f_ip2pj*f_ip2;
      }
      b0 += -0.125*f_n*f_m3mjpn + 0.125*f_n*f_mjpn + 5.125*f_m2mjpn*f_m2pn + 1.375*f_m1pn*f_m3mjpn - 1.625*f_m2mjpn*f_m3pn + 1.375*f_m2mjpn*f_n - 0.625*f_m1mjpn*f_m3pn - 1.625*f_m1pn*f_mjpn - 0.375*f_m2pn*f_m3mjpn + 5.125*f_m1pn*f_m1mjpn - 8.875*f_m1pn*f_m2mjpn + 0.125*f_m3pn*f_m3mjpn - 0.625*f_m2pn*f_mjpn + 0.125*f_m3pn*f_mjpn - 0.375*f_m1mjpn*f_n + 0.875*f_m1mjpn*f_m2pn;
      b1 += 0.5*f_n*f_m3mjpn - 0.5*f_m2pn*f_m3mjpn + 3.0*f_m1mjpn*f_m3pn - 3.0*f_m2mjpn*f_n - 3.0*f_m1pn*f_m3mjpn + 3.0*f_m2pn*f_mjpn + 0.5*f_m2mjpn*f_m3pn - 11.5*f_m2pn*f_m1mjpn + 11.5*f_m1pn*f_m2mjpn + 0.5*f_m1pn*f_mjpn - 0.5*f_m3pn*f_mjpn - 0.5*f_m1mjpn*f_n;
      b2 += 0.25*f_n*f_m3mjpn - 0.5*f_n*f_mjpn + 12.25*f_m1pn*f_m2mjpn + 2.75*f_m2pn*f_m3mjpn - 2.5*f_m1mjpn*f_m3pn - 2.5*f_m2mjpn*f_n + 2.75*f_m2mjpn*f_m3pn + 2.75*f_m1pn*f_mjpn - 2.5*f_m1pn*f_m3mjpn + 12.25*f_m2pn*f_m1mjpn - 12.5*f_m2pn*f_m2mjpn - 0.5*f_m3pn*f_m3mjpn - 2.5*f_m2pn*f_mjpn + 0.25*f_m3pn*f_mjpn + 2.75*f_m1mjpn*f_n - 12.5*f_m1pn*f_m1mjpn;
      b3 += -0.5*f_n*f_m3mjpn - 0.5*f_m2pn*f_m3mjpn - f_m1mjpn*f_m3pn + f_m2mjpn*f_n + f_m1pn*f_m3mjpn - f_m2pn*f_mjpn + 0.5*f_m2mjpn*f_m3pn + 1.5*f_m2pn*f_m1mjpn - 1.5*f_m1pn*f_m2mjpn + 0.5*f_m1pn*f_mjpn + 0.5*f_m3pn*f_mjpn - 0.5*f_m1mjpn*f_n;
      b4 += -0.375*f_n*f_m3mjpn + 0.375*f_n*f_mjpn - 3.375*f_m1pn*f_m2mjpn - 1.125*f_m2pn*f_m3mjpn + 1.125*f_m1mjpn*f_m3pn + 1.125*f_m2mjpn*f_n - 1.125*f_m2mjpn*f_m3pn - 1.125*f_m1pn*f_mjpn + 1.125*f_m1pn*f_m3mjpn - 3.375*f_m2pn*f_m1mjpn + 3.375*f_m2pn*f_m2mjpn + 0.375*f_m3pn*f_m3mjpn + 1.125*f_m2pn*f_mjpn - 0.375*f_m3pn*f_mjpn - 1.125*f_m1mjpn*f_n + 3.375*f_m1pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
  *a6 = b6;
  *a7 = b7;
}
        
int cf_a33_find_extreme_diff3(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a33_compute_coeffs_diff3(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_quadratic_approximation_1_4(a1_0, a1_1, a1_2, a1_3, a1_4, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a33_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a33_compute_coeffs_diff3(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_linear_approximation_1_4(a1_0, a1_1, a1_2, a1_3, a1_4, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a33_find_zero_diff3: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a33_compute_coeffs_diff4(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order=4) = sum(a_k*r^k, k=0..7) where y=j+r
     f1(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double b6 = 0.0;
  double b7 = 0.0;
  double f_ipj, f_m3mjpn, f_ip2pj, f_im1, f_mjpn, f_n, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_m2pn, f_ip2, f_ip3pj, f_im1pj, f_ip1, f_ip1pj, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_mjpn = F(-j+n);
      f_n = F(n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip2pj = F(i+2+j);
        f_im1 = F(i-1);
        f_i = F(i);
        f_ip2 = F(i+2);
        f_ip3pj = F(i+3+j);
        f_im1pj = F(i-1+j);
        f_ip1 = F(i+1);
        f_ip1pj = F(i+1+j);
        b0 += 3.0*f_im1pj*f_ip1 + f_ipj*f_ip2 - 12.0*f_ipj*f_ip1 + 3.0*f_ip3pj*f_ip1 - 0.5*f_ip2*f_im1pj + f_ip2pj*f_i - 12.0*f_ip2pj*f_ip1 + 18.0*f_ip1*f_ip1pj - f_ipj*f_i + 0.5*f_ip2*f_ip3pj + 0.5*f_i*f_im1pj - 0.5*f_i*f_ip3pj - f_ip2pj*f_ip2;
        b1 += -5.0*f_im1pj*f_ip1 + 2.0*f_ipj*f_ip2 - 3.5*f_ip2*f_ip3pj + 2.0*f_ip2pj*f_i + 5.0*f_ip3pj*f_ip1 + 0.5*f_ip2*f_im1pj + 10.0*f_ipj*f_ip1 - 10.0*f_ip2pj*f_ip1 - f_im1pj*f_im1 - 14.0*f_ipj*f_i + 5.5*f_i*f_im1pj + 9.0*f_i*f_ip1pj + f_ip3pj*f_im1 + 2.0*f_ipj*f_im1 - 2.5*f_i*f_ip3pj - 2.0*f_ip2pj*f_im1 - 9.0*f_ip2*f_ip1pj + 10.0*f_ip2pj*f_ip2;
        b2 += -9.0*f_ip1pj*f_im1 + 4.5*f_ip2*f_ip3pj - 9.0*f_ipj*f_ip2 - 3.0*f_im1pj*f_ip1 + 21.0*f_ipj*f_ip1 - 12.0*f_ip3pj*f_ip1 + 1.5*f_ip2*f_im1pj - 33.0*f_ip2pj*f_i - 15.0*f_ipj*f_i - 45.0*f_ip1*f_ip1pj + 39.0*f_ip2pj*f_ip1 + 1.5*f_i*f_im1pj + 36.0*f_i*f_ip1pj - 3.0*f_ip3pj*f_im1 + 3.0*f_ipj*f_im1 + 10.5*f_i*f_ip3pj + 9.0*f_ip2pj*f_im1 + 18.0*f_ip2*f_ip1pj - 15.0*f_ip2pj*f_ip2;
        b3 += 9.0*f_ip1pj*f_im1 - 1.5*f_ip2*f_ip3pj + 6.0*f_ipj*f_ip2 + 4.5*f_im1pj*f_ip1 - 18.0*f_ipj*f_ip1 + 4.5*f_ip3pj*f_ip1 - 1.5*f_ip2*f_im1pj + 18.0*f_ip2pj*f_i - 18.0*f_ip2pj*f_ip1 + 27.0*f_ip1*f_ip1pj + 18.0*f_ipj*f_i - 4.5*f_i*f_im1pj - 27.0*f_i*f_ip1pj + 1.5*f_im1pj*f_im1 + 1.5*f_ip3pj*f_im1 - 6.0*f_ipj*f_im1 - 4.5*f_i*f_ip3pj - 6.0*f_ip2pj*f_im1 - 9.0*f_ip2*f_ip1pj + 6.0*f_ip2pj*f_ip2;
      }
      b0 += 0.5*f_n*f_m3mjpn - 3.0*f_m1pn*f_m3mjpn + 0.5*f_m2mjpn*f_m3pn - 3.0*f_m2mjpn*f_n + 3.0*f_m1mjpn*f_m3pn + 3.0*f_m2pn*f_mjpn - 0.5*f_m2pn*f_m3mjpn + 11.5*f_m1pn*f_m2mjpn - 11.5*f_m2pn*f_m1mjpn + 0.5*f_m1pn*f_mjpn - 0.5*f_m3pn*f_mjpn - 0.5*f_m1mjpn*f_n;
      b1 += 0.5*f_n*f_m3mjpn - f_n*f_mjpn + 24.5*f_m1pn*f_m2mjpn + 5.5*f_m2pn*f_m3mjpn + 5.5*f_m2mjpn*f_m3pn - 5.0*f_m2mjpn*f_n - 5.0*f_m1mjpn*f_m3pn + 5.5*f_m1pn*f_mjpn - 5.0*f_m1pn*f_m3mjpn - 25.0*f_m1pn*f_m1mjpn + 24.5*f_m2pn*f_m1mjpn - f_m3pn*f_m3mjpn - 5.0*f_m2pn*f_mjpn + 0.5*f_m3pn*f_mjpn + 5.5*f_m1mjpn*f_n - 25.0*f_m2pn*f_m2mjpn;
      b2 += -1.5*f_n*f_m3mjpn + 3.0*f_m1pn*f_m3mjpn + 1.5*f_m2mjpn*f_m3pn + 3.0*f_m2mjpn*f_n - 3.0*f_m1mjpn*f_m3pn - 3.0*f_m2pn*f_mjpn - 1.5*f_m2pn*f_m3mjpn - 4.5*f_m1pn*f_m2mjpn + 4.5*f_m2pn*f_m1mjpn + 1.5*f_m1pn*f_mjpn + 1.5*f_m3pn*f_mjpn - 1.5*f_m1mjpn*f_n;
      b3 += -1.5*f_n*f_m3mjpn + 1.5*f_n*f_mjpn - 13.5*f_m1pn*f_m2mjpn - 4.5*f_m2pn*f_m3mjpn - 4.5*f_m2mjpn*f_m3pn + 4.5*f_m2mjpn*f_n + 4.5*f_m1mjpn*f_m3pn - 4.5*f_m1pn*f_mjpn + 4.5*f_m1pn*f_m3mjpn + 13.5*f_m1pn*f_m1mjpn - 13.5*f_m2pn*f_m1mjpn + 1.5*f_m3pn*f_m3mjpn + 4.5*f_m2pn*f_mjpn - 1.5*f_m3pn*f_mjpn - 4.5*f_m1mjpn*f_n + 13.5*f_m2pn*f_m2mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
  *a6 = b6;
  *a7 = b7;
}
        
int cf_a33_find_extreme_diff4(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a33_compute_coeffs_diff4(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_quadratic_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a33_find_zero_diff4(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a33_compute_coeffs_diff4(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_linear_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a33_find_zero_diff4: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a33_compute_coeffs_diff5(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order=5) = sum(a_k*r^k, k=0..7) where y=j+r
     f1(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double b6 = 0.0;
  double b7 = 0.0;
  double f_ipj, f_m3mjpn, f_ip2pj, f_im1, f_m1mjpn, f_n, f_mjpn, f_m3pn, f_i, f_m2mjpn, f_m2pn, f_ip2, f_ip3pj, f_im1pj, f_ip1, f_ip1pj, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_m1mjpn = F(-1-j+n);
      f_n = F(n);
      f_mjpn = F(-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip2pj = F(i+2+j);
        f_im1 = F(i-1);
        f_i = F(i);
        f_ip2 = F(i+2);
        f_ip3pj = F(i+3+j);
        f_im1pj = F(i-1+j);
        f_ip1 = F(i+1);
        f_ip1pj = F(i+1+j);
        b0 += -5.0*f_im1pj*f_ip1 + 2.0*f_ipj*f_ip2 - 3.5*f_ip2*f_ip3pj + 2.0*f_ip2pj*f_i + 5.0*f_ip3pj*f_ip1 + 0.5*f_ip2*f_im1pj + 10.0*f_ipj*f_ip1 - 14.0*f_ipj*f_i - f_im1pj*f_im1 - 10.0*f_ip2pj*f_ip1 + 5.5*f_i*f_im1pj + 9.0*f_i*f_ip1pj + f_ip3pj*f_im1 + 2.0*f_ipj*f_im1 - 2.5*f_i*f_ip3pj - 2.0*f_ip2pj*f_im1 - 9.0*f_ip2*f_ip1pj + 10.0*f_ip2pj*f_ip2;
        b1 += -18.0*f_ip1pj*f_im1 + 9.0*f_ip2*f_ip3pj - 18.0*f_ipj*f_ip2 - 6.0*f_im1pj*f_ip1 + 42.0*f_ipj*f_ip1 - 24.0*f_ip3pj*f_ip1 + 3.0*f_ip2*f_im1pj - 66.0*f_ip2pj*f_i + 78.0*f_ip2pj*f_ip1 - 90.0*f_ip1*f_ip1pj - 30.0*f_ipj*f_i + 3.0*f_i*f_im1pj + 72.0*f_i*f_ip1pj - 6.0*f_ip3pj*f_im1 + 6.0*f_ipj*f_im1 + 21.0*f_i*f_ip3pj + 18.0*f_ip2pj*f_im1 + 36.0*f_ip2*f_ip1pj - 30.0*f_ip2pj*f_ip2;
        b2 += 27.0*f_ip1pj*f_im1 - 4.5*f_ip2*f_ip3pj + 18.0*f_ipj*f_ip2 + 13.5*f_im1pj*f_ip1 - 54.0*f_ipj*f_ip1 + 13.5*f_ip3pj*f_ip1 - 4.5*f_ip2*f_im1pj + 54.0*f_ip2pj*f_i + 54.0*f_ipj*f_i + 81.0*f_ip1*f_ip1pj - 54.0*f_ip2pj*f_ip1 - 13.5*f_i*f_im1pj - 81.0*f_i*f_ip1pj + 4.5*f_im1pj*f_im1 + 4.5*f_ip3pj*f_im1 - 18.0*f_ipj*f_im1 - 13.5*f_i*f_ip3pj - 18.0*f_ip2pj*f_im1 - 27.0*f_ip2*f_ip1pj + 18.0*f_ip2pj*f_ip2;
      }
      b0 += 0.5*f_n*f_m3mjpn - f_n*f_mjpn + 24.5*f_m1pn*f_m2mjpn + 5.5*f_m2pn*f_m3mjpn - 5.0*f_m1mjpn*f_m3pn - 5.0*f_m2mjpn*f_n + 5.5*f_m2mjpn*f_m3pn + 5.5*f_m1pn*f_mjpn - 5.0*f_m1pn*f_m3mjpn - 25.0*f_m2pn*f_m2mjpn - 25.0*f_m1pn*f_m1mjpn - f_m3pn*f_m3mjpn - 5.0*f_m2pn*f_mjpn + 0.5*f_m3pn*f_mjpn + 5.5*f_m1mjpn*f_n + 24.5*f_m2pn*f_m1mjpn;
      b1 += -3.0*f_n*f_m3mjpn - 3.0*f_m2pn*f_m3mjpn - 6.0*f_m1mjpn*f_m3pn + 6.0*f_m2mjpn*f_n + 6.0*f_m1pn*f_m3mjpn - 6.0*f_m2pn*f_mjpn + 3.0*f_m2mjpn*f_m3pn + 9.0*f_m2pn*f_m1mjpn - 9.0*f_m1pn*f_m2mjpn + 3.0*f_m1pn*f_mjpn + 3.0*f_m3pn*f_mjpn - 3.0*f_m1mjpn*f_n;
      b2 += -4.5*f_n*f_m3mjpn + 4.5*f_n*f_mjpn - 40.5*f_m1pn*f_m2mjpn - 13.5*f_m2pn*f_m3mjpn + 13.5*f_m1mjpn*f_m3pn + 13.5*f_m2mjpn*f_n - 13.5*f_m2mjpn*f_m3pn - 13.5*f_m1pn*f_mjpn + 13.5*f_m1pn*f_m3mjpn + 40.5*f_m2pn*f_m2mjpn + 40.5*f_m1pn*f_m1mjpn + 4.5*f_m3pn*f_m3mjpn + 13.5*f_m2pn*f_mjpn - 4.5*f_m3pn*f_mjpn - 13.5*f_m1mjpn*f_n - 40.5*f_m2pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
  *a6 = b6;
  *a7 = b7;
}
        
int cf_a33_find_extreme_diff5(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a33_compute_coeffs_diff5(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_quadratic_approximation_1_2(a1_0, a1_1, a1_2, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a33_find_zero_diff5(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a33_compute_coeffs_diff5(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_linear_approximation_1_2(a1_0, a1_1, a1_2, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a33_find_zero_diff5: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a33_compute_coeffs_diff6(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order=6) = sum(a_k*r^k, k=0..7) where y=j+r
     f1(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double b6 = 0.0;
  double b7 = 0.0;
  double f_im1, f_m3mjpn, f_ipj, f_mjpn, f_n, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip3pj, f_ip2, f_ip1, f_ip1pj, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_mjpn = F(-j+n);
      f_n = F(n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_im1 = F(i-1);
        f_ipj = F(i+j);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip3pj = F(i+3+j);
        f_ip2 = F(i+2);
        f_ip1 = F(i+1);
        f_ip1pj = F(i+1+j);
        b0 += -18.0*f_ip1pj*f_im1 + 9.0*f_ip2*f_ip3pj - 18.0*f_ipj*f_ip2 - 6.0*f_im1pj*f_ip1 + 42.0*f_ipj*f_ip1 - 24.0*f_ip3pj*f_ip1 + 3.0*f_ip2*f_im1pj - 66.0*f_ip2pj*f_i - 30.0*f_ipj*f_i - 90.0*f_ip1*f_ip1pj + 78.0*f_ip2pj*f_ip1 + 3.0*f_i*f_im1pj + 72.0*f_i*f_ip1pj - 6.0*f_ip3pj*f_im1 + 6.0*f_ipj*f_im1 + 21.0*f_i*f_ip3pj + 18.0*f_ip2pj*f_im1 + 36.0*f_ip2*f_ip1pj - 30.0*f_ip2pj*f_ip2;
        b1 += 54.0*f_ip1pj*f_im1 - 9.0*f_ip2*f_ip3pj + 36.0*f_ipj*f_ip2 + 27.0*f_im1pj*f_ip1 - 108.0*f_ipj*f_ip1 + 27.0*f_ip3pj*f_ip1 - 9.0*f_ip2*f_im1pj + 108.0*f_ip2pj*f_i - 108.0*f_ip2pj*f_ip1 + 162.0*f_ip1*f_ip1pj + 108.0*f_ipj*f_i - 27.0*f_i*f_im1pj - 162.0*f_i*f_ip1pj + 9.0*f_im1pj*f_im1 + 9.0*f_ip3pj*f_im1 - 36.0*f_ipj*f_im1 - 27.0*f_i*f_ip3pj - 36.0*f_ip2pj*f_im1 - 54.0*f_ip2*f_ip1pj + 36.0*f_ip2pj*f_ip2;
      }
      b0 += -3.0*f_n*f_m3mjpn + 6.0*f_m1pn*f_m3mjpn + 3.0*f_m2mjpn*f_m3pn + 6.0*f_m2mjpn*f_n - 6.0*f_m1mjpn*f_m3pn - 6.0*f_m2pn*f_mjpn - 3.0*f_m2pn*f_m3mjpn - 9.0*f_m1pn*f_m2mjpn + 9.0*f_m2pn*f_m1mjpn + 3.0*f_m1pn*f_mjpn + 3.0*f_m3pn*f_mjpn - 3.0*f_m1mjpn*f_n;
      b1 += -9.0*f_n*f_m3mjpn + 9.0*f_n*f_mjpn - 81.0*f_m1pn*f_m2mjpn - 27.0*f_m2pn*f_m3mjpn - 27.0*f_m2mjpn*f_m3pn + 27.0*f_m2mjpn*f_n + 27.0*f_m1mjpn*f_m3pn - 27.0*f_m1pn*f_mjpn + 27.0*f_m1pn*f_m3mjpn - 81.0*f_m2pn*f_m1mjpn + 81.0*f_m2pn*f_m2mjpn + 9.0*f_m3pn*f_m3mjpn + 27.0*f_m2pn*f_mjpn - 9.0*f_m3pn*f_mjpn - 27.0*f_m1mjpn*f_n + 81.0*f_m1pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
  *a6 = b6;
  *a7 = b7;
}
        
int cf_a33_find_extreme_diff6(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a2_2 = 0.0;
  double a2_3 = 0.0;
  double a2_4 = 0.0;
  double a2_5 = 0.0;
  double a2_6 = 0.0;
  double a2_7 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  double a3_2 = 0.0;
  double a3_3 = 0.0;
  double a3_4 = 0.0;
  double a3_5 = 0.0;
  double a3_6 = 0.0;
  double a3_7 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a1_2 = a2_2;
    a1_3 = a2_3;
    a1_4 = a2_4;
    a1_5 = a2_5;
    a1_6 = a2_6;
    a1_7 = a2_7;
    a2_0 = a3_0;
    a2_1 = a3_1;
    a2_2 = a3_2;
    a2_3 = a3_3;
    a2_4 = a3_4;
    a2_5 = a3_5;
    a2_6 = a3_6;
    a2_7 = a3_7;
    cf_a33_compute_coeffs_diff6(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3, &a3_4, &a3_5, &a3_6, &a3_7);
    count ++;
    if (count<3)
      continue;
    cf_quadratic_approximation_3_1(a1_0, a1_1, a2_0, a2_1, a3_0, a3_1, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a33_find_zero_diff6(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a33_compute_coeffs_diff6(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_linear_approximation_1_1(a1_0, a1_1, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a33_find_zero_diff6: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a33_compute_coeffs_diff7(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order=7) = sum(a_k*r^k, k=0..7) where y=j+r
     f1(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double b6 = 0.0;
  double b7 = 0.0;
  double f_im1, f_m3mjpn, f_ipj, f_m1mjpn, f_n, f_mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip3pj, f_ip2, f_ip1, f_ip1pj, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_m1mjpn = F(-1-j+n);
      f_n = F(n);
      f_mjpn = F(-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_im1 = F(i-1);
        f_ipj = F(i+j);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip3pj = F(i+3+j);
        f_ip2 = F(i+2);
        f_ip1 = F(i+1);
        f_ip1pj = F(i+1+j);
        b0 += 54.0*f_ip1pj*f_im1 - 9.0*f_ip2*f_ip3pj + 36.0*f_ipj*f_ip2 + 27.0*f_im1pj*f_ip1 - 108.0*f_ipj*f_ip1 + 27.0*f_ip3pj*f_ip1 - 9.0*f_ip2*f_im1pj + 108.0*f_ip2pj*f_i + 108.0*f_ipj*f_i + 162.0*f_ip1*f_ip1pj - 108.0*f_ip2pj*f_ip1 - 27.0*f_i*f_im1pj - 162.0*f_i*f_ip1pj + 9.0*f_im1pj*f_im1 + 9.0*f_ip3pj*f_im1 - 36.0*f_ipj*f_im1 - 27.0*f_i*f_ip3pj - 36.0*f_ip2pj*f_im1 - 54.0*f_ip2*f_ip1pj + 36.0*f_ip2pj*f_ip2;
      }
      b0 += -9.0*f_n*f_m3mjpn + 9.0*f_n*f_mjpn - 81.0*f_m1pn*f_m2mjpn - 27.0*f_m2pn*f_m3mjpn + 27.0*f_m1mjpn*f_m3pn + 27.0*f_m2mjpn*f_n - 27.0*f_m2mjpn*f_m3pn - 27.0*f_m1pn*f_mjpn + 27.0*f_m1pn*f_m3mjpn + 81.0*f_m1pn*f_m1mjpn - 81.0*f_m2pn*f_m1mjpn + 9.0*f_m3pn*f_m3mjpn + 27.0*f_m2pn*f_mjpn - 9.0*f_m3pn*f_mjpn - 27.0*f_m1mjpn*f_n + 81.0*f_m2pn*f_m2mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
  *a6 = b6;
  *a7 = b7;
}
        
int cf_a33_find_extreme_diff7(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a2_2 = 0.0;
  double a2_3 = 0.0;
  double a2_4 = 0.0;
  double a2_5 = 0.0;
  double a2_6 = 0.0;
  double a2_7 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  double a3_2 = 0.0;
  double a3_3 = 0.0;
  double a3_4 = 0.0;
  double a3_5 = 0.0;
  double a3_6 = 0.0;
  double a3_7 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a1_2 = a2_2;
    a1_3 = a2_3;
    a1_4 = a2_4;
    a1_5 = a2_5;
    a1_6 = a2_6;
    a1_7 = a2_7;
    a2_0 = a3_0;
    a2_1 = a3_1;
    a2_2 = a3_2;
    a2_3 = a3_3;
    a2_4 = a3_4;
    a2_5 = a3_5;
    a2_6 = a3_6;
    a2_7 = a3_7;
    cf_a33_compute_coeffs_diff7(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3, &a3_4, &a3_5, &a3_6, &a3_7);
    count ++;
    if (count<3)
      continue;
    cf_quadratic_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a33_find_zero_diff7(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a2_2 = 0.0;
  double a2_3 = 0.0;
  double a2_4 = 0.0;
  double a2_5 = 0.0;
  double a2_6 = 0.0;
  double a2_7 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  double a3_2 = 0.0;
  double a3_3 = 0.0;
  double a3_4 = 0.0;
  double a3_5 = 0.0;
  double a3_6 = 0.0;
  double a3_7 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a1_2 = a2_2;
    a1_3 = a2_3;
    a1_4 = a2_4;
    a1_5 = a2_5;
    a1_6 = a2_6;
    a1_7 = a2_7;
    a2_0 = a3_0;
    a2_1 = a3_1;
    a2_2 = a3_2;
    a2_3 = a3_3;
    a2_4 = a3_4;
    a2_5 = a3_5;
    a2_6 = a3_6;
    a2_7 = a3_7;
    cf_a33_compute_coeffs_diff7(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3, &a3_4, &a3_5, &a3_6, &a3_7);
    count ++;
    if (count<3)
      continue;
    cf_linear_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1);
    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a33_find_zero_diff7: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
void cf_a33_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order) = sum(a_k*r^k, k=0..7) where y=j+r
     f1(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  switch (order)
  {
    case 0: cf_a33_compute_coeffs_diff0(j, fm, n, m, a0, a1, a2, a3, a4, a5, a6, a7); break;
    case 1: cf_a33_compute_coeffs_diff1(j, fm, n, m, a0, a1, a2, a3, a4, a5, a6, a7); break;
    case 2: cf_a33_compute_coeffs_diff2(j, fm, n, m, a0, a1, a2, a3, a4, a5, a6, a7); break;
    case 3: cf_a33_compute_coeffs_diff3(j, fm, n, m, a0, a1, a2, a3, a4, a5, a6, a7); break;
    case 4: cf_a33_compute_coeffs_diff4(j, fm, n, m, a0, a1, a2, a3, a4, a5, a6, a7); break;
    case 5: cf_a33_compute_coeffs_diff5(j, fm, n, m, a0, a1, a2, a3, a4, a5, a6, a7); break;
    case 6: cf_a33_compute_coeffs_diff6(j, fm, n, m, a0, a1, a2, a3, a4, a5, a6, a7); break;
    case 7: cf_a33_compute_coeffs_diff7(j, fm, n, m, a0, a1, a2, a3, a4, a5, a6, a7); break;
    default:
      *a0 = 0.0;
      *a1 = 0.0;
      *a2 = 0.0;
      *a3 = 0.0;
      *a4 = 0.0;
      *a5 = 0.0;
      *a6 = 0.0;
      *a7 = 0.0;
  }
}
        
int cf_a33_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  switch (order)
  {
    case 0: return cf_a33_find_extreme_diff0(j0, j1, fm, n, m, result);
    case 1: return cf_a33_find_extreme_diff1(j0, j1, fm, n, m, result);
    case 2: return cf_a33_find_extreme_diff2(j0, j1, fm, n, m, result);
    case 3: return cf_a33_find_extreme_diff3(j0, j1, fm, n, m, result);
    case 4: return cf_a33_find_extreme_diff4(j0, j1, fm, n, m, result);
    case 5: return cf_a33_find_extreme_diff5(j0, j1, fm, n, m, result);
    case 6: return cf_a33_find_extreme_diff6(j0, j1, fm, n, m, result);
    case 7: return cf_a33_find_extreme_diff7(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
int cf_a33_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  switch (order)
  {
    case 0: return cf_a33_find_zero_diff0(j0, j1, fm, n, m, result);
    case 1: return cf_a33_find_zero_diff1(j0, j1, fm, n, m, result);
    case 2: return cf_a33_find_zero_diff2(j0, j1, fm, n, m, result);
    case 3: return cf_a33_find_zero_diff3(j0, j1, fm, n, m, result);
    case 4: return cf_a33_find_zero_diff4(j0, j1, fm, n, m, result);
    case 5: return cf_a33_find_zero_diff5(j0, j1, fm, n, m, result);
    case 6: return cf_a33_find_zero_diff6(j0, j1, fm, n, m, result);
    case 7: return cf_a33_find_zero_diff7(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
double cf_a33_evaluate(double y, double *fm, int n, int m, int order)
{
  double a0 = 0.0;
  double a1 = 0.0;
  double a2 = 0.0;
  double a3 = 0.0;
  double a4 = 0.0;
  double a5 = 0.0;
  double a6 = 0.0;
  double a7 = 0.0;
  /*
  int j = floor((y<0?-y:y));
  double r = (y<0?-y:y) - j;
  */
  int j = floor (y);
  double r = y - j;
  cf_a33_compute_coeffs(j, fm, n, m, order, &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);
  return a0+(a1+(a2+(a3+(a4+(a5+(a6+(a7)*r)*r)*r)*r)*r)*r)*r;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double cf_a33_f1_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  switch (order)
  {
    case 0: return F(i) + (0.5*(F(i+1)) - 0.5*(F(i-1)) + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)) + (0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s)*s)*s;
    case 1: return 0.5*(F(i+1)) - 0.5*(F(i-1)) + (-(F(i+2)) - 5.0*(F(i)) + 4.0*(F(i+1)) + 2.0*(F(i-1)) + (1.5*(F(i+2)) + 4.5*(F(i)) - 4.5*(F(i+1)) - 1.5*(F(i-1)))*s)*s;
    case 2: return -(F(i+2)) - 5.0*(F(i)) + 4.0*(F(i+1)) + 2.0*(F(i-1)) + (3.0*(F(i+2)) + 9.0*(F(i)) - 9.0*(F(i+1)) - 3.0*(F(i-1)))*s;
  }
  return 0.0;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double cf_a33_f2_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  switch (order)
  {
    case 0: return F(i) + (0.5*(F(i+1)) - 0.5*(F(i-1)) + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)) + (0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s)*s)*s;
    case 1: return 0.5*(F(i+1)) - 0.5*(F(i-1)) + (-(F(i+2)) - 5.0*(F(i)) + 4.0*(F(i+1)) + 2.0*(F(i-1)) + (1.5*(F(i+2)) + 4.5*(F(i)) - 4.5*(F(i+1)) - 1.5*(F(i-1)))*s)*s;
    case 2: return -(F(i+2)) - 5.0*(F(i)) + 4.0*(F(i+1)) + 2.0*(F(i-1)) + (3.0*(F(i+2)) + 9.0*(F(i)) - 9.0*(F(i+1)) - 3.0*(F(i-1)))*s;
  }
  return 0.0;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a22_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5)
{
  /* int(f1(x)*f2(x+y), x=0..L-y) = sum(a_k*r^k, k=0..5) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double f_im1, f_m3mjpn, f_ipj, f_ip1pj, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_im1 = F(i-1);
        f_ipj = F(i+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip1 = F(i+1);
        b0 += -0.03333333333333333*f_im1pj*f_ip1 - 0.03333333333333333*f_ip1pj*f_im1 + 0.1916666666666667*f_ipj*f_ip1 + 0.5333333333333333*f_ipj*f_i + 0.2583333333333333*f_ip1*f_ip1pj - 0.05833333333333333*f_i*f_im1pj + 0.1916666666666667*f_i*f_ip1pj + 0.008333333333333333*f_im1pj*f_im1 - 0.05833333333333333*f_ipj*f_im1;
        b1 += 0.08333333333333333*f_im1pj*f_ip1 - 0.08333333333333333*f_ip1pj*f_im1 - 0.5833333333333333*f_ipj*f_ip1 - 0.5*f_ipj*f_i + 0.5*f_ip1*f_ip1pj - 0.08333333333333333*f_i*f_im1pj + 0.5833333333333333*f_i*f_ip1pj + 0.08333333333333333*f_ipj*f_im1;
        b2 += -0.04166666666666667*f_ip1*f_im1pj - 0.04166666666666667*f_im1*f_ip1pj + 0.3333333333333333*f_ipj*f_ip1 - 0.6666666666666667*f_ipj*f_i - 0.04166666666666667*f_im1pj*f_im1 + 0.25*f_ip2pj*f_ip1 + 0.3333333333333333*f_i*f_im1pj + 0.3333333333333333*f_i*f_ip1pj - 0.5416666666666667*f_ip1*f_ip1pj + 0.08333333333333333*f_ipj*f_im1;
        b3 += 0.125*f_im1*f_ip1pj - 0.04166666666666667*f_im1pj*f_ip1 + 0.125*f_ipj*f_ip1 + 0.1666666666666667*f_ip2pj*f_i + 0.04166666666666667*f_ip2pj*f_ip1 - 0.125*f_ip1*f_ip1pj + 0.5*f_ipj*f_i - 0.1666666666666667*f_i*f_im1pj - 0.5*f_i*f_ip1pj + 0.04166666666666667*f_im1pj*f_im1 - 0.125*f_ipj*f_im1 - 0.04166666666666667*f_ip2pj*f_im1;
        b4 += 0.04166666666666667*f_ip1*f_im1pj + 0.04166666666666667*f_ip2pj*f_i - 0.125*f_ipj*f_ip1 - 0.04166666666666667*f_ip2pj*f_ip1 + 0.125*f_ip1*f_ip1pj + 0.125*f_ipj*f_i - 0.04166666666666667*f_i*f_im1pj - 0.125*f_i*f_ip1pj;
        b5 += -0.025*f_im1*f_ip1pj - 0.008333333333333333*f_ip1*f_im1pj + 0.025*f_ipj*f_ip1 - 0.01666666666666667*f_ip2pj*f_i + 0.008333333333333333*f_ip2pj*f_ip1 - 0.025*f_ip1*f_ip1pj - 0.05*f_ipj*f_i + 0.01666666666666667*f_i*f_im1pj + 0.05*f_i*f_ip1pj - 0.008333333333333333*f_im1pj*f_im1 + 0.025*f_ipj*f_im1 + 0.008333333333333333*f_ip2pj*f_im1;
      }
      b0 += -0.05833333333333333*f_m2mjpn*f_m3pn - 0.03333333333333333*f_m1mjpn*f_m3pn + 0.008333333333333333*f_m3mjpn*f_m3pn - 0.05833333333333333*f_m2pn*f_m3mjpn - 0.03333333333333333*f_m1pn*f_m3mjpn + 0.1916666666666667*f_m2mjpn*f_m1pn + 0.1916666666666667*f_m1mjpn*f_m2pn + 0.5333333333333333*f_m2mjpn*f_m2pn + 0.2583333333333333*f_m1mjpn*f_m1pn;
      b1 += -0.08333333333333333*f_m2mjpn*f_m3pn + 0.08333333333333333*f_m1mjpn*f_m3pn + 0.08333333333333333*f_m2pn*f_m3mjpn - 0.08333333333333333*f_m1pn*f_m3mjpn - 0.5*f_m2mjpn*f_m2pn - 0.5833333333333333*f_m1mjpn*f_m2pn - 0.5*f_m1mjpn*f_m1pn + 0.5833333333333333*f_m2mjpn*f_m1pn;
      b2 += 0.3333333333333333*f_m2mjpn*f_m3pn - 0.04166666666666667*f_m1mjpn*f_m3pn - 0.04166666666666667*f_m3mjpn*f_m3pn + 0.08333333333333333*f_m2pn*f_m3mjpn + 0.2083333333333333*f_m1pn*f_m3mjpn + 0.5833333333333333*f_m2pn*f_m1mjpn + 0.2083333333333333*f_m1pn*f_m1mjpn - 0.6666666666666667*f_m2pn*f_m2mjpn - 0.6666666666666667*f_m2mjpn*f_m1pn;
      b3 += -0.1666666666666667*f_m2mjpn*f_m3pn - 0.04166666666666667*f_m1mjpn*f_m3pn + 0.04166666666666667*f_m3mjpn*f_m3pn - 0.1666666666666667*f_m2pn*f_m3mjpn - 0.04166666666666667*f_m1pn*f_m3mjpn + 0.6666666666666667*f_m2mjpn*f_m2pn - 0.1666666666666667*f_m1mjpn*f_m2pn + 0.04166666666666667*f_m1mjpn*f_m1pn - 0.1666666666666667*f_m2mjpn*f_m1pn;
      b4 += 0.04166666666666667*f_m2pn*f_m3mjpn - 0.04166666666666667*f_m2mjpn*f_m3pn + 0.04166666666666667*f_m1mjpn*f_m3pn - 0.04166666666666667*f_m1pn*f_m3mjpn + 0.04166666666666667*f_m1pn*f_m2mjpn - 0.04166666666666667*f_m1mjpn*f_m2pn;
      b5 += -0.008333333333333333*f_m1mjpn*f_m3pn + 0.01666666666666667*f_m2mjpn*f_m3pn - 0.008333333333333333*f_m3pn*f_m3mjpn - 0.008333333333333333*f_m1pn*f_m3mjpn + 0.01666666666666667*f_m2pn*f_m3mjpn - 0.03333333333333333*f_m2pn*f_m2mjpn + 0.01666666666666667*f_m2pn*f_m1mjpn + 0.01666666666666667*f_m1pn*f_m2mjpn - 0.008333333333333333*f_m1pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
}
        
int cf_a22_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a22_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5);
    cf_quadratic_approximation_1_5(a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a22_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a22_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5);
    cf_linear_approximation_1_5(a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a22_find_zero_diff0: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a22_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order=1) = sum(a_k*r^k, k=0..5) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double f_im1, f_m3mjpn, f_ipj, f_ip1pj, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_im1 = F(i-1);
        f_ipj = F(i+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip1 = F(i+1);
        b0 += 0.08333333333333333*f_im1pj*f_ip1 - 0.08333333333333333*f_ip1pj*f_im1 - 0.5833333333333333*f_ipj*f_ip1 - 0.5*f_ipj*f_i + 0.5*f_ip1*f_ip1pj - 0.08333333333333333*f_i*f_im1pj + 0.5833333333333333*f_i*f_ip1pj + 0.08333333333333333*f_ipj*f_im1;
        b1 += -0.08333333333333333*f_ip1*f_im1pj - 0.08333333333333333*f_im1*f_ip1pj + 0.6666666666666667*f_ipj*f_ip1 - 1.333333333333333*f_ipj*f_i - 0.08333333333333333*f_im1pj*f_im1 + 0.5*f_ip2pj*f_ip1 + 0.6666666666666667*f_i*f_im1pj + 0.6666666666666667*f_i*f_ip1pj - 1.083333333333333*f_ip1*f_ip1pj + 0.1666666666666667*f_ipj*f_im1;
        b2 += 0.375*f_im1*f_ip1pj - 0.125*f_im1pj*f_ip1 + 0.375*f_ipj*f_ip1 + 0.5*f_ip2pj*f_i + 0.125*f_ip2pj*f_ip1 - 0.375*f_ip1*f_ip1pj + 1.5*f_ipj*f_i - 0.5*f_i*f_im1pj - 1.5*f_i*f_ip1pj + 0.125*f_im1pj*f_im1 - 0.375*f_ipj*f_im1 - 0.125*f_ip2pj*f_im1;
        b3 += 0.1666666666666667*f_ip1*f_im1pj + 0.1666666666666667*f_ip2pj*f_i - 0.5*f_ipj*f_ip1 + 0.5*f_ipj*f_i + 0.5*f_ip1*f_ip1pj - 0.1666666666666667*f_ip2pj*f_ip1 - 0.1666666666666667*f_i*f_im1pj - 0.5*f_i*f_ip1pj;
        b4 += -0.125*f_im1*f_ip1pj - 0.04166666666666667*f_ip1*f_im1pj + 0.125*f_ipj*f_ip1 - 0.08333333333333333*f_ip2pj*f_i + 0.04166666666666667*f_ip2pj*f_ip1 - 0.125*f_ip1*f_ip1pj - 0.25*f_ipj*f_i + 0.08333333333333333*f_i*f_im1pj + 0.25*f_i*f_ip1pj - 0.04166666666666667*f_im1pj*f_im1 + 0.125*f_ipj*f_im1 + 0.04166666666666667*f_ip2pj*f_im1;
      }
      b0 += -0.08333333333333333*f_m1pn*f_m3mjpn - 0.08333333333333333*f_m2mjpn*f_m3pn + 0.08333333333333333*f_m1mjpn*f_m3pn + 0.08333333333333333*f_m2pn*f_m3mjpn - 0.5*f_m2mjpn*f_m2pn + 0.5833333333333333*f_m2mjpn*f_m1pn - 0.5833333333333333*f_m1mjpn*f_m2pn - 0.5*f_m1mjpn*f_m1pn;
      b1 += -1.333333333333333*f_m2mjpn*f_m1pn + 0.1666666666666667*f_m2pn*f_m3mjpn + 0.6666666666666667*f_m2mjpn*f_m3pn + 0.4166666666666667*f_m1pn*f_m3mjpn - 0.08333333333333333*f_m1mjpn*f_m3pn + 1.166666666666667*f_m2pn*f_m1mjpn - 1.333333333333333*f_m2pn*f_m2mjpn - 0.08333333333333333*f_m3mjpn*f_m3pn + 0.4166666666666667*f_m1pn*f_m1mjpn;
      b2 += -0.5*f_m2mjpn*f_m1pn - 0.5*f_m2pn*f_m3mjpn - 0.5*f_m2mjpn*f_m3pn - 0.125*f_m1pn*f_m3mjpn - 0.125*f_m1mjpn*f_m3pn + 2.0*f_m2mjpn*f_m2pn + 0.125*f_m1mjpn*f_m1pn + 0.125*f_m3mjpn*f_m3pn - 0.5*f_m1mjpn*f_m2pn;
      b3 += -0.1666666666666667*f_m1pn*f_m3mjpn - 0.1666666666666667*f_m2mjpn*f_m3pn + 0.1666666666666667*f_m2pn*f_m3mjpn + 0.1666666666666667*f_m1mjpn*f_m3pn - 0.1666666666666667*f_m1mjpn*f_m2pn + 0.1666666666666667*f_m1pn*f_m2mjpn;
      b4 += -0.04166666666666667*f_m1pn*f_m1mjpn - 0.04166666666666667*f_m1pn*f_m3mjpn - 0.04166666666666667*f_m1mjpn*f_m3pn + 0.08333333333333333*f_m2pn*f_m3mjpn + 0.08333333333333333*f_m2mjpn*f_m3pn - 0.1666666666666667*f_m2pn*f_m2mjpn + 0.08333333333333333*f_m1pn*f_m2mjpn - 0.04166666666666667*f_m3pn*f_m3mjpn + 0.08333333333333333*f_m2pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
}
        
int cf_a22_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a22_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5);
    cf_quadratic_approximation_1_4(a1_0, a1_1, a1_2, a1_3, a1_4, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a22_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a22_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5);
    cf_linear_approximation_1_4(a1_0, a1_1, a1_2, a1_3, a1_4, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a22_find_zero_diff1: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a22_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order=2) = sum(a_k*r^k, k=0..5) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double f_im1, f_m3mjpn, f_ipj, f_ip1pj, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_im1 = F(i-1);
        f_ipj = F(i+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip1 = F(i+1);
        b0 += -0.08333333333333333*f_ip1*f_im1pj - 0.08333333333333333*f_im1*f_ip1pj + 0.6666666666666667*f_ipj*f_ip1 - 1.333333333333333*f_ipj*f_i - 0.08333333333333333*f_im1pj*f_im1 + 0.5*f_ip2pj*f_ip1 + 0.6666666666666667*f_i*f_im1pj + 0.6666666666666667*f_i*f_ip1pj - 1.083333333333333*f_ip1*f_ip1pj + 0.1666666666666667*f_ipj*f_im1;
        b1 += 0.75*f_im1*f_ip1pj - 0.25*f_im1pj*f_ip1 + 0.75*f_ipj*f_ip1 + f_ip2pj*f_i + 0.25*f_ip2pj*f_ip1 - 0.75*f_ip1*f_ip1pj + 3.0*f_ipj*f_i - f_i*f_im1pj - 3.0*f_i*f_ip1pj + 0.25*f_im1pj*f_im1 - 0.75*f_ipj*f_im1 - 0.25*f_ip2pj*f_im1;
        b2 += 0.5*f_ip1*f_im1pj + 0.5*f_ip2pj*f_i - 1.5*f_ipj*f_ip1 + 1.5*f_ipj*f_i + 1.5*f_ip1*f_ip1pj - 0.5*f_ip2pj*f_ip1 - 0.5*f_i*f_im1pj - 1.5*f_i*f_ip1pj;
        b3 += -0.5*f_im1*f_ip1pj - 0.1666666666666667*f_ip1*f_im1pj + 0.5*f_ipj*f_ip1 - 0.3333333333333333*f_ip2pj*f_i + 0.1666666666666667*f_ip2pj*f_ip1 - 0.5*f_ip1*f_ip1pj - f_ipj*f_i + 0.3333333333333333*f_i*f_im1pj + f_i*f_ip1pj - 0.1666666666666667*f_im1pj*f_im1 + 0.5*f_ipj*f_im1 + 0.1666666666666667*f_ip2pj*f_im1;
      }
      b0 += 0.4166666666666667*f_m1pn*f_m3mjpn - 0.08333333333333333*f_m1mjpn*f_m3pn - 0.08333333333333333*f_m3mjpn*f_m3pn + 0.1666666666666667*f_m2pn*f_m3mjpn + 0.6666666666666667*f_m2mjpn*f_m3pn + 1.166666666666667*f_m2pn*f_m1mjpn + 0.4166666666666667*f_m1pn*f_m1mjpn - 1.333333333333333*f_m2mjpn*f_m1pn - 1.333333333333333*f_m2pn*f_m2mjpn;
      b1 += -0.25*f_m1pn*f_m3mjpn - 0.25*f_m1mjpn*f_m3pn + 0.25*f_m3mjpn*f_m3pn - f_m2pn*f_m3mjpn - f_m2mjpn*f_m3pn + 4.0*f_m2mjpn*f_m2pn - f_m1mjpn*f_m2pn - f_m2mjpn*f_m1pn + 0.25*f_m1mjpn*f_m1pn;
      b2 += -0.5*f_m2mjpn*f_m3pn + 0.5*f_m1mjpn*f_m3pn + 0.5*f_m2pn*f_m3mjpn - 0.5*f_m1pn*f_m3mjpn + 0.5*f_m1pn*f_m2mjpn - 0.5*f_m1mjpn*f_m2pn;
      b3 += 0.3333333333333333*f_m2pn*f_m3mjpn + 0.3333333333333333*f_m2mjpn*f_m3pn - 0.1666666666666667*f_m3pn*f_m3mjpn - 0.1666666666666667*f_m1pn*f_m3mjpn - 0.1666666666666667*f_m1mjpn*f_m3pn - 0.6666666666666667*f_m2pn*f_m2mjpn + 0.3333333333333333*f_m2pn*f_m1mjpn - 0.1666666666666667*f_m1pn*f_m1mjpn + 0.3333333333333333*f_m1pn*f_m2mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
}
        
int cf_a22_find_extreme_diff2(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a22_compute_coeffs_diff2(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5);
    cf_quadratic_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a22_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a22_compute_coeffs_diff2(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5);
    cf_linear_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a22_find_zero_diff2: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a22_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order=3) = sum(a_k*r^k, k=0..5) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double f_im1, f_m3mjpn, f_ipj, f_ip1pj, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_im1 = F(i-1);
        f_ipj = F(i+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip1 = F(i+1);
        b0 += 0.75*f_im1*f_ip1pj - 0.25*f_im1pj*f_ip1 + 0.75*f_ipj*f_ip1 + f_ip2pj*f_i + 0.25*f_ip2pj*f_ip1 - 0.75*f_ip1*f_ip1pj + 3.0*f_ipj*f_i - f_i*f_im1pj - 3.0*f_i*f_ip1pj + 0.25*f_im1pj*f_im1 - 0.75*f_ipj*f_im1 - 0.25*f_ip2pj*f_im1;
        b1 += f_ip1*f_im1pj + f_ip2pj*f_i - 3.0*f_ipj*f_ip1 + 3.0*f_ipj*f_i + 3.0*f_ip1*f_ip1pj - f_ip2pj*f_ip1 - f_i*f_im1pj - 3.0*f_i*f_ip1pj;
        b2 += -1.5*f_im1*f_ip1pj - 0.5*f_ip1*f_im1pj + 1.5*f_ipj*f_ip1 - f_ip2pj*f_i + 0.5*f_ip2pj*f_ip1 - 1.5*f_ip1*f_ip1pj - 3.0*f_ipj*f_i + f_i*f_im1pj + 3.0*f_i*f_ip1pj - 0.5*f_im1pj*f_im1 + 1.5*f_ipj*f_im1 + 0.5*f_ip2pj*f_im1;
      }
      b0 += 0.25*f_m1mjpn*f_m1pn - f_m2pn*f_m3mjpn - f_m2mjpn*f_m3pn - 0.25*f_m1pn*f_m3mjpn - 0.25*f_m1mjpn*f_m3pn + 4.0*f_m2mjpn*f_m2pn - f_m2mjpn*f_m1pn + 0.25*f_m3mjpn*f_m3pn - f_m1mjpn*f_m2pn;
      b1 += -f_m1pn*f_m3mjpn - f_m2mjpn*f_m3pn + f_m2pn*f_m3mjpn + f_m1mjpn*f_m3pn - f_m1mjpn*f_m2pn + f_m1pn*f_m2mjpn;
      b2 += f_m1pn*f_m2mjpn - 0.5*f_m1pn*f_m3mjpn - 0.5*f_m1mjpn*f_m3pn + f_m2pn*f_m3mjpn + f_m2mjpn*f_m3pn - 2.0*f_m2pn*f_m2mjpn - 0.5*f_m1pn*f_m1mjpn - 0.5*f_m3pn*f_m3mjpn + f_m2pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
}
        
int cf_a22_find_extreme_diff3(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a22_compute_coeffs_diff3(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5);
    cf_quadratic_approximation_1_2(a1_0, a1_1, a1_2, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a22_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a22_compute_coeffs_diff3(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5);
    cf_linear_approximation_1_2(a1_0, a1_1, a1_2, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a22_find_zero_diff3: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a22_compute_coeffs_diff4(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order=4) = sum(a_k*r^k, k=0..5) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double f_ip2pj, f_m3mjpn, f_i, f_im1, f_ip1pj, f_m1mjpn, f_m3pn, f_ipj, f_m2mjpn, f_m2pn, f_im1pj, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ip2pj = F(i+2+j);
        f_i = F(i);
        f_im1 = F(i-1);
        f_ip1pj = F(i+1+j);
        f_ipj = F(i+j);
        f_im1pj = F(i-1+j);
        f_ip1 = F(i+1);
        b0 += f_ip1*f_im1pj + f_ip2pj*f_i - 3.0*f_ipj*f_ip1 + 3.0*f_ipj*f_i + 3.0*f_ip1*f_ip1pj - f_ip2pj*f_ip1 - f_i*f_im1pj - 3.0*f_i*f_ip1pj;
        b1 += -3.0*f_im1*f_ip1pj - f_ip1*f_im1pj + 3.0*f_ipj*f_ip1 - 2.0*f_ip2pj*f_i + f_ip2pj*f_ip1 - 3.0*f_ip1*f_ip1pj - 6.0*f_ipj*f_i + 2.0*f_i*f_im1pj + 6.0*f_i*f_ip1pj - f_im1pj*f_im1 + 3.0*f_ipj*f_im1 + f_ip2pj*f_im1;
      }
      b0 += -f_m2mjpn*f_m3pn + f_m1mjpn*f_m3pn + f_m2pn*f_m3mjpn - f_m1pn*f_m3mjpn + f_m1pn*f_m2mjpn - f_m1mjpn*f_m2pn;
      b1 += 2.0*f_m2pn*f_m3mjpn + 2.0*f_m2mjpn*f_m3pn - f_m3pn*f_m3mjpn - f_m1pn*f_m3mjpn - f_m1mjpn*f_m3pn - 4.0*f_m2pn*f_m2mjpn + 2.0*f_m2pn*f_m1mjpn + 2.0*f_m1pn*f_m2mjpn - f_m1pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
}
        
int cf_a22_find_extreme_diff4(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a2_2 = 0.0;
  double a2_3 = 0.0;
  double a2_4 = 0.0;
  double a2_5 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  double a3_2 = 0.0;
  double a3_3 = 0.0;
  double a3_4 = 0.0;
  double a3_5 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a1_2 = a2_2;
    a1_3 = a2_3;
    a1_4 = a2_4;
    a1_5 = a2_5;
    a2_0 = a3_0;
    a2_1 = a3_1;
    a2_2 = a3_2;
    a2_3 = a3_3;
    a2_4 = a3_4;
    a2_5 = a3_5;
    cf_a22_compute_coeffs_diff4(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3, &a3_4, &a3_5);
    count ++;
    if (count<3)
      continue;
    cf_quadratic_approximation_3_1(a1_0, a1_1, a2_0, a2_1, a3_0, a3_1, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a22_find_zero_diff4(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a22_compute_coeffs_diff4(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5);
    cf_linear_approximation_1_1(a1_0, a1_1, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a22_find_zero_diff4: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a22_compute_coeffs_diff5(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order=5) = sum(a_k*r^k, k=0..5) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double f_im1, f_m3mjpn, f_ipj, f_ip1pj, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_im1 = F(i-1);
        f_ipj = F(i+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip1 = F(i+1);
        b0 += -3.0*f_im1*f_ip1pj - f_ip1*f_im1pj + 3.0*f_ipj*f_ip1 - 2.0*f_ip2pj*f_i + f_ip2pj*f_ip1 - 3.0*f_ip1*f_ip1pj - 6.0*f_ipj*f_i + 2.0*f_i*f_im1pj + 6.0*f_i*f_ip1pj - f_im1pj*f_im1 + 3.0*f_ipj*f_im1 + f_ip2pj*f_im1;
      }
      b0 += -f_m1pn*f_m1mjpn - f_m1pn*f_m3mjpn - f_m1mjpn*f_m3pn + 2.0*f_m2pn*f_m3mjpn + 2.0*f_m2mjpn*f_m3pn - 4.0*f_m2pn*f_m2mjpn + 2.0*f_m1pn*f_m2mjpn - f_m3pn*f_m3mjpn + 2.0*f_m2pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
}
        
int cf_a22_find_extreme_diff5(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a2_2 = 0.0;
  double a2_3 = 0.0;
  double a2_4 = 0.0;
  double a2_5 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  double a3_2 = 0.0;
  double a3_3 = 0.0;
  double a3_4 = 0.0;
  double a3_5 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a1_2 = a2_2;
    a1_3 = a2_3;
    a1_4 = a2_4;
    a1_5 = a2_5;
    a2_0 = a3_0;
    a2_1 = a3_1;
    a2_2 = a3_2;
    a2_3 = a3_3;
    a2_4 = a3_4;
    a2_5 = a3_5;
    cf_a22_compute_coeffs_diff5(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3, &a3_4, &a3_5);
    count ++;
    if (count<3)
      continue;
    cf_quadratic_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a22_find_zero_diff5(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a2_2 = 0.0;
  double a2_3 = 0.0;
  double a2_4 = 0.0;
  double a2_5 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  double a3_2 = 0.0;
  double a3_3 = 0.0;
  double a3_4 = 0.0;
  double a3_5 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a1_2 = a2_2;
    a1_3 = a2_3;
    a1_4 = a2_4;
    a1_5 = a2_5;
    a2_0 = a3_0;
    a2_1 = a3_1;
    a2_2 = a3_2;
    a2_3 = a3_3;
    a2_4 = a3_4;
    a2_5 = a3_5;
    cf_a22_compute_coeffs_diff5(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3, &a3_4, &a3_5);
    count ++;
    if (count<3)
      continue;
    cf_linear_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1);
    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a22_find_zero_diff5: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
void cf_a22_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order) = sum(a_k*r^k, k=0..5) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  switch (order)
  {
    case 0: cf_a22_compute_coeffs_diff0(j, fm, n, m, a0, a1, a2, a3, a4, a5); break;
    case 1: cf_a22_compute_coeffs_diff1(j, fm, n, m, a0, a1, a2, a3, a4, a5); break;
    case 2: cf_a22_compute_coeffs_diff2(j, fm, n, m, a0, a1, a2, a3, a4, a5); break;
    case 3: cf_a22_compute_coeffs_diff3(j, fm, n, m, a0, a1, a2, a3, a4, a5); break;
    case 4: cf_a22_compute_coeffs_diff4(j, fm, n, m, a0, a1, a2, a3, a4, a5); break;
    case 5: cf_a22_compute_coeffs_diff5(j, fm, n, m, a0, a1, a2, a3, a4, a5); break;
    default:
      *a0 = 0.0;
      *a1 = 0.0;
      *a2 = 0.0;
      *a3 = 0.0;
      *a4 = 0.0;
      *a5 = 0.0;
  }
}
        
int cf_a22_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  switch (order)
  {
    case 0: return cf_a22_find_extreme_diff0(j0, j1, fm, n, m, result);
    case 1: return cf_a22_find_extreme_diff1(j0, j1, fm, n, m, result);
    case 2: return cf_a22_find_extreme_diff2(j0, j1, fm, n, m, result);
    case 3: return cf_a22_find_extreme_diff3(j0, j1, fm, n, m, result);
    case 4: return cf_a22_find_extreme_diff4(j0, j1, fm, n, m, result);
    case 5: return cf_a22_find_extreme_diff5(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
int cf_a22_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  switch (order)
  {
    case 0: return cf_a22_find_zero_diff0(j0, j1, fm, n, m, result);
    case 1: return cf_a22_find_zero_diff1(j0, j1, fm, n, m, result);
    case 2: return cf_a22_find_zero_diff2(j0, j1, fm, n, m, result);
    case 3: return cf_a22_find_zero_diff3(j0, j1, fm, n, m, result);
    case 4: return cf_a22_find_zero_diff4(j0, j1, fm, n, m, result);
    case 5: return cf_a22_find_zero_diff5(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
double cf_a22_evaluate(double y, double *fm, int n, int m, int order)
{
  double a0 = 0.0;
  double a1 = 0.0;
  double a2 = 0.0;
  double a3 = 0.0;
  double a4 = 0.0;
  double a5 = 0.0;
  /*
  int j = floor((y<0?-y:y));
  double r = (y<0?-y:y) - j;
  */
  int j = floor (y);
  double r = y - j;
  cf_a22_compute_coeffs(j, fm, n, m, order, &a0, &a1, &a2, &a3, &a4, &a5);
  return a0+(a1+(a2+(a3+(a4+(a5)*r)*r)*r)*r)*r;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double cf_a22_f1_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  switch (order)
  {
    case 0: return F(i) + (0.5*(F(i+1)) - 0.5*(F(i-1)) + (-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s)*s;
    case 1: return 0.5*(F(i+1)) - 0.5*(F(i-1)) + (-2.0*(F(i)) + (F(i+1)) + (F(i-1)))*s;
    case 2: return -2.0*(F(i)) + (F(i+1)) + (F(i-1));
  }
  return 0.0;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double cf_a22_f2_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  switch (order)
  {
    case 0: return F(i) + (0.5*(F(i+1)) - 0.5*(F(i-1)) + (-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s)*s;
    case 1: return 0.5*(F(i+1)) - 0.5*(F(i-1)) + (-2.0*(F(i)) + (F(i+1)) + (F(i-1)))*s;
    case 2: return -2.0*(F(i)) + (F(i+1)) + (F(i-1));
  }
  return 0.0;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e33_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7)
{
  /* int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y) = sum(a_k*r^k, k=0..7) where y=j+r
     f1(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double b6 = 0.0;
  double b7 = 0.0;
  double f_ipj, f_m3mjpn, f_im1, f_mjpn, f_n, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip3pj, f_ip2, f_ip1, f_ip1pj, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_mjpn = F(-j+n);
      f_n = F(n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_im1 = F(i-1);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip3pj = F(i+3+j);
        f_ip2 = F(i+2);
        f_ip1 = F(i+1);
        f_ip1pj = F(i+1+j);
        b0 += 0.3654761904761905*f_ipj*f_ip1pj + 0.03571428571428571*f_ip1pj*f_im1 - 0.03571428571428571*f_ip1*f_im1 + 0.03571428571428571*f_ip2pj*f_i - 0.003571428571428571*f_ip2*f_im1pj - 0.03571428571428571*f_ip2*f_i + 0.003571428571428571*f_ip2pj*f_im1pj + 0.4047619047619048*(f_ipj*f_ipj) - 0.004761904761904762*f_ip2pj*f_ip2 - 0.05595238095238095*f_ipj*f_im1pj + 0.03571428571428571*f_ip1*f_im1pj - 0.05595238095238095*f_ip2*f_ip1 + 0.03571428571428571*f_ipj*f_ip2 + 0.003571428571428571*f_ip2*f_im1 + 0.002380952380952381*(f_im1*f_im1) - 0.03571428571428571*f_ipj*f_ip2pj + 0.05595238095238095*f_ip2pj*f_ip1 - 0.004761904761904762*f_im1pj*f_im1 - 0.05595238095238095*f_ip2pj*f_ip1pj + 0.05595238095238095*f_ipj*f_im1 - 0.05595238095238095*f_i*f_im1 - 0.3654761904761905*f_ipj*f_ip1 + 0.002380952380952381*(f_im1pj*f_im1pj) + 0.4047619047619048*(f_ip1*f_ip1) - 0.8095238095238095*f_ip1*f_ip1pj + 0.002380952380952381*(f_ip2*f_ip2) + 0.05595238095238095*f_i*f_im1pj + 0.4047619047619048*(f_i*f_i) + 0.05595238095238095*f_ip2*f_ip1pj - 0.03571428571428571*f_im1pj*f_ip1pj + 0.3654761904761905*f_i*f_ip1 + 0.4047619047619048*(f_ip1pj*f_ip1pj) - 0.8095238095238095*f_ipj*f_i + 0.002380952380952381*(f_ip2pj*f_ip2pj) - 0.3654761904761905*f_i*f_ip1pj - 0.003571428571428571*f_ip2pj*f_im1;
        b1 += 0.1*f_ip1pj*f_im1 - 0.1*f_ipj*f_ip2 - 0.1*f_im1pj*f_ip1 + 0.1*f_ip2pj*f_i + 0.008333333333333333*f_ip2*f_im1pj + 1.191666666666667*f_ipj*f_ip1 + (f_ip1pj*f_ip1pj) - 0.09166666666666667*f_ip2pj*f_ip1 - f_ip1*f_ip1pj + f_ipj*f_i + 0.09166666666666667*f_i*f_im1pj - 1.191666666666667*f_i*f_ip1pj - 0.09166666666666667*f_ipj*f_im1 - 0.008333333333333333*f_ip2pj*f_im1 - (f_ipj*f_ipj) + 0.09166666666666667*f_ip2*f_ip1pj;
        b2 += -f_ipj*f_ip1pj + 0.01666666666666667*f_ip1pj*f_im1 + 0.01666666666666667*f_ipj*f_ip2 + 0.01666666666666667*f_ip1*f_im1pj + 0.01666666666666667*f_ip2pj*f_i + 0.008333333333333333*f_ip2*f_im1pj - 0.5916666666666667*f_ipj*f_ip1 + 1.133333333333333*f_ipj*f_i + 1.133333333333333*f_ip1*f_ip1pj - 0.5583333333333333*f_ip2pj*f_ip1 + 0.5*f_ip2pj*f_ip1pj - 0.5583333333333333*f_i*f_im1pj - 0.5916666666666667*f_i*f_ip1pj + 0.03333333333333333*f_im1pj*f_im1 - 0.05833333333333333*f_ipj*f_im1 + 0.008333333333333333*f_ip2pj*f_im1 - 0.05833333333333333*f_ip2*f_ip1pj + 0.03333333333333333*f_ip2pj*f_ip2 + 0.5*f_ipj*f_im1pj;
        b3 += -0.6666666666666667*f_ipj*f_ip1pj - 0.125*f_ip1pj*f_im1 - 0.5416666666666667*f_ip2pj*f_i - 0.04166666666666667*f_ip2*f_im1pj + 1.75*(f_ipj*f_ipj) + 0.04166666666666667*f_ip2pj*f_ip2 - 0.6666666666666667*f_ipj*f_im1pj + 0.2083333333333333*f_im1pj*f_ip1 + 0.125*f_ipj*f_ip2 + 0.1666666666666667*f_ipj*f_ip2pj - 1.208333333333333*f_ip2pj*f_ip1 - 0.04166666666666667*f_im1pj*f_im1 + 1.333333333333333*f_ip2pj*f_ip1pj + 0.125*f_ipj*f_im1 - 0.9583333333333333*f_ipj*f_ip1 + 0.3333333333333333*f_ip3pj*f_ip1 - 0.08333333333333333*(f_im1pj*f_im1pj) + 1.625*f_ip1*f_ip1pj - 0.3333333333333333*f_ip3pj*f_ip1pj + 0.5416666666666667*f_i*f_im1pj - 0.125*f_ip2*f_ip1pj + 0.1666666666666667*f_im1pj*f_ip1pj - 1.75*(f_ip1pj*f_ip1pj) - 1.625*f_ipj*f_i + 0.08333333333333333*(f_ip2pj*f_ip2pj) + 1.625*f_i*f_ip1pj + 0.04166666666666667*f_ip2pj*f_im1;
        b4 += 1.75*f_ipj*f_ip1pj + 0.25*(f_ip1pj*f_ip1pj) - 0.08333333333333333*f_ip2pj*f_i + 0.04166666666666667*f_ip2*f_im1pj - 0.125*f_ip2pj*f_im1pj + 0.125*f_ipj*f_ip3pj - (f_ipj*f_ipj) + 0.08333333333333333*f_ip2pj*f_ip2 - 0.375*f_ipj*f_im1pj - 0.25*f_im1pj*f_ip1 - 0.08333333333333333*f_ipj*f_ip2 - 0.5*f_ipj*f_ip2pj - 0.125*f_ip2pj*f_ip3pj + f_ip2pj*f_ip1 - 1.25*f_ip2pj*f_ip1pj - 0.04166666666666667*f_ip2*f_ip3pj + f_ipj*f_ip1 - 0.25*f_ip3pj*f_ip1 + 0.25*(f_im1pj*f_im1pj) - 1.5*f_ip1*f_ip1pj + 0.25*f_ip3pj*f_ip1pj - 0.04166666666666667*f_i*f_im1pj + 0.25*f_im1pj*f_ip1pj + 0.04166666666666667*f_i*f_ip3pj + 0.08333333333333333*f_ipj*f_i + 0.5*(f_ip2pj*f_ip2pj);
        b5 += 0.4*f_ipj*f_ip1pj + 0.75*(f_ip1pj*f_ip1pj) - 0.03333333333333333*f_ip2pj*f_i - 0.008333333333333333*f_ip2*f_im1pj + 0.3*f_ip2pj*f_im1pj - 0.3*f_ipj*f_ip3pj - 0.01666666666666667*f_ip3pj*f_im1 - 0.95*(f_ipj*f_ipj) - 0.1666666666666667*f_ip2pj*f_ip2 + 1.3*f_ipj*f_im1pj + 0.08333333333333333*f_ip1*f_im1pj - 0.03333333333333333*f_ipj*f_ip2 + 0.5*f_ipj*f_ip2pj - 0.3*f_ip2pj*f_ip3pj + 0.1666666666666667*f_ip2pj*f_ip1 + 0.01666666666666667*f_im1pj*f_im1 - 1.4*f_ip2pj*f_ip1pj + 0.05833333333333333*f_ip2*f_ip3pj - 0.03333333333333333*f_ipj*f_im1 - 0.1666666666666667*f_ipj*f_ip1 - 0.08333333333333333*f_ip3pj*f_ip1 - 0.3*(f_im1pj*f_im1pj) + 0.5*f_ip3pj*f_ip1pj - 0.09166666666666667*f_i*f_im1pj + 0.15*f_ip2*f_ip1pj - f_im1pj*f_ip1pj + 0.05*(f_ip3pj*f_ip3pj) + 0.04166666666666667*f_i*f_ip3pj + 0.2333333333333333*f_ipj*f_i + 0.45*(f_ip2pj*f_ip2pj) - 0.15*f_i*f_ip1pj + 0.03333333333333333*f_ip2pj*f_im1;
        b6 += -1.333333333333333*f_ipj*f_ip1pj + 0.05*f_ip1pj*f_im1 - 0.25*(f_ip1pj*f_ip1pj) - 0.008333333333333333*f_ip2*f_im1pj - 0.25*f_ip2pj*f_im1pj + 0.25*f_ipj*f_ip3pj + 0.01666666666666667*f_ip3pj*f_im1 + 1.083333333333333*(f_ipj*f_ipj) + 0.08333333333333333*f_ip2pj*f_ip2 - 0.9166666666666667*f_ipj*f_im1pj + 0.01666666666666667*f_im1pj*f_ip1 + 0.05*f_ipj*f_ip2 - 0.1666666666666667*f_ipj*f_ip2pj + 0.5833333333333333*f_ip2pj*f_ip3pj - 0.2166666666666667*f_ip2pj*f_ip1 + 1.666666666666667*f_ip2pj*f_ip1pj - 0.025*f_ip2*f_ip3pj - 0.01666666666666667*f_ipj*f_im1 + 0.1833333333333333*f_ip2pj*f_i - 0.1166666666666667*f_ipj*f_ip1 + 0.06666666666666667*f_ip3pj*f_ip1 + 0.1666666666666667*(f_im1pj*f_im1pj) + 0.25*f_ip1*f_ip1pj - 0.6666666666666667*f_ip3pj*f_ip1pj - 0.008333333333333333*f_i*f_im1pj - 0.1*f_ip2*f_ip1pj + 0.8333333333333333*f_im1pj*f_ip1pj - 0.08333333333333333*(f_ip3pj*f_ip3pj) - 0.05833333333333333*f_i*f_ip3pj + 0.08333333333333333*f_ipj*f_i - 0.9166666666666667*(f_ip2pj*f_ip2pj) - 0.2*f_i*f_ip1pj - 0.05*f_ip2pj*f_im1;
        b7 += 0.4285714285714286*f_ipj*f_ip1pj - 0.02142857142857143*f_ip1pj*f_im1 - 0.04285714285714286*f_ip2pj*f_i + 0.003571428571428571*f_ip2*f_im1pj + 0.07142857142857143*f_ip2pj*f_im1pj - 0.07142857142857143*f_ipj*f_ip3pj - 0.003571428571428571*f_ip3pj*f_im1 - 0.2857142857142857*(f_ipj*f_ipj) - 0.01428571428571429*f_ip2pj*f_ip2 + 0.2142857142857143*f_ipj*f_im1pj - 0.01071428571428571*f_ip1*f_im1pj - 0.01428571428571429*f_ipj*f_ip2 - 0.2142857142857143*f_ip2pj*f_ip3pj + 0.04285714285714286*f_ip2pj*f_ip1 - 0.003571428571428571*f_im1pj*f_im1 - 0.4285714285714286*f_ip2pj*f_ip1pj + 0.003571428571428571*f_ip2*f_ip3pj + 0.01428571428571429*f_ipj*f_im1 + 0.04285714285714286*f_ipj*f_ip1 - 0.01071428571428571*f_ip3pj*f_ip1 - 0.03571428571428571*(f_im1pj*f_im1pj) - 0.06428571428571429*f_ip1*f_ip1pj + 0.2142857142857143*f_ip3pj*f_ip1pj + 0.01071428571428571*f_i*f_im1pj + 0.02142857142857143*f_ip2*f_ip1pj - 0.2142857142857143*f_im1pj*f_ip1pj + 0.03571428571428571*(f_ip3pj*f_ip3pj) + 0.01071428571428571*f_i*f_ip3pj - 0.04285714285714286*f_ipj*f_i + 0.2857142857142857*(f_ip2pj*f_ip2pj) + 0.06428571428571429*f_i*f_ip1pj + 0.01428571428571429*f_ip2pj*f_im1;
      }
      b0 += 0.002380952380952381*(f_mjpn*f_mjpn) + 0.002380952380952381*(f_m3pn*f_m3pn) + 0.003571428571428571*f_m3mjpn*f_mjpn - 0.03571428571428571*f_m1pn*f_m3pn + 0.4047619047619048*(f_m1pn*f_m1pn) - 0.004761904761904762*f_m3mjpn*f_m3pn - 0.003571428571428571*f_m3pn*f_mjpn - 0.05595238095238095*f_m1mjpn*f_mjpn - 0.05595238095238095*f_m2mjpn*f_m3mjpn + 0.4047619047619048*(f_m1mjpn*f_m1mjpn) + 0.05595238095238095*f_m2mjpn*f_m3pn + 0.002380952380952381*(f_n*f_n) + 0.3654761904761905*f_m1pn*f_m2pn - 0.03571428571428571*f_m2mjpn*f_mjpn + 0.05595238095238095*f_m1mjpn*f_n - 0.05595238095238095*f_m2pn*f_m3pn + 0.3654761904761905*f_m1mjpn*f_m2mjpn + 0.03571428571428571*f_m2mjpn*f_n + 0.4047619047619048*(f_m2pn*f_m2pn) - 0.03571428571428571*f_m1mjpn*f_m3mjpn + 0.03571428571428571*f_m2pn*f_mjpn + 0.05595238095238095*f_m2pn*f_m3mjpn - 0.3654761904761905*f_m1mjpn*f_m2pn + 0.05595238095238095*f_m1pn*f_mjpn + 0.4047619047619048*(f_m2mjpn*f_m2mjpn) - 0.03571428571428571*f_m2pn*f_n - 0.003571428571428571*f_n*f_m3mjpn - 0.004761904761904762*f_n*f_mjpn + 0.03571428571428571*f_m1pn*f_m3mjpn + 0.003571428571428571*f_n*f_m3pn + 0.03571428571428571*f_m1mjpn*f_m3pn - 0.8095238095238095*f_m1mjpn*f_m1pn - 0.3654761904761905*f_m1pn*f_m2mjpn - 0.05595238095238095*f_m1pn*f_n + 0.002380952380952381*(f_m3mjpn*f_m3mjpn) - 0.8095238095238095*f_m2pn*f_m2mjpn;
      b1 += -0.008333333333333333*f_n*f_m3mjpn + 1.191666666666667*f_m2pn*f_m1mjpn + 0.1*f_m1pn*f_m3mjpn - (f_m1mjpn*f_m1mjpn) + 0.1*f_m2mjpn*f_n - (f_m2pn*f_m2pn) - 0.09166666666666667*f_m2pn*f_m3mjpn + 0.09166666666666667*f_m1pn*f_mjpn - 0.1*f_m1mjpn*f_m3pn + f_m2pn*f_m2mjpn + f_m1pn*f_m1mjpn - 1.191666666666667*f_m1pn*f_m2mjpn - 0.1*f_m2pn*f_mjpn + 0.008333333333333333*f_m3pn*f_mjpn - 0.09166666666666667*f_m1mjpn*f_n + 0.09166666666666667*f_m2mjpn*f_m3pn;
      b2 += 0.008333333333333333*f_n*f_m3mjpn + 0.5*f_m2pn*f_m3pn + 0.03333333333333333*f_n*f_mjpn + 1.133333333333333*f_m1pn*f_m1mjpn + 0.01666666666666667*f_m1pn*f_m3mjpn - 0.5*f_m1mjpn*f_m2mjpn + 0.01666666666666667*f_m2mjpn*f_n - 0.05833333333333333*f_m2pn*f_m3mjpn - 0.5583333333333333*f_m1pn*f_mjpn - 0.5583333333333333*f_m2mjpn*f_m3pn + 0.01666666666666667*f_m1mjpn*f_m3pn - 1.091666666666667*f_m2pn*f_m1mjpn - 0.5*f_m2pn*f_m1pn + 1.133333333333333*f_m2pn*f_m2mjpn + 0.03333333333333333*f_m3pn*f_m3mjpn + 0.01666666666666667*f_m2pn*f_mjpn + 0.008333333333333333*f_m3pn*f_mjpn - 0.05833333333333333*f_m1mjpn*f_n - 0.09166666666666667*f_m1pn*f_m2mjpn + 0.5*f_m1mjpn*f_mjpn;
      b3 += -0.08333333333333333*(f_mjpn*f_mjpn) - 0.08333333333333333*(f_m3pn*f_m3pn) + 0.1666666666666667*f_m1pn*f_m3pn - 0.08333333333333333*(f_m1pn*f_m1pn) - 0.04166666666666667*f_m3pn*f_m3mjpn - 0.04166666666666667*f_m3pn*f_mjpn - 0.6666666666666667*f_m1mjpn*f_mjpn + 0.3333333333333333*f_m1mjpn*f_m3mjpn + 1.666666666666667*(f_m1mjpn*f_m1mjpn) + 0.5416666666666667*f_m2mjpn*f_m3pn - 1.333333333333333*f_m1mjpn*f_m2mjpn + 0.1666666666666667*f_m2mjpn*f_mjpn - 0.4583333333333333*f_m2mjpn*f_n - 0.6666666666666667*f_m2pn*f_m3pn - 1.333333333333333*f_m2pn*f_m1pn + 0.125*f_m1mjpn*f_n + 1.666666666666667*(f_m2pn*f_m2pn) + 0.5416666666666667*f_m1pn*f_mjpn + 0.125*f_m2pn*f_m3mjpn - 1.708333333333333*f_m2pn*f_m2mjpn + 0.2083333333333333*f_m2pn*f_mjpn - 0.08333333333333333*(f_m2mjpn*f_m2mjpn) + 0.04166666666666667*f_n*f_m3mjpn - 0.04166666666666667*f_n*f_mjpn - 0.4583333333333333*f_m1pn*f_m3mjpn + 0.2083333333333333*f_m1mjpn*f_m3pn - 1.708333333333333*f_m1pn*f_m1mjpn + 2.958333333333333*f_m1pn*f_m2mjpn + 0.3333333333333333*f_m2pn*f_n - 0.2916666666666667*f_m2pn*f_m1mjpn;
      b4 += 0.25*(f_mjpn*f_mjpn) + 0.25*(f_m3pn*f_m3pn) - 0.125*f_m3mjpn*f_mjpn + 0.25*f_m1pn*f_m3pn - 0.5*(f_m1pn*f_m1pn) + 0.04166666666666667*f_m3pn*f_mjpn - 0.375*f_m1mjpn*f_mjpn - 0.25*f_m1mjpn*f_m3mjpn - 0.5*(f_m2mjpn*f_m2mjpn) - 0.04166666666666667*f_m2mjpn*f_m3pn + 1.375*f_m1mjpn*f_m2mjpn + 0.25*f_m2mjpn*f_mjpn + 0.25*f_m2mjpn*f_n - 0.375*f_m2pn*f_m3pn + 1.375*f_m2pn*f_m1pn + 0.04166666666666667*f_m1mjpn*f_n - 0.75*(f_m2pn*f_m2pn) + 0.125*f_m2mjpn*f_m3mjpn - 0.04166666666666667*f_m1pn*f_mjpn + 0.04166666666666667*f_m2pn*f_m3mjpn + 0.9583333333333333*f_m2pn*f_m1mjpn - 0.25*f_m2pn*f_mjpn - 0.75*(f_m1mjpn*f_m1mjpn) - 0.25*f_m2pn*f_n - 0.04166666666666667*f_n*f_m3mjpn + 0.25*f_m1pn*f_m3mjpn - 0.125*f_m3pn*f_n - 0.25*f_m1mjpn*f_m3pn - 0.9583333333333333*f_m1pn*f_m2mjpn + 0.125*f_m1pn*f_n;
      b5 += -0.3*(f_mjpn*f_mjpn) - 0.3*(f_m3pn*f_m3pn) + 0.3*f_m3mjpn*f_mjpn - f_m1pn*f_m3pn - 0.5*(f_m1pn*f_m1pn) + 0.01666666666666667*f_m3pn*f_m3mjpn - 0.008333333333333333*f_m3pn*f_mjpn + 1.3*f_m1mjpn*f_mjpn + 0.3*f_m2mjpn*f_m3mjpn - 0.5*(f_m2mjpn*f_m2mjpn) - 0.09166666666666667*f_m2mjpn*f_m3pn - 0.05*(f_n*f_n) + 1.7*f_m2pn*f_m1pn - f_m2mjpn*f_mjpn + 0.08333333333333333*f_m2mjpn*f_n + 1.3*f_m2pn*f_m3pn + 1.7*f_m2mjpn*f_m1mjpn - 0.09166666666666667*f_m1mjpn*f_n - 1.25*(f_m2pn*f_m2pn) - 0.5*f_m1mjpn*f_m3mjpn - 0.09166666666666667*f_m1pn*f_mjpn - 0.09166666666666667*f_m2pn*f_m3mjpn + 0.4166666666666667*f_m2pn*f_m2mjpn + 0.08333333333333333*f_m2pn*f_mjpn - 1.25*(f_m1mjpn*f_m1mjpn) - 0.5*f_m2pn*f_n - 0.008333333333333333*f_n*f_m3mjpn + 0.01666666666666667*f_n*f_mjpn + 0.08333333333333333*f_m1pn*f_m3mjpn + 0.3*f_m3pn*f_n + 0.08333333333333333*f_m1mjpn*f_m3pn + 0.4166666666666667*f_m1pn*f_m1mjpn - 0.4083333333333333*f_m1pn*f_m2mjpn + 0.3*f_m1pn*f_n - 0.05*(f_m3mjpn*f_m3mjpn) - 0.4083333333333333*f_m2pn*f_m1mjpn;
      b6 += 0.1666666666666667*(f_mjpn*f_mjpn) + 0.1666666666666667*(f_m3pn*f_m3pn) - 0.25*f_m3mjpn*f_mjpn + 0.8333333333333333*f_m1pn*f_m3pn + (f_m1pn*f_m1pn) - 0.008333333333333333*f_m3pn*f_mjpn - 0.9166666666666667*f_m1mjpn*f_mjpn + 0.6666666666666667*f_m1mjpn*f_m3mjpn + (f_m2mjpn*f_m2mjpn) - 0.008333333333333333*f_m2mjpn*f_m3pn + 0.08333333333333333*(f_n*f_n) - 2.25*f_m2mjpn*f_m1mjpn + 0.8333333333333333*f_m2mjpn*f_mjpn - 0.01666666666666667*f_m2mjpn*f_n - 0.9166666666666667*f_m2pn*f_m3pn - 2.25*f_m2pn*f_m1pn + 0.008333333333333333*f_m1mjpn*f_n + 1.25*(f_m2pn*f_m2pn) - 0.5833333333333333*f_m2mjpn*f_m3mjpn - 0.008333333333333333*f_m1pn*f_mjpn + 0.008333333333333333*f_m2pn*f_m3mjpn - 0.025*f_m2pn*f_m1mjpn + 0.01666666666666667*f_m2pn*f_mjpn + 1.25*(f_m1mjpn*f_m1mjpn) + 0.6666666666666667*f_m2pn*f_n + 0.008333333333333333*f_n*f_m3mjpn - 0.01666666666666667*f_m1pn*f_m3mjpn - 0.25*f_m3pn*f_n + 0.01666666666666667*f_m1mjpn*f_m3pn + 0.025*f_m1pn*f_m2mjpn - 0.5833333333333333*f_m1pn*f_n + 0.08333333333333333*(f_m3mjpn*f_m3mjpn);
      b7 += -0.03571428571428571*(f_mjpn*f_mjpn) - 0.03571428571428571*(f_m3pn*f_m3pn) + 0.07142857142857143*f_m3mjpn*f_mjpn + 0.2142857142857143*f_m2pn*f_m3pn - 0.3214285714285714*(f_m1pn*f_m1pn) - 0.003571428571428571*f_m3pn*f_m3mjpn + 0.003571428571428571*f_m3pn*f_mjpn + 0.2142857142857143*f_m1mjpn*f_mjpn + 0.2142857142857143*f_m2mjpn*f_m3mjpn - 0.3214285714285714*(f_m2mjpn*f_m2mjpn) + 0.01071428571428571*f_m2mjpn*f_m3pn - 0.03571428571428571*(f_n*f_n) + 0.6428571428571429*f_m2pn*f_m1pn - 0.2142857142857143*f_m2mjpn*f_mjpn - 0.01071428571428571*f_m2mjpn*f_n - 0.2142857142857143*f_m1pn*f_m3pn + 0.6428571428571429*f_m2mjpn*f_m1mjpn + 0.01071428571428571*f_m1mjpn*f_n - 0.3214285714285714*(f_m2pn*f_m2pn) - 0.2142857142857143*f_m1mjpn*f_m3mjpn + 0.01071428571428571*f_m1pn*f_mjpn + 0.01071428571428571*f_m2pn*f_m3mjpn - 0.03214285714285714*f_m2pn*f_m2mjpn - 0.01071428571428571*f_m2pn*f_mjpn - 0.3214285714285714*(f_m1mjpn*f_m1mjpn) - 0.2142857142857143*f_m2pn*f_n + 0.003571428571428571*f_n*f_m3mjpn - 0.003571428571428571*f_n*f_mjpn - 0.01071428571428571*f_m1pn*f_m3mjpn + 0.07142857142857143*f_m3pn*f_n - 0.01071428571428571*f_m1mjpn*f_m3pn - 0.03214285714285714*f_m1pn*f_m1mjpn + 0.03214285714285714*f_m1pn*f_m2mjpn + 0.2142857142857143*f_m1pn*f_n - 0.03571428571428571*(f_m3mjpn*f_m3mjpn) + 0.03214285714285714*f_m2pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
  *a6 = b6;
  *a7 = b7;
}
        
int cf_e33_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e33_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_quadratic_approximation_1_7(a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, a1_6, a1_7, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e33_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e33_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_linear_approximation_1_7(a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, a1_6, a1_7, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e33_find_zero_diff0: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e33_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=1) = sum(a_k*r^k, k=0..7) where y=j+r
     f1(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double b6 = 0.0;
  double b7 = 0.0;
  double f_im1, f_m3mjpn, f_ipj, f_mjpn, f_n, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip3pj, f_ip2, f_ip1, f_ip1pj, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_mjpn = F(-j+n);
      f_n = F(n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_im1 = F(i-1);
        f_ipj = F(i+j);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip3pj = F(i+3+j);
        f_ip2 = F(i+2);
        f_ip1 = F(i+1);
        f_ip1pj = F(i+1+j);
        b0 += 0.1*f_ip1pj*f_im1 - 0.1*f_ipj*f_ip2 - 0.1*f_im1pj*f_ip1 + 0.1*f_ip2pj*f_i + 0.008333333333333333*f_ip2*f_im1pj + 1.191666666666667*f_ipj*f_ip1 + (f_ip1pj*f_ip1pj) + f_ipj*f_i - f_ip1*f_ip1pj - 0.09166666666666667*f_ip2pj*f_ip1 + 0.09166666666666667*f_i*f_im1pj - 1.191666666666667*f_i*f_ip1pj - 0.09166666666666667*f_ipj*f_im1 - 0.008333333333333333*f_ip2pj*f_im1 - (f_ipj*f_ipj) + 0.09166666666666667*f_ip2*f_ip1pj;
        b1 += -2.0*f_ipj*f_ip1pj + 0.03333333333333333*f_ip1pj*f_im1 + 0.03333333333333333*f_ipj*f_ip2 + 0.03333333333333333*f_ip1*f_im1pj + 0.03333333333333333*f_ip2pj*f_i + 0.01666666666666667*f_ip2*f_im1pj - 1.183333333333333*f_ipj*f_ip1 - 1.116666666666667*f_ip2pj*f_ip1 + 2.266666666666667*f_ip1*f_ip1pj + 2.266666666666667*f_ipj*f_i + f_ip2pj*f_ip1pj - 1.116666666666667*f_i*f_im1pj - 1.183333333333333*f_i*f_ip1pj + 0.06666666666666667*f_im1pj*f_im1 - 0.1166666666666667*f_ipj*f_im1 + 0.01666666666666667*f_ip2pj*f_im1 - 0.1166666666666667*f_ip2*f_ip1pj + 0.06666666666666667*f_ip2pj*f_ip2 + f_ipj*f_im1pj;
        b2 += -2.0*f_ipj*f_ip1pj - 0.375*f_ip1pj*f_im1 - 1.625*f_ip2pj*f_i - 0.125*f_ip2*f_im1pj + 5.25*(f_ipj*f_ipj) + 0.125*f_ip2pj*f_ip2 - 2.0*f_ipj*f_im1pj + 0.625*f_im1pj*f_ip1 + 0.375*f_ipj*f_ip2 + 0.5*f_ipj*f_ip2pj - 3.625*f_ip2pj*f_ip1 - 0.125*f_im1pj*f_im1 + 4.0*f_ip2pj*f_ip1pj + 0.375*f_ipj*f_im1 - 2.875*f_ipj*f_ip1 + f_ip3pj*f_ip1 - 0.25*(f_im1pj*f_im1pj) + 4.875*f_ip1*f_ip1pj - f_ip3pj*f_ip1pj + 1.625*f_i*f_im1pj - 0.375*f_ip2*f_ip1pj + 0.5*f_im1pj*f_ip1pj - 5.25*(f_ip1pj*f_ip1pj) - 4.875*f_ipj*f_i + 0.25*(f_ip2pj*f_ip2pj) + 4.875*f_i*f_ip1pj + 0.125*f_ip2pj*f_im1;
        b3 += 7.0*f_ipj*f_ip1pj + (f_ip1pj*f_ip1pj) - 0.3333333333333333*f_ip2pj*f_i + 0.1666666666666667*f_ip2*f_im1pj - 0.5*f_ip2pj*f_im1pj + 0.5*f_ipj*f_ip3pj - 4.0*(f_ipj*f_ipj) + 0.3333333333333333*f_ip2pj*f_ip2 - 1.5*f_ipj*f_im1pj - f_im1pj*f_ip1 - 0.3333333333333333*f_ipj*f_ip2 - 2.0*f_ipj*f_ip2pj - 0.5*f_ip2pj*f_ip3pj + 4.0*f_ip2pj*f_ip1 - 5.0*f_ip2pj*f_ip1pj - 0.1666666666666667*f_ip2*f_ip3pj + 4.0*f_ipj*f_ip1 - f_ip3pj*f_ip1 + (f_im1pj*f_im1pj) - 6.0*f_ip1*f_ip1pj + f_ip3pj*f_ip1pj - 0.1666666666666667*f_i*f_im1pj + f_im1pj*f_ip1pj + 0.1666666666666667*f_i*f_ip3pj + 0.3333333333333333*f_ipj*f_i + 2.0*(f_ip2pj*f_ip2pj);
        b4 += 2.0*f_ipj*f_ip1pj + 3.75*(f_ip1pj*f_ip1pj) - 0.1666666666666667*f_ip2pj*f_i - 0.04166666666666667*f_ip2*f_im1pj + 1.5*f_ip2pj*f_im1pj - 1.5*f_ipj*f_ip3pj - 0.08333333333333333*f_ip3pj*f_im1 - 4.75*(f_ipj*f_ipj) - 0.8333333333333333*f_ip2pj*f_ip2 + 6.5*f_ipj*f_im1pj + 0.4166666666666667*f_ip1*f_im1pj - 0.1666666666666667*f_ipj*f_ip2 + 2.5*f_ipj*f_ip2pj - 1.5*f_ip2pj*f_ip3pj + 0.8333333333333333*f_ip2pj*f_ip1 + 0.08333333333333333*f_im1pj*f_im1 - 7.0*f_ip2pj*f_ip1pj + 0.2916666666666667*f_ip2*f_ip3pj - 0.1666666666666667*f_ipj*f_im1 - 0.8333333333333333*f_ipj*f_ip1 - 0.4166666666666667*f_ip3pj*f_ip1 - 1.5*(f_im1pj*f_im1pj) + 2.5*f_ip3pj*f_ip1pj - 0.4583333333333333*f_i*f_im1pj + 0.75*f_ip2*f_ip1pj - 5.0*f_im1pj*f_ip1pj + 0.25*(f_ip3pj*f_ip3pj) + 0.2083333333333333*f_i*f_ip3pj + 1.166666666666667*f_ipj*f_i + 2.25*(f_ip2pj*f_ip2pj) - 0.75*f_i*f_ip1pj + 0.1666666666666667*f_ip2pj*f_im1;
        b5 += -8.0*f_ipj*f_ip1pj + 0.3*f_ip1pj*f_im1 - 1.5*(f_ip1pj*f_ip1pj) - 0.05*f_ip2*f_im1pj - 1.5*f_ip2pj*f_im1pj + 1.5*f_ipj*f_ip3pj + 0.1*f_ip3pj*f_im1 + 6.5*(f_ipj*f_ipj) + 0.5*f_ip2pj*f_ip2 - 5.5*f_ipj*f_im1pj + 0.1*f_im1pj*f_ip1 + 0.3*f_ipj*f_ip2 - f_ipj*f_ip2pj + 3.5*f_ip2pj*f_ip3pj - 1.3*f_ip2pj*f_ip1 + 10.0*f_ip2pj*f_ip1pj - 0.15*f_ip2*f_ip3pj - 0.1*f_ipj*f_im1 + 1.1*f_ip2pj*f_i - 0.7*f_ipj*f_ip1 + 0.4*f_ip3pj*f_ip1 + (f_im1pj*f_im1pj) + 1.5*f_ip1*f_ip1pj - 4.0*f_ip3pj*f_ip1pj - 0.05*f_i*f_im1pj - 0.6*f_ip2*f_ip1pj + 5.0*f_im1pj*f_ip1pj - 0.5*(f_ip3pj*f_ip3pj) - 0.35*f_i*f_ip3pj + 0.5*f_ipj*f_i - 5.5*(f_ip2pj*f_ip2pj) - 1.2*f_i*f_ip1pj - 0.3*f_ip2pj*f_im1;
        b6 += 3.0*f_ipj*f_ip1pj - 0.15*f_ip1pj*f_im1 - 0.3*f_ip2pj*f_i + 0.025*f_ip2*f_im1pj + 0.5*f_ip2pj*f_im1pj - 0.5*f_ipj*f_ip3pj - 0.025*f_ip3pj*f_im1 - 2.0*(f_ipj*f_ipj) - 0.1*f_ip2pj*f_ip2 + 1.5*f_ipj*f_im1pj - 0.075*f_ip1*f_im1pj - 0.1*f_ipj*f_ip2 - 1.5*f_ip2pj*f_ip3pj + 0.3*f_ip2pj*f_ip1 - 0.025*f_im1pj*f_im1 - 3.0*f_ip2pj*f_ip1pj + 0.025*f_ip2*f_ip3pj + 0.1*f_ipj*f_im1 + 0.3*f_ipj*f_ip1 - 0.075*f_ip3pj*f_ip1 - 0.25*(f_im1pj*f_im1pj) - 0.45*f_ip1*f_ip1pj + 1.5*f_ip3pj*f_ip1pj + 0.075*f_i*f_im1pj + 0.15*f_ip2*f_ip1pj - 1.5*f_im1pj*f_ip1pj + 0.25*(f_ip3pj*f_ip3pj) + 0.075*f_i*f_ip3pj - 0.3*f_ipj*f_i + 2.0*(f_ip2pj*f_ip2pj) + 0.45*f_i*f_ip1pj + 0.1*f_ip2pj*f_im1;
      }
      b0 += -0.008333333333333333*f_n*f_m3mjpn + 1.191666666666667*f_m2pn*f_m1mjpn + 0.1*f_m1pn*f_m3mjpn - (f_m1mjpn*f_m1mjpn) + 0.1*f_m2mjpn*f_n - (f_m2pn*f_m2pn) + 0.09166666666666667*f_m2mjpn*f_m3pn + 0.09166666666666667*f_m1pn*f_mjpn - 0.09166666666666667*f_m2pn*f_m3mjpn + f_m1pn*f_m1mjpn - 1.191666666666667*f_m1pn*f_m2mjpn + f_m2pn*f_m2mjpn - 0.1*f_m2pn*f_mjpn + 0.008333333333333333*f_m3pn*f_mjpn - 0.09166666666666667*f_m1mjpn*f_n - 0.1*f_m1mjpn*f_m3pn;
      b1 += 0.01666666666666667*f_n*f_m3mjpn + f_m2pn*f_m3pn + 0.06666666666666667*f_n*f_mjpn + 2.266666666666667*f_m1pn*f_m1mjpn + 0.03333333333333333*f_m1pn*f_m3mjpn + 0.06666666666666667*f_m3pn*f_m3mjpn - f_m1mjpn*f_m2mjpn + 0.03333333333333333*f_m2mjpn*f_n - 1.116666666666667*f_m2mjpn*f_m3pn - f_m2pn*f_m1pn - 1.116666666666667*f_m1pn*f_mjpn - 0.1166666666666667*f_m2pn*f_m3mjpn - 2.183333333333333*f_m2pn*f_m1mjpn - 0.1833333333333333*f_m1pn*f_m2mjpn + 2.266666666666667*f_m2pn*f_m2mjpn + 0.03333333333333333*f_m2pn*f_mjpn + 0.01666666666666667*f_m3pn*f_mjpn - 0.1166666666666667*f_m1mjpn*f_n + 0.03333333333333333*f_m1mjpn*f_m3pn + f_m1mjpn*f_mjpn;
      b2 += -0.25*(f_mjpn*f_mjpn) - 0.25*(f_m3pn*f_m3pn) + 0.5*f_m1pn*f_m3pn - 0.25*(f_m1pn*f_m1pn) - 0.125*f_m3pn*f_m3mjpn - 0.125*f_m3pn*f_mjpn + 0.5*f_m2mjpn*f_mjpn + f_m1mjpn*f_m3mjpn - 0.25*(f_m2mjpn*f_m2mjpn) + 0.625*f_m1mjpn*f_m3pn - 4.0*f_m2pn*f_m1pn - 2.0*f_m1mjpn*f_mjpn - 1.375*f_m2mjpn*f_n - 2.0*f_m2pn*f_m3pn - 4.0*f_m1mjpn*f_m2mjpn + 0.375*f_m1mjpn*f_n + 5.0*(f_m2pn*f_m2pn) + 1.625*f_m1pn*f_mjpn - 1.375*f_m1pn*f_m3mjpn - 5.125*f_m2pn*f_m2mjpn + 0.625*f_m2pn*f_mjpn + 5.0*(f_m1mjpn*f_m1mjpn) + 0.125*f_n*f_m3mjpn - 0.125*f_n*f_mjpn + 0.375*f_m2pn*f_m3mjpn + 1.625*f_m2mjpn*f_m3pn + 8.875*f_m1pn*f_m2mjpn - 0.875*f_m2pn*f_m1mjpn + f_m2pn*f_n - 5.125*f_m1pn*f_m1mjpn;
      b3 += (f_mjpn*f_mjpn) + (f_m3pn*f_m3pn) - 0.5*f_m3mjpn*f_mjpn + f_m1pn*f_m3pn + 0.1666666666666667*f_m3pn*f_mjpn + f_m2mjpn*f_mjpn + 0.5*f_m2mjpn*f_m3mjpn - 2.0*(f_m2mjpn*f_m2mjpn) - f_m1mjpn*f_m3pn + 5.5*f_m2pn*f_m1pn - 1.5*f_m1mjpn*f_mjpn + f_m2mjpn*f_n - 1.5*f_m2pn*f_m3pn - 2.0*(f_m1pn*f_m1pn) + 0.1666666666666667*f_m1mjpn*f_n - 3.0*(f_m2pn*f_m2pn) - f_m1mjpn*f_m3mjpn + 5.5*f_m1mjpn*f_m2mjpn - 0.1666666666666667*f_m1pn*f_mjpn + f_m1pn*f_m3mjpn - 3.833333333333333*f_m1pn*f_m2mjpn - f_m2pn*f_mjpn - 3.0*(f_m1mjpn*f_m1mjpn) + 0.5*f_m1pn*f_n - 0.1666666666666667*f_n*f_m3mjpn + 0.1666666666666667*f_m2pn*f_m3mjpn - 0.5*f_m3pn*f_n - 0.1666666666666667*f_m2mjpn*f_m3pn + 3.833333333333333*f_m2pn*f_m1mjpn - f_m2pn*f_n;
      b4 += -1.5*(f_mjpn*f_mjpn) - 1.5*(f_m3pn*f_m3pn) + 1.5*f_m3mjpn*f_mjpn + 6.5*f_m2pn*f_m3pn - 6.25*(f_m2pn*f_m2pn) + 0.08333333333333333*f_m3pn*f_m3mjpn - 0.04166666666666667*f_m3pn*f_mjpn + 6.5*f_m1mjpn*f_mjpn + 1.5*f_m2mjpn*f_m3mjpn - 6.25*(f_m1mjpn*f_m1mjpn) + 0.4166666666666667*f_m1mjpn*f_m3pn - 0.25*(f_n*f_n) + 8.5*f_m2mjpn*f_m1mjpn - 5.0*f_m2mjpn*f_mjpn - 0.4583333333333333*f_m1mjpn*f_n - 5.0*f_m1pn*f_m3pn + 8.5*f_m2pn*f_m1pn + 0.4166666666666667*f_m2mjpn*f_n - 2.5*(f_m1pn*f_m1pn) - 2.5*f_m1mjpn*f_m3mjpn - 0.4583333333333333*f_m1pn*f_mjpn - 0.4583333333333333*f_m2pn*f_m3mjpn + 2.083333333333333*f_m1pn*f_m1mjpn + 0.4166666666666667*f_m2pn*f_mjpn - 2.5*(f_m2mjpn*f_m2mjpn) + 1.5*f_m1pn*f_n - 0.04166666666666667*f_n*f_m3mjpn + 0.08333333333333333*f_n*f_mjpn + 0.4166666666666667*f_m1pn*f_m3mjpn + 1.5*f_m3pn*f_n - 0.4583333333333333*f_m2mjpn*f_m3pn - 2.041666666666667*f_m1pn*f_m2mjpn + 2.083333333333333*f_m2pn*f_m2mjpn - 2.5*f_m2pn*f_n - 0.25*(f_m3mjpn*f_m3mjpn) - 2.041666666666667*f_m2pn*f_m1mjpn;
      b5 += (f_mjpn*f_mjpn) + (f_m3pn*f_m3pn) - 1.5*f_m3mjpn*f_mjpn - 5.5*f_m2pn*f_m3pn + 7.5*(f_m2pn*f_m2pn) - 0.05*f_m3pn*f_mjpn - 5.5*f_m1mjpn*f_mjpn + 4.0*f_m1mjpn*f_m3mjpn + 7.5*(f_m1mjpn*f_m1mjpn) + 0.1*f_m1mjpn*f_m3pn + 0.5*(f_n*f_n) - 13.5*f_m2pn*f_m1pn + 5.0*f_m2mjpn*f_mjpn + 0.05*f_m1mjpn*f_n + 5.0*f_m1pn*f_m3pn - 13.5*f_m2mjpn*f_m1mjpn - 0.1*f_m2mjpn*f_n + 6.0*(f_m1pn*f_m1pn) - 3.5*f_m2mjpn*f_m3mjpn - 0.05*f_m1pn*f_mjpn + 0.05*f_m2pn*f_m3mjpn + 0.15*f_m1pn*f_m2mjpn + 0.1*f_m2pn*f_mjpn + 6.0*(f_m2mjpn*f_m2mjpn) - 3.5*f_m1pn*f_n + 0.05*f_n*f_m3mjpn - 0.1*f_m1pn*f_m3mjpn - 1.5*f_m3pn*f_n - 0.05*f_m2mjpn*f_m3pn - 0.15*f_m2pn*f_m1mjpn + 4.0*f_m2pn*f_n + 0.5*(f_m3mjpn*f_m3mjpn);
      b6 += -0.25*(f_mjpn*f_mjpn) - 0.25*(f_m3pn*f_m3pn) + 0.5*f_m3mjpn*f_mjpn - 1.5*f_m1pn*f_m3pn - 2.25*(f_m2pn*f_m2pn) - 0.025*f_m3pn*f_m3mjpn + 0.025*f_m3pn*f_mjpn + 1.5*f_m1mjpn*f_mjpn + 1.5*f_m2mjpn*f_m3mjpn - 2.25*(f_m1mjpn*f_m1mjpn) - 0.075*f_m1mjpn*f_m3pn - 0.25*(f_n*f_n) + 4.5*f_m2pn*f_m1pn - 1.5*f_m2mjpn*f_mjpn + 0.075*f_m1mjpn*f_n + 1.5*f_m2pn*f_m3pn + 4.5*f_m2mjpn*f_m1mjpn - 0.075*f_m2mjpn*f_n - 2.25*(f_m1pn*f_m1pn) - 1.5*f_m1mjpn*f_m3mjpn + 0.075*f_m1pn*f_mjpn + 0.075*f_m2pn*f_m3mjpn - 0.225*f_m1pn*f_m1mjpn - 0.075*f_m2pn*f_mjpn - 2.25*(f_m2mjpn*f_m2mjpn) + 1.5*f_m1pn*f_n + 0.025*f_n*f_m3mjpn - 0.025*f_n*f_mjpn - 0.075*f_m1pn*f_m3mjpn + 0.5*f_m3pn*f_n + 0.075*f_m2mjpn*f_m3pn + 0.225*f_m1pn*f_m2mjpn - 0.225*f_m2pn*f_m2mjpn - 1.5*f_m2pn*f_n - 0.25*(f_m3mjpn*f_m3mjpn) + 0.225*f_m2pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
  *a6 = b6;
  *a7 = b7;
}
        
int cf_e33_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e33_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_quadratic_approximation_1_6(a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, a1_6, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e33_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e33_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_linear_approximation_1_6(a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, a1_6, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e33_find_zero_diff1: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e33_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=2) = sum(a_k*r^k, k=0..7) where y=j+r
     f1(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double b6 = 0.0;
  double b7 = 0.0;
  double f_ipj, f_m3mjpn, f_im1, f_m1mjpn, f_n, f_mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip3pj, f_ip2, f_ip1, f_ip1pj, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_m1mjpn = F(-1-j+n);
      f_n = F(n);
      f_mjpn = F(-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_im1 = F(i-1);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip3pj = F(i+3+j);
        f_ip2 = F(i+2);
        f_ip1 = F(i+1);
        f_ip1pj = F(i+1+j);
        b0 += -2.0*f_ipj*f_ip1pj + 0.03333333333333333*f_ip1pj*f_im1 + 0.03333333333333333*f_ipj*f_ip2 + 0.03333333333333333*f_ip1*f_im1pj + 0.03333333333333333*f_ip2pj*f_i + 0.01666666666666667*f_ip2*f_im1pj - 1.183333333333333*f_ipj*f_ip1 + 2.266666666666667*f_ipj*f_i + 2.266666666666667*f_ip1*f_ip1pj - 1.116666666666667*f_ip2pj*f_ip1 + f_ip2pj*f_ip1pj - 1.116666666666667*f_i*f_im1pj - 1.183333333333333*f_i*f_ip1pj + 0.06666666666666667*f_im1pj*f_im1 - 0.1166666666666667*f_ipj*f_im1 + 0.01666666666666667*f_ip2pj*f_im1 - 0.1166666666666667*f_ip2*f_ip1pj + 0.06666666666666667*f_ip2pj*f_ip2 + f_ipj*f_im1pj;
        b1 += -4.0*f_ipj*f_ip1pj - 0.75*f_ip1pj*f_im1 - 3.25*f_ip2pj*f_i - 0.25*f_ip2*f_im1pj + 10.5*(f_ipj*f_ipj) + 0.25*f_ip2pj*f_ip2 - 4.0*f_ipj*f_im1pj + 1.25*f_im1pj*f_ip1 + 0.75*f_ipj*f_ip2 + f_ipj*f_ip2pj - 7.25*f_ip2pj*f_ip1 - 0.25*f_im1pj*f_im1 + 8.0*f_ip2pj*f_ip1pj + 0.75*f_ipj*f_im1 - 5.75*f_ipj*f_ip1 + 2.0*f_ip3pj*f_ip1 - 0.5*(f_im1pj*f_im1pj) + 9.75*f_ip1*f_ip1pj - 2.0*f_ip3pj*f_ip1pj + 3.25*f_i*f_im1pj - 0.75*f_ip2*f_ip1pj + f_im1pj*f_ip1pj - 10.5*(f_ip1pj*f_ip1pj) - 9.75*f_ipj*f_i + 0.5*(f_ip2pj*f_ip2pj) + 9.75*f_i*f_ip1pj + 0.25*f_ip2pj*f_im1;
        b2 += 21.0*f_ipj*f_ip1pj + 3.0*(f_ip1pj*f_ip1pj) - f_ip2pj*f_i + 0.5*f_ip2*f_im1pj - 1.5*f_ip2pj*f_im1pj + 1.5*f_ipj*f_ip3pj - 12.0*(f_ipj*f_ipj) + f_ip2pj*f_ip2 - 4.5*f_ipj*f_im1pj - 3.0*f_im1pj*f_ip1 - f_ipj*f_ip2 - 6.0*f_ipj*f_ip2pj - 1.5*f_ip2pj*f_ip3pj + 12.0*f_ip2pj*f_ip1 - 15.0*f_ip2pj*f_ip1pj - 0.5*f_ip2*f_ip3pj + 12.0*f_ipj*f_ip1 - 3.0*f_ip3pj*f_ip1 + 3.0*(f_im1pj*f_im1pj) - 18.0*f_ip1*f_ip1pj + 3.0*f_ip3pj*f_ip1pj - 0.5*f_i*f_im1pj + 3.0*f_im1pj*f_ip1pj + 0.5*f_i*f_ip3pj + f_ipj*f_i + 6.0*(f_ip2pj*f_ip2pj);
        b3 += 8.0*f_ipj*f_ip1pj + 15.0*(f_ip1pj*f_ip1pj) - 0.6666666666666667*f_ip2pj*f_i - 0.1666666666666667*f_ip2*f_im1pj + 6.0*f_ip2pj*f_im1pj - 6.0*f_ipj*f_ip3pj - 0.3333333333333333*f_ip3pj*f_im1 - 19.0*(f_ipj*f_ipj) - 3.333333333333333*f_ip2pj*f_ip2 + 26.0*f_ipj*f_im1pj + 1.666666666666667*f_ip1*f_im1pj - 0.6666666666666667*f_ipj*f_ip2 + 10.0*f_ipj*f_ip2pj - 6.0*f_ip2pj*f_ip3pj + 3.333333333333333*f_ip2pj*f_ip1 + 0.3333333333333333*f_im1pj*f_im1 - 28.0*f_ip2pj*f_ip1pj + 1.166666666666667*f_ip2*f_ip3pj - 0.6666666666666667*f_ipj*f_im1 - 3.333333333333333*f_ipj*f_ip1 - 1.666666666666667*f_ip3pj*f_ip1 - 6.0*(f_im1pj*f_im1pj) + 10.0*f_ip3pj*f_ip1pj - 1.833333333333333*f_i*f_im1pj + 3.0*f_ip2*f_ip1pj - 20.0*f_im1pj*f_ip1pj + (f_ip3pj*f_ip3pj) + 0.8333333333333333*f_i*f_ip3pj + 4.666666666666667*f_ipj*f_i + 9.0*(f_ip2pj*f_ip2pj) - 3.0*f_i*f_ip1pj + 0.6666666666666667*f_ip2pj*f_im1;
        b4 += -40.0*f_ipj*f_ip1pj + 1.5*f_ip1pj*f_im1 - 7.5*(f_ip1pj*f_ip1pj) - 0.25*f_ip2*f_im1pj - 7.5*f_ip2pj*f_im1pj + 7.5*f_ipj*f_ip3pj + 0.5*f_ip3pj*f_im1 + 32.5*(f_ipj*f_ipj) + 2.5*f_ip2pj*f_ip2 - 27.5*f_ipj*f_im1pj + 0.5*f_im1pj*f_ip1 + 1.5*f_ipj*f_ip2 - 5.0*f_ipj*f_ip2pj + 17.5*f_ip2pj*f_ip3pj - 6.5*f_ip2pj*f_ip1 + 50.0*f_ip2pj*f_ip1pj - 0.75*f_ip2*f_ip3pj - 0.5*f_ipj*f_im1 + 5.5*f_ip2pj*f_i - 3.5*f_ipj*f_ip1 + 2.0*f_ip3pj*f_ip1 + 5.0*(f_im1pj*f_im1pj) + 7.5*f_ip1*f_ip1pj - 20.0*f_ip3pj*f_ip1pj - 0.25*f_i*f_im1pj - 3.0*f_ip2*f_ip1pj + 25.0*f_im1pj*f_ip1pj - 2.5*(f_ip3pj*f_ip3pj) - 1.75*f_i*f_ip3pj + 2.5*f_ipj*f_i - 27.5*(f_ip2pj*f_ip2pj) - 6.0*f_i*f_ip1pj - 1.5*f_ip2pj*f_im1;
        b5 += 18.0*f_ipj*f_ip1pj - 0.9*f_ip1pj*f_im1 - 1.8*f_ip2pj*f_i + 0.15*f_ip2*f_im1pj + 3.0*f_ip2pj*f_im1pj - 3.0*f_ipj*f_ip3pj - 0.15*f_ip3pj*f_im1 - 12.0*(f_ipj*f_ipj) - 0.6*f_ip2pj*f_ip2 + 9.0*f_ipj*f_im1pj - 0.45*f_ip1*f_im1pj - 0.6*f_ipj*f_ip2 - 9.0*f_ip2pj*f_ip3pj + 1.8*f_ip2pj*f_ip1 - 0.15*f_im1pj*f_im1 - 18.0*f_ip2pj*f_ip1pj + 0.15*f_ip2*f_ip3pj + 0.6*f_ipj*f_im1 + 1.8*f_ipj*f_ip1 - 0.45*f_ip3pj*f_ip1 - 1.5*(f_im1pj*f_im1pj) - 2.7*f_ip1*f_ip1pj + 9.0*f_ip3pj*f_ip1pj + 0.45*f_i*f_im1pj + 0.9*f_ip2*f_ip1pj - 9.0*f_im1pj*f_ip1pj + 1.5*(f_ip3pj*f_ip3pj) + 0.45*f_i*f_ip3pj - 1.8*f_ipj*f_i + 12.0*(f_ip2pj*f_ip2pj) + 2.7*f_i*f_ip1pj + 0.6*f_ip2pj*f_im1;
      }
      b0 += 0.01666666666666667*f_n*f_m3mjpn + f_m2pn*f_m3pn + 0.06666666666666667*f_n*f_mjpn - 0.1833333333333333*f_m1pn*f_m2mjpn + 0.03333333333333333*f_m1pn*f_m3mjpn + 0.06666666666666667*f_m3pn*f_m3mjpn - f_m1mjpn*f_m2mjpn + 0.03333333333333333*f_m2mjpn*f_n + 0.03333333333333333*f_m1mjpn*f_m3pn - f_m2pn*f_m1pn - 1.116666666666667*f_m1pn*f_mjpn - 0.1166666666666667*f_m2pn*f_m3mjpn + 2.266666666666667*f_m1pn*f_m1mjpn + 2.266666666666667*f_m2pn*f_m2mjpn - 2.183333333333333*f_m2pn*f_m1mjpn + 0.03333333333333333*f_m2pn*f_mjpn + 0.01666666666666667*f_m3pn*f_mjpn - 0.1166666666666667*f_m1mjpn*f_n - 1.116666666666667*f_m2mjpn*f_m3pn + f_m1mjpn*f_mjpn;
      b1 += -0.5*(f_mjpn*f_mjpn) - 0.5*(f_m3pn*f_m3pn) + f_m1pn*f_m3pn - 0.5*(f_m1pn*f_m1pn) - 0.25*f_m3pn*f_m3mjpn - 0.25*f_m3pn*f_mjpn + f_m2mjpn*f_mjpn + 2.0*f_m1mjpn*f_m3mjpn - 0.5*(f_m2mjpn*f_m2mjpn) + 3.25*f_m2mjpn*f_m3pn - 8.0*f_m2pn*f_m1pn - 4.0*f_m1mjpn*f_mjpn + 0.75*f_m1mjpn*f_n - 4.0*f_m2pn*f_m3pn - 8.0*f_m1mjpn*f_m2mjpn - 2.75*f_m2mjpn*f_n + 10.0*(f_m2pn*f_m2pn) + 3.25*f_m1pn*f_mjpn - 2.75*f_m1pn*f_m3mjpn - 1.75*f_m2pn*f_m1mjpn + 1.25*f_m2pn*f_mjpn + 10.0*(f_m1mjpn*f_m1mjpn) + 0.25*f_n*f_m3mjpn - 0.25*f_n*f_mjpn + 0.75*f_m2pn*f_m3mjpn + 1.25*f_m1mjpn*f_m3pn - 10.25*f_m2pn*f_m2mjpn - 10.25*f_m1pn*f_m1mjpn + 2.0*f_m2pn*f_n + 17.75*f_m1pn*f_m2mjpn;
      b2 += 3.0*(f_mjpn*f_mjpn) + 3.0*(f_m3pn*f_m3pn) - 1.5*f_m3mjpn*f_mjpn + 3.0*f_m1pn*f_m3pn + 0.5*f_m3pn*f_mjpn + 3.0*f_m2mjpn*f_mjpn - 3.0*f_m1mjpn*f_m3mjpn - 6.0*(f_m2mjpn*f_m2mjpn) - 0.5*f_m2mjpn*f_m3pn + 16.5*f_m1mjpn*f_m2mjpn - 4.5*f_m1mjpn*f_mjpn + 0.5*f_m1mjpn*f_n - 4.5*f_m2pn*f_m3pn - 6.0*(f_m1pn*f_m1pn) + 3.0*f_m2mjpn*f_n - 9.0*(f_m2pn*f_m2pn) + 1.5*f_m2mjpn*f_m3mjpn + 16.5*f_m2pn*f_m1pn - 0.5*f_m1pn*f_mjpn + 3.0*f_m1pn*f_m3mjpn + 11.5*f_m2pn*f_m1mjpn - 3.0*f_m2pn*f_mjpn - 9.0*(f_m1mjpn*f_m1mjpn) - 3.0*f_m2pn*f_n - 0.5*f_n*f_m3mjpn + 0.5*f_m2pn*f_m3mjpn - 1.5*f_m3pn*f_n - 3.0*f_m1mjpn*f_m3pn - 11.5*f_m1pn*f_m2mjpn + 1.5*f_m1pn*f_n;
      b3 += -6.0*(f_mjpn*f_mjpn) - 6.0*(f_m3pn*f_m3pn) + 6.0*f_m3mjpn*f_mjpn - 20.0*f_m1pn*f_m3pn - 10.0*(f_m1pn*f_m1pn) + 0.3333333333333333*f_m3pn*f_m3mjpn - 0.1666666666666667*f_m3pn*f_mjpn + 26.0*f_m1mjpn*f_mjpn + 6.0*f_m2mjpn*f_m3mjpn - 10.0*(f_m2mjpn*f_m2mjpn) - 1.833333333333333*f_m2mjpn*f_m3pn - (f_n*f_n) + 34.0*f_m2pn*f_m1pn - 20.0*f_m2mjpn*f_mjpn + 1.666666666666667*f_m2mjpn*f_n + 26.0*f_m2pn*f_m3pn + 34.0*f_m2mjpn*f_m1mjpn - 1.833333333333333*f_m1mjpn*f_n - 25.0*(f_m2pn*f_m2pn) - 10.0*f_m1mjpn*f_m3mjpn - 1.833333333333333*f_m1pn*f_mjpn - 1.833333333333333*f_m2pn*f_m3mjpn - 8.166666666666667*f_m1pn*f_m2mjpn + 1.666666666666667*f_m2pn*f_mjpn - 25.0*(f_m1mjpn*f_m1mjpn) - 10.0*f_m2pn*f_n - 0.1666666666666667*f_n*f_m3mjpn + 0.3333333333333333*f_n*f_mjpn + 1.666666666666667*f_m1pn*f_m3mjpn + 6.0*f_m3pn*f_n + 1.666666666666667*f_m1mjpn*f_m3pn + 8.333333333333333*f_m2pn*f_m2mjpn + 8.333333333333333*f_m1pn*f_m1mjpn + 6.0*f_m1pn*f_n - (f_m3mjpn*f_m3mjpn) - 8.166666666666667*f_m2pn*f_m1mjpn;
      b4 += 5.0*(f_mjpn*f_mjpn) + 5.0*(f_m3pn*f_m3pn) - 7.5*f_m3mjpn*f_mjpn + 25.0*f_m1pn*f_m3pn + 30.0*(f_m1pn*f_m1pn) - 0.25*f_m3pn*f_mjpn - 27.5*f_m1mjpn*f_mjpn + 20.0*f_m1mjpn*f_m3mjpn + 30.0*(f_m2mjpn*f_m2mjpn) - 0.25*f_m2mjpn*f_m3pn + 2.5*(f_n*f_n) - 67.5*f_m2mjpn*f_m1mjpn + 25.0*f_m2mjpn*f_mjpn - 0.5*f_m2mjpn*f_n - 27.5*f_m2pn*f_m3pn - 67.5*f_m2pn*f_m1pn + 0.25*f_m1mjpn*f_n + 37.5*(f_m2pn*f_m2pn) - 17.5*f_m2mjpn*f_m3mjpn - 0.25*f_m1pn*f_mjpn + 0.25*f_m2pn*f_m3mjpn - 0.75*f_m2pn*f_m1mjpn + 0.5*f_m2pn*f_mjpn + 37.5*(f_m1mjpn*f_m1mjpn) + 20.0*f_m2pn*f_n + 0.25*f_n*f_m3mjpn - 0.5*f_m1pn*f_m3mjpn - 7.5*f_m3pn*f_n + 0.5*f_m1mjpn*f_m3pn + 0.75*f_m1pn*f_m2mjpn - 17.5*f_m1pn*f_n + 2.5*(f_m3mjpn*f_m3mjpn);
      b5 += -1.5*(f_mjpn*f_mjpn) - 1.5*(f_m3pn*f_m3pn) + 3.0*f_m3mjpn*f_mjpn + 9.0*f_m2pn*f_m3pn - 13.5*(f_m1pn*f_m1pn) - 0.15*f_m3pn*f_m3mjpn + 0.15*f_m3pn*f_mjpn + 9.0*f_m1mjpn*f_mjpn + 9.0*f_m2mjpn*f_m3mjpn - 13.5*(f_m2mjpn*f_m2mjpn) + 0.45*f_m2mjpn*f_m3pn - 1.5*(f_n*f_n) + 27.0*f_m2mjpn*f_m1mjpn - 9.0*f_m2mjpn*f_mjpn - 0.45*f_m2mjpn*f_n - 9.0*f_m1pn*f_m3pn + 27.0*f_m2pn*f_m1pn + 0.45*f_m1mjpn*f_n - 13.5*(f_m2pn*f_m2pn) - 9.0*f_m1mjpn*f_m3mjpn + 0.45*f_m1pn*f_mjpn + 0.45*f_m2pn*f_m3mjpn + 1.35*f_m1pn*f_m2mjpn - 0.45*f_m2pn*f_mjpn - 13.5*(f_m1mjpn*f_m1mjpn) - 9.0*f_m2pn*f_n + 0.15*f_n*f_m3mjpn - 0.15*f_n*f_mjpn - 0.45*f_m1pn*f_m3mjpn + 3.0*f_m3pn*f_n - 0.45*f_m1mjpn*f_m3pn - 1.35*f_m2pn*f_m2mjpn - 1.35*f_m1pn*f_m1mjpn + 9.0*f_m1pn*f_n - 1.5*(f_m3mjpn*f_m3mjpn) + 1.35*f_m2pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
  *a6 = b6;
  *a7 = b7;
}
        
int cf_e33_find_extreme_diff2(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e33_compute_coeffs_diff2(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_quadratic_approximation_1_5(a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e33_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e33_compute_coeffs_diff2(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_linear_approximation_1_5(a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e33_find_zero_diff2: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e33_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=3) = sum(a_k*r^k, k=0..7) where y=j+r
     f1(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double b6 = 0.0;
  double b7 = 0.0;
  double f_ipj, f_m3mjpn, f_im1, f_mjpn, f_n, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip3pj, f_ip2, f_ip1, f_ip1pj, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_mjpn = F(-j+n);
      f_n = F(n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_im1 = F(i-1);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip3pj = F(i+3+j);
        f_ip2 = F(i+2);
        f_ip1 = F(i+1);
        f_ip1pj = F(i+1+j);
        b0 += -4.0*f_ipj*f_ip1pj - 0.75*f_ip1pj*f_im1 - 3.25*f_ip2pj*f_i - 0.25*f_ip2*f_im1pj + 10.5*(f_ipj*f_ipj) + 0.25*f_ip2pj*f_ip2 - 4.0*f_ipj*f_im1pj + 1.25*f_im1pj*f_ip1 + 0.75*f_ipj*f_ip2 + f_ipj*f_ip2pj - 7.25*f_ip2pj*f_ip1 - 0.25*f_im1pj*f_im1 + 8.0*f_ip2pj*f_ip1pj + 0.75*f_ipj*f_im1 - 5.75*f_ipj*f_ip1 + 2.0*f_ip3pj*f_ip1 - 0.5*(f_im1pj*f_im1pj) + 9.75*f_ip1*f_ip1pj - 2.0*f_ip3pj*f_ip1pj + 3.25*f_i*f_im1pj - 0.75*f_ip2*f_ip1pj + f_im1pj*f_ip1pj - 10.5*(f_ip1pj*f_ip1pj) - 9.75*f_ipj*f_i + 0.5*(f_ip2pj*f_ip2pj) + 9.75*f_i*f_ip1pj + 0.25*f_ip2pj*f_im1;
        b1 += 42.0*f_ipj*f_ip1pj + 6.0*(f_ip1pj*f_ip1pj) - 2.0*f_ip2pj*f_i + f_ip2*f_im1pj - 3.0*f_ip2pj*f_im1pj + 3.0*f_ipj*f_ip3pj - 24.0*(f_ipj*f_ipj) + 2.0*f_ip2pj*f_ip2 - 9.0*f_ipj*f_im1pj - 6.0*f_im1pj*f_ip1 - 2.0*f_ipj*f_ip2 - 12.0*f_ipj*f_ip2pj - 3.0*f_ip2pj*f_ip3pj + 24.0*f_ip2pj*f_ip1 - 30.0*f_ip2pj*f_ip1pj - f_ip2*f_ip3pj + 24.0*f_ipj*f_ip1 - 6.0*f_ip3pj*f_ip1 + 6.0*(f_im1pj*f_im1pj) - 36.0*f_ip1*f_ip1pj + 6.0*f_ip3pj*f_ip1pj - f_i*f_im1pj + 6.0*f_im1pj*f_ip1pj + f_i*f_ip3pj + 2.0*f_ipj*f_i + 12.0*(f_ip2pj*f_ip2pj);
        b2 += 24.0*f_ipj*f_ip1pj + 45.0*(f_ip1pj*f_ip1pj) - 2.0*f_ip2pj*f_i - 0.5*f_ip2*f_im1pj + 18.0*f_ip2pj*f_im1pj - 18.0*f_ipj*f_ip3pj - f_ip3pj*f_im1 - 57.0*(f_ipj*f_ipj) - 10.0*f_ip2pj*f_ip2 + 78.0*f_ipj*f_im1pj + 5.0*f_ip1*f_im1pj - 2.0*f_ipj*f_ip2 + 30.0*f_ipj*f_ip2pj - 18.0*f_ip2pj*f_ip3pj + 10.0*f_ip2pj*f_ip1 + f_im1pj*f_im1 - 84.0*f_ip2pj*f_ip1pj + 3.5*f_ip2*f_ip3pj - 2.0*f_ipj*f_im1 - 10.0*f_ipj*f_ip1 - 5.0*f_ip3pj*f_ip1 - 18.0*(f_im1pj*f_im1pj) + 30.0*f_ip3pj*f_ip1pj - 5.5*f_i*f_im1pj + 9.0*f_ip2*f_ip1pj - 60.0*f_im1pj*f_ip1pj + 3.0*(f_ip3pj*f_ip3pj) + 2.5*f_i*f_ip3pj + 14.0*f_ipj*f_i + 27.0*(f_ip2pj*f_ip2pj) - 9.0*f_i*f_ip1pj + 2.0*f_ip2pj*f_im1;
        b3 += -160.0*f_ipj*f_ip1pj + 6.0*f_ip1pj*f_im1 - 30.0*(f_ip1pj*f_ip1pj) - f_ip2*f_im1pj - 30.0*f_ip2pj*f_im1pj + 30.0*f_ipj*f_ip3pj + 2.0*f_ip3pj*f_im1 + 130.0*(f_ipj*f_ipj) + 10.0*f_ip2pj*f_ip2 - 110.0*f_ipj*f_im1pj + 2.0*f_im1pj*f_ip1 + 6.0*f_ipj*f_ip2 - 20.0*f_ipj*f_ip2pj + 70.0*f_ip2pj*f_ip3pj - 26.0*f_ip2pj*f_ip1 + 200.0*f_ip2pj*f_ip1pj - 3.0*f_ip2*f_ip3pj - 2.0*f_ipj*f_im1 + 22.0*f_ip2pj*f_i - 14.0*f_ipj*f_ip1 + 8.0*f_ip3pj*f_ip1 + 20.0*(f_im1pj*f_im1pj) + 30.0*f_ip1*f_ip1pj - 80.0*f_ip3pj*f_ip1pj - f_i*f_im1pj - 12.0*f_ip2*f_ip1pj + 100.0*f_im1pj*f_ip1pj - 10.0*(f_ip3pj*f_ip3pj) - 7.0*f_i*f_ip3pj + 10.0*f_ipj*f_i - 110.0*(f_ip2pj*f_ip2pj) - 24.0*f_i*f_ip1pj - 6.0*f_ip2pj*f_im1;
        b4 += 90.0*f_ipj*f_ip1pj - 4.5*f_ip1pj*f_im1 - 9.0*f_ip2pj*f_i + 0.75*f_ip2*f_im1pj + 15.0*f_ip2pj*f_im1pj - 15.0*f_ipj*f_ip3pj - 0.75*f_ip3pj*f_im1 - 60.0*(f_ipj*f_ipj) - 3.0*f_ip2pj*f_ip2 + 45.0*f_ipj*f_im1pj - 2.25*f_ip1*f_im1pj - 3.0*f_ipj*f_ip2 - 45.0*f_ip2pj*f_ip3pj + 9.0*f_ip2pj*f_ip1 - 0.75*f_im1pj*f_im1 - 90.0*f_ip2pj*f_ip1pj + 0.75*f_ip2*f_ip3pj + 3.0*f_ipj*f_im1 + 9.0*f_ipj*f_ip1 - 2.25*f_ip3pj*f_ip1 - 7.5*(f_im1pj*f_im1pj) - 13.5*f_ip1*f_ip1pj + 45.0*f_ip3pj*f_ip1pj + 2.25*f_i*f_im1pj + 4.5*f_ip2*f_ip1pj - 45.0*f_im1pj*f_ip1pj + 7.5*(f_ip3pj*f_ip3pj) + 2.25*f_i*f_ip3pj - 9.0*f_ipj*f_i + 60.0*(f_ip2pj*f_ip2pj) + 13.5*f_i*f_ip1pj + 3.0*f_ip2pj*f_im1;
      }
      b0 += -0.5*(f_mjpn*f_mjpn) - 0.5*(f_m3pn*f_m3pn) + f_m1pn*f_m3pn - 0.5*(f_m1pn*f_m1pn) - 0.25*f_m3pn*f_m3mjpn - 0.25*f_m3pn*f_mjpn + f_m2mjpn*f_mjpn + 2.0*f_m1mjpn*f_m3mjpn - 0.5*(f_m2mjpn*f_m2mjpn) + 1.25*f_m1mjpn*f_m3pn - 8.0*f_m2pn*f_m1pn - 4.0*f_m1mjpn*f_mjpn - 2.75*f_m2mjpn*f_n - 4.0*f_m2pn*f_m3pn - 8.0*f_m1mjpn*f_m2mjpn + 0.75*f_m1mjpn*f_n + 10.0*(f_m2pn*f_m2pn) + 3.25*f_m1pn*f_mjpn - 2.75*f_m1pn*f_m3mjpn - 10.25*f_m1pn*f_m1mjpn + 1.25*f_m2pn*f_mjpn + 10.0*(f_m1mjpn*f_m1mjpn) + 0.25*f_n*f_m3mjpn - 0.25*f_n*f_mjpn + 0.75*f_m2pn*f_m3mjpn + 3.25*f_m2mjpn*f_m3pn - 1.75*f_m2pn*f_m1mjpn + 17.75*f_m1pn*f_m2mjpn + 2.0*f_m2pn*f_n - 10.25*f_m2pn*f_m2mjpn;
      b1 += 6.0*(f_mjpn*f_mjpn) + 6.0*(f_m3pn*f_m3pn) - 3.0*f_m3mjpn*f_mjpn + 6.0*f_m1pn*f_m3pn + f_m3pn*f_mjpn + 6.0*f_m2mjpn*f_mjpn + 3.0*f_m2mjpn*f_m3mjpn - 12.0*(f_m2mjpn*f_m2mjpn) - 6.0*f_m1mjpn*f_m3pn + 33.0*f_m2pn*f_m1pn - 9.0*f_m1mjpn*f_mjpn + 6.0*f_m2mjpn*f_n - 9.0*f_m2pn*f_m3pn - 12.0*(f_m1pn*f_m1pn) + f_m1mjpn*f_n - 18.0*(f_m2pn*f_m2pn) - 6.0*f_m1mjpn*f_m3mjpn + 33.0*f_m1mjpn*f_m2mjpn - f_m1pn*f_mjpn + 6.0*f_m1pn*f_m3mjpn - 23.0*f_m1pn*f_m2mjpn - 6.0*f_m2pn*f_mjpn - 18.0*(f_m1mjpn*f_m1mjpn) + 3.0*f_m1pn*f_n - f_n*f_m3mjpn + f_m2pn*f_m3mjpn - 3.0*f_m3pn*f_n - f_m2mjpn*f_m3pn + 23.0*f_m2pn*f_m1mjpn - 6.0*f_m2pn*f_n;
      b2 += -18.0*(f_mjpn*f_mjpn) - 18.0*(f_m3pn*f_m3pn) + 18.0*f_m3mjpn*f_mjpn + 78.0*f_m2pn*f_m3pn - 75.0*(f_m2pn*f_m2pn) + f_m3pn*f_m3mjpn - 0.5*f_m3pn*f_mjpn + 78.0*f_m1mjpn*f_mjpn + 18.0*f_m2mjpn*f_m3mjpn - 75.0*(f_m1mjpn*f_m1mjpn) + 5.0*f_m1mjpn*f_m3pn - 3.0*(f_n*f_n) + 102.0*f_m2mjpn*f_m1mjpn - 60.0*f_m2mjpn*f_mjpn - 5.5*f_m1mjpn*f_n - 60.0*f_m1pn*f_m3pn + 102.0*f_m2pn*f_m1pn + 5.0*f_m2mjpn*f_n - 30.0*(f_m1pn*f_m1pn) - 30.0*f_m1mjpn*f_m3mjpn - 5.5*f_m1pn*f_mjpn - 5.5*f_m2pn*f_m3mjpn + 25.0*f_m2pn*f_m2mjpn + 5.0*f_m2pn*f_mjpn - 30.0*(f_m2mjpn*f_m2mjpn) + 18.0*f_m1pn*f_n - 0.5*f_n*f_m3mjpn + f_n*f_mjpn + 5.0*f_m1pn*f_m3mjpn + 18.0*f_m3pn*f_n - 5.5*f_m2mjpn*f_m3pn + 25.0*f_m1pn*f_m1mjpn - 24.5*f_m1pn*f_m2mjpn - 30.0*f_m2pn*f_n - 3.0*(f_m3mjpn*f_m3mjpn) - 24.5*f_m2pn*f_m1mjpn;
      b3 += 20.0*(f_mjpn*f_mjpn) + 20.0*(f_m3pn*f_m3pn) - 30.0*f_m3mjpn*f_mjpn - 110.0*f_m2pn*f_m3pn + 150.0*(f_m2pn*f_m2pn) - f_m3pn*f_mjpn - 110.0*f_m1mjpn*f_mjpn + 80.0*f_m1mjpn*f_m3mjpn + 150.0*(f_m1mjpn*f_m1mjpn) + 2.0*f_m1mjpn*f_m3pn + 10.0*(f_n*f_n) - 270.0*f_m2pn*f_m1pn + 100.0*f_m2mjpn*f_mjpn + f_m1mjpn*f_n + 100.0*f_m1pn*f_m3pn - 270.0*f_m2mjpn*f_m1mjpn - 2.0*f_m2mjpn*f_n + 120.0*(f_m1pn*f_m1pn) - 70.0*f_m2mjpn*f_m3mjpn - f_m1pn*f_mjpn + f_m2pn*f_m3mjpn + 3.0*f_m1pn*f_m2mjpn + 2.0*f_m2pn*f_mjpn + 120.0*(f_m2mjpn*f_m2mjpn) - 70.0*f_m1pn*f_n + f_n*f_m3mjpn - 2.0*f_m1pn*f_m3mjpn - 30.0*f_m3pn*f_n - f_m2mjpn*f_m3pn - 3.0*f_m2pn*f_m1mjpn + 80.0*f_m2pn*f_n + 10.0*(f_m3mjpn*f_m3mjpn);
      b4 += -7.5*(f_mjpn*f_mjpn) - 7.5*(f_m3pn*f_m3pn) + 15.0*f_m3mjpn*f_mjpn - 45.0*f_m1pn*f_m3pn - 67.5*(f_m2pn*f_m2pn) - 0.75*f_m3pn*f_m3mjpn + 0.75*f_m3pn*f_mjpn + 45.0*f_m1mjpn*f_mjpn + 45.0*f_m2mjpn*f_m3mjpn - 67.5*(f_m1mjpn*f_m1mjpn) - 2.25*f_m1mjpn*f_m3pn - 7.5*(f_n*f_n) + 135.0*f_m2pn*f_m1pn - 45.0*f_m2mjpn*f_mjpn + 2.25*f_m1mjpn*f_n + 45.0*f_m2pn*f_m3pn + 135.0*f_m2mjpn*f_m1mjpn - 2.25*f_m2mjpn*f_n - 67.5*(f_m1pn*f_m1pn) - 45.0*f_m1mjpn*f_m3mjpn + 2.25*f_m1pn*f_mjpn + 2.25*f_m2pn*f_m3mjpn - 6.75*f_m2pn*f_m2mjpn - 2.25*f_m2pn*f_mjpn - 67.5*(f_m2mjpn*f_m2mjpn) + 45.0*f_m1pn*f_n + 0.75*f_n*f_m3mjpn - 0.75*f_n*f_mjpn - 2.25*f_m1pn*f_m3mjpn + 15.0*f_m3pn*f_n + 2.25*f_m2mjpn*f_m3pn - 6.75*f_m1pn*f_m1mjpn + 6.75*f_m1pn*f_m2mjpn - 45.0*f_m2pn*f_n - 7.5*(f_m3mjpn*f_m3mjpn) + 6.75*f_m2pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
  *a6 = b6;
  *a7 = b7;
}
        
int cf_e33_find_extreme_diff3(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e33_compute_coeffs_diff3(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_quadratic_approximation_1_4(a1_0, a1_1, a1_2, a1_3, a1_4, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e33_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e33_compute_coeffs_diff3(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_linear_approximation_1_4(a1_0, a1_1, a1_2, a1_3, a1_4, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e33_find_zero_diff3: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e33_compute_coeffs_diff4(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=4) = sum(a_k*r^k, k=0..7) where y=j+r
     f1(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double b6 = 0.0;
  double b7 = 0.0;
  double f_ipj, f_m3mjpn, f_ip2pj, f_im1, f_mjpn, f_n, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_m2pn, f_im1pj, f_ip3pj, f_ip2, f_ip1, f_ip1pj, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_mjpn = F(-j+n);
      f_n = F(n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip2pj = F(i+2+j);
        f_im1 = F(i-1);
        f_i = F(i);
        f_im1pj = F(i-1+j);
        f_ip3pj = F(i+3+j);
        f_ip2 = F(i+2);
        f_ip1 = F(i+1);
        f_ip1pj = F(i+1+j);
        b0 += 42.0*f_ipj*f_ip1pj + 6.0*(f_ip1pj*f_ip1pj) - 2.0*f_ip2pj*f_i + f_ip2*f_im1pj - 3.0*f_ip2pj*f_im1pj + 3.0*f_ipj*f_ip3pj - 24.0*(f_ipj*f_ipj) + 2.0*f_ip2pj*f_ip2 - 9.0*f_ipj*f_im1pj - 6.0*f_im1pj*f_ip1 - 2.0*f_ipj*f_ip2 - 12.0*f_ipj*f_ip2pj - 3.0*f_ip2pj*f_ip3pj + 24.0*f_ip2pj*f_ip1 - 30.0*f_ip2pj*f_ip1pj - f_ip2*f_ip3pj + 24.0*f_ipj*f_ip1 - 6.0*f_ip3pj*f_ip1 + 6.0*(f_im1pj*f_im1pj) - 36.0*f_ip1*f_ip1pj + 6.0*f_ip3pj*f_ip1pj - f_i*f_im1pj + 6.0*f_im1pj*f_ip1pj + f_i*f_ip3pj + 2.0*f_ipj*f_i + 12.0*(f_ip2pj*f_ip2pj);
        b1 += 48.0*f_ipj*f_ip1pj + 90.0*(f_ip1pj*f_ip1pj) - 4.0*f_ip2pj*f_i - f_ip2*f_im1pj + 36.0*f_ip2pj*f_im1pj - 36.0*f_ipj*f_ip3pj - 2.0*f_ip3pj*f_im1 - 114.0*(f_ipj*f_ipj) - 20.0*f_ip2pj*f_ip2 + 156.0*f_ipj*f_im1pj + 10.0*f_ip1*f_im1pj - 4.0*f_ipj*f_ip2 + 60.0*f_ipj*f_ip2pj - 36.0*f_ip2pj*f_ip3pj + 20.0*f_ip2pj*f_ip1 + 2.0*f_im1pj*f_im1 - 168.0*f_ip2pj*f_ip1pj + 7.0*f_ip2*f_ip3pj - 4.0*f_ipj*f_im1 - 20.0*f_ipj*f_ip1 - 10.0*f_ip3pj*f_ip1 - 36.0*(f_im1pj*f_im1pj) + 60.0*f_ip3pj*f_ip1pj - 11.0*f_i*f_im1pj + 18.0*f_ip2*f_ip1pj - 120.0*f_im1pj*f_ip1pj + 6.0*(f_ip3pj*f_ip3pj) + 5.0*f_i*f_ip3pj + 28.0*f_ipj*f_i + 54.0*(f_ip2pj*f_ip2pj) - 18.0*f_i*f_ip1pj + 4.0*f_ip2pj*f_im1;
        b2 += -480.0*f_ipj*f_ip1pj + 18.0*f_ip1pj*f_im1 - 90.0*(f_ip1pj*f_ip1pj) - 3.0*f_ip2*f_im1pj - 90.0*f_ip2pj*f_im1pj + 90.0*f_ipj*f_ip3pj + 6.0*f_ip3pj*f_im1 + 390.0*(f_ipj*f_ipj) + 30.0*f_ip2pj*f_ip2 - 330.0*f_ipj*f_im1pj + 6.0*f_im1pj*f_ip1 + 18.0*f_ipj*f_ip2 - 60.0*f_ipj*f_ip2pj + 210.0*f_ip2pj*f_ip3pj - 78.0*f_ip2pj*f_ip1 + 600.0*f_ip2pj*f_ip1pj - 9.0*f_ip2*f_ip3pj - 6.0*f_ipj*f_im1 + 66.0*f_ip2pj*f_i - 42.0*f_ipj*f_ip1 + 24.0*f_ip3pj*f_ip1 + 60.0*(f_im1pj*f_im1pj) + 90.0*f_ip1*f_ip1pj - 240.0*f_ip3pj*f_ip1pj - 3.0*f_i*f_im1pj - 36.0*f_ip2*f_ip1pj + 300.0*f_im1pj*f_ip1pj - 30.0*(f_ip3pj*f_ip3pj) - 21.0*f_i*f_ip3pj + 30.0*f_ipj*f_i - 330.0*(f_ip2pj*f_ip2pj) - 72.0*f_i*f_ip1pj - 18.0*f_ip2pj*f_im1;
        b3 += 360.0*f_ipj*f_ip1pj - 18.0*f_ip1pj*f_im1 - 36.0*f_ip2pj*f_i + 3.0*f_ip2*f_im1pj + 60.0*f_ip2pj*f_im1pj - 60.0*f_ipj*f_ip3pj - 3.0*f_ip3pj*f_im1 - 240.0*(f_ipj*f_ipj) - 12.0*f_ip2pj*f_ip2 + 180.0*f_ipj*f_im1pj - 9.0*f_ip1*f_im1pj - 12.0*f_ipj*f_ip2 - 180.0*f_ip2pj*f_ip3pj + 36.0*f_ip2pj*f_ip1 - 3.0*f_im1pj*f_im1 - 360.0*f_ip2pj*f_ip1pj + 3.0*f_ip2*f_ip3pj + 12.0*f_ipj*f_im1 + 36.0*f_ipj*f_ip1 - 9.0*f_ip3pj*f_ip1 - 30.0*(f_im1pj*f_im1pj) - 54.0*f_ip1*f_ip1pj + 180.0*f_ip3pj*f_ip1pj + 9.0*f_i*f_im1pj + 18.0*f_ip2*f_ip1pj - 180.0*f_im1pj*f_ip1pj + 30.0*(f_ip3pj*f_ip3pj) + 9.0*f_i*f_ip3pj - 36.0*f_ipj*f_i + 240.0*(f_ip2pj*f_ip2pj) + 54.0*f_i*f_ip1pj + 12.0*f_ip2pj*f_im1;
      }
      b0 += 6.0*(f_mjpn*f_mjpn) + 6.0*(f_m3pn*f_m3pn) - 3.0*f_m3mjpn*f_mjpn + 6.0*f_m1pn*f_m3pn + f_m3pn*f_mjpn + 6.0*f_m2mjpn*f_mjpn - 6.0*f_m1mjpn*f_m3mjpn - 12.0*(f_m2mjpn*f_m2mjpn) - f_m2mjpn*f_m3pn + 33.0*f_m1mjpn*f_m2mjpn - 9.0*f_m1mjpn*f_mjpn + f_m1mjpn*f_n - 9.0*f_m2pn*f_m3pn - 12.0*(f_m1pn*f_m1pn) + 6.0*f_m2mjpn*f_n - 18.0*(f_m2pn*f_m2pn) + 3.0*f_m2mjpn*f_m3mjpn + 33.0*f_m2pn*f_m1pn - f_m1pn*f_mjpn + 6.0*f_m1pn*f_m3mjpn + 23.0*f_m2pn*f_m1mjpn - 6.0*f_m2pn*f_mjpn - 18.0*(f_m1mjpn*f_m1mjpn) - 6.0*f_m2pn*f_n - f_n*f_m3mjpn + f_m2pn*f_m3mjpn - 3.0*f_m3pn*f_n - 6.0*f_m1mjpn*f_m3pn - 23.0*f_m1pn*f_m2mjpn + 3.0*f_m1pn*f_n;
      b1 += -36.0*(f_mjpn*f_mjpn) - 36.0*(f_m3pn*f_m3pn) + 36.0*f_m3mjpn*f_mjpn - 120.0*f_m1pn*f_m3pn - 60.0*(f_m1pn*f_m1pn) + 2.0*f_m3pn*f_m3mjpn - f_m3pn*f_mjpn + 156.0*f_m1mjpn*f_mjpn + 36.0*f_m2mjpn*f_m3mjpn - 60.0*(f_m2mjpn*f_m2mjpn) - 11.0*f_m2mjpn*f_m3pn - 6.0*(f_n*f_n) + 204.0*f_m2pn*f_m1pn - 120.0*f_m2mjpn*f_mjpn + 10.0*f_m2mjpn*f_n + 156.0*f_m2pn*f_m3pn + 204.0*f_m2mjpn*f_m1mjpn - 11.0*f_m1mjpn*f_n - 150.0*(f_m2pn*f_m2pn) - 60.0*f_m1mjpn*f_m3mjpn - 11.0*f_m1pn*f_mjpn - 11.0*f_m2pn*f_m3mjpn + 50.0*f_m1pn*f_m1mjpn + 10.0*f_m2pn*f_mjpn - 150.0*(f_m1mjpn*f_m1mjpn) - 60.0*f_m2pn*f_n - f_n*f_m3mjpn + 2.0*f_n*f_mjpn + 10.0*f_m1pn*f_m3mjpn + 36.0*f_m3pn*f_n + 10.0*f_m1mjpn*f_m3pn - 49.0*f_m1pn*f_m2mjpn + 50.0*f_m2pn*f_m2mjpn + 36.0*f_m1pn*f_n - 6.0*(f_m3mjpn*f_m3mjpn) - 49.0*f_m2pn*f_m1mjpn;
      b2 += 60.0*(f_mjpn*f_mjpn) + 60.0*(f_m3pn*f_m3pn) - 90.0*f_m3mjpn*f_mjpn + 300.0*f_m1pn*f_m3pn + 360.0*(f_m1pn*f_m1pn) - 3.0*f_m3pn*f_mjpn - 330.0*f_m1mjpn*f_mjpn + 240.0*f_m1mjpn*f_m3mjpn + 360.0*(f_m2mjpn*f_m2mjpn) - 3.0*f_m2mjpn*f_m3pn + 30.0*(f_n*f_n) - 810.0*f_m2mjpn*f_m1mjpn + 300.0*f_m2mjpn*f_mjpn - 6.0*f_m2mjpn*f_n - 330.0*f_m2pn*f_m3pn - 810.0*f_m2pn*f_m1pn + 3.0*f_m1mjpn*f_n + 450.0*(f_m2pn*f_m2pn) - 210.0*f_m2mjpn*f_m3mjpn - 3.0*f_m1pn*f_mjpn + 3.0*f_m2pn*f_m3mjpn - 9.0*f_m2pn*f_m1mjpn + 6.0*f_m2pn*f_mjpn + 450.0*(f_m1mjpn*f_m1mjpn) + 240.0*f_m2pn*f_n + 3.0*f_n*f_m3mjpn - 6.0*f_m1pn*f_m3mjpn - 90.0*f_m3pn*f_n + 6.0*f_m1mjpn*f_m3pn + 9.0*f_m1pn*f_m2mjpn - 210.0*f_m1pn*f_n + 30.0*(f_m3mjpn*f_m3mjpn);
      b3 += -30.0*(f_mjpn*f_mjpn) - 30.0*(f_m3pn*f_m3pn) + 60.0*f_m3mjpn*f_mjpn + 180.0*f_m2pn*f_m3pn - 270.0*(f_m1pn*f_m1pn) - 3.0*f_m3pn*f_m3mjpn + 3.0*f_m3pn*f_mjpn + 180.0*f_m1mjpn*f_mjpn + 180.0*f_m2mjpn*f_m3mjpn - 270.0*(f_m2mjpn*f_m2mjpn) + 9.0*f_m2mjpn*f_m3pn - 30.0*(f_n*f_n) + 540.0*f_m2mjpn*f_m1mjpn - 180.0*f_m2mjpn*f_mjpn - 9.0*f_m2mjpn*f_n - 180.0*f_m1pn*f_m3pn + 540.0*f_m2pn*f_m1pn + 9.0*f_m1mjpn*f_n - 270.0*(f_m2pn*f_m2pn) - 180.0*f_m1mjpn*f_m3mjpn + 9.0*f_m1pn*f_mjpn + 9.0*f_m2pn*f_m3mjpn - 27.0*f_m1pn*f_m1mjpn - 9.0*f_m2pn*f_mjpn - 270.0*(f_m1mjpn*f_m1mjpn) - 180.0*f_m2pn*f_n + 3.0*f_n*f_m3mjpn - 3.0*f_n*f_mjpn - 9.0*f_m1pn*f_m3mjpn + 60.0*f_m3pn*f_n - 9.0*f_m1mjpn*f_m3pn + 27.0*f_m1pn*f_m2mjpn - 27.0*f_m2pn*f_m2mjpn + 180.0*f_m1pn*f_n - 30.0*(f_m3mjpn*f_m3mjpn) + 27.0*f_m2pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
  *a6 = b6;
  *a7 = b7;
}
        
int cf_e33_find_extreme_diff4(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e33_compute_coeffs_diff4(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_quadratic_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e33_find_zero_diff4(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e33_compute_coeffs_diff4(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_linear_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e33_find_zero_diff4: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e33_compute_coeffs_diff5(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=5) = sum(a_k*r^k, k=0..7) where y=j+r
     f1(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double b6 = 0.0;
  double b7 = 0.0;
  double f_ipj, f_m3mjpn, f_ip2pj, f_im1, f_mjpn, f_n, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_m2pn, f_im1pj, f_ip3pj, f_ip2, f_ip1, f_ip1pj, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_mjpn = F(-j+n);
      f_n = F(n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip2pj = F(i+2+j);
        f_im1 = F(i-1);
        f_i = F(i);
        f_im1pj = F(i-1+j);
        f_ip3pj = F(i+3+j);
        f_ip2 = F(i+2);
        f_ip1 = F(i+1);
        f_ip1pj = F(i+1+j);
        b0 += 48.0*f_ipj*f_ip1pj + 90.0*(f_ip1pj*f_ip1pj) - 4.0*f_ip2pj*f_i - f_ip2*f_im1pj + 36.0*f_ip2pj*f_im1pj - 36.0*f_ipj*f_ip3pj - 2.0*f_ip3pj*f_im1 - 114.0*(f_ipj*f_ipj) - 20.0*f_ip2pj*f_ip2 + 156.0*f_ipj*f_im1pj + 10.0*f_ip1*f_im1pj - 4.0*f_ipj*f_ip2 + 60.0*f_ipj*f_ip2pj - 36.0*f_ip2pj*f_ip3pj + 20.0*f_ip2pj*f_ip1 + 2.0*f_im1pj*f_im1 - 168.0*f_ip2pj*f_ip1pj + 7.0*f_ip2*f_ip3pj - 4.0*f_ipj*f_im1 - 20.0*f_ipj*f_ip1 - 10.0*f_ip3pj*f_ip1 - 36.0*(f_im1pj*f_im1pj) + 60.0*f_ip3pj*f_ip1pj - 11.0*f_i*f_im1pj + 18.0*f_ip2*f_ip1pj - 120.0*f_im1pj*f_ip1pj + 6.0*(f_ip3pj*f_ip3pj) + 5.0*f_i*f_ip3pj + 28.0*f_ipj*f_i + 54.0*(f_ip2pj*f_ip2pj) - 18.0*f_i*f_ip1pj + 4.0*f_ip2pj*f_im1;
        b1 += -960.0*f_ipj*f_ip1pj + 36.0*f_ip1pj*f_im1 - 180.0*(f_ip1pj*f_ip1pj) - 6.0*f_ip2*f_im1pj - 180.0*f_ip2pj*f_im1pj + 180.0*f_ipj*f_ip3pj + 12.0*f_ip3pj*f_im1 + 780.0*(f_ipj*f_ipj) + 60.0*f_ip2pj*f_ip2 - 660.0*f_ipj*f_im1pj + 12.0*f_im1pj*f_ip1 + 36.0*f_ipj*f_ip2 - 120.0*f_ipj*f_ip2pj + 420.0*f_ip2pj*f_ip3pj - 156.0*f_ip2pj*f_ip1 + 1200.0*f_ip2pj*f_ip1pj - 18.0*f_ip2*f_ip3pj - 12.0*f_ipj*f_im1 + 132.0*f_ip2pj*f_i - 84.0*f_ipj*f_ip1 + 48.0*f_ip3pj*f_ip1 + 120.0*(f_im1pj*f_im1pj) + 180.0*f_ip1*f_ip1pj - 480.0*f_ip3pj*f_ip1pj - 6.0*f_i*f_im1pj - 72.0*f_ip2*f_ip1pj + 600.0*f_im1pj*f_ip1pj - 60.0*(f_ip3pj*f_ip3pj) - 42.0*f_i*f_ip3pj + 60.0*f_ipj*f_i - 660.0*(f_ip2pj*f_ip2pj) - 144.0*f_i*f_ip1pj - 36.0*f_ip2pj*f_im1;
        b2 += 1080.0*f_ipj*f_ip1pj - 54.0*f_ip1pj*f_im1 - 108.0*f_ip2pj*f_i + 9.0*f_ip2*f_im1pj + 180.0*f_ip2pj*f_im1pj - 180.0*f_ipj*f_ip3pj - 9.0*f_ip3pj*f_im1 - 720.0*(f_ipj*f_ipj) - 36.0*f_ip2pj*f_ip2 + 540.0*f_ipj*f_im1pj - 27.0*f_ip1*f_im1pj - 36.0*f_ipj*f_ip2 - 540.0*f_ip2pj*f_ip3pj + 108.0*f_ip2pj*f_ip1 - 9.0*f_im1pj*f_im1 - 1080.0*f_ip2pj*f_ip1pj + 9.0*f_ip2*f_ip3pj + 36.0*f_ipj*f_im1 + 108.0*f_ipj*f_ip1 - 27.0*f_ip3pj*f_ip1 - 90.0*(f_im1pj*f_im1pj) - 162.0*f_ip1*f_ip1pj + 540.0*f_ip3pj*f_ip1pj + 27.0*f_i*f_im1pj + 54.0*f_ip2*f_ip1pj - 540.0*f_im1pj*f_ip1pj + 90.0*(f_ip3pj*f_ip3pj) + 27.0*f_i*f_ip3pj - 108.0*f_ipj*f_i + 720.0*(f_ip2pj*f_ip2pj) + 162.0*f_i*f_ip1pj + 36.0*f_ip2pj*f_im1;
      }
      b0 += -36.0*(f_mjpn*f_mjpn) - 36.0*(f_m3pn*f_m3pn) + 36.0*f_m3mjpn*f_mjpn + 156.0*f_m2pn*f_m3pn - 150.0*(f_m2pn*f_m2pn) + 2.0*f_m3pn*f_m3mjpn - f_m3pn*f_mjpn + 156.0*f_m1mjpn*f_mjpn + 36.0*f_m2mjpn*f_m3mjpn - 150.0*(f_m1mjpn*f_m1mjpn) + 10.0*f_m1mjpn*f_m3pn - 6.0*(f_n*f_n) + 204.0*f_m2mjpn*f_m1mjpn - 120.0*f_m2mjpn*f_mjpn - 11.0*f_m1mjpn*f_n - 120.0*f_m1pn*f_m3pn + 204.0*f_m2pn*f_m1pn + 10.0*f_m2mjpn*f_n - 60.0*(f_m1pn*f_m1pn) - 60.0*f_m1mjpn*f_m3mjpn - 11.0*f_m1pn*f_mjpn - 11.0*f_m2pn*f_m3mjpn - 49.0*f_m1pn*f_m2mjpn + 10.0*f_m2pn*f_mjpn - 60.0*(f_m2mjpn*f_m2mjpn) + 36.0*f_m1pn*f_n - f_n*f_m3mjpn + 2.0*f_n*f_mjpn + 10.0*f_m1pn*f_m3mjpn + 36.0*f_m3pn*f_n - 11.0*f_m2mjpn*f_m3pn + 50.0*f_m2pn*f_m2mjpn + 50.0*f_m1pn*f_m1mjpn - 60.0*f_m2pn*f_n - 6.0*(f_m3mjpn*f_m3mjpn) - 49.0*f_m2pn*f_m1mjpn;
      b1 += 120.0*(f_mjpn*f_mjpn) + 120.0*(f_m3pn*f_m3pn) - 180.0*f_m3mjpn*f_mjpn - 660.0*f_m2pn*f_m3pn + 900.0*(f_m2pn*f_m2pn) - 6.0*f_m3pn*f_mjpn - 660.0*f_m1mjpn*f_mjpn + 480.0*f_m1mjpn*f_m3mjpn + 900.0*(f_m1mjpn*f_m1mjpn) + 12.0*f_m1mjpn*f_m3pn + 60.0*(f_n*f_n) - 1620.0*f_m2pn*f_m1pn + 600.0*f_m2mjpn*f_mjpn + 6.0*f_m1mjpn*f_n + 600.0*f_m1pn*f_m3pn - 1620.0*f_m2mjpn*f_m1mjpn - 12.0*f_m2mjpn*f_n + 720.0*(f_m1pn*f_m1pn) - 420.0*f_m2mjpn*f_m3mjpn - 6.0*f_m1pn*f_mjpn + 6.0*f_m2pn*f_m3mjpn + 18.0*f_m1pn*f_m2mjpn + 12.0*f_m2pn*f_mjpn + 720.0*(f_m2mjpn*f_m2mjpn) - 420.0*f_m1pn*f_n + 6.0*f_n*f_m3mjpn - 12.0*f_m1pn*f_m3mjpn - 180.0*f_m3pn*f_n - 6.0*f_m2mjpn*f_m3pn - 18.0*f_m2pn*f_m1mjpn + 480.0*f_m2pn*f_n + 60.0*(f_m3mjpn*f_m3mjpn);
      b2 += -90.0*(f_mjpn*f_mjpn) - 90.0*(f_m3pn*f_m3pn) + 180.0*f_m3mjpn*f_mjpn - 540.0*f_m1pn*f_m3pn - 810.0*(f_m2pn*f_m2pn) - 9.0*f_m3pn*f_m3mjpn + 9.0*f_m3pn*f_mjpn + 540.0*f_m1mjpn*f_mjpn + 540.0*f_m2mjpn*f_m3mjpn - 810.0*(f_m1mjpn*f_m1mjpn) - 27.0*f_m1mjpn*f_m3pn - 90.0*(f_n*f_n) + 1620.0*f_m2pn*f_m1pn - 540.0*f_m2mjpn*f_mjpn + 27.0*f_m1mjpn*f_n + 540.0*f_m2pn*f_m3pn + 1620.0*f_m2mjpn*f_m1mjpn - 27.0*f_m2mjpn*f_n - 810.0*(f_m1pn*f_m1pn) - 540.0*f_m1mjpn*f_m3mjpn + 27.0*f_m1pn*f_mjpn + 27.0*f_m2pn*f_m3mjpn + 81.0*f_m1pn*f_m2mjpn - 27.0*f_m2pn*f_mjpn - 810.0*(f_m2mjpn*f_m2mjpn) + 540.0*f_m1pn*f_n + 9.0*f_n*f_m3mjpn - 9.0*f_n*f_mjpn - 27.0*f_m1pn*f_m3mjpn + 180.0*f_m3pn*f_n + 27.0*f_m2mjpn*f_m3pn - 81.0*f_m2pn*f_m2mjpn - 81.0*f_m1pn*f_m1mjpn - 540.0*f_m2pn*f_n - 90.0*(f_m3mjpn*f_m3mjpn) + 81.0*f_m2pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
  *a6 = b6;
  *a7 = b7;
}
        
int cf_e33_find_extreme_diff5(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e33_compute_coeffs_diff5(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_quadratic_approximation_1_2(a1_0, a1_1, a1_2, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e33_find_zero_diff5(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e33_compute_coeffs_diff5(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_linear_approximation_1_2(a1_0, a1_1, a1_2, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e33_find_zero_diff5: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e33_compute_coeffs_diff6(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=6) = sum(a_k*r^k, k=0..7) where y=j+r
     f1(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double b6 = 0.0;
  double b7 = 0.0;
  double f_ipj, f_m3mjpn, f_im1, f_mjpn, f_n, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip3pj, f_ip2, f_ip1, f_ip1pj, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_mjpn = F(-j+n);
      f_n = F(n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_im1 = F(i-1);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip3pj = F(i+3+j);
        f_ip2 = F(i+2);
        f_ip1 = F(i+1);
        f_ip1pj = F(i+1+j);
        b0 += -960.0*f_ipj*f_ip1pj + 36.0*f_ip1pj*f_im1 - 180.0*(f_ip1pj*f_ip1pj) - 6.0*f_ip2*f_im1pj - 180.0*f_ip2pj*f_im1pj + 180.0*f_ipj*f_ip3pj + 12.0*f_ip3pj*f_im1 + 780.0*(f_ipj*f_ipj) + 60.0*f_ip2pj*f_ip2 - 660.0*f_ipj*f_im1pj + 12.0*f_im1pj*f_ip1 + 36.0*f_ipj*f_ip2 - 120.0*f_ipj*f_ip2pj + 420.0*f_ip2pj*f_ip3pj - 156.0*f_ip2pj*f_ip1 + 1200.0*f_ip2pj*f_ip1pj - 18.0*f_ip2*f_ip3pj - 12.0*f_ipj*f_im1 + 132.0*f_ip2pj*f_i - 84.0*f_ipj*f_ip1 + 48.0*f_ip3pj*f_ip1 + 120.0*(f_im1pj*f_im1pj) + 180.0*f_ip1*f_ip1pj - 480.0*f_ip3pj*f_ip1pj - 6.0*f_i*f_im1pj - 72.0*f_ip2*f_ip1pj + 600.0*f_im1pj*f_ip1pj - 60.0*(f_ip3pj*f_ip3pj) - 42.0*f_i*f_ip3pj + 60.0*f_ipj*f_i - 660.0*(f_ip2pj*f_ip2pj) - 144.0*f_i*f_ip1pj - 36.0*f_ip2pj*f_im1;
        b1 += 2160.0*f_ipj*f_ip1pj - 108.0*f_ip1pj*f_im1 - 216.0*f_ip2pj*f_i + 18.0*f_ip2*f_im1pj + 360.0*f_ip2pj*f_im1pj - 360.0*f_ipj*f_ip3pj - 18.0*f_ip3pj*f_im1 - 1440.0*(f_ipj*f_ipj) - 72.0*f_ip2pj*f_ip2 + 1080.0*f_ipj*f_im1pj - 54.0*f_ip1*f_im1pj - 72.0*f_ipj*f_ip2 - 1080.0*f_ip2pj*f_ip3pj + 216.0*f_ip2pj*f_ip1 - 18.0*f_im1pj*f_im1 - 2160.0*f_ip2pj*f_ip1pj + 18.0*f_ip2*f_ip3pj + 72.0*f_ipj*f_im1 + 216.0*f_ipj*f_ip1 - 54.0*f_ip3pj*f_ip1 - 180.0*(f_im1pj*f_im1pj) - 324.0*f_ip1*f_ip1pj + 1080.0*f_ip3pj*f_ip1pj + 54.0*f_i*f_im1pj + 108.0*f_ip2*f_ip1pj - 1080.0*f_im1pj*f_ip1pj + 180.0*(f_ip3pj*f_ip3pj) + 54.0*f_i*f_ip3pj - 216.0*f_ipj*f_i + 1440.0*(f_ip2pj*f_ip2pj) + 324.0*f_i*f_ip1pj + 72.0*f_ip2pj*f_im1;
      }
      b0 += 120.0*(f_mjpn*f_mjpn) + 120.0*(f_m3pn*f_m3pn) - 180.0*f_m3mjpn*f_mjpn + 600.0*f_m1pn*f_m3pn + 720.0*(f_m1pn*f_m1pn) - 6.0*f_m3pn*f_mjpn - 660.0*f_m1mjpn*f_mjpn + 480.0*f_m1mjpn*f_m3mjpn + 720.0*(f_m2mjpn*f_m2mjpn) - 6.0*f_m2mjpn*f_m3pn + 60.0*(f_n*f_n) - 1620.0*f_m2mjpn*f_m1mjpn + 600.0*f_m2mjpn*f_mjpn - 12.0*f_m2mjpn*f_n - 660.0*f_m2pn*f_m3pn - 1620.0*f_m2pn*f_m1pn + 6.0*f_m1mjpn*f_n + 900.0*(f_m2pn*f_m2pn) - 420.0*f_m2mjpn*f_m3mjpn - 6.0*f_m1pn*f_mjpn + 6.0*f_m2pn*f_m3mjpn - 18.0*f_m2pn*f_m1mjpn + 12.0*f_m2pn*f_mjpn + 900.0*(f_m1mjpn*f_m1mjpn) + 480.0*f_m2pn*f_n + 6.0*f_n*f_m3mjpn - 12.0*f_m1pn*f_m3mjpn - 180.0*f_m3pn*f_n + 12.0*f_m1mjpn*f_m3pn + 18.0*f_m1pn*f_m2mjpn - 420.0*f_m1pn*f_n + 60.0*(f_m3mjpn*f_m3mjpn);
      b1 += -180.0*(f_mjpn*f_mjpn) - 180.0*(f_m3pn*f_m3pn) + 360.0*f_m3mjpn*f_mjpn + 1080.0*f_m2pn*f_m3pn - 1620.0*(f_m1pn*f_m1pn) - 18.0*f_m3pn*f_m3mjpn + 18.0*f_m3pn*f_mjpn + 1080.0*f_m1mjpn*f_mjpn + 1080.0*f_m2mjpn*f_m3mjpn - 1620.0*(f_m2mjpn*f_m2mjpn) + 54.0*f_m2mjpn*f_m3pn - 180.0*(f_n*f_n) + 3240.0*f_m2mjpn*f_m1mjpn - 1080.0*f_m2mjpn*f_mjpn - 54.0*f_m2mjpn*f_n - 1080.0*f_m1pn*f_m3pn + 3240.0*f_m2pn*f_m1pn + 54.0*f_m1mjpn*f_n - 1620.0*(f_m2pn*f_m2pn) - 1080.0*f_m1mjpn*f_m3mjpn + 54.0*f_m1pn*f_mjpn + 54.0*f_m2pn*f_m3mjpn - 162.0*f_m2pn*f_m2mjpn - 54.0*f_m2pn*f_mjpn - 1620.0*(f_m1mjpn*f_m1mjpn) - 1080.0*f_m2pn*f_n + 18.0*f_n*f_m3mjpn - 18.0*f_n*f_mjpn - 54.0*f_m1pn*f_m3mjpn + 360.0*f_m3pn*f_n - 54.0*f_m1mjpn*f_m3pn - 162.0*f_m1pn*f_m1mjpn + 162.0*f_m1pn*f_m2mjpn + 1080.0*f_m1pn*f_n - 180.0*(f_m3mjpn*f_m3mjpn) + 162.0*f_m2pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
  *a6 = b6;
  *a7 = b7;
}
        
int cf_e33_find_extreme_diff6(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a2_2 = 0.0;
  double a2_3 = 0.0;
  double a2_4 = 0.0;
  double a2_5 = 0.0;
  double a2_6 = 0.0;
  double a2_7 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  double a3_2 = 0.0;
  double a3_3 = 0.0;
  double a3_4 = 0.0;
  double a3_5 = 0.0;
  double a3_6 = 0.0;
  double a3_7 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a1_2 = a2_2;
    a1_3 = a2_3;
    a1_4 = a2_4;
    a1_5 = a2_5;
    a1_6 = a2_6;
    a1_7 = a2_7;
    a2_0 = a3_0;
    a2_1 = a3_1;
    a2_2 = a3_2;
    a2_3 = a3_3;
    a2_4 = a3_4;
    a2_5 = a3_5;
    a2_6 = a3_6;
    a2_7 = a3_7;
    cf_e33_compute_coeffs_diff6(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3, &a3_4, &a3_5, &a3_6, &a3_7);
    count ++;
    if (count<3)
      continue;
    cf_quadratic_approximation_3_1(a1_0, a1_1, a2_0, a2_1, a3_0, a3_1, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e33_find_zero_diff6(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e33_compute_coeffs_diff6(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5, &a1_6, &a1_7);
    cf_linear_approximation_1_1(a1_0, a1_1, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e33_find_zero_diff6: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e33_compute_coeffs_diff7(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=7) = sum(a_k*r^k, k=0..7) where y=j+r
     f1(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double b6 = 0.0;
  double b7 = 0.0;
  double f_ipj, f_m3mjpn, f_im1, f_mjpn, f_n, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip3pj, f_ip2, f_ip1, f_ip1pj, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_mjpn = F(-j+n);
      f_n = F(n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_im1 = F(i-1);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip3pj = F(i+3+j);
        f_ip2 = F(i+2);
        f_ip1 = F(i+1);
        f_ip1pj = F(i+1+j);
        b0 += 2160.0*f_ipj*f_ip1pj - 108.0*f_ip1pj*f_im1 - 216.0*f_ip2pj*f_i + 18.0*f_ip2*f_im1pj + 360.0*f_ip2pj*f_im1pj - 360.0*f_ipj*f_ip3pj - 18.0*f_ip3pj*f_im1 - 1440.0*(f_ipj*f_ipj) - 72.0*f_ip2pj*f_ip2 + 1080.0*f_ipj*f_im1pj - 54.0*f_ip1*f_im1pj - 72.0*f_ipj*f_ip2 - 1080.0*f_ip2pj*f_ip3pj + 216.0*f_ip2pj*f_ip1 - 18.0*f_im1pj*f_im1 - 2160.0*f_ip2pj*f_ip1pj + 18.0*f_ip2*f_ip3pj + 72.0*f_ipj*f_im1 + 216.0*f_ipj*f_ip1 - 54.0*f_ip3pj*f_ip1 - 180.0*(f_im1pj*f_im1pj) - 324.0*f_ip1*f_ip1pj + 1080.0*f_ip3pj*f_ip1pj + 54.0*f_i*f_im1pj + 108.0*f_ip2*f_ip1pj - 1080.0*f_im1pj*f_ip1pj + 180.0*(f_ip3pj*f_ip3pj) + 54.0*f_i*f_ip3pj - 216.0*f_ipj*f_i + 1440.0*(f_ip2pj*f_ip2pj) + 324.0*f_i*f_ip1pj + 72.0*f_ip2pj*f_im1;
      }
      b0 += -180.0*(f_mjpn*f_mjpn) - 180.0*(f_m3pn*f_m3pn) + 360.0*f_m3mjpn*f_mjpn - 1080.0*f_m1pn*f_m3pn - 1620.0*(f_m2pn*f_m2pn) - 18.0*f_m3pn*f_m3mjpn + 18.0*f_m3pn*f_mjpn + 1080.0*f_m1mjpn*f_mjpn + 1080.0*f_m2mjpn*f_m3mjpn - 1620.0*(f_m1mjpn*f_m1mjpn) - 54.0*f_m1mjpn*f_m3pn - 180.0*(f_n*f_n) + 3240.0*f_m2pn*f_m1pn - 1080.0*f_m2mjpn*f_mjpn + 54.0*f_m1mjpn*f_n + 1080.0*f_m2pn*f_m3pn + 3240.0*f_m2mjpn*f_m1mjpn - 54.0*f_m2mjpn*f_n - 1620.0*(f_m1pn*f_m1pn) - 1080.0*f_m1mjpn*f_m3mjpn + 54.0*f_m1pn*f_mjpn + 54.0*f_m2pn*f_m3mjpn - 162.0*f_m1pn*f_m1mjpn - 54.0*f_m2pn*f_mjpn - 1620.0*(f_m2mjpn*f_m2mjpn) + 1080.0*f_m1pn*f_n + 18.0*f_n*f_m3mjpn - 18.0*f_n*f_mjpn - 54.0*f_m1pn*f_m3mjpn + 360.0*f_m3pn*f_n + 54.0*f_m2mjpn*f_m3pn + 162.0*f_m1pn*f_m2mjpn - 162.0*f_m2pn*f_m2mjpn - 1080.0*f_m2pn*f_n - 180.0*(f_m3mjpn*f_m3mjpn) + 162.0*f_m2pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
  *a6 = b6;
  *a7 = b7;
}
        
int cf_e33_find_extreme_diff7(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a2_2 = 0.0;
  double a2_3 = 0.0;
  double a2_4 = 0.0;
  double a2_5 = 0.0;
  double a2_6 = 0.0;
  double a2_7 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  double a3_2 = 0.0;
  double a3_3 = 0.0;
  double a3_4 = 0.0;
  double a3_5 = 0.0;
  double a3_6 = 0.0;
  double a3_7 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a1_2 = a2_2;
    a1_3 = a2_3;
    a1_4 = a2_4;
    a1_5 = a2_5;
    a1_6 = a2_6;
    a1_7 = a2_7;
    a2_0 = a3_0;
    a2_1 = a3_1;
    a2_2 = a3_2;
    a2_3 = a3_3;
    a2_4 = a3_4;
    a2_5 = a3_5;
    a2_6 = a3_6;
    a2_7 = a3_7;
    cf_e33_compute_coeffs_diff7(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3, &a3_4, &a3_5, &a3_6, &a3_7);
    count ++;
    if (count<3)
      continue;
    cf_quadratic_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e33_find_zero_diff7(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a1_6 = 0.0;
  double a1_7 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a2_2 = 0.0;
  double a2_3 = 0.0;
  double a2_4 = 0.0;
  double a2_5 = 0.0;
  double a2_6 = 0.0;
  double a2_7 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  double a3_2 = 0.0;
  double a3_3 = 0.0;
  double a3_4 = 0.0;
  double a3_5 = 0.0;
  double a3_6 = 0.0;
  double a3_7 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a1_2 = a2_2;
    a1_3 = a2_3;
    a1_4 = a2_4;
    a1_5 = a2_5;
    a1_6 = a2_6;
    a1_7 = a2_7;
    a2_0 = a3_0;
    a2_1 = a3_1;
    a2_2 = a3_2;
    a2_3 = a3_3;
    a2_4 = a3_4;
    a2_5 = a3_5;
    a2_6 = a3_6;
    a2_7 = a3_7;
    cf_e33_compute_coeffs_diff7(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3, &a3_4, &a3_5, &a3_6, &a3_7);
    count ++;
    if (count<3)
      continue;
    cf_linear_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1);
    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e33_find_zero_diff7: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
void cf_e33_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order) = sum(a_k*r^k, k=0..7) where y=j+r
     f1(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s**3 + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  switch (order)
  {
    case 0: cf_e33_compute_coeffs_diff0(j, fm, n, m, a0, a1, a2, a3, a4, a5, a6, a7); break;
    case 1: cf_e33_compute_coeffs_diff1(j, fm, n, m, a0, a1, a2, a3, a4, a5, a6, a7); break;
    case 2: cf_e33_compute_coeffs_diff2(j, fm, n, m, a0, a1, a2, a3, a4, a5, a6, a7); break;
    case 3: cf_e33_compute_coeffs_diff3(j, fm, n, m, a0, a1, a2, a3, a4, a5, a6, a7); break;
    case 4: cf_e33_compute_coeffs_diff4(j, fm, n, m, a0, a1, a2, a3, a4, a5, a6, a7); break;
    case 5: cf_e33_compute_coeffs_diff5(j, fm, n, m, a0, a1, a2, a3, a4, a5, a6, a7); break;
    case 6: cf_e33_compute_coeffs_diff6(j, fm, n, m, a0, a1, a2, a3, a4, a5, a6, a7); break;
    case 7: cf_e33_compute_coeffs_diff7(j, fm, n, m, a0, a1, a2, a3, a4, a5, a6, a7); break;
    default:
      *a0 = 0.0;
      *a1 = 0.0;
      *a2 = 0.0;
      *a3 = 0.0;
      *a4 = 0.0;
      *a5 = 0.0;
      *a6 = 0.0;
      *a7 = 0.0;
  }
}
        
int cf_e33_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  switch (order)
  {
    case 0: return cf_e33_find_extreme_diff0(j0, j1, fm, n, m, result);
    case 1: return cf_e33_find_extreme_diff1(j0, j1, fm, n, m, result);
    case 2: return cf_e33_find_extreme_diff2(j0, j1, fm, n, m, result);
    case 3: return cf_e33_find_extreme_diff3(j0, j1, fm, n, m, result);
    case 4: return cf_e33_find_extreme_diff4(j0, j1, fm, n, m, result);
    case 5: return cf_e33_find_extreme_diff5(j0, j1, fm, n, m, result);
    case 6: return cf_e33_find_extreme_diff6(j0, j1, fm, n, m, result);
    case 7: return cf_e33_find_extreme_diff7(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
int cf_e33_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  switch (order)
  {
    case 0: return cf_e33_find_zero_diff0(j0, j1, fm, n, m, result);
    case 1: return cf_e33_find_zero_diff1(j0, j1, fm, n, m, result);
    case 2: return cf_e33_find_zero_diff2(j0, j1, fm, n, m, result);
    case 3: return cf_e33_find_zero_diff3(j0, j1, fm, n, m, result);
    case 4: return cf_e33_find_zero_diff4(j0, j1, fm, n, m, result);
    case 5: return cf_e33_find_zero_diff5(j0, j1, fm, n, m, result);
    case 6: return cf_e33_find_zero_diff6(j0, j1, fm, n, m, result);
    case 7: return cf_e33_find_zero_diff7(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
double cf_e33_evaluate(double y, double *fm, int n, int m, int order)
{
  double a0 = 0.0;
  double a1 = 0.0;
  double a2 = 0.0;
  double a3 = 0.0;
  double a4 = 0.0;
  double a5 = 0.0;
  double a6 = 0.0;
  double a7 = 0.0;
  /*
  int j = floor((y<0?-y:y));
  double r = (y<0?-y:y) - j;
  */
  int j = floor (y);
  double r = y - j;
  cf_e33_compute_coeffs(j, fm, n, m, order, &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);
  return a0+(a1+(a2+(a3+(a4+(a5+(a6+(a7)*r)*r)*r)*r)*r)*r)*r;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double cf_e33_f1_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  switch (order)
  {
    case 0: return F(i) + (0.5*(F(i+1)) - 0.5*(F(i-1)) + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)) + (0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s)*s)*s;
    case 1: return 0.5*(F(i+1)) - 0.5*(F(i-1)) + (-(F(i+2)) - 5.0*(F(i)) + 4.0*(F(i+1)) + 2.0*(F(i-1)) + (1.5*(F(i+2)) + 4.5*(F(i)) - 4.5*(F(i+1)) - 1.5*(F(i-1)))*s)*s;
    case 2: return -(F(i+2)) - 5.0*(F(i)) + 4.0*(F(i+1)) + 2.0*(F(i-1)) + (3.0*(F(i+2)) + 9.0*(F(i)) - 9.0*(F(i+1)) - 3.0*(F(i-1)))*s;
  }
  return 0.0;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double cf_e33_f2_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  switch (order)
  {
    case 0: return F(i) + (0.5*(F(i+1)) - 0.5*(F(i-1)) + (-0.5*(F(i+2)) - 2.5*(F(i)) + 2.0*(F(i+1)) + (F(i-1)) + (0.5*(F(i+2)) + 1.5*(F(i)) - 1.5*(F(i+1)) - 0.5*(F(i-1)))*s)*s)*s;
    case 1: return 0.5*(F(i+1)) - 0.5*(F(i-1)) + (-(F(i+2)) - 5.0*(F(i)) + 4.0*(F(i+1)) + 2.0*(F(i-1)) + (1.5*(F(i+2)) + 4.5*(F(i)) - 4.5*(F(i+1)) - 1.5*(F(i-1)))*s)*s;
    case 2: return -(F(i+2)) - 5.0*(F(i)) + 4.0*(F(i+1)) + 2.0*(F(i-1)) + (3.0*(F(i+2)) + 9.0*(F(i)) - 9.0*(F(i+1)) - 3.0*(F(i-1)))*s;
  }
  return 0.0;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e22_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5)
{
  /* int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y) = sum(a_k*r^k, k=0..5) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double f_ipj, f_m3mjpn, f_im1, f_ip1pj, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_im1 = F(i-1);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip1 = F(i+1);
        b0 += 0.3833333333333333*f_ipj*f_ip1pj + 0.2583333333333333*(f_ip1pj*f_ip1pj) + 0.2583333333333333*(f_ip1*f_ip1) + 0.06666666666666667*f_ip1*f_im1pj - 0.06666666666666667*f_ip1*f_im1 + 0.06666666666666667*f_ip1pj*f_im1 + 0.008333333333333333*(f_im1*f_im1) + 0.008333333333333333*(f_im1pj*f_im1pj) + 0.3833333333333333*f_i*f_ip1 - 1.066666666666667*f_ipj*f_i - 0.5166666666666667*f_ip1*f_ip1pj + 0.1166666666666667*f_ipj*f_im1 - 0.3833333333333333*f_ipj*f_ip1 + 0.1166666666666667*f_i*f_im1pj - 0.3833333333333333*f_i*f_ip1pj - 0.01666666666666667*f_im1pj*f_im1 + 0.5333333333333333*(f_i*f_i) - 0.1166666666666667*f_i*f_im1 + 0.5333333333333333*(f_ipj*f_ipj) - 0.06666666666666667*f_im1pj*f_ip1pj - 0.1166666666666667*f_ipj*f_im1pj;
        b1 += 0.1666666666666667*f_ip1pj*f_im1 - 0.1666666666666667*f_ip1*f_im1pj + 1.166666666666667*f_ipj*f_ip1 + (f_ip1pj*f_ip1pj) + f_ipj*f_i - f_ip1*f_ip1pj + 0.1666666666666667*f_i*f_im1pj - 1.166666666666667*f_i*f_ip1pj - 0.1666666666666667*f_ipj*f_im1 - (f_ipj*f_ipj);
        b2 += -f_ipj*f_ip1pj + 0.08333333333333333*f_im1*f_ip1pj + 0.08333333333333333*f_ip1*f_im1pj - 0.6666666666666667*f_ipj*f_ip1 - 0.5*f_ip2pj*f_ip1 + 0.08333333333333333*f_im1pj*f_im1 + 1.333333333333333*f_ipj*f_i + 0.5*f_ip2pj*f_ip1pj - 0.6666666666666667*f_i*f_im1pj - 0.6666666666666667*f_i*f_ip1pj + 1.083333333333333*f_ip1*f_ip1pj - 0.1666666666666667*f_ipj*f_im1 + 0.5*f_ipj*f_im1pj;
        b3 += -0.25*f_im1*f_ip1pj - 0.1666666666666667*f_ipj*f_ip2pj + 0.08333333333333333*f_ip1*f_im1pj + 0.1666666666666667*f_im1pj*f_ip1pj - 0.75*(f_ip1pj*f_ip1pj) - 0.08333333333333333*(f_im1pj*f_im1pj) - 0.25*f_ipj*f_ip1 - 0.3333333333333333*f_ip2pj*f_i - 0.08333333333333333*f_ip2pj*f_ip1 + 0.25*f_ip1*f_ip1pj - f_ipj*f_i + 0.3333333333333333*f_ip2pj*f_ip1pj + 0.3333333333333333*f_i*f_im1pj + f_i*f_ip1pj + 0.08333333333333333*(f_ip2pj*f_ip2pj) + 0.25*f_ipj*f_im1 - 0.08333333333333333*f_im1pj*f_im1 + 0.08333333333333333*f_ip2pj*f_im1 + 0.75*(f_ipj*f_ipj) - 0.3333333333333333*f_ipj*f_im1pj;
        b4 += 0.5*f_ipj*f_ip1pj - 0.08333333333333333*f_im1pj*f_ip1 + 0.25*f_ipj*f_ip1 + 0.125*(f_im1pj*f_im1pj) - 0.08333333333333333*f_ip2pj*f_i - 0.125*(f_ip1pj*f_ip1pj) + 0.08333333333333333*f_ip2pj*f_ip1 - 0.25*f_ip1*f_ip1pj - 0.25*f_ipj*f_i - 0.25*f_ip2pj*f_ip1pj + 0.08333333333333333*f_i*f_im1pj + 0.25*f_i*f_ip1pj + 0.125*(f_ip2pj*f_ip2pj) - 0.125*(f_ipj*f_ipj) - 0.25*f_ipj*f_im1pj;
        b5 += 0.05*f_im1*f_ip1pj + 0.1*f_ipj*f_ip2pj + 0.01666666666666667*f_ip1*f_im1pj - 0.1*f_im1pj*f_ip1pj + 0.15*(f_ip1pj*f_ip1pj) - 0.05*(f_im1pj*f_im1pj) - 0.05*f_ipj*f_ip1 + 0.03333333333333333*f_ip2pj*f_i - 0.01666666666666667*f_ip2pj*f_ip1 + 0.05*f_ip1*f_ip1pj + 0.1*f_ipj*f_i - 0.2*f_ip2pj*f_ip1pj - 0.03333333333333333*f_i*f_im1pj - 0.1*f_i*f_ip1pj + 0.05*(f_ip2pj*f_ip2pj) - 0.05*f_ipj*f_im1 + 0.01666666666666667*f_im1pj*f_im1 - 0.01666666666666667*f_ip2pj*f_im1 - 0.15*(f_ipj*f_ipj) + 0.2*f_ipj*f_im1pj;
      }
      b0 += -0.1166666666666667*f_m2mjpn*f_m3mjpn - 0.06666666666666667*f_m1mjpn*f_m3mjpn - 0.1166666666666667*f_m2pn*f_m3pn - 0.3833333333333333*f_m1mjpn*f_m2pn + 0.1166666666666667*f_m2pn*f_m3mjpn + 0.3833333333333333*f_m1pn*f_m2pn + 0.008333333333333333*(f_m3pn*f_m3pn) + 0.2583333333333333*(f_m1pn*f_m1pn) - 0.06666666666666667*f_m1pn*f_m3pn + 0.3833333333333333*f_m1mjpn*f_m2mjpn + 0.5333333333333333*(f_m2pn*f_m2pn) + 0.06666666666666667*f_m1pn*f_m3mjpn + 0.1166666666666667*f_m2mjpn*f_m3pn - 0.3833333333333333*f_m1pn*f_m2mjpn - 0.01666666666666667*f_m3pn*f_m3mjpn - 0.5166666666666667*f_m1pn*f_m1mjpn - 1.066666666666667*f_m2pn*f_m2mjpn + 0.008333333333333333*(f_m3mjpn*f_m3mjpn) + 0.2583333333333333*(f_m1mjpn*f_m1mjpn) + 0.06666666666666667*f_m1mjpn*f_m3pn + 0.5333333333333333*(f_m2mjpn*f_m2mjpn);
      b1 += 0.1666666666666667*f_m2mjpn*f_m3pn - 0.1666666666666667*f_m1mjpn*f_m3pn - (f_m2pn*f_m2pn) - 0.1666666666666667*f_m2pn*f_m3mjpn + 0.1666666666666667*f_m1pn*f_m3mjpn + 1.166666666666667*f_m2pn*f_m1mjpn + f_m2pn*f_m2mjpn - 1.166666666666667*f_m1pn*f_m2mjpn - (f_m1mjpn*f_m1mjpn) + f_m1pn*f_m1mjpn;
      b2 += 0.5*f_m1mjpn*f_m3mjpn + 1.333333333333333*f_m1pn*f_m2mjpn - 0.4166666666666667*f_m1pn*f_m3mjpn - 0.1666666666666667*f_m2pn*f_m3mjpn - 0.5*f_m1pn*f_m2pn + 0.5*f_m2pn*f_m3pn + 0.08333333333333333*f_m1mjpn*f_m3pn - 0.6666666666666667*f_m2mjpn*f_m3pn + 1.333333333333333*f_m2pn*f_m2mjpn - 2.0*f_m2mjpn*f_m1mjpn - 0.4166666666666667*f_m1pn*f_m1mjpn + 0.08333333333333333*f_m3pn*f_m3mjpn + 1.5*(f_m1mjpn*f_m1mjpn) - 1.166666666666667*f_m2pn*f_m1mjpn;
      b3 += -0.8333333333333333*f_m1mjpn*f_m3mjpn + 0.6666666666666667*f_m2mjpn*f_m3mjpn - 0.3333333333333333*f_m2pn*f_m3pn - 1.333333333333333*(f_m2mjpn*f_m2mjpn) - 0.08333333333333333*f_m1pn*f_m1mjpn + 0.08333333333333333*f_m1pn*f_m3mjpn + 0.08333333333333333*f_m1mjpn*f_m3pn - 0.08333333333333333*(f_m3pn*f_m3pn) - 0.08333333333333333*(f_m1pn*f_m1pn) + 0.1666666666666667*f_m1pn*f_m3pn + 2.666666666666667*f_m2mjpn*f_m1mjpn + 0.6666666666666667*(f_m2pn*f_m2pn) + 0.3333333333333333*f_m2pn*f_m3mjpn + 0.3333333333333333*f_m2pn*f_m1mjpn - 0.3333333333333333*f_m1pn*f_m2pn + 0.3333333333333333*f_m1pn*f_m2mjpn - 1.333333333333333*f_m2pn*f_m2mjpn - 0.08333333333333333*f_m3pn*f_m3mjpn - 1.083333333333333*(f_m1mjpn*f_m1mjpn) + 0.3333333333333333*f_m2mjpn*f_m3pn - 0.08333333333333333*(f_m3mjpn*f_m3mjpn);
      b4 += 0.5*f_m1mjpn*f_m3mjpn - 0.75*f_m2mjpn*f_m3mjpn - 0.25*f_m2pn*f_m3pn + 0.08333333333333333*f_m1pn*f_m3mjpn + 0.25*f_m1pn*f_m2pn + 0.125*(f_m3pn*f_m3pn) - 0.125*(f_m1pn*f_m1pn) - 0.08333333333333333*f_m1mjpn*f_m3pn - 0.08333333333333333*f_m2pn*f_m3mjpn - 1.25*f_m2mjpn*f_m1mjpn + 0.08333333333333333*f_m2pn*f_m1mjpn - 0.08333333333333333*f_m1pn*f_m2mjpn + 0.125*(f_m3mjpn*f_m3mjpn) + (f_m2mjpn*f_m2mjpn) + 0.08333333333333333*f_m2mjpn*f_m3pn + 0.375*(f_m1mjpn*f_m1mjpn);
      b5 += -0.1*f_m1mjpn*f_m3mjpn + 0.2*f_m2pn*f_m3pn + 0.06666666666666667*f_m2pn*f_m2mjpn - 0.03333333333333333*f_m2mjpn*f_m3pn + 0.01666666666666667*f_m1pn*f_m3mjpn + 0.2*f_m2pn*f_m1pn - 0.05*(f_m3pn*f_m3pn) - 0.05*(f_m1pn*f_m1pn) + 0.2*f_m2mjpn*f_m3mjpn + 0.2*f_m2mjpn*f_m1mjpn - 0.2*(f_m2pn*f_m2pn) - 0.03333333333333333*f_m2pn*f_m3mjpn - 0.1*f_m1pn*f_m3pn - 0.03333333333333333*f_m2pn*f_m1mjpn + 0.01666666666666667*f_m3pn*f_m3mjpn + 0.01666666666666667*f_m1pn*f_m1mjpn - 0.03333333333333333*f_m1pn*f_m2mjpn - 0.05*(f_m3mjpn*f_m3mjpn) - 0.2*(f_m2mjpn*f_m2mjpn) + 0.01666666666666667*f_m1mjpn*f_m3pn - 0.05*(f_m1mjpn*f_m1mjpn);
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
}
        
int cf_e22_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e22_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5);
    cf_quadratic_approximation_1_5(a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e22_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e22_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5);
    cf_linear_approximation_1_5(a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e22_find_zero_diff0: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e22_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=1) = sum(a_k*r^k, k=0..5) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double f_im1, f_m3mjpn, f_ipj, f_ip1pj, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_im1 = F(i-1);
        f_ipj = F(i+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip1 = F(i+1);
        b0 += 0.1666666666666667*f_ip1pj*f_im1 - 0.1666666666666667*f_ip1*f_im1pj + 1.166666666666667*f_ipj*f_ip1 + (f_ip1pj*f_ip1pj) + f_ipj*f_i - f_ip1*f_ip1pj + 0.1666666666666667*f_i*f_im1pj - 1.166666666666667*f_i*f_ip1pj - 0.1666666666666667*f_ipj*f_im1 - (f_ipj*f_ipj);
        b1 += -2.0*f_ipj*f_ip1pj + 0.1666666666666667*f_im1*f_ip1pj + 0.1666666666666667*f_ip1*f_im1pj - 1.333333333333333*f_ipj*f_ip1 - f_ip2pj*f_ip1 + 0.1666666666666667*f_im1pj*f_im1 + 2.666666666666667*f_ipj*f_i + f_ip2pj*f_ip1pj - 1.333333333333333*f_i*f_im1pj - 1.333333333333333*f_i*f_ip1pj + 2.166666666666667*f_ip1*f_ip1pj - 0.3333333333333333*f_ipj*f_im1 + f_ipj*f_im1pj;
        b2 += -0.75*f_im1*f_ip1pj + 0.25*f_ip1*f_im1pj + 0.5*f_im1pj*f_ip1pj - 2.25*(f_ip1pj*f_ip1pj) - 0.25*(f_im1pj*f_im1pj) - 0.5*f_ipj*f_ip2pj - f_ip2pj*f_i - 3.0*f_ipj*f_i + 0.75*f_ip1*f_ip1pj - 0.25*f_ip2pj*f_ip1 + 0.75*f_ipj*f_im1 - 0.75*f_ipj*f_ip1 + f_i*f_im1pj + 3.0*f_i*f_ip1pj - 0.25*f_im1pj*f_im1 + f_ip2pj*f_ip1pj + 0.25*(f_ip2pj*f_ip2pj) + 0.25*f_ip2pj*f_im1 + 2.25*(f_ipj*f_ipj) - f_ipj*f_im1pj;
        b3 += 2.0*f_ipj*f_ip1pj - 0.3333333333333333*f_im1pj*f_ip1 + f_ipj*f_ip1 + 0.5*(f_im1pj*f_im1pj) - 0.3333333333333333*f_ip2pj*f_i - 0.5*(f_ip1pj*f_ip1pj) + 0.3333333333333333*f_ip2pj*f_ip1 - f_ip1*f_ip1pj - f_ipj*f_i - f_ip2pj*f_ip1pj + 0.3333333333333333*f_i*f_im1pj + f_i*f_ip1pj + 0.5*(f_ip2pj*f_ip2pj) - 0.5*(f_ipj*f_ipj) - f_ipj*f_im1pj;
        b4 += 0.25*f_im1*f_ip1pj + 0.08333333333333333*f_ip1*f_im1pj - 0.5*f_im1pj*f_ip1pj + 0.75*(f_ip1pj*f_ip1pj) - 0.25*(f_im1pj*f_im1pj) + 0.5*f_ipj*f_ip2pj + 0.1666666666666667*f_ip2pj*f_i + 0.5*f_ipj*f_i + 0.25*f_ip1*f_ip1pj - 0.08333333333333333*f_ip2pj*f_ip1 - 0.25*f_ipj*f_im1 - 0.25*f_ipj*f_ip1 - 0.1666666666666667*f_i*f_im1pj - 0.5*f_i*f_ip1pj + 0.08333333333333333*f_im1pj*f_im1 - f_ip2pj*f_ip1pj + 0.25*(f_ip2pj*f_ip2pj) - 0.08333333333333333*f_ip2pj*f_im1 - 0.75*(f_ipj*f_ipj) + f_ipj*f_im1pj;
      }
      b0 += 0.1666666666666667*f_m1pn*f_m3mjpn + 0.1666666666666667*f_m2mjpn*f_m3pn - (f_m2pn*f_m2pn) - 0.1666666666666667*f_m1mjpn*f_m3pn - 0.1666666666666667*f_m2pn*f_m3mjpn + f_m2pn*f_m2mjpn + f_m1pn*f_m1mjpn + 1.166666666666667*f_m2pn*f_m1mjpn - (f_m1mjpn*f_m1mjpn) - 1.166666666666667*f_m1pn*f_m2mjpn;
      b1 += f_m1mjpn*f_m3mjpn - 2.333333333333333*f_m2pn*f_m1mjpn - 0.8333333333333333*f_m1pn*f_m3mjpn - 4.0*f_m2mjpn*f_m1mjpn + 0.1666666666666667*f_m3pn*f_m3mjpn + f_m2pn*f_m3pn - f_m1pn*f_m2pn - 1.333333333333333*f_m2mjpn*f_m3pn - 0.3333333333333333*f_m2pn*f_m3mjpn + 2.666666666666667*f_m2pn*f_m2mjpn - 0.8333333333333333*f_m1pn*f_m1mjpn + 2.666666666666667*f_m1pn*f_m2mjpn + 3.0*(f_m1mjpn*f_m1mjpn) + 0.1666666666666667*f_m1mjpn*f_m3pn;
      b2 += 2.0*f_m2mjpn*f_m3mjpn - 2.5*f_m1mjpn*f_m3mjpn - f_m2pn*f_m3pn - 0.25*f_m1pn*f_m1mjpn + 0.25*f_m1pn*f_m3mjpn - 0.25*f_m3pn*f_m3mjpn - f_m1pn*f_m2pn - 0.25*(f_m3pn*f_m3pn) - 0.25*(f_m1pn*f_m1pn) + 0.5*f_m1pn*f_m3pn + 8.0*f_m2mjpn*f_m1mjpn + 2.0*(f_m2pn*f_m2pn) + f_m2pn*f_m3mjpn + 0.25*f_m1mjpn*f_m3pn + f_m2pn*f_m1mjpn - 4.0*f_m2pn*f_m2mjpn + f_m1pn*f_m2mjpn - 0.25*(f_m3mjpn*f_m3mjpn) - 3.25*(f_m1mjpn*f_m1mjpn) + f_m2mjpn*f_m3pn - 4.0*(f_m2mjpn*f_m2mjpn);
      b3 += -f_m2pn*f_m3pn + 2.0*f_m1mjpn*f_m3mjpn + 0.3333333333333333*f_m1pn*f_m3mjpn + 0.3333333333333333*f_m2mjpn*f_m3pn + f_m1pn*f_m2pn + 0.5*(f_m3pn*f_m3pn) - 0.5*(f_m1pn*f_m1pn) - 3.0*f_m2mjpn*f_m3mjpn - 0.3333333333333333*f_m2pn*f_m3mjpn - 5.0*f_m2mjpn*f_m1mjpn - 0.3333333333333333*f_m1pn*f_m2mjpn + 0.3333333333333333*f_m2pn*f_m1mjpn + 0.5*(f_m3mjpn*f_m3mjpn) + 4.0*(f_m2mjpn*f_m2mjpn) - 0.3333333333333333*f_m1mjpn*f_m3pn + 1.5*(f_m1mjpn*f_m1mjpn);
      b4 += f_m2mjpn*f_m3mjpn - 0.5*f_m1mjpn*f_m3mjpn - 0.5*f_m1pn*f_m3pn + 0.3333333333333333*f_m2pn*f_m2mjpn + 0.08333333333333333*f_m1pn*f_m3mjpn + 0.08333333333333333*f_m3pn*f_m3mjpn + f_m2pn*f_m1pn - 0.25*(f_m3pn*f_m3pn) - 0.25*(f_m1pn*f_m1pn) + f_m2pn*f_m3pn + f_m2mjpn*f_m1mjpn - (f_m2pn*f_m2pn) - 0.1666666666666667*f_m2pn*f_m3mjpn + 0.08333333333333333*f_m1mjpn*f_m3pn - 0.1666666666666667*f_m2pn*f_m1mjpn - 0.1666666666666667*f_m1pn*f_m2mjpn + 0.08333333333333333*f_m1pn*f_m1mjpn - 0.25*(f_m3mjpn*f_m3mjpn) - (f_m2mjpn*f_m2mjpn) - 0.1666666666666667*f_m2mjpn*f_m3pn - 0.25*(f_m1mjpn*f_m1mjpn);
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
}
        
int cf_e22_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e22_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5);
    cf_quadratic_approximation_1_4(a1_0, a1_1, a1_2, a1_3, a1_4, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e22_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e22_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5);
    cf_linear_approximation_1_4(a1_0, a1_1, a1_2, a1_3, a1_4, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e22_find_zero_diff1: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e22_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=2) = sum(a_k*r^k, k=0..5) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double f_ipj, f_m3mjpn, f_im1, f_ip1pj, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_im1 = F(i-1);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip1 = F(i+1);
        b0 += -2.0*f_ipj*f_ip1pj + 0.1666666666666667*f_im1*f_ip1pj + 0.1666666666666667*f_ip1*f_im1pj - 1.333333333333333*f_ipj*f_ip1 - f_ip2pj*f_ip1 + 0.1666666666666667*f_im1pj*f_im1 + 2.666666666666667*f_ipj*f_i + f_ip2pj*f_ip1pj - 1.333333333333333*f_i*f_im1pj - 1.333333333333333*f_i*f_ip1pj + 2.166666666666667*f_ip1*f_ip1pj - 0.3333333333333333*f_ipj*f_im1 + f_ipj*f_im1pj;
        b1 += -1.5*f_im1*f_ip1pj + 0.5*f_ip1*f_im1pj + f_im1pj*f_ip1pj - 4.5*(f_ip1pj*f_ip1pj) - 0.5*(f_im1pj*f_im1pj) - f_ipj*f_ip2pj - 2.0*f_ip2pj*f_i - 0.5*f_ip2pj*f_ip1 + 1.5*f_ip1*f_ip1pj - 6.0*f_ipj*f_i + 2.0*f_ip2pj*f_ip1pj + 2.0*f_i*f_im1pj + 6.0*f_i*f_ip1pj + 0.5*(f_ip2pj*f_ip2pj) - 1.5*f_ipj*f_ip1 + 1.5*f_ipj*f_im1 - 0.5*f_im1pj*f_im1 + 0.5*f_ip2pj*f_im1 + 4.5*(f_ipj*f_ipj) - 2.0*f_ipj*f_im1pj;
        b2 += 6.0*f_ipj*f_ip1pj - f_im1pj*f_ip1 + 3.0*f_ipj*f_ip1 + 1.5*(f_im1pj*f_im1pj) - f_ip2pj*f_i - 1.5*(f_ip1pj*f_ip1pj) + f_ip2pj*f_ip1 - 3.0*f_ip1*f_ip1pj - 3.0*f_ipj*f_i - 3.0*f_ip2pj*f_ip1pj + f_i*f_im1pj + 3.0*f_i*f_ip1pj + 1.5*(f_ip2pj*f_ip2pj) - 1.5*(f_ipj*f_ipj) - 3.0*f_ipj*f_im1pj;
        b3 += f_im1*f_ip1pj + 0.3333333333333333*f_ip1*f_im1pj - 2.0*f_im1pj*f_ip1pj + 3.0*(f_ip1pj*f_ip1pj) - (f_im1pj*f_im1pj) + 2.0*f_ipj*f_ip2pj + 0.6666666666666667*f_ip2pj*f_i - 0.3333333333333333*f_ip2pj*f_ip1 + f_ip1*f_ip1pj + 2.0*f_ipj*f_i - 4.0*f_ip2pj*f_ip1pj - 0.6666666666666667*f_i*f_im1pj - 2.0*f_i*f_ip1pj + (f_ip2pj*f_ip2pj) - f_ipj*f_ip1 - f_ipj*f_im1 + 0.3333333333333333*f_im1pj*f_im1 - 0.3333333333333333*f_ip2pj*f_im1 - 3.0*(f_ipj*f_ipj) + 4.0*f_ipj*f_im1pj;
      }
      b0 += f_m1mjpn*f_m3mjpn + 2.666666666666667*f_m1pn*f_m2mjpn - 0.8333333333333333*f_m1pn*f_m3mjpn - 0.3333333333333333*f_m2pn*f_m3mjpn - 4.0*f_m2mjpn*f_m1mjpn + f_m2pn*f_m3pn - f_m1pn*f_m2pn - 1.333333333333333*f_m2mjpn*f_m3pn + 0.1666666666666667*f_m1mjpn*f_m3pn + 0.1666666666666667*f_m3pn*f_m3mjpn - 0.8333333333333333*f_m1pn*f_m1mjpn - 2.333333333333333*f_m2pn*f_m1mjpn + 3.0*(f_m1mjpn*f_m1mjpn) + 2.666666666666667*f_m2pn*f_m2mjpn;
      b1 += f_m1pn*f_m3pn + 4.0*f_m2mjpn*f_m3mjpn - 2.0*f_m2pn*f_m3pn - 8.0*f_m2pn*f_m2mjpn + 0.5*f_m1pn*f_m3mjpn - 0.5*f_m3pn*f_m3mjpn - 2.0*f_m1pn*f_m2pn - 0.5*(f_m3pn*f_m3pn) - 0.5*(f_m1pn*f_m1pn) - 5.0*f_m1mjpn*f_m3mjpn + 16.0*f_m2mjpn*f_m1mjpn + 4.0*(f_m2pn*f_m2pn) + 2.0*f_m2pn*f_m3mjpn + 2.0*f_m2mjpn*f_m3pn - 0.5*f_m1pn*f_m1mjpn + 2.0*f_m1pn*f_m2mjpn + 2.0*f_m2pn*f_m1mjpn - 0.5*(f_m3mjpn*f_m3mjpn) - 6.5*(f_m1mjpn*f_m1mjpn) + 0.5*f_m1mjpn*f_m3pn - 8.0*(f_m2mjpn*f_m2mjpn);
      b2 += -3.0*f_m2pn*f_m3pn - 9.0*f_m2mjpn*f_m3mjpn + f_m1pn*f_m3mjpn - f_m1mjpn*f_m3pn + 3.0*f_m1pn*f_m2pn + 1.5*(f_m3pn*f_m3pn) - 1.5*(f_m1pn*f_m1pn) + 6.0*f_m1mjpn*f_m3mjpn - f_m2pn*f_m3mjpn - 15.0*f_m2mjpn*f_m1mjpn + f_m2pn*f_m1mjpn - f_m1pn*f_m2mjpn + 1.5*(f_m3mjpn*f_m3mjpn) + 12.0*(f_m2mjpn*f_m2mjpn) + f_m2mjpn*f_m3pn + 4.5*(f_m1mjpn*f_m1mjpn);
      b3 += 4.0*f_m2pn*f_m3pn + 4.0*f_m2mjpn*f_m3mjpn - 2.0*f_m1pn*f_m3pn - 0.6666666666666667*f_m1pn*f_m2mjpn + 0.3333333333333333*f_m1pn*f_m3mjpn + 0.3333333333333333*f_m3pn*f_m3mjpn + 4.0*f_m2pn*f_m1pn - (f_m3pn*f_m3pn) - (f_m1pn*f_m1pn) - 2.0*f_m1mjpn*f_m3mjpn + 4.0*f_m2mjpn*f_m1mjpn - 4.0*(f_m2pn*f_m2pn) - 0.6666666666666667*f_m2pn*f_m3mjpn - 0.6666666666666667*f_m2mjpn*f_m3pn + 1.333333333333333*f_m2pn*f_m2mjpn + 0.3333333333333333*f_m1pn*f_m1mjpn - 0.6666666666666667*f_m2pn*f_m1mjpn - (f_m3mjpn*f_m3mjpn) - 4.0*(f_m2mjpn*f_m2mjpn) + 0.3333333333333333*f_m1mjpn*f_m3pn - (f_m1mjpn*f_m1mjpn);
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
}
        
int cf_e22_find_extreme_diff2(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e22_compute_coeffs_diff2(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5);
    cf_quadratic_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e22_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e22_compute_coeffs_diff2(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5);
    cf_linear_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e22_find_zero_diff2: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e22_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=3) = sum(a_k*r^k, k=0..5) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double f_im1, f_m3mjpn, f_ipj, f_ip1pj, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_im1 = F(i-1);
        f_ipj = F(i+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip1 = F(i+1);
        b0 += -1.5*f_im1*f_ip1pj - f_ipj*f_ip2pj + 0.5*f_ip1*f_im1pj + f_im1pj*f_ip1pj - 4.5*(f_ip1pj*f_ip1pj) - 0.5*(f_im1pj*f_im1pj) - 1.5*f_ipj*f_ip1 - 2.0*f_ip2pj*f_i - 6.0*f_ipj*f_i + 1.5*f_ip1*f_ip1pj - 0.5*f_ip2pj*f_ip1 + 1.5*f_ipj*f_im1 + 2.0*f_ip2pj*f_ip1pj + 2.0*f_i*f_im1pj + 6.0*f_i*f_ip1pj - 0.5*f_im1pj*f_im1 + 0.5*(f_ip2pj*f_ip2pj) + 0.5*f_ip2pj*f_im1 + 4.5*(f_ipj*f_ipj) - 2.0*f_ipj*f_im1pj;
        b1 += 12.0*f_ipj*f_ip1pj - 2.0*f_im1pj*f_ip1 + 6.0*f_ipj*f_ip1 + 3.0*(f_im1pj*f_im1pj) - 2.0*f_ip2pj*f_i - 3.0*(f_ip1pj*f_ip1pj) + 2.0*f_ip2pj*f_ip1 - 6.0*f_ip1*f_ip1pj - 6.0*f_ipj*f_i - 6.0*f_ip2pj*f_ip1pj + 2.0*f_i*f_im1pj + 6.0*f_i*f_ip1pj + 3.0*(f_ip2pj*f_ip2pj) - 3.0*(f_ipj*f_ipj) - 6.0*f_ipj*f_im1pj;
        b2 += 3.0*f_im1*f_ip1pj + 6.0*f_ipj*f_ip2pj + f_ip1*f_im1pj - 6.0*f_im1pj*f_ip1pj + 9.0*(f_ip1pj*f_ip1pj) - 3.0*(f_im1pj*f_im1pj) - 3.0*f_ipj*f_ip1 + 2.0*f_ip2pj*f_i + 6.0*f_ipj*f_i + 3.0*f_ip1*f_ip1pj - f_ip2pj*f_ip1 - 3.0*f_ipj*f_im1 - 12.0*f_ip2pj*f_ip1pj - 2.0*f_i*f_im1pj - 6.0*f_i*f_ip1pj + f_im1pj*f_im1 + 3.0*(f_ip2pj*f_ip2pj) - f_ip2pj*f_im1 - 9.0*(f_ipj*f_ipj) + 12.0*f_ipj*f_im1pj;
      }
      b0 += f_m1pn*f_m3pn - 5.0*f_m1mjpn*f_m3mjpn - 2.0*f_m2pn*f_m3pn + 2.0*f_m1pn*f_m2mjpn + 0.5*f_m1pn*f_m3mjpn - 0.5*f_m3pn*f_m3mjpn - 2.0*f_m1pn*f_m2pn - 0.5*(f_m3pn*f_m3pn) - 0.5*(f_m1pn*f_m1pn) + 4.0*f_m2mjpn*f_m3mjpn + 16.0*f_m2mjpn*f_m1mjpn + 4.0*(f_m2pn*f_m2pn) + 2.0*f_m2pn*f_m3mjpn + 0.5*f_m1mjpn*f_m3pn - 8.0*f_m2pn*f_m2mjpn + 2.0*f_m2pn*f_m1mjpn - 0.5*f_m1pn*f_m1mjpn - 0.5*(f_m3mjpn*f_m3mjpn) - 6.5*(f_m1mjpn*f_m1mjpn) + 2.0*f_m2mjpn*f_m3pn - 8.0*(f_m2mjpn*f_m2mjpn);
      b1 += -6.0*f_m2pn*f_m3pn + 12.0*f_m1mjpn*f_m3mjpn + 2.0*f_m1pn*f_m3mjpn + 2.0*f_m2mjpn*f_m3pn + 6.0*f_m1pn*f_m2pn + 3.0*(f_m3pn*f_m3pn) - 3.0*(f_m1pn*f_m1pn) - 18.0*f_m2mjpn*f_m3mjpn - 2.0*f_m2pn*f_m3mjpn - 30.0*f_m2mjpn*f_m1mjpn - 2.0*f_m1pn*f_m2mjpn + 2.0*f_m2pn*f_m1mjpn + 3.0*(f_m3mjpn*f_m3mjpn) + 24.0*(f_m2mjpn*f_m2mjpn) - 2.0*f_m1mjpn*f_m3pn + 9.0*(f_m1mjpn*f_m1mjpn);
      b2 += 12.0*f_m2pn*f_m3pn - 6.0*f_m1mjpn*f_m3mjpn - 6.0*f_m1pn*f_m3pn + f_m1pn*f_m1mjpn + f_m1pn*f_m3mjpn + f_m3pn*f_m3mjpn + 12.0*f_m2pn*f_m1pn - 3.0*(f_m3pn*f_m3pn) - 3.0*(f_m1pn*f_m1pn) + 12.0*f_m2mjpn*f_m3mjpn + 12.0*f_m2mjpn*f_m1mjpn - 12.0*(f_m2pn*f_m2pn) - 2.0*f_m2pn*f_m3mjpn + f_m1mjpn*f_m3pn - 2.0*f_m1pn*f_m2mjpn - 2.0*f_m2pn*f_m1mjpn + 4.0*f_m2pn*f_m2mjpn - 3.0*(f_m3mjpn*f_m3mjpn) - 12.0*(f_m2mjpn*f_m2mjpn) - 2.0*f_m2mjpn*f_m3pn - 3.0*(f_m1mjpn*f_m1mjpn);
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
}
        
int cf_e22_find_extreme_diff3(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e22_compute_coeffs_diff3(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5);
    cf_quadratic_approximation_1_2(a1_0, a1_1, a1_2, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e22_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e22_compute_coeffs_diff3(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5);
    cf_linear_approximation_1_2(a1_0, a1_1, a1_2, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e22_find_zero_diff3: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e22_compute_coeffs_diff4(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=4) = sum(a_k*r^k, k=0..5) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double f_ipj, f_m3mjpn, f_ip2pj, f_im1, f_ip1pj, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_m2pn, f_im1pj, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip2pj = F(i+2+j);
        f_im1 = F(i-1);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_im1pj = F(i-1+j);
        f_ip1 = F(i+1);
        b0 += 12.0*f_ipj*f_ip1pj - 2.0*f_im1pj*f_ip1 + 6.0*f_ipj*f_ip1 + 3.0*(f_im1pj*f_im1pj) - 2.0*f_ip2pj*f_i - 3.0*(f_ip1pj*f_ip1pj) + 2.0*f_ip2pj*f_ip1 - 6.0*f_ip1*f_ip1pj - 6.0*f_ipj*f_i - 6.0*f_ip2pj*f_ip1pj + 2.0*f_i*f_im1pj + 6.0*f_i*f_ip1pj + 3.0*(f_ip2pj*f_ip2pj) - 3.0*(f_ipj*f_ipj) - 6.0*f_ipj*f_im1pj;
        b1 += 6.0*f_im1*f_ip1pj + 2.0*f_ip1*f_im1pj - 12.0*f_im1pj*f_ip1pj + 18.0*(f_ip1pj*f_ip1pj) - 6.0*(f_im1pj*f_im1pj) + 12.0*f_ipj*f_ip2pj + 4.0*f_ip2pj*f_i - 2.0*f_ip2pj*f_ip1 + 6.0*f_ip1*f_ip1pj + 12.0*f_ipj*f_i - 6.0*f_ipj*f_ip1 - 4.0*f_i*f_im1pj - 12.0*f_i*f_ip1pj + 6.0*(f_ip2pj*f_ip2pj) - 24.0*f_ip2pj*f_ip1pj - 6.0*f_ipj*f_im1 + 2.0*f_im1pj*f_im1 - 2.0*f_ip2pj*f_im1 - 18.0*(f_ipj*f_ipj) + 24.0*f_ipj*f_im1pj;
      }
      b0 += -6.0*f_m2pn*f_m3pn - 18.0*f_m2mjpn*f_m3mjpn + 2.0*f_m1pn*f_m3mjpn - 2.0*f_m1mjpn*f_m3pn + 6.0*f_m1pn*f_m2pn + 3.0*(f_m3pn*f_m3pn) - 3.0*(f_m1pn*f_m1pn) + 12.0*f_m1mjpn*f_m3mjpn - 2.0*f_m2pn*f_m3mjpn - 30.0*f_m2mjpn*f_m1mjpn + 2.0*f_m2pn*f_m1mjpn - 2.0*f_m1pn*f_m2mjpn + 3.0*(f_m3mjpn*f_m3mjpn) + 24.0*(f_m2mjpn*f_m2mjpn) + 2.0*f_m2mjpn*f_m3pn + 9.0*(f_m1mjpn*f_m1mjpn);
      b1 += 24.0*f_m2pn*f_m3pn + 24.0*f_m2mjpn*f_m3mjpn - 12.0*f_m1pn*f_m3pn - 4.0*f_m2pn*f_m1mjpn + 2.0*f_m1pn*f_m3mjpn + 2.0*f_m3pn*f_m3mjpn + 24.0*f_m2pn*f_m1pn - 6.0*(f_m3pn*f_m3pn) - 6.0*(f_m1pn*f_m1pn) - 12.0*f_m1mjpn*f_m3mjpn + 24.0*f_m2mjpn*f_m1mjpn - 24.0*(f_m2pn*f_m2pn) - 4.0*f_m2pn*f_m3mjpn - 4.0*f_m2mjpn*f_m3pn + 2.0*f_m1pn*f_m1mjpn + 8.0*f_m2pn*f_m2mjpn - 4.0*f_m1pn*f_m2mjpn - 6.0*(f_m3mjpn*f_m3mjpn) - 24.0*(f_m2mjpn*f_m2mjpn) + 2.0*f_m1mjpn*f_m3pn - 6.0*(f_m1mjpn*f_m1mjpn);
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
}
        
int cf_e22_find_extreme_diff4(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a2_2 = 0.0;
  double a2_3 = 0.0;
  double a2_4 = 0.0;
  double a2_5 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  double a3_2 = 0.0;
  double a3_3 = 0.0;
  double a3_4 = 0.0;
  double a3_5 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a1_2 = a2_2;
    a1_3 = a2_3;
    a1_4 = a2_4;
    a1_5 = a2_5;
    a2_0 = a3_0;
    a2_1 = a3_1;
    a2_2 = a3_2;
    a2_3 = a3_3;
    a2_4 = a3_4;
    a2_5 = a3_5;
    cf_e22_compute_coeffs_diff4(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3, &a3_4, &a3_5);
    count ++;
    if (count<3)
      continue;
    cf_quadratic_approximation_3_1(a1_0, a1_1, a2_0, a2_1, a3_0, a3_1, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e22_find_zero_diff4(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e22_compute_coeffs_diff4(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3, &a1_4, &a1_5);
    cf_linear_approximation_1_1(a1_0, a1_1, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e22_find_zero_diff4: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e22_compute_coeffs_diff5(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=5) = sum(a_k*r^k, k=0..5) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  double b5 = 0.0;
  double f_im1, f_m3mjpn, f_ipj, f_ip1pj, f_m1mjpn, f_m3pn, f_i, f_m2mjpn, f_ip2pj, f_m2pn, f_im1pj, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m3mjpn = F(-3-j+n);
      f_m1mjpn = F(-1-j+n);
      f_m3pn = F(-3+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_im1 = F(i-1);
        f_ipj = F(i+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip2pj = F(i+2+j);
        f_im1pj = F(i-1+j);
        f_ip1 = F(i+1);
        b0 += 6.0*f_im1*f_ip1pj + 2.0*f_ip1*f_im1pj - 12.0*f_im1pj*f_ip1pj + 18.0*(f_ip1pj*f_ip1pj) - 6.0*(f_im1pj*f_im1pj) + 12.0*f_ipj*f_ip2pj + 4.0*f_ip2pj*f_i + 12.0*f_ipj*f_i + 6.0*f_ip1*f_ip1pj - 2.0*f_ip2pj*f_ip1 - 6.0*f_ipj*f_im1 - 24.0*f_ip2pj*f_ip1pj - 4.0*f_i*f_im1pj - 12.0*f_i*f_ip1pj + 2.0*f_im1pj*f_im1 - 6.0*f_ipj*f_ip1 + 6.0*(f_ip2pj*f_ip2pj) - 2.0*f_ip2pj*f_im1 - 18.0*(f_ipj*f_ipj) + 24.0*f_ipj*f_im1pj;
      }
      b0 += 24.0*f_m2pn*f_m3pn - 12.0*f_m1mjpn*f_m3mjpn - 12.0*f_m1pn*f_m3pn + 8.0*f_m2pn*f_m2mjpn + 2.0*f_m1pn*f_m3mjpn + 2.0*f_m3pn*f_m3mjpn + 24.0*f_m2pn*f_m1pn - 6.0*(f_m3pn*f_m3pn) - 6.0*(f_m1pn*f_m1pn) + 24.0*f_m2mjpn*f_m3mjpn + 24.0*f_m2mjpn*f_m1mjpn - 24.0*(f_m2pn*f_m2pn) - 4.0*f_m2pn*f_m3mjpn + 2.0*f_m1mjpn*f_m3pn - 4.0*f_m2pn*f_m1mjpn - 4.0*f_m1pn*f_m2mjpn + 2.0*f_m1pn*f_m1mjpn - 6.0*(f_m3mjpn*f_m3mjpn) - 24.0*(f_m2mjpn*f_m2mjpn) - 4.0*f_m2mjpn*f_m3pn - 6.0*(f_m1mjpn*f_m1mjpn);
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
  *a4 = b4;
  *a5 = b5;
}
        
int cf_e22_find_extreme_diff5(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a2_2 = 0.0;
  double a2_3 = 0.0;
  double a2_4 = 0.0;
  double a2_5 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  double a3_2 = 0.0;
  double a3_3 = 0.0;
  double a3_4 = 0.0;
  double a3_5 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a1_2 = a2_2;
    a1_3 = a2_3;
    a1_4 = a2_4;
    a1_5 = a2_5;
    a2_0 = a3_0;
    a2_1 = a3_1;
    a2_2 = a3_2;
    a2_3 = a3_3;
    a2_4 = a3_4;
    a2_5 = a3_5;
    cf_e22_compute_coeffs_diff5(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3, &a3_4, &a3_5);
    count ++;
    if (count<3)
      continue;
    cf_quadratic_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e22_find_zero_diff5(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a1_4 = 0.0;
  double a1_5 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a2_2 = 0.0;
  double a2_3 = 0.0;
  double a2_4 = 0.0;
  double a2_5 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  double a3_2 = 0.0;
  double a3_3 = 0.0;
  double a3_4 = 0.0;
  double a3_5 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a1_2 = a2_2;
    a1_3 = a2_3;
    a1_4 = a2_4;
    a1_5 = a2_5;
    a2_0 = a3_0;
    a2_1 = a3_1;
    a2_2 = a3_2;
    a2_3 = a3_3;
    a2_4 = a3_4;
    a2_5 = a3_5;
    cf_e22_compute_coeffs_diff5(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3, &a3_4, &a3_5);
    count ++;
    if (count<3)
      continue;
    cf_linear_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1);
    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e22_find_zero_diff5: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
void cf_e22_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order) = sum(a_k*r^k, k=0..5) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s**2 + (0.5*(F(i+1)) - 0.5*(F(i-1)))*s + (F(i))), i=0..N-1) where s=x-i */
  switch (order)
  {
    case 0: cf_e22_compute_coeffs_diff0(j, fm, n, m, a0, a1, a2, a3, a4, a5); break;
    case 1: cf_e22_compute_coeffs_diff1(j, fm, n, m, a0, a1, a2, a3, a4, a5); break;
    case 2: cf_e22_compute_coeffs_diff2(j, fm, n, m, a0, a1, a2, a3, a4, a5); break;
    case 3: cf_e22_compute_coeffs_diff3(j, fm, n, m, a0, a1, a2, a3, a4, a5); break;
    case 4: cf_e22_compute_coeffs_diff4(j, fm, n, m, a0, a1, a2, a3, a4, a5); break;
    case 5: cf_e22_compute_coeffs_diff5(j, fm, n, m, a0, a1, a2, a3, a4, a5); break;
    default:
      *a0 = 0.0;
      *a1 = 0.0;
      *a2 = 0.0;
      *a3 = 0.0;
      *a4 = 0.0;
      *a5 = 0.0;
  }
}
        
int cf_e22_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  switch (order)
  {
    case 0: return cf_e22_find_extreme_diff0(j0, j1, fm, n, m, result);
    case 1: return cf_e22_find_extreme_diff1(j0, j1, fm, n, m, result);
    case 2: return cf_e22_find_extreme_diff2(j0, j1, fm, n, m, result);
    case 3: return cf_e22_find_extreme_diff3(j0, j1, fm, n, m, result);
    case 4: return cf_e22_find_extreme_diff4(j0, j1, fm, n, m, result);
    case 5: return cf_e22_find_extreme_diff5(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
int cf_e22_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  switch (order)
  {
    case 0: return cf_e22_find_zero_diff0(j0, j1, fm, n, m, result);
    case 1: return cf_e22_find_zero_diff1(j0, j1, fm, n, m, result);
    case 2: return cf_e22_find_zero_diff2(j0, j1, fm, n, m, result);
    case 3: return cf_e22_find_zero_diff3(j0, j1, fm, n, m, result);
    case 4: return cf_e22_find_zero_diff4(j0, j1, fm, n, m, result);
    case 5: return cf_e22_find_zero_diff5(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
double cf_e22_evaluate(double y, double *fm, int n, int m, int order)
{
  double a0 = 0.0;
  double a1 = 0.0;
  double a2 = 0.0;
  double a3 = 0.0;
  double a4 = 0.0;
  double a5 = 0.0;
  /*
  int j = floor((y<0?-y:y));
  double r = (y<0?-y:y) - j;
  */
  int j = floor (y);
  double r = y - j;
  cf_e22_compute_coeffs(j, fm, n, m, order, &a0, &a1, &a2, &a3, &a4, &a5);
  return a0+(a1+(a2+(a3+(a4+(a5)*r)*r)*r)*r)*r;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double cf_e22_f1_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  switch (order)
  {
    case 0: return F(i) + (0.5*(F(i+1)) - 0.5*(F(i-1)) + (-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s)*s;
    case 1: return 0.5*(F(i+1)) - 0.5*(F(i-1)) + (-2.0*(F(i)) + (F(i+1)) + (F(i-1)))*s;
    case 2: return -2.0*(F(i)) + (F(i+1)) + (F(i-1));
  }
  return 0.0;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double cf_e22_f2_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  switch (order)
  {
    case 0: return F(i) + (0.5*(F(i+1)) - 0.5*(F(i-1)) + (-(F(i)) + 0.5*(F(i+1)) + 0.5*(F(i-1)))*s)*s;
    case 1: return 0.5*(F(i+1)) - 0.5*(F(i-1)) + (-2.0*(F(i)) + (F(i+1)) + (F(i-1)))*s;
    case 2: return -2.0*(F(i)) + (F(i+1)) + (F(i-1));
  }
  return 0.0;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e11_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double f_ipj, f_ip2pj, f_ip1pj, f_m1mjpn, f_i, f_m2mjpn, f_m2pn, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m1mjpn = F(-1-j+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip2pj = F(i+2+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip1 = F(i+1);
        b0 += 0.3333333333333333*f_ipj*f_ip1pj + 0.3333333333333333*(f_ip1pj*f_ip1pj) + 0.3333333333333333*(f_ip1*f_ip1) - 0.3333333333333333*f_ipj*f_ip1 + 0.3333333333333333*f_i*f_ip1 - 0.6666666666666667*f_ipj*f_i - 0.6666666666666667*f_ip1*f_ip1pj - 0.3333333333333333*f_i*f_ip1pj + 0.3333333333333333*(f_i*f_i) + 0.3333333333333333*(f_ipj*f_ipj);
        b1 += (f_ip1pj*f_ip1pj) + f_ipj*f_ip1 + f_ipj*f_i - f_ip1*f_ip1pj - f_i*f_ip1pj - (f_ipj*f_ipj);
        b2 += -f_ipj*f_ip1pj - (f_ip1pj*f_ip1pj) - f_ipj*f_ip1 - f_ip2pj*f_ip1 + 2.0*f_ip1*f_ip1pj + f_ip2pj*f_ip1pj + (f_ipj*f_ipj);
        b3 += 0.6666666666666667*f_ipj*f_ip1pj - 0.3333333333333333*f_ip2pj*f_i + 0.3333333333333333*f_ipj*f_ip1 - 0.3333333333333333*f_ipj*f_i - 0.6666666666666667*f_ip1*f_ip1pj + 0.3333333333333333*f_ip2pj*f_ip1 - 0.6666666666666667*f_ip2pj*f_ip1pj + 0.6666666666666667*f_i*f_ip1pj + 0.3333333333333333*(f_ip2pj*f_ip2pj) - 0.3333333333333333*(f_ipj*f_ipj);
      }
      b0 += 0.3333333333333333*f_m1pn*f_m2pn + 0.3333333333333333*(f_m2pn*f_m2pn) + 0.3333333333333333*f_m1mjpn*f_m2mjpn + 0.3333333333333333*(f_m1pn*f_m1pn) - 0.3333333333333333*f_m1pn*f_m2mjpn - 0.6666666666666667*f_m1mjpn*f_m1pn - 0.6666666666666667*f_m2mjpn*f_m2pn + 0.3333333333333333*(f_m2mjpn*f_m2mjpn) - 0.3333333333333333*f_m1mjpn*f_m2pn + 0.3333333333333333*(f_m1mjpn*f_m1mjpn);
      b1 += -(f_m2pn*f_m2pn) + f_m1mjpn*f_m1pn - f_m2mjpn*f_m1pn + f_m2mjpn*f_m2pn - (f_m1mjpn*f_m1mjpn) + f_m1mjpn*f_m2pn;
      b2 += -f_m1pn*f_m2pn + (f_m2pn*f_m2pn) - f_m1mjpn*f_m2mjpn - f_m1mjpn*f_m2pn + f_m1pn*f_m2mjpn + (f_m1mjpn*f_m1mjpn);
      b3 += 0.6666666666666667*f_m2pn*f_m1pn - 0.3333333333333333*(f_m2pn*f_m2pn) + 0.6666666666666667*f_m2mjpn*f_m1mjpn - 0.3333333333333333*(f_m1pn*f_m1pn) + 0.3333333333333333*f_m1mjpn*f_m2pn - 0.3333333333333333*f_m2mjpn*f_m2pn + 0.3333333333333333*f_m2mjpn*f_m1pn - 0.3333333333333333*(f_m2mjpn*f_m2mjpn) - 0.3333333333333333*f_m1mjpn*f_m1pn - 0.3333333333333333*(f_m1mjpn*f_m1mjpn);
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int cf_e11_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e11_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    cf_quadratic_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e11_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e11_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    cf_linear_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e11_find_zero_diff0: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e11_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=1) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double f_ipj, f_ip2pj, f_ip1pj, f_m1mjpn, f_i, f_m2mjpn, f_m2pn, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m1mjpn = F(-1-j+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip2pj = F(i+2+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip1 = F(i+1);
        b0 += (f_ip1pj*f_ip1pj) + f_ipj*f_ip1 + f_ipj*f_i - f_ip1*f_ip1pj - f_i*f_ip1pj - (f_ipj*f_ipj);
        b1 += -2.0*f_ipj*f_ip1pj - 2.0*(f_ip1pj*f_ip1pj) - 2.0*f_ipj*f_ip1 - 2.0*f_ip2pj*f_ip1 + 4.0*f_ip1*f_ip1pj + 2.0*f_ip2pj*f_ip1pj + 2.0*(f_ipj*f_ipj);
        b2 += 2.0*f_ipj*f_ip1pj - f_ip2pj*f_i + f_ipj*f_ip1 - f_ipj*f_i - 2.0*f_ip1*f_ip1pj + f_ip2pj*f_ip1 - 2.0*f_ip2pj*f_ip1pj + 2.0*f_i*f_ip1pj + (f_ip2pj*f_ip2pj) - (f_ipj*f_ipj);
      }
      b0 += -(f_m2pn*f_m2pn) - f_m2mjpn*f_m1pn + f_m2mjpn*f_m2pn + f_m1mjpn*f_m2pn - (f_m1mjpn*f_m1mjpn) + f_m1mjpn*f_m1pn;
      b1 += -2.0*f_m1mjpn*f_m2mjpn + 2.0*(f_m2pn*f_m2pn) - 2.0*f_m1pn*f_m2pn - 2.0*f_m1mjpn*f_m2pn + 2.0*(f_m1mjpn*f_m1mjpn) + 2.0*f_m1pn*f_m2mjpn;
      b2 += -(f_m1mjpn*f_m1mjpn) - (f_m1pn*f_m1pn) - (f_m2pn*f_m2pn) + 2.0*f_m2mjpn*f_m1mjpn - f_m1mjpn*f_m1pn + f_m2mjpn*f_m1pn - f_m2mjpn*f_m2pn + 2.0*f_m2pn*f_m1pn + f_m1mjpn*f_m2pn - (f_m2mjpn*f_m2mjpn);
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int cf_e11_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e11_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    cf_quadratic_approximation_1_2(a1_0, a1_1, a1_2, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e11_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e11_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    s = cf_find_real_zero_in_01_2(a1_0, a1_1, a1_2);
    //printf("j,s=%d, %f\n",j,s);
    if (s>=0.0 && s<=1.0)
      {
        *result = (double) (j) + s;
        status = 0;
        break;
      }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e11_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=2) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double f_ipj, f_ip2pj, f_ip1pj, f_m1mjpn, f_i, f_m2mjpn, f_m2pn, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m1mjpn = F(-1-j+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip2pj = F(i+2+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip1 = F(i+1);
        b0 += -2.0*f_ipj*f_ip1pj - 2.0*(f_ip1pj*f_ip1pj) - 2.0*f_ipj*f_ip1 - 2.0*f_ip2pj*f_ip1 + 4.0*f_ip1*f_ip1pj + 2.0*f_ip2pj*f_ip1pj + 2.0*(f_ipj*f_ipj);
        b1 += 4.0*f_ipj*f_ip1pj - 2.0*f_ip2pj*f_i + 2.0*f_ipj*f_ip1 - 2.0*f_ipj*f_i - 4.0*f_ip1*f_ip1pj + 2.0*f_ip2pj*f_ip1 - 4.0*f_ip2pj*f_ip1pj + 4.0*f_i*f_ip1pj + 2.0*(f_ip2pj*f_ip2pj) - 2.0*(f_ipj*f_ipj);
      }
      b0 += -2.0*f_m1pn*f_m2pn + 2.0*(f_m2pn*f_m2pn) - 2.0*f_m1mjpn*f_m2mjpn + 2.0*f_m1pn*f_m2mjpn + 2.0*(f_m1mjpn*f_m1mjpn) - 2.0*f_m1mjpn*f_m2pn;
      b1 += -2.0*(f_m1mjpn*f_m1mjpn) - 2.0*(f_m1pn*f_m1pn) - 2.0*(f_m2pn*f_m2pn) + 4.0*f_m2mjpn*f_m1mjpn + 2.0*f_m1mjpn*f_m2pn - 2.0*f_m1mjpn*f_m1pn + 2.0*f_m2mjpn*f_m1pn + 4.0*f_m2pn*f_m1pn - 2.0*f_m2mjpn*f_m2pn - 2.0*(f_m2mjpn*f_m2mjpn);
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int cf_e11_find_extreme_diff2(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a2_2 = 0.0;
  double a2_3 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  double a3_2 = 0.0;
  double a3_3 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a1_2 = a2_2;
    a1_3 = a2_3;
    a2_0 = a3_0;
    a2_1 = a3_1;
    a2_2 = a3_2;
    a2_3 = a3_3;
    cf_e11_compute_coeffs_diff2(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3);
    count ++;
    if (count<3)
      continue;
    cf_quadratic_approximation_3_1(a1_0, a1_1, a2_0, a2_1, a3_0, a3_1, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e11_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e11_compute_coeffs_diff2(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    cf_linear_approximation_1_1(a1_0, a1_1, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e11_find_zero_diff2: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e11_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=3) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double f_ipj, f_ip2pj, f_ip1pj, f_m1mjpn, f_i, f_m2mjpn, f_m2pn, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m1mjpn = F(-1-j+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip2pj = F(i+2+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip1 = F(i+1);
        b0 += 4.0*f_ipj*f_ip1pj - 2.0*f_ip2pj*f_i + 2.0*f_ipj*f_ip1 - 2.0*f_ipj*f_i - 4.0*f_ip1*f_ip1pj + 2.0*f_ip2pj*f_ip1 - 4.0*f_ip2pj*f_ip1pj + 4.0*f_i*f_ip1pj + 2.0*(f_ip2pj*f_ip2pj) - 2.0*(f_ipj*f_ipj);
      }
      b0 += -2.0*(f_m1mjpn*f_m1mjpn) - 2.0*(f_m1pn*f_m1pn) - 2.0*(f_m2pn*f_m2pn) + 4.0*f_m2mjpn*f_m1mjpn - 2.0*f_m2mjpn*f_m2pn + 2.0*f_m1mjpn*f_m2pn - 2.0*f_m1mjpn*f_m1pn + 4.0*f_m2pn*f_m1pn + 2.0*f_m2mjpn*f_m1pn - 2.0*(f_m2mjpn*f_m2mjpn);
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int cf_e11_find_extreme_diff3(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a2_2 = 0.0;
  double a2_3 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  double a3_2 = 0.0;
  double a3_3 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a1_2 = a2_2;
    a1_3 = a2_3;
    a2_0 = a3_0;
    a2_1 = a3_1;
    a2_2 = a3_2;
    a2_3 = a3_3;
    cf_e11_compute_coeffs_diff3(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3);
    count ++;
    if (count<3)
      continue;
    cf_quadratic_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e11_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a2_2 = 0.0;
  double a2_3 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  double a3_2 = 0.0;
  double a3_3 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a1_2 = a2_2;
    a1_3 = a2_3;
    a2_0 = a3_0;
    a2_1 = a3_1;
    a2_2 = a3_2;
    a2_3 = a3_3;
    cf_e11_compute_coeffs_diff3(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3);
    count ++;
    if (count<3)
      continue;
    cf_linear_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1);
    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e11_find_zero_diff3: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
void cf_e11_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */
  switch (order)
  {
    case 0: cf_e11_compute_coeffs_diff0(j, fm, n, m, a0, a1, a2, a3); break;
    case 1: cf_e11_compute_coeffs_diff1(j, fm, n, m, a0, a1, a2, a3); break;
    case 2: cf_e11_compute_coeffs_diff2(j, fm, n, m, a0, a1, a2, a3); break;
    case 3: cf_e11_compute_coeffs_diff3(j, fm, n, m, a0, a1, a2, a3); break;
    default:
      *a0 = 0.0;
      *a1 = 0.0;
      *a2 = 0.0;
      *a3 = 0.0;
  }
}
        
int cf_e11_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  switch (order)
  {
    case 0: return cf_e11_find_extreme_diff0(j0, j1, fm, n, m, result);
    case 1: return cf_e11_find_extreme_diff1(j0, j1, fm, n, m, result);
    case 2: return cf_e11_find_extreme_diff2(j0, j1, fm, n, m, result);
    case 3: return cf_e11_find_extreme_diff3(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
int cf_e11_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  switch (order)
  {
    case 0: return cf_e11_find_zero_diff0(j0, j1, fm, n, m, result);
    case 1: return cf_e11_find_zero_diff1(j0, j1, fm, n, m, result);
    case 2: return cf_e11_find_zero_diff2(j0, j1, fm, n, m, result);
    case 3: return cf_e11_find_zero_diff3(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
double cf_e11_evaluate(double y, double *fm, int n, int m, int order)
{
  double a0 = 0.0;
  double a1 = 0.0;
  double a2 = 0.0;
  double a3 = 0.0;
  /*
  int j = floor((y<0?-y:y));
  double r = (y<0?-y:y) - j;
  */
  int j = floor (y);
  double r = y - j;
  cf_e11_compute_coeffs(j, fm, n, m, order, &a0, &a1, &a2, &a3);
  return a0+(a1+(a2+(a3)*r)*r)*r;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double cf_e11_f1_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  switch (order)
  {
    case 0: return F(i) + (-(F(i)) + (F(i+1)))*s;
    case 1: return -(F(i)) + (F(i+1));
  }
  return 0.0;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double cf_e11_f2_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  switch (order)
  {
    case 0: return F(i) + (-(F(i)) + (F(i+1)))*s;
    case 1: return -(F(i)) + (F(i+1));
  }
  return 0.0;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a00_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1)
{
  /* int(f1(x)*f2(x+y), x=0..L-y) = sum(a_k*r^k, k=0..1) where y=j+r
     f1(x)=sum([0<=s<1]*(F(i)), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*(F(i)), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double f_ipj, f_ip1pj, f_m2pn, f_i, f_m2mjpn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m2pn = F(-2+n);
      f_m2mjpn = F(-2-j+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        b0 += f_ipj*f_i;
        b1 += f_i*f_ip1pj - f_ipj*f_i;
      }
      b0 += f_m2mjpn*f_m2pn;
      b1 += -f_m2mjpn*f_m2pn;
    }
  }
  *a0 = b0;
  *a1 = b1;
}
        
int cf_a00_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a2_0 = a3_0;
    a2_1 = a3_1;
    cf_a00_compute_coeffs_diff0(j, fm, n, m, &a3_0, &a3_1);
    count ++;
    if (count<3)
      continue;
    cf_quadratic_approximation_3_1(a1_0, a1_1, a2_0, a2_1, a3_0, a3_1, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a00_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a00_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1);
    cf_linear_approximation_1_1(a1_0, a1_1, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a00_find_zero_diff0: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a00_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order=1) = sum(a_k*r^k, k=0..1) where y=j+r
     f1(x)=sum([0<=s<1]*(F(i)), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*(F(i)), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double f_ipj, f_ip1pj, f_m2pn, f_i, f_m2mjpn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m2pn = F(-2+n);
      f_m2mjpn = F(-2-j+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        b0 += f_i*f_ip1pj - f_ipj*f_i;
      }
      b0 += -f_m2mjpn*f_m2pn;
    }
  }
  *a0 = b0;
  *a1 = b1;
}
        
int cf_a00_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a2_0 = a3_0;
    a2_1 = a3_1;
    cf_a00_compute_coeffs_diff1(j, fm, n, m, &a3_0, &a3_1);
    count ++;
    if (count<3)
      continue;
    cf_quadratic_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a00_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a2_0 = a3_0;
    a2_1 = a3_1;
    cf_a00_compute_coeffs_diff1(j, fm, n, m, &a3_0, &a3_1);
    count ++;
    if (count<3)
      continue;
    cf_linear_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1);
    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a00_find_zero_diff1: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
void cf_a00_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order) = sum(a_k*r^k, k=0..1) where y=j+r
     f1(x)=sum([0<=s<1]*(F(i)), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*(F(i)), i=0..N-1) where s=x-i */
  switch (order)
  {
    case 0: cf_a00_compute_coeffs_diff0(j, fm, n, m, a0, a1); break;
    case 1: cf_a00_compute_coeffs_diff1(j, fm, n, m, a0, a1); break;
    default:
      *a0 = 0.0;
      *a1 = 0.0;
  }
}
        
int cf_a00_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  switch (order)
  {
    case 0: return cf_a00_find_extreme_diff0(j0, j1, fm, n, m, result);
    case 1: return cf_a00_find_extreme_diff1(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
int cf_a00_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  switch (order)
  {
    case 0: return cf_a00_find_zero_diff0(j0, j1, fm, n, m, result);
    case 1: return cf_a00_find_zero_diff1(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
double cf_a00_evaluate(double y, double *fm, int n, int m, int order)
{
  double a0 = 0.0;
  double a1 = 0.0;
  /*
  int j = floor((y<0?-y:y));
  double r = (y<0?-y:y) - j;
  */
  int j = floor (y);
  double r = y - j;
  cf_a00_compute_coeffs(j, fm, n, m, order, &a0, &a1);
  return a0+(a1)*r;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double cf_a00_f1_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  switch (order)
  {
    case 0: return F(i);
  }
  return 0.0;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double cf_a00_f2_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  switch (order)
  {
    case 0: return F(i);
  }
  return 0.0;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e00_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1)
{
  /* int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y) = sum(a_k*r^k, k=0..1) where y=j+r
     f1(x)=sum([0<=s<1]*(F(i)), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*(F(i)), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double f_ipj, f_ip1pj, f_m2pn, f_i, f_m2mjpn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m2pn = F(-2+n);
      f_m2mjpn = F(-2-j+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        b0 += (f_i*f_i) + (f_ipj*f_ipj) - 2.0*f_ipj*f_i;
        b1 += (f_ip1pj*f_ip1pj) - (f_ipj*f_ipj) + 2.0*f_ipj*f_i - 2.0*f_i*f_ip1pj;
      }
      b0 += (f_m2pn*f_m2pn) + (f_m2mjpn*f_m2mjpn) - 2.0*f_m2pn*f_m2mjpn;
      b1 += -(f_m2mjpn*f_m2mjpn) - (f_m2pn*f_m2pn) + 2.0*f_m2pn*f_m2mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
}
        
int cf_e00_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a2_0 = a3_0;
    a2_1 = a3_1;
    cf_e00_compute_coeffs_diff0(j, fm, n, m, &a3_0, &a3_1);
    count ++;
    if (count<3)
      continue;
    cf_quadratic_approximation_3_1(a1_0, a1_1, a2_0, a2_1, a3_0, a3_1, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e00_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_e00_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1);
    cf_linear_approximation_1_1(a1_0, a1_1, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e00_find_zero_diff0: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_e00_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=1) = sum(a_k*r^k, k=0..1) where y=j+r
     f1(x)=sum([0<=s<1]*(F(i)), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*(F(i)), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double f_ipj, f_ip1pj, f_m2pn, f_i, f_m2mjpn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m2pn = F(-2+n);
      f_m2mjpn = F(-2-j+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        b0 += (f_ip1pj*f_ip1pj) - (f_ipj*f_ipj) + 2.0*f_ipj*f_i - 2.0*f_i*f_ip1pj;
      }
      b0 += -(f_m2pn*f_m2pn) - (f_m2mjpn*f_m2mjpn) + 2.0*f_m2pn*f_m2mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
}
        
int cf_e00_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a2_0 = a3_0;
    a2_1 = a3_1;
    cf_e00_compute_coeffs_diff1(j, fm, n, m, &a3_0, &a3_1);
    count ++;
    if (count<3)
      continue;
    cf_quadratic_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_e00_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a2_0 = a3_0;
    a2_1 = a3_1;
    cf_e00_compute_coeffs_diff1(j, fm, n, m, &a3_0, &a3_1);
    count ++;
    if (count<3)
      continue;
    cf_linear_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1);
    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_e00_find_zero_diff1: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
void cf_e00_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order) = sum(a_k*r^k, k=0..1) where y=j+r
     f1(x)=sum([0<=s<1]*(F(i)), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*(F(i)), i=0..N-1) where s=x-i */
  switch (order)
  {
    case 0: cf_e00_compute_coeffs_diff0(j, fm, n, m, a0, a1); break;
    case 1: cf_e00_compute_coeffs_diff1(j, fm, n, m, a0, a1); break;
    default:
      *a0 = 0.0;
      *a1 = 0.0;
  }
}
        
int cf_e00_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  switch (order)
  {
    case 0: return cf_e00_find_extreme_diff0(j0, j1, fm, n, m, result);
    case 1: return cf_e00_find_extreme_diff1(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
int cf_e00_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  switch (order)
  {
    case 0: return cf_e00_find_zero_diff0(j0, j1, fm, n, m, result);
    case 1: return cf_e00_find_zero_diff1(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
double cf_e00_evaluate(double y, double *fm, int n, int m, int order)
{
  double a0 = 0.0;
  double a1 = 0.0;
  /*
  int j = floor((y<0?-y:y));
  double r = (y<0?-y:y) - j;
  */
  int j = floor (y);
  double r = y - j;
  cf_e00_compute_coeffs(j, fm, n, m, order, &a0, &a1);
  return a0+(a1)*r;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double cf_e00_f1_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  switch (order)
  {
    case 0: return F(i);
  }
  return 0.0;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double cf_e00_f2_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  switch (order)
  {
    case 0: return F(i);
  }
  return 0.0;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a11_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* int(f1(x)*f2(x+y), x=0..L-y) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double f_ipj, f_ip2pj, f_ip1pj, f_m1mjpn, f_i, f_m2mjpn, f_m2pn, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m1mjpn = F(-1-j+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip2pj = F(i+2+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip1 = F(i+1);
        b0 += 0.1666666666666667*f_i*f_ip1pj + 0.3333333333333333*f_ipj*f_i + 0.3333333333333333*f_ip1*f_ip1pj + 0.1666666666666667*f_ipj*f_ip1;
        b1 += 0.5*f_i*f_ip1pj - 0.5*f_ipj*f_i + 0.5*f_ip1*f_ip1pj - 0.5*f_ipj*f_ip1;
        b2 += 0.5*f_ip2pj*f_ip1 - f_ip1*f_ip1pj + 0.5*f_ipj*f_ip1;
        b3 += -0.1666666666666667*f_ipj*f_ip1 + 0.1666666666666667*f_ip2pj*f_i - 0.1666666666666667*f_ip2pj*f_ip1 + 0.3333333333333333*f_ip1*f_ip1pj + 0.1666666666666667*f_ipj*f_i - 0.3333333333333333*f_i*f_ip1pj;
      }
      b0 += 0.1666666666666667*f_m2pn*f_m1mjpn + 0.1666666666666667*f_m1pn*f_m2mjpn + 0.3333333333333333*f_m1pn*f_m1mjpn + 0.3333333333333333*f_m2mjpn*f_m2pn;
      b1 += -0.5*f_m2pn*f_m1mjpn + 0.5*f_m1pn*f_m2mjpn - 0.5*f_m1pn*f_m1mjpn - 0.5*f_m2mjpn*f_m2pn;
      b2 += 0.5*f_m2pn*f_m1mjpn - 0.5*f_m2mjpn*f_m1pn;
      b3 += 0.1666666666666667*f_m2pn*f_m2mjpn - 0.1666666666666667*f_m2pn*f_m1mjpn - 0.1666666666666667*f_m1pn*f_m2mjpn + 0.1666666666666667*f_m1pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int cf_a11_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a11_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    cf_quadratic_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a11_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a11_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    cf_linear_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a11_find_zero_diff0: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a11_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order=1) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double f_ipj, f_ip2pj, f_ip1pj, f_m1mjpn, f_i, f_m2mjpn, f_m2pn, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m1mjpn = F(-1-j+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip2pj = F(i+2+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip1 = F(i+1);
        b0 += 0.5*f_i*f_ip1pj - 0.5*f_ipj*f_i + 0.5*f_ip1*f_ip1pj - 0.5*f_ipj*f_ip1;
        b1 += f_ip2pj*f_ip1 - 2.0*f_ip1*f_ip1pj + f_ipj*f_ip1;
        b2 += -0.5*f_ipj*f_ip1 + 0.5*f_ip2pj*f_i - 0.5*f_ip2pj*f_ip1 + f_ip1*f_ip1pj + 0.5*f_ipj*f_i - f_i*f_ip1pj;
      }
      b0 += -0.5*f_m2mjpn*f_m2pn - 0.5*f_m2pn*f_m1mjpn - 0.5*f_m1pn*f_m1mjpn + 0.5*f_m1pn*f_m2mjpn;
      b1 += f_m2pn*f_m1mjpn - f_m2mjpn*f_m1pn;
      b2 += 0.5*f_m1pn*f_m1mjpn + 0.5*f_m2pn*f_m2mjpn - 0.5*f_m1pn*f_m2mjpn - 0.5*f_m2pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int cf_a11_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a11_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    cf_quadratic_approximation_1_2(a1_0, a1_1, a1_2, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a11_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a11_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    s = cf_find_real_zero_in_01_2(a1_0, a1_1, a1_2);
    //printf("j,s=%d, %f\n",j,s);
    if (s>=0.0 && s<=1.0)
      {
        *result = (double) (j) + s;
        status = 0;
        break;
      }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a11_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order=2) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double f_ip2pj, f_ipj, f_ip1pj, f_m1mjpn, f_i, f_m2mjpn, f_m2pn, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m1mjpn = F(-1-j+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ip2pj = F(i+2+j);
        f_ipj = F(i+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip1 = F(i+1);
        b0 += f_ip2pj*f_ip1 - 2.0*f_ip1*f_ip1pj + f_ipj*f_ip1;
        b1 += -f_ipj*f_ip1 + f_ip2pj*f_i - f_ip2pj*f_ip1 + 2.0*f_ip1*f_ip1pj + f_ipj*f_i - 2.0*f_i*f_ip1pj;
      }
      b0 += f_m2pn*f_m1mjpn - f_m2mjpn*f_m1pn;
      b1 += -f_m2pn*f_m1mjpn + f_m1pn*f_m1mjpn - f_m1pn*f_m2mjpn + f_m2pn*f_m2mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int cf_a11_find_extreme_diff2(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a2_2 = 0.0;
  double a2_3 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  double a3_2 = 0.0;
  double a3_3 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a1_2 = a2_2;
    a1_3 = a2_3;
    a2_0 = a3_0;
    a2_1 = a3_1;
    a2_2 = a3_2;
    a2_3 = a3_3;
    cf_a11_compute_coeffs_diff2(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3);
    count ++;
    if (count<3)
      continue;
    cf_quadratic_approximation_3_1(a1_0, a1_1, a2_0, a2_1, a3_0, a3_1, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a11_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_a11_compute_coeffs_diff2(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    cf_linear_approximation_1_1(a1_0, a1_1, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a11_find_zero_diff2: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void cf_a11_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order=3) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double f_ipj, f_ip2pj, f_ip1pj, f_m1mjpn, f_i, f_m2mjpn, f_m2pn, f_ip1, f_m1pn;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      f_m1mjpn = F(-1-j+n);
      f_m2mjpn = F(-2-j+n);
      f_m2pn = F(-2+n);
      f_m1pn = F(-1+n);
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip2pj = F(i+2+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip1 = F(i+1);
        b0 += -f_ipj*f_ip1 + f_ip2pj*f_i - f_ip2pj*f_ip1 + 2.0*f_ip1*f_ip1pj + f_ipj*f_i - 2.0*f_i*f_ip1pj;
      }
      b0 += f_m2pn*f_m2mjpn - f_m2pn*f_m1mjpn - f_m1pn*f_m2mjpn + f_m1pn*f_m1mjpn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int cf_a11_find_extreme_diff3(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a2_2 = 0.0;
  double a2_3 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  double a3_2 = 0.0;
  double a3_3 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a1_2 = a2_2;
    a1_3 = a2_3;
    a2_0 = a3_0;
    a2_1 = a3_1;
    a2_2 = a3_2;
    a2_3 = a3_3;
    cf_a11_compute_coeffs_diff3(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3);
    count ++;
    if (count<3)
      continue;
    cf_quadratic_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
int cf_a11_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  double a2_0 = 0.0;
  double a2_1 = 0.0;
  double a2_2 = 0.0;
  double a2_3 = 0.0;
  double a3_0 = 0.0;
  double a3_1 = 0.0;
  double a3_2 = 0.0;
  double a3_3 = 0.0;
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    a1_0 = a2_0;
    a1_1 = a2_1;
    a1_2 = a2_2;
    a1_3 = a2_3;
    a2_0 = a3_0;
    a2_1 = a3_1;
    a2_2 = a3_2;
    a2_3 = a3_3;
    cf_a11_compute_coeffs_diff3(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3);
    count ++;
    if (count<3)
      continue;
    cf_linear_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1);
    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_a11_find_zero_diff3: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
void cf_a11_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int(f1(x)*f2(x+y), x=0..L-y), y, order) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */
  switch (order)
  {
    case 0: cf_a11_compute_coeffs_diff0(j, fm, n, m, a0, a1, a2, a3); break;
    case 1: cf_a11_compute_coeffs_diff1(j, fm, n, m, a0, a1, a2, a3); break;
    case 2: cf_a11_compute_coeffs_diff2(j, fm, n, m, a0, a1, a2, a3); break;
    case 3: cf_a11_compute_coeffs_diff3(j, fm, n, m, a0, a1, a2, a3); break;
    default:
      *a0 = 0.0;
      *a1 = 0.0;
      *a2 = 0.0;
      *a3 = 0.0;
  }
}
        
int cf_a11_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  switch (order)
  {
    case 0: return cf_a11_find_extreme_diff0(j0, j1, fm, n, m, result);
    case 1: return cf_a11_find_extreme_diff1(j0, j1, fm, n, m, result);
    case 2: return cf_a11_find_extreme_diff2(j0, j1, fm, n, m, result);
    case 3: return cf_a11_find_extreme_diff3(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
int cf_a11_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  switch (order)
  {
    case 0: return cf_a11_find_zero_diff0(j0, j1, fm, n, m, result);
    case 1: return cf_a11_find_zero_diff1(j0, j1, fm, n, m, result);
    case 2: return cf_a11_find_zero_diff2(j0, j1, fm, n, m, result);
    case 3: return cf_a11_find_zero_diff3(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
double cf_a11_evaluate(double y, double *fm, int n, int m, int order)
{
  double a0 = 0.0;
  double a1 = 0.0;
  double a2 = 0.0;
  double a3 = 0.0;
  /*
  int j = floor((y<0?-y:y));
  double r = (y<0?-y:y) - j;
  */
  int j = floor (y);
  double r = y - j;
  cf_a11_compute_coeffs(j, fm, n, m, order, &a0, &a1, &a2, &a3);
  return a0+(a1+(a2+(a3)*r)*r)*r;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double cf_a11_f1_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  switch (order)
  {
    case 0: return F(i) + (-(F(i)) + (F(i+1)))*s;
    case 1: return -(F(i)) + (F(i+1));
  }
  return 0.0;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double cf_a11_f2_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  switch (order)
  {
    case 0: return F(i) + (-(F(i)) + (F(i+1)))*s;
    case 1: return -(F(i)) + (F(i+1));
  }
  return 0.0;
}
        
void cf_quadratic_approximation_3_0(double a1_0, double a2_0, double a3_0, double* p0, double* p1, double* p2)
{
  *p0 = 4.3209876543209880e-01*a1_0+5.8024691358024694e-01*a2_0+-1.2345679012345680e-02*a3_0;
  *p1 = -6.9135802469135799e-01*a1_0+4.9382716049382719e-01*a2_0+1.9753086419753091e-01*a3_0;
  *p2 = 2.4691358024691359e-01*a1_0+-4.9382716049382719e-01*a2_0+2.4691358024691359e-01*a3_0;
}
            
void cf_quadratic_approximation_1_0(double a1_0, double* p0, double* p1, double* p2)
{
  *p0 = 1.0000000000000000e+00*a1_0;
  *p1 = 0.0000000000000000e+00*a1_0;
  *p2 = 0.0000000000000000e+00*a1_0;
}
            
void cf_linear_approximation_3_0(double a1_0, double a2_0, double a3_0, double* p0, double* p1)
{
  *p0 = 5.5555555555555558e-01*a1_0+3.3333333333333331e-01*a2_0+1.1111111111111110e-01*a3_0;
  *p1 = -4.4444444444444442e-01*a1_0+0.0000000000000000e+00*a2_0+4.4444444444444442e-01*a3_0;
}
            
void cf_linear_approximation_1_0(double a1_0, double* p0, double* p1)
{
  *p0 = 1.0000000000000000e+00*a1_0;
  *p1 = 0.0000000000000000e+00*a1_0;
}
            
void cf_quadratic_approximation_3_1(double a1_0, double a1_1, double a2_0, double a2_1, double a3_0, double a3_1, double* p0, double* p1, double* p2)
{
  *p0 = 4.3209876543209880e-01*a1_0+2.5925925925925930e-01*a1_1+5.8024691358024694e-01*a2_0+2.7160493827160492e-01*a2_1+-1.2345679012345680e-02*a3_0+-8.6419753086419748e-02*a3_1;
  *p1 = -6.9135802469135799e-01*a1_0+-1.8518518518518520e-01*a1_1+4.9382716049382719e-01*a2_0+2.8395061728395060e-01*a2_1+1.9753086419753091e-01*a3_0+1.2345679012345680e-02*a3_1;
  *p2 = 2.4691358024691359e-01*a1_0+0.0000000000000000e+00*a1_1+-4.9382716049382719e-01*a2_0+-2.4691358024691359e-01*a2_1+2.4691358024691359e-01*a3_0+2.4691358024691359e-01*a3_1;
}
            
void cf_quadratic_approximation_1_1(double a1_0, double a1_1, double* p0, double* p1, double* p2)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1;
  *p2 = 0.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1;
}
            
void cf_linear_approximation_3_1(double a1_0, double a1_1, double a2_0, double a2_1, double a3_0, double a3_1, double* p0, double* p1)
{
  *p0 = 5.5555555555555558e-01*a1_0+2.5925925925925930e-01*a1_1+3.3333333333333331e-01*a2_0+1.4814814814814811e-01*a2_1+1.1111111111111110e-01*a3_0+3.7037037037037042e-02*a3_1;
  *p1 = -4.4444444444444442e-01*a1_0+-1.8518518518518520e-01*a1_1+0.0000000000000000e+00*a2_0+3.7037037037037042e-02*a2_1+4.4444444444444442e-01*a3_0+2.5925925925925930e-01*a3_1;
}
            
void cf_linear_approximation_1_1(double a1_0, double a1_1, double* p0, double* p1)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1;
}
            
void cf_quadratic_approximation_3_2(double a1_0, double a1_1, double a1_2, double a2_0, double a2_1, double a2_2, double a3_0, double a3_1, double a3_2, double* p0, double* p1, double* p2)
{
  *p0 = 4.3209876543209880e-01*a1_0+2.5925925925925930e-01*a1_1+1.8518518518518520e-01*a1_2+5.8024691358024694e-01*a2_0+2.7160493827160492e-01*a2_1+1.7283950617283950e-01*a2_2+-1.2345679012345680e-02*a3_0+-8.6419753086419748e-02*a3_1+-8.6419753086419748e-02*a3_2;
  *p1 = -6.9135802469135799e-01*a1_0+-1.8518518518518520e-01*a1_1+-7.4074074074074070e-02*a1_2+4.9382716049382719e-01*a2_0+2.8395061728395060e-01*a2_1+1.9753086419753091e-01*a2_2+1.9753086419753091e-01*a3_0+1.2345679012345680e-02*a3_1+-2.4691358024691360e-02*a3_2;
  *p2 = 2.4691358024691359e-01*a1_0+0.0000000000000000e+00*a1_1+-3.7037037037037042e-02*a1_2+-4.9382716049382719e-01*a2_0+-2.4691358024691359e-01*a2_1+-1.6049382716049379e-01*a2_2+2.4691358024691359e-01*a3_0+2.4691358024691359e-01*a3_1+2.0987654320987650e-01*a3_2;
}
            
void cf_quadratic_approximation_1_2(double a1_0, double a1_1, double a1_2, double* p0, double* p1, double* p2)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2;
  *p2 = 0.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2;
}
            
void cf_linear_approximation_3_2(double a1_0, double a1_1, double a1_2, double a2_0, double a2_1, double a2_2, double a3_0, double a3_1, double a3_2, double* p0, double* p1)
{
  *p0 = 5.5555555555555558e-01*a1_0+2.5925925925925930e-01*a1_1+1.6666666666666671e-01*a1_2+3.3333333333333331e-01*a2_0+1.4814814814814811e-01*a2_1+9.2592592592592587e-02*a2_2+1.1111111111111110e-01*a3_0+3.7037037037037042e-02*a3_1+1.8518518518518521e-02*a3_2;
  *p1 = -4.4444444444444442e-01*a1_0+-1.8518518518518520e-01*a1_1+-1.1111111111111110e-01*a1_2+0.0000000000000000e+00*a2_0+3.7037037037037042e-02*a2_1+3.7037037037037042e-02*a2_2+4.4444444444444442e-01*a3_0+2.5925925925925930e-01*a3_1+1.8518518518518520e-01*a3_2;
}
            
void cf_linear_approximation_1_2(double a1_0, double a1_1, double a1_2, double* p0, double* p1)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+-1.6666666666666671e-01*a1_2;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2;
}
            
void cf_quadratic_approximation_3_3(double a1_0, double a1_1, double a1_2, double a1_3, double a2_0, double a2_1, double a2_2, double a2_3, double a3_0, double a3_1, double a3_2, double a3_3, double* p0, double* p1, double* p2)
{
  *p0 = 4.3209876543209880e-01*a1_0+2.5925925925925930e-01*a1_1+1.8518518518518520e-01*a1_2+1.4382716049382721e-01*a1_3+5.8024691358024694e-01*a2_0+2.7160493827160492e-01*a2_1+1.7283950617283950e-01*a2_2+1.2530864197530861e-01*a2_3+-1.2345679012345680e-02*a3_0+-8.6419753086419748e-02*a3_1+-8.6419753086419748e-02*a3_2+-7.8395061728395055e-02*a3_3;
  *p1 = -6.9135802469135799e-01*a1_0+-1.8518518518518520e-01*a1_1+-7.4074074074074070e-02*a1_2+-3.4567901234567898e-02*a1_3+4.9382716049382719e-01*a2_0+2.8395061728395060e-01*a2_1+1.9753086419753091e-01*a2_2+1.5061728395061730e-01*a2_3+1.9753086419753091e-01*a3_0+1.2345679012345680e-02*a3_1+-2.4691358024691360e-02*a3_2+-3.4567901234567898e-02*a3_3;
  *p2 = 2.4691358024691359e-01*a1_0+0.0000000000000000e+00*a1_1+-3.7037037037037042e-02*a1_2+-4.3209876543209881e-02*a1_3+-4.9382716049382719e-01*a2_0+-2.4691358024691359e-01*a2_1+-1.6049382716049379e-01*a2_2+-1.1728395061728400e-01*a2_3+2.4691358024691359e-01*a3_0+2.4691358024691359e-01*a3_1+2.0987654320987650e-01*a3_2+1.7901234567901231e-01*a3_3;
}
            
void cf_quadratic_approximation_1_3(double a1_0, double a1_1, double a1_2, double a1_3, double* p0, double* p1, double* p2)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+5.0000000000000003e-02*a1_3;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+-5.9999999999999998e-01*a1_3;
  *p2 = 0.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+1.5000000000000000e+00*a1_3;
}
            
void cf_linear_approximation_3_3(double a1_0, double a1_1, double a1_2, double a1_3, double a2_0, double a2_1, double a2_2, double a2_3, double a3_0, double a3_1, double a3_2, double a3_3, double* p0, double* p1)
{
  *p0 = 5.5555555555555558e-01*a1_0+2.5925925925925930e-01*a1_1+1.6666666666666671e-01*a1_2+1.2222222222222220e-01*a1_3+3.3333333333333331e-01*a2_0+1.4814814814814811e-01*a2_1+9.2592592592592587e-02*a2_2+6.6666666666666666e-02*a2_3+1.1111111111111110e-01*a3_0+3.7037037037037042e-02*a3_1+1.8518518518518521e-02*a3_2+1.1111111111111110e-02*a3_3;
  *p1 = -4.4444444444444442e-01*a1_0+-1.8518518518518520e-01*a1_1+-1.1111111111111110e-01*a1_2+-7.7777777777777779e-02*a1_3+0.0000000000000000e+00*a2_0+3.7037037037037042e-02*a2_1+3.7037037037037042e-02*a2_2+3.3333333333333333e-02*a2_3+4.4444444444444442e-01*a3_0+2.5925925925925930e-01*a3_1+1.8518518518518520e-01*a3_2+1.4444444444444440e-01*a3_3;
}
            
void cf_linear_approximation_1_3(double a1_0, double a1_1, double a1_2, double a1_3, double* p0, double* p1)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+-1.6666666666666671e-01*a1_2+-2.0000000000000001e-01*a1_3;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+9.0000000000000002e-01*a1_3;
}
            
void cf_quadratic_approximation_3_4(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double* p0, double* p1, double* p2)
{
  *p0 = 4.3209876543209880e-01*a1_0+2.5925925925925930e-01*a1_1+1.8518518518518520e-01*a1_2+1.4382716049382721e-01*a1_3+1.1746031746031750e-01*a1_4+5.8024691358024694e-01*a2_0+2.7160493827160492e-01*a2_1+1.7283950617283950e-01*a2_2+1.2530864197530861e-01*a2_3+9.7707231040564377e-02*a2_4+-1.2345679012345680e-02*a3_0+-8.6419753086419748e-02*a3_1+-8.6419753086419748e-02*a3_2+-7.8395061728395055e-02*a3_3+-7.0194003527336860e-02*a3_4;
  *p1 = -6.9135802469135799e-01*a1_0+-1.8518518518518520e-01*a1_1+-7.4074074074074070e-02*a1_2+-3.4567901234567898e-02*a1_3+-1.6931216931216929e-02*a1_4+4.9382716049382719e-01*a2_0+2.8395061728395060e-01*a2_1+1.9753086419753091e-01*a2_2+1.5061728395061730e-01*a2_3+1.2134038800705470e-01*a2_4+1.9753086419753091e-01*a3_0+1.2345679012345680e-02*a3_1+-2.4691358024691360e-02*a3_2+-3.4567901234567898e-02*a3_3+-3.6684303350970018e-02*a3_4;
  *p2 = 2.4691358024691359e-01*a1_0+0.0000000000000000e+00*a1_1+-3.7037037037037042e-02*a1_2+-4.3209876543209881e-02*a1_3+-4.2328042328042333e-02*a1_4+-4.9382716049382719e-01*a2_0+-2.4691358024691359e-01*a2_1+-1.6049382716049379e-01*a2_2+-1.1728395061728400e-01*a2_3+-9.1710758377425039e-02*a2_4+2.4691358024691359e-01*a3_0+2.4691358024691359e-01*a3_1+2.0987654320987650e-01*a3_2+1.7901234567901231e-01*a3_3+1.5520282186948850e-01*a3_4;
}
            
void cf_quadratic_approximation_1_4(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double* p0, double* p1, double* p2)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+5.0000000000000003e-02*a1_3+8.5714285714285715e-02*a1_4;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+-5.9999999999999998e-01*a1_3+-9.1428571428571426e-01*a1_4;
  *p2 = 0.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+1.5000000000000000e+00*a1_3+1.7142857142857140e+00*a1_4;
}
            
void cf_linear_approximation_3_4(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double* p0, double* p1)
{
  *p0 = 5.5555555555555558e-01*a1_0+2.5925925925925930e-01*a1_1+1.6666666666666671e-01*a1_2+1.2222222222222220e-01*a1_3+9.6296296296296297e-02*a1_4+3.3333333333333331e-01*a2_0+1.4814814814814811e-01*a2_1+9.2592592592592587e-02*a2_2+6.6666666666666666e-02*a2_3+5.1851851851851850e-02*a2_4+1.1111111111111110e-01*a3_0+3.7037037037037042e-02*a3_1+1.8518518518518521e-02*a3_2+1.1111111111111110e-02*a3_3+7.4074074074074068e-03*a3_4;
  *p1 = -4.4444444444444442e-01*a1_0+-1.8518518518518520e-01*a1_1+-1.1111111111111110e-01*a1_2+-7.7777777777777779e-02*a1_3+-5.9259259259259262e-02*a1_4+0.0000000000000000e+00*a2_0+3.7037037037037042e-02*a2_1+3.7037037037037042e-02*a2_2+3.3333333333333333e-02*a2_3+2.9629629629629631e-02*a2_4+4.4444444444444442e-01*a3_0+2.5925925925925930e-01*a3_1+1.8518518518518520e-01*a3_2+1.4444444444444440e-01*a3_3+1.1851851851851850e-01*a3_4;
}
            
void cf_linear_approximation_1_4(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double* p0, double* p1)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+-1.6666666666666671e-01*a1_2+-2.0000000000000001e-01*a1_3+-2.0000000000000001e-01*a1_4;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+9.0000000000000002e-01*a1_3+8.0000000000000004e-01*a1_4;
}
            
void cf_quadratic_approximation_3_5(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a2_5, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double a3_5, double* p0, double* p1, double* p2)
{
  *p0 = 4.3209876543209880e-01*a1_0+2.5925925925925930e-01*a1_1+1.8518518518518520e-01*a1_2+1.4382716049382721e-01*a1_3+1.1746031746031750e-01*a1_4+9.9206349206349215e-02*a1_5+5.8024691358024694e-01*a2_0+2.7160493827160492e-01*a2_1+1.7283950617283950e-01*a2_2+1.2530864197530861e-01*a2_3+9.7707231040564377e-02*a2_4+7.9805996472663135e-02*a2_5+-1.2345679012345680e-02*a3_0+-8.6419753086419748e-02*a3_1+-8.6419753086419748e-02*a3_2+-7.8395061728395055e-02*a3_3+-7.0194003527336860e-02*a3_4+-6.3051146384479714e-02*a3_5;
  *p1 = -6.9135802469135799e-01*a1_0+-1.8518518518518520e-01*a1_1+-7.4074074074074070e-02*a1_2+-3.4567901234567898e-02*a1_3+-1.6931216931216929e-02*a1_4+-7.9365079365079361e-03*a1_5+4.9382716049382719e-01*a2_0+2.8395061728395060e-01*a2_1+1.9753086419753091e-01*a2_2+1.5061728395061730e-01*a2_3+1.2134038800705470e-01*a2_4+1.0141093474426810e-01*a2_5+1.9753086419753091e-01*a3_0+1.2345679012345680e-02*a3_1+-2.4691358024691360e-02*a3_2+-3.4567901234567898e-02*a3_3+-3.6684303350970018e-02*a3_4+-3.6155202821869493e-02*a3_5;
  *p2 = 2.4691358024691359e-01*a1_0+0.0000000000000000e+00*a1_1+-3.7037037037037042e-02*a1_2+-4.3209876543209881e-02*a1_3+-4.2328042328042333e-02*a1_4+-3.9682539682539680e-02*a1_5+-4.9382716049382719e-01*a2_0+-2.4691358024691359e-01*a2_1+-1.6049382716049379e-01*a2_2+-1.1728395061728400e-01*a2_3+-9.1710758377425039e-02*a2_4+-7.4955908289241618e-02*a2_5+2.4691358024691359e-01*a3_0+2.4691358024691359e-01*a3_1+2.0987654320987650e-01*a3_2+1.7901234567901231e-01*a3_3+1.5520282186948850e-01*a3_4+1.3668430335097001e-01*a3_5;
}
            
void cf_quadratic_approximation_1_5(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double* p0, double* p1, double* p2)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+5.0000000000000003e-02*a1_3+8.5714285714285715e-02*a1_4+1.0714285714285710e-01*a1_5;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+-5.9999999999999998e-01*a1_3+-9.1428571428571426e-01*a1_4+-1.0714285714285710e+00*a1_5;
  *p2 = 0.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+1.5000000000000000e+00*a1_3+1.7142857142857140e+00*a1_4+1.7857142857142860e+00*a1_5;
}
            
void cf_linear_approximation_3_5(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a2_5, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double a3_5, double* p0, double* p1)
{
  *p0 = 5.5555555555555558e-01*a1_0+2.5925925925925930e-01*a1_1+1.6666666666666671e-01*a1_2+1.2222222222222220e-01*a1_3+9.6296296296296297e-02*a1_4+7.9365079365079361e-02*a1_5+3.3333333333333331e-01*a2_0+1.4814814814814811e-01*a2_1+9.2592592592592587e-02*a2_2+6.6666666666666666e-02*a2_3+5.1851851851851850e-02*a2_4+4.2328042328042333e-02*a2_5+1.1111111111111110e-01*a3_0+3.7037037037037042e-02*a3_1+1.8518518518518521e-02*a3_2+1.1111111111111110e-02*a3_3+7.4074074074074068e-03*a3_4+5.2910052910052907e-03*a3_5;
  *p1 = -4.4444444444444442e-01*a1_0+-1.8518518518518520e-01*a1_1+-1.1111111111111110e-01*a1_2+-7.7777777777777779e-02*a1_3+-5.9259259259259262e-02*a1_4+-4.7619047619047623e-02*a1_5+0.0000000000000000e+00*a2_0+3.7037037037037042e-02*a2_1+3.7037037037037042e-02*a2_2+3.3333333333333333e-02*a2_3+2.9629629629629631e-02*a2_4+2.6455026455026460e-02*a2_5+4.4444444444444442e-01*a3_0+2.5925925925925930e-01*a3_1+1.8518518518518520e-01*a3_2+1.4444444444444440e-01*a3_3+1.1851851851851850e-01*a3_4+1.0052910052910050e-01*a3_5;
}
            
void cf_linear_approximation_1_5(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double* p0, double* p1)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+-1.6666666666666671e-01*a1_2+-2.0000000000000001e-01*a1_3+-2.0000000000000001e-01*a1_4+-1.9047619047619049e-01*a1_5;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+9.0000000000000002e-01*a1_3+8.0000000000000004e-01*a1_4+7.1428571428571430e-01*a1_5;
}
            
void cf_quadratic_approximation_3_6(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a2_5, double a2_6, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double a3_5, double a3_6, double* p0, double* p1, double* p2)
{
  *p0 = 4.3209876543209880e-01*a1_0+2.5925925925925930e-01*a1_1+1.8518518518518520e-01*a1_2+1.4382716049382721e-01*a1_3+1.1746031746031750e-01*a1_4+9.9206349206349215e-02*a1_5+8.5831863609641387e-02*a1_6+5.8024691358024694e-01*a2_0+2.7160493827160492e-01*a2_1+1.7283950617283950e-01*a2_2+1.2530864197530861e-01*a2_3+9.7707231040564377e-02*a2_4+7.9805996472663135e-02*a2_5+6.7313345091122870e-02*a2_6+-1.2345679012345680e-02*a3_0+-8.6419753086419748e-02*a3_1+-8.6419753086419748e-02*a3_2+-7.8395061728395055e-02*a3_3+-7.0194003527336860e-02*a3_4+-6.3051146384479714e-02*a3_5+-5.7025279247501469e-02*a3_6;
  *p1 = -6.9135802469135799e-01*a1_0+-1.8518518518518520e-01*a1_1+-7.4074074074074070e-02*a1_2+-3.4567901234567898e-02*a1_3+-1.6931216931216929e-02*a1_4+-7.9365079365079361e-03*a1_5+-2.9394473838918280e-03*a1_6+4.9382716049382719e-01*a2_0+2.8395061728395060e-01*a2_1+1.9753086419753091e-01*a2_2+1.5061728395061730e-01*a2_3+1.2134038800705470e-01*a2_4+1.0141093474426810e-01*a2_5+8.7007642563198123e-02*a2_6+1.9753086419753091e-01*a3_0+1.2345679012345680e-02*a3_1+-2.4691358024691360e-02*a3_2+-3.4567901234567898e-02*a3_3+-3.6684303350970018e-02*a3_4+-3.6155202821869493e-02*a3_5+-3.4685479129923570e-02*a3_6;
  *p2 = 2.4691358024691359e-01*a1_0+0.0000000000000000e+00*a1_1+-3.7037037037037042e-02*a1_2+-4.3209876543209881e-02*a1_3+-4.2328042328042333e-02*a1_4+-3.9682539682539680e-02*a1_5+-3.6743092298647848e-02*a1_6+-4.9382716049382719e-01*a2_0+-2.4691358024691359e-01*a2_1+-1.6049382716049379e-01*a2_2+-1.1728395061728400e-01*a2_3+-9.1710758377425039e-02*a2_4+-7.4955908289241618e-02*a2_5+-6.3198118753674315e-02*a2_6+2.4691358024691359e-01*a3_0+2.4691358024691359e-01*a3_1+2.0987654320987650e-01*a3_2+1.7901234567901231e-01*a3_3+1.5520282186948850e-01*a3_4+1.3668430335097001e-01*a3_5+1.2198706643151090e-01*a3_6;
}
            
void cf_quadratic_approximation_1_6(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double* p0, double* p1, double* p2)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+5.0000000000000003e-02*a1_3+8.5714285714285715e-02*a1_4+1.0714285714285710e-01*a1_5+1.1904761904761900e-01*a1_6;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+-5.9999999999999998e-01*a1_3+-9.1428571428571426e-01*a1_4+-1.0714285714285710e+00*a1_5+-1.1428571428571430e+00*a1_6;
  *p2 = 0.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+1.5000000000000000e+00*a1_3+1.7142857142857140e+00*a1_4+1.7857142857142860e+00*a1_5+1.7857142857142860e+00*a1_6;
}
            
void cf_linear_approximation_3_6(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a2_5, double a2_6, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double a3_5, double a3_6, double* p0, double* p1)
{
  *p0 = 5.5555555555555558e-01*a1_0+2.5925925925925930e-01*a1_1+1.6666666666666671e-01*a1_2+1.2222222222222220e-01*a1_3+9.6296296296296297e-02*a1_4+7.9365079365079361e-02*a1_5+6.7460317460317457e-02*a1_6+3.3333333333333331e-01*a2_0+1.4814814814814811e-01*a2_1+9.2592592592592587e-02*a2_2+6.6666666666666666e-02*a2_3+5.1851851851851850e-02*a2_4+4.2328042328042333e-02*a2_5+3.5714285714285712e-02*a2_6+1.1111111111111110e-01*a3_0+3.7037037037037042e-02*a3_1+1.8518518518518521e-02*a3_2+1.1111111111111110e-02*a3_3+7.4074074074074068e-03*a3_4+5.2910052910052907e-03*a3_5+3.9682539682539680e-03*a3_6;
  *p1 = -4.4444444444444442e-01*a1_0+-1.8518518518518520e-01*a1_1+-1.1111111111111110e-01*a1_2+-7.7777777777777779e-02*a1_3+-5.9259259259259262e-02*a1_4+-4.7619047619047623e-02*a1_5+-3.9682539682539680e-02*a1_6+0.0000000000000000e+00*a2_0+3.7037037037037042e-02*a2_1+3.7037037037037042e-02*a2_2+3.3333333333333333e-02*a2_3+2.9629629629629631e-02*a2_4+2.6455026455026460e-02*a2_5+2.3809523809523812e-02*a2_6+4.4444444444444442e-01*a3_0+2.5925925925925930e-01*a3_1+1.8518518518518520e-01*a3_2+1.4444444444444440e-01*a3_3+1.1851851851851850e-01*a3_4+1.0052910052910050e-01*a3_5+8.7301587301587297e-02*a3_6;
}
            
void cf_linear_approximation_1_6(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double* p0, double* p1)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+-1.6666666666666671e-01*a1_2+-2.0000000000000001e-01*a1_3+-2.0000000000000001e-01*a1_4+-1.9047619047619049e-01*a1_5+-1.7857142857142860e-01*a1_6;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+9.0000000000000002e-01*a1_3+8.0000000000000004e-01*a1_4+7.1428571428571430e-01*a1_5+6.4285714285714290e-01*a1_6;
}
            
void cf_quadratic_approximation_3_7(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double a1_7, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a2_5, double a2_6, double a2_7, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double a3_5, double a3_6, double a3_7, double* p0, double* p1, double* p2)
{
  *p0 = 4.3209876543209880e-01*a1_0+2.5925925925925930e-01*a1_1+1.8518518518518520e-01*a1_2+1.4382716049382721e-01*a1_3+1.1746031746031750e-01*a1_4+9.9206349206349215e-02*a1_5+8.5831863609641387e-02*a1_6+7.5617283950617287e-02*a1_7+5.8024691358024694e-01*a2_0+2.7160493827160492e-01*a2_1+1.7283950617283950e-01*a2_2+1.2530864197530861e-01*a2_3+9.7707231040564377e-02*a2_4+7.9805996472663135e-02*a2_5+6.7313345091122870e-02*a2_6+5.8127572016460911e-02*a2_7+-1.2345679012345680e-02*a3_0+-8.6419753086419748e-02*a3_1+-8.6419753086419748e-02*a3_2+-7.8395061728395055e-02*a3_3+-7.0194003527336860e-02*a3_4+-6.3051146384479714e-02*a3_5+-5.7025279247501469e-02*a3_6+-5.1954732510288072e-02*a3_7;
  *p1 = -6.9135802469135799e-01*a1_0+-1.8518518518518520e-01*a1_1+-7.4074074074074070e-02*a1_2+-3.4567901234567898e-02*a1_3+-1.6931216931216929e-02*a1_4+-7.9365079365079361e-03*a1_5+-2.9394473838918280e-03*a1_6+0.0000000000000000e+00*a1_7+4.9382716049382719e-01*a2_0+2.8395061728395060e-01*a2_1+1.9753086419753091e-01*a2_2+1.5061728395061730e-01*a2_3+1.2134038800705470e-01*a2_4+1.0141093474426810e-01*a2_5+8.7007642563198123e-02*a2_6+7.6131687242798354e-02*a2_7+1.9753086419753091e-01*a3_0+1.2345679012345680e-02*a3_1+-2.4691358024691360e-02*a3_2+-3.4567901234567898e-02*a3_3+-3.6684303350970018e-02*a3_4+-3.6155202821869493e-02*a3_5+-3.4685479129923570e-02*a3_6+-3.2921810699588480e-02*a3_7;
  *p2 = 2.4691358024691359e-01*a1_0+0.0000000000000000e+00*a1_1+-3.7037037037037042e-02*a1_2+-4.3209876543209881e-02*a1_3+-4.2328042328042333e-02*a1_4+-3.9682539682539680e-02*a1_5+-3.6743092298647848e-02*a1_6+-3.3950617283950622e-02*a1_7+-4.9382716049382719e-01*a2_0+-2.4691358024691359e-01*a2_1+-1.6049382716049379e-01*a2_2+-1.1728395061728400e-01*a2_3+-9.1710758377425039e-02*a2_4+-7.4955908289241618e-02*a2_5+-6.3198118753674315e-02*a2_6+-5.4526748971193417e-02*a2_7+2.4691358024691359e-01*a3_0+2.4691358024691359e-01*a3_1+2.0987654320987650e-01*a3_2+1.7901234567901231e-01*a3_3+1.5520282186948850e-01*a3_4+1.3668430335097001e-01*a3_5+1.2198706643151090e-01*a3_6+1.1008230452674900e-01*a3_7;
}
            
void cf_quadratic_approximation_1_7(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double a1_7, double* p0, double* p1, double* p2)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+5.0000000000000003e-02*a1_3+8.5714285714285715e-02*a1_4+1.0714285714285710e-01*a1_5+1.1904761904761900e-01*a1_6+1.2500000000000000e-01*a1_7;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+-5.9999999999999998e-01*a1_3+-9.1428571428571426e-01*a1_4+-1.0714285714285710e+00*a1_5+-1.1428571428571430e+00*a1_6+-1.1666666666666670e+00*a1_7;
  *p2 = 0.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+1.5000000000000000e+00*a1_3+1.7142857142857140e+00*a1_4+1.7857142857142860e+00*a1_5+1.7857142857142860e+00*a1_6+1.7500000000000000e+00*a1_7;
}
            
void cf_linear_approximation_3_7(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double a1_7, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a2_5, double a2_6, double a2_7, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double a3_5, double a3_6, double a3_7, double* p0, double* p1)
{
  *p0 = 5.5555555555555558e-01*a1_0+2.5925925925925930e-01*a1_1+1.6666666666666671e-01*a1_2+1.2222222222222220e-01*a1_3+9.6296296296296297e-02*a1_4+7.9365079365079361e-02*a1_5+6.7460317460317457e-02*a1_6+5.8641975308641979e-02*a1_7+3.3333333333333331e-01*a2_0+1.4814814814814811e-01*a2_1+9.2592592592592587e-02*a2_2+6.6666666666666666e-02*a2_3+5.1851851851851850e-02*a2_4+4.2328042328042333e-02*a2_5+3.5714285714285712e-02*a2_6+3.0864197530864199e-02*a2_7+1.1111111111111110e-01*a3_0+3.7037037037037042e-02*a3_1+1.8518518518518521e-02*a3_2+1.1111111111111110e-02*a3_3+7.4074074074074068e-03*a3_4+5.2910052910052907e-03*a3_5+3.9682539682539680e-03*a3_6+3.0864197530864200e-03*a3_7;
  *p1 = -4.4444444444444442e-01*a1_0+-1.8518518518518520e-01*a1_1+-1.1111111111111110e-01*a1_2+-7.7777777777777779e-02*a1_3+-5.9259259259259262e-02*a1_4+-4.7619047619047623e-02*a1_5+-3.9682539682539680e-02*a1_6+-3.3950617283950622e-02*a1_7+0.0000000000000000e+00*a2_0+3.7037037037037042e-02*a2_1+3.7037037037037042e-02*a2_2+3.3333333333333333e-02*a2_3+2.9629629629629631e-02*a2_4+2.6455026455026460e-02*a2_5+2.3809523809523812e-02*a2_6+2.1604938271604941e-02*a2_7+4.4444444444444442e-01*a3_0+2.5925925925925930e-01*a3_1+1.8518518518518520e-01*a3_2+1.4444444444444440e-01*a3_3+1.1851851851851850e-01*a3_4+1.0052910052910050e-01*a3_5+8.7301587301587297e-02*a3_6+7.7160493827160490e-02*a3_7;
}
            
void cf_linear_approximation_1_7(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double a1_7, double* p0, double* p1)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+-1.6666666666666671e-01*a1_2+-2.0000000000000001e-01*a1_3+-2.0000000000000001e-01*a1_4+-1.9047619047619049e-01*a1_5+-1.7857142857142860e-01*a1_6+-1.6666666666666671e-01*a1_7;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+9.0000000000000002e-01*a1_3+8.0000000000000004e-01*a1_4+7.1428571428571430e-01*a1_5+6.4285714285714290e-01*a1_6+5.8333333333333326e-01*a1_7;
}
            
double cf_find_real_zero_in_01_2(double a_0, double a_1, double a_2)
{
  
/* Code translated from http://www.netlib.org/toms/493, subroutine QUAD, with modifications. */
#define ABS(X) ((X)<0.0?-(X):(X))
double b, e, d, lr, sr;
//printf("a_0,a_1,a_2, e=%f, %f, %f\n", a_0, a_1, a_2);
if (a_2==0.0)
  {
    if (a_1!=0.0) return -a_0/a_1;
    return -1.0;
  }
else
  {
    if (a_0==0.0)
      return 0.0;
    b = a_1*0.5;
    if (ABS(b) < ABS(a_0))
    {
      e = a_2;
      if (a_0<0.0)
        e = -a_2;
      e = b*(b/ABS(a_0)) - e;
      d = sqrt(ABS(e))*sqrt(ABS(a_0));
    }
    else
    {
      e = 1.0 - (a_2/b)*(a_0/b);
      d = sqrt(ABS(e))*ABS(b);
    }
    if (e>=0)
    {
      if (b>=0.0) d=-d;
      lr = (-b+d)/a_2;
      if (lr==0.0)
        return 0.0;
      sr = (a_0/lr)/a_2;
      //printf("p(lr=%f)=%f\n", lr,a_0+lr*(a_1+lr*a_2));
      //printf("p(sr=%f)=%f\n", sr,a_0+sr*(a_1+sr*a_2));
      if (lr>=0 && lr<=1.0)
        return lr;
      return sr;
    }
  }

  return -1.0;
}
            

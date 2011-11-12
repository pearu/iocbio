
/* ipwf - Integrate PieceWise polynomial Functions.

  This file is generated using iocbio/analysis/src/generate_iocbio_ipwf_source.py.

  Author: Pearu Peterson
  Created: Oct 2011
*/
#include <math.h>
#include <stdio.h>
#include "iocbio_ipwf.h"

#define EPSPOS 2.2204460492503131e-16
#define EPSNEG 1.1102230246251565e-16
#define FLOATMIN -1.7976931348623157e+308
#define FLOATMAX 1.7976931348623157e+308
#define FIXZERO(X) ((-EPSNEG<(X)) && ((X)<EPSPOS)?0.0:(X))
    
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

void iocbio_ipwf_e11_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  //printf("void iocbio_ipwf_e11_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)\n");
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
        b0 += (-0.6666666666666667*f_ip1pj - 0.3333333333333333*f_ipj + 0.3333333333333333*f_i + 0.3333333333333333*f_ip1)*f_ip1 + 0.3333333333333333*(f_ip1pj*f_ip1pj) + (0.3333333333333333*f_i - 0.6666666666666667*f_ipj - 0.3333333333333333*f_ip1pj)*f_i + f_ipj*(0.3333333333333333*f_ipj + 0.3333333333333333*f_ip1pj);
        b1 += (-f_ipj + f_i + f_ip1)*f_ipj + f_ip1pj*(f_ip1pj - f_i - f_ip1);
        b2 += f_ipj*f_ipj + f_ip1*(-f_ip2pj - f_ipj) + f_ip1pj*(2.0*f_ip1 + f_ip2pj - f_ipj - f_ip1pj);
        b3 += (-0.3333333333333333*f_ipj - 0.3333333333333333*f_i + 0.3333333333333333*f_ip1)*f_ipj + (-0.6666666666666667*f_ip2pj + 0.6666666666666667*f_ipj + 0.6666666666666667*f_i - 0.6666666666666667*f_ip1)*f_ip1pj + f_ip2pj*(0.3333333333333333*f_ip2pj - 0.3333333333333333*f_i + 0.3333333333333333*f_ip1);
      }
      b0 += f_m2mjpn*(0.3333333333333333*f_m1mjpn - 0.6666666666666667*f_m2pn + 0.3333333333333333*f_m2mjpn) + (-0.6666666666666667*f_m1mjpn + 0.3333333333333333*f_m2pn - 0.3333333333333333*f_m2mjpn + 0.3333333333333333*f_m1pn)*f_m1pn + f_m2pn*(-0.3333333333333333*f_m1mjpn + 0.3333333333333333*f_m2pn) + 0.3333333333333333*(f_m1mjpn*f_m1mjpn);
      b1 += -f_m1pn*f_m2mjpn + f_m2pn*(-f_m2pn + f_m2mjpn) + (f_m2pn + f_m1pn - f_m1mjpn)*f_m1mjpn;
      b2 += f_m2mjpn*f_m1pn + f_m1mjpn*(-f_m2mjpn + f_m1mjpn) + (-f_m1pn + f_m2pn - f_m1mjpn)*f_m2pn;
      b3 += f_m2mjpn*(-0.3333333333333333*f_m2mjpn + 0.3333333333333333*f_m1pn + 0.6666666666666667*f_m1mjpn) + (-0.3333333333333333*f_m1pn - 0.3333333333333333*f_m1mjpn)*f_m1pn - 0.3333333333333333*(f_m1mjpn*f_m1mjpn) + (0.3333333333333333*f_m1mjpn + 0.6666666666666667*f_m1pn - 0.3333333333333333*f_m2mjpn - 0.3333333333333333*f_m2pn)*f_m2pn;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int iocbio_ipwf_e11_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result)
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
  //printf("int iocbio_ipwf_e11_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result)\n");
  for (j=start_j; j<end_j; ++j)
  {
    iocbio_ipwf_e11_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    iocbio_ipwf_quadratic_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1, &p2);
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
            
int iocbio_ipwf_e11_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result, double* slope)
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
  //printf("int iocbio_ipwf_e11_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result, double* slope)\n");
  for (j=start_j; j<end_j; ++j)
  {
    iocbio_ipwf_e11_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    iocbio_ipwf_linear_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("iocbio_ipwf_e11_find_zero_diff0: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            *slope = p1;
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

void iocbio_ipwf_e11_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=1) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  //printf("void iocbio_ipwf_e11_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)\n");
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
        b0 += (-f_ipj + f_i + f_ip1)*f_ipj + f_ip1pj*(f_ip1pj - f_i - f_ip1);
        b1 += (-2.0*f_ip2pj - 2.0*f_ipj)*f_ip1 + 2.0*(f_ipj*f_ipj) + f_ip1pj*(4.0*f_ip1 + 2.0*f_ip2pj - 2.0*f_ipj - 2.0*f_ip1pj);
        b2 += f_ip2pj*(f_ip2pj - f_i + f_ip1) + (-2.0*f_ip2pj + 2.0*f_ipj + 2.0*f_i - 2.0*f_ip1)*f_ip1pj + f_ipj*(-f_ipj - f_i + f_ip1);
      }
      b0 += -f_m1pn*f_m2mjpn + f_m2pn*(-f_m2pn + f_m2mjpn) + (f_m1pn + f_m2pn - f_m1mjpn)*f_m1mjpn;
      b1 += f_m2pn*(-2.0*f_m1pn + 2.0*f_m2pn - 2.0*f_m1mjpn) + 2.0*f_m1pn*f_m2mjpn + f_m1mjpn*(-2.0*f_m2mjpn + 2.0*f_m1mjpn);
      b2 += f_m2pn*(2.0*f_m1pn - f_m2pn - f_m2mjpn) + f_m1pn*(-f_m1pn + f_m2mjpn) - (f_m2mjpn*f_m2mjpn) + f_m1mjpn*(2.0*f_m2mjpn - f_m1pn + f_m2pn - f_m1mjpn);
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int iocbio_ipwf_e11_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result)
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
  //printf("int iocbio_ipwf_e11_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result)\n");
  for (j=start_j; j<end_j; ++j)
  {
    iocbio_ipwf_e11_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    iocbio_ipwf_quadratic_approximation_1_2(a1_0, a1_1, a1_2, &p0, &p1, &p2);
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
            
int iocbio_ipwf_e11_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result, double* slope)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  //printf("int iocbio_ipwf_e11_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result, double* slope)\n");
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  int dj = (start_j>end_j?-1:1);
  int count = (start_j<end_j?end_j-start_j:start_j-end_j);
  for (j=(dj==-1?start_j-1:start_j); count>0; j += dj, --count)
  {
    iocbio_ipwf_e11_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    s = iocbio_ipwf_find_real_zero_in_01_2(a1_0, a1_1, a1_2, dj, slope);
    //printf("j,s,dj,zero=%d, %e, %d, %e\n",j,s, dj, a1_0+s*(a1_1+s*(a1_2+s*a1_3)));
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

void iocbio_ipwf_e11_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=2) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  //printf("void iocbio_ipwf_e11_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)\n");
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
        b0 += (-2.0*f_ip2pj - 2.0*f_ipj)*f_ip1 + 2.0*(f_ipj*f_ipj) + f_ip1pj*(4.0*f_ip1 + 2.0*f_ip2pj - 2.0*f_ipj - 2.0*f_ip1pj);
        b1 += f_ip1pj*(-4.0*f_ip2pj + 4.0*f_ipj + 4.0*f_i - 4.0*f_ip1) + (-2.0*f_ipj - 2.0*f_i + 2.0*f_ip1)*f_ipj + (2.0*f_ip2pj - 2.0*f_i + 2.0*f_ip1)*f_ip2pj;
      }
      b0 += f_m1mjpn*(-2.0*f_m2mjpn - 2.0*f_m2pn + 2.0*f_m1mjpn) + 2.0*f_m2mjpn*f_m1pn + f_m2pn*(-2.0*f_m1pn + 2.0*f_m2pn);
      b1 += -2.0*(f_m2mjpn*f_m2mjpn) + f_m1pn*(-2.0*f_m1pn + 2.0*f_m2mjpn) + f_m1mjpn*(4.0*f_m2mjpn - 2.0*f_m1mjpn - 2.0*f_m1pn) + f_m2pn*(4.0*f_m1pn - 2.0*f_m2mjpn - 2.0*f_m2pn + 2.0*f_m1mjpn);
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int iocbio_ipwf_e11_find_extreme_diff2(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  //printf("int iocbio_ipwf_e11_find_extreme_diff2(int j0, int j1, double *fm, int n, int m, double* result)\n");
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
    iocbio_ipwf_e11_compute_coeffs_diff2(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3);
    count ++;
    if (count<3)
      continue;
    iocbio_ipwf_quadratic_approximation_3_1(a1_0, a1_1, a2_0, a2_1, a3_0, a3_1, &p0, &p1, &p2);
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
            
int iocbio_ipwf_e11_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result, double* slope)
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
  //printf("int iocbio_ipwf_e11_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result, double* slope)\n");
  for (j=start_j; j<end_j; ++j)
  {
    iocbio_ipwf_e11_compute_coeffs_diff2(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    iocbio_ipwf_linear_approximation_1_1(a1_0, a1_1, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("iocbio_ipwf_e11_find_zero_diff2: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            *slope = p1;
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

void iocbio_ipwf_e11_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order=3) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  //printf("void iocbio_ipwf_e11_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)\n");
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
        b0 += f_ip1pj*(-4.0*f_ip2pj + 4.0*f_ipj + 4.0*f_i - 4.0*f_ip1) + (-2.0*f_ipj - 2.0*f_i + 2.0*f_ip1)*f_ipj + (2.0*f_ip2pj - 2.0*f_i + 2.0*f_ip1)*f_ip2pj;
      }
      b0 += f_m1pn*(-2.0*f_m1pn - 2.0*f_m1mjpn) + (4.0*f_m1pn - 2.0*f_m2pn + 2.0*f_m1mjpn)*f_m2pn - 2.0*(f_m1mjpn*f_m1mjpn) + f_m2mjpn*(4.0*f_m1mjpn - 2.0*f_m2pn + 2.0*f_m1pn - 2.0*f_m2mjpn);
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int iocbio_ipwf_e11_find_extreme_diff3(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  //printf("int iocbio_ipwf_e11_find_extreme_diff3(int j0, int j1, double *fm, int n, int m, double* result)\n");
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
    iocbio_ipwf_e11_compute_coeffs_diff3(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3);
    count ++;
    if (count<3)
      continue;
    iocbio_ipwf_quadratic_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1, &p2);
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
            
int iocbio_ipwf_e11_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result, double* slope)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  //printf("int iocbio_ipwf_e11_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result, double* slope)\n");
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
    iocbio_ipwf_e11_compute_coeffs_diff3(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3);
    count ++;
    if (count<3)
      continue;
    iocbio_ipwf_linear_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1);
    if (p1!=0.0)
    {
       s = -p0/p1;
       ////printf("iocbio_ipwf_e11_find_zero_diff3: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            *slope = p1;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
void iocbio_ipwf_e11_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */
  //printf("void iocbio_ipwf_e11_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3)\n");
  switch (order)
  {
    case 0:  iocbio_ipwf_e11_compute_coeffs_diff0(j, fm, n, m, a0, a1, a2, a3); break;
    case 1:  iocbio_ipwf_e11_compute_coeffs_diff1(j, fm, n, m, a0, a1, a2, a3); break;
    case 2:  iocbio_ipwf_e11_compute_coeffs_diff2(j, fm, n, m, a0, a1, a2, a3); break;
    case 3:  iocbio_ipwf_e11_compute_coeffs_diff3(j, fm, n, m, a0, a1, a2, a3); break;
    default:
      *a0 = 0.0;
      *a1 = 0.0;
      *a2 = 0.0;
      *a3 = 0.0;
  }
}
        
int iocbio_ipwf_e11_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  //printf("int iocbio_ipwf_e11_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)\n");
  switch (order)
  {
    case 0: return iocbio_ipwf_e11_find_extreme_diff0(j0, j1, fm, n, m, result);
    case 1: return iocbio_ipwf_e11_find_extreme_diff1(j0, j1, fm, n, m, result);
    case 2: return iocbio_ipwf_e11_find_extreme_diff2(j0, j1, fm, n, m, result);
    case 3: return iocbio_ipwf_e11_find_extreme_diff3(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
int iocbio_ipwf_e11_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result, double* slope)
{
  switch (order)
  {
    case 0: return iocbio_ipwf_e11_find_zero_diff0(j0, j1, fm, n, m, result, slope);
    case 1: return iocbio_ipwf_e11_find_zero_diff1(j0, j1, fm, n, m, result, slope);
    case 2: return iocbio_ipwf_e11_find_zero_diff2(j0, j1, fm, n, m, result, slope);
    case 3: return iocbio_ipwf_e11_find_zero_diff3(j0, j1, fm, n, m, result, slope);
    default:
      *result = 0.0;
      *slope = 0.0;
  }
  return -2;
}
        
double iocbio_ipwf_e11_evaluate(double y, double *fm, int n, int m, int order)
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
  //printf("double iocbio_ipwf_e11_evaluate(double y, double *fm, int n, int m, int order)\n");
  iocbio_ipwf_e11_compute_coeffs(j, fm, n, m, order, &a0, &a1, &a2, &a3);
  return a0+(a1+(a2+(a3)*r)*r)*r;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double iocbio_ipwf_e11_f1_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  //printf("double iocbio_ipwf_e11_f1_evaluate(double x, double *f, int n, int order)\n");
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

double iocbio_ipwf_e11_f2_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  //printf("double iocbio_ipwf_e11_f2_evaluate(double x, double *f, int n, int order)\n");
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
#define F(I) ((I)<0?(f[(I)+n]):((I)>=n?f[(I)-n]:f[(I)]))

void iocbio_ipwf_ep11_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 1;
  double *f = fm;
  //printf("void iocbio_ipwf_ep11_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)\n");
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double f_ipj, f_ip1pj, f_i, f_ip1, f_ip2pj;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip1 = F(i+1);
        f_ip2pj = F(i+2+j);
        b0 += (-0.6666666666666667*f_ip1pj - 0.3333333333333333*f_ipj + 0.3333333333333333*f_i + 0.3333333333333333*f_ip1)*f_ip1 + 0.3333333333333333*(f_ip1pj*f_ip1pj) + (0.3333333333333333*f_i - 0.6666666666666667*f_ipj - 0.3333333333333333*f_ip1pj)*f_i + f_ipj*(0.3333333333333333*f_ipj + 0.3333333333333333*f_ip1pj);
        b1 += (-f_ipj + f_i + f_ip1)*f_ipj + f_ip1pj*(f_ip1pj - f_i - f_ip1);
        b2 += f_ipj*f_ipj + f_ip1*(-f_ip2pj - f_ipj) + f_ip1pj*(2.0*f_ip1 + f_ip2pj - f_ipj - f_ip1pj);
        b3 += (-0.3333333333333333*f_ipj - 0.3333333333333333*f_i + 0.3333333333333333*f_ip1)*f_ipj + (-0.6666666666666667*f_ip2pj + 0.6666666666666667*f_ipj + 0.6666666666666667*f_i - 0.6666666666666667*f_ip1)*f_ip1pj + f_ip2pj*(0.3333333333333333*f_ip2pj - 0.3333333333333333*f_i + 0.3333333333333333*f_ip1);
      }
      b0 += 0;
      b1 += 0;
      b2 += 0;
      b3 += 0;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int iocbio_ipwf_ep11_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result)
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
  //printf("int iocbio_ipwf_ep11_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result)\n");
  for (j=start_j; j<end_j; ++j)
  {
    iocbio_ipwf_ep11_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    iocbio_ipwf_quadratic_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1, &p2);
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
            
int iocbio_ipwf_ep11_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result, double* slope)
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
  //printf("int iocbio_ipwf_ep11_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result, double* slope)\n");
  for (j=start_j; j<end_j; ++j)
  {
    iocbio_ipwf_ep11_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    iocbio_ipwf_linear_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("iocbio_ipwf_ep11_find_zero_diff0: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            *slope = p1;
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
#define F(I) ((I)<0?(f[(I)+n]):((I)>=n?f[(I)-n]:f[(I)]))

void iocbio_ipwf_ep11_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L), y, order=1) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 1;
  double *f = fm;
  //printf("void iocbio_ipwf_ep11_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)\n");
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double f_ip2pj, f_ipj, f_ip1, f_i, f_ip1pj;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      
      for(i=0;i<=k;++i)
      {
        f_ip2pj = F(i+2+j);
        f_ipj = F(i+j);
        f_ip1 = F(i+1);
        f_i = F(i);
        f_ip1pj = F(i+1+j);
        b0 += (-f_ipj + f_i + f_ip1)*f_ipj + f_ip1pj*(f_ip1pj - f_i - f_ip1);
        b1 += (-2.0*f_ip2pj - 2.0*f_ipj)*f_ip1 + 2.0*(f_ipj*f_ipj) + f_ip1pj*(4.0*f_ip1 + 2.0*f_ip2pj - 2.0*f_ipj - 2.0*f_ip1pj);
        b2 += f_ip2pj*(f_ip2pj - f_i + f_ip1) + (-2.0*f_ip2pj + 2.0*f_ipj + 2.0*f_i - 2.0*f_ip1)*f_ip1pj + f_ipj*(-f_ipj - f_i + f_ip1);
      }
      b0 += 0;
      b1 += 0;
      b2 += 0;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int iocbio_ipwf_ep11_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result)
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
  //printf("int iocbio_ipwf_ep11_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result)\n");
  for (j=start_j; j<end_j; ++j)
  {
    iocbio_ipwf_ep11_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    iocbio_ipwf_quadratic_approximation_1_2(a1_0, a1_1, a1_2, &p0, &p1, &p2);
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
            
int iocbio_ipwf_ep11_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result, double* slope)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  //printf("int iocbio_ipwf_ep11_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result, double* slope)\n");
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  int dj = (start_j>end_j?-1:1);
  int count = (start_j<end_j?end_j-start_j:start_j-end_j);
  for (j=(dj==-1?start_j-1:start_j); count>0; j += dj, --count)
  {
    iocbio_ipwf_ep11_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    s = iocbio_ipwf_find_real_zero_in_01_2(a1_0, a1_1, a1_2, dj, slope);
    //printf("j,s,dj,zero=%d, %e, %d, %e\n",j,s, dj, a1_0+s*(a1_1+s*(a1_2+s*a1_3)));
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
#define F(I) ((I)<0?(f[(I)+n]):((I)>=n?f[(I)-n]:f[(I)]))

void iocbio_ipwf_ep11_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L), y, order=2) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 1;
  double *f = fm;
  //printf("void iocbio_ipwf_ep11_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)\n");
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double f_ip2pj, f_ip1, f_ip1pj, f_i, f_ipj;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      
      for(i=0;i<=k;++i)
      {
        f_ip2pj = F(i+2+j);
        f_ip1 = F(i+1);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ipj = F(i+j);
        b0 += (-2.0*f_ip2pj - 2.0*f_ipj)*f_ip1 + 2.0*(f_ipj*f_ipj) + f_ip1pj*(4.0*f_ip1 + 2.0*f_ip2pj - 2.0*f_ipj - 2.0*f_ip1pj);
        b1 += f_ip1pj*(-4.0*f_ip2pj + 4.0*f_ipj + 4.0*f_i - 4.0*f_ip1) + (-2.0*f_ipj - 2.0*f_i + 2.0*f_ip1)*f_ipj + (2.0*f_ip2pj - 2.0*f_i + 2.0*f_ip1)*f_ip2pj;
      }
      b0 += 0;
      b1 += 0;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int iocbio_ipwf_ep11_find_extreme_diff2(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  //printf("int iocbio_ipwf_ep11_find_extreme_diff2(int j0, int j1, double *fm, int n, int m, double* result)\n");
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
    iocbio_ipwf_ep11_compute_coeffs_diff2(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3);
    count ++;
    if (count<3)
      continue;
    iocbio_ipwf_quadratic_approximation_3_1(a1_0, a1_1, a2_0, a2_1, a3_0, a3_1, &p0, &p1, &p2);
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
            
int iocbio_ipwf_ep11_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result, double* slope)
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
  //printf("int iocbio_ipwf_ep11_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result, double* slope)\n");
  for (j=start_j; j<end_j; ++j)
  {
    iocbio_ipwf_ep11_compute_coeffs_diff2(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    iocbio_ipwf_linear_approximation_1_1(a1_0, a1_1, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("iocbio_ipwf_ep11_find_zero_diff2: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            *slope = p1;
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
#define F(I) ((I)<0?(f[(I)+n]):((I)>=n?f[(I)-n]:f[(I)]))

void iocbio_ipwf_ep11_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L), y, order=3) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 1;
  double *f = fm;
  //printf("void iocbio_ipwf_ep11_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)\n");
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double f_ip2pj, f_ip1pj, f_i, f_ip1, f_ipj;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      
      for(i=0;i<=k;++i)
      {
        f_ip2pj = F(i+2+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip1 = F(i+1);
        f_ipj = F(i+j);
        b0 += f_ip1pj*(-4.0*f_ip2pj + 4.0*f_ipj + 4.0*f_i - 4.0*f_ip1) + (-2.0*f_ipj - 2.0*f_i + 2.0*f_ip1)*f_ipj + (2.0*f_ip2pj - 2.0*f_i + 2.0*f_ip1)*f_ip2pj;
      }
      b0 += 0;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int iocbio_ipwf_ep11_find_extreme_diff3(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  //printf("int iocbio_ipwf_ep11_find_extreme_diff3(int j0, int j1, double *fm, int n, int m, double* result)\n");
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
    iocbio_ipwf_ep11_compute_coeffs_diff3(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3);
    count ++;
    if (count<3)
      continue;
    iocbio_ipwf_quadratic_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1, &p2);
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
            
int iocbio_ipwf_ep11_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result, double* slope)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  //printf("int iocbio_ipwf_ep11_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result, double* slope)\n");
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
    iocbio_ipwf_ep11_compute_coeffs_diff3(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3);
    count ++;
    if (count<3)
      continue;
    iocbio_ipwf_linear_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1);
    if (p1!=0.0)
    {
       s = -p0/p1;
       ////printf("iocbio_ipwf_ep11_find_zero_diff3: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            *slope = p1;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
void iocbio_ipwf_ep11_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int((f1(x)-f1(x+y))*(f2(x)-f2(x+y)), x=0..L-y), y, order) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */
  //printf("void iocbio_ipwf_ep11_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3)\n");
  switch (order)
  {
    case 0:  iocbio_ipwf_ep11_compute_coeffs_diff0(j, fm, n, m, a0, a1, a2, a3); break;
    case 1:  iocbio_ipwf_ep11_compute_coeffs_diff1(j, fm, n, m, a0, a1, a2, a3); break;
    case 2:  iocbio_ipwf_ep11_compute_coeffs_diff2(j, fm, n, m, a0, a1, a2, a3); break;
    case 3:  iocbio_ipwf_ep11_compute_coeffs_diff3(j, fm, n, m, a0, a1, a2, a3); break;
    default:
      *a0 = 0.0;
      *a1 = 0.0;
      *a2 = 0.0;
      *a3 = 0.0;
  }
}
        
int iocbio_ipwf_ep11_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  //printf("int iocbio_ipwf_ep11_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)\n");
  switch (order)
  {
    case 0: return iocbio_ipwf_ep11_find_extreme_diff0(j0, j1, fm, n, m, result);
    case 1: return iocbio_ipwf_ep11_find_extreme_diff1(j0, j1, fm, n, m, result);
    case 2: return iocbio_ipwf_ep11_find_extreme_diff2(j0, j1, fm, n, m, result);
    case 3: return iocbio_ipwf_ep11_find_extreme_diff3(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
int iocbio_ipwf_ep11_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result, double* slope)
{
  switch (order)
  {
    case 0: return iocbio_ipwf_ep11_find_zero_diff0(j0, j1, fm, n, m, result, slope);
    case 1: return iocbio_ipwf_ep11_find_zero_diff1(j0, j1, fm, n, m, result, slope);
    case 2: return iocbio_ipwf_ep11_find_zero_diff2(j0, j1, fm, n, m, result, slope);
    case 3: return iocbio_ipwf_ep11_find_zero_diff3(j0, j1, fm, n, m, result, slope);
    default:
      *result = 0.0;
      *slope = 0.0;
  }
  return -2;
}
        
double iocbio_ipwf_ep11_evaluate(double y, double *fm, int n, int m, int order)
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
  //printf("double iocbio_ipwf_ep11_evaluate(double y, double *fm, int n, int m, int order)\n");
  iocbio_ipwf_ep11_compute_coeffs(j, fm, n, m, order, &a0, &a1, &a2, &a3);
  return a0+(a1+(a2+(a3)*r)*r)*r;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double iocbio_ipwf_ep11_f1_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  //printf("double iocbio_ipwf_ep11_f1_evaluate(double x, double *f, int n, int order)\n");
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

double iocbio_ipwf_ep11_f2_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  //printf("double iocbio_ipwf_ep11_f2_evaluate(double x, double *f, int n, int order)\n");
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

void iocbio_ipwf_a00_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1)
{
  /* int(-f1(x)*f2(x+y), x=0..L-y) = sum(a_k*r^k, k=0..1) where y=j+r
     f1(x)=sum([0<=s<1]*(F(i)), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*(F(i)), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  //printf("void iocbio_ipwf_a00_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1)\n");
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
        b0 += -f_ipj*f_i;
        b1 += (f_ipj - f_ip1pj)*f_i;
      }
      b0 += -f_m2mjpn*f_m2pn;
      b1 += f_m2mjpn*f_m2pn;
    }
  }
  *a0 = b0;
  *a1 = b1;
}
        
int iocbio_ipwf_a00_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  //printf("int iocbio_ipwf_a00_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result)\n");
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
    iocbio_ipwf_a00_compute_coeffs_diff0(j, fm, n, m, &a3_0, &a3_1);
    count ++;
    if (count<3)
      continue;
    iocbio_ipwf_quadratic_approximation_3_1(a1_0, a1_1, a2_0, a2_1, a3_0, a3_1, &p0, &p1, &p2);
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
            
int iocbio_ipwf_a00_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result, double* slope)
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
  //printf("int iocbio_ipwf_a00_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result, double* slope)\n");
  for (j=start_j; j<end_j; ++j)
  {
    iocbio_ipwf_a00_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1);
    iocbio_ipwf_linear_approximation_1_1(a1_0, a1_1, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("iocbio_ipwf_a00_find_zero_diff0: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            *slope = p1;
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

void iocbio_ipwf_a00_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1)
{
  /* diff(int(-f1(x)*f2(x+y), x=0..L-y), y, order=1) = sum(a_k*r^k, k=0..1) where y=j+r
     f1(x)=sum([0<=s<1]*(F(i)), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*(F(i)), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  //printf("void iocbio_ipwf_a00_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1)\n");
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
        b0 += (f_ipj - f_ip1pj)*f_i;
      }
      b0 += f_m2mjpn*f_m2pn;
    }
  }
  *a0 = b0;
  *a1 = b1;
}
        
int iocbio_ipwf_a00_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  //printf("int iocbio_ipwf_a00_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result)\n");
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
    iocbio_ipwf_a00_compute_coeffs_diff1(j, fm, n, m, &a3_0, &a3_1);
    count ++;
    if (count<3)
      continue;
    iocbio_ipwf_quadratic_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1, &p2);
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
            
int iocbio_ipwf_a00_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result, double* slope)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  //printf("int iocbio_ipwf_a00_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result, double* slope)\n");
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
    iocbio_ipwf_a00_compute_coeffs_diff1(j, fm, n, m, &a3_0, &a3_1);
    count ++;
    if (count<3)
      continue;
    iocbio_ipwf_linear_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1);
    if (p1!=0.0)
    {
       s = -p0/p1;
       ////printf("iocbio_ipwf_a00_find_zero_diff1: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            *slope = p1;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
void iocbio_ipwf_a00_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1)
{
  /* diff(int(-f1(x)*f2(x+y), x=0..L-y), y, order) = sum(a_k*r^k, k=0..1) where y=j+r
     f1(x)=sum([0<=s<1]*(F(i)), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*(F(i)), i=0..N-1) where s=x-i */
  //printf("void iocbio_ipwf_a00_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1)\n");
  switch (order)
  {
    case 0:  iocbio_ipwf_a00_compute_coeffs_diff0(j, fm, n, m, a0, a1); break;
    case 1:  iocbio_ipwf_a00_compute_coeffs_diff1(j, fm, n, m, a0, a1); break;
    default:
      *a0 = 0.0;
      *a1 = 0.0;
  }
}
        
int iocbio_ipwf_a00_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  //printf("int iocbio_ipwf_a00_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)\n");
  switch (order)
  {
    case 0: return iocbio_ipwf_a00_find_extreme_diff0(j0, j1, fm, n, m, result);
    case 1: return iocbio_ipwf_a00_find_extreme_diff1(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
int iocbio_ipwf_a00_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result, double* slope)
{
  switch (order)
  {
    case 0: return iocbio_ipwf_a00_find_zero_diff0(j0, j1, fm, n, m, result, slope);
    case 1: return iocbio_ipwf_a00_find_zero_diff1(j0, j1, fm, n, m, result, slope);
    default:
      *result = 0.0;
      *slope = 0.0;
  }
  return -2;
}
        
double iocbio_ipwf_a00_evaluate(double y, double *fm, int n, int m, int order)
{
  double a0 = 0.0;
  double a1 = 0.0;
  /*
  int j = floor((y<0?-y:y));
  double r = (y<0?-y:y) - j;
  */
  int j = floor (y);
  double r = y - j;
  //printf("double iocbio_ipwf_a00_evaluate(double y, double *fm, int n, int m, int order)\n");
  iocbio_ipwf_a00_compute_coeffs(j, fm, n, m, order, &a0, &a1);
  return a0+(a1)*r;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double iocbio_ipwf_a00_f1_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  //printf("double iocbio_ipwf_a00_f1_evaluate(double x, double *f, int n, int order)\n");
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

double iocbio_ipwf_a00_f2_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  //printf("double iocbio_ipwf_a00_f2_evaluate(double x, double *f, int n, int order)\n");
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

void iocbio_ipwf_a11_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* int(-f1(x)*f2(x+y), x=0..L-y) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  //printf("void iocbio_ipwf_a11_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)\n");
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
        b0 += -0.1666666666666667*f_ip1pj*f_i - 0.3333333333333333*f_i*f_ipj + f_ip1*(-0.3333333333333333*f_ip1pj - 0.1666666666666667*f_ipj);
        b1 += (0.5*f_ipj - 0.5*f_ip1pj)*f_i + (0.5*f_ipj - 0.5*f_ip1pj)*f_ip1;
        b2 += f_ip1*(f_ip1pj - 0.5*f_ip2pj - 0.5*f_ipj);
        b3 += (0.3333333333333333*f_ip1pj - 0.1666666666666667*f_ip2pj - 0.1666666666666667*f_ipj)*f_i + (-0.3333333333333333*f_ip1pj + 0.1666666666666667*f_ip2pj + 0.1666666666666667*f_ipj)*f_ip1;
      }
      b0 += -0.1666666666666667*f_m1mjpn*f_m2pn - 0.3333333333333333*f_m2pn*f_m2mjpn + (-0.3333333333333333*f_m1mjpn - 0.1666666666666667*f_m2mjpn)*f_m1pn;
      b1 += 0.5*f_m2pn*f_m1mjpn + 0.5*f_m2mjpn*f_m2pn + f_m1pn*(-0.5*f_m2mjpn + 0.5*f_m1mjpn);
      b2 += 0.5*f_m2mjpn*f_m1pn - 0.5*f_m2pn*f_m1mjpn;
      b3 += -0.1666666666666667*f_m2pn*f_m2mjpn + 0.1666666666666667*f_m1mjpn*f_m2pn + f_m1pn*(-0.1666666666666667*f_m1mjpn + 0.1666666666666667*f_m2mjpn);
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int iocbio_ipwf_a11_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result)
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
  //printf("int iocbio_ipwf_a11_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result)\n");
  for (j=start_j; j<end_j; ++j)
  {
    iocbio_ipwf_a11_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    iocbio_ipwf_quadratic_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1, &p2);
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
            
int iocbio_ipwf_a11_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result, double* slope)
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
  //printf("int iocbio_ipwf_a11_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result, double* slope)\n");
  for (j=start_j; j<end_j; ++j)
  {
    iocbio_ipwf_a11_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    iocbio_ipwf_linear_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("iocbio_ipwf_a11_find_zero_diff0: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            *slope = p1;
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

void iocbio_ipwf_a11_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int(-f1(x)*f2(x+y), x=0..L-y), y, order=1) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  //printf("void iocbio_ipwf_a11_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)\n");
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
        b0 += (0.5*f_ipj - 0.5*f_ip1pj)*f_i + (0.5*f_ipj - 0.5*f_ip1pj)*f_ip1;
        b1 += (2.0*f_ip1pj - f_ip2pj - f_ipj)*f_ip1;
        b2 += (-f_ip1pj + 0.5*f_ip2pj + 0.5*f_ipj)*f_ip1 + f_i*(f_ip1pj - 0.5*f_ip2pj - 0.5*f_ipj);
      }
      b0 += 0.5*f_m2pn*f_m1mjpn + 0.5*f_m2mjpn*f_m2pn + f_m1pn*(-0.5*f_m2mjpn + 0.5*f_m1mjpn);
      b1 += f_m1pn*f_m2mjpn - f_m2pn*f_m1mjpn;
      b2 += 0.5*f_m2pn*f_m1mjpn - 0.5*f_m2pn*f_m2mjpn + f_m1pn*(-0.5*f_m1mjpn + 0.5*f_m2mjpn);
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int iocbio_ipwf_a11_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result)
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
  //printf("int iocbio_ipwf_a11_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result)\n");
  for (j=start_j; j<end_j; ++j)
  {
    iocbio_ipwf_a11_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    iocbio_ipwf_quadratic_approximation_1_2(a1_0, a1_1, a1_2, &p0, &p1, &p2);
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
            
int iocbio_ipwf_a11_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result, double* slope)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  //printf("int iocbio_ipwf_a11_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result, double* slope)\n");
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  int dj = (start_j>end_j?-1:1);
  int count = (start_j<end_j?end_j-start_j:start_j-end_j);
  for (j=(dj==-1?start_j-1:start_j); count>0; j += dj, --count)
  {
    iocbio_ipwf_a11_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    s = iocbio_ipwf_find_real_zero_in_01_2(a1_0, a1_1, a1_2, dj, slope);
    //printf("j,s,dj,zero=%d, %e, %d, %e\n",j,s, dj, a1_0+s*(a1_1+s*(a1_2+s*a1_3)));
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

void iocbio_ipwf_a11_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int(-f1(x)*f2(x+y), x=0..L-y), y, order=2) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  //printf("void iocbio_ipwf_a11_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)\n");
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
        b0 += (2.0*f_ip1pj - f_ip2pj - f_ipj)*f_ip1;
        b1 += f_i*(2.0*f_ip1pj - f_ip2pj - f_ipj) + (-2.0*f_ip1pj + f_ip2pj + f_ipj)*f_ip1;
      }
      b0 += f_m1pn*f_m2mjpn - f_m2pn*f_m1mjpn;
      b1 += f_m1mjpn*f_m2pn - f_m2pn*f_m2mjpn + f_m1pn*(-f_m1mjpn + f_m2mjpn);
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int iocbio_ipwf_a11_find_extreme_diff2(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  //printf("int iocbio_ipwf_a11_find_extreme_diff2(int j0, int j1, double *fm, int n, int m, double* result)\n");
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
    iocbio_ipwf_a11_compute_coeffs_diff2(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3);
    count ++;
    if (count<3)
      continue;
    iocbio_ipwf_quadratic_approximation_3_1(a1_0, a1_1, a2_0, a2_1, a3_0, a3_1, &p0, &p1, &p2);
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
            
int iocbio_ipwf_a11_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result, double* slope)
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
  //printf("int iocbio_ipwf_a11_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result, double* slope)\n");
  for (j=start_j; j<end_j; ++j)
  {
    iocbio_ipwf_a11_compute_coeffs_diff2(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    iocbio_ipwf_linear_approximation_1_1(a1_0, a1_1, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("iocbio_ipwf_a11_find_zero_diff2: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            *slope = p1;
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

void iocbio_ipwf_a11_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int(-f1(x)*f2(x+y), x=0..L-y), y, order=3) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  //printf("void iocbio_ipwf_a11_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)\n");
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
        b0 += f_i*(2.0*f_ip1pj - f_ip2pj - f_ipj) + (-2.0*f_ip1pj + f_ip2pj + f_ipj)*f_ip1;
      }
      b0 += f_m1mjpn*f_m2pn - f_m2pn*f_m2mjpn + f_m1pn*(-f_m1mjpn + f_m2mjpn);
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int iocbio_ipwf_a11_find_extreme_diff3(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  //printf("int iocbio_ipwf_a11_find_extreme_diff3(int j0, int j1, double *fm, int n, int m, double* result)\n");
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
    iocbio_ipwf_a11_compute_coeffs_diff3(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3);
    count ++;
    if (count<3)
      continue;
    iocbio_ipwf_quadratic_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1, &p2);
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
            
int iocbio_ipwf_a11_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result, double* slope)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  //printf("int iocbio_ipwf_a11_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result, double* slope)\n");
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
    iocbio_ipwf_a11_compute_coeffs_diff3(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3);
    count ++;
    if (count<3)
      continue;
    iocbio_ipwf_linear_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1);
    if (p1!=0.0)
    {
       s = -p0/p1;
       ////printf("iocbio_ipwf_a11_find_zero_diff3: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            *slope = p1;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
void iocbio_ipwf_a11_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int(-f1(x)*f2(x+y), x=0..L-y), y, order) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */
  //printf("void iocbio_ipwf_a11_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3)\n");
  switch (order)
  {
    case 0:  iocbio_ipwf_a11_compute_coeffs_diff0(j, fm, n, m, a0, a1, a2, a3); break;
    case 1:  iocbio_ipwf_a11_compute_coeffs_diff1(j, fm, n, m, a0, a1, a2, a3); break;
    case 2:  iocbio_ipwf_a11_compute_coeffs_diff2(j, fm, n, m, a0, a1, a2, a3); break;
    case 3:  iocbio_ipwf_a11_compute_coeffs_diff3(j, fm, n, m, a0, a1, a2, a3); break;
    default:
      *a0 = 0.0;
      *a1 = 0.0;
      *a2 = 0.0;
      *a3 = 0.0;
  }
}
        
int iocbio_ipwf_a11_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  //printf("int iocbio_ipwf_a11_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)\n");
  switch (order)
  {
    case 0: return iocbio_ipwf_a11_find_extreme_diff0(j0, j1, fm, n, m, result);
    case 1: return iocbio_ipwf_a11_find_extreme_diff1(j0, j1, fm, n, m, result);
    case 2: return iocbio_ipwf_a11_find_extreme_diff2(j0, j1, fm, n, m, result);
    case 3: return iocbio_ipwf_a11_find_extreme_diff3(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
int iocbio_ipwf_a11_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result, double* slope)
{
  switch (order)
  {
    case 0: return iocbio_ipwf_a11_find_zero_diff0(j0, j1, fm, n, m, result, slope);
    case 1: return iocbio_ipwf_a11_find_zero_diff1(j0, j1, fm, n, m, result, slope);
    case 2: return iocbio_ipwf_a11_find_zero_diff2(j0, j1, fm, n, m, result, slope);
    case 3: return iocbio_ipwf_a11_find_zero_diff3(j0, j1, fm, n, m, result, slope);
    default:
      *result = 0.0;
      *slope = 0.0;
  }
  return -2;
}
        
double iocbio_ipwf_a11_evaluate(double y, double *fm, int n, int m, int order)
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
  //printf("double iocbio_ipwf_a11_evaluate(double y, double *fm, int n, int m, int order)\n");
  iocbio_ipwf_a11_compute_coeffs(j, fm, n, m, order, &a0, &a1, &a2, &a3);
  return a0+(a1+(a2+(a3)*r)*r)*r;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double iocbio_ipwf_a11_f1_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  //printf("double iocbio_ipwf_a11_f1_evaluate(double x, double *f, int n, int order)\n");
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

double iocbio_ipwf_a11_f2_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  //printf("double iocbio_ipwf_a11_f2_evaluate(double x, double *f, int n, int order)\n");
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
#define F(I) ((I)<0?(f[(I)+n]):((I)>=n?f[(I)-n]:f[(I)]))

void iocbio_ipwf_ap11_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* int(-f1(x)*f2(x+y), x=0..L) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 1;
  double *f = fm;
  //printf("void iocbio_ipwf_ap11_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)\n");
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double f_ipj, f_ip1pj, f_i, f_ip1, f_ip2pj;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip1 = F(i+1);
        f_ip2pj = F(i+2+j);
        b0 += -0.1666666666666667*f_ip1pj*f_i - 0.3333333333333333*f_i*f_ipj + f_ip1*(-0.3333333333333333*f_ip1pj - 0.1666666666666667*f_ipj);
        b1 += (0.5*f_ipj - 0.5*f_ip1pj)*f_i + (0.5*f_ipj - 0.5*f_ip1pj)*f_ip1;
        b2 += f_ip1*(f_ip1pj - 0.5*f_ip2pj - 0.5*f_ipj);
        b3 += (0.3333333333333333*f_ip1pj - 0.1666666666666667*f_ip2pj - 0.1666666666666667*f_ipj)*f_i + (-0.3333333333333333*f_ip1pj + 0.1666666666666667*f_ip2pj + 0.1666666666666667*f_ipj)*f_ip1;
      }
      b0 += 0;
      b1 += 0;
      b2 += 0;
      b3 += 0;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int iocbio_ipwf_ap11_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result)
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
  //printf("int iocbio_ipwf_ap11_find_extreme_diff0(int j0, int j1, double *fm, int n, int m, double* result)\n");
  for (j=start_j; j<end_j; ++j)
  {
    iocbio_ipwf_ap11_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    iocbio_ipwf_quadratic_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1, &p2);
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
            
int iocbio_ipwf_ap11_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result, double* slope)
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
  //printf("int iocbio_ipwf_ap11_find_zero_diff0(int j0, int j1, double *fm, int n, int m, double* result, double* slope)\n");
  for (j=start_j; j<end_j; ++j)
  {
    iocbio_ipwf_ap11_compute_coeffs_diff0(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    iocbio_ipwf_linear_approximation_1_3(a1_0, a1_1, a1_2, a1_3, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("iocbio_ipwf_ap11_find_zero_diff0: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            *slope = p1;
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
#define F(I) ((I)<0?(f[(I)+n]):((I)>=n?f[(I)-n]:f[(I)]))

void iocbio_ipwf_ap11_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int(-f1(x)*f2(x+y), x=0..L), y, order=1) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 1;
  double *f = fm;
  //printf("void iocbio_ipwf_ap11_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)\n");
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double f_ipj, f_ip1pj, f_i, f_ip1, f_ip2pj;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      
      for(i=0;i<=k;++i)
      {
        f_ipj = F(i+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip1 = F(i+1);
        f_ip2pj = F(i+2+j);
        b0 += (0.5*f_ipj - 0.5*f_ip1pj)*f_i + (0.5*f_ipj - 0.5*f_ip1pj)*f_ip1;
        b1 += (2.0*f_ip1pj - f_ip2pj - f_ipj)*f_ip1;
        b2 += (-f_ip1pj + 0.5*f_ip2pj + 0.5*f_ipj)*f_ip1 + f_i*(f_ip1pj - 0.5*f_ip2pj - 0.5*f_ipj);
      }
      b0 += 0;
      b1 += 0;
      b2 += 0;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int iocbio_ipwf_ap11_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result)
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
  //printf("int iocbio_ipwf_ap11_find_extreme_diff1(int j0, int j1, double *fm, int n, int m, double* result)\n");
  for (j=start_j; j<end_j; ++j)
  {
    iocbio_ipwf_ap11_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    iocbio_ipwf_quadratic_approximation_1_2(a1_0, a1_1, a1_2, &p0, &p1, &p2);
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
            
int iocbio_ipwf_ap11_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result, double* slope)
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  //printf("int iocbio_ipwf_ap11_find_zero_diff1(int j0, int j1, double *fm, int n, int m, double* result, double* slope)\n");
  double a1_0 = 0.0;
  double a1_1 = 0.0;
  double a1_2 = 0.0;
  double a1_3 = 0.0;
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  int dj = (start_j>end_j?-1:1);
  int count = (start_j<end_j?end_j-start_j:start_j-end_j);
  for (j=(dj==-1?start_j-1:start_j); count>0; j += dj, --count)
  {
    iocbio_ipwf_ap11_compute_coeffs_diff1(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    s = iocbio_ipwf_find_real_zero_in_01_2(a1_0, a1_1, a1_2, dj, slope);
    //printf("j,s,dj,zero=%d, %e, %d, %e\n",j,s, dj, a1_0+s*(a1_1+s*(a1_2+s*a1_3)));
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
#define F(I) ((I)<0?(f[(I)+n]):((I)>=n?f[(I)-n]:f[(I)]))

void iocbio_ipwf_ap11_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int(-f1(x)*f2(x+y), x=0..L), y, order=2) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 1;
  double *f = fm;
  //printf("void iocbio_ipwf_ap11_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)\n");
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double f_ip2pj, f_ip1pj, f_ip1, f_i, f_ipj;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      
      for(i=0;i<=k;++i)
      {
        f_ip2pj = F(i+2+j);
        f_ip1pj = F(i+1+j);
        f_ip1 = F(i+1);
        f_i = F(i);
        f_ipj = F(i+j);
        b0 += (2.0*f_ip1pj - f_ip2pj - f_ipj)*f_ip1;
        b1 += f_i*(2.0*f_ip1pj - f_ip2pj - f_ipj) + (-2.0*f_ip1pj + f_ip2pj + f_ipj)*f_ip1;
      }
      b0 += 0;
      b1 += 0;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int iocbio_ipwf_ap11_find_extreme_diff2(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  //printf("int iocbio_ipwf_ap11_find_extreme_diff2(int j0, int j1, double *fm, int n, int m, double* result)\n");
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
    iocbio_ipwf_ap11_compute_coeffs_diff2(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3);
    count ++;
    if (count<3)
      continue;
    iocbio_ipwf_quadratic_approximation_3_1(a1_0, a1_1, a2_0, a2_1, a3_0, a3_1, &p0, &p1, &p2);
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
            
int iocbio_ipwf_ap11_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result, double* slope)
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
  //printf("int iocbio_ipwf_ap11_find_zero_diff2(int j0, int j1, double *fm, int n, int m, double* result, double* slope)\n");
  for (j=start_j; j<end_j; ++j)
  {
    iocbio_ipwf_ap11_compute_coeffs_diff2(j, fm, n, m, &a1_0, &a1_1, &a1_2, &a1_3);
    iocbio_ipwf_linear_approximation_1_1(a1_0, a1_1, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("iocbio_ipwf_ap11_find_zero_diff2: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            *slope = p1;
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
#define F(I) ((I)<0?(f[(I)+n]):((I)>=n?f[(I)-n]:f[(I)]))

void iocbio_ipwf_ap11_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int(-f1(x)*f2(x+y), x=0..L), y, order=3) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */

  int p, i;
  int k = n - 1;
  double *f = fm;
  //printf("void iocbio_ipwf_ap11_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)\n");
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double f_ip2pj, f_ip1pj, f_i, f_ip1, f_ipj;
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      
      for(i=0;i<=k;++i)
      {
        f_ip2pj = F(i+2+j);
        f_ip1pj = F(i+1+j);
        f_i = F(i);
        f_ip1 = F(i+1);
        f_ipj = F(i+j);
        b0 += f_i*(2.0*f_ip1pj - f_ip2pj - f_ipj) + (-2.0*f_ip1pj + f_ip2pj + f_ipj)*f_ip1;
      }
      b0 += 0;
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
}
        
int iocbio_ipwf_ap11_find_extreme_diff3(int j0, int j1, double *fm, int n, int m, double* result)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  //printf("int iocbio_ipwf_ap11_find_extreme_diff3(int j0, int j1, double *fm, int n, int m, double* result)\n");
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
    iocbio_ipwf_ap11_compute_coeffs_diff3(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3);
    count ++;
    if (count<3)
      continue;
    iocbio_ipwf_quadratic_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1, &p2);
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
            
int iocbio_ipwf_ap11_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result, double* slope)
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  //printf("int iocbio_ipwf_ap11_find_zero_diff3(int j0, int j1, double *fm, int n, int m, double* result, double* slope)\n");
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
    iocbio_ipwf_ap11_compute_coeffs_diff3(j, fm, n, m, &a3_0, &a3_1, &a3_2, &a3_3);
    count ++;
    if (count<3)
      continue;
    iocbio_ipwf_linear_approximation_3_0(a1_0, a2_0, a3_0, &p0, &p1);
    if (p1!=0.0)
    {
       s = -p0/p1;
       ////printf("iocbio_ipwf_ap11_find_zero_diff3: j=%d, p0=%f, p1=%f, s=%f\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            *slope = p1;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            
void iocbio_ipwf_ap11_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3)
{
  /* diff(int(-f1(x)*f2(x+y), x=0..L-y), y, order) = sum(a_k*r^k, k=0..3) where y=j+r
     f1(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i
     f2(x)=sum([0<=s<1]*((-(F(i)) + (F(i+1)))*s + (F(i))), i=0..N-1) where s=x-i */
  //printf("void iocbio_ipwf_ap11_compute_coeffs(int j, double *fm, int n, int m, int order, double* a0, double* a1, double* a2, double* a3)\n");
  switch (order)
  {
    case 0:  iocbio_ipwf_ap11_compute_coeffs_diff0(j, fm, n, m, a0, a1, a2, a3); break;
    case 1:  iocbio_ipwf_ap11_compute_coeffs_diff1(j, fm, n, m, a0, a1, a2, a3); break;
    case 2:  iocbio_ipwf_ap11_compute_coeffs_diff2(j, fm, n, m, a0, a1, a2, a3); break;
    case 3:  iocbio_ipwf_ap11_compute_coeffs_diff3(j, fm, n, m, a0, a1, a2, a3); break;
    default:
      *a0 = 0.0;
      *a1 = 0.0;
      *a2 = 0.0;
      *a3 = 0.0;
  }
}
        
int iocbio_ipwf_ap11_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)
{
  //printf("int iocbio_ipwf_ap11_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)\n");
  switch (order)
  {
    case 0: return iocbio_ipwf_ap11_find_extreme_diff0(j0, j1, fm, n, m, result);
    case 1: return iocbio_ipwf_ap11_find_extreme_diff1(j0, j1, fm, n, m, result);
    case 2: return iocbio_ipwf_ap11_find_extreme_diff2(j0, j1, fm, n, m, result);
    case 3: return iocbio_ipwf_ap11_find_extreme_diff3(j0, j1, fm, n, m, result);
    default:
      *result = 0.0;
  }
  return -2;
}
        
int iocbio_ipwf_ap11_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result, double* slope)
{
  switch (order)
  {
    case 0: return iocbio_ipwf_ap11_find_zero_diff0(j0, j1, fm, n, m, result, slope);
    case 1: return iocbio_ipwf_ap11_find_zero_diff1(j0, j1, fm, n, m, result, slope);
    case 2: return iocbio_ipwf_ap11_find_zero_diff2(j0, j1, fm, n, m, result, slope);
    case 3: return iocbio_ipwf_ap11_find_zero_diff3(j0, j1, fm, n, m, result, slope);
    default:
      *result = 0.0;
      *slope = 0.0;
  }
  return -2;
}
        
double iocbio_ipwf_ap11_evaluate(double y, double *fm, int n, int m, int order)
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
  //printf("double iocbio_ipwf_ap11_evaluate(double y, double *fm, int n, int m, int order)\n");
  iocbio_ipwf_ap11_compute_coeffs(j, fm, n, m, order, &a0, &a1, &a2, &a3);
  return a0+(a1+(a2+(a3)*r)*r)*r;
}
        
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

double iocbio_ipwf_ap11_f1_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  //printf("double iocbio_ipwf_ap11_f1_evaluate(double x, double *f, int n, int order)\n");
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

double iocbio_ipwf_ap11_f2_evaluate(double x, double *f, int n, int order)
{
  int i = floor(x);
  double s = x - floor(x);
  //printf("double iocbio_ipwf_ap11_f2_evaluate(double x, double *f, int n, int order)\n");
  switch (order)
  {
    case 0: return F(i) + (-(F(i)) + (F(i+1)))*s;
    case 1: return -(F(i)) + (F(i+1));
  }
  return 0.0;
}
        
void iocbio_ipwf_quadratic_approximation_3_0(double a1_0, double a2_0, double a3_0, double* p0, double* p1, double* p2)
{
  *p0 = 4.3209876543209880e-01*a1_0+5.8024691358024694e-01*a2_0+-1.2345679012345680e-02*a3_0;
  *p1 = -6.9135802469135799e-01*a1_0+4.9382716049382719e-01*a2_0+1.9753086419753091e-01*a3_0;
  *p2 = 2.4691358024691359e-01*a1_0+-4.9382716049382719e-01*a2_0+2.4691358024691359e-01*a3_0;
}
            
void iocbio_ipwf_quadratic_approximation_1_0(double a1_0, double* p0, double* p1, double* p2)
{
  *p0 = 1.0000000000000000e+00*a1_0;
  *p1 = 0.0000000000000000e+00*a1_0;
  *p2 = 0.0000000000000000e+00*a1_0;
}
            
void iocbio_ipwf_linear_approximation_3_0(double a1_0, double a2_0, double a3_0, double* p0, double* p1)
{
  *p0 = 5.5555555555555558e-01*a1_0+3.3333333333333331e-01*a2_0+1.1111111111111110e-01*a3_0;
  *p1 = -4.4444444444444442e-01*a1_0+0.0000000000000000e+00*a2_0+4.4444444444444442e-01*a3_0;
}
            
void iocbio_ipwf_linear_approximation_1_0(double a1_0, double* p0, double* p1)
{
  *p0 = 1.0000000000000000e+00*a1_0;
  *p1 = 0.0000000000000000e+00*a1_0;
}
            
void iocbio_ipwf_quadratic_approximation_3_1(double a1_0, double a1_1, double a2_0, double a2_1, double a3_0, double a3_1, double* p0, double* p1, double* p2)
{
  *p0 = 4.3209876543209880e-01*a1_0+2.5925925925925930e-01*a1_1+5.8024691358024694e-01*a2_0+2.7160493827160492e-01*a2_1+-1.2345679012345680e-02*a3_0+-8.6419753086419748e-02*a3_1;
  *p1 = -6.9135802469135799e-01*a1_0+-1.8518518518518520e-01*a1_1+4.9382716049382719e-01*a2_0+2.8395061728395060e-01*a2_1+1.9753086419753091e-01*a3_0+1.2345679012345680e-02*a3_1;
  *p2 = 2.4691358024691359e-01*a1_0+0.0000000000000000e+00*a1_1+-4.9382716049382719e-01*a2_0+-2.4691358024691359e-01*a2_1+2.4691358024691359e-01*a3_0+2.4691358024691359e-01*a3_1;
}
            
void iocbio_ipwf_quadratic_approximation_1_1(double a1_0, double a1_1, double* p0, double* p1, double* p2)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1;
  *p2 = 0.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1;
}
            
void iocbio_ipwf_linear_approximation_3_1(double a1_0, double a1_1, double a2_0, double a2_1, double a3_0, double a3_1, double* p0, double* p1)
{
  *p0 = 5.5555555555555558e-01*a1_0+2.5925925925925930e-01*a1_1+3.3333333333333331e-01*a2_0+1.4814814814814811e-01*a2_1+1.1111111111111110e-01*a3_0+3.7037037037037042e-02*a3_1;
  *p1 = -4.4444444444444442e-01*a1_0+-1.8518518518518520e-01*a1_1+0.0000000000000000e+00*a2_0+3.7037037037037042e-02*a2_1+4.4444444444444442e-01*a3_0+2.5925925925925930e-01*a3_1;
}
            
void iocbio_ipwf_linear_approximation_1_1(double a1_0, double a1_1, double* p0, double* p1)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1;
}
            
void iocbio_ipwf_quadratic_approximation_3_2(double a1_0, double a1_1, double a1_2, double a2_0, double a2_1, double a2_2, double a3_0, double a3_1, double a3_2, double* p0, double* p1, double* p2)
{
  *p0 = 4.3209876543209880e-01*a1_0+2.5925925925925930e-01*a1_1+1.8518518518518520e-01*a1_2+5.8024691358024694e-01*a2_0+2.7160493827160492e-01*a2_1+1.7283950617283950e-01*a2_2+-1.2345679012345680e-02*a3_0+-8.6419753086419748e-02*a3_1+-8.6419753086419748e-02*a3_2;
  *p1 = -6.9135802469135799e-01*a1_0+-1.8518518518518520e-01*a1_1+-7.4074074074074070e-02*a1_2+4.9382716049382719e-01*a2_0+2.8395061728395060e-01*a2_1+1.9753086419753091e-01*a2_2+1.9753086419753091e-01*a3_0+1.2345679012345680e-02*a3_1+-2.4691358024691360e-02*a3_2;
  *p2 = 2.4691358024691359e-01*a1_0+0.0000000000000000e+00*a1_1+-3.7037037037037042e-02*a1_2+-4.9382716049382719e-01*a2_0+-2.4691358024691359e-01*a2_1+-1.6049382716049379e-01*a2_2+2.4691358024691359e-01*a3_0+2.4691358024691359e-01*a3_1+2.0987654320987650e-01*a3_2;
}
            
void iocbio_ipwf_quadratic_approximation_1_2(double a1_0, double a1_1, double a1_2, double* p0, double* p1, double* p2)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2;
  *p2 = 0.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2;
}
            
void iocbio_ipwf_linear_approximation_3_2(double a1_0, double a1_1, double a1_2, double a2_0, double a2_1, double a2_2, double a3_0, double a3_1, double a3_2, double* p0, double* p1)
{
  *p0 = 5.5555555555555558e-01*a1_0+2.5925925925925930e-01*a1_1+1.6666666666666671e-01*a1_2+3.3333333333333331e-01*a2_0+1.4814814814814811e-01*a2_1+9.2592592592592587e-02*a2_2+1.1111111111111110e-01*a3_0+3.7037037037037042e-02*a3_1+1.8518518518518521e-02*a3_2;
  *p1 = -4.4444444444444442e-01*a1_0+-1.8518518518518520e-01*a1_1+-1.1111111111111110e-01*a1_2+0.0000000000000000e+00*a2_0+3.7037037037037042e-02*a2_1+3.7037037037037042e-02*a2_2+4.4444444444444442e-01*a3_0+2.5925925925925930e-01*a3_1+1.8518518518518520e-01*a3_2;
}
            
void iocbio_ipwf_linear_approximation_1_2(double a1_0, double a1_1, double a1_2, double* p0, double* p1)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+-1.6666666666666671e-01*a1_2;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2;
}
            
void iocbio_ipwf_quadratic_approximation_3_3(double a1_0, double a1_1, double a1_2, double a1_3, double a2_0, double a2_1, double a2_2, double a2_3, double a3_0, double a3_1, double a3_2, double a3_3, double* p0, double* p1, double* p2)
{
  *p0 = 4.3209876543209880e-01*a1_0+2.5925925925925930e-01*a1_1+1.8518518518518520e-01*a1_2+1.4382716049382721e-01*a1_3+5.8024691358024694e-01*a2_0+2.7160493827160492e-01*a2_1+1.7283950617283950e-01*a2_2+1.2530864197530861e-01*a2_3+-1.2345679012345680e-02*a3_0+-8.6419753086419748e-02*a3_1+-8.6419753086419748e-02*a3_2+-7.8395061728395055e-02*a3_3;
  *p1 = -6.9135802469135799e-01*a1_0+-1.8518518518518520e-01*a1_1+-7.4074074074074070e-02*a1_2+-3.4567901234567898e-02*a1_3+4.9382716049382719e-01*a2_0+2.8395061728395060e-01*a2_1+1.9753086419753091e-01*a2_2+1.5061728395061730e-01*a2_3+1.9753086419753091e-01*a3_0+1.2345679012345680e-02*a3_1+-2.4691358024691360e-02*a3_2+-3.4567901234567898e-02*a3_3;
  *p2 = 2.4691358024691359e-01*a1_0+0.0000000000000000e+00*a1_1+-3.7037037037037042e-02*a1_2+-4.3209876543209881e-02*a1_3+-4.9382716049382719e-01*a2_0+-2.4691358024691359e-01*a2_1+-1.6049382716049379e-01*a2_2+-1.1728395061728400e-01*a2_3+2.4691358024691359e-01*a3_0+2.4691358024691359e-01*a3_1+2.0987654320987650e-01*a3_2+1.7901234567901231e-01*a3_3;
}
            
void iocbio_ipwf_quadratic_approximation_1_3(double a1_0, double a1_1, double a1_2, double a1_3, double* p0, double* p1, double* p2)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+5.0000000000000003e-02*a1_3;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+-5.9999999999999998e-01*a1_3;
  *p2 = 0.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+1.5000000000000000e+00*a1_3;
}
            
void iocbio_ipwf_linear_approximation_3_3(double a1_0, double a1_1, double a1_2, double a1_3, double a2_0, double a2_1, double a2_2, double a2_3, double a3_0, double a3_1, double a3_2, double a3_3, double* p0, double* p1)
{
  *p0 = 5.5555555555555558e-01*a1_0+2.5925925925925930e-01*a1_1+1.6666666666666671e-01*a1_2+1.2222222222222220e-01*a1_3+3.3333333333333331e-01*a2_0+1.4814814814814811e-01*a2_1+9.2592592592592587e-02*a2_2+6.6666666666666666e-02*a2_3+1.1111111111111110e-01*a3_0+3.7037037037037042e-02*a3_1+1.8518518518518521e-02*a3_2+1.1111111111111110e-02*a3_3;
  *p1 = -4.4444444444444442e-01*a1_0+-1.8518518518518520e-01*a1_1+-1.1111111111111110e-01*a1_2+-7.7777777777777779e-02*a1_3+0.0000000000000000e+00*a2_0+3.7037037037037042e-02*a2_1+3.7037037037037042e-02*a2_2+3.3333333333333333e-02*a2_3+4.4444444444444442e-01*a3_0+2.5925925925925930e-01*a3_1+1.8518518518518520e-01*a3_2+1.4444444444444440e-01*a3_3;
}
            
void iocbio_ipwf_linear_approximation_1_3(double a1_0, double a1_1, double a1_2, double a1_3, double* p0, double* p1)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+-1.6666666666666671e-01*a1_2+-2.0000000000000001e-01*a1_3;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+9.0000000000000002e-01*a1_3;
}
            
void iocbio_ipwf_quadratic_approximation_3_4(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double* p0, double* p1, double* p2)
{
  *p0 = 4.3209876543209880e-01*a1_0+2.5925925925925930e-01*a1_1+1.8518518518518520e-01*a1_2+1.4382716049382721e-01*a1_3+1.1746031746031750e-01*a1_4+5.8024691358024694e-01*a2_0+2.7160493827160492e-01*a2_1+1.7283950617283950e-01*a2_2+1.2530864197530861e-01*a2_3+9.7707231040564377e-02*a2_4+-1.2345679012345680e-02*a3_0+-8.6419753086419748e-02*a3_1+-8.6419753086419748e-02*a3_2+-7.8395061728395055e-02*a3_3+-7.0194003527336860e-02*a3_4;
  *p1 = -6.9135802469135799e-01*a1_0+-1.8518518518518520e-01*a1_1+-7.4074074074074070e-02*a1_2+-3.4567901234567898e-02*a1_3+-1.6931216931216929e-02*a1_4+4.9382716049382719e-01*a2_0+2.8395061728395060e-01*a2_1+1.9753086419753091e-01*a2_2+1.5061728395061730e-01*a2_3+1.2134038800705470e-01*a2_4+1.9753086419753091e-01*a3_0+1.2345679012345680e-02*a3_1+-2.4691358024691360e-02*a3_2+-3.4567901234567898e-02*a3_3+-3.6684303350970018e-02*a3_4;
  *p2 = 2.4691358024691359e-01*a1_0+0.0000000000000000e+00*a1_1+-3.7037037037037042e-02*a1_2+-4.3209876543209881e-02*a1_3+-4.2328042328042333e-02*a1_4+-4.9382716049382719e-01*a2_0+-2.4691358024691359e-01*a2_1+-1.6049382716049379e-01*a2_2+-1.1728395061728400e-01*a2_3+-9.1710758377425039e-02*a2_4+2.4691358024691359e-01*a3_0+2.4691358024691359e-01*a3_1+2.0987654320987650e-01*a3_2+1.7901234567901231e-01*a3_3+1.5520282186948850e-01*a3_4;
}
            
void iocbio_ipwf_quadratic_approximation_1_4(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double* p0, double* p1, double* p2)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+5.0000000000000003e-02*a1_3+8.5714285714285715e-02*a1_4;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+-5.9999999999999998e-01*a1_3+-9.1428571428571426e-01*a1_4;
  *p2 = 0.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+1.5000000000000000e+00*a1_3+1.7142857142857140e+00*a1_4;
}
            
void iocbio_ipwf_linear_approximation_3_4(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double* p0, double* p1)
{
  *p0 = 5.5555555555555558e-01*a1_0+2.5925925925925930e-01*a1_1+1.6666666666666671e-01*a1_2+1.2222222222222220e-01*a1_3+9.6296296296296297e-02*a1_4+3.3333333333333331e-01*a2_0+1.4814814814814811e-01*a2_1+9.2592592592592587e-02*a2_2+6.6666666666666666e-02*a2_3+5.1851851851851850e-02*a2_4+1.1111111111111110e-01*a3_0+3.7037037037037042e-02*a3_1+1.8518518518518521e-02*a3_2+1.1111111111111110e-02*a3_3+7.4074074074074068e-03*a3_4;
  *p1 = -4.4444444444444442e-01*a1_0+-1.8518518518518520e-01*a1_1+-1.1111111111111110e-01*a1_2+-7.7777777777777779e-02*a1_3+-5.9259259259259262e-02*a1_4+0.0000000000000000e+00*a2_0+3.7037037037037042e-02*a2_1+3.7037037037037042e-02*a2_2+3.3333333333333333e-02*a2_3+2.9629629629629631e-02*a2_4+4.4444444444444442e-01*a3_0+2.5925925925925930e-01*a3_1+1.8518518518518520e-01*a3_2+1.4444444444444440e-01*a3_3+1.1851851851851850e-01*a3_4;
}
            
void iocbio_ipwf_linear_approximation_1_4(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double* p0, double* p1)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+-1.6666666666666671e-01*a1_2+-2.0000000000000001e-01*a1_3+-2.0000000000000001e-01*a1_4;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+9.0000000000000002e-01*a1_3+8.0000000000000004e-01*a1_4;
}
            
void iocbio_ipwf_quadratic_approximation_3_5(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a2_5, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double a3_5, double* p0, double* p1, double* p2)
{
  *p0 = 4.3209876543209880e-01*a1_0+2.5925925925925930e-01*a1_1+1.8518518518518520e-01*a1_2+1.4382716049382721e-01*a1_3+1.1746031746031750e-01*a1_4+9.9206349206349215e-02*a1_5+5.8024691358024694e-01*a2_0+2.7160493827160492e-01*a2_1+1.7283950617283950e-01*a2_2+1.2530864197530861e-01*a2_3+9.7707231040564377e-02*a2_4+7.9805996472663135e-02*a2_5+-1.2345679012345680e-02*a3_0+-8.6419753086419748e-02*a3_1+-8.6419753086419748e-02*a3_2+-7.8395061728395055e-02*a3_3+-7.0194003527336860e-02*a3_4+-6.3051146384479714e-02*a3_5;
  *p1 = -6.9135802469135799e-01*a1_0+-1.8518518518518520e-01*a1_1+-7.4074074074074070e-02*a1_2+-3.4567901234567898e-02*a1_3+-1.6931216931216929e-02*a1_4+-7.9365079365079361e-03*a1_5+4.9382716049382719e-01*a2_0+2.8395061728395060e-01*a2_1+1.9753086419753091e-01*a2_2+1.5061728395061730e-01*a2_3+1.2134038800705470e-01*a2_4+1.0141093474426810e-01*a2_5+1.9753086419753091e-01*a3_0+1.2345679012345680e-02*a3_1+-2.4691358024691360e-02*a3_2+-3.4567901234567898e-02*a3_3+-3.6684303350970018e-02*a3_4+-3.6155202821869493e-02*a3_5;
  *p2 = 2.4691358024691359e-01*a1_0+0.0000000000000000e+00*a1_1+-3.7037037037037042e-02*a1_2+-4.3209876543209881e-02*a1_3+-4.2328042328042333e-02*a1_4+-3.9682539682539680e-02*a1_5+-4.9382716049382719e-01*a2_0+-2.4691358024691359e-01*a2_1+-1.6049382716049379e-01*a2_2+-1.1728395061728400e-01*a2_3+-9.1710758377425039e-02*a2_4+-7.4955908289241618e-02*a2_5+2.4691358024691359e-01*a3_0+2.4691358024691359e-01*a3_1+2.0987654320987650e-01*a3_2+1.7901234567901231e-01*a3_3+1.5520282186948850e-01*a3_4+1.3668430335097001e-01*a3_5;
}
            
void iocbio_ipwf_quadratic_approximation_1_5(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double* p0, double* p1, double* p2)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+5.0000000000000003e-02*a1_3+8.5714285714285715e-02*a1_4+1.0714285714285710e-01*a1_5;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+-5.9999999999999998e-01*a1_3+-9.1428571428571426e-01*a1_4+-1.0714285714285710e+00*a1_5;
  *p2 = 0.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+1.5000000000000000e+00*a1_3+1.7142857142857140e+00*a1_4+1.7857142857142860e+00*a1_5;
}
            
void iocbio_ipwf_linear_approximation_3_5(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a2_5, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double a3_5, double* p0, double* p1)
{
  *p0 = 5.5555555555555558e-01*a1_0+2.5925925925925930e-01*a1_1+1.6666666666666671e-01*a1_2+1.2222222222222220e-01*a1_3+9.6296296296296297e-02*a1_4+7.9365079365079361e-02*a1_5+3.3333333333333331e-01*a2_0+1.4814814814814811e-01*a2_1+9.2592592592592587e-02*a2_2+6.6666666666666666e-02*a2_3+5.1851851851851850e-02*a2_4+4.2328042328042333e-02*a2_5+1.1111111111111110e-01*a3_0+3.7037037037037042e-02*a3_1+1.8518518518518521e-02*a3_2+1.1111111111111110e-02*a3_3+7.4074074074074068e-03*a3_4+5.2910052910052907e-03*a3_5;
  *p1 = -4.4444444444444442e-01*a1_0+-1.8518518518518520e-01*a1_1+-1.1111111111111110e-01*a1_2+-7.7777777777777779e-02*a1_3+-5.9259259259259262e-02*a1_4+-4.7619047619047623e-02*a1_5+0.0000000000000000e+00*a2_0+3.7037037037037042e-02*a2_1+3.7037037037037042e-02*a2_2+3.3333333333333333e-02*a2_3+2.9629629629629631e-02*a2_4+2.6455026455026460e-02*a2_5+4.4444444444444442e-01*a3_0+2.5925925925925930e-01*a3_1+1.8518518518518520e-01*a3_2+1.4444444444444440e-01*a3_3+1.1851851851851850e-01*a3_4+1.0052910052910050e-01*a3_5;
}
            
void iocbio_ipwf_linear_approximation_1_5(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double* p0, double* p1)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+-1.6666666666666671e-01*a1_2+-2.0000000000000001e-01*a1_3+-2.0000000000000001e-01*a1_4+-1.9047619047619049e-01*a1_5;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+9.0000000000000002e-01*a1_3+8.0000000000000004e-01*a1_4+7.1428571428571430e-01*a1_5;
}
            
void iocbio_ipwf_quadratic_approximation_3_6(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a2_5, double a2_6, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double a3_5, double a3_6, double* p0, double* p1, double* p2)
{
  *p0 = 4.3209876543209880e-01*a1_0+2.5925925925925930e-01*a1_1+1.8518518518518520e-01*a1_2+1.4382716049382721e-01*a1_3+1.1746031746031750e-01*a1_4+9.9206349206349215e-02*a1_5+8.5831863609641387e-02*a1_6+5.8024691358024694e-01*a2_0+2.7160493827160492e-01*a2_1+1.7283950617283950e-01*a2_2+1.2530864197530861e-01*a2_3+9.7707231040564377e-02*a2_4+7.9805996472663135e-02*a2_5+6.7313345091122870e-02*a2_6+-1.2345679012345680e-02*a3_0+-8.6419753086419748e-02*a3_1+-8.6419753086419748e-02*a3_2+-7.8395061728395055e-02*a3_3+-7.0194003527336860e-02*a3_4+-6.3051146384479714e-02*a3_5+-5.7025279247501469e-02*a3_6;
  *p1 = -6.9135802469135799e-01*a1_0+-1.8518518518518520e-01*a1_1+-7.4074074074074070e-02*a1_2+-3.4567901234567898e-02*a1_3+-1.6931216931216929e-02*a1_4+-7.9365079365079361e-03*a1_5+-2.9394473838918280e-03*a1_6+4.9382716049382719e-01*a2_0+2.8395061728395060e-01*a2_1+1.9753086419753091e-01*a2_2+1.5061728395061730e-01*a2_3+1.2134038800705470e-01*a2_4+1.0141093474426810e-01*a2_5+8.7007642563198123e-02*a2_6+1.9753086419753091e-01*a3_0+1.2345679012345680e-02*a3_1+-2.4691358024691360e-02*a3_2+-3.4567901234567898e-02*a3_3+-3.6684303350970018e-02*a3_4+-3.6155202821869493e-02*a3_5+-3.4685479129923570e-02*a3_6;
  *p2 = 2.4691358024691359e-01*a1_0+0.0000000000000000e+00*a1_1+-3.7037037037037042e-02*a1_2+-4.3209876543209881e-02*a1_3+-4.2328042328042333e-02*a1_4+-3.9682539682539680e-02*a1_5+-3.6743092298647848e-02*a1_6+-4.9382716049382719e-01*a2_0+-2.4691358024691359e-01*a2_1+-1.6049382716049379e-01*a2_2+-1.1728395061728400e-01*a2_3+-9.1710758377425039e-02*a2_4+-7.4955908289241618e-02*a2_5+-6.3198118753674315e-02*a2_6+2.4691358024691359e-01*a3_0+2.4691358024691359e-01*a3_1+2.0987654320987650e-01*a3_2+1.7901234567901231e-01*a3_3+1.5520282186948850e-01*a3_4+1.3668430335097001e-01*a3_5+1.2198706643151090e-01*a3_6;
}
            
void iocbio_ipwf_quadratic_approximation_1_6(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double* p0, double* p1, double* p2)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+5.0000000000000003e-02*a1_3+8.5714285714285715e-02*a1_4+1.0714285714285710e-01*a1_5+1.1904761904761900e-01*a1_6;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+-5.9999999999999998e-01*a1_3+-9.1428571428571426e-01*a1_4+-1.0714285714285710e+00*a1_5+-1.1428571428571430e+00*a1_6;
  *p2 = 0.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+1.5000000000000000e+00*a1_3+1.7142857142857140e+00*a1_4+1.7857142857142860e+00*a1_5+1.7857142857142860e+00*a1_6;
}
            
void iocbio_ipwf_linear_approximation_3_6(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a2_5, double a2_6, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double a3_5, double a3_6, double* p0, double* p1)
{
  *p0 = 5.5555555555555558e-01*a1_0+2.5925925925925930e-01*a1_1+1.6666666666666671e-01*a1_2+1.2222222222222220e-01*a1_3+9.6296296296296297e-02*a1_4+7.9365079365079361e-02*a1_5+6.7460317460317457e-02*a1_6+3.3333333333333331e-01*a2_0+1.4814814814814811e-01*a2_1+9.2592592592592587e-02*a2_2+6.6666666666666666e-02*a2_3+5.1851851851851850e-02*a2_4+4.2328042328042333e-02*a2_5+3.5714285714285712e-02*a2_6+1.1111111111111110e-01*a3_0+3.7037037037037042e-02*a3_1+1.8518518518518521e-02*a3_2+1.1111111111111110e-02*a3_3+7.4074074074074068e-03*a3_4+5.2910052910052907e-03*a3_5+3.9682539682539680e-03*a3_6;
  *p1 = -4.4444444444444442e-01*a1_0+-1.8518518518518520e-01*a1_1+-1.1111111111111110e-01*a1_2+-7.7777777777777779e-02*a1_3+-5.9259259259259262e-02*a1_4+-4.7619047619047623e-02*a1_5+-3.9682539682539680e-02*a1_6+0.0000000000000000e+00*a2_0+3.7037037037037042e-02*a2_1+3.7037037037037042e-02*a2_2+3.3333333333333333e-02*a2_3+2.9629629629629631e-02*a2_4+2.6455026455026460e-02*a2_5+2.3809523809523812e-02*a2_6+4.4444444444444442e-01*a3_0+2.5925925925925930e-01*a3_1+1.8518518518518520e-01*a3_2+1.4444444444444440e-01*a3_3+1.1851851851851850e-01*a3_4+1.0052910052910050e-01*a3_5+8.7301587301587297e-02*a3_6;
}
            
void iocbio_ipwf_linear_approximation_1_6(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double* p0, double* p1)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+-1.6666666666666671e-01*a1_2+-2.0000000000000001e-01*a1_3+-2.0000000000000001e-01*a1_4+-1.9047619047619049e-01*a1_5+-1.7857142857142860e-01*a1_6;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+9.0000000000000002e-01*a1_3+8.0000000000000004e-01*a1_4+7.1428571428571430e-01*a1_5+6.4285714285714290e-01*a1_6;
}
            
void iocbio_ipwf_quadratic_approximation_3_7(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double a1_7, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a2_5, double a2_6, double a2_7, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double a3_5, double a3_6, double a3_7, double* p0, double* p1, double* p2)
{
  *p0 = 4.3209876543209880e-01*a1_0+2.5925925925925930e-01*a1_1+1.8518518518518520e-01*a1_2+1.4382716049382721e-01*a1_3+1.1746031746031750e-01*a1_4+9.9206349206349215e-02*a1_5+8.5831863609641387e-02*a1_6+7.5617283950617287e-02*a1_7+5.8024691358024694e-01*a2_0+2.7160493827160492e-01*a2_1+1.7283950617283950e-01*a2_2+1.2530864197530861e-01*a2_3+9.7707231040564377e-02*a2_4+7.9805996472663135e-02*a2_5+6.7313345091122870e-02*a2_6+5.8127572016460911e-02*a2_7+-1.2345679012345680e-02*a3_0+-8.6419753086419748e-02*a3_1+-8.6419753086419748e-02*a3_2+-7.8395061728395055e-02*a3_3+-7.0194003527336860e-02*a3_4+-6.3051146384479714e-02*a3_5+-5.7025279247501469e-02*a3_6+-5.1954732510288072e-02*a3_7;
  *p1 = -6.9135802469135799e-01*a1_0+-1.8518518518518520e-01*a1_1+-7.4074074074074070e-02*a1_2+-3.4567901234567898e-02*a1_3+-1.6931216931216929e-02*a1_4+-7.9365079365079361e-03*a1_5+-2.9394473838918280e-03*a1_6+0.0000000000000000e+00*a1_7+4.9382716049382719e-01*a2_0+2.8395061728395060e-01*a2_1+1.9753086419753091e-01*a2_2+1.5061728395061730e-01*a2_3+1.2134038800705470e-01*a2_4+1.0141093474426810e-01*a2_5+8.7007642563198123e-02*a2_6+7.6131687242798354e-02*a2_7+1.9753086419753091e-01*a3_0+1.2345679012345680e-02*a3_1+-2.4691358024691360e-02*a3_2+-3.4567901234567898e-02*a3_3+-3.6684303350970018e-02*a3_4+-3.6155202821869493e-02*a3_5+-3.4685479129923570e-02*a3_6+-3.2921810699588480e-02*a3_7;
  *p2 = 2.4691358024691359e-01*a1_0+0.0000000000000000e+00*a1_1+-3.7037037037037042e-02*a1_2+-4.3209876543209881e-02*a1_3+-4.2328042328042333e-02*a1_4+-3.9682539682539680e-02*a1_5+-3.6743092298647848e-02*a1_6+-3.3950617283950622e-02*a1_7+-4.9382716049382719e-01*a2_0+-2.4691358024691359e-01*a2_1+-1.6049382716049379e-01*a2_2+-1.1728395061728400e-01*a2_3+-9.1710758377425039e-02*a2_4+-7.4955908289241618e-02*a2_5+-6.3198118753674315e-02*a2_6+-5.4526748971193417e-02*a2_7+2.4691358024691359e-01*a3_0+2.4691358024691359e-01*a3_1+2.0987654320987650e-01*a3_2+1.7901234567901231e-01*a3_3+1.5520282186948850e-01*a3_4+1.3668430335097001e-01*a3_5+1.2198706643151090e-01*a3_6+1.1008230452674900e-01*a3_7;
}
            
void iocbio_ipwf_quadratic_approximation_1_7(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double a1_7, double* p0, double* p1, double* p2)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+5.0000000000000003e-02*a1_3+8.5714285714285715e-02*a1_4+1.0714285714285710e-01*a1_5+1.1904761904761900e-01*a1_6+1.2500000000000000e-01*a1_7;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+0.0000000000000000e+00*a1_2+-5.9999999999999998e-01*a1_3+-9.1428571428571426e-01*a1_4+-1.0714285714285710e+00*a1_5+-1.1428571428571430e+00*a1_6+-1.1666666666666670e+00*a1_7;
  *p2 = 0.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+1.5000000000000000e+00*a1_3+1.7142857142857140e+00*a1_4+1.7857142857142860e+00*a1_5+1.7857142857142860e+00*a1_6+1.7500000000000000e+00*a1_7;
}
            
void iocbio_ipwf_linear_approximation_3_7(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double a1_7, double a2_0, double a2_1, double a2_2, double a2_3, double a2_4, double a2_5, double a2_6, double a2_7, double a3_0, double a3_1, double a3_2, double a3_3, double a3_4, double a3_5, double a3_6, double a3_7, double* p0, double* p1)
{
  *p0 = 5.5555555555555558e-01*a1_0+2.5925925925925930e-01*a1_1+1.6666666666666671e-01*a1_2+1.2222222222222220e-01*a1_3+9.6296296296296297e-02*a1_4+7.9365079365079361e-02*a1_5+6.7460317460317457e-02*a1_6+5.8641975308641979e-02*a1_7+3.3333333333333331e-01*a2_0+1.4814814814814811e-01*a2_1+9.2592592592592587e-02*a2_2+6.6666666666666666e-02*a2_3+5.1851851851851850e-02*a2_4+4.2328042328042333e-02*a2_5+3.5714285714285712e-02*a2_6+3.0864197530864199e-02*a2_7+1.1111111111111110e-01*a3_0+3.7037037037037042e-02*a3_1+1.8518518518518521e-02*a3_2+1.1111111111111110e-02*a3_3+7.4074074074074068e-03*a3_4+5.2910052910052907e-03*a3_5+3.9682539682539680e-03*a3_6+3.0864197530864200e-03*a3_7;
  *p1 = -4.4444444444444442e-01*a1_0+-1.8518518518518520e-01*a1_1+-1.1111111111111110e-01*a1_2+-7.7777777777777779e-02*a1_3+-5.9259259259259262e-02*a1_4+-4.7619047619047623e-02*a1_5+-3.9682539682539680e-02*a1_6+-3.3950617283950622e-02*a1_7+0.0000000000000000e+00*a2_0+3.7037037037037042e-02*a2_1+3.7037037037037042e-02*a2_2+3.3333333333333333e-02*a2_3+2.9629629629629631e-02*a2_4+2.6455026455026460e-02*a2_5+2.3809523809523812e-02*a2_6+2.1604938271604941e-02*a2_7+4.4444444444444442e-01*a3_0+2.5925925925925930e-01*a3_1+1.8518518518518520e-01*a3_2+1.4444444444444440e-01*a3_3+1.1851851851851850e-01*a3_4+1.0052910052910050e-01*a3_5+8.7301587301587297e-02*a3_6+7.7160493827160490e-02*a3_7;
}
            
void iocbio_ipwf_linear_approximation_1_7(double a1_0, double a1_1, double a1_2, double a1_3, double a1_4, double a1_5, double a1_6, double a1_7, double* p0, double* p1)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+-1.6666666666666671e-01*a1_2+-2.0000000000000001e-01*a1_3+-2.0000000000000001e-01*a1_4+-1.9047619047619049e-01*a1_5+-1.7857142857142860e-01*a1_6+-1.6666666666666671e-01*a1_7;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+9.0000000000000002e-01*a1_3+8.0000000000000004e-01*a1_4+7.1428571428571430e-01*a1_5+6.4285714285714290e-01*a1_6+5.8333333333333326e-01*a1_7;
}
            
double iocbio_ipwf_find_real_zero_in_01_2(double a_0, double a_1, double a_2, int direction, double *slope)
{
  
/* Code translated from http://www.netlib.org/toms/493, subroutine QUAD, with modifications. */
#define ABS(X) ((X)<0.0?-(X):(X))
double b, e, d, lr, sr, ls, ss;
//printf("a_0,a_1,a_2, e=%f, %f, %f\n", a_0, a_1, a_2);
if (a_2==0.0)
  {
    if (a_1!=0.0)
      {
        *slope = a_1;
        sr = -a_0/a_1;
        return FIXZERO(sr);
      }
    return -1.0;
  }
else
  {
    if (a_0==0.0)
      {
        *slope = a_1;
        return 0.0;
      }
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
      lr = FIXZERO(lr);
      if (lr==0.0)
        {
          *slope = a_1;
          return 0.0;
        }
      sr = (a_0/lr)/a_2;
      sr = FIXZERO(sr);
      ls = a_1+2.0*a_2*lr;
      ss = a_1+2.0*a_2*sr;
      //printf("p(lr=%f)=%f\n", lr,a_0+lr*(a_1+lr*a_2));
      //printf("p(sr=%f)=%f\n", sr,a_0+sr*(a_1+sr*a_2));
      if (lr>=0.0 && lr<=1.0 && (sr<0.0 || direction <0))
        { *slope = ls; return lr; }
      else
        { *slope = ss; return sr; }
    }
  }

  *slope = 0.0;
  return -1.0;
}
            

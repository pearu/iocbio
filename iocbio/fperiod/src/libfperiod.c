/*
  Source code of libfperiod software library.

  This file is part of the IOCBio project but can be used as a
  standalone software program. We ask to acknowledge the use of the
  software in scientific articles by citing the following paper:

    Pearu Peterson, Mari Kalda, Marko Vendelin.
    Real-time Determination of Sarcomere Length of a Single Cardiomyocyte during Contraction.
    <Journal information to be updated>.

  See http://iocbio.googlecode.com/ for more information.

  License: BSD, see the footer of this file.
  Author: Pearu Peterson
  Created: November 2011
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "libfperiod.h"
#define MAX(X, Y) ((X)<(Y)?(Y):(X))
#define MIN(X, Y) ((X)>(Y)?(Y):(X))
#define IFLOOR(X) ((int)floor(X))
void iocbio_objective(double *y, int k, double *f, int n, int m, int order, int method, double *r)
{
  int j;
#ifndef ENABLE_METHOD
  double (*evaluate)(double, double*, int, int, int) = iocbio_ipwf_e11_evaluate;
#else
  double (*evaluate)(double, double*, int, int, int) = NULL;
  switch (method)
    {
    case 0:
      evaluate = iocbio_ipwf_e11_evaluate;
      break;
    case 1:
      evaluate = iocbio_ipwf_a11_evaluate;
      break;
    case 2:
      evaluate = iocbio_ipwf_ep11_evaluate;
      break;
    case 3:
      evaluate = iocbio_ipwf_ap11_evaluate;
      break;
    case 5:
      evaluate = iocbio_ipwf_a00_evaluate;
      break;
    default:
      printf("iocbio_objective: method value not in [0, 1, 2, 3, 5], got %d\n", method);
      return;
    }
#endif
  for (j=0; j<k; ++j)
    r[j] = evaluate(y[j], f, n, m, order);
}
/**
  Estimate fundamental period of sequences.

  Parameters
  ----------
  f : double*
    Specify pointer to the beggining of the first sequence.
  n : int
    Specify the length of a sequence.
  m : int
    Specify the number of sequences stored in f using row-storage order.
  initial_period: double
    Specify estimate of the fundamental period. Set initial_period=0.0
    if no estimate.
  detrend : {0,1}
    When true then detrend sequences before finding the fundamental period.
  method : {0,1}
    Specify objective function. [NOT AVAILABLE IN LIBFPERIOD] 
       0: F(f)=int_0^{L-y} (f(x+y)-f(x))^2 dx
       1: F(f)=-int_0^{L-y} f(x+y)*f(x) dx

  Returns
  -------
  fperiod : double
    If fperiod>0 then its value is the fundamental period (see also
    note about initial_period).
    If fperiod==-1.0 then invalid method parameter was specifies.
    If fperiod==-2.0 then no fundamental period could be determined
    because no non-zero minimum point of the objective function
    exists.

 */
double iocbio_fperiod(double *f, int n, int m, double initial_period, int detrend, int method)
{
  double* cache = NULL;
  double fperiod;
  if (detrend)
    {
      cache = (double*)malloc(sizeof(double)*n*m);
      if (cache==NULL)
	{
	  printf("iocbio_fperiod: memory allocation error\n");
	  return 0.0;
	}
    }
  fperiod = iocbio_fperiod_cached(f, n, m, initial_period, detrend, method, cache);
  if (detrend && cache != NULL)
    free(cache);
  return fperiod;
}
/**

   iocbio_fperiod_cached is same as iocbio_fperiod but with extra
   cache argument. The size of the cache must be n*m when detrend!=0,
   otherwise cache is not referenced.

*/
double iocbio_fperiod_cached(double *f, int n, int m, double initial_period, int detrend, int method, double *cache)
{
  /* This function returns the second (counted from the origin,
     inclusive) minimum point of objective function that defines the
     fundamental period of f.
   */
  int start_j = IFLOOR(initial_period);
  int end_j = start_j + 1;
  double extreme = 0.0;
  double slope = 0.0;
  double convexity = 0.0;
  double* f2 = (detrend?cache:f);
  int status;
#ifndef ENABLE_METHOD
  int (*find_zero)(int, int, double*, int, int, int, double*, double*) = iocbio_ipwf_e11_find_zero;
  double (*evaluate)(double, double*, int, int, int) = iocbio_ipwf_e11_evaluate;
#else
  int (*find_zero)(int, int, double*, int, int, int, double*, double*) = NULL;
  double (*evaluate)(double, double*, int, int, int) = NULL;

  switch (method)
    {
    case 0:
      find_zero = iocbio_ipwf_e11_find_zero;
      evaluate = iocbio_ipwf_e11_evaluate;
      break;
    case 1:
      find_zero = iocbio_ipwf_a11_find_zero;
      evaluate = iocbio_ipwf_a11_evaluate;
      break;
    case 2:
      find_zero = iocbio_ipwf_ep11_find_zero;
      evaluate = iocbio_ipwf_ep11_evaluate;
      break;
    case 3:
      find_zero = iocbio_ipwf_ap11_find_zero;
      evaluate = iocbio_ipwf_ap11_evaluate;
      break;
    case 5:
      find_zero = iocbio_ipwf_a00_find_zero;
      evaluate = iocbio_ipwf_a00_evaluate;
      break;
    default:
      printf("iocbio_fperiod_cached: method value not in [0, 1, 2, 3, 5], got %d\n", method);
      return -1.0;
    }
  //printf("iocbio_fperiod_cached[%d](n=%d, m=%d, initial_period=%f, detrend=%d, method=%d)\n", iocbio_fperiod_cached_call_level, n, m, initial_period, detrend, method);
#endif
  if (detrend)
    iocbio_detrend(f, n, m, initial_period, cache);
  start_j = MAX(0, start_j);
  end_j = MIN(n-1, end_j);
  if (start_j==0)
    { /* Initial period was not specified. Doing full search for
	 finding the first non-zero minimum point of the objective
	 function. Direction of search is to the right.
       */
      status = find_zero(1, n, f2, n, m, 1, &extreme, &convexity);
      if (status==0)
	{
	  if (convexity>0) /* extreme is minimum point. */
	    return extreme;
	  if (convexity<0) /* extreme is maximum point, finding the next extreme.. */
	    {
	      /* First try if the extreme is in the same interval by approaching it from right */
	      status = find_zero(IFLOOR(extreme)+1, IFLOOR(extreme), f2, n, m, 1, &extreme, &convexity);
	      if (status==0 && convexity>0) /* extreme is minimum point */
		return extreme;
	      /* Ok, continuing the search to the right */
	      status = find_zero(IFLOOR(extreme)+1, n, f2, n, m, 1, &extreme, &convexity);
	      if (status==0 && convexity>0) /* extreme is minimum point */
		return extreme;
	    }
	}
      //printf("iocbio_fperiod_cached: status=%d, initial_period=%f, extreme=%f(%f), convexity=%f\n", status, initial_period, extreme, IFLOOR(extreme), convexity);
      //printf("iocbio_fperiod_cached: objective function has no non-zero minimum.\n");
      return -2.0;
    }
  /* When initial_period is given, it is assumed to be in between the first
     to maximum points of the objective function. If it is to the right of the
     second maximum point then the returned result might not correspond
     to the fundamental period of f. If it is to the left of the first maximum
     point then the fundamental period will be determined via full search. */
  /* For efficiency, first assume floor(fperiod) == floor(initial_period) */
  status = find_zero(start_j, end_j, f2, n, m, 1, &extreme, &convexity);
  if (status==0)
    {
      if (convexity>0.0)  /* extreme is minimum point */
	return extreme;
      if (convexity<0.0) /* extreme is maximum point */
	{
	  /* First try if the extreme is in the same interval by approaching it from right */
	  status = find_zero(IFLOOR(extreme)+1, IFLOOR(extreme), f2, n, m, 1, &extreme, &convexity);
	  if (status==0 && convexity>0) /* extreme is minimum point */
	    return extreme;
	}
    }
  /* Decide in which direction to search for the minimum point */
  slope = evaluate(initial_period, f2, n, m, 1);
  if (slope<0.0) /* the minimum point is to the right of floor(initial_period)+1 */
    {
      status = find_zero(end_j, n, f2, n, m, 1, &extreme, &convexity);
      if (status==0 && convexity>0.0)  /* extreme is minimum point */
	return extreme;
      if (status!=0) /* no zero point found, looking to the left. */
	{
	  status = find_zero(start_j, 1, f2, n, m, 1, &extreme, &convexity);
	  if (status==0 && convexity>0.0) /* extreme is minimum point */
	    return extreme;
	  if (status==0 && convexity<0.0) /* skipping maximum point */
	    {
	      /* Check the case where minimum and maximum (extreme found from right) is in the same interval */
	      status = find_zero(IFLOOR(extreme), IFLOOR(extreme)+1, f2, n, m, 1, &extreme, &convexity);
	      if (status==0 && convexity>0.0)
		return extreme;
	      /* Continue the search to the left */
	      status = find_zero(IFLOOR(extreme), 1, f2, n, m, 1, &extreme, &convexity);
	      if (status==0 && convexity>0.0)
		return extreme;
	    }
	}
    }
  else if (slope>0.0) /* the minimum point is to the left of floor(initial_period) */
    {
      status = find_zero(start_j, 0, f2, n, m, 1, &extreme, &convexity);
      if (status==0 && convexity>0.0) /* extreme is minimum point */
	{
	  if (IFLOOR(extreme)!=0) /* note that the interval [0,1] cannot contain two minimum points */
	    return extreme;
	  /* Doing full scan */
	  return iocbio_fperiod_cached(f2, n, m, 0.0, 0, method, cache);
	}
    }
  //printf("iocbio_fperiod_cached: status=%d, initial_period=%f, extreme=%f, slope=%f, convexity=%f\n", status, initial_period, extreme, slope, convexity);
  //printf("iocbio_fperiod_cached: objective function has no non-zero minimum.\n");
  return -2.0;
}
#define UPDATE_DETREND1_ARRAY(GT, FORCE, N) \
      if (FORCE || prev_extreme_point GT 0)\
	{\
	  average_point = 0.5*(extreme_point+prev_extreme_point);\
	  average_value = 0.5*(extreme_value+prev_extreme_value);\
	  if (prev_average_point GT 0)\
	    {\
	      slope = (average_value-prev_average_value)/(average_point-prev_average_point);\
	      while (k<N)\
		{\
		  r[rstride*(k)] = f[fstride*(k)] - (prev_average_value + (k-prev_average_point)*slope);\
		  k++;\
		}\
	    }\
	  prev_average_point = average_point;\
	  prev_average_value = average_value;\
	}
#define ISMAX(index) (f[fstride*((index)-(1+smoothness))]<=f[fstride*((index)+smoothness)]) && (f[fstride*((index)-smoothness)]>=f[fstride*((index)+(1+smoothness))])
#define ISMIN(index) (f[fstride*((index)-(1+smoothness))]>=f[fstride*((index)+smoothness)]) && (f[fstride*((index)-smoothness)]<=f[fstride*((index)+(1+smoothness))])
#define ISEXTREME(index) (ISMIN(index)?-1:(ISMAX(index)?1:0))
void iocbio_detrend(double *f, int n, int m, double period, double *r)
{
  int j;
  for(j=0; j<m; ++j)
    iocbio_detrend1(f+j*n, n, 1, period, r+j*n, 1);
}
void iocbio_detrend1(double *f, int n, int fstride, double period, double *r, int rstride)
{
  int smoothness = 0;
  double prev_extreme_point = -1;
  double prev_average_point = -1;
  double prev_extreme_value = 0;
  double prev_average_value = 0;
  double extreme_point = 0;
  double average_point = -1;
  double extreme_value = f[0];
  double average_value = 0;
  double v, slope;
  int k = 0;
  int flag=0, new_flag;
  int count=0;
  int i,j;
  if (period>0)
    smoothness = 0.25*period-0.5;
  if (smoothness<0)
    smoothness = 0;
  for (i=1+smoothness;i<n-(1+smoothness); ++i)
    {
      new_flag = ISEXTREME(i);
      if (!new_flag)
	continue;
      if (flag==new_flag)
	count++;
      else
	count = 1;
      flag = new_flag;

      if (smoothness)
	{
	  v = 0.0;
	  for (j=-smoothness;j<1+smoothness; ++j)
	    v += f[fstride*(i+j)];
	  v /= (1+2*smoothness);
	}
      else
	v = f[fstride*i];
      if (count>1)
	{
	  extreme_point = (extreme_point*(count-1)+i)/count;
	  extreme_value = (extreme_value*(count-1)+v)/count;
	  continue;
	}

      UPDATE_DETREND1_ARRAY(>=, 0, average_point);

      prev_extreme_point = extreme_point;
      prev_extreme_value = extreme_value;
      extreme_point = i;
      extreme_value = v;
    }

  UPDATE_DETREND1_ARRAY(>, 0, average_point);

  prev_extreme_point = extreme_point;
  prev_extreme_value = extreme_value;

  extreme_point = n-1;
  extreme_value = f[fstride*(n-1)];

  UPDATE_DETREND1_ARRAY(>=, 1, n);

  while (k<n) // f is constant
    {
      r[rstride*k] = f[fstride*(k)];
      k++;
    }
}
#define EPSPOS 2.2204460492503131e-16
#define EPSNEG 1.1102230246251565e-16
#define FIXZERO(X) ((-EPSNEG<(X)) && ((X)<EPSPOS)?0.0:(X))
#define FRAC_1_3 3.333333333333333e-1
void iocbio_ipwf_e11_compute_coeffs_diff0(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

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
        b0 += FRAC_1_3*(f_i*(f_i - 2*f_ipj) + (f_ip1 + f_i - f_ipj)*f_ip1 + (f_ipj*f_ipj) + (f_ip1pj + f_ipj - f_i - 2*f_ip1)*f_ip1pj);
        b1 += (f_ip1pj - f_ip1 - f_i)*f_ip1pj + f_ipj*(f_ip1 + f_i - f_ipj);
        b2 += (2*f_ip1 - f_ipj + f_ip2pj - f_ip1pj)*f_ip1pj + (f_ipj*f_ipj) + (-f_ip2pj - f_ipj)*f_ip1;
        b3 += FRAC_1_3*((2*f_i + 2*f_ipj - 2*f_ip1 - 2*f_ip2pj)*f_ip1pj + f_ip2pj*(f_ip1 + f_ip2pj - f_i) + f_ipj*(f_ip1 - f_i - f_ipj));
      }
      b0 += FRAC_1_3*(f_m1pn*(f_m1pn - 2*f_m1mjpn) + f_m2mjpn*(f_m1mjpn + f_m2mjpn - f_m1pn - 2*f_m2pn) + (f_m1mjpn*f_m1mjpn) + f_m2pn*(f_m2pn + f_m1pn - f_m1mjpn));
      b1 += -f_m2mjpn*f_m1pn + f_m2pn*(f_m2mjpn - f_m2pn) + f_m1mjpn*(f_m1pn + f_m2pn - f_m1mjpn);
      b2 += f_m2pn*(f_m2pn - f_m1pn) + f_m2mjpn*f_m1pn + f_m1mjpn*(f_m1mjpn - f_m2pn - f_m2mjpn);
      b3 += FRAC_1_3*(f_m2pn*(2*f_m1pn + f_m1mjpn - f_m2pn) - (f_m1mjpn*f_m1mjpn) + f_m1pn*(-f_m1mjpn - f_m1pn) + f_m2mjpn*(2*f_m1mjpn + f_m1pn - f_m2mjpn - f_m2pn));
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
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
void iocbio_ipwf_e11_compute_coeffs_diff1(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

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
        b0 += (f_ip1pj - f_ip1 - f_i)*f_ip1pj + f_ipj*(f_ip1 + f_i - f_ipj);
        b1 += 2*((f_ipj*f_ipj) + (-f_ip2pj - f_ipj)*f_ip1) + (4*f_ip1 + 2*(-f_ipj + f_ip2pj - f_ip1pj))*f_ip1pj;
        b2 += 2*(f_i + f_ipj - f_ip1 - f_ip2pj)*f_ip1pj + f_ip2pj*(f_ip1 + f_ip2pj - f_i) + f_ipj*(f_ip1 - f_i - f_ipj);
      }
      b0 += -f_m2mjpn*f_m1pn + f_m2pn*(f_m2mjpn + f_m1mjpn - f_m2pn) + f_m1mjpn*(f_m1pn - f_m1mjpn);
      b1 += 2*(f_m1mjpn*(f_m1mjpn - f_m2mjpn) + f_m2pn*(f_m2pn - f_m1mjpn - f_m1pn) + f_m1pn*f_m2mjpn);
      b2 += f_m1mjpn*(2*f_m2mjpn - f_m1mjpn + f_m2pn - f_m1pn) + f_m2pn*(-f_m2pn - f_m2mjpn) - (f_m2mjpn*f_m2mjpn) + f_m1pn*(2*f_m2pn - f_m1pn + f_m2mjpn);
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
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
void iocbio_ipwf_e11_compute_coeffs_diff2(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

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
        b0 += 2*((f_ipj*f_ipj) + (-f_ip2pj - f_ipj)*f_ip1) + (4*f_ip1 + 2*(-f_ipj + f_ip2pj - f_ip1pj))*f_ip1pj;
        b1 += 2*(f_ip2pj*(f_ip1 + f_ip2pj - f_i) + f_ipj*(f_ip1 - f_i - f_ipj)) + 4*(f_i + f_ipj - f_ip1 - f_ip2pj)*f_ip1pj;
      }
      b0 += 2*(f_m2pn*(f_m2pn - f_m1pn) + f_m2mjpn*f_m1pn + f_m1mjpn*(f_m1mjpn - f_m2pn - f_m2mjpn));
      b1 += f_m2mjpn*(4*f_m1mjpn + 2*(-f_m2pn + f_m1pn - f_m2mjpn)) + f_m2pn*(4*f_m1pn + 2*(-f_m2pn + f_m1mjpn)) + 2*(f_m1pn*(-f_m1mjpn - f_m1pn) - (f_m1mjpn*f_m1mjpn));
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
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
void iocbio_ipwf_e11_compute_coeffs_diff3(int j, double *fm, int n, int m, double* a0, double* a1, double* a2, double* a3)
{
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

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
        b0 += 2*(f_ip2pj*(f_ip1 + f_ip2pj - f_i) + f_ipj*(f_ip1 - f_i - f_ipj)) + 4*(f_i + f_ipj - f_ip1 - f_ip2pj)*f_ip1pj;
      }
      b0 += f_m1mjpn*(4*f_m2mjpn + 2*(-f_m1pn + f_m2pn - f_m1mjpn)) + f_m2pn*(4*f_m1pn + 2*(-f_m2pn - f_m2mjpn)) + 2*(f_m1pn*(f_m2mjpn - f_m1pn) - (f_m2mjpn*f_m2mjpn));
    }
  }
  *a0 = b0;
  *a1 = b1;
  *a2 = b2;
  *a3 = b3;
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
void iocbio_ipwf_linear_approximation_3_0(double a1_0, double a2_0, double a3_0, double* p0, double* p1)
{
  *p0 = 5.5555555555555558e-01*a1_0+3.3333333333333331e-01*a2_0+1.1111111111111110e-01*a3_0;
  *p1 = -4.4444444444444442e-01*a1_0+0.0000000000000000e+00*a2_0+4.4444444444444442e-01*a3_0;
}
void iocbio_ipwf_linear_approximation_1_1(double a1_0, double a1_1, double* p0, double* p1)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1;
}
void iocbio_ipwf_linear_approximation_1_3(double a1_0, double a1_1, double a1_2, double a1_3, double* p0, double* p1)
{
  *p0 = 1.0000000000000000e+00*a1_0+0.0000000000000000e+00*a1_1+-1.6666666666666671e-01*a1_2+-2.0000000000000001e-01*a1_3;
  *p1 = 0.0000000000000000e+00*a1_0+1.0000000000000000e+00*a1_1+1.0000000000000000e+00*a1_2+9.0000000000000002e-01*a1_3;
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
/*
Copyright (c) 2011, Pearu Peterson
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
  This file provides the following functions:

    fperiod_compute_period - estimate the fundamental period of a sequence, high-level interface

    fperiod_find_acf_maximum - find maximum of ACF of a sequence
    fperiod_subtract_average - return fluctuations of a sequence
    fperiod_subtract_average1 - return fluctuations of a sequence
    fperiod_acf - evaluate ACF of a sequence

  Author: Pearu Peterson
  Created: March 2011
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "fperiod.h"

static void compute_coeffs(int j, double* f, int n, int m, double* a, double *b, double*c, double *d);
static void compute_coeffs_1(int j, double* f, int n, int m, double* c);
static void compute_coeffs_2(int j, double* f, int n, int m, double *b);
static void compute_coeffs_32(int j, double* f, int n, int m, double* a, double *b);
static void compute_coeffs_320(int j, double* f, int n, int m, double* a, double *b, double *d);
static void fperiod_power(double* f, int n, int m, double exp);

#define EPSPOS 2.2204460492503131e-16
#define EPSNEG 1.1102230246251565e-16
#define FLOATMIN -1.7976931348623157e+308
#define FLOATMAX 1.7976931348623157e+308

#define ENABLE_ACF_SUPERPOSITION

double fperiod_compute_period(double* f, int n, int m, double structure_size, double exp, int method)
{
  double period;
#ifdef ENABLE_ACF_SUPERPOSITION
  double *r = (double*)malloc(sizeof(double)*(m*n));
#else
  double *r = (double*)malloc(sizeof(double)*(n));
#endif
  period = fperiod_compute_period_cache(f, n, m, structure_size, exp, method, r);
  free(r);
  return period;
}

double fperiod_compute_period_cache(double* f, int n, int m, double structure_size, double exp, int method, double *r)
{
  int lbound, ubound, smoothness;
  int j, k;
  double period = 0.0, p;
  if (r==0) return -3.0;
  if (structure_size>0)
    {
      lbound = 0.5*structure_size;
      ubound = 1.5*structure_size;
      smoothness = structure_size/10.0;
    }
  else
    {
      lbound = 1;
      ubound = n-1;
      smoothness = 0;
    }
#ifdef ENABLE_ACF_SUPERPOSITION
  fperiod_subtract_average(f, n, m, smoothness, r);
  //fperiod_subtract_average(r, n, m, smoothness, f);
  fperiod_power(r, n, m, exp);
  switch (method)
    {
    case 0: period = fperiod_find_acf_maximum(r, n, m, lbound, ubound); break;
    case 1: period = fperiod_find_acf_d2_minimum(r, n, m, lbound, ubound); break;
    }
#else
  k = 1;
  for(j=0; j<m; ++j)
    {
      fperiod_subtract_average1(f+j*n, n, 1, smoothness, r, 1);
      p = fperiod_find_acf_maximum(r, n, 1, lbound, ubound);
      if (p>0)
	{
	  //printf("p=%f\n",p);
	  period = (period*(k-1)+p)/k;
	  k ++;
	}
    }
#endif
  return period;


  //fperiod_subtract_average(r, n, m, smoothness, f);

}

/*
Parameters
----------

  f - an array of n*m values
  n - length of a sample sequence
  m - number of different sample sequences
  lbound - lower bound of a range containing fundamental period, must be greater than 0
  ubound - upper bound of a range containing fundamental period, must be less than n-1

Returns
-------
  y - fundamental period of a sequence; if no period were found, y will be 0;
      if argument values are invalid, y will be negative with the following
      meanings:

      y = -1   --- values of lbound or ubound are invalid
      y = -2   --- values of m or n are invalid
 */
#define ISZERO(r) ((r)<largest_non_positive && (r)>smallest_non_negative)
#define ISNONPOS(r) ((r)<largest_non_positive)
#define ISNONNEG(r) ((r)>smallest_non_negative)
#define ISLTONE(r) ISNONNEG(1-(r))

double fperiod_find_acf_maximum(double* f, int n, int m, int lbound, int ubound)
{
  int j, j_max_v = 0;
  double y = 0.0, y2 = 0.0;
  double max_v = FLOATMIN;
  double a, b, c, d, c1, d1, r, v;
  double smallest_non_negative = -(n*m*EPSNEG);
  double largest_non_positive = (n*m*EPSPOS);

  if (lbound<=0 || lbound>=ubound || ubound>n-1)
    return -1;
  if (n<3 || m<0)
    return -2;

  //compute_coeffs_1(lbound, f, n, m, &c1);
  for (j=lbound+1;j<ubound;++j)
    {
      compute_coeffs(j, f, n, m, &a, &b, &c, &d);
      if (ISZERO(a))
	{
	  if (ISZERO(b))
	    continue;
	  r = -0.5*c/b;
	}
      else
	{
	  d1 = b*b-a*c;      

	  if (ISZERO(d1))
	    r = -b/a;
	  else if (d1>0)
	    /* note that r = (-b+sqrt(d1))/a would violate the negativity of second derivative */
	    r = -(b+sqrt(d1))/a;
	  else
	    {
	      continue;
	    }
	}
      if (ISNONNEG(r) && ISLTONE(r))
	{
	  v = ((a/3.0*r + b)*r+c)*r + d/3.0;
	  if (v>max_v)
	    {
	      max_v = v;
	      y = (double)j + r;
	      j_max_v = j;
	    }
	}
    }

  if (y==0.0)
    {
      printf("fperiod_find_acf_maximum: no acf maximum found within range [%d, %d]\n", lbound, ubound);
    }

  return y;
}

double fperiod_find_acf_d2_minimum(double* f, int n, int m, int lbound, int ubound)
{
  int j, j_min_b = 0, j_max_v = 0;
  double y = 0.0, y2 = 0.0;
  double max_d = FLOATMIN;
  double max_v = FLOATMIN;
  double min_b = FLOATMAX;
  double a, b, c, d, c1, d1, r, v, a1, b1;
  double smallest_non_negative = -(n*m*EPSNEG);
  double largest_non_positive = (n*m*EPSPOS);
  double vm1, v0, v1;

  if (lbound<=0 || lbound>=ubound || ubound>n-1)
    return -1;
  if (n<3 || m<0)
    return -2;

  compute_coeffs_32(lbound, f, n, m, &a1, &b1);
  for (j=lbound+1;j<ubound;++j)
    {
      a = a1;
      b = b1;
      compute_coeffs_32(j, f, n, m, &a1, &b1);
      if (!(ISNONNEG(a1) && ISNONPOS(a)))
	  continue;

      //TODO: catch 0-division
      vm1 = b;
      v0 = b1;
      v1 = b1 + a1;
      r = -0.5*(v1-vm1)/(vm1+v1-2.0*v0);
      //r = -0.5*(2.0*(b1+a1)-b)/(2.0*a1+b);

      if (ISLTONE(-r) && ISLTONE(r))
	{
	  //TODO: compute minimum of parabola
	  if (b1<min_b)
	    {
	      y = (double)j + r;
	      min_b = b1;	     
	    }
	}
      else
	printf("fperiod_find_acf_d2_minimum: j,r=%d, %f, a, a1, b, b1=%f, %f, %f, %f\n", j, r, a, a1, b, b1);
    }

  if (y==0.0)
    {
      printf("fperiod_find_acf_d2_minimum: no acf_d2 minimum found within range [%d, %d]\n", lbound, ubound);
    }

  return y;
}

/*
Parameters
----------
  y - argument to acf, must be nonnegative and less than n-1
  f - an array of n*m values
  n - length of a sample sequence
  m - number of sample sequences

Returns
-------
  acf - acf value at y
 */
double fperiod_acf(double y, double* f, int n, int m)
{
  int j=floor(y);
  double r = y-j;
  double a=0, b=0, c=0, d=0;
  compute_coeffs(j, f, n, m, &a, &b, &c, &d);
  //compute_coeffs_1(j, f, n, m, &c);
  //compute_coeffs_320(j, f, n, m, &a, &b, &d);
  return ((a/3.0*r + b)*r+c)*r + d/3.0;
}

double fperiod_acf_d1(double y, double* f, int n, int m)
{
  int j=floor(y);
  double r = y-j;
  double a=0, b=0, c=0, d=0;
  compute_coeffs(j, f, n, m, &a, &b, &c, &d);
  return (a*r + 2*b)*r+c;
}

double fperiod_acf_d2(double y, double* f, int n, int m)
{
  int j=floor(y);
  double r = y-j;
  double a=0, b=0, c=0, d=0;
  compute_coeffs(j, f, n, m, &a, &b, &c, &d);
  return 2*(a*r + b);
}

static
void compute_coeffs_1(int j, double* f, int n, int m, double* c)
{
  int i,p,o;
  int k = n-2-j;
  double c1 = 0.0;
  double f0;

  if (k>=0)
    {
      for (p=0;p<m;++p)
	{
	  o = p*n;
	  f0 = f[o];

	  for (i=1; i<k; ++i)
	    {
	      c1 += (f[o+i]-f0)*(f[o+i+j+1]-f[o+i+j-1]);
	      if (isnan(c1))
		{
		  printf("compute_coeffs_1: c1=%f, p,i,j,k,o=%d,%d,%d,%d,%d:::%f, %f\n",c1, p,i,j,k,o, (f[o+i]-f0), (f[o+i+j+1]-f[o+i+j-1]));
		  break;
		}
	    }
	  if (isnan(c1))
	    break;
	  c1 += (f[o+k+1]-f0)*(-f[o+k+j+1]-f[o+k+j]+f[o+k+1]+f0);
	  c1 += (f[o+k]-f0)*(f[o+k+j+1]-f[o+k+j-1]);

	}
    }
  if (isnan(c1))
    {
      printf("compute_coeffs_1: c1=%f\n",c1);
    }
  *c = c1;
}

static
void compute_coeffs_320(int j, double* f, int n, int m, double* a, double *b, double *d)
{
  int i,p,o;
  int k = n-2-j;
  double a1=0, b1=0, d1=0;
  double f0;
  double fi;
  if (k>=0)
    {
      for (p=0;p<m;++p)
	{
	  o = p*n;
	  f0 = f[o];
	  for (i=1; i<k; ++i)
	    {
	      fi = f[o+i]-f0;
	      a1 += fi*(f[o+i+j+2]-3*f[o+i+j+1]+3*f[o+i+j]-f[o+i+j-1]);
	      b1 += fi*(f[o+i+j+1]-2*f[o+i+j]+f[o+i+j-1]);

	      d1 += fi*(f[o+i+j+1]+4*f[o+i+j]+f[o+i+j-1]-2*f[o+i]-f[o+i+1]-3*f0);
	    }

	  a1 += (f[o+k]-f0)*(f[o+k]-2*f[o+k+1]-2*f[o+k+j+1]+3*f[o+k+j]-f[o+k+j-1]+f0);
	  a1 += (f[o+k+1]-f0)*(f[o+k+j+1]-f[o+k+j]+f[o+k+1]-f0);
	  
	  b1 += (f[o+k+1]-f0)*(f[o+k+j]-f[o+k+1]);
	  b1 += (f[o+k]-f0)*(f[o+k+1]-2*f[o+k+j]+f[o+k+j-1]);


	  
	  d1 += (f[o+k]-f0)*(-2*f[o+k]-f[o+k+1]+f[o+k+j+1]+4*f[o+k+j]+f[o+k+j-1]-3*f0);
	  d1 += (f[o+k+1]-f0)*(2*f[o+k+j+1]+f[o+k+j]-f[o+k+1]-2*f0);
	}
    }
  *a = a1;
  *b = b1;
  *d = d1;
}

static
void compute_coeffs_32(int j, double* f, int n, int m, double* a, double *b)
{
  int i,p,o;
  int k = n-2-j;
  double a1=0, b1=0;
  double f0;
  double fi;
  if (k>=0)
    {
      for (p=0;p<m;++p)
	{
	  o = p*n;
	  f0 = f[o];
	  for (i=1; i<k; ++i)
	    {
	      fi = f[o+i]-f0;
	      a1 += fi*(f[o+i+j+2]-3*f[o+i+j+1]+3*f[o+i+j]-f[o+i+j-1]);
	      b1 += fi*(f[o+i+j+1]-2*f[o+i+j]+f[o+i+j-1]);
	    }

	  a1 += (f[o+k]-f0)*(f[o+k]-2*f[o+k+1]-2*f[o+k+j+1]+3*f[o+k+j]-f[o+k+j-1]+f0);
	  a1 += (f[o+k+1]-f0)*(f[o+k+j+1]-f[o+k+j]+f[o+k+1]-f0);
	  
	  b1 += (f[o+k+1]-f0)*(f[o+k+j]-f[o+k+1]);
	  b1 += (f[o+k]-f0)*(f[o+k+1]-2*f[o+k+j]+f[o+k+j-1]);
	}
    }
  *a = a1;
  *b = b1;
}

static
void compute_coeffs_2(int j, double* f, int n, int m, double *b)
{
  int i,p,o;
  int k = n-2-j;
  double b1=0;
  double f0;
  double fi;
  if (k>=0)
    {
      for (p=0;p<m;++p)
	{
	  o = p*n;
	  f0 = f[o];
	  for (i=1; i<k; ++i)
	    {
	      fi = f[o+i]-f0;
	      b1 += fi*(f[o+i+j+1]-2*f[o+i+j]+f[o+i+j-1]);
	    }	  
	  b1 += (f[o+k+1]-f0)*(f[o+k+j]-f[o+k+1]);
	  b1 += (f[o+k]-f0)*(f[o+k+1]-2*f[o+k+j]+f[o+k+j-1]);
	}
    }
  *b = b1;
}

static
void compute_coeffs(int j, double* f, int n, int m, double* a, double *b, double*c, double *d)
{
  int i,p,o;
  int k = n-2-j;
  double a1=0, b1=0, c1=0, d1=0;
  double f0;
  double fi;
  if (k>=0)
    {
      for (p=0;p<m;++p)
	{
	  o = p*n;
	  f0 = f[o];
	  for (i=1; i<k; ++i)
	    {
	      fi = f[o+i]-f0;
	      a1 += fi*(f[o+i+j+2]-3*f[o+i+j+1]+3*f[o+i+j]-f[o+i+j-1]);
	      b1 += fi*(f[o+i+j+1]-2*f[o+i+j]+f[o+i+j-1]);
	      c1 += (f[o+i]-f0)*(f[o+i+j+1]-f[o+i+j-1]);
	      d1 += fi*(f[o+i+j+1]+4*f[o+i+j]+f[o+i+j-1]-2*f[o+i]-f[o+i+1]-3*f0);
	    }

	  a1 += (f[o+k]-f0)*(f[o+k]-2*f[o+k+1]-2*f[o+k+j+1]+3*f[o+k+j]-f[o+k+j-1]+f0);
	  a1 += (f[o+k+1]-f0)*(f[o+k+j+1]-f[o+k+j]+f[o+k+1]-f0);
	  
	  b1 += (f[o+k+1]-f0)*(f[o+k+j]-f[o+k+1]);
	  b1 += (f[o+k]-f0)*(f[o+k+1]-2*f[o+k+j]+f[o+k+j-1]);

	  c1 += (f[o+k+1]-f0)*(-f[o+k+j+1]-f[o+k+j]+f[o+k+1]+f0);
	  c1 += (f[o+k]-f0)*(f[o+k+j+1]-f[o+k+j-1]);
	  
	  d1 += (f[o+k]-f0)*(-2*f[o+k]-f[o+k+1]+f[o+k+j+1]+4*f[o+k+j]+f[o+k+j-1]-3*f0);
	  d1 += (f[o+k+1]-f0)*(2*f[o+k+j+1]+f[o+k+j]-f[o+k+1]-2*f0);
	}
    }
  *a = a1;
  *b = b1;
  *c = c1;
  *d = d1;
}

/*
Parameters
----------
  f - an array of n*m values
  n - length of a sample sequence
  m - number of sample sequences
  smoothness - smoothing parameter, it is the half-width of constant
    smoothing kernel. Suggested value is floor(s/10) where s is the
    minimal width of structures

Returns
-------
  r - an array of n*m values containing line by line fluctuations
 */
void fperiod_subtract_average(double* f, int n, int m, int structure_size, double* r)
{
  int j;
  int smoothness;
  if (structure_size>0)
    smoothness = ((float)structure_size/2.0-1.0)/2.0;
  else
    smoothness = 0;
  for(j=0; j<m; ++j)
    fperiod_subtract_average1(f+j*n, n, 1, smoothness, r+j*n, 1);
}

void fperiod_power(double* f, int n, int m, double exp)
{
  int i, j;
  double v;
  if (exp==1.0)
    return;
  if (exp==0.0)
    {
      for(j=0; j<m; ++j)
	for(i=0; i<n; ++i)
	  {
	    v = f[j*n+i];
	    if (v<0)
	      f[j*n+i] = -1.0;
	    else if (v==0.0)
	      f[j*n+i] = 0.0;
	    else
	      f[j*n+i] = 1.0;
	  }
    }
  else
    {
      for(j=0; j<m; ++j)
	for(i=0; i<n; ++i)
	  {
	    v = f[j*n+i];
	    if (v<0)
	      f[j*n+i] = -pow(-v, exp);
	    else
	      f[j*n+i] = pow(v, exp);
	  }
    }
}

void fperiod_subtract_average_2d(double* f, int n, int m, int smoothness, double* r)
{
  int i, j;
  for(i=0; i<n; ++i)
    fperiod_subtract_average1(f+i, m, n, smoothness, r+i, n);
  for(j=0; j<m; ++j)
    for(i=0; i<n; ++i)
      f[j*n+i] = f[j*n+i] - r[j*n+i];
  for(j=0; j<m; ++j)
    fperiod_subtract_average1(f+j*n, n, 1, smoothness, r+j*n, 1);
  //memcpy(r, f, sizeof(double)*m*n);
}

void fperiod_subtract_average1(double* f, int n, int fstride, int smoothness, double* r, int rstride)
{
#define ISMAX(index) (f[fstride*((index)-(1+smoothness))]<=f[fstride*((index)+smoothness)]) && (f[fstride*((index)-smoothness)]>=f[fstride*((index)+(1+smoothness))])
#define ISMIN(index) (f[fstride*((index)-(1+smoothness))]>=f[fstride*((index)+smoothness)]) && (f[fstride*((index)-smoothness)]<=f[fstride*((index)+(1+smoothness))])
#define ISEXTREME(index) (ISMAX(index)?1:(ISMIN(index)?-1:0))
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
      
      v = 0.0;
      for (j=-smoothness;j<1+smoothness; ++j)
	v += f[fstride*(i+j)];
      v /= (1+2*smoothness);

      if (count>1)
	{
	  extreme_point = (extreme_point*(count-1)+i)/count;
	  extreme_value = (extreme_value*(count-1)+v)/count;
	  continue;
	}

      if (prev_extreme_point>=0)
	{
	  average_point = 0.5*(extreme_point+prev_extreme_point);
	  average_value = 0.5*(extreme_value+prev_extreme_value);
	  if (prev_average_point>=0)
	    {
	      slope = (average_value-prev_average_value)/(average_point-prev_average_point);
	      while (k<average_point)
		{
		  //if (average_point-prev_average_point>4)
		  //  r[rstride*k] = 0;
		  //else
		  r[rstride*(k)] = f[fstride*(k)] - (prev_average_value + (k-prev_average_point)*slope);
		  k++;
		}
	    }
	  prev_average_point = average_point;
	  prev_average_value = average_value;
	}
      prev_extreme_point = extreme_point;
      prev_extreme_value = extreme_value;
      extreme_point = i;
      extreme_value = v;
    }

  if (prev_extreme_point>0)
    {
      average_point = 0.5*(extreme_point+prev_extreme_point);
      average_value = 0.5*(extreme_value+prev_extreme_value);
      if (prev_average_point>0)
	{
	  slope = (average_value-prev_average_value)/(average_point-prev_average_point);
	  while (k<average_point)
	    {
	      r[rstride*k] = f[fstride*(k)] - (prev_average_value + (k-prev_average_point)*slope);
	      k++;
	    }
	}
      prev_average_point = average_point;
      prev_average_value = average_value;
    }

  prev_extreme_point = extreme_point;
  prev_extreme_value = extreme_value;

  extreme_point = n-1;
  extreme_value = f[fstride*(n-1)];

  average_point = 0.5*(extreme_point+prev_extreme_point);
  average_value = 0.5*(extreme_value+prev_extreme_value);

  if (prev_average_point>=0)
    {
      slope = (average_value-prev_average_value)/(average_point-prev_average_point);
      while (k<n)
	{
	  r[rstride*k] = f[fstride*(k)] - (prev_average_value + (k-prev_average_point)*slope);
	  k++;
	}
    }

  while (k<n) // f is constant
    {
      r[rstride*k] = f[fstride*(k)];
      k++;
    }

}

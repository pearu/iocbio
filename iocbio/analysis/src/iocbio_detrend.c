/* iocbio_detrend - data processing routines for finding trends.

  This file provides the following interface functions:

    iocbio_detrend - subtract trend values from an array
    iocbio_trend - find a trend in an array

  and computational/auxiliary routines:
    iocbio_detrend1 - subtract trend values from a sequence
    iocbio_compute_trend_spline_data - compute trend spline data, useful for illustrations

  Author: Pearu Peterson
  Created: October 2011
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "iocbio_detrend.h"

/* Auxiliary macros */
#define UPDATE_DETREND1_ARRAY(GT, FORCE, N)	\
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
#define UPDATE_TREND_SPLINE_DATA(GT, FORCE, N)	\
      if (FORCE || prev_extreme_point GT 0)\
	{\
	  average_point = 0.5*(extreme_point+prev_extreme_point);\
	  average_value = 0.5*(extreme_value+prev_extreme_value);\
	  average_positions[*nof_averages] = average_point;\
          average_values[*nof_averages] = average_value;\
          (*nof_averages) ++;				\
	  prev_average_point = average_point;\
	  prev_average_value = average_value;\
	}
#define ISMAX(index) (f[fstride*((index)-(1+smoothness))]<=f[fstride*((index)+smoothness)]) && (f[fstride*((index)-smoothness)]>=f[fstride*((index)+(1+smoothness))])
#define ISMIN(index) (f[fstride*((index)-(1+smoothness))]>=f[fstride*((index)+smoothness)]) && (f[fstride*((index)-smoothness)]<=f[fstride*((index)+(1+smoothness))])
#define ISEXTREME(index) (ISMIN(index)?-1:(ISMAX(index)?1:0))

/*
  iocbio_detrend function subtracts trend from an array of values
  assuming independent rows.

  Input parameters
  ----------------

  f : double*
    Specify pointer to an array using row-storage order.
  n, m : int
    Specify number of columns and rows of the array
  period : double
    Specify estimated period of detrended array. This is used
    for convolving the input array with uniform smoothing kernel
    before finding the trend. When period==0, no blurring is
    applied.

  Output parameters
  -----------------
  r : double*
    Specify pointer to an array where the results are stored
    in n columns and m rows.
 */
void iocbio_detrend(double *f, int n, int m, double period, double *r)
{
  int j;
  for(j=0; j<m; ++j)
    iocbio_detrend1(f+j*n, n, 1, period, r+j*n, 1);
}

/*
  iocbio_trend function computes trend from an array of values
  assuming independent rows.

  Input parameters
  ----------------

  f : double*
    Specify pointer to an array using row-storage order.
  n, m : int
    Specify number of columns and rows of the array
  period : double
    Specify estimated period of detrended array. This is used
    for convolving the input array with uniform smoothing kernel
    before finding the trend. When period==0, no blurring is
    applied.

  Output parameters
  -----------------
  r : double*
    Specify pointer to an array where the results are stored
    in n columns and m rows.
 */
void iocbio_trend(double *f, int n, int m, double period, double *r)
{
  int i, j;
  iocbio_detrend(f, n, m, period, r);
  for(j=0; j<m; ++j)
    for(i=0; i<n; ++i)
      r[j*n+i] = f[j*n+i] - r[j*n+i];
}

/*
  iocbio_detrend1 function subtracts trend from a sequence of values.

  Input parameters
  ----------------

  f : double*
    Specify pointer to a sequence using row-storage order.
  n : int
    Specify the number of values in a sequence
  fstride : int
    Specify the stride for accessing the sequence values
  period : double
    Specify estimated period of detrended sequences. This is used
    for convolving the input sequence with uniform smoothing kernel
    before finding the trend. When period==0, no blurring is
    applied.
  rstride : int
    Specify the stride for storing the results.

  Output parameters
  -----------------
  r : double*
    Specify pointer to a sequence where the results are stored.

 */

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

void iocbio_compute_trend_spline_data(double *f, int n, double period, 
				      int* nof_extremes, double* extreme_positions, double* extreme_values,
				      int* nof_averages, double* average_positions, double* average_values)
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
  double v;
  int flag=0, new_flag;
  int count=0;
  int i,j;
  int fstride = 1;
  if (period>0)
    smoothness = 0.25*period-0.5;
  *nof_extremes = 0;
  *nof_averages = 0;
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
	v += f[i+j];
      v /= (1+2*smoothness);

      if (count>1)
	{
	  extreme_point = (extreme_point*(count-1)+i)/count;
	  extreme_value = (extreme_value*(count-1)+v)/count;
	  continue;
	}

      extreme_positions[*nof_extremes] = extreme_point;
      extreme_values[*nof_extremes] = extreme_value;
      (*nof_extremes) ++;

      UPDATE_TREND_SPLINE_DATA(>=, 0, average_point);

      prev_extreme_point = extreme_point;
      prev_extreme_value = extreme_value;
      extreme_point = i;
      extreme_value = v;
    }

  extreme_positions[*nof_extremes] = extreme_point;
  extreme_values[*nof_extremes] = extreme_value;
  (*nof_extremes) ++;
  UPDATE_TREND_SPLINE_DATA(>, 0, average_point);

  prev_extreme_point = extreme_point;
  prev_extreme_value = extreme_value;

  extreme_point = n-1;
  extreme_value = f[n-1];

  extreme_positions[*nof_extremes] = extreme_point;
  extreme_values[*nof_extremes] = extreme_value;
  (*nof_extremes) ++;
  UPDATE_TREND_SPLINE_DATA(>=, 1, n);
}
				      

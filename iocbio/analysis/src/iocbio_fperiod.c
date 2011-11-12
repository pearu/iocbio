/* iocbio_fperiod - routines for finding fundamental period in a signal

  This file provides the following functions:

     iocbio_fperiod - return fundamental period of an array
     iocbio_fperiod_cached - worker routine of iocbio_fperiod
     iocbio_objective - evaluate objective function

  Author: Pearu Peterson
  Created: October 2011
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "iocbio_fperiod.h"
#include "iocbio_ipwf.h"
#include "iocbio_detrend.h"


#define ABS(X) ((X)<0?-(X):(X))
#define MAX(X, Y) ((X)<(Y)?(Y):(X))
#define MIN(X, Y) ((X)>(Y)?(Y):(X))
#define IFLOOR(X) ((int)floor(X))

/* Evaluate objective function.
 */

void iocbio_objective(double *y, int k, double *f, int n, int m, int order, int method, double *r)
{
  int j;
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
  for (j=0; j<k; ++j)
    r[j] = evaluate(y[j], f, n, m, order);
}

/*
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
    Specify objective function. 
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
      cache = malloc(sizeof(double)*n*m);
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
	  status = find_zero(start_j, 0, f2, n, m, 1, &extreme, &convexity);
	  if (status==0 && convexity>0.0) /* extreme is minimum point */
	    return extreme;
	  if (status==0 && convexity<0.0) /* skipping maximum point */
	    {
	      /* Check the case where minimum and maximum (extreme found from right) is in the same interval */
	      status = find_zero(IFLOOR(extreme), IFLOOR(extreme)+1, f2, n, m, 1, &extreme, &convexity);
	      if (status==0 && convexity>0.0)
		return extreme;
	      /* Continue the search to the left */
	      status = find_zero(IFLOOR(extreme), 0, f2, n, m, 1, &extreme, &convexity);
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

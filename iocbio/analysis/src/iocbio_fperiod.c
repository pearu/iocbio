/* iocbio_fperiod - routines for finding fundamental period in a signal

  This file provides the following functions:
     iocbio_fperiod - return fundamental period of an array

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
    default:
      printf("iocbio_objective: method value not in [0, 1], got %d\n", method);
      return;
    }

  for (j=0; j<k; ++j)
    r[j] = evaluate(y[j], f, n, m, order);
}

double iocbio_fperiod(double *f, int n, int m, double initial_period, int detrend, int method)
{
  double* cache = malloc(sizeof(double)*n*m);
  double fperiod;
  fperiod = iocbio_fperiod_cached(f, n, m, initial_period, detrend, method, cache);
  free(cache);
  return fperiod;
}

double iocbio_fperiod_cached(double *f, int n, int m, double initial_period, int detrend, int method, double *cache)
{
  /* This function returns the second (counted from the origin,
     inclusive) minimum point of dissimilarity measure e11 that
     defines the fundamental period of f. The logic of this routine is
     built up from the assumption that the e11 resembles
     (1-cos(2*pi*y/p))/(n-y) in the vicinity of the origin. If this is
     not the case then the fundamental period of f might not be
     well-defined.
   */
  int start_j = floor(initial_period);
  int end_j = start_j + 1;
  double fperiod = -1.0;
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
    default:
      printf("iocbio_fperiod_cached: method value not in [0, 1], got %d\n", method);
      return -1.0;
    }

  if (detrend)
    iocbio_detrend(f, n, m, initial_period, cache);
  start_j = MAX(0, start_j);
  end_j = MIN(n-1, end_j);
  if (start_j==0)
    { /* Initial period was not specified.
       */
      status = find_zero(1, n, f2, n, m, 1, &extreme, &convexity);
      if (status==0)
	{
	  if (convexity<0) /* extreme is a maximum point, initial period is
			      approximated as twice of the extreme point. */
	    {
	      if (floor(extreme*2.0)>0.0) /* to avoid recursion */
		return iocbio_fperiod_cached(f2, n, m, 2.0*extreme, 0, method, cache);
	      else
		; /* this should never happen */
	    }
	  if (convexity>0) /* extreme is minimum point. This is unexpected but will do. */
	    return extreme;
	}
      /* Hmm, E is monotonous, no fundamental period could be determined. */
      return fperiod;
    }
  /* For efficiency, first assume floor(fperiod) == floor(initial_period) */
  status = find_zero(start_j, end_j, f2, n, m, 1, &extreme, &convexity);
  if (status==0)
    {
      if (convexity>0.0) /* extreme is minimum point */
	return extreme;
      /* Hmm, extreme corresponds to maximum point. We have to guess
	 whether it is the first or some subsequent maximum point.
	 Will assume that it is the second maximum point because
	 2/3*extreme has more potential to be the fundamental period
	 than 2*extreme. If extreme is the third (or forth, etc) maximum
	 point then the initial estimate must be reduced anyway. If extreme is
	 the first maximum point then when the zero minimum is found, the
	 search will be reset to the case where initial period was not
	 specified. Must be careful not to enter infinite recursion..
       */
      fperiod = iocbio_fperiod_cached(f2, n, m, 2.0*extreme/3.0, 0, method, cache);
      if (floor(fperiod)==0.0)
	/* So, extreme was the first maximum point after all. */
	fperiod = iocbio_fperiod_cached(f2, n, m, 2.0*extreme, 0, method, cache);
      return fperiod;
    }
  /* No extreme point were found.. now we have to decide whether the initial_period
     underestimated or overestimated the fundamental period in order to search
     it from the correct side .*/
  slope = evaluate(initial_period, f2, n, m, 1);
  if (slope<0.0)
    /* fundamental period should be larger than initial_period */
    end_j = n;
  else if (slope>0.0)
    /* fundamental period should be smaller than initial_period */
    end_j = 0;
  else
    /* Strange that the first assumption did not work. */
    return initial_period;

  status = find_zero(start_j, end_j, f2, n, m, 1, &extreme, &convexity);
  if (status==0)
    {
      if (convexity>0.0) /* extreme is minimum point */
	{
	  if (floor(extreme)==0.0) /* found first minimum, doing full scan */
	    fperiod = iocbio_fperiod_cached(f2, n, m, 0.0, 0, method, cache);
	  else
	    fperiod = extreme;
	}
      if (convexity<0.0) /* extreme is maximum point */
	    ; /* this should never happen */
    }
  return fperiod;
}

/* Analytic autocorrelation functions.

See acf.h for documentation.

Auhtor: Pearu Peterson
Created: September 2010
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "acf.h"

#define NEGEPS 1.1102230246251565e-16
#define EPS 2.2204460492503131e-16
#define MAX(x, y) ((x)<(y)?(y):(x))
#define MIN(x, y) ((x)<(y)?(x):(y))
#define GET_NODE_VALUE(F,N,INDEX) (((INDEX)<0 || (INDEX)>=(N))?0.0:(F)[INDEX])

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

#define F77NAME(func) func ## _

extern int F77NAME(lmdif1)(
			   // subroutine fcn(m,n,x,fvec,iflag)
			   void (*)(int*,int*,double*,double*,int*),
			   int*,         // m
			   int*,         // n
			   double*,      // x
			   double*,      // fvec
			   double*,      // tol
			   int*,         // info
			   int*,         // iwa
			   double*,      // wa
			   int*          // lwa
			   );

/* Constant: f(x) is piecewise constant */
static int acf_coeff_indices_Constant[2] = {0,1};
static double acf_coeffs_Constant[2*2] = {
  1., 0.,
  -1.,1.
};
static double acf_data_Constant[2] = {0,0};

/* Linear: f(x) is piecewise linear */
static int acf_coeff_indices_Linear[4] = {-1,0,1,2};
static double acf_coeffs_Linear[4*4] = {
  1./6., 2./3., 1./6., 0.,
  -0.5,0.,0.5,0.,
  0.5,-1.,0.5,0.,
  -1./6.,0.5,-0.5,1./6.
};
static double acf_data_Linear[4] = {0,0,0,0};

/* CatmullRom: f(x) is piecewise cubic */
static int acf_coeff_indices_CatmullRom[8] = {-3,-2,-1,0,1,2,3,4};
static double acf_coeffs_CatmullRom[8*8] = {
  1./560., -1./28., 71./560., 57./70., 71./560., -1./28.,  1./560., 0.,
  -1./240., 1./10.,-11./16.,   0.,    11./16.,  -1./10.,  1./240., 0.,
  -1./240.,-1./60., 29./48.,  -7./6.,  29./48.,  -1./60., -1./240., 0.,
  1./48.,  -1./6.,  13./48.,   0.,   -13./48.,   1./6.,  -1./48.,  0.,
  -1./48.,  1./6., -23./48.,   2./3., -23./48.,   1./6.,  -1./48.,  0.,
  1./240., -1./40., 13./240., -1./24., -1./48.,   7./120.,-3./80.,  1./120.,
  1./240., -1./30.,  9./80.,  -5./24., 11./48.,  -3./20., 13./240.,-1./120.,
  -1./560., 1./80., -3./80.,   1./16., -1./16.,   3./80., -1./80.,  1./560.
};
static double acf_data_CatmullRom[8] = {0,0,0,0,0,0,0,0};

typedef struct {
  ACFInterpolationMethod mth;
  int iy;
  double* f;
  double* data;
  int sz;
} ACFStateType;

static ACFStateType acf_state = {ACFUnspecified, 0, NULL, NULL, 0};

static
int acf_calculate_data(double* f, int n, int iy, 
		       ACFInterpolationMethod mth,
		       double** data, int* sz)
{
  int start, end, i, j, k, m, k1;
  int* coeff_indices;
  double d, fk0, fk1;
  double* coeffs = NULL;
  if (mth==acf_state.mth &&
      iy==acf_state.iy &&
      f==acf_state.f &&
      acf_state.data != NULL &&
      acf_state.sz > 0
      )
    {
      *data = acf_state.data;
      *sz = acf_state.sz;
      return 0;
    }

  switch (mth)
    {
    case ACFInterpolationConstant:
      *sz = 2;
      coeff_indices = acf_coeff_indices_Constant;
      coeffs = acf_coeffs_Constant;
      *data = acf_data_Constant;
      break;
    case ACFInterpolationLinear:
      *sz = 4;
      coeff_indices = acf_coeff_indices_Linear;
      coeffs = acf_coeffs_Linear;
      *data = acf_data_Linear;
      break;
    case ACFInterpolationCatmullRom:
      *sz = 8;
      coeff_indices = acf_coeff_indices_CatmullRom;
      coeffs = acf_coeffs_CatmullRom;
      *data = acf_data_CatmullRom;
      break;
    default: 
      *sz = 0;
      *data = NULL;
      return -1;
    }
  for (j=0; j<*sz; ++j) (*data)[j] = 0.0;
  start = MAX(iy, -coeff_indices[0]-2);
  end = n + coeff_indices[*sz-1];
  for (j=start; j<end; ++j)
    {
      k = j - iy;
      fk0 = GET_NODE_VALUE(f, n, k);
      for (i=0; i<*sz; ++i)
	{
	  d = 0.0;
	  for (m=0; m<*sz; ++m)
	    {
	      k1 = coeff_indices[m] + j;
	      fk1 = GET_NODE_VALUE(f, n, k1);
	      d += coeffs[i*(*sz)+m] * fk1;
	    }
	  (*data)[i] += fk0 * d;
	}
    }
  acf_state.mth = mth;
  acf_state.iy = iy;
  acf_state.f = f;
  acf_state.data = *data;
  acf_state.sz = *sz;
  return 0;
}

double acf_evaluate(double* f, int n, double y, ACFInterpolationMethod mth)
{
  int iy = floor((y<0?-y:y));
  double dy = y - iy;
  int sz, i;
  double *data = NULL;
  double result = 0.0;
  double p = 1.0;
  acf_calculate_data(f, n, iy, mth, &data, &sz);
  assert(data!=NULL);
  for (i=0; i<sz; ++i, p *= dy)
    result += data[i] * p;
  return result;
}

double acf_maximum_point(double* f, int n, int start_j, ACFInterpolationMethod mth)
{
  int j, sz;
  double *data = NULL;
  double a,b,c,d;
  double s;
  double fy;
  switch (mth)
    {
      case ACFInterpolationLinear:
	for (j=MAX(1,start_j); j<n-1; ++j)
	  {
	    acf_calculate_data(f, n, j, mth, &data, &sz);
	    a = acf_data_Linear[3];
	    b = acf_data_Linear[2];
	    c = acf_data_Linear[1];
	    d = acf_data_Linear[0];
	    s = b*b-3*a*c;
	    if (s < 0)
	      continue;
	    fy = -(b+sqrt(s))/(3.*a);
	    if (fy<-NEGEPS || (1-fy)<EPS)
	      continue;
	    return j + fy;
	  }
	break;
    default: ;
      /* not implemented */
    }
  return -1.0;
}

double acf_sine_fit(double* f, int n, int start_j, ACFInterpolationMethod mth)
{
  double a = acf_evaluate(f, n, 0.0, mth);
  double p0 = acf_maximum_point(f, n, start_j, ACFInterpolationLinear);
  double omega = 2.0*M_PI/p0;
  double tol;
  double *fvec = NULL;
  int iwa;
  double *wa = NULL;
  int m,n1,lwa,info;
  void fnc(int* m,int* n1,double* x,double* fvec,int* iflag)
  {
    assert(n1==1);
    double omega = x[0];
    int j;
    for (j=0; j<*m; ++j)
      fvec[j] = acf_evaluate(f, n, j, mth) - a*cos(omega*j)*(n-j)/n;
  }
  m = n;
  n1 = 1;
  lwa = m*n1+5*n1+m;
  info = 0;
  tol = 1e-8;
  fvec = (double*)malloc(sizeof(double)*m);
  wa = (double*)malloc(sizeof(double)*lwa);
  F77NAME(lmdif1)(fnc,&m,&n1,&omega,fvec,&tol,&info,&iwa,wa,&lwa);
  free(fvec);
  free(wa);
  return omega;
}

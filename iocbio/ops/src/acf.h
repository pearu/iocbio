/* Analytic autocorrelation functions.

   This code provides the following functions:

     double acf_evaluate(double* f, int n, double y, InterpolationMethod mth)
     double acf_maximum_point(double* f, int n, int start_j, InterpolationMethod mth)
     double acf_sine_fit(double* f, int n, int start_j, ACFInterpolationMethod mth)
     
   * acf_evaluate computes the value of an analytic autocorrelation
   function ACF(f)(y) of a piecewice polynomial function f(x). The
   parameters to acf_evaluate function are as follows:
   
     f - a pointer to function values f(i) at node points i=0,...,n-1
     n - the number of node points
     y - the argument to the ACF
     mth - interpolation method for finding f values between node points.
     
   Currently the impelented interpolation methods are Constant, Linear,
   and CatmullRom spline.
   
   Note that f(i)=0, i<0 or i>=n, is assumed.

   * acf_maximum_point returns the point after start_j where
   ACF(f)(y) obtains its maximum. start_j should be a floor of
   f(x) period for optimal computation.

   * acf_sine_fit returns sine fit parameter omega such that
       ACF(f)(y) = a * cos(omega*j)*(n-j)/n
     where a = ACF(f)(0). Note that acf_sine_fit uses
     acf_maximum_point as initial guess for omega = 2*pi/max_point,
     hence the start_j argument.

Auhtor: Pearu Peterson
Created: September 2010
 */

#ifndef ACF_H_INCLUDE
#define ACF_H_INCLUDE

typedef enum {ACFInterpolationConstant=0, 
	      ACFInterpolationLinear=1, 
	      ACFInterpolationCatmullRom=2, 
	      ACFUnspecified=999} ACFInterpolationMethod;

extern double acf_evaluate(double* f, int n, double y, ACFInterpolationMethod mth);
extern double acf_maximum_point(double* f, int n, int start_j, ACFInterpolationMethod mth);
extern double acf_sine_fit(double* f, int n, int start_j, ACFInterpolationMethod mth);
extern double acf_sine_power_spectrum(double* f, int n, double omega, ACFInterpolationMethod mth);

#endif

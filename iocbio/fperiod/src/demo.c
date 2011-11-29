/* Demo program for estimating fundamental period of a sine signal.

   This program is part of the IOCBio project. See
   http://iocbio.googlecode.com/ for more information.

   To compile, run:

     cc demo.c libfperiod.c -o demo -lm

   Usage: 

     demo [<n> [<period> [<detrend>]]]
   
   where n is the length of the signal, period is the expected
   fundamental period, and detrend!=0 enables using the detrend
   algorithm.
   
   Example session:

$ ./demo 20 5.4 0
Signal definition: f[i]=sin(2*pi*i/5.400000), i=0,1,..,19
Number of repetitive patterns in the signal=3.703704
Detrend algorithm is disabled
Expected  fundamental period=5.400000
Estimated fundamental period=5.389328
Relative error of the estimate=0.197630%
$

 */

/* 
   Author: Pearu Peterson
   Created: November 2011
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "libfperiod.h"

int main(int argc, char *argv[])
{

  int n = 30; /* default length of a signal (one row) */
  int m = 1; /* number of signal rows */
  double *f = NULL; /* holds signal data */
  double period = 5.3; /* default value of expected period */
  double initial_period = 0.0; /* 0.0 means unspecified */
  int detrend = 0; /* when non-zero then detrend algorithm is applied
		      prior the fundamental period estimation. The
		      detrend algorithm removes background field from
		      the signal. */
  int method = 0; /* unused argument for libfperiod */
  double fperiod; /* estimated fundamental period */
  int i;

  /* Read parameters from program arguments */
  switch(argc)
    {
    case 4: 
      detrend = atoi(argv[3]);
    case 3:
      period = atof(argv[2]); 
    case 2:
      n = atoi(argv[1]); 
    case 1:
      break;
    default:
      printf("Unexpected number of arguments: %d\nUsage: %s [<length of a signal> [<period of a signal> [<use detrend>]]]", argc, argv[0]);
    }

  /* Initialize signal */
  printf("Signal definition: f[i]=sin(2*pi*i/%f), i=0,1,..,%d\n", period, n-1);
  printf("Number of repetitive patterns in the signal=%f\n", n/period);
  if (detrend)
    printf("Detrend algorithm is enabled\n");
  else
    printf("Detrend algorithm is disabled\n");
  f = (double*)malloc(sizeof(double)*n*m); 
  for (i=0;i<n;++i) 
    {
      f[i] = sin(2.0*M_PI/period*i);
      if (n<20)
	printf("f[%d] = %f\n", i, f[i]);
    }

  /* Estimate fundamental period of the signal */
  fperiod = iocbio_fperiod(f, n, m, initial_period, detrend, method);

  /* Show results */
  if (fperiod<=0.0)
    printf("Failed to compute fundamental period: status=%f\n", fperiod);
  else
    {
      printf("Expected  fundamental period=%f\n", period);
      printf("Estimated fundamental period=%f\n", fperiod);
      printf("Relative error of the estimate=%f%%\n", 100*(1.0-fperiod/period));
    }
  return 0;
}

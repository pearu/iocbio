#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "libfperiod.h"

int main(int argc, char *argv[])
{

  int n = 30; /* default length of a signal (one row) */
  int m = 1; /* number of signal rows */
  double *f = NULL; /* holds signal data */
  double period = 4.3; /* default value of expected period */
  double initial_period = 0.0; /* 0.0 means unspecified */
  int detrend = 0; /* when non-zero then detrend algorithm is applied
		      prior the fundamental period estimation. The
		      detrend algorithm removes background field from
		      the signal. */
  int method = 0; /* unused argument for libfperiod */
  double fperiod; /* estimated fundamental period */
  int i;

  switch(argc)
    {
    case 1: break;
    case 2: 
      n = atoi(argv[1]); 
      break;
    case 3: 
      n = atoi(argv[1]); 
      period = atof(argv[2]); 
      break;
    default:
      printf("Unexpected number of arguments: %d\nUsage: %s [<length of a signal> [<period of a signal>]]", argc, argv[0]);
    }

  printf("Signal f[i]=sin(2*pi*i/%f), i=0,1,..,%d\n", period, n-1);
  printf("Nof repetitive patterns in the signal=%f\n", n/period);

  f = (double*)malloc(sizeof(double)*n*m); 
  for (i=0;i<n;++i) /* initialize signal */
    {
      f[i] = sin(2.0*M_PI/period*i);
      if (n<20)
	printf("f[%d] = %f\n", i, f[i]);
    }
  /* Estimate fundamental period of the signal */
  fperiod = iocbio_fperiod(f, n, m, initial_period, detrend, method);

  printf("Expected  fundamental period=%f\n", period);
  printf("Estimated fundamental period=%f\n", fperiod);
  printf("Relative error of the estimate=%f%%\n", 100*(1.0-fperiod/period));

  return 0;
}

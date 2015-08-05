# Introduction #

IOCBio provides a C library `libfperiod` and Python package `iocbio.fperiod` with tools to determine the fundamental period of a signal with an application to estimating sarcomere length of a single cardiomyocyte from its microscopy image. The underlying theory of the method is described in a paper

  * Pearu Peterson, Mari Kalda, Marko Vendelin. [Real-time Determination of Sarcomere Length of a Single Cardiomyocyte during Contraction](http://ajpcell.physiology.org/content/early/2012/12/19/ajpcell.00032.2012.abstract) _Am J Physiol Cell Physiol_, 304(6):C519–C531, 2013.

This page describes how to use the provided software.
Please file all found bugs and feature requests to Issues.

# Usage from a C program #

First, download `libfperiod` source code. The source code consists of three files:

  * [libfperiod.c](http://iocbio.googlecode.com/svn/trunk/iocbio/fperiod/src/libfperiod.c) - source code to be compiled and linked with your C program
  * [libfperiod.h](http://iocbio.googlecode.com/svn/trunk/iocbio/fperiod/src/libfperiod.h) - header file to be included by the C program
  * [demo.c](http://iocbio.googlecode.com/svn/trunk/iocbio/fperiod/src/demo.c) - demo code illustrating the basic usage of determining the fundamental period of a sine function.

Second, study the demo program above that shows how to call the `iocbio_fperiod` function from your C program. Here follows an example session:
```
$ cc demo.c libfperiod.c -o demo -lm
$ ./demo 20 5.4
Signal definition: f[i]=sin(2*pi*i/5.400000), i=0,1,..,19
Number of repetitive patterns in the signal=3.703704
Detrend algorithm is disabled
Expected  fundamental period=5.400000
Estimated fundamental period=5.389328
Relative error of the estimate=0.197630%
```

The `libfperiod` API exposes the following functions to users (see `libfperiod.h` for C definitions):

  * iocbio\_fperiod - compute fundamental period of a sequence
  * iocbio\_fperiod\_cached - same as iocbio\_fperiod but with extra cache argument that can be used to avoid malloc/free cycle in repetitive computations of the fundamental period.
  * iocbio\_objective - evaluate the similarity measure that defines the fundamental period of a signal sequence as its first non-zero minimum point
  * iocbio\_detrend - apply detrend algorithm to a signal sequence

# Usage from a Python program #

First, you must install `iocbio` Python package version `1.3` or newer.

Here follows an example Python session using the `iocbio.fperiod` package:
```
>>> from iocbio import fperiod
>>> print fperiod.fperiod.__doc__
fperiod - Function signature:
  period = fperiod(f,[initial_period,detrend,method])
Required arguments:
  f : input rank-2 array('d') with bounds (m,n)
Optional arguments:
  initial_period := 0.0 input float
  detrend := 0 input int
  method := 0 input int
Return objects:
  period : float

>>> from numpy import sin, arange, pi
>>> N = 20; i = arange(N)
>>> P = 5.4
>>> f = sin(2*pi*i/P)
>>> Pest = fperiod.fperiod(f)
>>> print 'Estimated fundamental period:',Pest
Estimated fundamental period: 5.3893279719
>>> print 'Relative error of the estimate:',(1-Pest/P)*100,'%'
Relative error of the estimate: 0.197630149933 %
```
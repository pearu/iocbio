"""Provides regression functions.

The :func:`regress` function can be used to smoothen noisy 3D images
using local averaging or local linear regression with various kernels
and boundary conditions.

Example
-------

The following example illustrates local linear regression method for
1D data with `Poisson noise
<http://en.wikipedia.org/wiki/Poisson_distribution>`_::

  from numpy import *
  from scipy.stats import poisson
  from iocbio.ops import regress
  x = arange(0,2*pi,0.1)
  data = 50+7*sin(x)+5*sin(2*x)
  data_with_noise = poisson.rvs (data)
  data_estimate = regress(data_with_noise, (0.1, ), method='linear', kernel='tricube', boundary='periodic')

.. image:: ../_static/regress_1d.png
  :width: 60%


Module content
--------------
"""
# Author: Pearu Peterson
# Created: September 2009

__all__ = ['regress']

import sys

try:
    from . import regress_ext
except ImportError, msg:
    print msg

def regress(data, scales,
            kernel='uniform', 
            method='average',
            boundary='finite',
            verbose = True):
    """
    Estimate a scalar field from noisy observations (data).

    Parameters
    ----------

    images : numpy.ndarray
      Noisy data.

    scales : tuple
      Kernel scaling parameters. ``1/scales[i]`` defines the
      half-width of a kernel for i-th dimension. For example, if
      `scales=(0.1, 0.5)` then `2/0.1+1=11` and `2/0.5+1=5`
      neighboring data points will averaged (using kernel weights) in
      1st and 2nd dimension, respectively.

    kernel : {'epanechnikov', 'uniform', 'triangular', 'quartic', 'triweight', 'tricube'}
      Smoothing kernel type, see `kernel types
      <http://en.wikipedia.org/wiki/Kernel_(statistics)>`_ for
      definitions.

    method : {'average', 'linear'} 
      Smoothing method, see `kernel smoothers
      <http://en.wikipedia.org/wiki/Kernel_smoother>`_ for method
      descriptions.

    boundary : {'constant', 'finite', 'periodic', 'reflective'}

      Boundary condition define how data values are extrapolated when
      kernel support goes out of the data range. The following table
      summarizes how boundary conditions and extrapolated values are
      related (``N`` denotes the last valid index for ``data`` and
      ``i>=1``):

      +--------------------+-----------------------------------------------+
      | Boundary condition | Extrapolation relations                       |
      +====================+===============================================+
      | constant           | data[N + i] = data[N], data[-i] = data[1]     |
      +--------------------+-----------------------------------------------+
      | finite             | data[N + i] = NaN, data[-i] = NaN             |
      +--------------------+-----------------------------------------------+
      | periodic           | data[N + i] = data[i], data[-i] = data[N - i] |
      +--------------------+-----------------------------------------------+
      | reflective         | data[N + i] = data[N - i], data[-i] = data[i] |
      +--------------------+-----------------------------------------------+

    verbose : bool
      when True then show the progress of computations to terminal.

    Returns
    -------
    new_data : numpy.ndarray
      smoothed data

    See also
    --------
    :mod:`iocbio.ops.regression`
    """
    kernel_types = dict (epanechnikov=0, uniform=1, 
                         triangular = 2, quartic=3,
                         triweight=4, tricube=5)

    smoothing_methods = dict(average=0, 
                             linear=1)

    boundary_conditions = dict(constant=0,   # out of boundary points are equal to closest boundary point
                               finite = 1,   # ignore out of boundary points
                               periodic=2,   # periodic boundaries
                               reflective=3, # boundary is a mirror
                               )
    if kernel not in kernel_types:
        raise ValueError('kernel type must be %s but got %s' \
                             % ('|'.join(map (str, kernel_types)), kernel))
    if method not in smoothing_methods:
        raise ValueError('smoothing method must be %s but got %s' \
                             % ('|'.join(map (str, smoothing_methods)), method))
    if boundary not in boundary_conditions:
        raise ValueError('boundary condition must be %s but got %s' \
                             % ('|'.join(map (str, bondary_conditions)), boundary))
    if verbose:
        def write_func(fmt, *args):
            sys.stdout.write(fmt % args)
            sys.stdout.flush()
    else:
        write_func = None
    return regress_ext.regress (data, scales, kernel_types[kernel],
                                smoothing_methods[method], boundary_conditions[boundary],
                                write_func)

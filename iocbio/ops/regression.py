"""Provides regression functions.

The :func:`regress` function can be used to smoothen noisy images
(upto 3D) using local averaging or local linear regression with
various kernels and boundary conditions. In addition, the
:func:`regress` function can be used for computing gradients of images

The :func:`kernel` function can be used for calculating the values
of regression kernels.

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
  data_with_noise = poisson.rvs (data).astype(data.dtype)
  data_estimate, data_gradient = regress(data_with_noise, (0.1, ), method='linear', kernel='tricube', boundary='periodic')

..
   import matplotlib.pyplot as plt
   plt.plot(x,data,x,data_with_noise,x,data_estimate)
   plt.title('Local linear regression for recovering data = 50+7*sin(x)+5*sin(2*x)')
   plt.legend(['data', 'data with Poisson noise', 'estimated data'])
   plt.xlabel('x'); plt.ylabel('y')
   plt.savefig('regress_1d.png')

.. image:: ../_static/regress_1d.png
  :width: 60%

..
   import matplotlib.pyplot as plt
   from iocbio.ops.regression import kernel_types, kernel
   x = arange(-(1/0.1),(1/0.1)+1)
   l = []
   for kernel_name in kernel_types:
       l.append(kernel_name)
       y = kernel((0.1,), kernel_name)
       plt.plot(x, y)
   plt.legend(l)
   plt.savefig('regress_kernels.png')

Available kernels
-----------------

The following plot shows regression kernels for ``scales=(0.1, )``:

.. image:: ../_static/regress_kernels.png
  :width: 60%


Module content
--------------
"""
# Author: Pearu Peterson
# Created: September 2009

__all__ = ['regress', 'kernel']

import sys

try:
    from . import regress_ext
except ImportError, msg:
    print msg

kernel_types = dict (epanechnikov=0, uniform=1, 
                     triangular = 2, quartic=3,
                     triweight=4, tricube=5, gaussian=6)

def kernel (scales, kernel = 'uniform'):
    """ Calculate regression kernel.

    Parameters
    ----------
    scales : tuple
      Kernel scaling parameters. ``1/scales[i]`` defines the
      half-width of a kernel for i-th dimension. For example, if
      ``scales=(0.1, 0.5)`` then ``2/0.1+1=21`` and ``2/0.5+1=5``
      node points will be used in 1st and 2nd dimension,
      respectively. Normally, scales[i] is smaller or equal to 1.

    kernel : {'epanechnikov', 'uniform', 'triangular', 'quartic', 'triweight', 'tricube', 'gaussian'}
      Smoothing kernel type, see `kernel types
      <http://en.wikipedia.org/wiki/Kernel_(statistics)>`_ and
      `tri-cube kernel
      <http://en.wikipedia.org/wiki/Local_regression>`_ for
      definitions. Available kernels are visualized also in
      :mod:`iocbio.ops.regression`.

    Returns
    -------
    kernel_data : numpy.ndarray
      Kernel values at node points.

    Notes
    -----
    For kernels that have non-zero values at the boundaries (uniform,
    gaussian), the boundary values are multiplied by 0.5 to minimize
    error in discrete convolution.

    The scale of gaussian kernel is choosen such that the boundary
    value of discrete gaussian is 100x smaller than the center value.

    See also
    --------
    :mod:`iocbio.ops.regression`, regress
    """
    return regress_ext.kernel(tuple(scales), kernel_types[kernel])

def regress(data, scales,
            kernel='uniform', 
            method='average',
            boundary='finite',
            verbose = True,
            options = None):
    """
    Estimate a scalar field from noisy observations (data).

    Parameters
    ----------

    images : numpy.ndarray
      Noisy data.

    scales : tuple
      Kernel scaling parameters. ``1/scales[i]`` defines the
      half-width of a kernel for i-th dimension. For example, if
      `scales=(0.1, 0.5)` then `2/0.1+1=21` and `2/0.5+1=5`
      neighboring data points will averaged (using kernel weights) in
      1st and 2nd dimension, respectively.
      Normally, scales[i] is smaller or equal to 1.

    kernel : {'epanechnikov', 'uniform', 'triangular', 'quartic', 'triweight', 'tricube', 'gaussian'}
      Smoothing kernel type, see `kernel types
      <http://en.wikipedia.org/wiki/Kernel_(statistics)>`_ and 
      `tri-cube kernel <http://en.wikipedia.org/wiki/Local_regression>`_ for
      definitions. Available kernels are visualized also in
      :mod:`iocbio.ops.regression`.

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
      When True then show the progress of computations to terminal.

    options : `iocbio.utils.Options`
      Specify regression parameters from command line. This will override
      parameters specified in function call. The following options attributes
      are used: ``options.kernel``, ``options.method``, ``options.boundary``.

    Returns
    -------
    new_data : numpy.ndarray
      Smoothed data.
    new_data_grad : numpy.ndarray
      Gradient of smoothed data only if method=='linear', otherwise
      nothing is returned.

    See also
    --------
    :mod:`iocbio.ops.regression`, kernel
    """
    smoothing_methods = dict(average=0, 
                             linear=1)

    boundary_conditions = dict(constant=0,   # out of boundary points are equal to closest boundary point
                               finite = 1,   # ignore out of boundary points
                               periodic=2,   # periodic boundaries
                               reflective=3, # boundary is a mirror
                               )
    if options is not None:
        kernel = options.get(kernel=kernel)
        method = options.get(method=method)
        boundary = options.get(boundary=boundary)

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
    result, grad = regress_ext.regress (data, tuple(scales), kernel_types[kernel],
                                        smoothing_methods[method], boundary_conditions[boundary],
                                        write_func)
    if method=='average':
        return result
    return result, grad

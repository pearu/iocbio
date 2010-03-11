"""Provides convolve function.

Example
-------

The following example illustrates how to convolve data with a kernel::

  from numpy import *
  from scipy.stats import poisson
  from iocbio.ops import convolve
  kernel = array([0,1,2,2,2,1,0])
  x = arange(0,2*pi,0.1)
  data = 50+7*sin(x)+5*sin(2*x)
  data_with_noise = poisson.rvs(data)
  data_convolved = convolve(kernel, data_with_noise)

.. image:: ../_static/convolve_1d.png
  :width: 60%

Module content
--------------
"""

from __future__ import division

__all__ = ['convolve']

from scipy import fftpack
from . import fft_tasks
from .. import utils

def convolve(kernel, data, options = None):
    """
    Convolve kernel and data using FFT.

    Parameters
    ----------
    kernel : numpy.ndarray
      The center of kernel is assumed to be in the middle of kernel array.
      If kernel has smaller size than data then kernel will be expanded
      with its boundary values. So, for many application the kernel
      must have ``kernel[0]==kernel[-1]==0``.
    data : numpy.ndarray
    options : optparse.Values
      options.float_type defines FFT algorithms floating point type:
      float or double

    Returns
    -------
    result : numpy.ndarray

    See also
    --------
    :mod:`iocbio.ops.convolution`
    """
    float_type = None
    if options is not None:
        float_type = options.float_type
    if float_type is None:
        float_type = 'double'
    task = fft_tasks.FFTTasks(data.shape, float_type, options=options)
    if kernel.shape != data.shape:
        # assuming that kernel has smaller size than data
        kernel = utils.expand_to_shape(kernel, data.shape, data.dtype)
    kernel = fftpack.fftshift(kernel)
    kernel = kernel / kernel.sum()
    task.set_convolve_kernel(kernel)
    return task.convolve(data)

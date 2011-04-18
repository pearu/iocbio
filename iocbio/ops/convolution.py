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

"""

from __future__ import division

__all__ = ['convolve']

from scipy import fftpack
from . import fft_tasks
from .. import utils

def convolve(kernel, data, kernel_background = None, options = None):
    """
    Convolve kernel and data using FFT algorithm.

    Parameters
    ----------
    kernel : numpy.ndarray
      Specify convolution kernel.  The center of the kernel is assumed
      to be in the middle of kernel array.
    data : numpy.ndarray
      Specify convolution data.
    kernel_background : {int, float, None}
      If kernel has smaller size than data then kernel will be expanded
      with kernel_background value. By default, kernel_background value
      is ``kernel.min()``.
    options : {`iocbio.utils.Options`, None}
      options.float_type defines FFT algorithms floating point type:
      ``'float'`` (32-bit float) or ``'double'`` (64-bit float).

    Returns
    -------
    result : numpy.ndarray

    See also
    --------
    :mod:`iocbio.ops.convolution`
    """
    if options is None:
        options = utils.Options()
    else:
        options = utils.Options(options)
    float_type = options.get (float_type='double')
    task = fft_tasks.FFTTasks(data.shape, float_type, options=options)
    if kernel.shape != data.shape:
        # assuming that kernel has smaller size than data
        kernel = utils.expand_to_shape(kernel, data.shape, float_type, background=kernel_background)

    kernel = fftpack.fftshift(kernel)

    kernel = kernel / kernel.sum()

    task.set_convolve_kernel(kernel)
    result = task.convolve(data)

    return result

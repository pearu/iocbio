"""Provides apply_window function.

The :func:`apply_window` function can be used to multiply 3D images
with a window that suppresses the boundaries to zero and leaves the
inside image values constant.

Multiplication by such a window provides a robust way to periodize 3D
images and so minimize `Gibbs phenomenon
<http://en.wikipedia.org/wiki/Gibbs_phenomenon>`_ for tools using
Fourier transform images.

Example
-------

The following example illustrates the result of applying window
to a constant data=1, that is, the result will be window function
itself::


  from numpy import *
  from iocbio.ops import apply_window
  data = ones (50)
  window0 = apply_window(data, (1/20., ), smoothness=0, background=0.2)
  window1 = apply_window(data, (1/20., ), smoothness=1, background=0)

.. image:: ../_static/apply_window_1d.png
  :width: 60%


Module content
--------------
.. autofunction:: apply_window
"""

__all__ = ['apply_window']

try:
    from .apply_window_ext import apply_window_inplace
except ImportError, msg:
    print msg

def apply_window(data, scales, smoothness=1, 
                 background=0.0, inplace=False):
    """ Multiply data with window function.

    Each value in data will be multiplied with ``w=w[0]*w[1]*..``
    where ``w[k] = f(scales[k] * min(index[k],
    shape[k]-index[k]-1))``, ``k=0,1,..``, and ``f(x)`` is
    ``2*smoothness+1`` times differentiable function such that
    ``f(x)=1`` if ``x>1`` and ``f(0)=0``.

    Parameters
    ----------
    data : numpy.ndarray
    scales : tuple
      Scaling factors. ``1/scales[i]`` defines the width of
      non-constant window.

    smoothness : int
      Specifies smoothness of the window so that it will be
      ``2*smoothness+1`` times continuously differentiable.

    background : float
      Specifies the background level of data. When non-zero,
      the result of apply window is ``(data - background) * window + background``.

    inplace : bool
      When True then multiplication is performed in-situ.

    Returns
    -------
    data : numpy.ndarray
    """
    if not inplace:
        data = data.copy()
    apply_window_inplace(data, scales, smoothness, background)
    return data

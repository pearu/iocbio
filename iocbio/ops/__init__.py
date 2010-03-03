"""
Operations on microscope images.


.. currentmodule:: iocbio.ops

The :mod:`iocbio.ops` provides the following modules for for
manipulating microscope images.

.. autosummary::

  regression
  convolution
  window
"""

from .regression import regress
from .convolution import convolve
from .window import apply_window

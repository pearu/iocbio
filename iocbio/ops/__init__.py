"""
Operations on microscope images.


.. currentmodule:: iocbio.ops

The :mod:`iocbio.ops` provides the following modules for for
manipulating microscope images.

.. autosummary::

  regression
  convolution
  window

Package content
---------------
"""

__autodoc__ = ['regression', 'convolution', 'window', 'regress', 'convolve','apply_window']

from .regression import regress
from .convolution import convolve
from .window import apply_window

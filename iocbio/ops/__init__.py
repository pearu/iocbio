"""
Operations on microscope images.

.. currentmodule:: iocbio.ops

The :mod:`iocbio.ops` provides the following modules for for
manipulating microscope images.

.. autosummary::

  regression
  convolution
  window
  fft_tasks
  autocorrelation
  filters

Package content
---------------
"""

__autodoc__ = ['regression', 'convolution', 'window', 'regress', 'convolve','apply_window',
               'fft_tasks', 'FFTTasks', 
               'autocorrelation', 'acf', 'acf_argmax', 'acf_sinefit',
               'filters', 'convolve_discrete_gauss']

from .regression import regress
from .convolution import convolve
from .window import apply_window
from .fft_tasks import FFTTasks
from .autocorrelation import acf, acf_argmax, acf_sinefit
from .filters import convolve_discrete_gauss

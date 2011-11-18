"""Fundamental period estimation tools

Overview
========

.. currentmodule:: iocbio.fperiod

The :mod:`iocbio.fperiod` provides the following tools:

.. autosummary::
  fperiod
  detrend
  objective
  trend_spline
  ipwf
"""

__autodoc__ = ['fperiod', 'detrend']

from .fperiod_ext import fperiod, fperiod_cached, detrend, objective, trend_spline
from . import ipwf

""" Provides autocorrelation function routines.

This module provides tools for evaluating autocorrelation functions
(ACF) of piecewise polynomial (constant, linear, cubic) functions with
finite support, finding local maximum points of ACFs, and fitting ACF
with a sine function.

Example
-------

>>> from numpy import *
>>> from iocbio.ops.autocorrelation import acf, acf_argmax, acf_sinefit
>>> # Define a signal:
>>> dx = 0.05
>>> x = arange(0,2*pi,dx)
>>> N = len(x)
>>> f = 7*sin(5*x)+6*sin(9*x)
>>> # f is shown in the upper right plot below
>>> # Calculate autocorrelation function:
>>> y = arange(0,N,0.1)
>>> af = acf (f, y, method='linear')
>>> # af is shown in the lower right plot below
>>> # Find the first maximum of the autocorrelation function:
>>> y_max1 = acf_argmax(f, method='linear')
>>> # The first maximum gives period estimate for f
>>> print 'period=',dx*y_max1
period= 0.720986042859
>>> print 'frequency=',2*pi/(y_max1*dx)
frequency= 8.71471142806
>>> # Find the second maximum of the autocorrelation function:
>>> y_max2 = acf_argmax(f, start_j=y_max1+1, method='linear')
>>> print y_max1, y_max2
14.4197208572 27.0918670436
>>> # Find sine-fit of the autocorrelation function:
>>> omega = acf_sinefit(f, method='linear')
>>> # The parameter omega in A*cos (omega*y)*(N-y)/N gives
>>> # another period estimate for f:
>>> print 'period=',2*pi/(omega/dx)
period= 0.695207021807
>>> print 'frequency=', omega/dx
frequency= 9.03786226273


Dominant frequency
------------------

The following plot demonstrates the use of autocorrelation function
for finding dominant frequency of a signal.

.. image:: ../_static/acf_dominant_frequency.png
  :width: 60%

Module content
--------------
"""
# Author: Pearu Peterson
# Created: September 2010

from __future__ import division
__all__ = ['acf', 'acf_argmax', 'acf_sinefit', 'acf_sine_power_spectrum']

try:
    from . import acf_ext
except ImportError, msg:
    print msg

def acf(f, y, method='linear'):
    """ Evaluate autocorrelation function ACF(f(x))(y).

    Parameters
    ----------
    f : {numpy.array, sequence}
      A sequence of f(x) values at nodal points x=0,...,N-1. Note that
      at x<=-1 and x>=N, f(x) is assumed to be zero. Hence the
      autocorrelation function has also finite support [-N, N].
    y : {numpy.array, sequence, float}
      An argument to ACF where the values should be evaluated.
    method : {'constant', 'linear', 'catmullrom'}
      Name of the interpolation method used to evaluate f(x) in
      between nodal points.

    Returns
    -------
    values : {numpy.array, sequence}
      The values of ACF(f(x)) at y.

    See also
    --------
    iocbio.ops.autocorrelation, acf_argmax, acf_sinefit
    """
    mth = dict(constant=0, linear=1, catmullrom=2, cubic=2)[method.lower()]
    return acf_ext.acf(f, y, mth)

def acf_argmax(f, start_j=1, method='linear'):
    """ Find the local maximum point of the autocorrelation function ACF(f(x)).

    Parameters
    ----------
    f : {numpy.array, sequence}
      A sequence of f(x) values at nodal points x=0,...,N-1. Note that
      at x<=-1 and x>=N, f(x) is assumed to be zero. Hence the
      autocorrelation function has also finite support [-N, N].
    start_j : int
      Left starting point of the local maximum point search.
    method : {'constant', 'linear', 'catmullrom'}
      Name of the interpolation method used to evaluate f(x) in
      between nodal points.

    Returns
    -------
    y_max : float
      Point where ACF(f(x)) obtains local maximum.

    See also
    --------
    iocbio.ops.autocorrelation, acf, acf_sinefit
    """
    mth = dict(constant=0, linear=1, catmullrom=2, cubic=2)[method.lower()]
    return acf_ext.acf_argmax (f, int(start_j), mth)

def acf_sinefit(f, start_j=1, method='linear'):
    """ Find the parameter omega of a sine-fit function for ACF(f(x)).

    The sine-fit function is defined as

      SF(y) = A*cos(omega*y)*(N-y)/N

    that approximates the autocorrelation function of a function with
    finite support [0, N-1]. Here A denotes ACF(f (x))(0).

    Parameters
    ----------
    f : {numpy.array, sequence}
      A sequence of f(x) values at nodal points x=0,...,N-1. Note that
      at x<=-1 and x>=N, f(x) is assumed to be zero. Hence the
      autocorrelation function has also finite support [-N, N].
    start_j : int
      Left starting point of the local maximum point search. The
      algorithm uses omega = 2*pi/p as a starting point for parameter
      search. Here p = acf_argmax(f, start_j, method).
    method : {'constant', 'linear', 'catmullrom'}
      Name of the interpolation method used to evaluate f(x) in
      between nodal points.

    Returns
    -------
    y_max : float
      Point where ACF(f(x)) obtains local maximum.

    See also
    --------
    iocbio.ops.autocorrelation, acf, acf_argmax
    """
    mth = dict(constant=0, linear=1, catmullrom=2, cubic=2)[method.lower()]
    return acf_ext.acf_sinefit(f, start_j, mth)

def acf_sine_power_spectrum(f, omega):
    """ Evaluate sine power spectrum SinePower(f(x))(omega).

    Definition::

      SinePower(f(x))(omega) = int_0^N (ACF(f(x))(y) - A*cos(omega*y)*(N-y)/N)))^2 dy

    Parameters
    ----------
    f : {numpy.array, sequence}
      A sequence of f(x) values at nodal points x=0,...,N-1.
    omega : {numpy.array, sequence, float}
      An argument to SinePower(f(x))(omega) where the values should be
      evaluated.

    Returns
    -------
    values : {numpy.array, sequence}
      The values of SinePower(f(x))(omega) at omega.

    Notes
    -----
    The function is currently implemented only for constant
    interpolation of f (x).

    See also
    --------
    iocbio.ops.autocorrelation, acf, acf_sinefit
    """
    method = 'constant'
    mth = dict(constant=0, linear=1, catmullrom=2, cubic=2)[method.lower()]
    return acf_ext.acf_sine_power_spectrum(f, omega, mth)

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
from __future__ import division

__autodoc__ = ['fperiod', 'detrend']

from .fperiod_ext import fperiod, fperiod_cached, detrend, objective, trend_spline
from . import ipwf

def fperiod_acf_fft(image, detrend=True, quad_approx=True, zero_padding=2):
    """ Estimate fundamental period of an image using the maximum point
    of average ACF of image lines computed via FFT. 

    WARNING: This function is implemented to demonstrate various flaws
    of used method. In practical application usage of fperiod function
    is highly recommended.

    Parameters
    ----------
    image : ndarray
      Specify image array. The periodic pattern is assumed to be
      aligned in rows of the image.
    detrend: bool
      When True then detrend the image prior spectrum computation.
    quad_approx: bool
      When True then use quadratic approximation of the spectrum peak.
    zero_padding : int
      Positive integer used for padding Fourier transform of ACF with
      zeros. For periodic images use zero_padding=1. For finite images
      use zero_padding=2.

    Returns
    -------
    fperiod : float
      Estimated fundamental period of an image.
    """
    from numpy import fft, zeros, where, sqrt
    if detrend:
        from .fperiod_ext import detrend
        image = detrend(image)
    N = image.shape[-1]
    fimage = fft.fft(image, axis=-1)
    mfimage = abs(fimage*fimage.conjugate ())
    if len (image.shape)>1:
        mfimage = mfimage.mean(axis=0)
    if zero_padding>1:
        padded = zeros(N*zero_padding)
        padded[:N//2+1] = mfimage[:N//2+1]
        padded[-N//2:] = mfimage[-N//2:]
        mfimage = padded
    acf = fft.ifft(mfimage).real[:N//2]

    imx = 0
    for i in range (1,len (acf)-1):
        if acf[i-1]<=acf[i] and acf[i]>=acf[i+1]:
            imx = i
            break

    if quad_approx:
        denom = (acf[imx-1]+acf[imx+1]-2*acf[imx])
        if denom:
            di = 0.5*(acf[imx-1] - acf[imx+1])/denom
            assert abs(di)<=1,`imx,di`
            return (imx+di)*(1.0/zero_padding)
    return imx*(1.0/zero_padding)

def fperiod_fft(image, detrend=True, quad_approx=True):
    """ Estimate fundamental period of an image using the maximum point
    of the Fourier spectrum of the image.

    WARNING: This function is implemented to demonstrate various flaws
    of used method. In practical application usage of fperiod function
    is highly recommended.

    Parameters
    ----------
    image : ndarray
      Specify image array. The periodic pattern is assumed to be
      aligned in rows of the image.
    detrend: bool
      When True then detrend the image prior spectrum computation.
    quad_approx: bool
      When True then use quadratic approximation of the spectrum peak.

    Returns
    -------
    fperiod : float
      Estimated fundamental period of an image.
    """
    from numpy import fft, zeros, where
    if detrend:
        from .fperiod_ext import detrend
        image = detrend(image)

    N = image.shape[-1]
    fimage = fft.fft(image, axis=-1)

    mfimage = abs(fimage*fimage.conjugate ())
    if len (image.shape)>1:
        mfimage = mfimage.mean (axis=0)
    power_spectrum = zeros(N//2+1)
    power_spectrum[0] = mfimage[0]
    power_spectrum[N//2] = mfimage[N//2]
    power_spectrum[1:N//2] = mfimage[1:N//2] + mfimage[::-1][:N//2-1] # NumRecip

    imx = where(power_spectrum==power_spectrum.max())[0][0]    

    if imx==0:
        return None
        from matplotlib import pyplot as plt
        plt.subplot(211)
        if len (image.shape)>1:
            for line in image:
                plt.plot(line)
        else:
            plt.plot(image)
        plt.subplot(212)
        plt.plot(power_spectrum)
        plt.show ()
        return None

    if quad_approx:
        if imx<N//2:
            #print N, N//2+1, imx
            numer = 0.5*(power_spectrum[imx-1] - power_spectrum[imx+1])
            denom = (power_spectrum[imx-1]+power_spectrum[imx+1]-2*power_spectrum[imx])
            if denom and abs (numer) <= abs (denom):
                return N/(imx + numer/denom)
            elif 1:
                pass
            else:
                from matplotlib import pyplot as plt
                plt.subplot(211)
                if len (image.shape)>1:
                    for line in image:
                        plt.plot(line)
                else:
                    plt.plot(image)
                plt.subplot(212)
                plt.plot(power_spectrum)
                plt.show ()

    return N/imx

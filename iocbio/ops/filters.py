
__all__ = ['convolve_discrete_gauss']

import numpy

def convolve_discrete_gauss(seq, t):
    """ Return convolved sequence with discrete Gaussian kernel.

    Parameters
    ----------
    seq : numpy.ndarray
    t : float
      Scale parameter.

    Notes
    -----

    The discrete Gaussian kernel ``T(n,t)`` is defined as
    ``exp(-t)*I_n(t)`` where ``I_n(t)`` is the modified Bessel
    functions of integer order. The Fourier transform of ``T(n,t)`` is
    ``exp((cos(2*pi*k/N)-1)*t)`` where ``k=0..N-1``.  

    The half-height-width of the discrete Gaussian kernel ``T(n,t)``
    is approximately ``2.36*sqrt(t)``. So, to smooth out details
    over W pixels, use ``t = (W/2.36)**2``.

    References
    ----------
    http://www.nada.kth.se/~tony/abstracts/Lin90-PAMI.html
    http://en.wikipedia.org/wiki/Scale-space_implementation
    """
    n = len(seq)
    fseq = numpy.fft.fft (seq)
    theta = 2*numpy.pi*numpy.arange(0, n, 1, dtype=float)/n
    fker = numpy.exp((numpy.cos(theta)-1)*t)
    fres = fseq * fker
    res = numpy.fft.ifft (fres)
    res = res.real
    return res

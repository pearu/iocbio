"""Provides FFTTasks class.

Examples
--------

The following example illustrates the usage of ``fft`` and ``ifft``
methods:

  >>> from iocbio.ops.fft_tasks import FFTTasks
  >>> task = FFTTasks((4,))
  >>> print task.fft([1,2,3,4])
  [ 10.+0.j  -2.+2.j  -2.+0.j  -2.-2.j]
  >>> print task.ifft(task.fft([1,2,3,4]))
  [ 1.  2.  3.  4.]

The following example illustrates the usage of ``convolve`` method:

  >>> task = FFTTasks((8,), float_type='double')
  >>> task.set_convolve_kernel([0,0,1,2,2,1,0,0])
  >>> print task.convolve([1,0,0,0,0,0,0,0]).round() # kernel
  [ 0. -0.  1.  2.  2.  1.  0. -0.]
  >>> print task.convolve([1,1,0,0,0,0,0,0]).round()
  [ 0. -0.  1.  3.  4.  3.  1. -0.]

The following example illustrates finding optimal FFT sizes with different inputs:
  >>> for sz in [7,13,63,65,129,1023,1025,2049]:
      print '%s -> %s gives speed up %.3fx' % ((sz,)+FFTTasks.get_optimal_fft_size(sz, return_speedup=True, max_nof_tries=100))
  7 -> 8 gives speed up 1.286x
  13 -> 16 gives speed up 1.288x
  63 -> 64 gives speed up 1.253x
  65 -> 66 gives speed up 1.069x
  129 -> 132 gives speed up 1.924x
  1002 -> 1024 gives speed up 4.713x
  1004 -> 1024 gives speed up 2.235x
  1023 -> 1024 gives speed up 2.137x
  1025 -> 1050 gives speed up 1.523x
  2049 -> 2100 gives speed up 4.244x

"""

from __future__ import division
__all__ = ['FFTTasks']


import os
import numpy

from ..utils import mul_seq, VERBOSE, Options

class FFTTasks(object):
    """ Optimized cache for Fourier transforms using `FFTW <http://www.fftw.org/>`_ with operations.

    See also
    --------
    iocbio.ops.fft_tasks, __init__
    """

    _wisdoms = {}

    @staticmethod
    def load_wisdoms(_cache=[]):
        """Load fftw wisdom from a disk.

        See also
        --------
        iocbio.ops.fft_tasks, save_wisdom
        """
        if _cache:
            return
        try:
            import fftw3
        except ImportError, msg:
            fftw3 = None
            print 'FFTTasks.load_wisdoms: %s' % (msg)
        try:
            import fftw3f
        except ImportError, msg:
            fftw3f = None
            print 'FFTTasks.load_wisdoms: %s' % (msg)
        for fftw in [fftw3, fftw3f]:
            if fftw is None:
                continue
            wisdom_file_name = os.path.join('.iocbio','fft_tasks','%s_wisdom_data.txt' % (fftw.__name__))
            if os.path.isfile(wisdom_file_name):
                if VERBOSE:
                    print 'Loading wisdom from file %r' % (wisdom_file_name)
                try:
                    fftw.import_wisdom_from_file(wisdom_file_name)
                except IOError, msg:
                    print 'Failed to load wisdom from file %r: %s' % (wisdom_file_name, msg)
                    continue
                _cache.append(wisdom_file_name)
                FFTTasks._wisdoms[wisdom_file_name] = fftw.export_wisdom_to_string()
    @staticmethod
    def save_wisdoms(_cache=[]):
        """Save fftw wisdom to a disk.

        See also
        --------
        iocbio.ops.fft_tasks, load_wisdom
        """
        if _cache:
            return
        try:
            import fftw3
        except ImportError, msg:
            fftw3 = None
            print 'FFTTasks.load_wisdoms: %s' % (msg)
        try:
            import fftw3f
        except ImportError, msg:
            fftw3f = None
            print 'FFTTasks.load_wisdoms: %s' % (msg)
        import atexit
        for fftw in [fftw3, fftw3f]:
            if fftw is None:
                continue
            wisdom_file_name = os.path.join('.iocbio','fft_tasks','%s_wisdom_data.txt' % (fftw.__name__))
            dirpath = os.path.dirname(wisdom_file_name)
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath)
            def save_wisdom(fftw=fftw, wisdom_file_name=wisdom_file_name):
                wisdom = FFTTasks._wisdoms.get(wisdom_file_name)
                if wisdom is not None and wisdom==fftw.export_wisdom_to_string():
                    #print 'Wisdom for %s has not changed, saving is not needed.' % (fftw.__name__)
                    return
                if VERBOSE:
                    print 'Saving wisdom to file %r' % (wisdom_file_name)
                try:
                    fftw.export_wisdom_to_file (wisdom_file_name)
                except Exception, msg:
                    print 'Failed to export wisdom to file %r: %s' % (wisdom_file_name, msg)
            atexit.register(save_wisdom)
            _cache.append(wisdom_file_name)
            save_wisdom()

    flops_cache = {}

    @classmethod
    def get_optimal_fft_size(cls, size, return_speedup=False, max_nof_tries=None):
        """Compute optimal FFT size from a given size.

        Usually optimal FFT size is a power of two but on the other
        hand, achieving power of two may be memory expensive and there
        may exist sizes that give more efficient FFT computation.  For
        example, if the input size is 65 then extending the FFT size
        to 128 is less efficient compared to extending the size, say,
        to 66.

        The method runs number of fft transforms (up-to the next power
        of 2) and used the actual FLOPS for finding optimal FFT
        size. So, the initial call may take some time but the results
        will be cached for subsequent calls. Note that if size>1024
        then the time spent of computing fft up to sizes <=2048 can be
        considerable. To restrict this optimization, specify
        ``max_nof_tries``.

        Parameters
        ----------
        size : int
          Specify estimate for FFT size.
        return_speedup : bool
          When True then return speed up factor.

        Returns
        -------
        optimal_size : int
          Optimal FFT size.
        speedup : float
          Speed up factor of using optimal size. Returned only if ``return_speedup`` is True.

        See also
        --------
        iocbio.ops.fft_tasks
        """
        if size==2**int(numpy.log2(size)):
            if return_speedup:
                return size, 1
            return size
        try:
            import fftw3f as fftw
        except ImportError:
            import fftw3 as fftw
        max_size = 2**int(numpy.log2(size)+1)
        if max_nof_tries is not None:
            max_size = min (size+max_nof_tries, max_size)
        min_flops = 1e9
        optimal_size = size
        flops_cache = cls.flops_cache
        for sz in range (size, max_size + 1):
            flops = flops_cache.get(sz)
            if flops is None:
                cache = numpy.empty((sz,), dtype=numpy.complex64)
                plan = fftw.Plan(cache, cache, direction='forward', flags=['estimate'])
                iplan = fftw.Plan(cache, cache, direction='backward', flags=['estimate'])
                flops = sum(plan.get_flops())+ sum(iplan.get_flops())
                fftw.destroy_plan (plan)
                fftw.destroy_plan (iplan)
                flops_cache[sz] = flops
            if flops < min_flops:
                min_flops = flops
                optimal_size = sz
        if return_speedup:
            return optimal_size, flops_cache[size] / flops_cache[optimal_size]
        return optimal_size

    def __init__(self, shape, float_type=None, options = None):
        """ Construct an instance of FFTTasks.

        Parameters
        ----------
        shape : tuple
          Specify array shape for FFT.
        float_type : {None, 'single', 'double'}
          Specify floating point type.
        options : {None, `iocbio.utils.Options`}
          Specify command line options:
            options.float_type
            options.fftw_plan_flags
            options.fftw_threads

        See also
        --------
        iocbio.ops.fft_tasks
        """
        if options is None:
            options = Options()
        flags = [options.get(fftw_plan_flags='estimate')]
        if float_type is None:
            float_type = options.get(float_type='single')

        self.load_wisdoms()

        self.shape = shape
        self.float_type = float_type

        threads = getattr(options, 'fftw_threads', 1)

        if float_type=='single':
            import fftw3f as fftw # hint: on failure to import select double float type
            cache = numpy.empty(shape, numpy.complex64)
            self.float_dtype = numpy.float32
            self.complex_dtype = numpy.complex64
        elif float_type=='double':
            import fftw3 as fftw
            cache = numpy.empty(shape, numpy.complex128)
            self.float_dtype = numpy.float64
            self.complex_dtype = numpy.complex128
        else:
            raise NotImplementedError (`float_type`)

        self._cache = cache
        self.fftw = fftw

        if VERBOSE:
            print 'Computing fftw wisdom (flags=%s, threads=%s, shape=%s, float=%s),'\
                ' be patient, it may take a while..'\
            % (flags, threads, shape, float_type), 
        self._fft_plan = fftw.Plan(cache, cache, direction='forward', flags=flags, nthreads=threads)
        self._ifft_plan = fftw.Plan(cache, cache, direction='backward', flags=flags, nthreads=threads)
        if VERBOSE:
            print 'done'

        self.save_wisdoms()

        self.convolve_kernel_fourier = None

    def clear(self):
        del self._fft_plan
        del self._ifft_plan
        del self._cache

    def fft(self, data):
        """Compute FFT of data.

        Parameters
        ----------
        data : :numpy:`ndarray`
          Input data of the same shape as ``task.shape``.

        Returns
        -------
        data_f : :numpy:`ndarray`
          Fourier transform of data.

        See also
        --------
        iocbio.ops.fft_tasks, ifft
        """
        cache = self._cache
        cache[:] = data
        self._fft_plan.execute()
        return cache.copy()

    def ifft(self, data, asreal=False):
        """Compute inverse FFT of data.

        Parameters
        ----------
        data : :numpy:`ndarray`
          Input data of the same shape as ``task.shape``.
        asreal : bool
          Return real part of the result.

        Returns
        -------
        data_if : :numpy:`ndarray`
          Inverse Fourier transform of data.

        See also
        --------
        iocbio.ops.fft_tasks, fft
        """
        cache = self._cache
        cache[:] = data
        self._ifft_plan.execute()
        if asreal:
            return cache.real / mul_seq(cache.shape)
        return cache / mul_seq(cache.shape)

    def set_convolve_kernel(self, kernel):
        """ Set convolve kernel.

        Parameters
        ----------
        kernel : :numpy:`ndarray`
          Specify kernel for the `convolve` method.

        See also
        --------
        iocbio.ops.fft_tasks, convolve, set_convolve_fourier_kernel
        """
        cache = self._cache
        cache[:] = kernel
        self._fft_plan.execute()
        self.set_convolve_fourier_kernel(cache.copy())

    def set_convolve_fourier_kernel(self, kernel_f):
        """ Set convolve kernel in Fourier transform.

        Parameters
        ----------
        kernel_f : :numpy:`ndarray`
          Specify kernel in Fourier form for the `convolve` method.

        See also
        --------
        iocbio.ops.fft_tasks, convolve, set_convolve_kernel
        """
        assert kernel_f.shape==self.shape,`kernel_f.shape, self.shape`
        kernel_f = kernel_f.astype(self.complex_dtype)
        self.convolve_kernel_fourier = kernel_f
        self.convolve_kernel_fourier_normal = kernel_f / mul_seq(kernel_f.shape)
        self.convolve_kernel_fourier_conj = kernel_f.conj()

    def convolve(self, data, inplace=True):
        """Compute convolution of data and convolve kernel.

        Parameters
        ----------
          data : :numpy:`ndarray`
            Specify data to be convolved with kernel. Kernel must
            be specified with `set_convolve_kernel` methods.

        Returns
        -------
          result : :numpy:`ndarray`
            The result of convolution.

        See also
        --------
        iocbio.ops.fft_tasks, set_convolve_kernel, set_convolve_fourier_kernel
        """
        kernel_f = self.convolve_kernel_fourier_normal
        if kernel_f is None:
            raise TypeError ('Convolve kernel not specified')
        cache = self._cache
        if not inplace:
            orig_cache = cache.copy()
        cache[:] = data
        self._fft_plan.execute()
        cache *= kernel_f
        self._ifft_plan.execute()
        result = cache.real.copy()
        if not inplace:
            self._cache[:] = orig_cache
        return result

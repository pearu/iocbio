
import os
import numpy

from ..utils import mul_seq

class FFTTasks:
    """ Holds optimized cache for Fourier transforms
    and implements various computational tasks using the cache.
    """

    _wisdoms = {}

    @staticmethod
    def load_wisdoms(_cache=[]):
        if _cache:
            return
        import fftw3
        import fftw3f
        for fftw in [fftw3, fftw3f]:
            wisdom_file_name = os.path.join('.ioc','fft_tasks','%s_wisdom_data.txt' % (fftw.__name__))
            if os.path.isfile(wisdom_file_name):
                print 'Loading wisdom from file %r' % (wisdom_file_name)
                fftw.import_wisdom_from_file(wisdom_file_name)
                _cache.append(wisdom_file_name)
                FFTTasks._wisdoms[wisdom_file_name] = fftw.export_wisdom_to_string()
    @staticmethod
    def save_wisdoms(_cache=[]):
        if _cache:
            return
        import fftw3
        import fftw3f
        import atexit
        for fftw in [fftw3, fftw3f]:
            wisdom_file_name = os.path.join('.ioc','fft_tasks','%s_wisdom_data.txt' % (fftw.__name__))
            dirpath = os.path.dirname(wisdom_file_name)
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath)
            def save_wisdom(fftw=fftw, wisdom_file_name=wisdom_file_name):
                wisdom = FFTTasks._wisdoms.get(wisdom_file_name)
                if wisdom is not None and wisdom==fftw.export_wisdom_to_string():
                    #print 'Wisdom for %s has not changed, saving is not needed.' % (fftw.__name__)
                    return
                print 'Saving wisdom to file %r' % (wisdom_file_name)
                fftw.export_wisdom_to_file (wisdom_file_name)
            atexit.register(save_wisdom)
            _cache.append(wisdom_file_name)
            save_wisdom()
    flops_cache = {}

    @classmethod
    def get_optimal_fft_size(cls, size):
        if size==2**int(numpy.log2(size)):
            return size
        import fftw3f as fftw
        max_size = 2**int(numpy.log2(size)+1)
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
        return optimal_size

    def __init__(self, shape, float_type=None, options = None):

        if options is None:
            flags = ['estimate']
        else:
            flags = [options.fftw_plan_flags]

        if float_type is None:
            float_type = 'single'

        self.load_wisdoms()

        self.shape = shape
        self.float_type = float_type

        threads = getattr(options, 'fftw_threads', 1)

        if float_type=='single':
            import fftw3f as fftw
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

        print 'Computing fftw wisdom (flags=%s, threads=%s), be patient, it may take a while..' % (flags, threads), 
        self._fft_plan = fftw.Plan(cache, cache, direction='forward', flags=flags, nthreads=threads)
        self._ifft_plan = fftw.Plan(cache, cache, direction='backward', flags=flags, nthreads=threads)
        print 'done'

        self.save_wisdoms()

        self.convolve_kernel_fourier = None

    def fft(self, data):
        cache = self._cache
        cache[:] = data
        self._fft_plan.execute()
        return cache.copy()

    def ifft(self, data):
        cache = self._cache
        cache[:] = data
        self._ifft_plan.execute()
        return cache.real / mul_seq(cache.shape)

    def set_convolve_kernel(self, kernel):
        cache = self._cache
        cache[:] = kernel
        self._fft_plan.execute()
        self.set_convolve_fourier_kernel(cache.copy())

    def set_convolve_fourier_kernel(self, kernel_f):
        kernel_f = kernel_f.astype(self.complex_dtype)
        self.convolve_kernel_fourier = kernel_f
        self.convolve_kernel_fourier_normal = kernel_f / mul_seq(kernel_f.shape)
        self.convolve_kernel_fourier_conj = kernel_f.conj()

    def convolve(self, data):
        kernel_f = self.convolve_kernel_fourier_normal
        #kernel_f = self.convolve_kernel_fourier
        if kernel_f is None:
            raise TypeError ('Convolve kernel not specified')
        cache = self._cache
        cache[:] = data
        self._fft_plan.execute()
        cache *= kernel_f
        self._ifft_plan.execute()
        return cache.real.copy()

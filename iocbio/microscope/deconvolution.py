"""Provides deconvolve function.

Example
-------

The following example illustrates the deconvolution of convolved
test data that is degraded with Poisson noise::

  import numpy
  import scipy.stats
  from iocbio.ops import convolve
  from iocbio.microscope.deconvolution import deconvolve
  from iocbio.io import ImageStack

  kernel = numpy.array([0,1,3,1,0])
  test_data = numpy.array([0, 0,0,0,2, 0,0,0,0, 0,1,0,1, 0,0,0])*50
  data = convolve(kernel, test_data)
  degraded_data = scipy.stats.poisson.rvs(numpy.where(data<=0, 1e-16, data)).astype(data.dtype)
  
  psf = ImageStack(kernel, voxel_sizes = (1,))
  stack = ImageStack(degraded_data, voxel_sizes = (1,))
  
  deconvolved_data = deconvolve(psf, stack).images

.. image:: ../_static/deconvolve_poisson_1d.png
  :width: 60%

Module content
--------------
"""

from __future__ import division
__autodoc__ = ['deconvolve', 'Deconvolve', 'DeconvolveRLPoisson']
__all__ = ['deconvolve']

import os
import sys
from ..io import ImageStack
from ..utils import ProgressBar, encode, tostr, expand_to_shape, contract_to_shape
from ..utils import mul_seq, float2dtype, Options
from .psf import normalize_uint8, discretize
from ..io import RowFile
import time
import numpy
import numpy as np
from .dv import dv
from .snr import estimate_snr
from scipy import fftpack
from scipy import ndimage
import scipy.stats
import numpy.testing.utils as numpy_utils

try:
    from . import ops_ext
except ImportError, msg:
    print msg

from ..ops import FFTTasks


def get_coherent_images(psf, stack, dtype):
    """
    Return PSF and stack images so that they have
      - same orientation and voxel sizes
      - same fft-optimal shape
      - same floating point type
    and
      - the center of PSF is shifted to origin
      - PSF is normalized such that convolve (PSF, 1) = 1
    """
    psf_angle = psf.get_rotation_angle() or 0
    stack_angle = stack.get_rotation_angle() or 0

    psf_voxels = psf.get_voxel_sizes()
    stack_voxels = stack.get_voxel_sizes()

    psf_images = psf.images
    stack_images = stack.images

    if psf_angle != stack_angle:
        rotation = psf_angle - stack_angle
        psf_images = ndimage.rotate(psf_images, rotation, axes=(-1,-2))
        print 'PSF was rotated by', rotation

    if psf_voxels != stack_voxels:
        zoom_factors = tuple([a/b for a,b in zip(psf_voxels, stack_voxels)])
        psf_images = ndimage.zoom (psf_images, zoom_factors)
        print 'PSF was zoomed by', zoom_factors
    
    max_shape = [max(a,b) for a,b in zip(psf_images.shape, stack_images.shape)]
    optimal_shape = tuple (map(FFTTasks.get_optimal_fft_size, max_shape))
    psf_images = expand_to_shape(psf_images, optimal_shape, dtype)
    stack_images = expand_to_shape(stack_images, optimal_shape, dtype)

    psf_images = fftpack.fftshift(psf_images)
    psf_images /= psf_images.sum()

    return psf_images, stack_images

def fourier_sphere(shape, diameters, eps = 1e-3):
    """ Return a Fourier transform of an ellipsoid with diameters.
    eps is accuracy parameter. The results of this functions will
    be cached to .iocbio/deconvolve/fourier_sphere_*.tif file and
    reused is subsequent calls.
    """
    cache_fn = os.path.join('.iocbio','deconvolve', 
                            'fourier_sphere_'+encode('shape=%s, diameters=%s, eps=%s' % (shape, diameters, eps))+'.tif')
    if os.path.isfile (cache_fn):
        print 'Using cached Fourier transform of a sphere:',cache_fn
        images = ImageStack.load(cache_fn).images
        assert images.shape==shape,`images.shape, shape`
        return images

    def write_func (fmt, *args):
        sys.stdout.write (fmt % args)
        sys.stdout.flush()

    print 'Computing Fourier transform of a sphere, be patient, it may take a quite while .. '
    images = ops_ext.fourier_sphere(shape, diameters, eps, write_func)
    print 'done'
    ImageStack(images).save (cache_fn)
    return images

def thr(a, b):
    return np.abs(a - b).sum() / abs(float(b.sum()))

def mse(a, b):
    return ((a-b)**2).mean()

def idiv(a, b):
    return (a * numpy.log(a/b) - a + b).mean()


def hasharr(arr):
    import hashlib
    return hashlib.sha1 (arr).hexdigest()

class Deconvolve(FFTTasks):
    """
    Base class for deconvolution worker classes.

    Deconvolve classes must implement the following methods:
      compute_estimate

    See also
    --------
    __init__, iocbio.ops.FFTTasks
    """

    def __init__(self, shape, options = None):
        """Construct an instance of Deconvolve.

        Parameters
        ----------
        shape : tuple
          Specify array shape for deconvolution algorithm.
        options : {None, `iocbio.utils.Options`}
          Specify command line options.
          See :ref:`iocbio-deconvolve` for options explanation.

        See also
        --------
        iocbio.ops.FFTTasks
        """
        FFTTasks.__init__ (self, shape, options = options)
        self.options = options
        self.convergence_epsilon = options.get(convergence_epsilon=0.05)

        self.cache_dir = None

        self.set_save_data(None, shape, self.float_dtype)
        self.lambda_ = options.get(rltv_lambda=0.0)
        self.count = None

    def get_suffix(self):
        return '_deconvolved'

    def set_cache_dir (self, cache_dir):
        """Set working directory.
        """
        self.cache_dir = cache_dir

    def set_save_data(self, pathinfo, data_shape, data_type):
        """Set data for saving results.
        """
        self.save_pathinfo = pathinfo
        self.save_data_shape = data_shape
        self.save_data_type = data_type

    def set_test_data(self):
        """Set test data.

        The input data is assumed to be a smooth field and the test
        data will be degraded (convolved and with Poisson noise) input
        data.
        """
        self.test_data = None
        options = self.options
        snr = None
        if options.get(degrade_input=False):
            self.test_data = self.data

            degraded_path = os.path.join(self.cache_dir, 'degraded.tif')

            print 'Adjusting SNR of input data..',
            snr = options.get(degrade_input_snr=0.0)
            data = self.convolve(self.test_data)
            data = np.where(data<=0, 1e-16, data)
            max_data = data.max()
            if snr==0.0:
                snr = numpy.sqrt(max_data)
            else:
                coeff = snr*snr/max_data
                data *= coeff
                self.test_data *= coeff
                print 'scaling test data by %r' % (coeff),
            print 'SNR=',snr,
            print 'done'

            if options.get(first_estimate='input image')=='last result' and os.path.isfile(degraded_path):
                print 'Loading degraded image.'
                self.data = ImageStack.load(degraded_path).images
            else:
                print 'Degrading image with Poisson noise..',
                import scipy.stats
                self.data = scipy.stats.poisson.rvs(data).astype(data.dtype)
                print 'done.'
                print 'Saving degraded image.'
                self.save(self.data, 'degraded.tif', True)
            d4 = ops_ext.kullback_leibler_divergence(self.data.astype(numpy.float64), data.astype (numpy.float64), 5.0)
            print 'Kullback-Leibler divergence of degraded image', d4, ' (should be close to 1/2)'

        if snr is None:
            snr = estimate_snr(self.data)
        print 'Input image has signal-to-noise ratio', snr
        print 'Suggested RLTV regularization parameter: %s[blocky]..%s[honeycomb]' % (43/snr, 60/snr)
        self.snr = snr

    def compute_estimate(self, estimate):
        """
        Update estimate array in-place and return (e,s,u,n) where

        Parameters
        ----------
        estimate : :numpy:`ndarray`
          Last estimate array. The method updates this estimate array
          in-place.

        Returns
        -------
        e : float
          Sum of exact photons.
        s : float
          Sum of stable photons.
        u : float
          Sum of unstable photons.
        n : float
          Sum of negative photons
        
        See also
        --------
        Deconvolve
        """
        print '%s must implement compute_estimate method' % (self.__class__.__name__)
        return 0, estimate.sum(), 0, 0
    
    def save (self, estimate, filename, use_estimate_shape=False):
        """ Save estimate to file.

        Parameters
        ----------
        estimate : :numpy:`ndarray`
        filename : str
        use_estimate_shape : bool
          If False then task shape will be used.
        """
        if self.cache_dir is None:
            return
        f = os.path.join(self.cache_dir, filename)
        ImageStack(contract_to_shape(estimate, 
                                     estimate.shape if use_estimate_shape else self.save_data_shape, 
                                     estimate.dtype),
                   self.save_pathinfo,
                   options = self.options).save(f)

    def deconvolve(self):
        """ Execute deconvolution iteration and return estimate.
        """
        options = self.options
        save_intermediate_results = options.get(save_intermediate_results=False)

        data_to_save = ('count', 't', 'mn', 'mx', 'tau1', 'tau2', 'leak', 'e', 's', 'u', 'n','u_esu', 'mse', 'mem',
                        'klic')

        data_file_name = os.path.join(self.cache_dir, 'deconvolve_data.txt')

        input_data = numpy.array(self.data, dtype=self.float_dtype)
        count = -1
        append_data_file = False
        first_estimate = options.get(first_estimate='input image')
        if first_estimate=='input image':
            estimate = input_data.copy()
        elif first_estimate=='convolved input image':
            estimate = self.convolve(input_data)
        elif first_estimate=='2x convolved input image':
            estimate = self.convolve(self.convolve(input_data))
        elif first_estimate=='last result':
            if os.path.isfile(data_file_name):
                data_file = RowFile(data_file_name)
                data = data_file.read()
                data_file.close()
                counts = map (int, data['count'])
                count = counts[-1]
                fn =os.path.join(self.cache_dir, 'result_%s.tif' % (count)) 
                append_data_file = True
                print 'Loading the last result from %r.' % (fn)
                stack = ImageStack.load(fn)
                estimate = numpy.array(stack.images, dtype=self.float_type)
                f = open(os.path.join(self.cache_dir, 'deconvolve_data_%s_%s.txt' % (counts[0],count)), 'w')
                fi = open(data_file_name)
                f.write(fi.read())
                fi.close()
                f.close()
            else:
                print 'Found no results in %r, using input image as estimate.' % (self.cache_dir)
                estimate = input_data.copy()
        else:
            raise NotImplementedError(`first_estimate`)

        prev_estimate = estimate.copy()
        initial_photon_count = input_data.sum()

        print 'Initial photon count: %.3f' % (initial_photon_count)
        print 'Initial minimum: %.3f' % (estimate.min())
        print 'Initial maximum: %.3f' % (estimate.max())
	
        max_count = options.get(max_nof_iterations=50)

        bar = ProgressBar(0, max_count, totalWidth=40, show_percentage=False)

        data_norm2 = (input_data**2).sum()
        if options.get(rltv_estimate_lambda=False) or options.get(rltv_compute_lambda_lsq=False):
            data_to_save += ('lambda_lsq',)

        if self.test_data is not None:
            data_to_save += ('mseo',)
            test_data_norm2 = (self.test_data**2).sum()

        data_file = RowFile(data_file_name,
                            titles = data_to_save,
                            append = append_data_file)
        data_file.comment('DeconvolveSysArgv: %s' % (' '.join(map(str, sys.argv))))

        if 'mseo' in data_file.extra_titles and 'mseo' not in data_to_save:
            data_to_save += ('mseo',)

        stop_message = ''
        stop = count >= max_count
        if stop:
            stop_message = 'The number of iterations reached to maximal count: %s' % (max_count)
        else:
            if save_intermediate_results:
                self.save(estimate, 'result_%sm1.tif' % (count+1))
        try:
            min_mse = 1e300
            min_mseo = 1e300
            min_tau = 1e300
            max_lambda = 0.0
            while not stop:
                count += 1
                self.count = count
                info_map = {}
                ittime = time.time()
                
                prev2_estimate = prev_estimate.copy()
                prev_estimate = estimate.copy()
  
                e,s,u,n = self.compute_estimate(estimate)
                
                info_map['E/S/U/N=%s/%s/%s/%s'] = int(e), int(s), int(u), int(n)
                photon_leak = 1.0 - (e+s+u)/initial_photon_count
                info_map['LEAK=%s%%'] = 100*photon_leak

                if 'leak' in data_to_save:
                    leak = 100*photon_leak

                if 'u_esu' in data_to_save:
                    u_esu = u/(e+s+u)
                    #info_map['U/ESU=%s'] = u_esu

                if 'mn' in data_to_save:
                    mn, mx = estimate.min(), estimate.max()

                if 'mse' in data_to_save:
                    eh = self.convolve(estimate, inplace=False)
                    mse = ((eh - input_data)**2).sum() / data_norm2
                    info_map['MSE=%s'] = mse

                if 'klic' in data_to_save:
                    klic = ops_ext.kullback_leibler_divergence(input_data, eh, 1.0)
                    info_map['KLIC=%s'] = klic

                if 'mseo' in data_to_save:
                    mseo = ((estimate - self.test_data)**2).sum() / test_data_norm2
                    info_map['MSEO=%s'] = mseo

                if 'tau1' in data_to_save:
                    tau1 = abs(estimate - prev_estimate).sum() / abs(prev_estimate).sum()
                    tau2 = abs(estimate - prev2_estimate).sum() / abs(prev2_estimate).sum()
                    info_map['TAU1/2=%s/%s'] = (tau1, tau2)

                if 'lambda_lsq' in data_to_save:
                    lambda_lsq = self.lambda_lsq
                    if lambda_lsq > max_lambda:
                        max_lambda = lambda_lsq
                    info_map['LAM/MX=%s/%s'] = lambda_lsq, max_lambda

                if 'mem' in data_to_save:
                    mem = int(numpy_utils.memusage()/2**20)
                    #info_map['MEM=%sMB'] = mem

                info_map['TIME=%ss'] = t = time.time() - ittime

                bar.updateComment(' '+', '.join([k%(tostr(info_map[k])) for k in sorted(info_map)]))
                bar(count)

                if 'mse' in data_to_save and mse < min_mse:
                    min_mse = mse
                    #self.save(discretize(estimate), 'deconvolved_%s_min_mse.tif' % (count))

                if 'mseo' in data_to_save and mseo < min_mseo:
                    min_mseo = mseo
                    #self.save(discretize(estimate), 'deconvolved_%s_min_mseo.tif' % (count))

                if save_intermediate_results:
                    self.save(estimate, 'result_%s.tif' % (count))

                # Stopping criteria:
                stop = True
                if abs(photon_leak) > 0.2:
                    stop_message = 'Photons leak is too large: %.3f%%>20%%' % (photon_leak*100)
                elif not u and not int (n):
                    stop_message = 'The number of non converging photons reached to zero.'
                elif count >= max_count:
                    stop_message = 'The number of iterations reached to maximal count: %s' % (max_count)
                elif 'tau1' in data_to_save and tau1 <= float(options.get(rltv_stop_tau=0.0)):
                    stop_message = 'Desired tau-threshold achieved'
                else:
                    stop = False

                exec 'data_file.write(%s)' % (', '.join (data_to_save))
                if not save_intermediate_results and stop:
                    self.save(estimate, 'result_%s.tif' % (count))

        except KeyboardInterrupt:
            stop_message = 'Iteration was interrupted by user.'

        print
        bar.updateComment (' '+stop_message)
        bar(count)
        print

        data_file.close()

        return estimate

class DeconvolveRLPoisson (Deconvolve):
    """
    Worker class for deconvolving stack images against PSF assuming Poisson noise.

    See also
    --------
    Deconvolve
    """

    def __init__(self, psf_images, stack_images, voxel_sizes, options):

        if psf_images is not None:
            assert psf_images.shape==stack_images.shape,`psf_images.shape, stack_images.shape`

        Deconvolve.__init__(self, stack_images.shape, options)

        if psf_images is not None:
            self.set_convolve_kernel(psf_images)

        if stack_images.min() < 0:
            print 'Cutting negative values'
            stack_images = numpy.where (stack_images < 0, 0, stack_images)

        self.test_data = None
        self.data = stack_images.astype(self.float_dtype)
        self.voxel_sizes = [s*1e9 for s in voxel_sizes]
        self.lambda_lsq = None
        self.lambda_lsq_coeff = None

    def get_suffix(self):
        if self.options.get(rltv_estimate_lambda=False):
            return '_deconvolved_mul'
        return '_deconvolved_mul_l%s' % (self.lambda_)
    
    def compute_estimate(self, estimate):
        options = self.options
        psf_f = self.convolve_kernel_fourier
        adj_psf_f = self.convolve_kernel_fourier_conj
        cache = self._cache

        # Execute: cache = convolve(PSF, estimate), non-normalized
        cache[:] = estimate
        self._fft_plan.execute()
        cache *= psf_f
        self._ifft_plan.execute()

        # Execute: cache = data/cache
        ops_ext.inverse_division_inplace(cache, self.data)

        # Execute: cache = convolve(PSF(-), cache), inverse of non-normalized
        self._fft_plan.execute()
        cache *= adj_psf_f
        self._ifft_plan.execute()
        # note that 1/mul_seq (cache.shape) factor cancels out

        dv_estimate = None
        if options.get(rltv_compute_lambda_lsq=False) or options.get(rltv_estimate_lambda=False):
            dv_estimate = ops_ext.div_unit_grad(estimate, self.voxel_sizes)
            lambda_lsq = ((1.0-cache.real)*dv_estimate).sum() / (dv_estimate*dv_estimate).sum()
            if self.lambda_lsq_coeff is None:
                lambda_lsq_coeff_path = os.path.join(self.cache_dir, 'lambda_lsq_coeff.txt')
                lambda_lsq_coeff = options.get(rltv_lambda_lsq_coeff=0.0)
                if lambda_lsq_coeff in [None, 0.0] and options.get(first_estimate='input image')=='last result' and os.path.isfile (lambda_lsq_coeff_path):
                    try:
                        lambda_lsq_coeff = float(open (lambda_lsq_coeff_path).read ())
                    except Exception, msg:
                        print 'Failed to read lambda_lsq_coeff cache: %s' % (msg)
                if lambda_lsq_coeff == 0.0:
                    # C * lambda_0 = 50/SNR
                    lambda_lsq_coeff = 50.0/self.snr/lambda_lsq
                if lambda_lsq_coeff < 0:
                    print 'Negative lambda_lsq, skip storing lambda_lsq_coeff'
                else:
                    self.lambda_lsq_coeff = lambda_lsq_coeff
                    f = open(lambda_lsq_coeff_path, 'w')
                    f.write(str (lambda_lsq_coeff))
                    f.close()
                print 'lambda-opt=',50.0/self.snr
                print 'lambda-lsq-0=',lambda_lsq
                print 'lambda-lsq-coeff=', lambda_lsq_coeff
            else:
                lambda_lsq_coeff = self.lambda_lsq_coeff
            lambda_lsq *= lambda_lsq_coeff
            self.lambda_lsq = lambda_lsq
        elif self.lambda_:
            dv_estimate = ops_ext.div_unit_grad(estimate, self.voxel_sizes)

        if options.get(rltv_estimate_lambda=False):
            self.lambda_ = lambda_lsq
            cache /= (1.0 - lambda_lsq*dv_estimate)
        elif self.lambda_:
            cache /= (1.0 - self.lambda_*dv_estimate)
        else: # TV is disabled
            pass

        # Execute: estimate *= cache.real
        result = ops_ext.update_estimate_poisson(estimate, cache, self.convergence_epsilon)
        return result

class DeconvolveRLGauss (Deconvolve):
    """
    Worker class for deconvolving stack images against PSF assuming Gaussian noise.
    [EXPERIMENTAL]

    See also
    --------
    Deconvolve
    """
    def __init__(self, psf_images, stack_images, voxel_sizes, options):

        if psf_images is not None:
            assert psf_images.shape==stack_images.shape,`psf_images.shape, stack_images.shape`

        Deconvolve.__init__(self, stack_images.shape, options)

        if psf_images is not None:
            self.set_convolve_kernel(psf_images)

        self.data = stack_images.astype(self.float_dtype)
        self.voxel_sizes = [s*1e9 for s in voxel_sizes]
        self.alpha = float(options.get(rltv_alpha=1.0))

        psf_f = self.convolve_kernel_fourier
        adj_psf_f = self.convolve_kernel_fourier_conj
        psf_adj_psf_f = self.psf_adj_psf_f = psf_f * adj_psf_f

        cache = self._cache
        cache[:] = self.data
        self._fft_plan.execute()
        cache *= psf_adj_psf_f
        self._ifft_plan.execute()
        self.psf_adj_data = cache.real / mul_seq(cache.shape)

    def get_suffix(self):
        if self.options.get(rltv_estimate_lambda=False):
            return '_deconvolved_add_a%s' % (self.alpha)
        return '_deconvolved_add_l%s_a%s' % (self.lambda_, self.alpha)

    def compute_estimate(self, estimate):
        cache = self._cache    
        cache[:] = estimate
        # Execute: cache = convolve(estimate, convolve(PSF, PSF(-)))
        self._fft_plan.execute()
        cache *= self.psf_adj_psf_f
        self._ifft_plan.execute()

        # Execute: cache = convolve(PSF(-), data) - f * cache
        ops_ext.inverse_subtraction_inplace(cache, self.psf_adj_data, 1.0/mul_seq (cache.shape))

        # Regularization:
        if self.options.get(rltv_estimate_lambda=False):
            dv_estimate = ops_ext.div_unit_grad(estimate, self.voxel_ratios)
            lambda_ = - (cache.real * dv_estimate).sum () / (dv_estimate*dv_estimate).sum()
            lambda_ *= options.get(rltv_lambda_lsq_coeff=1.0)
            self.lambda_ = lambda_
            cache += (lambda_/self.alpha) * dv_estimate
        elif self.lambda_:
            dv_estimate = ops_ext.div_unit_grad(estimate, self.voxel_ratios)
            cache += self.lambda_/self.alpha * dv_estimate
        else:
            pass

        # Execute: estimate += alpha * cache.real
        return ops_ext.update_estimate_gauss(estimate, cache, self.convergence_epsilon, self.alpha)

class DeconvolveRLPoissonSphere(DeconvolveRLPoisson):
    """
    Worker class for deconvolving stack images against sphere with diameter.

    See also
    --------
    DeconvolveRLPoisson
    """

    def __init__(self, diameter, stack_images, voxel_sizes, options):    
        DeconvolveRLPoisson.__init__ (self, None, stack_images, voxel_sizes, options)
        diameters = [diameter / h for h in voxel_sizes]
        sphere_ft = fourier_sphere(self.shape, diameters, 5e-5)
        self.set_convolve_fourier_kernel(sphere_ft)

    def get_suffix(self):
        if self.options.get(rltv_estimate_lambda=False):
            return '_deconvolved_sphere_mul'
        return '_deconvolved_sphere_mul_l%s' % (self.lambda_)

def deconvolve(psf, stack, working_dir = None, data_type = None,
               options = None):
    """Deconvolve stack of images against given PSF.

    Parameters
    ----------
    psf : `iocbio.io.image_stack.ImageStack`
      PSF
    stack : `iocbio.io.image_stack.ImageStack`
      Scanned images.
    working_dir : {None, str}
      Directory name where to save intermediate results.
      When None then a temporary directory will be created.
    data_type : {None, str}
      Desired data type of deconvolution estimate to be returned.
    options : {None, `iocbio.utils.Options`}
      See :ref:`iocbio-deconvolve` for information about options.
      The following options attributes are used: 
      float_type, apply_window, rltv_algorithm_type, degrade_data,
      first_estimate, rltv_stop_tau, save_intermediate_results

    Returns
    -------
    estimate : `iocbio.io.image_stack.ImageStack`
      Last deconvolution estimate. See ``working_dir`` for other estimates.

    See also
    --------
    iocbio.microscope.deconvolution
    """
    options = Options(options)
    if working_dir is None:
        import tempfile
        working_dir = tempfile.mkdtemp('-iocbio.deconvolve')

    if data_type is None:
        data_type = stack.images.dtype

    dtype = float2dtype(options.get(float_type='single'))

    phf0, data = get_coherent_images(psf, stack, dtype)

    if options.get(apply_window=False):
        background = (stack.pathinfo.get_background() or [0,0])[0]
        voxel_sizes = stack.get_voxel_sizes()
        smoothness = int(options.get(smoothness=1))
        window_width = options.get(window_width=None)
        if window_width is None:
            dr = stack.get_lateral_resolution()
            dz = stack.get_axial_resolution()
            if dr is None or dz is None:
                window_width = 3.0
                scales = tuple([s/(window_width*min(voxel_sizes)) for s in voxel_sizes])
            else:
                print 'lateral resolution: %.3f um (%.1f x %.1f px^2)' % (1e6*dr, dr/voxel_sizes[1], dr/voxel_sizes[2])
                print 'axial resolution: %.3f um (%.1fpx)' % (1e6*dz, dz / voxel_sizes[0])
                vz,vy,vx = voxel_sizes
                m = 0.3
                scales = (m*vz/dz, m*vy/dr, m*vx/dr)
        else:
            scales = tuple([s/(window_width*min(voxel_sizes)) for s in voxel_sizes])
        print 'Window size in pixels:', [1/s for s in scales]
        from iocbio.ops.apply_window_ext import apply_window_inplace
        apply_window_inplace(data, scales, smoothness, background)


    mode = options.get(rltv_algorithm_type='multiplicative').lower()
    Cls = dict(multiplicative=DeconvolveRLPoisson,
               additive = DeconvolveRLGauss)[mode]

    task = Cls(phf0, data, stack.get_voxel_sizes(), options)

    task.set_cache_dir(working_dir)
    task.set_save_data(stack.pathinfo, data.shape, data_type)
    task.set_test_data()

    estimate = task.deconvolve()

    estimate = contract_to_shape(estimate, 
                                 stack.images.shape, 
                                 data_type)

    return ImageStack(estimate, stack.pathinfo, suffix=task.get_suffix(),
                      options = options)

def deconvolve_sphere(psf, diameter, deconvolve_dir,
                      data_type = None,
                      options = None,
                      **kws):
    """
    Deconvolve given psf against a sphere with diameter [meter].

    See also
    --------
    deconvolve
    """
    if data_type is None:
        data_type = psf.images.dtype

    shape = tuple(map (FFTTasks.get_optimal_fft_size, psf.images.shape))

    background = (psf.pathinfo.get_background() or [0,0])[0]
    #assert background <= psf.images.min(),`background, psf.images.min()`

    psf_images = expand_to_shape(psf.images, shape) - psf.images.min()

    ImageStack (psf_images, psf.pathinfo).save('tmp.tif')

    print background, psf_images.min (), psf_images.max (), psf.images.max ()

    task = DeconvolveRLPoissonSphere (diameter, psf_images, psf.get_voxel_sizes(), options)
    task.set_cache_dir (os.path.join(deconvolve_dir,'ioc.deconvolve_sphere'))

    estimate = task.deconvolve()

    estimate = contract_to_shape(estimate, 
                                 psf.images.shape,
                                 data_type)

    return ImageStack(estimate, psf.pathinfo,  suffix=task.get_suffix(),
                      options = options)

def deconvolve_smooth(psf, stack, deconvolve_dir,
                      data_type = None,
                      options = None):
    """
    Smooth stack with given PSF: deconvolve (PSF, convolve(PSF, stack))

    See also
    --------
    deconvolve
    """
    if data_type is None:
        data_type = stack.images.dtype

    dtype = float2dtype(options.get(float_type='single'))

    phf0, data = get_coherent_images(psf, stack, dtype)

    task = DeconvolveRLPoisson(phf0, data, stack.get_voxel_sizes(), options)
    task.set_cache_dir (os.path.join(deconvolve_dir,'iocbio.deconvolve_smooth'))

    orig_data = data
    task.data = task.convolve(data)

    estimate = task.deconvolve()

    estimate = contract_to_shape(estimate, 
                                 stack.images.shape, 
                                 data_type)

    return ImageStack(estimate, stack.pathinfo,  suffix=task.get_suffix(),
                      options = options)


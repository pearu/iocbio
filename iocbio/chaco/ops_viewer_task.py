
__all__ = ['OpsViewerTask']
import time
import numpy

from enthought.traits.api import Button, Any, Bool, Tuple, Enum, Float, Int, Instance
from enthought.traits.ui.api import View, VGroup, Item, HGroup, HSplit, Group, TupleEditor

from iocbio.ops.fft_tasks import FFTTasks
from iocbio.utils import Options
from iocbio.ops.local_extrema_ext import local_minima, local_maxima

from .base_viewer_task import BaseViewerTask
from .array_data_source import ArrayDataSource
from .timeit import TimeIt

_fft_worker_cache = [None, None]

from threading import Thread

class RunnerThread(Thread):

    def __init__ (self, func, args, finish_func):
        Thread.__init__(self)
        self.func = func
        self.args = args
        self.want_to_abort = False
        self.finish_func = finish_func

    def run(self):
        print 'started run'
        try:
            result = self.func (*self.args)
        except Exception, msg:
            print 'run exception: %s' % (msg)
            self.finish_func(msg)
        else:
            self.finish_func(result)
        print 'stopped run'

class OpsViewerTask(BaseViewerTask):

    boundary_map = dict (constant=0, finite=1, periodic=2, reflective=3)
    boundary = Enum(sorted(boundary_map))
    threshold = Float(0)
    max_nof_points = Int(100)
    runner_thread = Instance(RunnerThread)

    compute_fft_button = Button('Compute FFT')
    compute_ifft_button = Button('Compute IFFT')
    clear_fft_worker_button = Button('Clear FFT worker cache')
    find_local_maxima_button = Button('Find local maxima')
    find_local_minima_button = Button('Find local minima')
    clear_points_button = Button('Clear points')
    take_real_button = Button('Take real')
    take_imag_button = Button('Take imag')
    take_abs_button = Button('Take abs')
    discrete_gauss_blur_button = Button('Discrete Gauss blur')
    discrete_gauss_laplace_button = Button('Discrete Gauss Laplace')
    discrete_gauss_scales = Tuple (0.0,0.0,0.0)
    discrete_gauss_widths = Tuple (0.0,0.0,0.0)
    discrete_gauss_sizes = Tuple (0.0,0.0,0.0)
    stop_button = Button('Stop')

    traits_view = View (
        VGroup(
            HSplit(Item('compute_fft_button', show_label = False),
                   Item('compute_ifft_button', show_label = False),
                   Item('clear_fft_worker_button', show_label = False),
                   ),'_',
            HSplit(Item('boundary'), Item('threshold'),  Item ('max_nof_points'),
                   visible_when='is_real'),
            HSplit(Item('find_local_maxima_button', show_label = False),
                   Item('find_local_minima_button', show_label = False),
                   Item('clear_points_button', show_label = False),
                   Item('stop_button', show_label = False),
                   visible_when='is_real'
                   ),'_',
            HSplit(Item('take_real_button', show_label = False), 
                   Item('take_imag_button', show_label = False), 
                   Item('take_abs_button', show_label = False), 
                   visible_when='is_complex'),
            '_',
            Group(
            HSplit(Item('discrete_gauss_blur_button', show_label=False),
                   Item('discrete_gauss_laplace_button', show_label=False)),
            Item('discrete_gauss_sizes',  label='DG FWHM [um]', 
                 editor=TupleEditor (cols=3, labels=['Z','Y','X']),
                 ),
            Item('discrete_gauss_widths', label='DG FWHM [px]', 
                 editor=TupleEditor (cols=3, labels=['Z','Y','X']),
                 ),
            Item('discrete_gauss_scales', label='DG scales', 
                 editor=TupleEditor (cols=3, labels=['Z','Y','X']),
                 ),


            )
            ))

    @property
    def fft_worker(self):
        if _fft_worker_cache[0] != self.viewer.data.shape:
            timeit = TimeIt (self.viewer, 'creating FFT worker')
            _fft_worker_cache[0] = self.viewer.data.shape
            _fft_worker_cache[1] = FFTTasks(self.viewer.data.shape, options = Options(fftw_threads = 4))
            timeit.stop()
        return _fft_worker_cache[1]

    @fft_worker.deleter
    def fft_worker(self):
        timeit = TimeIt('removing FFT worker')
        _fft_worker_cache[0] = None
        _fft_worker_cache[1] = None

    def _clear_fft_worker_button_fired(self):
        fft_worker = self.fft_worker
        if fft_worker is not None:
            fft_worker.clear()
        del self.fft_worker

    def _discrete_gauss_scales_changed(self, old, new):
        if old==new: return
        self.discrete_gauss_widths = tuple([2.3548*t**0.5 for t in self.discrete_gauss_scales])

    def _discrete_gauss_widths_changed(self, old, new):
        if old==new: return
        self.discrete_gauss_scales = tuple([(w/2.3548)**2 for w in self.discrete_gauss_widths])
        self.discrete_gauss_sizes = tuple([w*s for w, s in zip(self.discrete_gauss_widths, self.viewer.voxel_sizes)])

    def _discrete_gauss_sizes_changed(self, old, new):
        if old==new: return
        self.discrete_gauss_widths = tuple([w/s for w, s in zip(self.discrete_gauss_sizes, self.viewer.voxel_sizes)])

    def _name_default (self):
        return 'Operations'

    def startup(self):
        # import fftw routines
        #self._fft_worker = FFTTasks(self.viewer.data.shape, options = Options(fftw_threads = 4))
        pass

    def _clear_points_button_fired(self):
        self.viewer.set_point_data([])
        self.viewer.reset()
        self.viewer.status = 'cleared point data'

    def _find_local_maxima_button_fired(self):
        data = self.viewer.data
        if not isinstance (data, numpy.ndarray):
            data = data[:] # tiffarray
        threshold = self.threshold
        boundary = self.boundary_map[self.boundary]
        timeit = TimeIt(self.viewer, 'computing local maxima')
        def cb(done, result, timeit=timeit):
            timeit.update('%.2f%% done, %s points sofar' % (done*100.0, len (result)))
            if len (result)>1000000:
                raise RuntimeError('too many points (more than billion), try blurring')
        try:
            l = local_maxima(data, threshold, boundary, cb)
        except Exception, msg:
            timeit.stop('failed with exception: %s' % (msg))
            raise
        timeit.stop()
        l.sort(reverse=True)
        self.viewer.set_point_data(l[:self.max_nof_points])
        self.viewer.reset()

    def compute_local_minima(self):
        data = self.viewer.data
        if not isinstance (data, numpy.ndarray):
            data = data[:] # tiffarray
        threshold = self.threshold
        boundary = self.boundary_map[self.boundary]
        timeit = TimeIt (self.viewer, 'computing local minima')
        def cb(done, result, timeit=timeit):
            timeit.update('%.2f%% done, %s points sofar' % (done*100.0, len (result)))
            if len (result)>1000000:
                raise RuntimeError('too many points (more than billion), try blurring')
        try:
            l = local_minima(data, threshold, boundary, cb)
        except Exception, msg:
            timeit.stop('failed with exception: %s' % (msg))
            raise
        timeit.stop()
        l.sort(reverse=False)
        self.viewer.set_point_data(l[:self.max_nof_points])
        self.viewer.reset()

    def _find_local_minima_button_fired(self):
        self.compute_local_minima ()
        #Thread(target=self.compute_local_minima).start()

    def _compute_fft_button_fired(self):

        data = self.viewer.data[:]
        timeit = TimeIt (self.viewer, 'computing FFT')
        fdata = self.fft_worker.fft(data)
        timeit.stop()
        self.viewer.add_data('FFT(%s)' % (self.viewer.selected_data), fdata)

    def _compute_ifft_button_fired(self):
        data = self.viewer.data[:]
        timeit = TimeIt (self.viewer, 'computing IFFT')
        fdata = self.fft_worker.ifft(data)
        timeit.stop()
        self.viewer.add_data('IFFT(%s)'% (self.viewer.selected_data), fdata)

    def _take_real_button_fired(self):
        timeit = TimeIt (self.viewer, 'computing REAL')
        data = self.viewer.data
        if not isinstance (data, numpy.ndarray):
            data = data[:] # tiffarray
        self.viewer.add_data('REAL(%s)'% (self.viewer.selected_data), data.real)

    def _take_imag_button_fired(self):
        timeit = TimeIt (self.viewer, 'computing IMAG')
        data = self.viewer.data
        if not isinstance (data, numpy.ndarray):
            data = data[:] # tiffarray
        self.viewer.add_data('IMAG(%s)'% (self.viewer.selected_data), data.imag)

    def _take_abs_button_fired(self):
        timeit = TimeIt (self.viewer, 'ABS')
        data = self.viewer.data
        if not isinstance (data, numpy.ndarray):
            data = data[:] # tiffarray
        self.viewer.add_data('ABS(%s)'% (self.viewer.selected_data), abs(data))

    def _discrete_gauss_blur_button_fired(self):
        data = self.viewer.data
        if not isinstance (data, numpy.ndarray):
            data = data[:] # tiffarray
        t0,t1,t2 = self.discrete_gauss_scales
        timeit = TimeIt (self.viewer, 'computing discrete gaussian blur')
        fdataname = 'FFT(%s)' % (self.viewer.selected_data)
        fdata = self.viewer.get_data(fdataname)
        if fdata is None:
            fdata = self.fft_worker.fft(data)
            self.viewer.add_data(fdataname, fdata)
        fkername = 'KernelDG[%s, %s, %s]' % (t0, t1, t2)
        fker = self.viewer.get_data(fkername)
        if fker is None:
            theta0 = 2*numpy.pi*numpy.fft.fftfreq(data.shape[0])
            theta1 = 2*numpy.pi*numpy.fft.fftfreq(data.shape[1])
            theta2 = 2*numpy.pi*numpy.fft.fftfreq(data.shape[2])
            fker0 = numpy.exp((numpy.cos(theta0)-1)*t0).reshape((data.shape[0],1,1))
            fker1 = numpy.exp((numpy.cos(theta1)-1)*t1).reshape((1,data.shape[1],1))
            fker2 = numpy.exp((numpy.cos(theta2)-1)*t2).reshape((1,1,data.shape[2]))
            fker = fker0 * fker1 * fker2
            self.viewer.add_data(fkername, fker)
        ffdata = fdata * fker
        result = self.fft_worker.ifft(ffdata)
        timeit.stop()
        self.viewer.add_data('DG[%s,%s,%s](%s)' % (t0,t1,t2,self.viewer.selected_data), result)

    def _discrete_gauss_laplace_button_fired(self):

        data = self.viewer.data
        if not isinstance (data, numpy.ndarray):
            data = data[:] # tiffarray
        t0,t1,t2 = self.discrete_gauss_scales
        timeit = TimeIt (self.viewer, 'computing discrete gaussian laplace')
        fdataname = 'FFT(%s)' % (self.viewer.selected_data)
        fdata = self.viewer.get_data(fdataname)
        if fdata is None:
            fdata = self.fft_worker.fft(data)
            self.viewer.add_data(fdataname, fdata)
        fkername = 'KernelLDG[%s, %s, %s]' % (t0, t1, t2)
        fker = self.viewer.get_data(fkername)
        if fker is None:
            theta0 = 2*numpy.pi*numpy.fft.fftfreq(data.shape[0])
            theta1 = 2*numpy.pi*numpy.fft.fftfreq(data.shape[1])
            theta2 = 2*numpy.pi*numpy.fft.fftfreq(data.shape[2])
            fker0 = t0 * numpy.exp((numpy.cos(theta0)-1)*t0)*(numpy.cos(theta0)-1.0)
            fker1 = t1 * numpy.exp((numpy.cos(theta1)-1)*t1)*(numpy.cos(theta1)-1.0)
            fker2 = t2 * numpy.exp((numpy.cos(theta2)-1)*t2)*(numpy.cos(theta2)-1.0)
            fker0 = fker0.reshape((data.shape[0],1,1))
            fker1 = fker1.reshape((1,data.shape[1],1))
            fker2 = fker2.reshape((1,1,data.shape[2]))
            fker = fker0 * fker1 * fker2
            self.viewer.add_data(fkername, fker)
        ffdata = fdata * fker
        result = self.fft_worker.ifft(ffdata)
        timeit.stop()
        self.viewer.add_data('LDG[%s,%s,%s](%s)' % (t0,t1,t2,self.viewer.selected_data), result)

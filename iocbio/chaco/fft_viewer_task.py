
__all__ = ['FFTViewerTask']
import numpy

from enthought.traits.api import Button, Any, Bool
from enthought.traits.ui.api import View, VGroup, Item
from .base_viewer_task import BaseViewerTask
from .array_data_source import ArrayDataSource

class FFTViewerTask(BaseViewerTask):


    compute_button = Button('Compute Fourier Transform')
    inverse = Bool(False)

    traits_view = View (
            Item('compute_button', show_label = False),
            Item('inverse'),
            )

    _fft_worker = Any

    def startup(self):
        # import fftw routines
        from iocbio.ops.fft_tasks import FFTTasks
        self._fft_worker = FFTTasks(self.viewer.data.shape)

    def _compute_button_fired(self):
        data = self.viewer.data
        if not isinstance (data, numpy.ndarray) or 1:
            data = data[:] # tiffarray
        # compute fft of data
        if self.inverse:
            fdata = self._fft_worker.ifft(data)
            self.viewer.add_data('IFFT', fdata)
        else:
            fdata = self._fft_worker.fft(data)
            self.viewer.add_data('FFT', fdata)
        #fdata[0,0,0] = 0
        #fdata = numpy.fft.fftshift (fdata)
        return
        
        fdata_source = ArrayDataSource(original_source = self.viewer.data_source,
                                       kind = self.viewer.data_source.kind,
                                       data = fdata)
        viewer = self.viewer.__class__(data_source = fdata_source)
        self.viewer.add_result(viewer)
        viewer.copy_tasks(self.viewer)

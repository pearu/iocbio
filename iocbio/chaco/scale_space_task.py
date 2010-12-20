
__all__ = ['FFTViewerTask']

import numpy

from enthought.traits.api import Button, Any, Array, Float
from enthought.traits.ui.api import View, VGroup, Item
from enthought.traits.ui.ui_editors.array_view_editor \
    import ArrayViewEditor
from enthought.chaco.api import Plot, ArrayPlotData, OverlayPlotContainer
from enthought.chaco.overlays.api import ContainerOverlay

from .base_viewer_task import BaseViewerTask
from .array_data_source import ArrayDataSource


class ScaleSpaceTask(BaseViewerTask):

    compute_button = Button('Compute Scale Space Data')

    scale = Float(0.0)
    results = Array

    traits_view = View (VGroup(
            Item('compute_button', show_label = False),
            Item('scale'),
            Item('results', editor = ArrayViewEditor (titles=['value', 'z', 'y', 'x'],
                                                      format = '%.1f', font='Arial 8'))
            ))

    _fft_worker = Any

    def startup(self):
        # import fftw routines
        #from iocbio.ops.fft_tasks import FFTTasks
        #self._fft_worker = FFTTasks(self.viewer.data.shape)
        pass
    
    def _results_default (self):
        return numpy.array([[0,0,0,0]])

    def _compute_button_fired(self):
        data = self.viewer.data
        if not isinstance (data, numpy.ndarray):
            data = data[:] # tiffarray
        # compute fft of data
        
        from iocbio.ops.local_extrema_ext import local_maxima
        l = local_maxima(data, 0)
        print len (l)
        l.sort(reverse=True)
        self.results = numpy.array(l)
        self.viewer.set_point_data(l)
        return

        fdata = self._fft_worker.fft(data)
        fdata[0,0,0] = 0
        fdata = numpy.fft.fftshift (fdata)
        fdata_source = ArrayDataSource(original_source = self.viewer.data_source,
                                       kind = self.viewer.data_source.kind,
                                       data = fdata)
        viewer = self.viewer.__class__(data_source = fdata_source)
        self.viewer.add_result(viewer)
        viewer.copy_tasks(self.viewer)


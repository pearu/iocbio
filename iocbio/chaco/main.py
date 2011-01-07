import sys
from enthought.etsconfig.api import ETSConfig
print 'Using ETS_TOOLKIT=%s' % (ETSConfig.toolkit)
#ETSConfig.toolkit='wx' # wx, qt4, null

from enthought.traits.api import HasStrictTraits, Instance, DelegatesTo, List, Str, on_trait_change
from enthought.traits.ui.api import Handler, View, Item, Group, Action, ListEditor, Tabbed

from iocbio.chaco.base_data_source import BaseDataSource
from iocbio.chaco.base_data_viewer import BaseDataViewer
from iocbio.chaco.image_stack_viewer import ImageStackViewer
from iocbio.chaco.image_timeseries_viewer import ImageTimeseriesViewer
from iocbio.chaco.tiff_data_source import TiffDataSource
from iocbio.chaco.array_data_source import ArrayDataSource
from iocbio.chaco.box_selector_task import BoxSelectorTask
from iocbio.chaco.slice_selector_task import SliceSelectorTask
from iocbio.chaco.fft_viewer_task import FFTViewerTask
from iocbio.chaco.ops_viewer_task import OpsViewerTask
from iocbio.chaco.scale_space_task import ScaleSpaceTask
from iocbio.chaco.points_task import PointsTask
from iocbio.chaco.table_plot_task import TablePlotTask
from iocbio.utils import bytes2str

class ViewHandler (Handler):

    def object_has_data_changed (self, info):
        #print 'object_has_data_changed', info.object.has_data, info.object.data_source.kind
        if info.object.has_data:
            if info.object.data_source.kind=='image_stack':
                info.object.data_viewer = None
                viewer = ImageStackViewer(data_source = info.object.data_source)
                info.object.data_viewer = viewer
                viewer.add_task(SliceSelectorTask)
                viewer.add_task(BoxSelectorTask)
                viewer.add_task(OpsViewerTask)
                viewer.add_task(PointsTask)
                return
            if info.object.data_source.kind=='image_timeseries':
                info.object.data_viewer = None
                viewer = ImageTimeseriesViewer(data_source = info.object.data_source)
                info.object.data_viewer = viewer
                viewer.add_task(TablePlotTask)
                #viewer.add_task(SliceSelectorTask)
                #viewer.add_task(BoxSelectorTask)
                #viewer.add_task(FFTViewerTask)
                return
            print '%s.object_has_data_changed: notimplemented kind=%r' % (self.__class__.__name__, info.object.data_source.kind)

    object_kind_changed = object_has_data_changed
    object_selected_channel_changed = object_has_data_changed

class ControlPanel (HasStrictTraits):
    
    data_source = Instance(BaseDataSource)
    data_viewer = Instance(BaseDataViewer)
    has_data = DelegatesTo('data_source')
    kind = DelegatesTo('data_source')
    selected_channel = DelegatesTo('data_source')
    memusage = DelegatesTo('data_viewer')
    status = DelegatesTo('data_viewer')
    statusbar = Str
    
    @on_trait_change('data_viewer.memusage, data_viewer.status')
    def update_statusbar(self):
        if hasattr (self, 'memusage'):
            memusage = bytes2str (self.memusage)
            self.statusbar = 'MEMUSAGE=%s STATUS: %s' % (memusage, self.status)

    traits_view = View (Tabbed(Item("data_source", style='custom', show_label=False, resizable=True),
                               Item('data_viewer', style='custom', resizable=True, show_label = False),
                               show_border = True, ),
                        buttons = ['OK'],
                        resizable = True,
                        handler = ViewHandler(),
                        height = 600,
                        width = 800,
                        statusbar = 'statusbar',

                        )

def analyze (file_name = ''):
    control_panel = ControlPanel(data_source = TiffDataSource(file_name=file_name))
    control_panel.configure_traits()
    return control_panel

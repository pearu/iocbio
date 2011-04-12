from __future__ import division

__all__ = ['ImageTimeseriesViewer']

import numpy

from enthought.traits.api import HasStrictTraits, Instance, Dict, Int, Bool, DelegatesTo, on_trait_change, Button, Any
from enthought.traits.ui.api import View, HSplit, VSplit, Item, VGrid, HGroup, Group, VGroup, Tabbed, RangeEditor
from enthought.enable.api import Component, ComponentEditor
from enthought.chaco.tools.api import ZoomTool, PanTool, ImageInspectorTool, ImageInspectorOverlay
from enthought.chaco.api import ArrayPlotData, Plot, GridContainer, PlotAxis
from enthought.chaco.tools.cursor_tool import BaseCursorTool
from enthought.chaco.default_colormaps import bone

from .base_data_source import BaseDataSource
from .base_data_viewer import BaseDataViewer

class ImageTimeseriesViewer(BaseDataViewer):
    
    plots = Dict

    plotdata = Instance(ArrayPlotData)
    image = Any
    
    time_index = Int
    time_points = Int
    time = Any

    traits_view = View(Tabbed (HSplit(VGroup(Item('plot', editor=ComponentEditor(), 
                                                  show_label = False,
                                                  resizable = True, label = 'View'),
                                             Item ("time_index", style='custom', editor=RangeEditor (low=0, high_name='time_points')),
                                             Item ("time", style='readonly'),
                                             ),
                                      Item('tasks', style='custom', show_label = False, label = 'Tasks'),
                                      ),
                               Item ('results', style='custom', show_label = False, label = 'Results'),
                               ),
                       )

    def _time_points_default (self):
        return self.data.shape[0]-1

    def _time_index_default (self):
        return 0
    
    def _time_default (self):
        return self.get_data_time(0, 0)

    def _plotdata_default(self):
        data = self.get_data_slice(0, 0)
        plotdata = ArrayPlotData()
        plotdata.set_data('xy', data) 
        return plotdata

    def _time_index_changed (self):
        self.select_xy_slice (self.time_index)

    def select_xy_slice(self, t):
        data = self.get_data_slice(t, 0)
        self.time = self.get_data_time(t, 0)
        self.plotdata.set_data('xy', data)
        self.image.invalidate_and_redraw()

    def reset (self):
        pass

    def redraw (self):
        self.image.invalidate_and_redraw()

    def get_plot(self):
        pixel_sizes = self.data_source.pixel_sizes
        shape = self.data.shape[1:]
        m = min(pixel_sizes)
        s = [int(d*sz/m) for d, sz in zip(shape, pixel_sizes)]
        plot_sizes = dict (xy = (s[1], s[0]))
        plot = Plot(self.plotdata, padding=30, fixed_preferred_size = plot_sizes['xy'],
                    )
        image = plot.img_plot('xy', colormap=bone)[0]
        image.overlays.append(ZoomTool(image))
        image.tools.append(PanTool(image, drag_button='right'))
        imgtool = ImageInspectorTool(image)
        image.tools.append(imgtool)
        overlay = ImageInspectorOverlay(component=image,
                                        bgcolor = 'white',
                                        image_inspector=imgtool)
        image.overlays.append(overlay)
        self.image = image

        self.plots =  dict(xy = image)
        return plot

 

from __future__ import division

__all__ = ['ImageStackViewer']
from collections import defaultdict
import numpy

from enthought.traits.api import HasStrictTraits, Instance, Dict, Int, Bool, DelegatesTo, on_trait_change, Button, Any, Enum, List
from enthought.traits.ui.api import View, HSplit, VSplit, Item, VGrid, HGroup, Group, VGroup, Tabbed
from enthought.enable.api import Component, ComponentEditor
from enthought.chaco.tools.api import ZoomTool, PanTool, ImageInspectorTool, ImageInspectorOverlay, ScatterInspector
from enthought.chaco.api import ArrayPlotData, Plot, GridContainer, PlotAxis, HPlotContainer, OverlayPlotContainer
from enthought.chaco.tools.cursor_tool import BaseCursorTool
from enthought.chaco.default_colormaps import bone

from .cursor_tool import CursorTool
from .markers import LowerLeftMarker, UpperRightMarker
from .base_data_source import BaseDataSource
from .base_data_viewer import BaseDataViewer


class ImageStackViewer(BaseDataViewer):
    
    plotdata = Instance(ArrayPlotData)
    plots = Dict # map of plot names to Plot.img_plot(..)[0] results

    current_slice = List ([0,0,0])

    traits_view = View(Tabbed (HSplit(
                Item('plot', editor=ComponentEditor(), 
                     show_label = False,
                     resizable = True, label = 'View'),
                VSplit(
                    Item('options', style='custom', show_label=False, resizable = True),
                    Item('tasks', style='custom', show_label = False, label = 'Tasks')),
                ),
                               Item ('results', style='custom', show_label = False, label = 'Results'),
                               ),
                       )
    
    def _complex_view_changed (self):
        self.reset()

    def _plotdata_default(self):
        arr = self.data
        plotdata = ArrayPlotData(zz=numpy.array ([[0]]),
                                 z_x = [], z_y=[],
                                 y_x = [], y_z=[],
                                 x_y = [], x_z=[],
                                 )
        plotdata.set_data('xy', arr[0])        
        plotdata.set_data('xz', arr[:,0])        
        plotdata.set_data('zy', arr[:,:,0].T)
        return plotdata

    def select_xy_slice(self, z):
        self.current_slice[0] = z
        data = self.get_data_slice(z, 0)
        points = self.get_points(z, 0)
        xdata = self.get_points_coords (z, 0, 2)
        ydata = self.get_points_coords (z, 0, 1)
        #xdata = [point.x for point in points]
        #ydata = [point.y for point in points]
        selections = [i for i, point in enumerate(points) if point.selected]

        self.plotdata.set_data('xy', data)
        self.plotdata.set_data('z_x', xdata)
        self.plotdata.set_data('z_y', ydata)

        self.plots['xyp'].index.metadata['selections'] = selections
        self.plots['xyp'].invalidate_and_redraw()
        self.plots['xy'].invalidate_and_redraw()

    def select_xz_slice(self, y):
        self.current_slice[1] = y
        data = self.get_data_slice(y, 1)
        points = self.get_points(y, 1)
        #xdata = [point.x for point in points]
        #zdata = [point.z for point in points]
        xdata = self.get_points_coords (y, 1, 2)
        zdata = self.get_points_coords (y, 1, 0)
        selections = [i for i, point in enumerate(points) if point.selected]

        self.plotdata.set_data('xz', data)
        self.plotdata.set_data('y_x', xdata)
        self.plotdata.set_data('y_z', zdata)

        self.plots['xzp'].index.metadata['selections'] = selections

        self.plots['xzp'].invalidate_and_redraw()
        self.plots['xz'].invalidate_and_redraw()

    def select_zy_slice(self, x):
        self.current_slice[2] = x
        data = self.get_data_slice(x, 2).T
        points = self.get_points(x, 2)
        #ydata = [point.y for point in points]
        #zdata = [point.z for point in points]
        ydata = self.get_points_coords (x, 2, 1)
        zdata = self.get_points_coords (x, 2, 0)
        selections = [i for i, point in enumerate(points) if point.selected]

        self.plotdata.set_data('zy', data)
        self.plotdata.set_data('x_y', ydata)
        self.plotdata.set_data('x_z', zdata)

        self.plots['zyp'].index.metadata['selections'] = selections

        self.plots['zyp'].invalidate_and_redraw()
        self.plots['zy'].invalidate_and_redraw()

    def redraw (self):
        for p in self.plots.values():
            p.invalidate_and_redraw()

    def reset (self):
        self.reset_points()

    def reset_points (self):
        if self.plots:
            self.select_xy_slice(self.current_slice[0])
            self.select_xz_slice(self.current_slice[1])
            self.select_zy_slice(self.current_slice[2])

    def _xyp_metadata_handler (self):
        z = self.current_slice[0]
        selections = self.plots['xyp'].index.metadata.get ('selections', [])
        points = self.get_points(z, 0)
        changed = False
        for i, point in enumerate (points):
            old_value = point.selected
            new_value = i in selections
            if old_value == new_value:
                continue
            point.selected = new_value
            changed = True
        if changed:
            self.select_xz_slice(self.current_slice[1])
            self.select_zy_slice(self.current_slice[2])

    def _xzp_metadata_handler (self):
        y = self.current_slice[1]
        selections = self.plots['xzp'].index.metadata.get ('selections', [])
        points = self.get_points(y, 1)
        changed = False
        for i,point in enumerate (points):
            old_value = point.selected
            new_value = i in selections
            if old_value == new_value:
                continue
            point.selected = new_value
            changed = True
        if changed:
            self.select_xy_slice(self.current_slice[0])
            self.select_zy_slice(self.current_slice[2])

    def _zyp_metadata_handler (self):
        x = self.current_slice[2]
        selections = self.plots['zyp'].index.metadata.get ('selections', [])
        points = self.get_points(x, 2)
        changed = False
        for i,point in enumerate (points):
            old_value = point.selected
            new_value = i in selections
            if old_value == new_value:
                continue
            point.selected = new_value
            changed = True
        if changed:
            self.select_xy_slice(self.current_slice[0])
            self.select_xz_slice(self.current_slice[1])

    def get_plot(self):
        pixel_sizes = self.data_source.voxel_sizes
        shape = self.data.shape
        m = min(pixel_sizes)
        s = [int(d*sz/m) for d, sz in zip(shape, pixel_sizes)]
        if 1: # else physical aspect ratio is enabled
            ss = max(s)/4
            s = [max(s,ss) for s in s]
        plot_sizes = dict (xy = (s[2], s[1]), xz = (s[2], s[0]), zy = (s[0],s[1]), zz=(s[0],s[0]))

        plots = GridContainer(shape=(2,2), spacing=(3, 3), padding = 50, aspect_ratio=1)
        pxy = Plot(self.plotdata, padding=1, fixed_preferred_size = plot_sizes['xy'],
                   x_axis=PlotAxis (orientation='top'),
                   )
        pxz = Plot(self.plotdata, padding=1, fixed_preferred_size = plot_sizes['xz'],
                   )
        pzy = Plot(self.plotdata, padding=1, fixed_preferred_size = plot_sizes['zy'],
                   #orientation = 'v',  # cannot use 'v' because of img_plot assumes row-major ordering
                   x_axis=PlotAxis(orientation='top'), 
                   y_axis=PlotAxis(orientation='right'),
                   )
        pzz = Plot(self.plotdata, padding=1, fixed_preferred_size = plot_sizes['zz'])

        plots.add(pxy, pzy, pxz, pzz)

        self.plots =  dict(xy = pxy.img_plot('xy', colormap=bone)[0],
                           xz = pxz.img_plot('xz', colormap=bone)[0],
                           zy = pzy.img_plot('zy', colormap=bone)[0],
                           zz = pzz.img_plot('zz')[0],
                           xyp = pxy.plot(('z_x', 'z_y'), type='scatter', color='orange', marker='circle', marker_size=3, 
                                          selection_marker_size = 3, selection_marker='circle')[0],
                           xzp = pxz.plot(('y_x', 'y_z'), type='scatter', color='orange', marker='circle', marker_size=3,
                                          selection_marker_size = 3, selection_marker='circle')[0],
                           zyp = pzy.plot(('x_z', 'x_y'), type='scatter', color='orange', marker='circle', marker_size=3,
                                          selection_marker_size = 3, selection_marker='circle')[0],
                           )

        for p in ['xy', 'xz', 'zy']:
            self.plots[p].overlays.append(ZoomTool(self.plots[p]))
            self.plots[p].tools.append(PanTool(self.plots[p], drag_button='right'))

            imgtool = ImageInspectorTool(self.plots[p])
            self.plots[p].tools.append(imgtool)
            overlay = ImageInspectorOverlay(component=self.plots[p],
                                            bgcolor = 'white',
                                            image_inspector=imgtool)
            self.plots['zz'].overlays.append(overlay)

            self.plots[p+'p'].tools.append (ScatterInspector(self.plots[p+'p'], selection_mode = 'toggle'))

        self.plots['xyp'].index.on_trait_change (self._xyp_metadata_handler, 'metadata_changed')
        self.plots['xzp'].index.on_trait_change (self._xzp_metadata_handler, 'metadata_changed')
        self.plots['zyp'].index.on_trait_change (self._zyp_metadata_handler, 'metadata_changed')

        plot = HPlotContainer()
        # todo: add colormaps
        plot.add(plots)
        return plot
 

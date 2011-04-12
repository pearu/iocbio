
from enthought.traits.api import HasStrictTraits, Instance, Int, DelegatesTo, Bool, on_trait_change, Button, Str, Range
from enthought.chaco.tools.cursor_tool import BaseCursorTool
from enthought.traits.ui.api import View, HSplit, VSplit, Item, VGrid, HGroup, Group, VGroup, Tabbed, RangeEditor

from .cursor_tool import CursorTool
from .markers import LowerLeftMarker, UpperRightMarker
from .base_viewer_task import BaseViewerTask

class LineSelectorTask (BaseViewerTask):

    min_x = min_y = Int(0)
    max_x = Int; max_y = Int
    
    corner_xll = Int (editor=RangeEditor (low_name='min_x', high_name='corner_xur', is_float=False))
    corner_yll = Int (editor=RangeEditor (low_name='min_y', high_name='corner_yur', is_float=False))

    corner_xur = Int (editor=RangeEditor (low_name='corner_xll', high_name='max_x', is_float=False))
    corner_yur = Int (editor=RangeEditor (low_name='corner_yll', high_name='max_y', is_float=False))

    cursor_xyll = Instance (BaseCursorTool)
    cursor_xyur = Instance (BaseCursorTool)

    have_cursors = Bool(False)
    visible = Bool(True)

    traits_view = View(VGroup(
            Item('visible', label='Visible'),
            Item('corner_xll', label='Xll'),
            Item('corner_yll', label='Yll'),
            Item('corner_xur', label='Xur'),
            Item('corner_yur', label='Yur'),
            #Item('create_button', show_label = False),
            #Item('message', show_label=False, style='readonly'),
            ))
    
    def startup(self):
        self._create_cursors()

    def _visible_changed(self):
        if self.have_cursors:
            b = self.visible
            self.cursor_xyll.visible = b
            self.cursor_xyur.visible = b
            self.viewer.redraw()

    def _max_x_default(self): return self.viewer.data.shape[2]
    def _max_y_default(self): return self.viewer.data.shape[1]

    @property
    def corner_ll(self):
        return self.corner_xll, self.corner_yll
    @property
    def corner_ur(self):
        return self.corner_xur, self.corner_yur

    def _corner_xll_changed(self):
        if self.have_cursors:
            self.cursor_xyll.current_position = (self.corner_xll, self.corner_yll)

    def _corner_yll_changed(self):
        if self.have_cursors:
            self.cursor_xyll.current_position = (self.corner_xll, self.corner_yll)

    def _corner_xur_changed(self):
        if self.have_cursors:
            self.cursor_xyur.current_position = (self.corner_xur, self.corner_yur)

    def _corner_yur_changed(self):
        if self.have_cursors:
            self.cursor_xyur.current_position = (self.corner_xur, self.corner_yur)
            
    @on_trait_change('cursor_xyll.current_position')
    def _on_cursor_xyll_change(self): 
        self.corner_xll, self.corner_yll = self.cursor_xyll.current_position

    @on_trait_change('cursor_xyur.current_position')
    def _on_cursor_xyur_change(self): 
        self.corner_xur, self.corner_yur = self.cursor_xyur.current_position

    def _create_cursors(self):
        plot_name = 'xy'
        plot = self.viewer.plots[plot_name]
        a, b = plot_name

        cursor = CursorTool(plot, color='green', marker=LowerLeftMarker(), marker_size=5, threshold=5)
        plot.overlays.append (cursor)
        cursor.current_position = eval('self.corner_%sll, self.corner_%sll' % (a, b))
        setattr (self, 'cursor_%sll' % (plot_name), cursor)

        cursor = CursorTool(plot, color='green', marker=UpperRightMarker(), marker_size=5, threshold=5)
        plot.overlays.append (cursor)
        cursor.current_position = eval('self.corner_%sur, self.corner_%sur' % (a, b))
        setattr (self, 'cursor_%sur' % (plot_name), cursor)

        self.have_cursors = True

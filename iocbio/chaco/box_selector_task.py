
from enthought.traits.api import HasStrictTraits, Instance, Int, DelegatesTo, Bool, on_trait_change, Button, Str, Range
from enthought.chaco.tools.cursor_tool import BaseCursorTool
from enthought.traits.ui.api import View, HSplit, VSplit, Item, VGrid, HGroup, Group, VGroup, Tabbed, RangeEditor
from .base_data_viewer import BaseDataViewer
from .cursor_tool import CursorTool
from .markers import LowerLeftMarker, UpperRightMarker
from .array_data_source import ArrayDataSource
from .base_viewer_task import BaseViewerTask

class BoxSelectorTask(BaseViewerTask):

    # corners of a volume slice
    min_x = min_y = min_z = Int(0)
    max_x = Int; max_y = Int; max_z = Int

    corner_xll = Int (editor=RangeEditor (low_name='min_x', high_name='corner_xur', is_float=False))
    corner_yll = Int (editor=RangeEditor (low_name='min_y', high_name='corner_yur', is_float=False))
    corner_zll = Int (editor=RangeEditor (low_name='min_z', high_name='corner_zur', is_float=False))
    corner_xur = Int (editor=RangeEditor (low_name='corner_xll', high_name='max_x', is_float=False))
    corner_yur = Int (editor=RangeEditor (low_name='corner_yll', high_name='max_y', is_float=False))
    corner_zur = Int (editor=RangeEditor (low_name='corner_zll', high_name='max_z', is_float=False))

    # cursors for volume slice
    cursor_xyll = Instance (BaseCursorTool)
    cursor_xyur = Instance (BaseCursorTool)
    cursor_xzll = Instance (BaseCursorTool)
    cursor_xzur = Instance (BaseCursorTool)
    cursor_zyll = Instance (BaseCursorTool)
    cursor_zyur = Instance (BaseCursorTool)

    have_cursors = Bool(False)
    visible = Bool(True)

    create_button = Button('Create new view with selected box')

    message = Str

    traits_view = View(VGroup(
            Item('visible', label='Visible'),
            Item('corner_xll', label='Xll'),
            Item('corner_yll', label='Yll'),
            Item('corner_zll', label='Zll'),
            Item('corner_xur', label='Xur'),
            Item('corner_yur', label='Yur'),
            Item('corner_zur', label='Zur'),
            Item('create_button', show_label = False),
            Item('message', show_label=False, style='readonly')))

    def startup(self):
        self._create_cursors()        

    def _visible_changed(self):
        if self.have_cursors:
            b = self.visible
            self.cursor_xyll.visible = b
            self.cursor_xzll.visible = b
            self.cursor_zyll.visible = b
            self.cursor_xyur.visible = b
            self.cursor_xzur.visible = b
            self.cursor_zyur.visible = b
            self.viewer.redraw()

    def _max_x_default(self): return self.viewer.data.shape[2]
    def _max_y_default(self): return self.viewer.data.shape[1]
    def _max_z_default(self): return self.viewer.data.shape[0]
    def _corner_xur_default(self): return self.max_x
    def _corner_yur_default(self): return self.max_y
    def _corner_zur_default(self): return self.max_z

    @property
    def corner_ll(self):
        return self.corner_xll, self.corner_yll, self.corner_zll
    @property
    def corner_ur(self):
        return self.corner_xur, self.corner_yur, self.corner_zur

    def _corner_xll_changed(self):
        if self.have_cursors:
            if self.cursor_xyll is not None:
                self.cursor_xyll.current_position = (self.corner_xll, self.corner_yll)
            if self.cursor_xzll is not None:
                self.cursor_xzll.current_position = (self.corner_xll, self.corner_zll)

    def _corner_yll_changed(self):
        if self.have_cursors:
            if self.cursor_xyll is not None:
                self.cursor_xyll.current_position = (self.corner_xll, self.corner_yll)
            if self.cursor_zyll is not None:
                self.cursor_zyll.current_position = (self.corner_zll, self.corner_yll)

    def _corner_zll_changed(self):
        if self.have_cursors:
            if self.cursor_xzll is not None:
                self.cursor_xzll.current_position = (self.corner_xll, self.corner_zll)
            if self.cursor_zyll is not None:
                self.cursor_zyll.current_position = (self.corner_zll, self.corner_yll)

    def _corner_xur_changed(self):
        if self.have_cursors:
            if self.cursor_xyur is not None:
                self.cursor_xyur.current_position = (self.corner_xur, self.corner_yur)
            if self.cursor_xzur is not None:
                self.cursor_xzur.current_position = (self.corner_xur, self.corner_zur)

    def _corner_yur_changed(self):
        if self.have_cursors:
            if self.cursor_xyur is not None:
                self.cursor_xyur.current_position = (self.corner_xur, self.corner_yur)
            if self.cursor_zyur is not None:
                self.cursor_zyur.current_position = (self.corner_zur, self.corner_yur)

    def _corner_zur_changed(self):
        if self.have_cursors:
            if self.cursor_xzur is not None:
                self.cursor_xzur.current_position = (self.corner_xur, self.corner_zur)
            if self.cursor_zyur is not None:
                self.cursor_zyur.current_position = (self.corner_zur, self.corner_yur)

    @on_trait_change('cursor_xyll.current_position')
    def _on_cursor_xyll_change(self): 
        self.corner_xll, self.corner_yll = self.cursor_xyll.current_position

    @on_trait_change('cursor_xzll.current_position')
    def _on_cursor_xzll_change(self): 
        self.corner_xll, self.corner_zll = self.cursor_xzll.current_position

    @on_trait_change('cursor_zyll.current_position')
    def _on_cursor_zyll_change(self): 
        self.corner_zll, self.corner_yll = self.cursor_zyll.current_position

    @on_trait_change('cursor_xyur.current_position')
    def _on_cursor_xyur_change(self): 
        self.corner_xur, self.corner_yur = self.cursor_xyur.current_position

    @on_trait_change('cursor_xzur.current_position')
    def _on_cursor_xzur_change(self): 
        self.corner_xur, self.corner_zur = self.cursor_xzur.current_position

    @on_trait_change('cursor_zyur.current_position')
    def _on_cursor_zyur_change(self): 
        self.corner_zur, self.corner_yur = self.cursor_zyur.current_position

    def _create_cursors(self):        
        for plot_name, plot in self.viewer.plots.items ():
            if plot_name not in ['xy', 'xz', 'zy']:
                continue
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

    def _create_button_fired(self):
        ll, ur = self.corner_ll, self.corner_ur
        data_source = ArrayDataSource(original_source = self.viewer.data_source,
                                      kind = self.viewer.data_source.kind,
                                      data = self.viewer.data[ll[2]:ur[2], ll[1]:ur[1], ll[0]:ur[0]]
                                      )
        viewer = self.viewer.__class__(data_source = data_source)
        self.viewer.add_result(viewer)
        viewer.copy_tasks(self.viewer)
        self.message = 'Last box selection is in Results/%s tab.' % (viewer.name)

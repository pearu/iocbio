
from enthought.traits.api import HasStrictTraits, Instance, Int, Bool, on_trait_change, Range, Tuple
from enthought.traits.ui.api import View, HSplit, VSplit, Item, VGrid, HGroup, Group, VGroup, Tabbed
from enthought.chaco.tools.cursor_tool import BaseCursorTool
from .cursor_tool import CursorTool
from .base_viewer_task import BaseViewerTask
from .base_data_viewer import BaseDataViewer


class SliceSelectorTask(BaseViewerTask):

    # indices of plane slices
    min_x = min_y = min_z = Int(0)
    max_x = Int; max_y = Int; max_z = Int
    value_x = Int; value_y = Int; value_z = Int
    slice_x = Range(low='min_x', high='max_x', value='value_x')
    slice_y = Range(low='min_y', high='max_y', value='value_y')
    slice_z = Range(low='min_z', high='max_z', value='value_z')

    last_slice = Tuple (Int, Int, Int)

    # cursors for plane slice
    cursor_xy = Instance (BaseCursorTool)
    cursor_xz = Instance (BaseCursorTool)
    cursor_zy = Instance (BaseCursorTool)

    have_cursors = Bool(False)
    visible = Bool(True)

    traits_view = View(VGroup(
            Item('visible', label='Visible'),
            Item('slice_x', label='X'),
            Item('slice_y', label='Y'), 
            Item('slice_z', label='Z'),
            ))

    def _max_x_default(self): return self.viewer.data.shape[2]-1
    def _max_y_default(self): return self.viewer.data.shape[1]-1
    def _max_z_default(self): return self.viewer.data.shape[0]-1
    def _value_z_default (self): return self.viewer.data.shape[0]//2
    def _value_y_default (self): return self.viewer.data.shape[1]//2
    def _value_x_default (self): return self.viewer.data.shape[2]//2

    def reset_slices(self):
        self.set_slices (*self.last_slice)

    def set_slices (self, z, y, x):
        self.last_slice = (z, y, x)
        if self.viewer.shifted_view:
            z = self.viewer.ifftshift (z, self.viewer.data.shape[0])
            y = self.viewer.ifftshift (y, self.viewer.data.shape[1])
            x = self.viewer.ifftshift (x, self.viewer.data.shape[2])
        self.set(slice_x=x, slice_y=y, slice_z=z)
        self.viewer.reset()

    def startup(self):
        self._create_cursors()
        self.viewer.select_zy_slice (self.slice_x)
        self.viewer.select_xz_slice (self.slice_y)
        self.viewer.select_xy_slice (self.slice_z)

    def _visible_changed(self):
        if self.have_cursors:
            b = self.visible
            self.cursor_xy.visible = b
            self.cursor_xz.visible = b
            self.cursor_zy.visible = b
            self.viewer.redraw()

    def _create_cursors(self):
        slice_dict = dict(x=self.slice_x, y=self.slice_y, z=self.slice_z)
        for plot_name in ['xy', 'xz', 'zy']:
            a1, a2 = plot_name

            plot = self.viewer.plots[plot_name]
            default_slice = (slice_dict[a1], slice_dict[a2])

            cursor = CursorTool(plot, color='red', threshold=3, marker_size = 1)
            plot.overlays.append(cursor)
            cursor.current_position = default_slice
            setattr (self, 'cursor_%s' % (plot_name), cursor)

        self.have_cursors = True

    def _slice_x_changed (self, old, new):
        if new < 0 or new >= self.viewer.data.shape[2]:
            self.slice_x = old
            return
        if self.have_cursors:
            self.cursor_xy.current_position = (self.slice_x, self.slice_y)
            self.cursor_xz.current_position = (self.slice_x, self.slice_z)

            self.viewer.select_zy_slice (self.slice_x)

    def _slice_y_changed (self, old, new):
        if new < 0 or new >= self.viewer.data.shape[1]:
            self.slice_y = old
            return
        if self.have_cursors:
            self.cursor_xy.current_position = (self.slice_x, self.slice_y)
            self.cursor_zy.current_position = (self.slice_z, self.slice_y)

            self.viewer.select_xz_slice (self.slice_y)

    def _slice_z_changed (self, old, new):
        if new < 0 or new >= self.viewer.data.shape[0]:
            self.slice_z = old
            return
        if self.have_cursors:
            self.cursor_xz.current_position = (self.slice_x, self.slice_z)
            self.cursor_zy.current_position = (self.slice_z, self.slice_y)

            self.viewer.select_xy_slice (self.slice_z)


    @on_trait_change('cursor_xy.current_position')
    def _on_cursor_xy_change(self): 
        self.slice_x, self.slice_y = self.cursor_xy.current_position

    @on_trait_change('cursor_xz.current_position')
    def _on_cursor_xz_change(self):
        self.slice_x, self.slice_z = self.cursor_xz.current_position

    @on_trait_change('cursor_zy.current_position')
    def _on_cursor_zy_change(self):
        self.slice_z, self.slice_y = self.cursor_zy.current_position

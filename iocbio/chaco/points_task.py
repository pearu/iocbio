
__all__ = ['PointsTask']

import numpy

from enthought.traits.api import Button, Any, Array, Float, DelegatesTo, on_trait_change, Dict, Bool, HasStrictTraits, List, Int, Tuple, Instance, on_trait_change
from enthought.traits.ui.api import View, VGroup, Item, ListEditor, TableEditor
from enthought.traits.ui.table_column import ObjectColumn, ExpressionColumn
from enthought.traits.ui.extras.checkbox_column import CheckboxColumn
from enthought.traits.ui.ui_editors.array_view_editor \
    import ArrayViewEditor
from enthought.chaco.api import Plot, ArrayPlotData, OverlayPlotContainer
from enthought.chaco.overlays.api import ContainerOverlay

from .base_viewer_task import BaseViewerTask
from .array_data_source import ArrayDataSource
from .point import Point

table_editor = TableEditor(
    columns = [
        ExpressionColumn(
            expression = '"(%3s, %3s, %3s)->%s" % (object.coordinates+ (object.value,))',
            label = '(Z, Y, X)->Value',
            ),
        CheckboxColumn(name = 'selected'),
        ],
    selected = 'selected_point',
    )


class PointsTask(BaseViewerTask):

    points = DelegatesTo('viewer')
    selected_point = Instance(Point)

    traits_view = View (#VGroup(
        Item('points', show_label = False, editor=table_editor),
        )

    def startup(self):
        pass

    def _selected_point_changed (self):
        if self.selected_point is not None:
            slice_selector = self.viewer.slice_selector
            if slice_selector is not None:
                slice_selector.set_slices(*self.selected_point.coordinates)

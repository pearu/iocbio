
__all__ = ['TablePlotTask']
from collections import defaultdict
import numpy

from enthought.traits.api import Button, Any, Array, Float, DelegatesTo, on_trait_change, Dict, Bool, HasStrictTraits, List, Int, Tuple, Instance, on_trait_change, Property, Str, cached_property
from enthought.traits.ui.api import View, VGroup, Item, ListEditor, TableEditor, EnumEditor
from enthought.traits.ui.table_column import ObjectColumn, ExpressionColumn
from enthought.traits.ui.extras.checkbox_column import CheckboxColumn
from enthought.traits.ui.ui_editors.array_view_editor \
    import ArrayViewEditor
from enthought.chaco.api import Plot, ArrayPlotData, OverlayPlotContainer, create_line_plot, HPlotContainer, PlotAxis, PlotGrid, GridContainer
from enthought.enable.api import Component, ComponentEditor
from enthought.chaco.overlays.api import ContainerOverlay
from enthought.chaco.tools.api import PanTool, ZoomTool, BroadcasterTool

from .base_viewer_task import BaseViewerTask
from .array_data_source import ArrayDataSource
from .point import Point


class TablePlotTask(BaseViewerTask):

    tables = DelegatesTo('viewer')

    table = Dict

    available_tables = Property
    selected_table = Str(editor=EnumEditor(name = 'available_tables', cols=1))

    plot = Instance (Component)

    traits_view = View (#VGroup(
        Item ('selected_table', style = 'simple', label='Table',  visible_when='len(tables) != 0'),
        Item('plot', editor=ComponentEditor(), 
             show_label = False,
             resizable = True, label = 'View', visible_when='tables'),
        )

    def startup(self):
        if self.tables:
            self.selected_table = self.available_tables[-1]

    @property
    def available_tables(self):
        return sorted(self.tables.keys ())

    def _selected_table_changed(self):
        self.table = self.tables[self.selected_table]
        self.plot = self.get_plot ()

    def _plot_default(self):
        return self.get_plot()

    def get_plot(self):
        #pd = ArrayPlotData()
        index_label = 'index'
        index = None
        colors = ['purple','blue','green','gold', 'orange', 'red', 'black']
        groups = defaultdict(lambda:[])

        pd = ArrayPlotData()

        index_values = None
        if 'relative time' in self.table:
            index_key = 'relative time'
            index_values = self.table[index_key]
        else:
            index_key = 'index'
        index_label = index_key

        for key, values in self.table.items():
            if index_values is None:
                index_values = range(len(values))
            if key==index_label:
                continue
            if key.startswith('stage '):
                label = key[6:].strip()
                group = groups['stages']
            elif key=='contact position':
                label = key
                group = groups['stages']
            elif key.startswith('fiber '):

                if key.endswith('deformation'):
                    label = key[5:-11].strip()
                    group = groups['fiber deformation']
                elif key.endswith('position'):
                    label = key[5:-8].strip()
                    group = groups['fiber position']
                else:
                    label = key[5:].strip ()
                    group = groups['fiber']
            elif key.startswith('sarcomere '):
                label = key[10:].strip()
                if label=='orientation': # this is artificial information
                    continue
                group = groups['sarcomere']
            else:
                label = key
                group = groups[key]

            group.append((index_label, label, index_key, key))
            pd.set_data(key, values)

        pd.set_data (index_key, index_values)

        if 'force' in self.table and 'stage right current' in self.table:
            group = groups['position-force']
            group.append(('position','force','stage right current','force'))



        n = len (groups)
        if n in [0,1,2,3,5,7]:
            shape = (n, 1)
        elif n in [4,6,8,10]:
            shape = (n//2,2)
        elif n in [9]:
            shape = (n//3,3)
        else:
            raise NotImplementedError (`n`)

        container = GridContainer(padding=10, #fill_padding=True,
                                  #bgcolor="lightgray", use_backbuffer=True,
                                  shape=shape, spacing=(0,0))

        for i, (group_label, group_info) in enumerate(groups.items ()):
            plot = Plot (pd)
            for j, (index_label, label, index_key, key) in enumerate(group_info):
                color = colors[j % len (colors)]
                plot.plot((index_key, key), name=label, color=color, x_label=index_label)
            plot.legend.visible = True
            plot.title = group_label
            plot.tools.append(PanTool(plot))
            zoom = ZoomTool(component=plot, tool_mode="box", always_on=False)
            plot.overlays.append(zoom)
            container.add (plot)

        return container

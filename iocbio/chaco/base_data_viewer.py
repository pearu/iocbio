
__all__ = ['BaseDataViewer']
import gc
import sys
from collections import defaultdict
import numpy
from numpy.testing import memusage

from enthought.traits.api import HasStrictTraits, Str, List, Instance, DelegatesTo, Bool, Enum, Any, cached_property, Property, Dict, Button, Int, on_trait_change
from enthought.traits.ui.api import Item, ListEditor, View, TableEditor, Group, EnumEditor, ListStrEditor
from enthought.traits.ui.table_column import ObjectColumn
from enthought.enable.api import Component

from iocbio.utils import bytes2str

from .base_data_source import BaseDataSource
from .point import Point

class DataInfo (HasStrictTraits):

    name = Str
    data = Any

_data = [None]
_data_table = {}

class Options (HasStrictTraits):

    memusage = Int
    status = Str('')

    is_complex = Bool
    complex_view = Enum ('real', 'imag', 'abs', 'complex', cols=4)
    shifted_view = Bool(False)

    #data_table = Dict
    points_table = Dict

    #data = Any # rank-3 array
    points = List(Point)

    available_data = List
    selected_data = Str (editor=EnumEditor(name = 'available_data', cols=4))
    remove_selected_data = Button

    traits_view = View(
        Group(Item('selected_data', style = 'simple', show_label=False),
              Item('remove_selected_data', show_label=False, resizable = False, visible_when='selected_data != "A"'),
              label = 'Available data', show_border=True)
        ,'_',
        Group(Item('complex_view', style='custom', show_label=False), label='Complex view',  visible_when='is_complex', show_border=True),
        '_',
        Item('shifted_view'),
        resizable = True,
        scrollable= True)

    @property
    def data (self):
        self.update_memusage()
        return _data[0]

    @data.setter
    def data (self, data):
        _data[0] = data
        self.update_memusage()

    @property
    def data_table(self):
        return _data_table

    def _points_changed(self):
        self.points_table[self.selected_data] = self.points

    def update_memusage(self):
        self.memusage = memusage()        

    def _status_changed(self):
        self.update_memusage()
    
    def update_available_data(self):
        self.available_data = sorted(self.data_table.keys())

    def _selected_data_changed (self):
        if self.data_table.keys():
            data = self.data_table[self.selected_data]
            self.data = data
            self.points = self.points_table[self.selected_data]
            self.is_complex = data.dtype.kind=='c'

    def _shifted_view_changed (self):
        self._selected_data_changed()

    def _remove_selected_data_fired(self):
        if len (self.data_table)<=1 or self.selected_data=='A':
            return
        name = self.selected_data
        index = self.available_data.index(name)
        if index<=0: index += 1
        else: index -= 1
        self.selected_data = self.available_data[index]
        del self.data_table[name]
        del self.points_table[name]
        self.update_available_data()
        self.update_memusage()

class BaseDataViewer (HasStrictTraits):

    options = Instance(Options)
    memusage = DelegatesTo('options')
    status = DelegatesTo('options')
    complex_view = DelegatesTo('options')
    shifted_view = DelegatesTo('options')
    #data_table = DelegatesTo('options')
    selected_data = DelegatesTo('options')
    points = DelegatesTo('options')
    points_table = DelegatesTo('options')

    is_complex = DelegatesTo('options')

    data_source = Instance(BaseDataSource)
    has_data = DelegatesTo('data_source')
    voxel_sizes = DelegatesTo('data_source')
    tables = DelegatesTo('data_source')

    plot = Instance (Component)

    results = List(editor=ListEditor(use_notebook=True,
                                     deletable=True,
                                     dock_style='tab',
                                     page_name='.name'))

    tasks = List(editor=ListEditor(use_notebook=True,
                                   deletable=False,
                                   dock_style='tab',
                                   page_name='.name'))

    traits_view = View()

    name = Str
    inited = Bool (False)

    def __init__ (self, *args, **kws):
        self.inited = False
        HasStrictTraits.__init__ (self, *args, **kws)        
        self.options = Options()
        self.add_data('A', self.data_source.data)
        self.options.selected_data = 'A'
        self.inited = True

    @on_trait_change('data_source.voxel_sizes')
    def replot(self):
        if self.inited:
            self.plot = self.get_plot()

    def _plot_default(self):
        return self.get_plot()

    def get_plot (self):
        raise NotImplementedError ('%s.get_plot method' % (self.__class__.__name__))
    
    def update_memusage(self):
        self.options.update_memusage()

    def reset(self):
        print '%s.reset: not implemented' % (self.__class__.__name__)

    def _shifted_view_changed(self):
        self.slice_selector.reset_slices()

    @property
    def data (self):
        return self.options.data
    @property
    def data_table(self):
        return self.options.data_table

    def _selected_data_changed (self):
        self.reset()

    def _name_default (self, _count = [0]):
        n = self.__class__.__name__
        if n.endswith ('Viewer'): n = n[:-6]
        c = _count[0]
        _count[0] += 1
        return '%s %s' % (n, c)

    def add_task(self, task_cls):
        assert issubclass(task_cls, BaseViewerTask), `task_cls`
        self.tasks.append (task_cls(viewer = self))

    def copy_tasks (self, other):
        for task in other.tasks:
            self.tasks.append (task.__class__ (viewer = self))

    slice_selector = Property(depends_on='tasks')

    @cached_property
    def _get_slice_selector(self):
        for task in self.tasks:
            if task.__class__.__name__ == 'SliceSelectorTask':
                return task
    
    def add_result (self, result):
        self.results.append(result)

    def add_data(self, name, data):
        self.data_table[name] = data
        self.points_table[name] = []
        self.options.update_available_data()
        self.selected_data = name

    def get_data(self, name):
        return self.data_table.get(name)

    def fftshift(self, obj, n=None):
        if n is not None:
            if isinstance(obj, int):
                return numpy.fft.fftshift(range(n))[obj]
            if isinstance(obj, list):
                r = numpy.fft.fftshift(range(n))
                return [r[zz] for zz in obj]
            raise NotImplementedError(type (obj))
        return numpy.fft.fftshift(obj)

    def ifftshift(self, obj, n=None):
        if n is not None:
            if isinstance(obj, int):
                return numpy.fft.ifftshift(range(n))[obj]
            if isinstance(obj, list):
                r = numpy.fft.ifftshift(range(n))
                return [r[o] for o in obj]
            raise NotImplementedError(type (obj))
        return numpy.fft.ifftshift(obj)

    def set_point_data(self, value_points):
        l = []
        for i, (v,z,y,x) in enumerate(value_points):
            l.append (Point (coordinates=(z,y,x), value=v, selected=not i))
        self.options.points = l
        assert self.points == self.options.points
        assert self.points == self.options.points_table[self.selected_data]

    def get_data_slice(self, index, axis):
        complex_view = self.complex_view
        shifted_view = self.shifted_view
        if shifted_view:
            index = self.fftshift(index, self.data.shape[axis])
        if axis==0:
            data = self.data[index]
        elif axis==1:
            data = self.data[:,index]
        elif axis==2:
            data = self.data[:,:,index]
        if complex_view=='real':
            data = data.real
        elif complex_view=='imag':
            data = data.imag
        elif complex_view=='abs':
            data = abs(data)
        elif complex_view=='complex':
            pass
        else:
            raise NotImplementedError (`complex_view`)
        if shifted_view:
            data = numpy.fft.fftshift(data)            
        return data

    def get_data_time(self, index, axis):
        if isinstance(self.data, numpy.ndarray):
            return
        if axis==0:
            return self.data.get_time(index)
        raise NotImplementedError ('get_data_time for axis=%s' % (axis))

    def get_points(self, index, axis):
        shifted_view = self.shifted_view
        if self.shifted_view:
            index = self.fftshift(index, self.data.shape[axis])
        l = []
        for point in self.points:
            if point.coordinates[axis]==index:
                l.append(point)
        return l

    def get_points_coords(self, index, axis, direction):
        shifted_view = self.shifted_view
        if self.shifted_view:
            index = self.fftshift(index, self.data.shape[axis])
        l = []
        for point in self.points:
            if point.coordinates[axis]==index:
                coords = point.coordinates[direction]
                if self.shifted_view:
                    coords = self.ifftshift(coords, self.data.shape[direction])
                l.append(coords)
        return l

from .base_viewer_task import BaseViewerTask

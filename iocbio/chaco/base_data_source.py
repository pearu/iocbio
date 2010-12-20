
__all__ = ['BaseDataSource']

from enthought.traits.api import HasStrictTraits, Bool, Any, Enum, Str, Tuple, Float

data_source_kinds = Enum('image_stack', 'image_timeseries', 'image_stack_timeseries', None)

_data = [None]

class BaseDataSource(HasStrictTraits):

    has_data = Bool(False) # set True when data is loaded

    description = Str
    kind = data_source_kinds
    voxel_sizes = Tuple(Float(1.0), Float(1.0), Float(1.0))
    pixel_sizes = Tuple(Float(1.0), Float(1.0))

    is_image_stack = Bool(True)
    is_image_timeseries = Bool(False)
    is_image_stack_timeseries = Bool(False)

    @property
    def data(self):
        return _data[0]

    @data.setter
    def data(self, data):
        _data[0] = data
        self.has_data = data is not None
        if data is None:
            print '%s: removing data' % (self.__class__.__name__)
        else:
            print '%s: setting data' % (self.__class__.__name__)

    def _kind_changed (self):
        self.is_image_stack = self.kind=='image_stack'
        self.is_image_timeseries = self.kind=='image_timeseries'
        self.is_image_stack_timeseries = self.kind=='image_stack_timeseries'

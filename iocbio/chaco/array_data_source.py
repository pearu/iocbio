
__all__ = ['ArrayDataSource']

from enthought.traits.api import Instance
from enthought.traits.ui.api import View, Group, Item
from .base_data_source import BaseDataSource

class ArrayDataSource(BaseDataSource):

    original_source = Instance(BaseDataSource)

    traits_view = View(Group(Item("original_source", style='readonly')))

    def get_pixel_sizes (self):
        return self.original_source.get_pixel_sizes()

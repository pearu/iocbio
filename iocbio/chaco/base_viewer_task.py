
from enthought.traits.api import HasStrictTraits, Instance, Str, DelegatesTo, Bool
from .base_data_viewer import BaseDataViewer

class BaseViewerTask(HasStrictTraits):

    viewer = Instance (BaseDataViewer)
    is_complex = DelegatesTo('viewer')
    is_real = Bool(True)
    name = Str

    def _is_complex_changed(self):
        self.is_real = not self.is_complex

    def __init__(self, viewer = None):
        HasStrictTraits.__init__ (self, viewer=viewer)
        self.startup()

    def startup (self):
        raise NotImplementedError ('%s must implement startup() method.' % (self.__class__.__name__))

    def _name_default (self):
        n = self.__class__.__name__
        if n.endswith ('Task'):
            n = n[:-4]
        return n

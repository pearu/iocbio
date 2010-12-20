
from enthought.traits.api import HasStrictTraits, Int, Tuple, Bool, Any, Property, cached_property

class Point(HasStrictTraits):
    
    coordinates = Tuple (Int, Int, Int)
    x = Property(depends_on='coordinates')
    y = Property(depends_on='coordinates')
    z = Property(depends_on='coordinates')

    selected = Bool(False)
    value = Any

    @cached_property
    def _get_z (self): return self.coordinates[0]
    @cached_property
    def _get_y (self): return self.coordinates[1]
    @cached_property
    def _get_x (self): return self.coordinates[2]

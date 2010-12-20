
__all__ = ['CursorTool']

from enthought.traits.api import Instance
from enthought.chaco.tools.cursor_tool import CursorTool2D
from enthought.chaco.tools.cursor_tool import CursorTool as _CursorTool
from enthought.enable.markers import CircleMarker
from enthought.chaco.base_2d_plot import Base2DPlot

class IocbioCursorTool2D(CursorTool2D):

    # Allow using subclasses of CircleMarker:
    marker = Instance(CircleMarker, ())
    invisible_layout = None

def CursorTool(component, *args, **kwds):
    if isinstance(component, Base2DPlot):
        return IocbioCursorTool2D(component, *args, **kwds)
    return _CursorTool(component, *args, **kwds)
CursorTool.__doc__ = _CursorTool.__doc__

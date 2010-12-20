
__all__ = ['LowerLeftMarker', 'UpperRightMarker']

import numpy
from enthought.enable.api import CircleMarker
from enthought.kiva.constants import NO_MARKER

class LowerLeftMarker(CircleMarker):
    
    kiva_marker = NO_MARKER
    circle_points = CircleMarker.circle_points.copy ()[4:]
    circle_points[0] = (0,0)

class UpperRightMarker(CircleMarker):
    
    kiva_marker = NO_MARKER
    circle_points = numpy.dot(CircleMarker.circle_points, numpy.array([[0,-1],[-1,0]]))[4:]
    circle_points[0] = (0,0)

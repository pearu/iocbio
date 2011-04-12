
__all__ = ['set_sarcomere_length_options']

import os
from optparse import OptionGroup, NO_DEFAULT
from iocbio.script_options import set_formatter

def set_sarcomere_length_options (parser):
    from ..io.script_options import get_microscope_options_group,  get_io_options_group
    set_formatter (parser)
    parser.set_usage('%prog [options] [ -i INPUT_PATH ]')
    parser.set_description('Estimate the length of sarcomere.')
    parser.add_option ('--input-path','-i',
                       type = 'file', metavar='PATH',
                       help = 'Specify input PATH of 3D images.'
                       )
    parser.add_option ('--roi-center-line',
                       help = 'Specify the coordinates of ROI center line in pixels: x0,y0,x1,y1')
    parser.add_option ('--roi-width',
                       type = 'int',
                       help = 'Specify the width of ROI in pixels')
    parser.add_option ('--nof-points',
                       type = 'int', default = 512,
                       help = 'Specify the number of interpolation points.')
    parser.add_option_group(get_io_options_group(parser))
    parser.add_option_group(get_microscope_options_group(parser))
 

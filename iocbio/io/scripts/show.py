#!/usr/bin/env python
# -*- python-mode -*-
# Author: Pearu Peterson
# Created: August 2009

from __future__ import division
import os

### START UPDATE SYS.PATH ###
### END UPDATE SYS.PATH ###

import numpy
from iocbio.io import ImageStack
from iocbio.optparse_gui import OptionParser
from iocbio.io.io import fix_path
from iocbio.utils import tostr
from iocbio.io.script_options import set_show_options

def runner (parser, options, args):
    
    if not hasattr(parser, 'runner'):
        options.output_path = None

    if args:
        if len (args)==1:
            if options.input_path:
                print >> sys.stderr, "WARNING: overwriting input path %r with %r" % (options.input_path,  args[0])
            options.input_path = args[0]

    if options.input_path is None:
        parser.error('Expected --input-path but got nothing')

    options.input_path = fix_path (options.input_path)

    stack = ImageStack.load(options.input_path, options=options)
    images = stack.images
    roll_axis = dict(XY=0, XZ=1, YZ=2).get(options.projection, 0)
    axis_labels = ('Z', 'Y', 'X')
    #roll_axis = int (options.roll_axis)
    voxel_sizes = stack.get_voxel_sizes()
    resolution = ['', '', '']

    title = '%s[%s]' % (os.path.basename (options.input_path), images.dtype)
    dr = stack.get_lateral_resolution()
    dz = stack.get_axial_resolution()


    if dr is not None:
        resolution[1] = tostr(dr/voxel_sizes[1]) + 'px'
        resolution[2] = tostr(dr/voxel_sizes[2]) + 'px'
    if dz is not None:
        resolution[0] = tostr(dz/voxel_sizes[0]) + 'px'

    resolution = tuple (resolution)
    if roll_axis:
        images = numpy.rollaxis(images, roll_axis)
        voxel_sizes = (voxel_sizes[roll_axis],) + voxel_sizes[:roll_axis] + voxel_sizes[roll_axis+1:]
        axis_labels = (axis_labels[roll_axis],) + axis_labels[:roll_axis] + axis_labels[roll_axis+1:]
        resolutions = (resolution[roll_axis],) + resolution[:roll_axis] + resolution[roll_axis+1:]

    xlabel = '%s, resol=%s, px size=%sum, size=%sum' \
        % (axis_labels[-1], resolution[-1], tostr(voxel_sizes[-1]*1e6),  tostr(voxel_sizes[-1]*1e6*images.shape[-1]))
    ylabel = '%s, resol=%s, px size=%sum, size=%sum' \
        % (axis_labels[-2], resolution[-2], tostr(voxel_sizes[-2]*1e6),  tostr(voxel_sizes[-2]*1e6*images.shape[-2]))

    import matplotlib.cm as cm
    import matplotlib.pyplot as pyplot
    from iocbio.io.tifffile import imshow

    if options.invert_cmap:
        cmap = getattr(cm, options.cmap+'_r', options.cmap)
    else:
        cmap = getattr(cm, options.cmap, 'gray')

    figure, subplot, image = imshow(images, title = title,
                                    #miniswhite=page.photometric=='miniswhite',
                                    interpolation=options.interpolation,
                                    cmap=cmap,
                                    dpi=options.dpi, isrgb=options.rgb,
                                    show_hist = options.histogram_bins,
                                    auto_scale = options.auto_scale)
    axes = figure.get_axes()[0]
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    pyplot.show()

def main ():
    parser = OptionParser()
    set_show_options (parser)
    if hasattr(parser, 'runner'):
        parser.runner = runner
    options, args = parser.parse_args()
    runner(parser, options, args)
    return

if __name__=="__main__":
    main()

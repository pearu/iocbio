#!/usr/bin/env python
# -*- python-mode -*-
"""
Estimate a scalar field from noisy observations (images) using local
constant or local linear regression.
"""
# Author: Pearu Peterson
# Created: September 2009

from __future__ import division
import os

### START UPDATE SYS.PATH ###
### END UPDATE SYS.PATH ###

import numpy
from iocbio.optparse_gui import OptionParser
from iocbio.io import ImageStack
from iocbio.ops.apply_window_ext import apply_window_inplace
from iocbio.io.io import fix_path
from iocbio.io import utils

import time

__usage__ = """\
%prog [options] [ INPUT_PATH [ OUTPUT_PATH ] ]

Description:
  %prog applies smooth window to images in INPUT_PATH to make images periodic."""

def runner(parser, options, args):

    smoothness = int(options.smoothness or 1)
    verbose = options.verbose if options.verbose is not None else True

    if not hasattr(parser, 'runner'):
        options.output_path = None

    if args:
        if len (args)==1:
            if options.input_path:
                print >> sys.stderr, "WARNING: overwriting input path %r with %r" % (options.input_path,  args[0])
            options.input_path = args[0]
        elif len(args)==2:
            if options.input_path:
                print >> sys.stderr, "WARNING: overwriting input path %r with %r" % (options.input_path,  args[0])
            options.input_path = args[0]
            options.output_path = args[1]
        else:
            parser.error("incorrect number of arguments (expected upto 2 but got %s)" % (len(args)))
    
    options.input_path = fix_path(options.input_path)
    stack = ImageStack.load(options.input_path, options=options)
    voxel_sizes = stack.get_voxel_sizes()
    new_images = stack.images.copy()
    background = (stack.pathinfo.get_background() or [0,0])[0]
    print 'Image has background', background
    window_width = options.window_width or None

    if window_width is None:
        dr = stack.get_lateral_resolution()
        dz = stack.get_axial_resolution()
        if dr is None or dz is None:
            window_width = 3.0
            scales = tuple([s/(window_width*min(voxel_sizes)) for s in voxel_sizes])
        else:
            print 'lateral resolution: %.3f um (%.1f x %.1f px^2)' % (1e6*dr, dr/voxel_sizes[1], dr/voxel_sizes[2])
            print 'axial resolution: %.3f um (%.1fpx)' % (1e6*dz, dz / voxel_sizes[0])
            vz,vy,vx = voxel_sizes
            m = 3
            scales = (m*vz/dz, m*vy/dr, m*vx/dr)
            window_width = '%.1fx%.1f' % (dz/m/vz, dr/m/vy)
    else:
        window_width = options.window_width
        scales = tuple([s/(window_width*min(voxel_sizes)) for s in voxel_sizes])

    print 'Window size in pixels:', [1/s for s in scales]
    apply_window_inplace (new_images, scales, smoothness, background)

    if options.output_path is None:
        b,e = os.path.splitext(options.input_path)
        suffix = '_window%s_%s' % (window_width, smoothness)
        options.output_path = b + suffix + (e or '.tif')
    options.output_path = fix_path(options.output_path)

    if verbose:
        print 'Leak: %.3f%%' % ( 100*(1-new_images.sum ()/stack.images.sum ()))
        print 'MSE:', ((new_images - stack.images)**2).mean()
        print 'Energy:', ((stack.images)**2).sum()
        print 'Saving result to',options.output_path
    ImageStack(new_images, stack.pathinfo).save(options.output_path)        

def main ():
    parser = OptionParser(__usage__)
    from iocbio.ops.script_options import set_apply_window_options
    set_apply_window_options (parser)
    if hasattr(parser, 'runner'):
        parser.runner = runner
    options, args = parser.parse_args()
    runner(parser, options, args)

if __name__ == '__main__':
    main()

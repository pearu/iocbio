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
import sys

### START UPDATE SYS.PATH ###
### END UPDATE SYS.PATH ###

import numpy
from iocbio.optparse_gui import OptionParser
from iocbio.io import ImageStack
from iocbio.ops import regress
from iocbio.io.io import fix_path

__usage__ = """\
%prog [options] [ INPUT_PATH [ OUTPUT_PATH ] ]

Description:
  %prog applies local regression methods to images in INPUT_PATH."""

def get_dtype_min_max(dtype):
    """
    Return possible minimum and maximum values of an integer type.
    """
    if isinstance (dtype, type):
        type_name = dtype.__name__
    else:
        type_name = str(dtype)
    if type_name.startswith ('uint'):
        return 0, getattr (numpy, type_name)(-1)
    elif type_name.startswith('int'):
        bits = int(type_name[3:])
        return int (getattr(numpy, type_name) (2**(bits-1))), int(getattr(numpy, type_name) (2**(bits-1)-1))
    raise NotImplementedError (`dtype, type_name`)


def runner(parser, options, args):

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
            if options.output_path:
                print >> sys.stderr, "WARNING: overwriting output path %r with %r" % (options.output_path,  args[1])
            options.output_path = args[1]
        else:
            parser.error("incorrect number of arguments (expected upto 2 but got %s)" % (len(args)))
    
    options.input_path = fix_path(options.input_path)

    kernel_width = options.kernel_width or None
    kernel_type = options.kernel
    smoothing_method = options.method
    boundary_condition = options.boundary


    stack = ImageStack.load(options.input_path, options=options)
    voxel_sizes = stack.get_voxel_sizes()

    if kernel_width is None:
        dr = stack.get_lateral_resolution()
        dz = stack.get_axial_resolution()
        if dr is None or dz is None:
            kernel_width = 3
        else:
            print 'lateral resolution: %.3f um (%.1f x %.1f px^2)' % (1e6*dr, dr/voxel_sizes[1], dr/voxel_sizes[2])
            print 'axial resolution: %.3f um (%.1fpx)' % (1e6*dz, dz / voxel_sizes[0])
            vz,vy,vx = voxel_sizes
            m = 1
            scales = (m*vz/dz, m*vy/dr, m*vx/dr)

    if kernel_width is not None:
        w = float(kernel_width) * min (voxel_sizes)  
        scales = tuple([s/w for s in voxel_sizes])    

    print 'Window sizes:', [1/s for s in scales]

    kdims = [1+2*(int(numpy.ceil(1/s))//2) for s in scales]
    k = 'x'.join (map (str, kdims))

    if options.output_path is None:
        b,e = os.path.splitext(options.input_path)
        suffix = '_%s_%s%s' % (smoothing_method, kernel_type, k)
        options.output_path = b + suffix + (e or '.tif')

    options.output_path = fix_path(options.output_path)

    if options.link_function == 'identity':
        images = stack.images
    elif options.link_function == 'log':
        images = stack.images
        mn, mx = get_dtype_min_max (images.dtype)
        images = numpy.log(images)
        images[numpy.where (numpy.isnan(images))] = numpy.log(mn)
    else:
        raise NotImplementedError (`options.link_function`)

    new_images = regress (images, scales,
                          kernel = kernel_type,
                          method = smoothing_method,
                          boundary = boundary_condition,
                          verbose = verbose)

    if options.link_function == 'identity':
        pass
    elif options.link_function == 'log':
        new_images = numpy.exp(new_images)
        new_images = numpy.nan_to_num(new_images)
    else:
        raise NotImplementedError (`options.link_function`)

    if verbose:
        print 'Leak: %.3f%%' % ( 100*(1-new_images.sum ()/stack.images.sum ()))
        print 'MSE:', ((new_images - stack.images)**2).mean()
        print 'Energy:', ((stack.images)**2).sum()
        print 'Saving result to',options.output_path
    ImageStack(new_images, stack.pathinfo).save(options.output_path)


def main ():
    parser = OptionParser(__usage__)
    from iocbio.ops.script_options import set_regress_options
    set_regress_options (parser)
    if hasattr(parser, 'runner'):
        parser.runner = runner
    options, args = parser.parse_args()
    runner(parser, options, args)
    return

if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- python-mode -*-
# Author: Pearu Peterson
# Created: September 2009

from __future__ import division
import os

### START UPDATE SYS.PATH ###
### END UPDATE SYS.PATH ###

import numpy
from iocbio.optparse_gui import OptionParser
from iocbio.io import ImageStack
from iocbio.io.io import fix_path
from iocbio import utils
from iocbio.ops.script_options import set_estimate_snr_options

import time

def runner(parser, options, args):

    if not hasattr(parser, 'runner'):
        options.output_path = None
    
    options.input_path = fix_path(options.input_path)
    stack = ImageStack.load(options.input_path, options=options)
    voxel_sizes = stack.get_voxel_sizes()

    dr = stack.get_lateral_resolution()
    dz = stack.get_axial_resolution()
    if dr is not None:
        print 'lateral resolution: %.3f um (%.1f x %.1f px^2)' % (1e6*dr, dr/voxel_sizes[1], dr/voxel_sizes[2])
        print 'axial resolution: %.3f um (%.1fpx)' % (1e6*dz, dz / voxel_sizes[0])
        vz,vy,vx = voxel_sizes
        m = 0.5/2
        scales = (m*vz/dz, m*vy/dr, m*vx/dr)
    else:
        raise NotImplementedError ('get_lateral_resolution')

    kdims = [1+2*(int(numpy.ceil(1/s))//2) for s in scales]
    k = 'x'.join (map (str, kdims))
    print 'Averaging window box:', k

    kernel_type = options.kernel
    smoothing_method = options.method
    boundary_condition = options.boundary

    mn, mx = stack.images.min (), stack.images.max()
    high_indices = numpy.where(stack.images >= mn + 0.9*(mx-mn))
    high = stack.images[high_indices]

    from iocbio.ops import regress
    average = regress (stack.images, scales,
                       kernel = kernel_type,
                       method = smoothing_method,
                       boundary = boundary_condition,
                       verbose = True)

    noise = stack.images - average

    var = regress (noise*noise, scales,
                   kernel = kernel_type,
                   method = smoothing_method,
                   boundary = boundary_condition,
                   verbose = True)

    indices = numpy.where (var != 0)
    
    print len(numpy.where (var==0)[0]), var.shape, var.dtype
    var[numpy.where (var==0)] = 1
    snr = average / numpy.sqrt(var)
    snr1 = snr[indices]
    print 'STACK min, max = %s, %s' % (mn, mx)
    print 'SNR min, max, mean = %s, %s, %s' % (snr1.min (), snr1.max (), snr1.mean ())

    ImageStack(snr, pathinfo=stack.pathinfo).save('snr.tif')

def main ():
    parser = OptionParser()
    set_estimate_snr_options (parser)
    if hasattr(parser, 'runner'):
        parser.runner = runner
    options, args = parser.parse_args()
    runner(parser, options, args)

if __name__ == '__main__':
    main()

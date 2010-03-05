#!/usr/bin/env python
# -*- python-mode -*-
"""
Apply Poisson noise to a scalar field.
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
from iocbio.io.io import fix_path
from iocbio.ops.script_options import set_apply_noise_options
from scipy.stats import poisson

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

    if options.output_path is None:
        b,e = os.path.splitext(options.input_path)
        suffix = '_noised_%s' % (options.noise_type)
        options.output_path = b + suffix + (e or '.tif')

    options.output_path = fix_path(options.output_path)
    stack = ImageStack.load(options.input_path, options=options)

    if options.noise_type == 'poisson':
        new_images = stack.images.copy()
        new_images[new_images <= 0] = 1
        new_images = poisson.rvs(new_images)
    else:
        raise NotImplementedError(`options.noise_type`)
    if verbose:
        print 'Saving result to',options.output_path
    ImageStack(new_images, stack.pathinfo).save(options.output_path)


def main ():
    parser = OptionParser()
    set_apply_noise_options (parser)
    if hasattr(parser, 'runner'):
        parser.runner = runner
    options, args = parser.parse_args()
    runner(parser, options, args)
    return

if __name__ == '__main__':
    main()

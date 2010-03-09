#!/usr/bin/env python
# -*- python-mode -*-
"""
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
from iocbio.ops import convolve
from iocbio.io.io import fix_path
from iocbio.ops.script_options import set_convolve_options

def runner(parser, options, args):

    if not hasattr(parser, 'runner'):
        options.output_path = None

    if args:
        raise NotImplementedError (`args`)
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
    
    options.kernel_path = fix_path(options.kernel_path)
    options.input_path = fix_path(options.input_path)

    if options.output_path is None:
        b,e = os.path.splitext(options.input_path)
        suffix = '_convolved' % ()
        options.output_path = b + suffix + (e or '.tif')

    options.output_path = fix_path(options.output_path)

    kernel = ImageStack.load(options.kernel_path, options=options)
    stack = ImageStack.load(options.input_path, options=options)

    result = convolve (kernel.images, stack.images, options=options)

    if 1:
        print 'Saving result to',options.output_path
    ImageStack(result, stack.pathinfo).save(options.output_path)

def main ():
    parser = OptionParser()
    set_convolve_options (parser)
    if hasattr(parser, 'runner'):
        parser.runner = runner
    options, args = parser.parse_args()
    runner(parser, options, args)

if __name__ == '__main__':
    main()

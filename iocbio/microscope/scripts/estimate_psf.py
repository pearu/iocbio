#!/usr/bin/env python 
# -*- python-mode -*-
"""
Front end script for deconvolving microscope images.
Execute this script with --help for usage information.
"""
# Author: Pearu Peterson
# Created: May 2009

import os
import sys

### START UPDATE SYS.PATH ###
### END UPDATE SYS.PATH ###

from iocbio.optparse_gui import OptionParser
from iocbio.io.io import fix_path
from iocbio.microscope import spots_to_psf
from iocbio import utils

def runner(parser, options, args):

    if not hasattr(parser, 'runner'):
        options.output_path = None
    
    if args:
        if len(args)==1:
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
        if options.cluster_background_level:
            suffix = '_psf_cbl%s_fz%s' % (options.cluster_background_level, options.psf_field_size)
        else:
            suffix = '_psf_fz%s' % (options.psf_field_size,)
        options.output_path = b + suffix + (e or '.tif')

    psf_dir = utils.get_path_dir(options.output_path, 'iocbio.estimate_psf')
    try:
        psf = spots_to_psf(options.input_path, psf_dir, options)
    except KeyboardInterrupt:
        print 'RECEIVED CTRL-C'
        return

    options.output_path = fix_path(options.output_path)
    if options.output_path is not None:
        print 'Saving results to %r' % (options.output_path)
        psf.save (options.output_path)


def main():
    parser = OptionParser()
    from iocbio.microscope.script_options import set_estimate_psf_options
    set_estimate_psf_options (parser)

    if hasattr(parser, 'runner'):
        parser.runner = runner

    options, args = parser.parse_args()

    runner(parser, options, args)
    return

if __name__ == "__main__":
    main()

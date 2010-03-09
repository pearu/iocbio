#!/usr/bin/env python 
# -*- python-mode -*-
"""
Front end script for deconvolving microscope images.
Execute this script with --help for usage information.
"""
# Author: Pearu Peterson
# Created: May 2009

import os

### START UPDATE SYS.PATH ###
### END UPDATE SYS.PATH ###

from iocbio.optparse_gui import OptionParser
from iocbio.microscope.deconvolution import deconvolve
from iocbio.io import ImageStack
from iocbio.io.io import fix_path, get_psf_path
from iocbio.microscope.script_options import set_deconvolve_options

file_extensions = ['.tif', '.lsm', 'tiff', '.raw']

def runner (parser, options, args):
    
    if not hasattr(parser, 'runner'):
        options.psf_path = None
        options.input_path = None
        options.output_path = None

    if args:
        if len(args) in [2,3]:
            if options.psf_path:
                print >> sys.stderr, "WARNING: overwriting psf path %r with %r" % (options.psf_path,  args[0])
            options.psf_path = args[0]
            if options.input_path:
                print >> sys.stderr, "WARNING: overwriting input path %r with %r" % (options.input_path,  args[1])
            options.input_path = args[1]
        else:
            parser.error("Incorrect number of arguments (expected 2 or 3 but got %r)" % ((args)))
        if len(args)==3:
            if options.output_path:
                print >> sys.stderr, "WARNING: overwriting output path %r with %r" % (options.output_path,  args[2])
            options.output_path = args[2]

    options.input_path = fix_path (options.input_path)
    if options.output_path is None:
        deconvolve_dir = get_path_dir(options.input_path, 'ioc.deconvolve')
    else:
        deconvolve_dir = get_path_dir(options.output_path, 'ioc.deconvolve')

    psf_path = get_psf_path(options)

    psf = ImageStack.load(psf_path, options=options)
    stack = ImageStack.load(options.input_path, options=options)
    deconvolved_image = deconvolve(psf, stack, deconvolve_dir, options=options)

    if options.output_path is None:
        b,e = os.path.splitext(options.input_path)
        suffix = deconvolved_image.pathinfo.suffix
        options.output_path = b + suffix + (e or '.tif')
    options.output_path = fix_path(options.output_path)

    if 1:
        print 'Saving result to %r' % (options.output_path)
    deconvolved_image.save(options.output_path)

def get_path_dir(path, suffix):
    """ Return a directory name with suffix that will be used to save data
    related to given path.
    """
    if os.path.isfile(path):
        path_dir = path+'.'+suffix
    elif os.path.isdir(path):
        path_dir = os.path.join(path, suffix)
    elif os.path.exists(path):
        raise ValueError ('Not a file or directory: %r' % path)
    else:
        base, ext = os.path.splitext(path)
        if ext in file_extensions:
            path_dir = path +'.'+suffix
        else:
            path_dir = os.path.join(path, suffix)
    return path_dir

def main ():
    parser = OptionParser()
    set_deconvolve_options (parser)
    if hasattr(parser, 'runner'):
        parser.runner = runner
    options, args = parser.parse_args()
    runner(parser, options, args)
    return

if __name__ == "__main__":
    main()

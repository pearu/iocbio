#!/usr/bin/env python
# -*- python-mode -*-
"""
Converts image files to ome-tiff files.
"""
# Author: Pearu Peterson
# Created: October 2010

from __future__ import division
import os

from iocbio.io.io import get_pathinfo
from iocbio.optparse_gui import OptionParser
from iocbio.io.script_options import set_ome_options

### START UPDATE SYS.PATH ###
### END UPDATE SYS.PATH ###

def runner (parser, options, args):
    pathinfo = get_pathinfo(options.input_path)
    print pathinfo.omexml()


def main ():
    parser = OptionParser()
    set_ome_options (parser)
    if hasattr(parser, 'runner'):
        parser.runner = runner
    options, args = parser.parse_args()
    runner(parser, options, args)
    return

if __name__ == '__main__':
    main()

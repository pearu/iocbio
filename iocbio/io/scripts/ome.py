#!/usr/bin/env python
# -*- python-mode -*-
"""
Converts image files to ome-tiff files.
"""
# Author: Pearu Peterson
# Created: October 2010

from __future__ import division
import os

### START UPDATE SYS.PATH ###
### END UPDATE SYS.PATH ###

from iocbio.io.io import get_pathinfo
from iocbio.optparse_gui import OptionParser
from iocbio.io.script_options import set_ome_options



def runner (parser, options, args):
    pathinfo = get_pathinfo(options.input_path)
    if pathinfo is not None:
        pathinfo.omexml(options)
        return
    config_files = []
    for root, dirs, files in os.walk(options.input_path):
        if 'configuration.txt' in files:
            config_files.append(os.path.join (root, 'configuration.txt'))

    for config_file in config_files:
        pathinfo = get_pathinfo(config_file)
        protocol = pathinfo.get_protocol()
        if protocol=='rics':
            print 'Skipping RICS data in %s' % (config_file)
            continue
        pathinfo.omexml(options)

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

#!/usr/bin/env python
# -*- python -*-
# Author: Pearu Peterson
# Created: June 2010

from __future__ import division


### START UPDATE SYS.PATH ###
### END UPDATE SYS.PATH ###

import os
import time
import numpy
from iocbio.optparse_gui import OptionParser
from iocbio.io import ImageStack
from iocbio.io.io import fix_path
from iocbio import utils
from iocbio.analysis.script_options import set_sarcomere_length_options
from iocbio.analysis.sarcomere import sarcomere_length

def runner(parser, options, args):

    if not hasattr(parser, 'runner'):
        options.output_path = None

    options.input_path = fix_path(options.input_path)
    stack = ImageStack.load(options.input_path, options=options)
    voxel_sizes = stack.get_voxel_sizes()

    roi_center_line = [int(n.strip()) for n in options.roi_center_line.split(',')]
    assert len (roi_center_line)==4,`roi_center_line`
    roi_width = int (options.roi_width)

    N = options.nof_points

    lines = []
    time_lst = []
    last_lines = []

    for i,image in enumerate(stack.images):
        result, labels = sarcomere_length (image, voxel_sizes[1:], roi_center_line, roi_width, N)
        if not lines:
            for r in result:
                lines.append([])
                last_lines.append([])
        for j,r in enumerate (result):
            last_lines[j].append(r)
            if len(last_lines[j])>=2:
                last_lines[j].pop(0)                
            lines[j].append(numpy.mean (last_lines[j]))
        time_lst.append (i)

    import matplotlib.pyplot as plt
    for line in lines:
        plt.plot (time_lst, line)

    plt.legend (labels)
    plt.show ()

    #ImageStack(stack.images, pathinfo=stack.pathinfo).save('marked.tif')


def main ():
    parser = OptionParser()
    set_sarcomere_length_options (parser)
    if hasattr(parser, 'runner'):
        parser.runner = runner
    options, args = parser.parse_args()
    runner(parser, options, args)

if __name__ == '__main__':
    main()

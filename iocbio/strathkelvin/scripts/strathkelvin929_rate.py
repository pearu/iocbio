#!/usr/bin/env python
# -*- python-mode -*-
# Author: Pearu Peterson
# Created: June 2010

from __future__ import division
import os

### START UPDATE SYS.PATH ###
### END UPDATE SYS.PATH ###

import re
import numpy

from iocbio.optparse_gui import OptionParser
from iocbio.strathkelvin.script_options import set_strathkelvin929_rate_options
from iocbio.strathkelvin.model import DataSlope


def splitline(line):
    r = line.split()
    if len(r)==3:
        r.append ('')
    return r

def runner(parser, options, args):
    n = options.nof_regression_points

    for filename in args:
        lines = open (filename, 'r').readlines ()
        new_lines = []
        dt = None
        skip = []
        for i, line in enumerate(lines):
            if i in skip:
                continue
            if line.startswith('#'):
                if line.startswith('# Configuration.rate_regression_points :'):
                    line = line.split(':')[0] + ': ' + str(n) + '\n'
                new_lines.append(line)
            else:
                words = splitline(line)
                if dt is None:
                    words2 = splitline(lines[i+1])
                    dt = float (words2[0]) - float (words[0])
                    slope = DataSlope (dt, n)
                    slope.add(float(words[1]))
                    slope.add(float(words2[1]))
                    words[2] = str(slope.slope[0])
                    words2[2] = str(slope.slope[1])
                    new_lines.append ('%18s %18s %18s %s\n' % tuple(words))
                    new_lines.append ('%18s %18s %18s %s\n' % tuple(words2))
                    skip.append (i+1)
                    
                else:
                    slope.add(float(words[1]))
                    words[2] = str(slope.slope[-1])
                    new_lines.append ('%18s %18s %18s %s\n' % tuple(words))
        f = open (filename, 'w')
        f.write(''.join (new_lines))
        f.close ()

def main ():
    parser = OptionParser()

    set_strathkelvin929_rate_options (parser)
    if hasattr(parser, 'runner'):
        parser.runner = runner
    options, args = parser.parse_args()
    runner(parser, options, args)
    return

if __name__ == '__main__':
    main()

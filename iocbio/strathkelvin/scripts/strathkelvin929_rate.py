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
from datetime import datetime

from iocbio.optparse_gui import OptionParser
from iocbio.strathkelvin.script_options import set_strathkelvin929_rate_options
from iocbio.strathkelvin.model import DataSlope, get_rate_factor, get_rate_unit
from iocbio.timeunit import Time, Seconds

def splitline(line):
    r = line.split(None, 3)
    if len(r)==3:
        r.append ('')
    return r

def runner(parser, options, args):
    n = options.nof_regression_points
    tryrun = options.tryrun
    use_n_from_file = n==0
    start_time_format = '%y-%m-%d %H:%M'
    
    wrong_rates_count = 0
    for filename in args:
        lines = open (filename, 'r').readlines ()
        new_lines = []
        dt = None
        skip = []
        volume_ml = None
        protocol = None
        volume_ml_line = None
        start_time = None
        timeunit = None
        oxygenunit = None
        rate_factor = None
        rateunit = None
        normsum = 0
        newnormsum = 0
        count = 0
        other_volumes = dict()
        for i, line in enumerate(lines):
            if i in skip:
                continue
            if line.startswith('#'):
                if line.startswith('# Configuration.rate_regression_points :'):
                    if use_n_from_file:
                        n = int(line.split (':',1)[-1].strip())
                    line = line.split(':')[0] + ': ' + str(n) + '\n'
                elif line.startswith('#  protocol :'):
                    protocol = line.split (':',1)[-1].strip()
                elif line.startswith ('#  start time :'):
                    start_time = datetime.strptime(line.split(':',1)[-1].strip(), start_time_format)
                elif line.startswith ('# Configuration.oxygen_units :'):
                    oxygenunit = line.split(':', 1)[-1].strip()
                elif line.startswith ('# Configuration.time_units :'):
                    timeunit = line.split(':', 1)[-1].strip()
                elif 'volume_ml :' in line:
                    if line.startswith('# %s.' % (protocol)):
                        volume_ml = float(line.split (':',1)[-1].strip ())
                    else:
                        t1, t2 = line.split (':',1)
                        other_volumes[t1[1:-1].strip()] = float(t2.strip())
                new_lines.append(line)
            else:
                words = splitline(line.strip())
                if dt is None:
                    words2 = splitline(lines[i+1])
                    dt = Seconds(Time(words2[0], timeunit) - Time(words[0], timeunit))
                    if volume_ml is None:
                        print 'WARNING: No volume_ml found for protocol %r in %r' % (protocol, filename)
                        for t1, t2 in other_volumes.iteritems():
                            print '  Using existing %s=%s as volume_ml' % (t1, t2)
                            volume_ml = t2
                    rate_factor = get_rate_factor(volume_ml, timeunit)

                    rateunit = get_rate_unit (timeunit, oxygenunit)
                    slope = DataSlope (dt, n)
                    slope.add(float(words[1]))
                    slope.add(float(words2[1]))
                    normsum += float(words[2])**2
                    normsum += float(words2[2])**2
                    s0 = slope.slope[0]*rate_factor
                    s1 = slope.slope[1]*rate_factor
                    newnormsum += s0**2
                    newnormsum += s1**2
                    words[2] = str(s0)
                    words2[2] = str(s1)
                    new_lines.append ('%18s %18s %18s %s\n' % tuple(words))
                    new_lines.append ('%18s %18s %18s %s\n' % tuple(words2))
                    skip.append (i+1)
                    count += 2
                else:
                    slope.add(float(words[1]))
                    normsum += float(words[2])**2
                    s1 = slope.slope[-1]*rate_factor
                    newnormsum += s1**2
                    words[2] = str(s1)
                    new_lines.append ('%18s %18s %18s %s\n' % tuple(words))
                    count += 1
        create_double_rate_file = False
        if count:
            orig_norm = (normsum/count)**0.5
            new_norm = (newnormsum/count)**0.5
            if abs (orig_norm-new_norm)>1e-6:
                print 'WARNING: %r rates have %.1fx different norms (protocol=%s)' % (filename,
                                                                                      new_norm/orig_norm,
                                                                                      protocol)
                if abs(new_norm/orig_norm-2)<1e-3 and options.nof_regression_points==0:
                    create_double_rate_file = True
                #print '  orig_norm=%r' % (orig_norm)
                #print '  new_norm=%r' % (new_norm)
                #print '  volume_ml=%r' % (volume_ml)
                #print '  rate_factor=%r' % (rate_factor)
                #print '  rateunit=%r' % (rateunit)
                wrong_rates_count += 1
            else:
                print '%r rates are OK (protocol=%s)' % (filename, protocol)
        if not tryrun:
            print 'Writing oxygen rates to',filename
            f = open (filename, 'w')
            f.write(''.join (new_lines))
            f.close ()
            if os.path.isfile(filename+'_RATE_CORRECTION'):
                os.remove(filename+'_RATE_CORRECTION')
        else:
            if create_double_rate_file:
                try:
                    f = open(filename+'_RATE_CORRECTION', 'w')
                    f.write('%s' % (new_norm/orig_norm))
                    f.close()
                except Exception, msg:
                    print 'Ignoring %s' % (msg)

    if wrong_rates_count:
        print 'WARNING: %s channel files out of %s have wrong rates' % (wrong_rates_count, len (args))

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

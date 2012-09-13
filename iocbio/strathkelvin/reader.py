
# Author: Pearu Peterson
# Created: June 2010

import numpy
from datetime import datetime
from iocbio.timeunit import Time, Seconds
from collections import defaultdict

def splitline(line):
    r = line.split(None, 3)
    if len(r)==3:
        r.append ('')
    return r

def reader (filename):
    """ Reader of strathkelvin929 output files.
    
    Parameters
    ----------
    filename : str
      Path to experiment file.

    Returns
    -------
    data : dict
      Dictionary of time, oxygen, respiration_rate, and event arrays.
    info : dict
      Dictionary of experimental parameters.
    """
    start_time_format = '%y-%m-%d %H:%M'
    lines = open (filename, 'r').readlines ()
    timeunit = None
    info = {}
    keys = []
    data = defaultdict (list)
    events = {}
    for i, line in enumerate(lines):
        if line.startswith('#'):
            if line.startswith('# Configuration.rate_regression_points :'):
                n = int(line.split (':',1)[-1].strip())
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
            if ':' in line:
                key, value = line[1:].split (':', 1)
                info[key.strip()] = value.strip()
            elif not keys:
                for i,w in enumerate (line[1:].strip().split()):
                    if i%2:
                        keys[-1] += ' ' + w.strip()
                    else:
                        keys.append (w.strip())
                print keys
            else:
                index = None
                for i,w in enumerate(splitline(line[1:].strip())):
                    if i==0:
                        time = float(Seconds(Time (w, timeunit)))
                        for index, t in enumerate (data[keys[0]]):
                            if t > value:
                                break
                    elif i<3:
                        value = float (w)
                    else:
                        events[time] = w
                #print 'Unprocessed line: %r' % (line)
        else:
            for i,w in enumerate(splitline(line.strip())):
                if i==0:
                    value = float(Seconds(Time (w, timeunit)))
                elif i<3:
                    value = float (w)
                else:
                    value = w
                data[keys[i]].append (value)
    for k in keys[:-1]:
        data[k] = numpy.array (data[k])
    info['events'] = events
    return data, info

if __name__=='__main__':
    import sys
    filename = sys.argv[1]
    data, info = reader (filename)
    print info
    print data.keys ()
    from matplotlib import pyplot as plt
    plt.plot (data['time [min]'], data['oxygen [umol/l]'])
    plt.show ()
    #print info

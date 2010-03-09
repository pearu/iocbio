#!/usr/bin/env python
# -*- python-mode -*-
# Author: Pearu Peterson
# Created: August 2009

from __future__ import division
import os

### START UPDATE SYS.PATH ###
### END UPDATE SYS.PATH ###

import numpy
from iocbio.io import ImageStack
from iocbio.optparse_gui import OptionParser
from iocbio.io.io import fix_path, RowFile
from iocbio.utils import tostr

def runner (parser, options, args):
    
    run_method = getattr(parser, 'run_method', 'subcommand')
    if os.name=='posix' and run_method == 'subprocess':
        print 'This script cannot be run using subprocess method. Choose subcommand to continue.'
        return

    if args:
        if len (args)==1:
            if options.input_path:
                print >> sys.stderr, "WARNING: overwriting input path %r with %r" % (options.input_path,  args[0])
            options.input_path = args[0]

    if options.input_path is None:
        parser.error('Expected --input-path but got nothing')

    data, titles = RowFile (options.input_path).read(with_titles = True)
    if options.print_keys:
        print 'rowfile keys:' + ', '.join(data.keys())
        print 'length:', len(data[data.keys()[0]] or [])
        return

    if not options.x_keys and not options.y_keys:
        options.x_keys = titles[0]
        options.y_keys = ','.join(titles[1:])
        parser.save_options(options, [])

    if not options.x_keys or not options.y_keys:
        print 'No x or y keys specified for the plot.'
        return

    x_keys = [str (k).strip() for k in options.x_keys.split (',')]
    y_keys = [str (k).strip() for k in options.y_keys.split (',')]

    from mpl_toolkits.axes_grid.parasite_axes import HostAxes, ParasiteAxes
    import matplotlib.pyplot as plt
    fig = plt.figure(1, figsize=(12,6))
    host = HostAxes(fig, [0.1, 0.1, 0.55, 0.8])
    host.axis["right"].set_visible(False)
    fig.add_axes(host)

    host.set_title(options.input_path)

    legend_list = []
    axes_ylim = []
    plot_axes = []

    for x_key in x_keys:
        x_data = data.get(x_key)
        if x_data is None:
            print 'Rowfile does not contain a column named %r' % (x_key)
            continue
        offset = 20
        host.set_xlabel(x_key)
        for y_key in y_keys:
            plot_str = 'plot'
            if y_key.startswith('log:'):
                y_key = y_key[4:]
                plot_str = 'semilogy'
            y_data = data.get(y_key)
            if y_data is None:
                print 'Rowfile does not contain a column named %r' % (y_key)
                continue
            if axes_ylim:
                ax = ParasiteAxes(host, sharex=host)
                host.parasites.append(ax)
                #ax.axis["right"].set_visible(True)
                #ax.axis["right"].major_ticklabels.set_visible(True)
                #ax.axis["right"].label.set_visible(True)
                new_axisline = ax._grid_helper.new_fixed_axis
                ax_line = ax.axis["right2"] = new_axisline(loc="right",
                                                  axes=ax,
                                                  offset=(offset,0))
                if plot_str=='semilogy':
                    offset += 60
                else:
                    offset += 40
            else:
                ax = host
                ax_line = ax.axis['left']
            ax.set_ylabel(y_key)
            p, = getattr(ax, plot_str)(x_data, y_data, label = y_key)
            axes_ylim.append((ax, min (y_data), max (y_data)))
            plot_axes.append((p, ax_line))
    [ax.set_ylim(mn,mx) for ax,mn,mx in axes_ylim]
    #[ax_line.label.set_color (p.get_color()) for p, ax in plot_axes]
    host.legend()

    def on_keypressed(event):
        key = event.key
        if key=='q':
            sys.exit(0)

    fig.canvas.mpl_connect('key_press_event', on_keypressed)

    plt.draw()
    plt.show()

def main ():
    parser = OptionParser()
    from iocbio.io.script_options import set_rowfile_plot_options
    set_rowfile_plot_options(parser)
    if hasattr(parser, 'runner'):
        parser.runner = runner
    options, args = parser.parse_args()
    runner(parser, options, args)
    return

if __name__=="__main__":
    main()

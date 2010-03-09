
__all__ = ['set_show_options', 'get_io_options_group',
           'get_microscope_options_group', 'set_convert_options',
           'set_rowfile_plot_options']

import os
from optparse import OptionGroup, NO_DEFAULT

def set_show_options(parser):
    import matplotlib
    if os.name == 'posix':
        matplotlib.use('GTkAgg')
        parser.run_methods = ['subcommand']
    parser.set_usage('%prog [options] [ [-i] INPUT_PATH ]')
    parser.set_description('Display INPUT_PATH as 3D image.')

    parser.add_option ('--input-path', '-i',
                       type = 'file', metavar='INPUT_PATH',
                       help = 'Specify input PATH of 3D images. Default: %default.'
                       )
    parser.add_option('--interpolation', default='nearest',
                      choices = sorted(['bilinear', 'nearest','bicubic', 'spline16', 
                                        'spline36', 'hanning', 'hamming', 'hermit', 
                                        'kaiser', 'quadric', 'catrom', 'gaussian', 
                                        'bessel', 'mitchell', 'sinc', 'lanczos']),
                      help="Specify image interpolation method. Default: %default.")
    #parser.add_option('--roll-axis', choices=['0','1','2'], help='Roll given axis to 0. Default: %default.', default='0')
    parser.add_option('--projection', choices=['XY','XZ','YZ'], help='Project image stack to given plane. Default: %default.', default='XY')
    parser.add_option('--rgb', action='store_true', default=False,
                      help="Display as RGB(A) color images. Default: %default.")
    parser.add_option('--dpi', type='int', default=96,
                      help="Specify plot resolution. Default: %default.")
    import matplotlib.cm as cm
    colormap_names = [name for name in cm.datad if not name.endswith ('_r')]
    lcolormap_names = sorted([name for name in colormap_names if name[0].islower()])
    ucolormap_names = sorted([name for name in colormap_names if name[0].isupper()])
    colormap_names = lcolormap_names + ucolormap_names

    parser.add_option('--cmap', choices = colormap_names, default='gray',
                      help='Specify `colormap <http://matplotlib.sourceforge.net/plot_directive/mpl_examples/pylab_examples/show_colormaps.hires.png>`_. Default: %default.')
    parser.add_option ('--invert-cmap', action='store_true', default=False,
                       help='Invert specified colormap. Default: %default.')

    parser.add_option ('--histogram-bins', type='int', default=0,
                       help = 'Specify the number of bins for histogram. 0 means no histogram. Default: %default.')

    parser.add_option ('--auto-scale', action='store_true', default=False,
                       help = 'Automatically scale each frame. Default: %default.')

    parser.add_option_group(get_io_options_group(parser))
    parser.add_option_group(get_microscope_options_group (parser))

def get_io_options_group(parser, group=None):
    if group is None:
        group = OptionGroup(parser, 'I/O options',
                            description = 'Specify I/O options for reading/writing 3D images.')
    group.add_option('--max-nof-stacks', type='int',
                     help = 'Specify the maximum number of stacks for reading. Default: %default.')
    group.add_option('--max-nof-stacks-none', dest='max_nof_stacks', action='store_const',
                     const = 'none',
                     help = 'Unspecify the --max-nof-stacks option.')
    return group

def get_microscope_options_group(parser):
    group = OptionGroup (parser, 'Microscope options',
                         description = '''\
Specify microscope environment options. Note that these options should \
be used only when input images do not contain information about required options. \
Be warned that options specified here will override the options values \
found in input image files, so it is recommended to keep the fields of \
microscope options empty.''')
    group.add_option ("--objective-na", dest='objective_na',
                      type = 'float',  metavar='FLOAT',
                      help='Specify the numerical aperture of microscope objectve. Default: %default.')
    group.add_option ("--excitation-wavelength", dest='excitation_wavelength',
                      type = 'float',  metavar='FLOAT',
                      help='Specify excitation wavelength in nm. Default: %default.')
    group.add_option ("--emission-wavelength", dest='emission_wavelength',
                      type = 'float',  metavar='FLOAT',
                      help='Specify emission wavelength in nm. Default: %default.')
    group.add_option ("--refractive-index", dest='refractive_index',
                      type = 'float', default=NO_DEFAULT, metavar='FLOAT',
                      help='Specify refractive index of immersion medium: Default: %default.')
    group.add_option ("--microscope-type", dest='microscope_type',
                      choices = ['<detect>', 'confocal', 'widefield'],
                      default = '<detect>',
                      help = 'Specify microscope type: confocal|widefield. Default: %default',
                      )
    return group

def set_convert_options (parser):
    import numpy
    parser.set_usage('%prog [options] [ [-i] INPUT_PATH  [ [-o] OUTPUT_PATH ]]')
    parser.set_description('Convert INPUT_PATH to specified type and format.')

    parser.add_option ('--input-path','-i',
                       type = 'file', metavar='INPUT_PATH',
                       help = 'Specify input PATH of 3D images. Default: %default.'
                       )
    parser.add_option ('--output-path','-o',
                       type = 'file', metavar='OUTPUT_PATH',
                       help = 'Specify output PATH of 3D images. Default: %default.'
                       )
    numpy_types = ['<detect>'] + sorted(set([t.__name__ for t in numpy.typeDict.values()]))
    parser.add_option("--output-type", dest="output_type",
                      choices = numpy_types, default = numpy_types[0],
                      help="Specify output image stack type. Default: %default.")
    parser.add_option("--scale",
                      action="store_true", dest="scale",
                      help='Specify whether to scale stack to the limits of the output type. Default: %default.')
    parser.add_option("--no-scale",
                      action="store_false", dest="scale",
                      help='See --scale option.')
    parser.add_option('--normalize',
                      choices = ['none', 'unit volume'],
                      help = 'Specify normalization. Default: %default.'
                      )
    parser.add_option('--output-ext', dest='output_ext',
                      choices = ['tif', 'vtk', 'data'],
                      default = 'tif',
                      help="Specify output format extension. Default: %default.")

def set_rowfile_plot_options (parser):
    import matplotlib
    if os.name == 'posix' and 1:
        matplotlib.use('GTkAgg')
        parser.run_methods = ['subcommand']
    parser.set_usage ('%prog [options] [[-i] ROWFILE]')
    parser.set_description('Plot data in ROWFILE')
    parser.add_option ('--input-path', '-i',
                       type = 'file', metavar='ROWFILE',
                       help = 'Specify path to ROWFILE. Default: %default.'
                       )
    parser.add_option('--print-keys', action='store_true',
                      help = 'Print keys of the rowfile and exit. Default: %default.')

    parser.add_option('--x-keys',
                      help = 'Specify keys for x-axis. Default: %default.')
    parser.add_option('--y-keys',
                      help = 'Specify keys for y-axis. Default: %default. When not specified then use all keys.')

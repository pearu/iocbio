import os
import sys
import glob
from optparse import OptionGroup, NO_DEFAULT

print __file__,'will be removed in future, use subpackage/script_options.py instead'

def set_plot_photons_options(parser):
    import matplotlib
    if os.name == 'posix' and 1:
        matplotlib.use('GTkAgg')
        parser.run_methods = ['subcommand']
    parser.set_usage ('''\
%prog [options] [[-i] PATH]

Description:
  %prog makes photons count plot from each frame in PATH.
''')
    parser.add_option ('--input-path', '-i',
                       type = 'file', metavar='PATH',
                       help = 'Specify path to PATH. Default: %default.'
                       )

def set_rowfile_plot_options (parser):
    import matplotlib
    if os.name == 'posix' and 1:
        matplotlib.use('GTkAgg')
        parser.run_methods = ['subcommand']
    parser.set_usage ('''\
%prog [options] [[-i] ROWFILE]

Description:
  %prog makes plots from the data in ROWFILE.
''')
    parser.add_option ('--input-path', '-i',
                       type = 'file', metavar='ROWFILE',
                       help = 'Specify path to ROWFILE. Default: %default.'
                       )
    parser.add_option('--print-keys', action='store_true',
                      help = 'Print keys of the rowfile and exit. Default: %default.')

    parser.add_option('--x-keys',
                      help = 'Specify keys for x-axis. Default: %default.')
    parser.add_option('--y-keys',
                      help = 'Specify keys for y-axis. Default: %default.')

def set_rics_options (parser):
    parser.set_usage("""\
%prog [options] [ [-i] INPUT_PATH ]

Description:
  %prog scans INPUT_PATH for RICS measurments.""")

    parser.add_option ('--input-path', '-i',
                       type = 'file', metavar='INPUT_PATH',
                       help = 'Specify input PATH of RICS images. Default: %default.'
                       )
    parser.add_option ('--output-path', '-o',
                       type = 'file', metavar='PATH',
                       help = 'Specify output PATH. Default: %default.'
                       )
    parser.add_option ('--resolution-factor',
                       default = 1.0,
                       type = 'float',
                       help = 'Specify factor for Airy resolution. Default: %default.')

    parser.add_option('--resolution-slotsize-map',
                      default = '',
                      help = 'Specify resolution slotsize map using the following syntax: ([resolution:]slotsize)[,([resolution:]slotsize)]*. Default: %default.')

    parser.add_option_group(get_regress_options_group (parser))
    parser.add_option_group(get_microscope_options_group (parser))

def set_show_options(parser):
    import matplotlib
    if os.name == 'posix':
        matplotlib.use('GTkAgg')
        parser.run_methods = ['subcommand']
    parser.set_usage("""\
%prog [options] [ [-i] INPUT_PATH ]

Description:
  %prog displays INPUT_PATH as 3D image""")

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
                      help='Specify colormap. Possible colormaps are available for view: http://matplotlib.sourceforge.net/plot_directive/mpl_examples/pylab_examples/show_colormaps.hires.png. Default: %default.')
    parser.add_option ('--invert-cmap', action='store_true', default=False,
                       help='Invert specified colormap. Default: %default.')

    parser.add_option ('--histogram-bins', type='int', default=0,
                       help = 'Specify the number of bins for histogram. 0 means no histogram. Default: %default.')

    parser.add_option ('--auto-scale', action='store_true', default=False,
                       help = 'Automatically scale each frame. Default: %default.')

    parser.add_option_group(get_io_options_group(parser))
    parser.add_option_group(get_microscope_options_group (parser))

def add_psflib_options (parser):
    from .io import get_psf_libs, psflib_dir
    parser.add_option ('--psf-lib',
                       choices = ['<select>'] + sorted(get_psf_libs().keys ()),
                       help = 'Select PSF library name (psflib_dir=%r).'%psflib_dir \
                           + ' Note that specifying --psf-path|--kernel-path options override this selection. Default: %default.'
                       )

def set_convolve_options(parser):
    add_psflib_options(parser)
    parser.add_option ('--kernel-path','-k',
                       type = 'file', metavar='PATH',
                       help = 'Specify PATH to 3D images to be used as a kernel. Default: %default.'
                       )
    parser.add_option ('--input-path', '-i',
                       type = 'file', metavar='PATH',
                       help = 'Specify input PATH of 3D images. Default: %default.'
                       )
    parser.add_option ('--output-path', '-o',
                       type = 'file', metavar='PATH',
                       help = 'Specify output PATH of 3D images. Default: %default.'
                       )
    parser.add_option_group(get_fft_options_group (parser))


def set_deconvolve_options(parser):
    add_psflib_options(parser)
    parser.add_option('--psf-path','-k', dest = 'psf_path',
                      type = 'file', metavar='PATH',
                      help = 'Specify PATH to PSF 3D images. Default: %default.'
                      )
    parser.add_option ('--input-path', '-i',
                       type = 'file', metavar='PATH',
                       help = 'Specify input PATH of 3D images. Default: %default.'
                       )
    parser.add_option ('--output-path','-o',
                       type = 'file', metavar='PATH',
                       help = 'Specify output PATH of 3D images. Default: %default.'
                       )
    parser.add_option ('--max-nof-iterations',
                       type = 'int', default=10,
                       help = 'Specify maximum number of iterations. Default: %default.')
    parser.add_option('--convergence-epsilon',
                      type = 'float', default=0.05,
                      help = 'Specify small positive number that determines the window for convergence criteria. Default: %default.')
    parser.add_option('--degrade-input', action='store_true',
                      help = 'Degrade input: apply noise to convolved input. Default: %default.')
    parser.add_option('--no-degrade-input', action='store_false', dest='degrade_input',
                      help = 'See --degrade-input.')
    parser.add_option ('--first-estimate',
                      choices = ['input image',
                                 'convolved input image',
                                 '2x convolved input image',
                                 'last result'
                                 ],
                      help = 'Specify first estimate for iteration. Default: %default.')
    parser.add_option('--save-intermediate-results',
                      action = 'store_true',
                      help = 'Save intermediate results. Default: %default.')
    parser.add_option('--no-save-intermediate-results',
                      dest = 'save_intermediate_results',
                      action = 'store_false',
                      help = 'See --save-intermediate-results option.')
    parser.add_option_group(get_apply_window_options_group (parser))
    parser.add_option_group(get_rltv_options_group (parser))
    parser.add_option_group(get_fft_options_group (parser))
    parser.add_option_group(get_runner_options_group (parser))
    parser.add_option_group(get_microscope_options_group (parser))

def set_deconvolve_with_sphere_options (parser):
    parser.add_option ('--sphere-diameter', dest='diameter',
                       type = 'float',
                       default = 170,
                       help = 'Specify sphere diameter in nanometers. Default: %default.')
    parser.add_option ('--input-path', '-i',
                       type = 'file', metavar='PATH',
                       help = 'Specify input PATH of 3D images. Default: %default.'
                       )
    parser.add_option ('--output-path','-o',
                       type = 'file', metavar='PATH',
                       help = 'Specify output PATH of 3D images. Default: %default.'
                       )
    parser.add_option ('--max-nof-iterations',
                       type = 'int', default=10,
                       help = 'Specify maximum number of iterations. Default: %default.')
    parser.add_option('--convergence-epsilon',
                      type = 'float', default=0.05,
                      help = 'Specify small positive number that determines the window for convergence criteria. Default: %default.')

    parser.add_option_group(get_fft_options_group (parser))
    parser.add_option_group(get_rltv_options_group (parser))
    parser.add_option_group(get_microscope_options_group (parser))


def set_convert_options (parser):
    import numpy
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
                      help="Specify output image extension. Default: %default.")

def set_regress_options (parser):
    parser.add_option ('--quiet', dest='verbose', action='store_false',
                       help = 'Disable output messages. Default: %default')
    parser.add_option ('--input-path','-i',
                       type = 'file', metavar='PATH',
                       help = 'Specify input PATH of 3D images. Default: %default.'
                       )
    parser.add_option ('--output-path','-o',
                       type = 'file', metavar='PATH',
                       help = 'Specify output PATH of 3D images. Default: %default.'
                       )
    parser.add_option ("--kernel-width", dest="kernel_width",
                       type = 'float',
                       help="Specify the width of kernel in minimal voxel size unit. Default: %default."
                       )
    parser.add_option ('--link-function',
                       choices = ['identity'], default='identity',
                       help="Specify link function to transform data before regression analysis. Default: %default.")

    get_regress_options_group (parser, group=parser)
    parser.add_option_group(get_io_options_group (parser))
    parser.add_option_group(get_microscope_options_group (parser))

def get_regress_options_group(parser, group=None):
    if group is None:
        group = OptionGroup (parser, 'Regression options',
                             description = '''\
Specify options for applying regression.''')
        group.add_option('--regress', action='store_true',
                         help = 'Apply regress to input images. Default: %default.')
        group.add_option('--no-regress', action='store_false',
                         dest = 'regress',
                         help = 'See --regress option.')

    group.add_option('--method', dest='method',
                      choices = ['average', 'linear'],
                      default = 'linear',
                      help="Select smoothing method: average, linear. Default: %default")
    group.add_option('--kernel', dest='kernel',
                       choices = ['epanechnikov', 'uniform', 'triangular', 'quartic', 'triweight', 'tricube'],
                       default = 'uniform',
                       help="Select smoothing kernel: epanechnikov, uniform, triangular, quartic, triweight, tricube. Default: %default.")

    group.add_option ('--boundary', dest='boundary',
                      choices = ['constant', 'finite', 'periodic', 'reflective'],
                      default = 'finite',
                      help="Specify boundary condition: constant, finite, periodic, reflective. Default: %default.")

    return group

def set_apply_window_options (parser):
    parser.add_option ('--quiet', dest = 'verbose',
                       action = 'store_false', default=True,
                       help = 'Disable output messages. Default: %default')
    parser.add_option ('--input-path','-i',
                       type = 'file', metavar='PATH',
                       help = 'Specify input PATH of 3D images. Default: %default.'
                       )
    parser.add_option ('--output-path','-o',
                       type = 'file', metavar='PATH',
                       help = 'Specify output PATH of 3D images. Default: %default.'
                       )
    get_apply_window_options_group(parser, group=parser)
    parser.add_option_group(get_microscope_options_group (parser))

def set_apply_noise_options (parser):
    parser.add_option ('--quiet', dest = 'verbose',
                       action = 'store_false', default=True,
                       help = 'Disable output messages. Default: %default')
    parser.add_option ('--input-path','-i',
                       type = 'file', metavar='PATH',
                       help = 'Specify input PATH of 3D images. Default: %default.'
                       )
    parser.add_option ('--output-path','-o',
                       type = 'file', metavar='PATH',
                       help = 'Specify output PATH of 3D images. Default: %default.'
                       )
    parser.add_option ("--noise-type", dest='noise_type',
                      choices = ['poisson'],
                      default = 'poisson',
                      help = 'Specify noise type: poisson. Default: %default',
                      )

def set_estimate_psf_options (parser):
    parser.set_usage ('''\
%prog [options] [ [-i] INPUT_PATH [ [-k] OUTPUT_PATH ] ]

Description:
  %prog finds PSF estimate from the measurments of microspheres.
  Intermediate results are saved to INPUT_PATH/ioc.compute_psf/ directory
  and estimated PSF is saved to OUTPUT_PATH
''')
    parser.add_option ("--measurement-path", '--input-path','-i',
                       type = 'file', metavar='MEASUREMENT_PATH', dest='input_path',
                       help = '''Specify PATH to microsphere measurments.\
To select directory PATH, find a file PATH/{PATHINFO.txt, configuration.txt, SCANINFO.txt} and select it. Default: %default.''',
                       )

    parser.add_option ("--psf-path", '--output-path','-o','-k',
                       type = 'file', metavar='PSF_PATH', dest='output_path',
                       help = '''Specify PATH for saving estimated PSF. Default: %default.'''
                       )
    #parser.add_option('--photon-counter-offset',
    #                  type = 'float',
    #                  help = 'Specify photon counter offset. Default: %default.'
    #                  )
    parser.add_option('--subtract-background-field', dest='subtract_background_field',
                      action = 'store_true',
                      help='Specify that subtraction of a background field should be carried out. Default: %default.')

    parser.add_option('--no-subtract-background-field', dest='subtract_background_field',
                      action = 'store_false',
                      help='See --subtract-background-field option.')

    parser.add_option ("--cluster-background-level",
                       dest = 'cluster_background_level',
                       type = 'float',  metavar='FLOAT',
                       help = """\
Specify maximum background level for defining PSF clusters: smaller
value means larger clusters (a good thing) and higher probability of
overlapping clusters (a bad thing); default will be estimated. Default: %default.""")

    parser.add_option ("--psf-field-size",
                       dest = 'psf_field_size',
                       type = 'float', default=7, metavar='FLOAT',
                       help = 'Specify PSF field size in lateral and axial resolution units. Default: %default.'
                       )

    parser.add_option ("--show-plots",
                       dest = 'show_plots', action = 'store_true',
                       help = 'Show estimated PSF during computation. Default: %default.',
                       )
    parser.add_option ("--no-show-plots",
                       dest = 'show_plots', action = 'store_false',
                       help = 'See --show-plot option.',
                       )

    parser.add_option ("--save-intermediate-results",
                       dest = 'save_intermediate_results', action = 'store_true',
                       help = 'Save intermediate results of computation. Default: %default.',
                       )
    parser.add_option ('--no-save-intermediate-results',
                       dest = 'save_intermediate_results', action = 'store_false',
                       help = 'See --save-intermediate-results option.'
                       )
    parser.add_option ('--select-candidates',
                       help = 'Specify which canditates are selected for computing the PSF estimate. Default: %default.')

    parser.add_option_group(get_io_options_group (parser))
    parser.add_option_group(get_microscope_options_group (parser))

def get_apply_window_options_group(parser, group=None):
    if group is None:
        group = OptionGroup (parser, 'Apply window options',
                             description = '''\
Specify options for applying window.''')
        group.add_option('--apply-window', action='store_true',
                         help = 'Apply window to input images. Default: %default.')
        group.add_option('--no-apply-window', action='store_false',
                         dest = 'apply_window',
                         help = 'See --apply-window option.')


    group.add_option ('--smoothness',
                      choices = ['0','1','2','3'], default = '1', 
                      help = 'Specify smoothness parameter n, the window will be (2*n+1)x continuously differentiable. Default: %default')
    group.add_option('--window-width',  metavar='FLOAT',
                     type = 'float',
                     help = 'Specify the width of a cutting window in minimal voxel size unit. Default: %default')

    return group

def get_fft_options_group (parser):
    group = OptionGroup (parser, 'FFT algorithm options',
                         description = '''\
Specify options for FFT algorithm.''')
    group.add_option ('--float-type',
                      choices = ['single', 'double'], default = 'single',
                      help = 'Specify floating point number type to be used in FFT computations. Default: %default.')

    group.add_option ('--fftw-plan-flags',
                      choices = ['patient', 'measure', 'estimate', 'exhaustive'],
                      default = 'measure',
                      help = 'Specify FFTW plan flags. Default: %default.')
    group.add_option ('--fftw-threads',
                      type=int, default=1,
                      help = 'Specify the number of threads for FFTW plan. Default: %default.')

    return group

def get_rltv_options_group(parser):
    group = OptionGroup (parser, 'Richardson-Lucy algorithm options',
                         description = '''\
Specify options for Richardson-Lucy deconvolution algorithm with total variation term.''')

    group.add_option ('--rltv-lambda',
                      type = 'float',
                      help = 'Specify RLTV regularization parameter. Default: %default')
    group.add_option ('--rltv-estimate-lambda', dest='rltv_estimate_lambda',
                      action='store_true',
                      help = 'Enable estimating RLTV parameter lambda. Default: %default.')
    group.add_option ('--no-rltv-estimate-lambda', 
                      dest='rltv_estimate_lambda', action='store_false',
                      help = 'See --rltv-estimate-lambda option.')
    #group.add_option ('--rltv-estimate-lambda-method',
    #                  choices = ['scalar', 'smoothed scalar'], default = 'scalar',
    #                  help = 'Specify method for estimating lambda. Default: %default.')
    group.add_option ('--rltv-algorithm-type',
                      choices = ['multiplicative', 'additive'], default='multiplicative',
                      help = 'Specify algorithm type. Use multiplicative with Poisson noise and additive with Gaussian noise. Default: %default.')
    group.add_option ('--rltv-alpha',
                      type = 'float',
                      help = 'Specify additive RLTV regularization parameter. Default: %default.')
    group.add_option ('--rltv-stop-tau',
                      type = 'float',
                      help = 'Specify parameter for tau-stopping criteria. Default: %default.')
    #group.add_option('--rltv-save-intermediate-results',
    #                 action = 'store_true',
    #                 help = 'Save intermediate results. Default: %default.')
    #group.add_option('--no-rltv-save-intermediate-results',
    #                 dest = 'rltv_save_intermediate_results',
    #                 action = 'store_false',
    #                 help = 'See --rltv-save-intermediate-results option.')
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

def get_runner_options_group(parser):
    group = OptionGroup (parser, 'Runner options',
                         description = 'Specify options for running processes.')

    default_runner_subcommand = ''
    if os.environ.has_key('SGE_ROOT'):
        default_runner_subcommand = 'qrsh -l virtual_free=4G'
    group.add_option('--runner-subcommand',
                     default = default_runner_subcommand,
                     help = 'Specify command prefix for running subcommand. Default: %default.',
                     )

    return group


__all__ = ['set_convolve_options',
           'get_fft_options_group',
           'set_regress_options',
           'get_regress_options_group',
           'get_apply_window_options_group',
           'set_apply_window_options',
           'set_apply_noise_options']

import os
from optparse import OptionGroup, NO_DEFAULT

def set_apply_window_options (parser):
    from ..io.script_options import get_microscope_options_group
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

def set_convolve_options(parser):
    from ..microscope.script_options import add_psflib_options
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

def set_regress_options (parser):
    from ..io.script_options import get_microscope_options_group, get_io_options_group
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

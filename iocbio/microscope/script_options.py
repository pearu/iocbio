
from optparse import OptionGroup
from iocbio.script_options import set_formatter

__all__ = ['add_psflib_options', 'set_estimate_psf_options',
           'set_deconvolve_options','get_rltv_options_group',
           'set_deconvolve_with_sphere_options']

def set_clusters_options (parser):
    set_formatter(parser)
    parser.set_usage ('%prog [options] [ -i INPUT_PATH ]')
    parser.set_description('Find clusters in INPUT_PATH field.')
    parser.add_option ('--input-path','-i',
                       type = 'file', metavar='INPUT_PATH', dest='input_path',
                       help = 'Specify INPUT_PATH.'
                       )
    parser.add_option ('--output-path','-o',
                       type = 'file', metavar='OUTPUT_PATH', dest='output_path',
                       help = 'Specify OUTPUT_PATH.'
                       )
    parser.add_option ('--detail-size', dest='detail_size',type = 'float', 
                       help = 'Specify typical size of a detail to be resolved, in micrometers.')
    from iocbio.ops.script_options import get_regress_options_group
    parser.add_option_group(get_regress_options_group (parser))
    from iocbio.ops.script_options import get_fft_options_group
    parser.add_option_group(get_fft_options_group (parser))

def add_psflib_options (parser):
    from ..io.io import get_psf_libs, psflib_dir
    parser.add_option ('--psf-lib',
                       choices = ['<select>'] + sorted(get_psf_libs().keys ()),
                       help = 'Select PSF library name (psflib_dir=%r).'%psflib_dir \
                           + ' Note that specifying --psf-path|--kernel-path options override this selection.'
                       )

def set_estimate_psf_options (parser):
    set_formatter(parser)
    parser.set_usage('%prog [options] [ [-i] INPUT_PATH [ [-o] OUTPUT_PATH ] ]')
    parser.set_description('''Find PSF estimate from the measurments of microspheres.

Intermediate results are saved to INPUT_PATH/iocbio.estimate_psf/
directory and estimated PSF is saved to OUTPUT_PATH
''')
    parser.add_option ("--measurement-path", '--input-path','-i',
                       type = 'file', metavar='INPUT_PATH', dest='input_path',
                       help = '''Specify PATH to microsphere measurments.\
To select directory PATH, find a file PATH/{PATHINFO.txt, configuration.txt, SCANINFO.txt} and select it.''',
                       )

    parser.add_option ("--psf-path", '--output-path','-o','-k',
                       type = 'file', metavar='OUTPUT_PATH', dest='output_path',
                       help = '''Specify PATH for saving estimated PSF.'''
                       )
    #parser.add_option('--photon-counter-offset',
    #                  type = 'float',
    #                  help = 'Specify photon counter offset.'
    #                  )
    parser.add_option('--subtract-background-field', dest='subtract_background_field',
                      action = 'store_true',
                      help='Specify that subtraction of a background field should be carried out.')

    parser.add_option('--no-subtract-background-field', dest='subtract_background_field',
                      action = 'store_false',
                      help='See ``--subtract-background-field option``.')

    parser.add_option ("--cluster-background-level",
                       dest = 'cluster_background_level',
                       type = 'float',  metavar='FLOAT',
                       help = """\
Specify maximum background level for defining PSF clusters: smaller
value means larger clusters (a good thing) and higher probability of
overlapping clusters (a bad thing); default will be estimated. """)

    parser.add_option ("--psf-field-size",
                       dest = 'psf_field_size',
                       type = 'float', default=7, metavar='FLOAT',
                       help = 'Specify PSF field size in lateral and axial resolution units.'
                       )

    parser.add_option ("--show-plots",
                       dest = 'show_plots', action = 'store_true',
                       help = 'Show estimated PSF during computation.',
                       )
    parser.add_option ("--no-show-plots",
                       dest = 'show_plots', action = 'store_false',
                       help = 'See ``--show-plot option``.',
                       )

    parser.add_option ("--save-intermediate-results",
                       dest = 'save_intermediate_results', action = 'store_true',
                       help = 'Save intermediate results of computation.',
                       )
    parser.add_option ('--no-save-intermediate-results',
                       dest = 'save_intermediate_results', action = 'store_false',
                       help = 'See ``--save-intermediate-results`` option.'
                       )
    parser.add_option ('--select-candidates',
                       help = 'Specify which canditates are selected for computing the PSF estimate.')

    from ..io.script_options import get_io_options_group, get_microscope_options_group
    parser.add_option_group(get_io_options_group (parser))
    parser.add_option_group(get_microscope_options_group (parser))

def set_deconvolve_options(parser):
    set_formatter(parser)
    parser.set_usage('%prog [options] [ [-k] PSF_PATH [-i] INPUT_PATH [ [-o] OUTPUT_PATH] ]')
    parser.set_description('Deconvolve INPUT_PATH with PSF_PATH.')
    add_psflib_options(parser)
    parser.add_option('--psf-path','-k', dest = 'psf_path',
                      type = 'file', metavar='PATH',
                      help = 'Specify PATH to PSF 3D images.'
                      )
    parser.add_option ('--input-path', '-i',
                       type = 'file', metavar='PATH',
                       help = 'Specify input PATH of 3D images.'
                       )
    parser.add_option ('--output-path','-o',
                       type = 'file', metavar='PATH',
                       help = 'Specify output PATH of 3D images.'
                       )
    parser.add_option ('--max-nof-iterations',
                       type = 'int', default=10,
                       help = 'Specify maximum number of iterations.')
    parser.add_option('--convergence-epsilon',
                      type = 'float', default=0.05,
                      help = 'Specify small positive number that determines the window for convergence criteria.')
    parser.add_option('--degrade-input', action='store_true',
                      help = 'Degrade input: apply noise to convolved input.')
    parser.add_option('--no-degrade-input', action='store_false', dest='degrade_input',
                      help = 'See ``--degrade-input``.')
    parser.add_option('--degrade-input-snr',
                      type = 'float', default=0.0,
                      help = 'Specify the signal-to-noise ratio when using --degrade-input.'\
                          'If set to 0, then snr will be estimated as sqrt(max(input image)).')
    parser.add_option ('--first-estimate',
                      choices = ['input image',
                                 'convolved input image',
                                 '2x convolved input image',
                                 'last result'
                                 ],
                      help = 'Specify first estimate for iteration.')
    parser.add_option('--save-intermediate-results',
                      action = 'store_true',
                      help = 'Save intermediate results.')
    parser.add_option('--no-save-intermediate-results',
                      dest = 'save_intermediate_results',
                      action = 'store_false',
                      help = 'See ``--save-intermediate-results`` option.')
    from ..ops.script_options import get_apply_window_options_group
    parser.add_option_group(get_apply_window_options_group (parser))
    parser.add_option_group(get_rltv_options_group (parser))
    from ..ops.script_options import get_fft_options_group
    parser.add_option_group(get_fft_options_group (parser))
    from ..script_options import get_runner_options_group
    parser.add_option_group(get_runner_options_group (parser))
    from ..io.script_options import get_io_options_group, get_microscope_options_group
    parser.add_option_group(get_microscope_options_group (parser))

def get_rltv_options_group(parser):
    group = OptionGroup (parser, 'Richardson-Lucy algorithm options',
                         description = '''\
Specify options for Richardson-Lucy deconvolution algorithm with total variation term.''')
    group.add_option ('--rltv-estimate-lambda', dest='rltv_estimate_lambda',
                      action='store_true',
                      help = 'Enable estimating RLTV parameter lambda.')
    group.add_option ('--no-rltv-estimate-lambda', 
                      dest='rltv_estimate_lambda', action='store_false',
                      help = 'See ``--rltv-estimate-lambda`` option.')
    group.add_option ('--rltv-lambda-lsq-coeff',
                      type = 'float',
                      help = 'Specify coefficient for RLTV regularization parameter.'\
                          'If set to 0 then the coefficent will be chosed such that lambda_lsq_0==50/SNR.')
    group.add_option ('--rltv-lambda',
                      type = 'float',
                      help = 'Specify RLTV regularization parameter.')
    group.add_option ('--rltv-compute-lambda-lsq',
                      action='store_true',
                      help = 'Compute RLTV parameter estimation lambda_lsq.')
    group.add_option ('--rltv-algorithm-type',
                      choices = ['multiplicative', 'additive'], default='multiplicative',
                      help = 'Specify algorithm type. Use multiplicative with Poisson noise and additive with Gaussian noise.')
    group.add_option ('--rltv-alpha',
                      type = 'float',
                      help = 'Specify additive RLTV regularization parameter.')
    group.add_option ('--rltv-stop-tau',
                      type = 'float',
                      help = 'Specify parameter for tau-stopping criteria.')
    return group

def set_deconvolve_with_sphere_options (parser):
    set_formatter(parser)
    parser.set_usage('%prog [options] [ [-i] INPUT_PATH [ [-o] OUTPUT_PATH ] ]')
    parser.set_description('Deconvolve INPUT_PATH with sphere.')
    parser.add_option ('--sphere-diameter', dest='diameter',
                       type = 'float',
                       default = 170,
                       help = 'Specify sphere diameter in nanometers.')
    parser.add_option ('--input-path', '-i',
                       type = 'file', metavar='PATH',
                       help = 'Specify input PATH of 3D images.'
                       )
    parser.add_option ('--output-path','-o',
                       type = 'file', metavar='PATH',
                       help = 'Specify output PATH of 3D images.'
                       )
    parser.add_option ('--max-nof-iterations',
                       type = 'int', default=10,
                       help = 'Specify maximum number of iterations.')
    parser.add_option('--convergence-epsilon',
                      type = 'float', default=0.05,
                      help = 'Specify small positive number that determines the window for convergence criteria.')

    from ..ops.script_options import get_fft_options_group
    from ..io.script_options import get_microscope_options_group
    parser.add_option_group(get_fft_options_group (parser))
    parser.add_option_group(get_rltv_options_group (parser))
    parser.add_option_group(get_microscope_options_group (parser))


import os
from optparse import OptionGroup
from iocbio.script_options import set_formatter, get_runner_options_group

__all__ = ['set_run_mirrors_options',
           'get_nidaqmx_options_group',
           'get_mirror_parameter_options_group']

def set_run_mirrors_options(parser):
    set_formatter(parser)
    parser.set_usage ('%prog [options]')
    parser.set_description('Run scanning mirrors.')

    if os.name == 'posix':
        parser.run_methods = ['subcommand']

    from iocbio.scanner.configuration import camera_area_width, camera_area_height

    parser.add_option('--roi-x0', type='int', default=1)
    parser.add_option('--roi-x1', type='int', default=camera_area_width)
    parser.add_option('--roi-y0', type='int', default=1)
    parser.add_option('--roi-y1', type='int', default=camera_area_height)
    parser.add_option('--orientation-angle', type='float', default=0)

    parser.add_option ('--pixel-time-usec', type='float')
    parser.add_option ('--scan-speed', type='float')
    parser.add_option('--image-width', type='int')
    parser.add_option('--pixel-size-um', type='float')
    parser.add_option('--image-height', type='int', default=2)

    parser.add_option ('--flyback', type='float')
    parser.add_option ('--flyback-range', type='string')

    parser.add_option_group (get_mirror_parameter_options_group(parser))
    parser.add_option_group (get_runner_options_group(parser))

    try:
        parser.add_option_group (get_nidaqmx_options_group(parser))
    except ImportError, msg:
        print '%s' % (msg)

    parser.add_option ('--task', 
                       default = 'initialize',
                       choices = ['initialize',
                                  'plot',
                                  'measure',
                                  'scan'])


def get_nidaqmx_options_group(parser, group=None):
    import nidaqmx
    from nidaqmx.libnidaqmx import make_pattern
    if group is None:
        group = OptionGroup(parser, 'NIDAQmx options')

    phys_channel_choices = []
    if nidaqmx.get_nidaqmx_version() is not None:
        for dev in nidaqmx.System().devices:
            phys_channel_choices.extend(dev.get_digital_input_lines())

    pattern = make_pattern(phys_channel_choices)
    group.add_option ('--start-trigger-digital-output-lines',
                       type = 'string',
                       help = 'Specify digital lines as a pattern ['+pattern+'].')

    group.add_option ('--start-trigger-terminal',
                       type = 'string',
                       help = 'Specify digital lines as a pattern ['+pattern+'].')

    phys_channel_choices = []
    if nidaqmx.get_nidaqmx_version() is not None:
        for dev in nidaqmx.System().devices:
            phys_channel_choices.extend(dev.get_analog_output_channels())
    pattern = make_pattern(phys_channel_choices)
    group.add_option ('--mirror-x-analog-output-channels',
                       help = 'Specify physical channel as a pattern ['+pattern+'].')
    group.add_option ('--mirror-y-analog-output-channels',
                       help = 'Specify physical channel as a pattern ['+pattern+'].')

    phys_channel_choices = []
    if nidaqmx.get_nidaqmx_version() is not None:
        for dev in nidaqmx.System().devices:
            phys_channel_choices.extend(dev.get_analog_input_channels())
    pattern = make_pattern(phys_channel_choices)
    #group.add_option ('--mirror-x-analog-input-channels',
    #                   help = 'Specify physical channel as a pattern ['+pattern+']. Default: %default.')
    #group.add_option ('--mirror-x-analog-input-2-channels',
    #                   help = 'Specify physical channel as a pattern ['+pattern+']. Default: %default.')
    #group.add_option ('--mirror-y-analog-input-channels',
    #                   help = 'Specify physical channel as a pattern ['+pattern+']. Default: %default.')
    group.add_option ('--mirror-x-error-analog-input-channels',
                       help = 'Specify physical channel as a pattern ['+pattern+'].')
    #group.add_option ('--mirror-x-error-analog-input-2-channels',
    #                   help = 'Specify physical channel as a pattern ['+pattern+']. Default: %default.')
    group.add_option ('--mirror-y-error-analog-input-channels',
                       help = 'Specify physical channel as a pattern ['+pattern+'].')

    return group

def get_mirror_parameter_options_group(parser, group=None):
    if group is None:
        group = OptionGroup(parser, 'Mirror parameters')

    group.add_option ('--param-i-offset', type='float', default=0)
    group.add_option ('--param-j-offset', type='float', default=0)
    group.add_option ('--param-t-offset', type='float', default=0)

    #group.add_option ('--param-flyback-alpha', type='float', default=1.0)
    #group.add_option ('--param-flyback-beta', type='float', default=1.0)

    return group
         


__all__ = ['get_runner_options_group']

import os
from optparse import OptionGroup, NO_DEFAULT


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

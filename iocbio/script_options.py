
__all__ = ['get_runner_options_group', 'set_formatter']

import os
from optparse import OptionGroup, NO_DEFAULT
from optparse import TitledHelpFormatter

class MyHelpFormatter(TitledHelpFormatter):

    def format_option(self, option):
        old_help = option.help
        default = option.default
        if isinstance (default, str) and ' ' in default:
            default = repr (default)
        if option.help is None:
            option.help = 'Specify a %s.' % (option.type) 
        if option.type=='choice':
            choices = []
            for choice in option.choices:
                if choice==option.default:
                    if ' ' in choice:
                        choice = repr(choice)
                    choice = '['+choice+']'
                else:
                    if ' ' in choice:
                        choice = repr(choice)
                choices.append (choice)
            option.help = '%s Choices: %s.'% (option.help, ', '.join(choices))
        else:
            if default != NO_DEFAULT:
                if option.action=='store_false':
                    option.help = '%s Default: %s.'% (option.help, not default)
                else:
                    option.help = '%s Default: %s.'% (option.help, default)

        result = TitledHelpFormatter.format_option (self, option)
        option.help = old_help
        return result

help_formatter = MyHelpFormatter()

def set_formatter(parser):
    """Set customized help formatter.
    """
    parser.formatter = help_formatter

def get_runner_options_group(parser):
    group = OptionGroup (parser, 'Runner options',
                         description = 'Specify options for running processes.')

    default_runner_subcommand = ''
    if os.environ.has_key('SGE_ROOT'):
        default_runner_subcommand = 'qrsh -l virtual_free=4G'
    group.add_option('--runner-subcommand',
                     default = default_runner_subcommand,
                     help = 'Specify command prefix for running subcommand.',
                     )

    return group


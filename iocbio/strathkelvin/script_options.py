
__all__ = ['set_strathkelvin929_options',
           'set_strathkelvin929_rate_options']

from iocbio.script_options import set_formatter

def set_strathkelvin929_options (parser):
    set_formatter(parser)
    parser.set_usage('%prog')
    parser.set_description('Wrapper of Strathkelvin 929 System software, the GUI program.')

def set_strathkelvin929_rate_options(parser):
    set_formatter(parser)
    parser.set_usage('%prog [options] <channel-files>')
    parser.set_description('Re-calculates oxygen respiration rates.')

    parser.add_option ('--nof-regression-points', '-n',
                       type = 'int', default=10,
                       help = 'Specify the number of regression points')


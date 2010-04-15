
__all__ = ['set_strathkelvin929_options']

from iocbio.script_options import set_formatter

def set_strathkelvin929_options (parser):
    set_formatter(parser)
    parser.set_usage('%prog')
    parser.set_description('Wrapper of Strathkelvin 929 System software, the GUI program.')

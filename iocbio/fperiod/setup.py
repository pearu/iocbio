
from os.path import join
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('fperiod',parent_package,top_path)

    config.add_extension('imageinterp', sources = [join('src','imageinterp.c')],
                         define_macros = [('PYTHON_EXTENSION', None)])

    config.add_extension('fperiod_ext', sources = [
            join('src','fperiod.pyf'),
            join('src','iocbio_detrend.c'),
            join('src','iocbio_fperiod.c'),
            join('src','iocbio_ipwf.c'),
            ],
                         #define_macros = [('F2PY_REPORT_ATEXIT','1')]
                         )

    config.add_extension('ipwf', sources = [
            join('src','ipwf.pyf'),
            join('src','iocbio_ipwf.c'),
            ])
    return config

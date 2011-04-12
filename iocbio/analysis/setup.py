
from os.path import join
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('analysis',parent_package,top_path)
    config.add_extension('lineinterp', sources = [join('src','lineinterp.c')])

    config.add_extension('fperiod', sources = [
            join('src','fperiod.pyf'),
            join('src','fperiod.c')])
    return config

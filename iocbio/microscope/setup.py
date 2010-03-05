
from os.path import join

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('microscope',parent_package,top_path)
    config.add_extension('ops_ext', join('src','ops_ext.c'))
    return config

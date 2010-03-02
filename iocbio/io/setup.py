
from os.path import join

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('io',parent_package,top_path)
    #config.add_extension('ops_ext', join('src','ops_ext.c'))
    #config.add_extension('apply_window_ext', join('src','apply_window_ext.c'))
    #config.add_extension('regress_ext', join('src','regress_ext.c'))

    return config

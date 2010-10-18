
from os.path import join

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('io',parent_package,top_path)
    config.add_extension('_tifffile', join('src','tifffile.c'))
    config.add_data_files('ome.xsd')
    return config

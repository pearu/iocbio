
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('iocbio',parent_package,top_path)

    #config.add_subpackage('microscope')

    config.make_svn_version_py()
    return config

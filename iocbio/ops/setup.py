

from os.path import join

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('ops',parent_package,top_path)
    config.add_extension('apply_window_ext', join('src','apply_window_ext.c'))
    config.add_library('fminpack',
                       sources = [join ('src', 'minpack', '*.f'),
                                  ],
                       )
    config.add_extension('regress_ext', 
                         sources = [join('src','regress_ext.c'),
                                    ],
                         )
    config.add_extension('acf_ext', 
                         sources = [join('src','acf_ext.c'),
                                    join('src','acf.c')],
                         libraries = ['fminpack'])
    return config

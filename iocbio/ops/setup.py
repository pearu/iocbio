

from os.path import join

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, NotFoundError
    config = Configuration('ops',parent_package,top_path)
    config.add_extension('apply_window_ext', join('src','apply_window_ext.c'))
    config.add_extension('local_extrema_ext', join('src','local_extrema_ext.c'))

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

    
    fftw3_info = get_info('fftw3')

    if fftw3_info:
        config.add_extension('discrete_gauss_ext', 
                             sources = [join('src','discrete_gauss_ext.c'),
                                        join('src','discrete_gauss.c')],
                             extra_info = fftw3_info)
    else:
        print 'FFTW3 not found: skipping discrete_gauss_ext extension'
    return config

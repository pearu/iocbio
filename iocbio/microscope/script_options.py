
__all__ = ['add_psflib_options']

def add_psflib_options (parser):
    from ..io.io import get_psf_libs, psflib_dir
    parser.add_option ('--psf-lib',
                       choices = ['<select>'] + sorted(get_psf_libs().keys ()),
                       help = 'Select PSF library name (psflib_dir=%r).'%psflib_dir \
                           + ' Note that specifying --psf-path|--kernel-path options override this selection. Default: %default.'
                       )

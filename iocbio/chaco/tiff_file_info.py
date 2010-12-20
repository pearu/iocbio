
__all__ = ['TiffFileInfo']

import os
import glob
from enthought.traits.api import Property, Enum, cached_property, Bool
from enthought.traits.ui.api import View, Tabbed, Item, VGroup
from enthought.traits.ui.file_dialog import MFileDialogModel
from libtiff import TIFFfile
from .base_data_source import data_source_kinds


class TiffFileInfo(MFileDialogModel):

    description = Property(depends_on = 'file_name')
    #preview = Property (depends_on = 'file_name')
    #kind = data_source_kinds
    is_ok = Bool (False)

    traits_view = View(VGroup(
            #Tabbed(
            Item ('description', style='readonly', show_label = False, resizable=True),
            #Item ('preview', style='readonly', show_label = False, resizable=True),
            #    scrollable=True,
            #    ),
            #Item('kind', label='Open as', style='custom'),
            ),
            resizable=True)

    @cached_property
    def _get_description(self):
        self.is_ok = False
        if not os.path.isfile (self.file_name):
            if os.path.exists (self.file_name):
                if os.path.isdir (self.file_name):
                    files = []
                    for ext in ['tif', 'lsm']:
                        files += glob.glob(self.file_name+'/*.'+ext)
                    n = len (self.file_name)
                    files = sorted([f[n+1:] for f in files])
                    return 'Directory contains:\n%s' % ('\n'.join (files))
                return 'not a file'
            return 'file does not exists'
        if os.path.basename(self.file_name)=='configuration.txt':
            return unicode(open(self.file_name).read(), errors='ignore')
            raise NotImplementedError('opening configuration.txt data')
        try:
            tiff = TIFFfile(self.file_name, verbose=True)
        except ValueError, msg:
            return 'not a TIFF file\n%s' % (msg)
        self.is_ok = True
        try:
            r = tiff.get_info()
        except Exception, msg:
            r = 'failed to get TIFF info: %s' % (msg)
        #TODO: set self.preview object here
        tiff.close ()
        return unicode(r, errors='ignore')

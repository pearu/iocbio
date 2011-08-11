#!/usr/bin/env python

import os

try:
    from gui import Model
    import gui_resources
    import gui_python

    from utils import get_appdata_directory
    working_dir = r'c:\iocbio\_install'
    if not os.path.isdir (working_dir):
        print 'Making directory', working_dir
        os.makedirs(working_dir)
    print 'chdir', working_dir
    os.chdir(working_dir)

    resources = [
        'iocbio'
        #'libfftw3'
        ]

    Model(#'iocbio_installer.log'
          ).run (resources)

    print __file__,'normal exit'
except:
    import traceback
    traceback.print_exc ()
    print __file__,'abnormal exit'

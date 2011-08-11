#!/usr/bin/env python

import os
import sys

if 0 and 'admin' not in sys.argv and len (sys.argv)==1:
    import subprocess
    print subprocess.call(['runas', '/user:Administrator', '"\"%s\" admin"' % sys.argv[0]])
    raw_input('Press ENTER to close this program...')
    sys.exit()

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

    installer_log_tmpl = 'iocbio_installer_%d.log'
    i = 0
    installer_log = installer_log_tmpl % i
    while os.path.isfile(installer_log):
        i += 1
        installer_log = installer_log_tmpl % i

    resources = [
        'iocbio'
        ]

    Model(#installer_log
          ).run (resources)

    print __file__,'normal exit'
except:
    import traceback
    traceback.print_exc ()
    print __file__,'abnormal exit'

raw_input('Press ENTER to close this program...')
sys.exit(0)

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

    working_dir = r'c:\iocbio\_install'

    installer_log_tmpl = os.path.join(working_dir, 'iocbio_installer_%d.log')
    i = 0
    installer_log = installer_log_tmpl % i
    while os.path.isfile(installer_log):
        i += 1
        installer_log = installer_log_tmpl % i

    resources = [
        'iocbio'
        ]

    Model(logfile=installer_log,
          working_dir = working_dir
          ).run (resources)

    print __file__,'normal exit'
except:
    import traceback
    traceback.print_exc ()
    print __file__,'abnormal exit'

raw_input('Press ENTER to close this program...')
sys.exit(0)

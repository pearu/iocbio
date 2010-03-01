#!/bin/sh

PYTHONPATH= wine python setup.py  build --compiler=mingw32 install_lib bdist_wininst --install-script=iocbio_nt_post_install.py 

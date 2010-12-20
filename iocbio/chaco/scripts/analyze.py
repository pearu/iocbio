#!/usr/bin/env python
# -*- python-mode -*-
# Author: Pearu Peterson
# Created: December 2010

from __future__ import division
import sys

### START UPDATE SYS.PATH ###
### END UPDATE SYS.PATH ###

from iocbio.chaco.main import analyze

if len(sys.argv)==2:
    file_name = sys.argv[1]
else:
    file_name = None
analyze(file_name=file_name)

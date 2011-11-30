# Script to generate libfperioc.{c,h} files.
#
# Author: Pearu Peterson
# Created: November 2011

import re
source_start_re = re.compile(r'\A\w+\s+(?P<name>[\w_]+)\s*\(.*?\)\s*\Z').match
source_end_re = re.compile(r'\A}\s*\Z').match
doc_start_re = re.compile(r'\A\s*/[*][*]\s*\Z').match
doc_end_re = re.compile(r'\A\s*[*]/\s*\Z').match

macro_start_re = re.compile(r'\A\s*[#]define\s+(?P<name>[\w_]+\s*(\(\s*[\w_,\s]*\s*\)|))').match

def get_c_functions(filename):
    """ Scan C source file for C functions.
    """
    doc = None
    name = None
    source = None
    macros = {}
    macro = None
    macro_name = None
    in_doc = False
    in_source = False
    in_macro = False
    for line in open(filename).readlines():
        if in_doc:
            doc += line
            if doc_end_re(line):
                in_doc = False
            continue
        if in_source:
            source += line
            if source_end_re(line):
                in_source = False
                yield name, doc, source
                name = None
                doc = None
                source = None
            continue
        if in_macro:
            macro += line
            if not line.rstrip().endswith("\\"):
                in_macro = False
                yield macro_name, None, macro
                macro_name = None
                macro = None
            continue
        if doc_start_re(line):
            doc = line
            in_doc = True
            continue
        if not in_doc and doc_start_re(line):
            doc = line
            in_doc = True
            continue
        if not in_source:
            m = source_start_re(line)
            if m:
                name = m.group('name')
                source = line
                in_source = True
                continue
        if not in_macro:
            m = macro_start_re(line)
            if m:
                macro = line
                macro_name = m.group('name').replace(' ','')
                if not line.rstrip().endswith("\\"):
                    yield macro_name, None, macro
                    macro_name = None
                    macro = None
                else:
                    in_macro = True
                continue
        #print in_doc, in_source

if __name__=='__main__':
    import os
    import sys
    import glob
    dirname = os.path.dirname(__file__)
    source_files = ['iocbio_fperiod.c', 'iocbio_detrend.c', 'iocbio_ipwf.c']

    c_file = open(os.path.join(dirname, 'libfperiod.c'), 'w')
    h_file = open(os.path.join(dirname, 'libfperiod.h'), 'w')

    c_file.write('''\
/*
  Source code of libfperiod software library.

  This file is part of the IOCBio project but can be used as a
  standalone software program. We ask to acknowledge the use of the
  software in scientific articles by citing the following paper:

    Pearu Peterson, Mari Kalda, Marko Vendelin.
    Real-time Determination of Sarcomere Length of a Single Cardiomyocyte during Contraction.
    <Journal information to be updated>.

  See http://iocbio.googlecode.com/ for more information.

  License: BSD, see the footer of this file.
  Author: Pearu Peterson
  Created: November 2011
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "libfperiod.h"
''')
    h_file.write('''\
/*
  Header file for libfperiod.c. See the C source file for documentation.

  This file is part of the IOCBio project but can be used as a
  standalone software program. We ask to acknowledge the use of the
  software in scientific articles by citing the following paper:

    Pearu Peterson, Mari Kalda, Marko Vendelin.
    Real-time Determination of Sarcomere Length of a Single Cardiomyocyte during Contraction.
    <Journal information to be updated>.

  See http://iocbio.googlecode.com/ for more information.

  License: BSD, see the footer of libfperiod.c.
  Author: Pearu Peterson
  Created: November 2011
 */

#ifndef LIBFPERIOD_H
#define LIBFPERIOD_H

#ifdef __cplusplus
extern "C" {
#endif
''')
    for filename in source_files:        
        for (name, doc, source) in get_c_functions(os.path.join(dirname,filename)):
            if name in ['iocbio_fperiod', 'iocbio_fperiod_cached','iocbio_objective',
                        'iocbio_detrend', 'iocbio_detrend1',
                        'iocbio_ipwf_e11_find_zero','iocbio_ipwf_e11_evaluate',
                        'IFLOOR(X)', 'ISEXTREME(index)', 'ISMAX(index)', 'ISMIN(index)',
                        'UPDATE_DETREND1_ARRAY(GT,FORCE,N)',
                        'FRAC_1_3','FIXZERO(X)','EPSNEG','EPSPOS',
                        'MIN(X,Y)','MAX(X,Y)',
                        'iocbio_ipwf_e11_compute_coeffs',
                        'iocbio_ipwf_e11_find_zero_diff0',
                        'iocbio_ipwf_e11_find_zero_diff1',
                        'iocbio_ipwf_e11_find_zero_diff2',
                        'iocbio_ipwf_e11_find_zero_diff3',
                        'iocbio_ipwf_e11_compute_coeffs_diff0',
                        'iocbio_ipwf_e11_compute_coeffs_diff1',
                        'iocbio_ipwf_find_real_zero_in_01_2',
                        'iocbio_ipwf_linear_approximation_1_3',
                        'iocbio_ipwf_e11_compute_coeffs_diff2',
                        'iocbio_ipwf_linear_approximation_1_1',
                        'iocbio_ipwf_e11_compute_coeffs_diff3',
                        'iocbio_ipwf_linear_approximation_3_0',
                        ]:
                if doc is not None:
                    c_file.write(doc)
                c_file.write(source)
                if source.startswith('#define'):
                    pass
                else:
                    h_file.write('extern %s;\n' % (source.split('\n', 1)[0]))
            else:
                pass
                print 'skipping %s in %s' % (name, filename)
    h_file.write('''\
#ifdef __cplusplus
}
#endif

#endif
''')
    c_file.write ('''\
/*
Copyright (c) 2011, Pearu Peterson
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
''')
    c_file.close()
    h_file.close()

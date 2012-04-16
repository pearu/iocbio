#!/usr/bin/env python
# -*- python-mode -*-
"""
Collect image stacks and sum them to one.
"""
# Author: Pearu Peterson
# Created: April 2012

from __future__ import division
import os

### START UPDATE SYS.PATH ###
### END UPDATE SYS.PATH ###

import numpy
from iocbio.io import ImageStack
from iocbio.optparse_gui import OptionParser
from iocbio.io.io import fix_path
from iocbio.io.script_options import set_sumstacks_options

def get_dtype_min_max(dtype):
    """
    Return possible minimum and maximum values of an integer type.
    """
    if isinstance (dtype, type):
        type_name = dtype.__name__
    else:
        type_name = str(dtype)
    if type_name.startswith ('uint'):
        return 0, getattr (numpy, type_name)(-1)
    elif type_name.startswith('int'):
        bits = int(type_name[3:])
        return int (getattr(numpy, type_name) (2**(bits-1))), int(getattr(numpy, type_name) (2**(bits-1)-1))
    raise NotImplementedError (`dtype, type_name`)

def runner (parser, options, args):
    
    if not hasattr(parser, 'runner'):
        options.output_path = None

    if args:
        if len (args)==1:
            if options.input_path:
                print >> sys.stderr, "WARNING: overwriting input path %r with %r" % (options.input_path,  args[0])
            options.input_path = args[0]
        elif len(args)==2:
            if options.input_path:
                print >> sys.stderr, "WARNING: overwriting input path %r with %r" % (options.input_path,  args[0])
            options.input_path = args[0]
            if options.output_path:
                print >> sys.stderr, "WARNING: overwriting output path %r with %r" % (options.output_path,  args[1])
            options.output_path = args[1]
        else:
            parser.error("Incorrect number of arguments (expected upto 2 but got %s)" % (len(args)))

    if options.input_path is None:
        parser.error('Expected --input-path but got nothing')

    options.input_path = fix_path (options.input_path)

    stack = ImageStack.load(options.input_path, options=options)
    numpy_types = numpy.typeDict.values()
    if options.output_type in ['<detect>', None]:
        output_type_name = stack.images.dtype.name
    else:
        output_type_name = options.output_type.lower()
    output_type = getattr (numpy, output_type_name, None)

    nof_stacks = stack.get_nof_stacks()
    old_shape = stack.images.shape
    new_shape = (nof_stacks, old_shape[0]//nof_stacks) + old_shape[1:]

    new_images = numpy.zeros (new_shape[1:], dtype=output_type_name)

    first_stack = None
    last_stack = None
    for i, stacki in enumerate(stack.images.reshape(new_shape)):
        if i==0:
            first_stack = stacki.astype (float)
            new_images[:] = stacki
        else:
            err_first = abs(stacki - first_stack).mean()
            err_last = abs(stacki - last_stack).mean()
            print ('Stack %i: mean abs difference from first and last stack: %.3f, %.3f' % (i+1, err_first, err_last))
            new_images += stacki
        last_stack = stacki.astype(float)

    output_path = options.output_path
    output_ext = options.output_ext
    if output_path is None:
        dn = os.path.dirname(options.input_path)
        bn = os.path.basename(options.input_path)
        if os.path.isfile(options.input_path):
            fn, ext = os.path.splitext (bn)
            fn += '_sumstacks%s' % (nof_stacks)
            type_part = None
            for t in numpy_types:
                if fn.endswith('_' + t.__name__):
                    type_part = t.__name__
                    break
            if type_part is None:
                output_path = os.path.join(dn, fn + '_' + output_type_name + '.' + output_ext)
            else:
                output_path = os.path.join(dn, fn[:-len(type_part)] + output_type_name + '.' + output_ext)
        elif os.path.isdir (options.input_path):
            bn += '_sumstacks%s' % (nof_stacks)
            output_path = os.path.join (dn, bn+'_'+output_type_name + '.' + output_ext)
        else:
            raise NotImplementedError ('%s is not file nor directory' % (options.input_path))

    output_path = fix_path(output_path)

    print 'Saving new stack to',output_path

    if output_ext=='tif':
        ImageStack(new_images, stack.pathinfo, options=options).save(output_path)
    elif output_ext=='data':
        from iocbio.microscope.psf import normalize_unit_volume, discretize
        value_resolution = stack.pathinfo.get_value_resolution()
        normal_images = normalize_unit_volume(new_images, stack.get_voxel_sizes())
        discrete = discretize(new_images / value_resolution)
        signal_indices = numpy.where(discrete>0)

        new_value_resolution = value_resolution * normal_images.max() / new_images.max()

        ImageStack(normal_images, stack.pathinfo,
                   value_resolution = new_value_resolution).save(output_path, zip(*signal_indices))
    elif output_ext=='vtk':
        from pyvtk import VtkData, StructuredPoints, PointData, Scalars
        vtk = VtkData (StructuredPoints (new_images.shape), PointData(Scalars(new_images.T.ravel())))
        vtk.tofile(output_path, 'binary')
    else:
        raise NotImplementedError (`output_ext`)

def main ():
    parser = OptionParser()
    set_sumstacks_options (parser)
    if hasattr(parser, 'runner'):
        parser.runner = runner
    options, args = parser.parse_args()
    runner(parser, options, args)
    return

if __name__ == '__main__':
    main()

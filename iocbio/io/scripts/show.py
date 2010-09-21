#!/usr/bin/env python
# -*- python-mode -*-
# Author: Pearu Peterson
# Created: August 2009

from __future__ import division
import os

### START UPDATE SYS.PATH ###
### END UPDATE SYS.PATH ###

import numpy
from iocbio.io import ImageStack
from iocbio.optparse_gui import OptionParser
from iocbio.io.io import fix_path
from iocbio.utils import tostr, Options
from iocbio.io.script_options import set_show_options

def runner (parser, options, args):
    
    options = Options (options)

    if not hasattr(parser, 'runner'):
        options.output_path = None

    if args:
        if len (args)==1:
            if options.input_path:
                print >> sys.stderr, "WARNING: overwriting input path %r with %r" % (options.input_path,  args[0])
            options.input_path = args[0]

    if options.input_path is None:
        parser.error('Expected --input-path but got nothing')

    options.input_path = fix_path (options.input_path)

    stack = ImageStack.load(options.input_path, options=options)
    images = stack.images

    roll_axis = dict(XY=0, XZ=1, YZ=2).get(options.projection, 0)
    axis_labels = ('Z', 'Y', 'X')
    #roll_axis = int (options.roll_axis)
    voxel_sizes = stack.get_voxel_sizes()
    resolution = ['', '', '']

    title = '%s[%s]' % (os.path.basename (options.input_path), images.dtype)
    dr = stack.get_lateral_resolution()
    dz = stack.get_axial_resolution()

    if voxel_sizes:
        if dr is not None:
            resolution[1] = tostr(dr/voxel_sizes[1]) + 'px'
            resolution[2] = tostr(dr/voxel_sizes[2]) + 'px'
        if dz is not None:
            resolution[0] = tostr(dz/voxel_sizes[0]) + 'px'

    resolution = tuple (resolution)
    if roll_axis:
        images = numpy.rollaxis(images, roll_axis)
        if voxel_sizes:
            voxel_sizes = (voxel_sizes[roll_axis],) + voxel_sizes[:roll_axis] + voxel_sizes[roll_axis+1:]
        axis_labels = (axis_labels[roll_axis],) + axis_labels[:roll_axis] + axis_labels[roll_axis+1:]
        resolutions = (resolution[roll_axis],) + resolution[:roll_axis] + resolution[roll_axis+1:]

    if voxel_sizes:
        xlabel = '%s, resol=%s, px size=%sum, size=%sum' \
            % (axis_labels[-1], resolution[-1], tostr(voxel_sizes[-1]*1e6),  tostr(voxel_sizes[-1]*1e6*images.shape[-1]))
        ylabel = '%s, resol=%s, px size=%sum, size=%sum' \
            % (axis_labels[-2], resolution[-2], tostr(voxel_sizes[-2]*1e6),  tostr(voxel_sizes[-2]*1e6*images.shape[-2]))
    else:
        xlabel = '%s' % (axis_labels[-1])
        ylabel = '%s' % (axis_labels[-2])

    import matplotlib.cm as cm
    import matplotlib.pyplot as pyplot
    from iocbio.io.tifffile import imshow

    if options.invert_cmap:
        cmap = getattr(cm, options.cmap+'_r', options.cmap)
    else:
        cmap = getattr(cm, options.cmap, 'gray')

    view_3d = options.get(view_3d = '')
    if view_3d:
        l = []
        for i,d in enumerate(view_3d.split (',')):
            d = d.strip()
            if d.lower()=='c':
                d = images.shape[i]//2
            else:
                try:
                    d = int(d)
                except:
                    d = images.shape[i]//2
            d = max(min(images.shape[i]-1, d), 0)
            l.append(d)
        view_3d = l

    def mark_image (image, c1, c2, ln, wd, mx):
        image[c1-wd:c1+wd+1,:ln] = mx
        image[c1-wd:c1+wd+1,-ln:] = mx
        image[:ln,c2-wd:c2+wd+1] = mx
        image[-ln:,c2-wd:c2+wd+1] = mx
        return image

    if view_3d:
        import scipy.ndimage as ndimage
        pyplot.rc('font', family='sans-serif', weight='normal', size=8)
        figure = pyplot.figure(dpi=options.get(dpi=96), 
                               figsize=(8, 8), frameon=True,
                               facecolor='1.0', edgecolor='w')

        yz_scale = voxel_sizes[1] / voxel_sizes[0]
        zx_scale = voxel_sizes[0] / voxel_sizes[2]
        yx_scale = voxel_sizes[1] / voxel_sizes[2]

        yx_image = images[view_3d[0]].copy ()
        zx_image = images[:,view_3d[1],:].copy ()
        yz_image = images[:,:,view_3d[2]].T.copy()

        image = numpy.zeros((yx_image.shape[0] + zx_image.shape[0]+1, yx_image.shape[1]+yz_image.shape[1]+1))
        wd = image.shape[0]//300
        ln = max(1, image.shape[1]//20)
        mx = yx_image.max()
        
        def fix_image (image, scale, c1, c2, ln, wd):
            mx = image.max()
            if scale==1:
                mark_image (image, c1, c2, ln, wd, mx)
            elif scale>1:
                image = ndimage.interpolation.zoom(image, [scale, 1.0], order=0)
                mark_image (image, c1*scale, c2, ln, wd, mx)
            else:
                image = ndimage.interpolation.zoom(image, [1.0, 1.0/scale], order=0)
                mark_image (image, c1, c2/scale, ln, wd, mx)
            return image

        yx_image = fix_image(yx_image, yx_scale, view_3d[1], view_3d[2], ln, wd)
        zx_image = fix_image(zx_image, zx_scale, view_3d[0], view_3d[2], ln, wd)
        yz_image = fix_image(yz_image, yz_scale, view_3d[1], view_3d[0], ln, wd)

        image = numpy.zeros((yx_image.shape[0] + zx_image.shape[0]+1, yx_image.shape[1]+yz_image.shape[1]+1))
        image[:yx_image.shape[0], :yx_image.shape[1]] = yx_image
        image[:yx_image.shape[0], yx_image.shape[1]+1:] = yz_image
        image[yx_image.shape[0]+1:, :yx_image.shape[1]] = zx_image
        image_plot = pyplot.imshow(image, 
                              interpolation=options.get(interpolation='nearest'),
                              cmap = cmap,
                              )
        pyplot.title(title, size=11)
        axes = pyplot.gca()

        xtickdata = [(0,'0'),
                     #(yx_image.shape[1]/2, 'X'),
                     (yx_image.shape[1],  '%.1fum' % (voxel_sizes[2]*images.shape[2]*1e6)),
                     ( (yx_image.shape[1]+image.shape[1])/2, 'Z'),
                     (image.shape[1]-1, '%.1fum' % (voxel_sizes[0]*images.shape[0]*1e6)),
                     (yx_image.shape[1]*view_3d[2]/images.shape[2],'X=%spx' % (view_3d[2])),
                     ]
        ytickdata = [(0,'0'),
                     #(yx_image.shape[0]/2, 'Y'),
                     (yx_image.shape[0], '%.1fum' % (voxel_sizes[1]*images.shape[1]*1e6)),
                     #((yx_image.shape[0]+image.shape[0])/2, 'Z'),
                     ( image.shape[0]-1, '%.1fum' % (voxel_sizes[0]*images.shape[0]*1e6)),
                     (yx_image.shape[0]*view_3d[1]/images.shape[1],'Y=%spx' % (view_3d[1])),
                     (yx_image.shape[0]+yz_image.shape[1]*view_3d[0]/images.shape[0], 'Z=%spx' % (view_3d[0])),
                     ]

        xtickdata.sort ()
        xticks, xticklabels = zip(*xtickdata)

        ytickdata.sort ()
        yticks, yticklabels = zip(*ytickdata)


        axes.set_xticks(xticks)
        axes.set_xticklabels (xticklabels)
        axes.set_yticks(yticks)
        axes.set_yticklabels (yticklabels)

        cbar = pyplot.colorbar(shrink=0.8)

        def on_keypressed(event):
            """Callback for key press event."""
            key = event.key
            if key == 'q':
                sys.exit(0)
        figure.canvas.mpl_connect('key_press_event', on_keypressed)
    else:
        figure, subplot, image = imshow(images, title = title,
                                        #miniswhite=page.photometric=='miniswhite',
                                        interpolation=options.interpolation,
                                        cmap=cmap,
                                        dpi=options.dpi, isrgb=options.rgb,
                                        show_hist = options.histogram_bins,
                                        auto_scale = options.auto_scale)
        axes = figure.get_axes()[0]
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
    output_path = options.get (output_path='')
    if output_path:
        pyplot.savefig (output_path)
        print 'wrote',output_path
        sys.exit(0)
    pyplot.show()

def main ():
    parser = OptionParser()
    set_show_options (parser)
    if hasattr(parser, 'runner'):
        parser.runner = runner
    options, args = parser.parse_args()
    runner(parser, options, args)
    return

if __name__=="__main__":
    main()

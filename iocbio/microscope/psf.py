"""Provides spots_to_psf function.

Notes for measuring PSF
-----------------------

To measure a PSF (point spread function) of an optical system, we use
microspheres (diameter 0.17um) that are placed on a slide with
approximate distance of (PFI) from each other. A stack of images is
taken with high enough resolution that guarantees good representation
of PSF as a function. The values of PSF (photon counts) must be small
to ensure that photon counter will not overflow.  The stack should
contain many (say, 7-10) PSF images to improve statistical analysis.
The stack size should be large enough so that the first and last
image in the stack would contain only background signal (or a very
small fingerprint of PSF compared to overall stack signal).

PSF estimation protocol
-----------------------

The estimated PSF can be found using the following protocol:

Superpose PSF canditates

  PSF measurments contain images of many microspheres. The
  :func:`spots_to_psf` function finds the corresponding clusters of
  PSF candidates and sums them together. 

Smoothen PSF superposition

  Smoothening PSF superposition will increase the stability of
  deconvolution with a sphere. Use :func:`iocbio.ops.regress`
  function for smoothening PSF data.

Periodize PSF superposition

  Periodizing PSF superposition will minimize the effects from
  non-periodic boundaries. Use :func:`iocbio.ops.apply_window`
  function.

Deconvolve PSF superposition with sphere

  Deconvolve PSF superposition with sphere eliminates the effects from
  microspheres having finite size.

Smoothen the result of deconvolution with sphere

  Deconvolution tends to amplify random signals while PSF is
  presumably a smooth function. Use :func:`iocbio.ops.regress`.


"""

__all__ = ['spots_to_psf']
__autodoc__ = __all__ + ['find_clusters']

import os
import sys
import numpy
import scipy.optimize
import scipy.signal
import scipy.ndimage
from .. import utils
from .cluster_tools import find_clusters
from ..io import ImageStack
#from .differentiator import coeffs_1st_order
from ..utils import tostr

def implot(image, image_title=''):

    from matplotlib.pylab import imshow, show, title
    imshow(image)
    title(image_title)
    show()
    return

    from multiprocessing import Process

    def func(image, image_title):
        from matplotlib.pylab import imshow, show, title
        imshow(image)
        title(image_title)
        show()
        
    p = Process(target=func, args=(image,image_title))
    p.start()
    p.join()

def mul(lst):
    return reduce(lambda x,y:x*y, lst, 1.0)

def normalize_uint8(images):
    min = images.min()
    max = images.max()
    return (255.0 * (images - min) / (max - min)).astype(numpy.uint8)

def discretize (images):
    min = images.min()
    max = images.max()
    if min>=0 and max<2**8:
        t = numpy.uint8
    elif min>=-2**8 and max<2**8:
        t = numpy.int8
    elif min>=0 and max<2**16:
        t = numpy.uint16
    elif min>=-2**16 and max<2**16:
        t = numpy.int16
    elif min>=0 and max<2**32:
        t = numpy.uint32
    elif min>=-2**32 and max<2**32:
        t = numpy.int32
    elif min>=0 and max<2**64:
        t = numpy.uint64
    elif min>=-2**64 and max<2**64:
        t = numpy.int64
    else:
        t = int
    return images.astype(t)
        
def normalize_unit_volume(images, voxel_sizes):
    voxel_volume = mul(voxel_sizes) * 1e18
    integral = images.sum() * voxel_volume
    return images / integral

def typename(dtype):
    if isinstance(dtype, type):
        return dtype.__name__
    return dtype.name

def highertype(dtype):
    if isinstance(dtype, numpy.ndarray):
        return dtype.astype(highertype(dtype.dtype))
    return dict(int8=numpy.int16,
                uint8=numpy.uint16,
                int16=numpy.int32,
                uint16=numpy.uint32,
                int32=numpy.int64,
                uint32=numpy.uint64,
                float32=numpy.float64,
                float64=getattr(numpy,'float128', dtype),
                ).get(typename(dtype), dtype)

def signedtype(dtype):
    if isinstance(dtype, numpy.ndarray):
        return dtype.astype(signedtype(dtype.dtype))
    return dict(uint8=numpy.uint8,
                uint16=numpy.uint16,
                uint32=numpy.uint32,
                uint64=numpy.uint64,
                ).get(typename(dtype), dtype)

def highersignedtype (dtype):
    if isinstance(dtype, numpy.ndarray):
        return dtype.astype(highersignedtype(dtype.dtype))
    return highertype(signedtype(dtype))

def fix_indices(i0, i1, mn, mx):
    while not (i0>=mn and i1<mx):
        i0 += 1
        i1 -= 1
    return i0, i1

def expand_indices(i0, i1, shape, max_i):
    while i1-i0 < shape:
        if i0>0:
            i0 -= 1
            if i1-i0==shape:
                break
        if i1 < max_i:
            i1 += 1
        elif i1-i0==shape:
            break
        elif i0>0:
            pass
        else:
            raise ValueError(`i0,i1,shape, max_i`) # unexpected condition
    return i0, i1

def odd_max(x,y):
    if x <= y:
        return y
    if x % 2:
        return x
    return x + 1

def maximum_centering (images,  (i0,i1,j0,j1,k0,k1), kernel, (si,sj,sk), (ri, rj, rk),
                       bar_count=None, quiet=False):

    ni,nj,nk=kernel.shape
    image = images[i0:i1, j0:j1, k0:k1]
    image_smooth = scipy.signal.convolve(image, kernel, 'valid')
    zi, zj, zk = image_smooth.shape
    mxi, mxj, mxk = scipy.ndimage.maximum_position(image_smooth)

    i0m,i1m = i0 - (zi//2 - mxi), i1 - (zi//2 - mxi)
    j0m,j1m = j0 - (zj//2 - mxj), j1 - (zj//2 - mxj)
    k0m,k1m = k0 - (zk//2 - mxk), k1 - (zk//2 - mxk)

    i0m = min(max(i0m, ri[0]), ri[1])
    i1m = min(max(i1m, ri[0]), ri[1])
    j0m = min(max(j0m, rj[0]), rj[1])
    j1m = min(max(j1m, rj[0]), rj[1])
    k0m = min(max(k0m, rk[0]), rk[1])
    k1m = min(max(k1m, rk[0]), rk[1])

    if not (si == i1m-i0m and sj == j1m-j0m and sk == k1m-k0m):
        msg = 'Centered image stack is too small: %s < %s' % ((i1m-i0m, j1m-j0m, k1m-k0m), (si, sj, sk))
        return (i0m,i1m,j0m,j1m,k0m,k1m), msg

    mx_dist = 3
    ci, cj, ck = scipy.ndimage.center_of_mass((image_smooth-image_smooth.min())**2)
    dist = numpy.sqrt((mxi-ci)**2 + (mxj-cj)**2 + (mxk-ck)**2)

    if dist>mx_dist:
        msg = 'Maximum position and center of mass are too far: %.2fvx>%.2fvx' % (dist, mx_dist)
        return (i0m,i1m,j0m,j1m,k0m,k1m), msg

    return (i0m,i1m,j0m,j1m,k0m,k1m), None

#@utils.time_it
def spots_to_psf(image_stack, psf_dir, options = None):
    """Extract PSF spots from microscope image stack.

    Parameters
    ----------
    image_stack : {str, `iocbio.io.image_stack.ImageStack`}
      Path to or image stack of PSF measurments.
    psf_dir : str
      Directory name where to save intermediate results.
    options : {None, `iocbio.utils.Options`}

      The following options attributes are used:

      `options.cluster_background_level` : float
        Specify maximum background level for finding clusters

      `options.subtract_background_field` : bool
        If True then subtract background field from PSF measurments.
        Use this option when PSF is measured with analog PMT.

      `options.subtract_z_gradient` : bool
        If True then subtract z gradient level from PSF measurments.
        Use this option when PSF is measured with widefield microscope
        where strong z gradients exist when measuring homogeneous
        solution.

      `options.save_intermediate_results` : bool
        If True then save intermediate results.

      `options.select_candidates` : comma separeted list of numbers
        Can be used to manually select PSF canditates. See the
        save intermediate results for specifying this list.

      `options.psf_field_size` : int
        Specify the size of one PSF in arbitrary units. Start with
        the size 7.

      `options.show_plots` : bool
        If True then display intermediate results to screen.

    Returns
    -------
    psf_stack : `iocbio.io.image_stack.ImageStack`
      Estimated PSF image stack.

    See also
    --------
    iocbio.microscope.cluster_tools.find_clusters, iocbio.microscope.psf
    """
    if options.show_plots:
        import pylab
        pylab.ion()
    psf_path = os.path.join(psf_dir, 'psf.tif')
    psf_sd_path = os.path.join(psf_dir, 'psf_sd.tif')
    psf_path_uint8 = os.path.join(psf_dir, 'psf_uint8.tif')
    psf_path_data = os.path.join(psf_dir, 'PSF.data')
    psf_path_smooth = os.path.join(psf_dir, 'psf_smooth.tif')
    psf_path_canditate = os.path.join(psf_dir, 'psf_canditate_%.3i.tif')
    psf_path_blurred = os.path.join(psf_dir, 'psf_blurred.tif')
    psf_path_sphere = os.path.join(psf_dir, 'psf_sphere.tif')
    psf_path_deconv_smooth = os.path.join(psf_dir, 'psf_deconv_smooth_%.1i.tif')
    background_path = os.path.join(psf_dir, 'background.tif')

    if isinstance(image_stack, (str, unicode)):
        image_stack = ImageStack.load(image_stack, options=options)

    print 'Extracting PSF from images..'
    images = image_stack.images
    voxel_sizes = image_stack.get_voxel_sizes()
    nof_stacks = image_stack.get_nof_stacks()
    rotation_angle = image_stack.get_rotation_angle()
    image_size = list(images.shape)
    image_size[0] //= nof_stacks
    microscope_type = image_stack.get_microscope_type()
    is_widefield = microscope_type=='widefield'
    print '  Microscope type: %s' % (microscope_type)
    print '  Voxel: %s um^3' % (' x '.join(['%.3f' % (v*1e6) for v in voxel_sizes]))
    print '  Image stack: %s (%s um^3)' % (\
        ' x '.join(map(str, image_size)),
        ' x '.join(['%.3f' % (v*s*1e6) for v,s in zip(voxel_sizes, image_size)]))
    print '  Number of stacks:', nof_stacks
    print '  Rotation angle: %s deg' % (rotation_angle)
    print '  Objective NA:',image_stack.get_objective_NA()
    print '  Excitation wavelength:',image_stack.get_excitation_wavelength()*1e9, 'nm'
    dr = image_stack.get_lateral_resolution()
    dz = image_stack.get_axial_resolution()
    print '  Lateral resolution: %.3f um (%.1f x %.1f px^2)' % (1e6*dr, dr/voxel_sizes[1], dr/voxel_sizes[2])
    print '  Axial resolution: %.3f um (%.1fpx)' % (1e6*dz, dz / voxel_sizes[0])
    r = 1
    nz,ny,nx = map(lambda i: max(1,int(i)), [(dz/voxel_sizes[0])/r, (dr/voxel_sizes[1])/r, (dr/voxel_sizes[2])/r])
    print '  Blurring steps:', ' x '.join(map(str, (nz,ny,nx)))
    mz,my,mx = [m/n for m,n in zip (image_size,[nz,ny,nx])]

    print '  Blurred image stack size:', ' x '.join(map(str, (mz,my,mx)))

    edge_indices1 = [n*image_size[0] for n in range(nof_stacks)]
    edge_indices2 = [(n+1)*image_size[0]-1 for n in range(nof_stacks)]
    background1 = images[(edge_indices1,)]
    background2 = images[(edge_indices2,)]

    if options.subtract_background_field:
        from ..ops.regression import regress
        scales = (0.5*voxel_sizes[1]/dr, 0.5*voxel_sizes[2]/dr)
        print '  Computing background field..'
        sys.stdout.flush()
        bg = background1.mean(0, dtype=float) if background1.shape[0]>1 else background1[0].astype(float)
        bg1 = regress(bg, scales)
        bg = background2.mean(0, dtype=float) if background2.shape[0]>1 else background2[0].astype(float)
        bg2 = regress(bg, scales)
        bg21 = bg2 - bg1

        print '    Background field means:', tostr(bg1.mean()), tostr(bg2.mean())
        print '    Background field average slope:', tostr(abs(bg21).mean())
        print '    Saving background field to %r' % (background_path)
        ImageStack(numpy.array ([bg1,bg2]), image_stack.pathinfo).save(background_path)
        #ImageStack(numpy.array([bg21]), image_stack.pathinfo).save('background_diff.tif')

        print '  Subtracting background field from images:'
        bar = utils.ProgressBar(0,images.shape[0], prefix='  ', show_percentage=False)
        n = 0
        for i in range(image_size[0]):
            bg = (bg21 * (float(i)/image_size[0]) + bg1)
            for k in range(nof_stacks):
                j = i+k*image_size[0]
                lowvalue_indices = numpy.where(images[j] < bg)
                bar.updateComment(' lowvalues: %f%%' % (100.0*len(lowvalue_indices[0])/images[j].size))
                images[j] -= bg
                images[j][lowvalue_indices] = 0
                bar(n)
                n += 1
        #ImageStack(images, image_stack.pathinfo).save('images_wo_background.tif')

        background1 = images[(edge_indices1,)]
        background2 = images[(edge_indices2,)]

    background_mean1 =  background1.mean()
    background_mean2 = background2.mean()
    background_mean = 0.5*(background_mean1 + background_mean2)
    background_slope = (background_mean2 - background_mean1) / (image_size[0]-1)
    background_var1 = background1.var()
    background_var2 = background2.var()
    background_var = 0.5*(background_var1 + background_var2)
    offset1 = background_mean1 - background_var1
    offset2 = background_mean2 - background_var2
    offset = 0.5*(offset1+offset2)
    if offset > 0:
        # to achive background with Poisson noise
        offset = offset + 1

    print '  Background mean/slope/var: %s/%s/%s' % (tostr(background_mean), tostr(background_slope), tostr(background_var))
    print '  Estimated offset: %s (+-%s)' % ((offset), tostr(0.5*abs((offset1-offset2))))

    #if options.photon_counter_offset is None:
    #    use_offset = int(offset)
    #else:
    #    use_offset = int(options.photon_counter_offset)

    if 1:
        pass

    elif options.subtract_z_gradient:
        print '  Substracting background gradient (slope=%s) from images' % (background_slope)
        bar = utils.ProgressBar(0,images.shape[0], prefix='  ', show_percentage=False)
        for k in range(nof_stacks):
            for i in range(image_size[0]):
                o = int((background_mean2-background_mean1) * i / (image_size[0]-1.0))
                if o:
                    j = i+k*image_size[0]
                    lowvalue_indices = numpy.where(images[j] < o)
                    bar.updateComment(' offset=%s, #lowvalues: %f%%' % (o, 100.0*len(lowvalue_indices[0])/images[j].size))
                    bar(j)
                    images[j] -= o
                    images[j][lowvalue_indices] = 0
        print

    #if use_offset and use_offset > 0:
    #    print '  Subtracting photon counter offset (%s) from images' % (use_offset)
    #    lowvalue_indices = numpy.where(images < use_offset)
    #    images -= use_offset
    #    images[lowvalue_indices] = 0

    blurred_images = numpy.zeros((mz,my,mx), float)
    for i in range (nz):
        for j in range (ny):
            for k in range(nx):
                assert nz*mz <= image_size[0],`nz,mz`
                a = images[i:nz*mz:nz, j:ny*my:ny, k:nx*mx:nx]
                assert a.shape==(mz,my,mx)
                blurred_images += a
    blurred_images /= nz*ny*nx

    if options.save_intermediate_results:
        ImageStack(blurred_images, image_stack,
                   shape = blurred_images.shape,
                   nof_stacks = 1,
                   voxel_sizes = [nn*vs for nn, vs in zip ((nz,ny,nx), voxel_sizes)]
                   ).save(psf_path_blurred)

    center_of_masses = []
    minimum_distance_between_centers = 1e9
    marked_images = blurred_images.copy()

    if options and options.cluster_background_level is not None:
        cluster_backround_level = float(options.cluster_background_level)
    else:
        cluster_backround_level = None

    clusters_list = find_clusters(blurred_images, background_level=cluster_backround_level,
                                  voxel_sizes = voxel_sizes)

    select_list = []
    if options.select_candidates:
        select_list = map (int, options.select_candidates.split (','))

    print '  Finding PSF shape and canditate slices:'
    psf_slice_list = []
    psf_shape = (0,0,0)
    field_size = int(options.psf_field_size or 7)

    for counter, (coordinates, values) in enumerate(clusters_list):
        mass = values.sum()
        center_of_coordinates = (coordinates.T * values).T.sum (axis=0)/float(mass)
        ci, cj, ck = map(int, map(round, center_of_coordinates * (nz, ny, nx)))
        i0, j0, k0 = ci - (nz*field_size)//2, cj - (ny*field_size)//2, ck - (nx*field_size)//2
        i1, j1, k1 = ci + (nz*field_size)//2, cj + (ny*field_size)//2, ck + (nx*field_size)//2
        i0, i1 = fix_indices(i0, i1, nz, image_size[0] - nz)
        j0, j1 = fix_indices(j0, j1, ny, image_size[1] - ny)
        k0, k1 = fix_indices(k0, k1, nx, image_size[2] - nx)
        psf_slice_list.append((i0,i1,j0,j1,k0,k1))
        # psf dimensions will be odd:
        psf_shape = odd_max(i1-i0, psf_shape[0]), odd_max(j1-j0, psf_shape[1]), odd_max(k1-k0, psf_shape[2])

    print '  PSF shape:', psf_shape

    print '  Centering PSF canditates:'
    psf_sum = numpy.zeros(psf_shape, highertype(images.dtype))
    max_bar_count = nof_stacks*len(psf_slice_list)
    bar = utils.ProgressBar(0,max_bar_count-1, prefix='  ')
    psf_count = -1
    nof_measurments = 0
    mx_dist = 3

    kernel = numpy.ones((nz,ny,nx))

    for counter, (i0,i1,j0,j1,k0,k1) in enumerate(psf_slice_list):
        i0, i1 = expand_indices(i0, i1, psf_shape[0], image_size[0]-1)
        j0, j1 = expand_indices(j0, j1, psf_shape[1], image_size[1]-1)
        k0, k1 = expand_indices(k0, k1, psf_shape[2], image_size[2]-1)
        center = numpy.array(psf_shape[1:])*0.5

        for n in range(nof_stacks):
            psf_count = counter * nof_stacks + n
            nn = n * image_size[0]

            (i0m, i1m, j0m, j1m, k0m, k1m), msg = maximum_centering(images, 
                                                                    (nn+i0,nn+i1,j0,j1,k0,k1), 
                                                                    kernel, 
                                                                    psf_shape,
                                                                    ((nn, nn+image_size[0]), (0, image_size[1]), (0, image_size[2])),
                                                                    )

            index_center = (i0m+i1m)//2, (j0m+j1m)//2, (k0m+k1m)//2

            psf_canditate = images[i0m:i1m, j0m:j1m, k0m:k1m]

            if 0:
                p = psf_canditate.sum(0, dtype=float)
                #r = 2*0.95*numpy.sqrt(p.max())
                #p[numpy.where(p<r)] = 0
                d1, d2 = numpy.indices(p.shape, dtype=float)
                d1 -= index_center[1]
                d2 -= index_center[2]
                v1 = (p * d1**2).sum(dtype=float)
                v2 = (p * d2**2).sum(dtype=float)
                v12 = (p * d1*d2).sum(dtype=float)
                alpha = numpy.arctan2(v12, v1) if v1 > abs(v12) else numpy.arctan2(v2, v12)
                cosa, sina = numpy.cos(alpha), numpy.sin(alpha)
                a = ((cosa * d1 + sina * d2)).var()
                b = ((-sina * d1 + cosa * d2)).var()
                print
                print alpha*180/numpy.pi, a,b

            psf = normalize_uint8(psf_canditate)

            psf[psf.shape[0]//2, :, psf.shape[2]//2] = 255
            psf[psf.shape[0]//2, psf.shape[1]//2, :] = 255
            psf[:, psf.shape[1]//2, psf.shape[2]//2] = 255

            if options.save_intermediate_results:
                ImageStack(psf, image_stack,
                           shape = psf.shape,
                           nof_stacks = 1,
                           background = (background_mean, background_slope),
                           ).save(psf_path_canditate % (psf_count))

            bar.updateComment(' %.3i.%.3i: center=%s' % (counter, n, index_center))
            bar(psf_count)

            if options.show_plots:
                pylab.imshow(psf[psf.shape[0]//2])
                pylab.title(msg or 'ok')
                pylab.draw()

            if msg is not None:
                print '\n  %s' % (msg)
                break

            if select_list and counter not in select_list:
                pass
            else:
                psf_sum += psf_canditate
                nof_measurments += 1

            if options.show_plots:
                pylab.imshow (psf_sum[psf_sum.shape[0]//2])
                pylab.title('Sum of %s PSF candiates' % (nof_measurments))
                pylab.draw ()

    bar(max_bar_count)
    print

    print '  Nof PSF measurements: %s' % (nof_measurments)

    if not nof_measurments:
        raise ValueError('No valid PSF canditates detected')    

    # For future, that is, the saved data can be reused for statistics
    print '  Saving integrated PSF to %s' % (psf_path)
    psf_stack = ImageStack(discretize(psf_sum), image_stack,
                           shape = psf_sum.shape,
                           background = (background_mean * nof_measurments, background_slope * nof_measurments),
                           nof_stacks = 1
                           )


    psf_resolution = 2*0.95/numpy.sqrt(nof_measurments)*numpy.sqrt(psf_sum.max())
    print '  PSF value resolution: %.2f' % (psf_resolution)
    psf_stack.pathinfo.set_value_resolution(psf_resolution)
    psf_stack.save(psf_path)

    return psf_stack

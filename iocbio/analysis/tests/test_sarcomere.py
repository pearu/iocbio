
from __future__ import division

import sys
import numpy

def shape_func(i,w,r):
    """ Evaluate shape function.
::
              ^
        ,_____|_____.          1
       /             \
    --'       +       '--   -> 0
              0              i
        <-w*(1-2r)->
    <-------- w -------->
    """
    i = abs(i)
    if i <= w*(1-2*r)/2:
        return 1
    if i <= w/2:
        return (1 + numpy.cos((i-(w/2-r*w))/(r*w)*numpy.pi))/2
    return 0

def compute_image(sarcomere_length_um,
                  sarcomere_width_um,
                  sarcomere_distance_um, # distance between neighboring sacromeres
                  roi_width_px, roi_length_px,
                  pixel_size_um):
    
    image = numpy.zeros ((roi_length_px, roi_width_px))
    sarcomere_length_px = sarcomere_length_um / pixel_size_um
    sarcomere_width_px = sarcomere_width_um / pixel_size_um
    sarcomere_distance_px = sarcomere_distance_um / pixel_size_um

    nof_sarcomeres_length = int(roi_length_px / sarcomere_length_px)
    nof_sarcomeres_width = int(roi_width_px / sarcomere_distance_px)
    
    print nof_sarcomeres_length, nof_sarcomeres_width, sarcomere_distance_px, sarcomere_length_px

    for j in range(nof_sarcomeres_width + 1):
        cj = j * sarcomere_distance_px
        for i in range(nof_sarcomeres_length + 1):
            ci = i * sarcomere_length_px
            #ci += numpy.random.normal (scale=sarcomere_length_px/40)
            lcj = cj #+ numpy.random.normal (scale=sarcomere_distance_px/20)

            for j0 in range(-sarcomere_width_px/2, sarcomere_width_px/2+1):
                if lcj+j0 >= image.shape[1] or lcj+j0<0:
                    continue
                sj = shape_func(j0, sarcomere_width_px, 0.4)

                for i0 in range(-sarcomere_length_px/2, sarcomere_length_px/2+1):
                    if ci+i0 >= image.shape[0] or ci+i0<0:
                        continue
                    si = shape_func(i0, sarcomere_length_px, 0.5)
                    image[ci+i0, lcj+j0] += si * sj * 100
    return image

def main ():
    from libtiff import TIFFimage
    #import matplotlib.pyplot as plt

    N = 400
    for i in range (N):
        L = 1.8 + 0.2*numpy.cos(max(0,i-N//5)/N*2*numpy.pi)
        print i,L
        image = compute_image(L, 2.25, 1.5, 100, 640, 0.274)
        tif = TIFFimage(image.astype(numpy.uint8).T, description='''
VoxelSizeX: 0.274e-6
VoxelSizeY: 0.274e-6
VoxelSizeZ: 1
MicroscopeType: widefield
ObjectiveNA: 1.2
ExcitationWavelength: 540.0
RefractiveIndex: 1.33
''')
        tif.write_file ('fakesacromere_exact/image_%06d.tif' % i)
        del tif

    #plt.imshow (image.T*256, interpolation='nearest', cmap='gray')
    #plt.show ()

if __name__ == '__main__':
    main ()

""" Provides ImageStack class.

Representing image stacks as ``PATH``-s
=======================================

Different microscope software can save 3D images in various formats.
These formats hold image data as well as topological and environment
information about microscope session. For example, a Zeiss microscope
system saves 3D images to a single ``.lsm`` file that contains also
microscope session data. On the other hand, SysBio microscope systems
save 3D images as a series of 2D TIFF images and the topological and
environment data is stored in a :file:`configuration.txt` file. Other
microscope systems may use other conventions.

To process stacks of images that are produced by these microscope
systems, by iocbio software, we have implemented an interface to various
storage formats to ease reading microscope data in by iocbio software
tools. Typically, a software tool reads 3D images from a disk,
processes these 3D images, and then saves them back to disk.  In the
following, 3D images stored in a disk are referred as
``PATH``. Usually the ``PATH`` is the name of a file or a directory
containing microscope images.

For reading, the following storage formats are supported:

  - a LSM file ``PATH`` (with extension ``.lsm``, created by the Zeiss
    microscope system),

  - a TIFF file ``PATH`` (with extension ``.tif`` or ``.tiff``) and a
    topological data file :file:`PATH + PATHINFO.txt` (created by iocbio
    software),

  - a directory ``PATH`` of raw files (with extension ``.raw``) and a
    topological data file :file:`PATH/SCANINFO.txt` (created by ImageJ
    software) or :file:`PATH/PATHINFO.txt`,

  - a directory ``PATH`` of TIFF files and a topological data file
    :file:`PATH/configuration.txt` (created by SysBio microscope system),

where topological data file contains information about voxel sizes and
rotation angle as well as the type of a microscope system (widefield
or confocal), emission and excitation wave lengths, the numerical
aperture of objective, the refrective index of medium, etc.

For writing, the following storage formats are supported:

  - a TIFF file ``PATH`` (with extension ``.tif``) and a topological
    data file :file:`PATH + PATHINFO.txt` (recommended format),
        
  - a TEXT file ``PATH`` (with extension ``.data``) and a topological
    data file :file:`PATH + PATHINFO.txt` (used by SysBio software),

  - a RAW file ``PATH`` (with extension ``.raw``) and a topological
    data file :file:`PATH + PATHINFO.txt`,

  - a directory ``PATH`` of RAW files and a topological data file
    :file:`PATH/PATHINFO.txt`.

Python interface
----------------

To load a 3D image to a Python program, use::

  from iocbio.io import ImageStack
  stack = ImageStack.load(PATH)

where ``PATH`` is a path to a file or directory containing a stack of
images.

To create a 3D image stack in a Python program and save it as .tif file,
use::

  from iocbio.io import ImageStack
  images = numpy.zeros((2,3,4))
  stack = ImageStack(images, voxel_sizes=...)
  stack.save('stack.tif')

Module content
--------------
"""
# Author: Pearu Peterson
# Created: 2009

__all__ = ['ImageStack']

import numpy
from . import io
from .pathinfo import PathInfo


class ImageStack(object):
    """
    Holds a 3D stack of images and its parameters.

    Attributes
    ----------

    images : :numpy:`ndarray`
      a 3D array of image stacks.
    pathinfo : `iocbio.io.pathinfo.PathInfo`
      holds microscope data

    See also
    --------
    :mod:`iocbio.io.image_stack`
    """
    
    @classmethod
    def load(cls, path, options=None):
        """
        Load microscope images from path to ImageStack object.

        Parameters
        ----------
        path : str
          File or directory name.
        options : {None, `iocbio.utils.Options`}
          Options specified in command line. Note that command line
          options override options stored in pathinfo attribute.

        Returns
        -------
          image_stack : `iocbio.io.image_stack.ImageStack`

        See also
        --------
        :class:`iocbio.io.image_stack.ImageStack`
        """
        images, pathinfo = io.load_image_stack(path, options=options)
        return cls (images, pathinfo=pathinfo, options=options)

    def __init__(self, images, pathinfo = None, options=None, **kws):
        """
        Construct `ImageStack` from an array.

        Parameters
        ----------
        images : :numpy:`ndarray`
          A 3D or 2D array.
        pathinfo : {None, `iocbio.io.pathinfo.PathInfo`}
          If pathinfo is None then it will be constructed from the
          kws mapping.
        options : {None, `iocbio.utils.Options`}
        kws : dict
          A dictionary of pathinfo keys.
        """
        if len (images.shape)==2:
            images = numpy.array([images])

        self.images = images
        if pathinfo is None:
            self.pathinfo = pathinfo = PathInfo ('.')
        elif isinstance(pathinfo, ImageStack):
            self.pathinfo = pathinfo.pathinfo.copy()
        else:
            self.pathinfo = pathinfo.copy()
        for key, value in kws.items():
            self.pathinfo.set(key, value)
        if 'sample_format' not in kws:
            self.pathinfo.set_sample_format(images.dtype)
        if 'shape' not in kws:
            self.pathinfo.set_shape(images.shape)
        assert images.shape == self.pathinfo.get_shape (), `images.shape, self.pathinfo.get_shape ()`
        self.options = options
        self.pathinfo.set_options(options)

    def save(self, path, indices=None):
        """
        Save image stack to path.

        Parameters
        ----------
        path : str
          File or directory name of a ``PATH``.
        indices : {None, tuple}
          Save only image data with indices. Applicable if path has
          ``.data`` extension.
        """
        io.save_image_stack(self, path, indices=indices, options=self.options)

    def get_voxel_sizes(self):
        """
        Return 3-tuple of voxels sizes in meters.
        """
        return self.pathinfo.get_voxel_sizes()

    def get_objective_NA(self):
        """
        Return numerical aperture of microscope objective.
        """
        NA = None
        if self.options is not None:
            NA = self.options.objective_na
        if NA is None:
            NA = self.pathinfo.get_objective_NA()
        if NA is None:
            print 'Warning: failed to determine objective NA. Use --objective-na=..'
        return NA

    def get_excitation_wavelength(self):
        """
        Return excitation wavelength in meters.
        """
        if self.options is not None:
            wl = self.options.excitation_wavelength
        else:
            wl = None
        if wl is None:
            wl = self.pathinfo.get_excitation_wavelength()
        if wl is None:
            print 'Warning: failed to determine excitation wavelength. Use --excitation-wavelength=..'
        return wl

    def get_emission_wavelength(self):
        """
        Return emission wavelength in meters.
        """
        if self.options is not None:
            wl = self.options.emission_wavelength
        else:
            wl = None
        if wl is None:
            wl = self.pathinfo.get_emission_wavelength()
        if wl is None:
            print 'Warning: failed to determine emission wavelength. Use --emission-wavelength=..'
        return wl

    def get_refractive_index(self):
        """
        Return refractive index of microscope medium.
        """
        if self.options is not None:
            v = self.options.refractive_index
        else:
            v = None
        if v is None:
            v = self.pathinfo.get_refractive_index()
        if v is None:
            print 'Warning: failed to determine refractive index of medium. Use --refractive-index=..'
        return v

    def get_lateral_resolution(self):
        """
        Return computed lateral resolution (in meters) of the
        microscope system.
        """
        NA = self.get_objective_NA()
        n = self.get_refractive_index()
        l = self.get_excitation_wavelength()
        t = self.get_microscope_type()
        if None in [NA, n, l, t]: return
        alpha = numpy.arcsin(NA/n)
        dr = l / n / numpy.sqrt (3-2*numpy.cos (alpha)-numpy.cos(2*alpha))
        if t=='confocal':
            return dr
        if t=='widefield':
            return 2*dr
        raise NotImplementedError(`t`)

    def get_axial_resolution(self):
        """
        Return computed axial resolution (in meters) of the microscope
        system.
        """
        NA = self.get_objective_NA()
        n = self.get_refractive_index()
        l = self.get_excitation_wavelength()
        t = self.get_microscope_type()        
        if None in [NA, n, l, t]: return
        alpha = numpy.arcsin(NA/n)
        dz = l / n / (1-numpy.cos(alpha))
        if t=='confocal':
            return dz
        if t=='widefield':
            return 2*dz
        raise NotImplementedError(`t`)
    
    def get_nof_stacks(self):
        """
        Return the number of image stacks contained in the
        :attr:`images` array.
        """
        nof_stacks = self.pathinfo.get_nof_stacks()
        if nof_stacks is None:
            return 1
        return nof_stacks

    def get_rotation_angle(self):
        """
        Return scanning angle in degrees.
        """
        rotation_angle = self.pathinfo.get_rotation_angle()
        if rotation_angle is None:
            return 0
        return rotation_angle

    def get_microscope_type(self):
        """
        Return microscope type: ``'confocal'`` or ``'widefield'``.
        """
        if self.options is not None:
            v = self.options.microscope_type
        else:
            v = None
        if v is None or v=='<detect>':
            v = self.pathinfo.get_microscope_type()
        if v is None:
            print 'Warning: failed to determine microscope type. Use --microscope-type=confocal|widefield'            
        return v

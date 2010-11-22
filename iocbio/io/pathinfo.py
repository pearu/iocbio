""" Provides PathInfo classes.

PathInfo instances hold microscope topological and environmental
information that is related to ``PATH``. Pathinfo instances are
usually created by the `iocbio.io.io.load_image_stack` function that
returns instances of one of the following classes:

.. autosummary::

  Scaninfo
  Configuration
  Tiffinfo
  Rawinfo

The following mappings are used to hold microscope information:

.. autodata:: objectives
.. autodata:: filters

Module content
--------------
"""

__autodoc__ = ['PathInfo']
__all__ = ['Scaninfo', 'Configuration', 'Tiffinfo', 'Rawinfo']

import os
import sys
import time
import numpy
from StringIO import StringIO
from . import tifffile
import tempfile


objectives = {'UPLSAPO_60xW_NA_1__x20':dict(refractive_index=1.33, NA = 1.2), # airy
              'UPlanFLN_10x_NA_0__x30':dict(refractive_index=1.0, NA=0.3),    # airy
              'CFI_Plan_Apochromat_VC_60xW_NA_1__x20':dict(refractive_index=1.33, NA = 1.2), # suga
              'CFI_Super_Plan_Fluor_ELWD_40xC_NA_0__x60':dict(refractive_index=1.0, NA = 0.6), # suga
              'CFI_Super_Plan_Fluor_ELWD_20xC_NA_0__x45':dict(refractive_index=1.0, NA=0.45),# suga
              'C-Apochromat 63x/1.20 W Korr UV-VIS-IR M27':dict(refractive_index=1.33, NA = 1.2), # zeiss
              'Plan-Apochromat 63x/1.40 Oil DIC M27':dict(refractive_index=1.5158, NA = 1.4), # zeiss
              }
"""
Contains a mapping between objectives and their parameters.

To find out currently recognized objectives, run:
  >>> from iocbio.io import pathinfo
  >>> print '\\n'.join(pathinfo.objectives.keys())
  CFI_Plan_Apochromat_VC_60xW_NA_1__x20
  Plan-Apochromat 63x/1.40 Oil DIC M27
  CFI_Super_Plan_Fluor_ELWD_20xC_NA_0__x45
  UPLSAPO_60xW_NA_1__x20
  C-Apochromat 63x/1.20 W Korr UV-VIS-IR M27
  UPlanFLN_10x_NA_0__x30

Parameters
----------
refractive_index : float
  Refractive index of a objective
NA : float
  Numerical aperture of a objective

Notes
-----
To add more objectives to the mapping, add the following
code to your Python program::

  from iocbio.io import pathinfo
  pathinfo.objectives['<objective name>'] = dict(refractive_index=.., NA=..)

"""

filters = {
           '__e1__c_550__s88':dict(wavelength=550),
           '__e2__c_725__s150':dict(wavelength=725),
           '__e585__s40':dict (),
           '__e1__c_465__s30_Di510':dict(turret='top'),
           '__e2__c_340__s26_Di400lp':dict(turret='top'),
           '__e3__c_562__s40_Di593_624__s40':dict(turret='top'),
           '__e4__c_482__s35_Di506':dict(turret='top'),
           '__e5__c_500__s24_Di520':dict(turret='top'),
           '__e6__c_543__s22_Di562':dict(turret='top'),
           '__e1__c_525__s50_Di560xr':dict(turret='bottom'),
           '__e2__c_460__s80_Di510xr':dict(turret='bottom'),
           '__e3__c_Mirror':dict(turret='bottom'),
           '__e4__c_536__s40_Di580xr':dict(turret='bottom'),
           '__e5__c_542__s27_Di580xr':dict(turret='bottom'),
           '__e6__c_593__s40_Di640xr':dict(turret='bottom'),
           }
"""
Contains a mapping between optical filters and their parameters.
This mapping is specific to SysBio microscope systems but can
be extended to others when necessary.

Parameters
----------
wavelength : float
  Wavelength in nanometers.
turret : {'top', 'bottom'}
  The location of a filter.
"""

dyes = {}

def get_tag_from_scaninfo(path, tagname, _cache={}):
    """ Return tag value from SCANINFO.txt formatted file.
    """
    info = _cache.get(path)
    if info is not None:
        return info.get(tagname)
    info = {}
    f = open (path,'r')
    for line in f.readlines():
        i = line.find(':')
        if i==-1: continue
        info[line[:i].strip()] = line[i+1:].strip()
    f.close ()
    _cache[path] = info
    return info.get(tagname)

def get_tag_from_configuration(path, tagname, _cache={}):
    """ Return tag value from configuration.txt formatted file.
    """
    info = _cache.get(path)
    if info is not None:
        if tagname is None:
            return info
        return info.get(tagname)
    if not os.path.isfile(path):
        return
    info = {}
    f = open(path,'r')
    is_string = False
    text = ''
    value_type = str
    for line in f.readlines():
        if is_string:
            if line.rstrip().endswith('"'):
                text += line.rstrip()[:-1]
                is_string = False
                info[tag] = text
                text = ''
            else:
                text += line
        else:
            i = line.find(' ')
            if i==-1: continue
            tag = line[:i].strip()
            text = line[i+1:].strip()
            if text.startswith('"'):
                is_string = True
                text = text[1:]
            else:
                if tag=='Float':
                    value_type = float
                elif tag=='Int':
                    value_type = int
                elif tag=='Bool':
                    value_type = bool
                elif tag=='String':
                    value_type = str
                else:
                    info[tag] = value_type(text)
    f.close ()
    _cache[path] = info
    if tagname is None:
        return info
    return info.get(tagname)

def get_tag_from_lsm_file(path, tagname, _cache={}):
    """ Return tag value from .lsm file by brute force.
    """
    info = _cache.get(path)
    if info is not None:
        return info.get(tagname)
    info = {}
    f = open (path,'r')
    flag = True
    l = []
    for line in f.readlines():
        sline = line.strip()
        if sline.startswith('BEGIN'):
            l.append(sline.split()[1].strip())
        elif l:
            if sline=='END':
                l.pop()
                continue
            i = sline.find('=')
            if i==-1: continue
            key = '.'.join(l + [sline[:i].rstrip()])
            info[key] = sline[i+1:].lstrip()
            #print key, info[key]
    f.close ()
    _cache[path] = info
    return info.get(tagname)

class PathInfo(object):
    """ Base class for storing microscope information.
    """

    def __init__(self, path):
        self.path = path
        self.microscope_type = None
        self.nof_stacks = None
        self.shape = None
        self.voxel_sizes = None
        self.image_time = None
        self.objective_NA = None
        self.excitation_wavelength = None
        self.emission_wavelength = None
        self.refractive_index = None
        self.rotation_angle = None
        self.value_resolution = None
        self.RL_lambda = None
        self.background = None
        self.sample_format = None
        self.protocol = None
        self.options = None

    def __str__ (self):
        l = []
        for attr in ['microscope_type', 'nof_stacks', 'shape',
                     'voxel_sizes', 'objective_NA', 'excitation_wavelength',
                     'emission_wavelength', 'refractive_index',
                     'rotation_angle', 'value_resolution', 'background', 'sample_format',
                     'protocol', 'options']:
            try:
                v = getattr (self, 'get_'+attr) ()
            except NotImplementedError, msg:
                v = None
            except AttributeError, msg:
                print '%s: no %s attribute' % (self.__class__.__name__, msg)
                v = None
            if v is None:
                continue
            l.append ('%s=%r' % (attr, v))
        return '%s(%r)[%s]' % (self.__class__.__name__, self.path, ', '.join (l))

    def get_shape (self):
        """
        Return a shape of image stack.
        """
        return self.shape

    def get_nof_stacks(self):
        """
        Return the number of image stacks.
        """
        return self.nof_stacks

    def get_voxel_sizes(self):
        """
        Return a tuple of voxel sizes in meters.
        """
        return self.voxel_sizes

    def get_image_time(self):
        """
        Return a list of time moments when image in a stack was taken in seconds.
        """
        return self.image_time

    def get_objective_NA (self):
        """
        Return numerical aperture of microscope objective.
        """
        return self.objective_NA

    def get_excitation_wavelength(self):
        """
        Return excitation wavelength in meters.
        """
        return self.excitation_wavelength

    def get_emission_wavelength(self):
        """
        Return emission wavelength in meters.
        """
        return self.emission_wavelength

    def get_refractive_index (self):
        """
        Return refrective index of environment.
        """
        return self.refractive_index

    def get_rotation_angle(self):
        """
        Return the scanning angle.
        """
        return self.rotation_angle

    def get_value_resolution(self):
        """
        Return voxel value resolution.
        """
        return self.value_resolution

    def get_microscope_type(self):
        """
        Return microscope type: ``'confocal'`` or ``'widefield'``.
        """
        return self.microscope_type

    def get_RL_lambda(self):
        return self.RL_lambda

    def get_background(self):
        """
        Return the level of background noise.
        """
        return self.background

    def get_sample_format(self):
        """
        Return sample format of voxel values: ``'int'``, ``'uint'``, ``'float'``, ``'complex'``
        """
        return self.sample_format

    def get_protocol(self):
        """
        Return microscope protocol.
        """
        return self.protocol

    def get_options (self):
        """
        Return command line options.
        """
        return self.options

    def set(self, key, value):
        """
        Setter of microscope information.
        """
        getattr(self, 'set_'+key)(value)

    def get(self, key):
        """
        Getter of microscope information.
        """
        mth = getattr(self, 'get_'+key, None)
        if mth is not None:
            return mth()
        return getattr(self, key, None)

    def set_suffix (self, suffix):
        assert isinstance (suffix, str),`suffix`
        self.suffix = suffix

    def set_microscope_type(self, type):
        if type is None:
            return
        assert type in ['confocal', 'widefield'],`type`
        self.microscope_type = type

    def set_shape(self, *shape):
        if len(shape)==1 and not isinstance(shape[0], int):
            self.set_shape(*shape[0])
        else:
            self.shape = shape

    def set_nof_stacks(self, n):
        if isinstance(n, str):
            n = int(n)
        assert isinstance(n, int),`n, type(n)`
        self.nof_stacks = n

    def set_voxel_sizes(self, *voxel_sizes):
        if len (voxel_sizes)==1  and not isinstance(voxel_sizes[0], (float, int)):
            self.set_voxel_sizes(*voxel_sizes[0])
        else:
            self.voxel_sizes = voxel_sizes

    def set_image_time(self, lst):
        assert isinstance(lst, list),`n, type(lst)`
        self.image_time = lst

    def set_objective_NA (self, objective_NA):
        if objective_NA is None:
            assert self.objective_NA is None,`self.objective_NA`
            return
        if isinstance(objective_NA, str):
            objective_NA = float(objective_NA)
        assert isinstance (objective_NA, (float, int)),`objective_NA, type(objective_NA)`
        self.objective_NA = objective_NA

    def set_excitation_wavelength(self, wavelength):
        if wavelength is None:
            return
        if isinstance(wavelength, str):
            wavelength = float(wavelength)
        assert isinstance (wavelength, (float, int)),`wavelength, type(wavelength)`
        if wavelength > 100:
            wavelength *= 1e-9
        self.excitation_wavelength = wavelength

    def set_emission_wavelength(self, wavelength):
        if wavelength is None:
            return
        if isinstance(wavelength, str):
            wavelength = float(wavelength)
        assert isinstance (wavelength, (float, int)),`wavelength, type(wavelength)`
        if wavelength > 100:
            wavelength *= 1e-9
        self.emission_wavelength = wavelength

    def set_refractive_index(self, refractive_index):
        if refractive_index is None:
            return
        if isinstance(refractive_index, str):
            refractive_index = float(refractive_index)
        assert isinstance (refractive_index, (float, int)),`refractive_index`
        self.refractive_index = refractive_index

    def set_rotation_angle(self, rotation_angle):
        if isinstance(rotation_angle, str):
            rotation_angle = float(rotation_angle)
        assert isinstance (rotation_angle, (float, int)),`rotation_angle`
        self.rotation_angle = rotation_angle

    def set_value_resolution(self, value_resolution):
        if isinstance(value_resolution, str):
            value_resolution = float(value_resolution)
        assert isinstance (value_resolution, (float, int)),`value_resolution`
        self.value_resolution = value_resolution

    def set_RL_lambda(self, value):
        if isinstance(value, str):
            value = float(value)
        assert isinstance (value, (float, int)),`value`
        self.RL_lambda = value        
        
    def set_background(self, background):
        if isinstance(background, str):
            background = eval(background)
        assert isinstance(background, tuple),`background`
        self.background = background

    def set_protocol (self, protocol):
        assert isinstance(protocol, str), `type(protocol)`
        assert protocol in ['rics', 'image']
        self.protocol = protocol

    def set_sample_format(self, sample_format):
        if isinstance (sample_format, numpy.dtype):
            for fmt in ['float', 'complex','int','uint']:
                if sample_format in numpy.sctypes[fmt]:
                    sample_format = fmt
                    break
        assert sample_format in ['uint','int','float','complex'],`sample_format`
        self.sample_format = sample_format

    def set_options (self, options):
        self.options = options

    def save(self, path):
        """
        Save PathInfo instance to a PATHINFO.txt file.
        """
        if isinstance (path, StringIO):
            f = path
        elif isinstance(path, str):
            f = open(path, 'w')
        else:
            raise NotImplementedError (`path`)
        shape = self.get_shape()
        if shape is not None:
            d = 'XYZT'
            for i in range(len(shape)):
                f.write('Dimension%s: %s\n' % (d[i], shape[len(shape)-i-1]))
        nof_stacks = self.get_nof_stacks()
        if nof_stacks is not None:
            f.write('NofStacks: %s\n' % (nof_stacks))
        voxel_sizes = self.get_voxel_sizes()
        if voxel_sizes is not None:
            d = 'XYZT'
            for i in range(len(voxel_sizes)):
                f.write('VoxelSize%s: %s\n' % (d[i], voxel_sizes[len (voxel_sizes)-i-1]))
        rotation_angle = self.get_rotation_angle()
        if rotation_angle is not None:
            f.write ('RotationAngle: %s\n' % (rotation_angle))
        microscope_type = self.get_microscope_type()
        if microscope_type is not None:
            f.write ('MicroscopeType: %s\n' % (microscope_type))
        objective_NA = self.get_objective_NA()
        if objective_NA is not None:
            f.write('ObjectiveNA: %s\n' % (objective_NA))
        excitation_wavelength = self.get_excitation_wavelength()
        if excitation_wavelength is not None:
            f.write('ExcitationWavelength: %s\n' % (excitation_wavelength*1e9)) #nm
            excitation_wavelength = self.get_excitation_wavelength()
        emission_wavelength = self.get_emission_wavelength()
        if emission_wavelength is not None:
            f.write('EmissionWavelength: %s\n' % (emission_wavelength*1e9)) #nm
        refractive_index = self.get_refractive_index()
        if refractive_index is not None:
            f.write ('RefractiveIndex: %s\n' % (refractive_index))
        value_resolution = self.get_value_resolution()
        if value_resolution is not None:
            f.write ('ValueResolution: %s\n' % (value_resolution))
        RL_lambda = self.get_RL_lambda()
        if RL_lambda is not None:
            f.write('DeconvolveRLTVLambda: %s\n' % (RL_lambda))
        background = self.get_background()
        if background is not None:
            f.write ('Background: %s\n' % (background,))
        protocol = self.get_protocol()
        if protocol is not None:
            f.write ('Protocol: %s\n' % (protocol,))
        sample_format = self.get_sample_format()
        if sample_format is not None:
            f.write ('SampleFormat: %s\n' % (sample_format,))
        f.write('PathinfoSysArgv: %s\n' % (' '.join(map(str, sys.argv))))
        f.write('PathinfoDate: %s\n' % (time.asctime()))
        f.write('PathInfoClass: %s\n' % (self.__class__.__name__))
        f.write('PathInfoPath: %s\n' % (self.path))

        options = self.options
        if options is not None:
            l = []
            for name,value in options.__dict__.items ():
                if name.startswith ('_') or value is None: continue
                l.append('%s=%r' % (name, value))
            f.write('Options: %s\n' % (';'.join(l)))
        if isinstance (path, StringIO):
            pass
        elif isinstance(path, str):
            f.close()        
        else:
            raise NotImplementedError (`path`)

    def copy(self):
        """
        Return a copy of a PathInfo instance.
        """
        pathinfo = self.__class__(self.path)
        pathinfo.nof_stacks = self.nof_stacks
        pathinfo.shape = self.shape
        pathinfo.voxel_sizes = self.voxel_sizes
        pathinfo.objective_NA = self.objective_NA
        pathinfo.excitation_wavelength = self.excitation_wavelength
        pathinfo.emission_wavelength = self.emission_wavelength
        pathinfo.refractive_index = self.refractive_index
        pathinfo.microscope_type = self.microscope_type
        pathinfo.RL_lambda = self.RL_lambda
        pathinfo.value_resolution = self.value_resolution
        pathinfo.background = self.background
        pathinfo.sample_format = self.sample_format
        pathinfo.protocol = self.protocol
        return pathinfo

    def get_lateral_resolution(self):
        """
        Return computed lateral resolution of microscope system.
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

    def get_detectors(self):
        """Return detectors information.

        Returns
        -------
        info : list
          A list of dictonaries with `name`, `index`, `pinhole` keys.
        """
        return

    def omexml(self):
        raise NotImplementedError('%s.omexml' % (self.__class__.__name__))

class Scaninfo(PathInfo):

    """
    PathInfo subclass with microscope information stored in a SCANINFO.txt files.
    """

    def get_options(self):
        if self.options is None:
            options = get_tag_from_scaninfo(self.path, 'Options')
            if options is not None:
                options_dict = {}
                for name_value in options.split(';'):
                    name, value = name_value.split('=', 1)
                    options_dict[name.strip()] = eval(value.strip())
                self.set_options(options_dict)
        return self.options

    def get_microscope_type(self):
        if self.microscope_type is None:
            microscope_type = get_tag_from_scaninfo(self.path, 'MicroscopeType')
            if microscope_type is not None:
                self.set_microscope_type(microscope_type)
        return self.microscope_type

    def get_shape(self):
        if self.shape is None:
            shape = []
            for d in 'TZYX':
                s = get_tag_from_scaninfo(self.path, 'Dimension'+d)
                if s is None:
                    assert not shape,`d, shape` # missing dimension??
                    continue
                shape.append(int(s))
            self.set_shape(*shape)
        return self.shape

    def get_nof_stacks(self):
        if self.nof_stacks is None:
            nof_stacks = get_tag_from_scaninfo(self.path, 'NofStacks')
            if nof_stacks is not None:
                self.set_nof_stacks(nof_stacks)
        return self.nof_stacks

    def get_voxel_sizes(self):
        if self.voxel_sizes is None:
            voxel_sizes = []
            for d in 'TZYX':
                s = get_tag_from_scaninfo(self.path, 'VoxelSize'+d)
                if s is None:
                    assert not voxel_sizes,`d, voxel_sizes` # missing dimension??
                    continue
                voxel_sizes.append(float(s))
            if not voxel_sizes:
                voxel_sizes = [1] * len(self.get_shape())
            self.set_voxel_sizes(*voxel_sizes)
        return self.voxel_sizes

    def get_objective_NA (self):
        if self.objective_NA is None:
            objective_NA = get_tag_from_scaninfo(self.path, 'ObjectiveNA')
            if objective_NA is None:
                objective_name = get_tag_from_scaninfo(self.path, 'ENTRY_OBJECTIVE')
                print objective_name, self.path
                objective_params = objectives.get(objective_name)
                if objective_params is not None:
                    objective_NA = objective_params.get('NA')
                else:
                    return None
            if objective_NA is not None:
                self.set_objective_NA(objective_NA)
        return self.objective_NA

    def get_excitation_wavelength(self):
        if self.excitation_wavelength is None:
            wavelength = get_tag_from_scaninfo(self.path, 'ExcitationWavelength')
            if wavelength is None:
                wavelength = get_tag_from_scaninfo(self.path, 'WAVELENGTH')
            if wavelength is not None:
                self.set_excitation_wavelength(wavelength)
        return self.excitation_wavelength

    def get_emission_wavelength(self):
        if self.emission_wavelength is None:
            wavelength = get_tag_from_scaninfo(self.path, 'EmissionWavelength')
            #if wavelength is None:
            #    wavelength = get_tag_from_scaninfo(self.path, 'WAVELENGTH')
            if wavelength is not None:
                self.set_emission_wavelength(wavelength)
        return self.emission_wavelength

    def get_refractive_index(self):
        if self.refractive_index is None:
            refractive_index = get_tag_from_scaninfo (self.path, 'RefractiveIndex')
            if refractive_index is None:
                objective_name = get_tag_from_scaninfo(self.path, 'ENTRY_OBJECTIVE')
                objective_params = objectives.get(objective_name)
                if objective_params is not None:
                    refractive_index = objective_params.get('refractive_index')
                else:
                    return None
            if refractive_index is not None:
                self.set_refractive_index(refractive_index)
        return self.refractive_index

    def get_rotation_angle(self):
        if self.rotation_angle is None:
            rotation_angle = get_tag_from_scaninfo (self.path, 'RotationAngle')
            if rotation_angle is None:
                rotation_angle = get_tag_from_scaninfo(self.path, 'ROTATION')
            if rotation_angle is not None:
                self.set_rotation_angle(rotation_angle)
        return self.rotation_angle

    def get_value_resolution(self):
        if self.value_resolution is None:
            value_resolution = get_tag_from_scaninfo (self.path, 'ValueResolution')
            if value_resolution is not None:
                self.set_value_resolution(value_resolution)
        return self.value_resolution

    def get_RL_lambda(self):
        if self.RL_lambda is None:
            RL_lambda = get_tag_from_scaninfo (self.path, 'DeconvolveRLTVLambda')
            if RL_lambda is not None:
                self.set_RL_lambda(RL_lambda)
        return self.RL_lambda
    
    def get_background(self):
        if self.background is None:
            background = get_tag_from_scaninfo (self.path, 'Background')
            if background is not None:
                self.set_background(background)
        return self.background

    def get_protocol(self):
        if self.protocol is None:
            protocol = get_tag_from_scaninfo (self.path, 'Protocol')
            if protocol is not None:
                self.set_protocol(protocol)
        return self.protocol

    def get_sample_format(self):
        if self.sample_format is None:
            sample_format = get_tag_from_scaninfo(self.path, 'SampleFormat')
            if sample_format is not None:
                self.set_sample_format(sample_format)
        return self.sample_format

class Configuration(PathInfo):

    """
    PathInfo subclass with microscope information stored in a configuration.txt file.
    """

    _widefield_protocol_modes = ['FluorescenceTransmission', 'Fluorescence_Transmission',
                                 'Fluorescence',
                                 'MyocyteMechanics', 'Myocyte_Mechanics',
                                 'MyocyteMechanicsFluorescence'
                                 ]
    _confocal_protocol_modes = ['Confocal_Image', 'ConfocalImage']

    _rics_protocol_modes = ['ConfocalRICS', 'Confocal_RICS']

    def has_imperx(self):
        path = self.path
        if 'Imperx' in path:
            return True
        protocol_mode = get_tag_from_configuration(path, 'main_protocol_mode')
        if protocol_mode in ['MyocyteMechanics', 'MyocyteMechanicsFluorescence']:
            return True

    def get_protocol(self):
        if self.protocol is None:
            protocol_mode = get_tag_from_configuration(self.path, 'main_protocol_mode')
            if protocol_mode in self._widefield_protocol_modes + self._confocal_protocol_modes:
                self.set_protocol('image')
            if protocol_mode in self._rics_protocol_modes:
                self.set_protocol('rics')
        return self.protocol

    def get_microscope_type(self):
        if self.microscope_type is None:
            protocol_mode = get_tag_from_configuration(self.path, 'main_protocol_mode')
            microscope_type = None
            if protocol_mode in self._confocal_protocol_modes + self._rics_protocol_modes:
                microscope_type = 'confocal'
            elif protocol_mode in self._widefield_protocol_modes:
                microscope_type = 'widefield'
            else:
                raise NotImplementedError (`protocol_mode`)
            if microscope_type is not None:
                self.set_microscope_type(microscope_type)
        return self.microscope_type

    def get_nof_stacks(self):
        if self.nof_stacks is None:
            shape = self.get_shape()
            nof_stacks = None
            if shape is not None:
                path = self.path
                protocol_mode = get_tag_from_configuration(path, 'main_protocol_mode')
                if protocol_mode in self._widefield_protocol_modes + self._confocal_protocol_modes + self._rics_protocol_modes:
                    if get_tag_from_configuration(path, 'PROTOCOL_Z_STACKER_Enable'):
                        n = int(get_tag_from_configuration(path, 'PROTOCOL_Z_STACKER_NumberOfFrames')) or 1
                        nof_stacks = shape[0] // n
                    else:
                        nof_stacks = 1
                else:
                    raise NotImplementedError (`protocol_mode`)
            if nof_stacks is not None:
                self.set_nof_stacks(nof_stacks)
        return self.nof_stacks



    def get_shape(self):
        if self.shape is None:
            path = self.path
            protocol_mode = get_tag_from_configuration(path, 'main_protocol_mode')
            n = int(get_tag_from_configuration(path, 'PROTOCOL_Z_STACKER_NumberOfFrames')) or 1
            if protocol_mode in self._confocal_protocol_modes + self._rics_protocol_modes:
                imagesize_x = int(get_tag_from_configuration(path, 'CONFOCAL_ImageSizeX')) #px
                imagesize_y = int(get_tag_from_configuration(path, 'CONFOCAL_ImageSizeY')) #px
                shape = (n, imagesize_y, imagesize_x)
            elif protocol_mode in self._widefield_protocol_modes:
                if self.has_imperx():
                    imagesize_x = int(get_tag_from_configuration(path, 'CAMERA_IMPERX_ImageSizeX')) #px
                    imagesize_y = int(get_tag_from_configuration(path, 'CAMERA_IMPERX_ImageSizeY')) #px
                else:
                    imagesize_x = int(get_tag_from_configuration(path, 'CAMERA_ANDOR_ImageSizeX')) #px
                    imagesize_y = int(get_tag_from_configuration(path, 'CAMERA_ANDOR_ImageSizeY')) #px
                shape = (n, imagesize_y, imagesize_x)
            else:
                raise NotImplementedError (`protocol_mode`)
            self.set_shape(*shape)
        return self.shape

    def get_voxel_sizes(self):
        path = self.path
        if self.voxel_sizes is None:
            protocol_mode = get_tag_from_configuration(path, 'main_protocol_mode')
            mn = float(get_tag_from_configuration(path, 'PROTOCOL_Z_STACKER_Minimum')) #um
            mx = float(get_tag_from_configuration(path, 'PROTOCOL_Z_STACKER_Maximum')) #um
            n = int(get_tag_from_configuration(path, 'PROTOCOL_Z_STACKER_NumberOfFrames')) or 1
            if protocol_mode in self._confocal_protocol_modes + self._rics_protocol_modes:
                pixelsize_x = float(get_tag_from_configuration(path, 'CONFOCAL_PixelSizeX')) #um
                pixelsize_y = float(get_tag_from_configuration(path, 'CONFOCAL_PixelSizeY')) #um
                voxel_sizes = (1e-6*(mx-mn)/(n), 1e-6*pixelsize_y, 1e-6*pixelsize_x)
            elif protocol_mode in self._widefield_protocol_modes:
                if self.has_imperx():
                    pixelsize_x = float (get_tag_from_configuration(path, 'CAMERA_IMPERX_PixelSize'))
                else:
                    pixelsize_x = float (get_tag_from_configuration(path, 'CAMERA_ANDOR_PixelSize'))
                pixelsize_y = pixelsize_x
                voxel_sizes = (1e-6*(mx-mn)/(n), 1e-6*pixelsize_y, 1e-6*pixelsize_x)
            else:
                raise NotImplementedError (`protocol_mode`)
            self.set_voxel_sizes(*voxel_sizes)
        return self.voxel_sizes

    def get_image_time (self):
        path = self.path
        print path
        if self.image_time is None:
            image_time = None
            protocol_mode = get_tag_from_configuration(path, 'main_protocol_mode')
            if protocol_mode in self._widefield_protocol_modes:
                time_path = None
                if self.has_imperx():
                    time_path = os.path.join (os.path.dirname(path), 'Imperx_index.txt')
                elif 'Andor' in path:
                    time_path = os.path.join (os.path.dirname(path), 'Andor_index.txt')
                if time_path is not None:
                    image_time = []
                    for line in open (time_path):
                        t, fn = line.strip().split()
                        image_time.append (float (t.strip()))
            if image_time is not None:
                self.set_image_time(image_time)
            else:
                raise NotImplementedError (`protocol_mode, path`)
        return self.image_time

    def get_objective_NA(self):
        path = self.path
        if self.objective_NA is None:
            protocol_mode = get_tag_from_configuration(path, 'main_protocol_mode')
            if protocol_mode in self._confocal_protocol_modes + self._rics_protocol_modes:
                objective_name = get_tag_from_configuration(path, 'olympus_optics_objective')
            elif protocol_mode in self._widefield_protocol_modes:
                objective_name = get_tag_from_configuration(path, 'optics_objective')
            else:
                raise NotImplementedError(`protocol_mode`)
            objective_params = objectives.get(objective_name)
            if objective_params is not None:
                objective_NA = objective_params.get('NA')
            else:
                raise NotImplementedError(`objective_name`)
            self.set_objective_NA (objective_NA)
        return self.objective_NA

    def get_refractive_index(self):
        path = self.path
        if self.refractive_index is None:
            protocol_mode = get_tag_from_configuration(path, 'main_protocol_mode')
            if protocol_mode in self._confocal_protocol_modes + self._rics_protocol_modes:
                objective_name = get_tag_from_configuration(path, 'olympus_optics_objective')
            elif protocol_mode in self._widefield_protocol_modes:
                objective_name = get_tag_from_configuration(path, 'optics_objective')
            else:
                raise NotImplementedError(`protocol_mode`)
            objective_params = objectives.get(objective_name)
            if objective_params is not None:
                refractive_index = objective_params.get('refractive_index')                
            else:
                raise NotImplementedError(`objective_name`)
            self.set_refractive_index (refractive_index)
        return self.refractive_index

    def get_excitation_wavelength(self):
        path = self.path
        if self.excitation_wavelength is None:
            protocol_mode = get_tag_from_configuration(path, 'main_protocol_mode')
            if protocol_mode in self._confocal_protocol_modes + self._rics_protocol_modes:
                for i in range (1,5):
                    line_enabled = int(get_tag_from_configuration(path, 'AOTF_Line%iEnable' % i))
                    if line_enabled:
                        line_freq = float(get_tag_from_configuration(path, 'AOTF_Line%iAcousticFrequency' % i))
                        line_pow = int(get_tag_from_configuration(path, 'AOTF_Line%iAcousticPower' % i))
                        if line_pow:
                            if 85 < line_freq < 100:
                                self.set_excitation_wavelength(633*1e-9)
                                break
                            elif 130 < line_freq < 145:
                                self.set_excitation_wavelength(473*1e-9)
                                break
                            raise NotImplementedError (`line_enabled, i, line_freq, line_pow`)
            elif protocol_mode in self._widefield_protocol_modes:
                self.set_excitation_wavelength(540*1e-9)
            else:
                raise NotImplementedError(`protocol_mode`)
        return self.excitation_wavelength

    def get_emission_wavelength(self):
        path = self.path
        if self.emission_wavelength is None:
            protocol_mode = get_tag_from_configuration(path, 'main_protocol_mode')
            wavelength = None
            if protocol_mode in self._confocal_protocol_modes + self._rics_protocol_modes:
                #XXX: emission wave length is defined by the used dye and filter combination
                filter_position = get_tag_from_configuration(path, 'thorlabs_filter_wheel_position')
                filter_params = filters.get(filter_position)
                if filter_params is not None:
                    wavelength = filter_params.get('wavelength')
            if wavelength is not None:
                self.set_emission_wavelength(wavelength*1e-9)
        return self.emission_wavelength

    def get_rotation_angle(self):
        path = self.path
        if self.rotation_angle is None:
            rotation_angle = None
            protocol_mode = get_tag_from_configuration(path, 'main_protocol_mode')
            if protocol_mode in self._confocal_protocol_modes + self._rics_protocol_modes:
                rotation_angle = get_tag_from_configuration(path, 'CONFOCAL_RotationAngle')
            elif protocol_mode in self._widefield_protocol_modes:
                rotation_angle = 0
            else:
                raise NotImplementedError(`protocol_mode`)                
            if rotation_angle is not None:
                self.set_rotation_angle (rotation_angle)
        return self.rotation_angle

    def get_sample_format(self):
        if self.sample_format is None:
            self.set_sample_format('uint')
        return self.sample_format

    def omexml(self, options=None):
        from .ome_configuration import OMEConfiguration
        OMEConfiguration (self.path).process()

class Tiffinfo(PathInfo):

    """
    PathInfo subclass with microscope information stored in .tif or .lsm file.
    """

    def __init__(self, path):
        PathInfo.__init__(self, path)
        self.tif = tif = tifffile.TIFFfile(path)

        if not tif.is_lsm:
            try:
                image_description = tif[0].image_description
            except AttributeError:
                image_description = ''
            pathinfo = tempfile.mktemp()
            f = open(pathinfo, 'w')
            f.write(image_description)
            f.close()
            self.pathinfo = Scaninfo(pathinfo)
            self.pathinfo.set_shape (*self.get_shape())

    def get_microscope_type(self):
        if self.microscope_type is None:
            tif = self.tif
            if tif.is_lsm:
                self.set_microscope_type('confocal')
            else:
                self.set_microscope_type(self.pathinfo.get_microscope_type())
        return self.microscope_type

    def get_shape(self):
        if self.shape is None:
            tif = self.tif[0]
            shape = tif.shape
            self.set_shape(*shape)
        return self.shape

    def get_voxel_sizes (self):
        if self.voxel_sizes is None:
            tif = self.tif
            if tif.is_lsm:
                lsmi = tif[0].cz_lsm_info
                self.set_voxel_sizes(lsmi.voxel_size_z, lsmi.voxel_size_y, lsmi.voxel_size_x)
            else:
                self.set_voxel_sizes(*self.pathinfo.get_voxel_sizes())
        return self.voxel_sizes

    def get_objective_NA(self):
        if self.objective_NA is None:
            tif = self.tif
            if tif.is_lsm:
                objective_name = get_tag_from_lsm_file(self.path, 'AcquisitionParameters.Objective')
                objective_params = objectives.get(objective_name)
                if objective_params is not None:
                    objective_NA = objective_params.get('NA')
                else:
                    raise NotImplementedError (`objective_name`)
                self.set_objective_NA(objective_NA)
            else:
                self.set_objective_NA(self.pathinfo.get_objective_NA())
        return self.objective_NA

    def get_refractive_index(self):
        if self.refractive_index is None:
            tif = self.tif
            if tif.is_lsm:
                objective_name = get_tag_from_lsm_file(self.path, 'AcquisitionParameters.Objective')
                objective_params = objectives.get(objective_name)
                if objective_params is not None:
                    refractive_index = objective_params.get('refractive_index')
                else:
                    raise NotImplementedError (`objective_name`)
                self.set_refractive_index(refractive_index)
            else:
                self.set_refractive_index(self.pathinfo.get_refractive_index())
        return self.refractive_index

    def get_excitation_wavelength(self):
        if self.excitation_wavelength is None:
            tif = self.tif
            if tif.is_lsm:
                value = get_tag_from_lsm_file(self.path, 'Tracks.Attenuators.Attenuator1.Wavelength')
                if value is not None:
                    excitation_wavelength = float(value.split()[0])
                    self.set_excitation_wavelength(excitation_wavelength*1e-9)
            else:
                self.set_excitation_wavelength(self.pathinfo.get_excitation_wavelength())
        return self.excitation_wavelength

    def get_rotation_angle(self):
        if self.rotation_angle is None:
            tif = self.tif
            if tif.is_lsm:
                value = get_tag_from_lsm_file(self.path, 'AcquisitionParameters.Rotation')
                if value is not None:
                    rotation_angle = float(value.split()[0])
                    self.set_rotation_angle(rotation_angle)
            else:
                self.set_rotation_angle(self.pathinfo.get_rotation_angle())  
        return self.rotation_angle

    def get_sample_format(self):
        if self.sample_format is None:
            tif = self.tif
            sample_format = getattr(tif, 'sample_format', None)
            if sample_format is not None:
                self.set_sample_format(sample_format)
        return self.sample_format

    def get_detectors(self):
        nof_detectors = int(get_tag_from_lsm_file(self.path, 'Tracks.Detectors.NumberDetectors'))
        l =[]
        for n in range (1, nof_detectors+1):
            name = get_tag_from_lsm_file(self.path, 'Tracks.Detectors.Detector%s.ImageChannelName' % (n))
            pinhole = get_tag_from_lsm_file(self.path, 'Tracks.Detectors.Detector%s.PinholeDiameter' % (n))
            pinhole, unit = pinhole.strip (). split()
            unit = {'\xb5'+'m':1e6}.get (unit, 1)
            l.append (dict(name=name, index=n-1, pinhole=float(pinhole) * unit))
        return l

class Rawinfo(PathInfo):

    """
    PathInfo subclass to represent raw data (.f32, .u8 etc) files with shape information stored in a .hdr file.
    """

    def _get_header_file(self):
        fn, ext = os.path.splitext(self.path)
        hdr = fn + '.hdr'
        return hdr

    def _get_ext_type(self):
        fn, ext = os.path.splitext(self.path)
        return ext[1:]

    def get_shape(self):
        if self.shape is None:
            shape = []
            for line in file (self._get_header_file ()):
                try:
                    d = int(line.strip())
                except ValueError:
                    break
                shape.append(d)
            self.set_shape (*shape)
        return self.shape

    def get_sample_format(self):
        if self.sample_format is None:
            e = self._get_ext_type ()
            sample_format = dict (f='float', i='int', u='uint').get (e[0])
            if sample_format is not None:
                self.set_sample_format (sample_format)
        return self.sample_format

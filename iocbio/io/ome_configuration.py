""" Configuration to OME-XML converter tools.

Module content
--------------
"""
from __future__ import absolute_import

__all__ = ['get_xml']

import os
import sys
import numpy
import tempfile
from uuid import uuid1 as uuid
from lxml import etree
from .ome import ome, ATTR, namespace_map, validate_xml, OMEBase
from . import pathinfo
from libtiff import TIFFfile, TIFFimage

def get_AcquiredDate(info):
    year, month, day, hour, minute, second, ns = map(int, info['m_time'].split())
    year += 1900
    return ome.AcquiredDate('%s-%s-%sT%s:%s:%s.%s' % (year,month, day, hour, minute, second, ns))

def get_Description (info):
    pass

class OMEConfiguration(OMEBase):

    _detectors = ['Imperx', 'Andor', 'Confocal']

    def __init__(self, path):
        OMEBase.__init__(self)
        dir_path = self.dir_path = os.path.dirname(path)
        self.config_path = path
        self.info_path = os.path.join(dir_path,'info.txt')
        
        config = self.config = pathinfo.get_tag_from_configuration(self.config_path, None) or {}
        info = self.info = pathinfo.get_tag_from_configuration(self.info_path, None) or {}

        data = self.data = {}

        for d in self._detectors:
            d_path = os.path.join(dir_path, '%s_index.txt' % (d))
            if os.path.isfile (d_path):
                d_index = {}
                for index, line in enumerate(open (d_path).readlines ()):
                    t, fn = line.strip().split()
                    t = float(t)
                    d_index[t, index] = os.path.join(dir_path, fn)
                data[d] = d_index

        if 0:
            for k in sorted (config):
                print '%s: %r' % (k,config[k])
    
    def get_AcquiredDate (self):
        year, month, day, hour, minute, second, ns = map(int, self.config['m_time'].split())
        year += 1900
        return '%s-%s-%sT%s:%s:%s.%s' % (year,month, day, hour, minute, second, ns)

    def iter_Experiment(self, func):
        mode = self.config['main_protocol_mode']
        descr = self.info.get('DESCRIPTION')
        e = func(Type="Other", ID='Experiment:%s' % (mode))
        if descr is not None:
            e.append(ome.Description(descr))
        return [e]

    def iter_Image(self, func):
        sys.stdout.write('iter_Image: reading image data from TIFF files\n')
        for detector in self.data:
            sys.stdout.write('  detector: %s\n' % (detector))
            d_index = self.data[detector]

            # write the content of tiff files to a single raw files
            f,fn,dtype = None, None, None
            time_set = set()
            mn, mx = None, None
            for t, index in sorted(d_index):
                sys.stdout.write('\r  copying TIFF image data to RAW file: %5s%% done' % (int(100.0*index/len(d_index))))
                sys.stdout.flush()
                time_set.add(t)
                tif = TIFFfile(d_index[t, index])
                samples, sample_names = tif.get_samples()
                assert len (sample_names)==1,`sample_names`
                data = samples[0]
                if mn is None:
                    mn, mx = data.min(), data.max()
                else:
                    mn = min (data.min(), mn)
                    mx = min (data.max(), mx)
                if f is None:
                    shape = list(data.shape)
                    dtype = data.dtype
                    fn = tempfile.mktemp(suffix='.raw', prefix='%s_%s_' % (detector, dtype))
                    f = open (fn, 'wb')
                else:
                    assert dtype is data.dtype,`dtype,data.dtype`
                    shape[0] += 1
                data.tofile(f)

            if f is None:
                continue
            f.close ()
            shape = tuple (shape)

            xsz = shape[2]
            ysz = shape[1]
            tsz = len(time_set)
            zsz = shape[0] // tsz
            if zsz==1:
                order = 'XYTZC'
            else:
                order = 'XYZTC'
            sys.stdout.write("\n  RAW file contains %sx%sx%sx%sx%s [XYZTC] array, dtype=%s, MIN/MAX=%s/%s\n" \
                                 % (xsz, ysz,zsz,tsz,1, dtype, mn,mx))
            assert zsz*tsz==shape[0]

            tif_filename = '%s.ome.tif' % (detector)
            sys.stdout.write("  creating memmap image for OME-TIF file %r..." % (tif_filename))
            sys.stdout.flush()
            mmap = numpy.memmap(fn, dtype=dtype, mode='r', shape=shape)
            tif_image = TIFFimage(mmap)
            tif_uuid = self._mk_uuid()
            self.tif_images[tif_filename, tif_uuid] = tif_image
            sys.stdout.write (' done\n')
            sys.stdout.flush()


            #todo attributes: PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ, TimeIncrement
            #todo elements: Channel, BIN:BinData, TiffData, MetadataOnly, Plane, SA:AnnotationRef
            tiffdata = ome.TiffData()
            pixels = ome.Pixels(tiffdata, DimensionOrder=order, ID='Pixels:%s' % (detector),
                                SizeX = str(xsz), SizeY = str(ysz), SizeZ = str(zsz), SizeT=str(tsz), SizeC = str(1),
                                Type = self.dtype2PixelIType (dtype),
                                )

            #todo attributes: Name
            #todo elements: ExperimenterRef, Description, ExperimentRef, GroupRef, DatasetRef , InstrumentRef,
            #               ObjectiveSettings, ImagingEnvironment, StageLabel, ROIRef, MicrobeamManipulationRef, AnnotationRef
            image = ome.Image (ome.AcquiredDate (self.get_AcquiredDate()),
                               pixels, 
                               ID='Image:%s' % (detector))
            yield image
        return


def make_ome_xml(path, validate=True):
    """ Return OME-XML string.

    Parameters
    ----------
    path : str
      Path to configuration.txt file.
    """
    return OMEConfiguration(path).process ()


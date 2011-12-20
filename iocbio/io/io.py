""" Provides functions to load and save image stacks.
"""

__autodoc__ = ['RowFile', 'load_image_stack', 'save_image_stack', 'get_pathinfo']

# Author: Pearu Peterson
# Created: 2009

__all__ = ['load_image_stack', 'save_image_stack', 'RowFile', 'get_pathinfo']

import re
import os
import sys
import time
import numpy
from StringIO import StringIO
from .tifffile20100410_py25 import TIFFfile
from .libtiff import TIFF
from glob import glob
from .. import utils
from .pathinfo import Tiffinfo, Scaninfo, Configuration, Rawinfo


tif_extensions = ['.tif', '.tiff', '.lsm'] # files will be read with tifffile
raw_extensions = ['.raw']                  # files will be read with numpy.fromfile, the fastest method
data_extensions = ['.data']

if os.name=='nt':
    psflib_dir = r'C:\iocbio\psflib'
else:
    psflib_dir = os.path.join(os.environ['HOME'], 'iocbio/psflib')
    if not os.path.exists (psflib_dir):
        psflib_dir = '/usr/local/share/iocbio/psflib'

def israwfile(path):
    ext = os.path.splitext(path)[1]
    if ext in raw_extensions:
        return True
    if re.match(r'[.][fuic](8|16|32|64|128|256|512)', ext):
        return True
    return False

def get_psf_libs():
    """
    Return a mapping of PSF file locations found in ``iocbio.io.io.psflib_dir``.
    """
    psf_files = glob(os.path.join(psflib_dir, '*', 'psf.tif'))
    return dict([(os.path.basename(os.path.dirname (fn)), os.path.abspath(fn)) for fn in psf_files])

def get_psf_path(options):
    """
    Return PSF file path determined by options.
    """
    psf_path = getattr(options, 'psf_path', None)
    if not psf_path:
        psflib = getattr(options, 'psf_lib', None)
        if psflib is None:
            raise ValueError ('Unable to find psf_path information from options: %s' % (options))
        if psflib=='<select>':
            raise ValueError ('You must either select PSF library name or specify --psf-path|--kernel-path option.')
        psflibs = get_psf_libs()
        psf_path = psflibs[psflib]
    return fix_path(psf_path)

def get_indexed_files(path, file_prefix):
    """ Return a list of sorted image file names in path.
    The file names must have specified prefix and end with
    number parts.
    """
    files = None
    for ext in raw_extensions + tif_extensions:
        files = glob(os.path.join(path,file_prefix+'[0-9]*'+ext))
        if files: break
    if not files:
        raise ValueError('No image files with prefix %r found in %r' % (file_prefix, path))
    prefix = suffix = files[0]
    for f in files:
        while not f.endswith (suffix) and suffix:
            suffix = suffix[1:]
        while not f.startswith(prefix) and prefix:
            prefix = prefix[:-1]
    #print `prefix,suffix`
    l = []
    for f in files:
        numpart = f[len(prefix):-len (suffix)]
        try:
            n = int (numpart)
        except ValueError, msg:
            print ('Failed to establish index from %r for file name %r (prefix=%r, suffix=%r): %s' % (numpart, f, prefix, suffix, msg))
            continue
        l.append((n,f))
    l.sort()
    return [f for (n,f) in l]

def fix_path (path):
    """
    Return valid path to microscope data in ``path``.
    """
    if path is None:
        raise ValueError('Expected path name but got None.')
    orig_path = path
    if os.path.isfile(path):
        path = os.path.realpath(path)
        if os.path.basename(path) in ['configuration.txt',
                                       'SCANINFO.txt',
                                       'PATHINFO.txt',
                                       'DECONVOLVEINFO.txt']:
            path = os.path.dirname(path)
        elif path.endswith('_DECONVOLVEINFO.txt'):
            path = path[:-19]
        elif path.endswith('_PATHINFO.txt'):
            path = path[:-13]
        if not os.path.isdir(path):
            if not os.path.isfile(path) and os.path.isfile(path+'f'):
                path = path + 'f'
        if orig_path != path:
            print 'Fixing',orig_path,'to',path
    return path

def get_pathinfo_file(path):
    """
    Return path to PATHINFO.txt file.
    """
    if os.path.isdir(path):
        for fn in [os.path.join(path, 'PATHINFO.txt'),
                   os.path.join(path, 'DECONVOLVEINFO.txt'),
                   ]:
            if os.path.isfile (fn):
                return fn
    b, e = os.path.splitext(path)
    for fn in [
        path + '_PATHINFO.txt',
        path + 'f_PATHINFO.txt',
        path[:-1] + '_PATHINFO.txt',
        path + '_DECONVOLVEINFO.txt',
        path + 'f_DECONVOLVEINFO.txt',
        path[:-1] + '_DECONVOLVEINFO.txt',
        ]:
        if os.path.isfile (fn):
            return fn
    return path + '_PATHINFO.txt'

def get_pathinfo (path):
    """ Return Pathinfo instance of path.

    Returns
    -------
    pathinfo : `iocbio.io.pathinfo.PathInfo`
    """
    if not os.path.exists(path):
        new_path = os.path.dirname(path)
        if not os.path.exists(new_path):
            raise ValueError('Path to image stack does not exist: %r' % (path))
        path = new_path

    path = fix_path(path)

    if os.path.isfile(path):
        dirpath = os.path.dirname(path)    
    else:
        dirpath = path

    pathinfo_txt = get_pathinfo_file(path)
    scaninfo_txt = os.path.join(dirpath, 'SCANINFO.txt')
    configuration_txt = os.path.join(dirpath, 'configuration.txt')

    if os.path.isfile(pathinfo_txt):
        pathinfo = Scaninfo(pathinfo_txt)
    elif os.path.isfile(scaninfo_txt):
        pathinfo = Scaninfo(scaninfo_txt)
    elif os.path.isfile(configuration_txt):
        pathinfo = Configuration(configuration_txt)
    elif israwfile(path):
        pathinfo = Rawinfo(path)
    else:
        pathinfo = None
    return pathinfo

def load_image_stack(path, options=None, file_prefix=None):
    """ 
    Load image stacks from path

    Parameters
    ----------
    path : str
      Path to microscope data file or directory.


    options : {None, optparse.Values}

    Returns
    -------
    images : numpy.ndarray
    pathinfo : `iocbio.io.pathinfo.PathInfo`

    Notes
    -----

    path is a directory of images or an image file, supported
    image extensions are .raw, .tif, .tiff, .lsm. For raw images
    the images shape must be available is some form in
    images directory. Currently supported forms are

      - ``SCANINFO.txt`` - generated by ImageJ
      - ``configuration.txt`` - generated by SysBio microscope application
      - ``*PATHINFO.txt`` - generated by `iocbio.io.save_image_stack`
    """
    options = utils.Options(options)
    if file_prefix is None:
        file_prefix = '*'
    if not os.path.exists(path):
        file_prefix = os.path.basename(path) + '*'
    pathinfo = get_pathinfo(path)

    if os.path.isfile(path):
        base, ext = os.path.splitext (path)
        ext = ext.lower()
        if ext in tif_extensions:

            if pathinfo is None:
                pathinfo = Tiffinfo(path)
                images = pathinfo.tif.asarray()
                if len (images.shape)==4:

                    detector = [d for d in pathinfo.get_detectors() if d['pinhole']][0]
                    print 'Using detector with non-zero pinhole:', detector                    
                    images = images[:,detector['index'],:,:]
            else:
                sample_format = pathinfo.get_sample_format()
                tif = TIFFfile(path, sample_format = sample_format)
                images = tif.asarray()
            pathinfo.set_shape(*images.shape)
            print '-> image array with shape=%s and dtype=%s' % (images.shape, images.dtype)
            return images, pathinfo
        elif israwfile(path):
            if pathinfo is None:
                raise ValueError('Cannot determine the shape information of image stacks from '\
                                     +`path`+' (no info files in the path directory)')
            shape = pathinfo.get_shape()
            sz = os.path.getsize(path)
            bytes = sz // (shape[0]*shape[1]*shape[2])
            assert bytes * shape[0] * shape[1] * shape[2] == sz, `sz, bytes, shape`
            sample_format = pathinfo.get_sample_format()
            if not sample_format:
                print 'Warning: failed to determine sample format from %r, assuming uint' % (path)
                print '         pathinfo=',pathinfo
                sample_format = 'uint'
            bits = 8*bytes
            image_type = numpy.typeDict[sample_format+str(bits)]
            images = numpy.fromfile(path, image_type)
            images.shape = shape
            pathinfo.set_shape(*shape)
            print '-> image array with shape=%s and dtype=%s' % (images.shape, images.dtype)
            return images, pathinfo
    elif os.path.isdir(path):
        images = None
        image_type = None

        indexed_files = get_indexed_files(path, file_prefix)
        
        if options is not None:
            max_nof_stacks = getattr(options, 'max_nof_stacks', None)
            if max_nof_stacks == 'none':
                max_nof_stacks = None
        else:
            max_nof_stacks = None

        if not indexed_files:
            raise ValueError('No files to be read (nn=%s, shape=%s)' % (len(indexed_files), shape))
        if max_nof_stacks:
            print 'Reading upto %s image files but upto %s stacks from %s:' % (len(indexed_files), max_nof_stacks, path)
        else:
            print 'Reading %s image files from %s:' % (len(indexed_files), path)

        bar = utils.ProgressBar(1,len(indexed_files), prefix='  ', show_percentage=False)
        max_i = None
        for i, filename in enumerate(indexed_files):
            if max_i is not None and i>=max_i:
                bar.updateComment(' Reached to a maximum nof stacks %s, breaking.' % (max_nof_stacks))
                bar(i)
                break
            bar.updateComment(' '+filename)
            bar(i)
            base, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext in tif_extensions:
                if images is None:
                    if pathinfo is None:
                        pathinfo = Tiffinfo(filename)
                        image = pathinfo.tif.asarray()
                    else:
                        tif = TIFFfile(filename)
                        image = tif.asarray()
                    shape = (len(indexed_files),) + image.shape
                    image_type = image.dtype
                    pathinfo.set_shape(*shape)
                    if max_nof_stacks is not None:
                        total_stacks = pathinfo.get_nof_stacks()
                        if total_stacks and max_nof_stacks < total_stacks:
                            shape = ((shape[0] // total_stacks) * max_nof_stacks,) + shape[1:]
                            pathinfo.set_shape(*shape)
                            pathinfo.set_nof_stacks(max_nof_stacks)
                            max_i = shape[0]
                    images = numpy.empty(shape, image_type)
                else:
                    if max_nof_stacks is not None and max_i is None:
                        max_i = max_nof_stacks
                    tif = TIFFfile(filename)
                    image = tif.asarray()                    
            elif ext in raw_extensions:
                if images is None:
                    if pathinfo is None:
                        raise ValueError('Cannot determine the shape information of image stacks from '\
                                             +`path`+' (no info files in the path directory)')
                    shape = pathinfo.get_shape()
                    sz = os.path.getsize(filename)
                    bytes = sz // (shape[1]*shape[2])
                    assert bytes * shape[1] * shape[2] == sz, `sz, bytes, shape`
                    bits = 8*bytes
                    sample_format = pathinfo.get_sample_format() or 'uint'
                    image_type = numpy.typeDict[sample_format+str(bits)]
                    if max_nof_stacks is not None:
                        total_stacks = pathinfo.get_nof_stacks()
                        if total_stacks and max_nof_stacks < total_stacks:
                            shape = (shape[0] // total_stacks) * max_nof_stacks
                            pathinfo.set_shape(*shape)
                            pathinfo.set_nof_stacks(max_nof_stacks)
                            max_i = shape[0]
                    images = numpy.empty(shape, image_type)
                image = numpy.fromfile (filename, image_type)
                image.shape  = shape[1:]
            else:
                raise NotImplementedError ('Reading image from '+`filename`)
            assert image.shape==shape[1:],`image.shape, shape[1:]`
            images[i] = image
        bar(i)
        print
        print '-> image array with shape=%s and dtype=%s' % (images.shape, images.dtype)
        return images, pathinfo
    elif not os.path.exists (path):
        raise IOError ('Image path does not exist: %r' % (path))
    raise NotImplementedError ('Reading image stack from '+`path`)

def read_csv(filename):
    """
    Read the content of a CSV file assuming that the first line
    contains labels and the rest of the lines contain numbers.

    Returns
    -------
    dct : dict
      Mapping of labels and values
    """
    import csv
    d = {}
    csvfile = open(filename)
    dialect = csv.Sniffer().sniff(csvfile.read(1024))
    csvfile.seek(0)
    reader = csv.reader(csvfile, dialect)
    titles = reader.next()[:-1]
    dct = dict([(t,[]) for t in titles])
    for row in reader:
        for t,d in zip(titles, row[:-1]):
            dct[t].append(eval(d))
    csvfile.seek(0)
    for t in dct:
        dct[t] = numpy.array(dct[t])
    #print len(csvfile.readlines())
    return dct

def get_rics_info(path, options=None):
    """ 
    Return RICS information. [EXPERIMENTAL]

    Returns
    -------
    dct : dict
    pathinfo : `iocbio.io.pathinfo.PathInfo`
    """
    options = utils.Options(options)
    import numpy
    path = fix_path(path)

    if os.path.isfile(path):
        dirpath = os.path.dirname(path)    
    else:
        dirpath = path

    configuration_txt = os.path.join(dirpath, 'configuration.txt')

    assert os.path.isfile(configuration_txt),` configuration_txt`
    assert os.path.isdir(path),`path`
    pathinfo = Configuration(configuration_txt)    

    rics_csv = os.path.join(dirpath, 'ProtocolConfocalRICS_rics.csv')
    assert os.path.isfile(rics_csv), `rics_csv`
    dct = read_csv(rics_csv)
    n = len(dct[dct.keys ()[0]])
    dct['filename'] = get_indexed_files(path, file_prefix='*')[:n]

    return dct, pathinfo

def save_image_stack(image_stack, path, indices=None,
                         options = None):
    """ Save ImageStack instance to path.

    Parameters
    ----------
    image_stack : `iocbio.io.image_stack.ImageStack`
    path : `iocbio.io.pathinfo.PathInfo`
    indices : {None, tuple}

    Notes
    -----
    <pathpart>PATHINFO.txt file will be created to path directory that
    contains the shape and other information about image stacks.
    """
    options = utils.Options(options)

    if os.path.isfile(path):
        dirpath = os.path.dirname(path)
        prefix, ext = os.path.splitext(path)
        ext = ext.lower()
        if not ext: ext = '.raw'
    elif os.path.isdir(path):
        dirpath = path
        ext = 'dir'
        prefix = 'image_'
    elif not os.path.exists(path):
        base, ext = os.path.splitext(path)
        ext = ext.lower()
        dirpath = path if not ext else os.path.dirname(path)
        if ext:
            prefix = base
        else:
            ext = 'dir'
            prefix = 'image_'

    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)

    if ext in tif_extensions + raw_extensions + data_extensions:
        pathinfo_txt = path + '_PATHINFO.txt'
    else:
        pathinfo_txt = os.path.join(dirpath, 'PATHINFO.txt')

    image_stack.pathinfo.save(pathinfo_txt)
    images = image_stack.images

    if ext in raw_extensions:
        f = open(path, 'wb')
        images.tofile(f)
        f.close()
    elif ext=='dir':
        for i in range(images.shape[0]):
            f = open(os.path.join(dirpath,  prefix + '%.5i.raw' % (i)), 'wb')
            images[i].tofile(f)
            f.close()
    elif ext in tif_extensions:
        tif = TIFF.open(path, mode='w')
        compression = options.get(tiff_compression = 'none')
        buf = StringIO()
        image_stack.pathinfo.save(buf)
        tif.SetField('ImageDescription', buf.getvalue ())
        tif.write_image(images, compression=compression)
        tif.close()
    elif ext=='.data':
        f = open(path, 'w')
        voxel_sizes = numpy.array(image_stack.get_voxel_sizes()) * 1e6 #um
        t = (voxel_sizes[2], voxel_sizes[1], voxel_sizes[0],voxel_sizes[0]*voxel_sizes[1]*voxel_sizes[2],)
        fmt = 4*'%14.6e' + '\n'
        f.write(fmt % t)
        if indices is None:
            indices = numpy.ndindex(*images.shape)
        center = numpy.array(images.shape)//2
        maxvalue = images.max()
        use_value_resolution = options.get(use_value_resolution = False)
        if use_value_resolution:
            value_resolution = image_stack.pathinfo.get_value_resolution() or 0
            print 'Using Value Resolution:', value_resolution
        else:
            value_resolution = 0
        for index in indices:
            value = images[index]
            if value > value_resolution:
                position = (index - center) * voxel_sizes
                t = (position[2], position[1], position[0], value,)
                f.write (fmt % t)
        f.close()
    else:
        raise NotImplementedError(`path, ext`)

class RowFile:
    """
    Represents a row file.

    The RowFile class is used for creating and reading row files.

    The format of the row file is the following:
    - row file may have a header line containg the titles of columns
    - lines starting with ``#`` are ignored as comment lines
    """

    def __init__(self, filename, titles = None, append=False):
        """
        Parameters
        ----------

        filename : str
          Path to a row file
        titles : {None, list}
          A list of column headers for writing mode.
        append : bool
          When True, new data will be appended to row file.
          Otherwise, the row file will be overwritten.
        """
        self.filename = filename
        dirname = os.path.dirname(self.filename)
        if not os.path.exists(dirname) and dirname:
            os.makedirs(dirname)
        self.file = None
        self.nof_cols = 0
        self.append = append
        self.extra_titles = ()
        if titles is not None:
            self.header(*titles)

        self.data_sep = ', '

    def __del__ (self):
        if self.file is not None:
            self.file.close()

    def header(self, *titles):
        """
        Write titles of columns to file.
        """
        data = None
        extra_titles = self.extra_titles
        if self.file is None:
            if os.path.isfile(self.filename) and self.append:
                data_file = RowFile(self.filename)
                data, data_titles = data_file.read(with_titles=True)
                data_file.close()
                if data_titles!=titles:
                    self.extra_titles = extra_titles = tuple([t for t in data_titles if t not in titles])
            self.file = open(self.filename, 'w')
            self.nof_cols = len(titles + extra_titles)
            self.comment('@,@'.join(titles + extra_titles))
            self.comment('To read data from this file, use ioc.microscope.io.RowFile(%r).read().' % (self.filename))

            if data is not None:
                for i in range(len(data[data_titles[0]])):
                    data_line = []
                    for t in titles + extra_titles:
                        if t in data_titles:
                            data_line.append(data[t][i])
                        else:
                            data_line.append(0)
                    self.write(*data_line)

    def comment (self, msg):
        """
        Write a comment to file.
        """
        if self.file is not None:
            self.file.write ('#%s\n' % msg)
            self.file.flush ()

    def write(self, *data):
        """
        Write a row of data to file.
        """
        if len (data) < self.nof_cols:
            data = data + (0, ) * (self.nof_cols - len (data))
        assert len (data)==self.nof_cols,`len (data), self.nof_cols`
        self.file.write(', '.join(map(str,data)) + '\n')
        self.file.flush()

    def _get_titles (self, line):
        if line.startswith('"'): # csv file header
            self.data_sep = '\t'
            return tuple([t[1:-1] for t in line.strip().split('\t')])
        return tuple([t.strip() for t in line[1:].split('@,@')])

    def read(self, with_titles = False):
        """
        Read data from a row file.

        Parameters
        ----------
        with_titles : bool
          When True, return also column titles.

        Returns
        -------
        data : dict
          A mapping of column values.
        titles : tuple
          Column titles.
        """
        f = open (self.filename, 'r')
        titles = None
        d = {}
        for line in f.readlines():
            if titles is None:
                titles = self._get_titles(line)
                for t in titles:
                    d[t] = []
                continue
            if line.startswith ('#'):
                continue
            data = line.strip().split(self.data_sep)
            for i, t in enumerate (titles):
                try:
                    v = float(data[i])
                except (IndexError,ValueError):
                    v = 0.0
                d[t].append(v)
        f.close()
        if with_titles:
            return d, titles
        return d

    def close (self):
        """
        Close row file.
        """
        if self.file is not None:
            print 'Closing ',self.filename
            self.file.close ()
            self.file = None

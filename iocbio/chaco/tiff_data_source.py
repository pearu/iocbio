from __future__ import division

__all__ = ['TiffDataSource']

import os
from collections import defaultdict
from libtiff import TIFFfile, TiffArray, TiffChannelsAndFiles, TiffFiles, TiffBase

from enthought.traits.api import Instance, List, Str, Dict, Enum, File, Button, Property, cached_property, Tuple, Float
from enthought.traits.ui.api import EnumEditor, View, Item, FileEditor, VGroup, Group, Tabbed, TupleEditor
from enthought.traits.ui.file_dialog import OpenFileDialog

from .base_data_source import BaseDataSource
from .tiff_file_info import TiffFileInfo

tiff_filter = [] #[u"TIFF file (*.tif)|*.tif|",u"|LSM file (*.lsm)|*.lsm"]

class TiffDataSource(BaseDataSource):

    open_button = Button ('Load TIFF file..')

    file_name = File (filter = tiff_filter)

    available_channels = List(Str)
    selected_channel = Str(editor=EnumEditor(name = 'available_channels', cols=1))

    # Private traits
    tiff = Instance (TiffBase)
    tiff_array = Instance(TiffArray)
    tiff_array_info = Dict

    view = View(
        Item ('open_button', show_label=False),'_',
        Item ('file_name', style='readonly', springy=True),
        Item ('selected_channel', style = 'custom', label='Channel'),
        Item ('kind', style='custom', label = 'View as'),
        Item ('voxel_sizes', editor=TupleEditor (cols=3, labels = ['Z', 'Y', 'X']), visible_when='is_image_stack'), 
        Item ('pixel_sizes', editor=TupleEditor (cols=2, labels = ['Y', 'X']), visible_when='is_image_timeseries'), 
        '_',
        Item ('description', style='readonly', show_label=False),
        scrollable = True,
        resizable = True)


    def _open_button_changed (self):
        tiffinfo = TiffFileInfo()
        fd = OpenFileDialog(file_name = os.path.abspath(self.file_name or ''),
                            filter = tiff_filter,
                            extensions = tiffinfo)
        if fd.edit_traits(view = 'open_file_view' ).result and tiffinfo.is_ok:
            self.file_name = fd.file_name
            #self.kind = tiffinfo.kind
            self.description = tiffinfo.description

    def _file_name_changed(self):
        print self.file_name
        if not os.path.isfile(self.file_name):
            raise ValueError ("File does not exist: %r" % (self.file_name))
        self.reset()

        tiff_array_info = {}
        if os.path.basename(self.file_name)=='configuration.txt':
            tiff_files = {}
            for d in ['Imperx', 'Andor', 'Confocal']:
                dir_path = os.path.dirname(self.file_name)
                d_path = os.path.join(dir_path, '%s_index.txt' % (d))
                if not os.path.isfile (d_path): # mari-ism support
                    d_path = os.path.join(dir_path, d, '%s_index.txt' % (d))
                if os.path.isfile (d_path):
                    d_index = {}
                    time_map = defaultdict(lambda:[])
                    for index, line in enumerate(open (d_path).readlines ()):
                        t, fn = line.strip().split()
                        t = float(t)
                        fn = os.path.join(os.path.dirname(d_path), fn)
                        d_index[t, index] = fn
                        time_map[fn].append(t)
                    files = [d_index[k] for k in sorted (d_index)]
                    tiff = TiffFiles(files, time_map = time_map)
                    channel_label = '%s' % (d)
                    tiff_files[channel_label] = tiff
                    tiff_array_info[channel_label] = dict (channel=channel_label, subfile_type=0, sample_index=0)
            tiff = TiffChannelsAndFiles(tiff_files)
        else:
            tiff = TIFFfile(self.file_name)

            for subfile_type in tiff.get_subfile_types():
                ifd = tiff.get_first_ifd(subfile_type=subfile_type)
                depth = tiff.get_depth(subfile_type=subfile_type)
                width = ifd.get_value('ImageWidth')
                height = ifd.get_value('ImageLength')
                if subfile_type!=0:
                    print '%s._file_name_changed: ignoring subfile_type %r' % (self.__class__.__name__, subfile_type)
                    continue

                for i, (name, dtype) in enumerate(zip (ifd.get_sample_names(), ifd.get_sample_dtypes())):
                    tname = str(dtype)
                    channel_label = '%s: %s [%sx%sx%s %s]' % (subfile_type, name, depth, height, width, tname)
                    tiff_array_info[channel_label] = dict (subfile_type=subfile_type, sample_index=i)

        self.tiff = tiff

        try:
            info = tiff.get_info()
        except Exception, msg:
            info = 'failed to get TIFF info: %s' % (msg)

        self.tiff_array_info = tiff_array_info
        self.available_channels = sorted (tiff_array_info)
        self.selected_channel = self.available_channels[0]
        self.description = info

    def _selected_channel_changed(self):
        channel = self.selected_channel
        if channel:
            tiff_array = self.tiff.get_tiff_array(**self.tiff_array_info[channel])
            self.tiff_array = tiff_array
            self.data = tiff_array
            self.voxel_sizes = tuple(tiff_array.get_voxel_sizes())
            self.pixel_sizes = tuple(tiff_array.get_pixel_sizes())

    def reset (self):
        self.selected_channel = ''
        self.available_channels = []
        self.tiff_array = None
        self.tiff_array_info = {}
        #self.kind = None
        self.data = None
        self.voxel_sizes = (1,1,1)
        self.pixel_sizes = (1,1)
        if self.tiff is not None:
            self.tiff.close()
            self.tiff = None

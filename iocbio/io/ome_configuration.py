""" Configuration to OME-XML converter tools.

Module content
--------------
"""
from __future__ import absolute_import

__all__ = ['get_xml']

import os
import re
import sys
import atexit
import numpy
import tempfile
import base64
from uuid import uuid1 as uuid
from lxml import etree
from .ome import ome, ATTR, namespace_map, validate_xml, OMEBase, sa, bf
from . import pathinfo
from libtiff import TIFFfile, TIFFimage

def pretty_name(name, split_words = False):
    # copied from sysbiosoft/microscope/trunk/python/microscope/config.py
    m = {'__S':'/', '__s':'/',
         '__M':'-', '__m':'-',
         '__P':'+', '__p':'+',
         '__C':':', '__c':':',
         '__U':'@@U@@', '__u':'@@U@@',
         '__E':'', '__e':'',
         '__X':'.', '__x':'.'
         }
    r = name
    for k,v in m.items():
        r = r.replace(k,v)
    r = r.replace('_',' ')
    if split_words:
        l = []
        for i,c in enumerate(r[:-1]):
            nc = r[i+1]
            l.append(c)
            if c.islower() and nc.isupper():
                l.append(' ')
        l.append(r[-1])
        r = ''.join(l)
    r = r.replace('@@U@@','_')
    return r


def get_AcquiredDate(info):
    year, month, day, hour, minute, second, ns = map(int, info['m_time'].split())
    year += 1900
    month += 1
    return ome.AcquiredDate('%s-%s-%sT%s:%s:%s.%s' % (year,month, day, hour, minute, second, ns))

def get_Description (info):
    pass

groups = dict(
    SysBio = dict (
        name = 'Laboratory of Systems Biology',
        #leader_id = 'markov',
        contact_id = 'sysbio',
        descr='''\
The Laboratory of Systems Biology (http://sysbio.ioc.ee) is a part of
the Center for Nonlinear Studies (http://cens.ioc.ee) in Institute Of
Cybernetics (http://www.ioc.ee), Tallinn University of Technology
(http://www.ttu.ee). Its aim is to study regulation of intracellular
processes and understand functional influences of intracellular
interactions.
''')
    )

people = dict (
    sysbio = dict (email='sysbio@sysbio.ioc.ee', institution='Laboratory of Systems Biology', groups = ['SysBio']),
    )

nikon_filters = dict(
    NikonTurretTopCube1 = dict(ex = ('FF01-465/30-25', 0.9), di = ('FF510-Di01-25x36', 0.9)),
    NikonTurretTopCube2 = dict(ex = ('FF01-340/26-25', 0.75), di = ('400DCLP', 0.92)),
    NikonTurretTopCube3 = dict(ex = ('FF01-562/40-25', 0.93), di = ('FF593-Di03-25x36', 0.93), em = ('FF01-624/40-25', 0.93)),
    NikonTurretTopCube4 = dict(ex = ('FF01-482/35-25', 0.93), di = ('FF506-Di03-25x36', 0.93)),
    NikonTurretTopCube5 = dict(ex = ('FF01-500/24-25', 0.93), di = ('FF520-Di02-25x36', 0.93)),
    NikonTurretTopCube6 = dict(ex = ('FF01-543/22-25', 0.93), di = ('FF562-Di02-25x36', 0.9)),
    NikonTurretBottomCube1 = dict(ex = ('FF02-525/50-25', 0.93), di = ('560DCXR', 0.9)),
    NikonTurretBottomCube2 = dict(ex = ('FF02-460/80-25', 0.9), di = ('FF510-Di01-25x36', 0.9)),
    NikonTurretBottomCube3 = dict(di = ('Mirror', 0.0)),
    NikonTurretBottomCube4 = dict(ex = ('FF01-536/40-25', 0.93), di = ('580DCXR', 0.9)),
    NikonTurretBottomCube5 = dict(ex = ('FF01-542/27-25', 0.93), di = ('580DCXR', 0.9)),
    NikonTurretBottomCube6 = dict(ex = ('FF01-593/40-25', 0.93), di = ('640DCXR', 0.9)),
    NikonIllumination = dict (ex = ('FF01-585/40-25', 0.9)),
    )

"""
Part no F73-492:
HC Dual Line Notch Beamsplitter 500/646

Reflection 488 and 633-638 nm >90% Transmission 420 - 471, 505 - 613
and 653 - 750 nm >90% Dimension 25,2 x 35,6 x 1,1 mm

http://www.ahf.de/art-HC_Dual_Line_Notch_Beamsplitter_500_646;F73-492.html
http://www.ahf.de/spectren/F73-492.txt
"""

confocal_filters = dict(
    ThorlabsWheelPosition1 = dict (em = ('FF01-550/88-25', 0.92)),
    ThorlabsWheelPosition2 = dict (em = ('FF01-725/150-25', 0.9)),
    OpticalTableJoiner = dict(di = ('LM01-503',0.95)),
    OpticalTableSplitter = dict(di = ('FF500/646-Di01-25x36',0.9)),
    AOTFLine1 = dict (ex = ('AOTF:Line1:%(status)s:%(freq)sMHz:%(power)sdBm', 0.96)),
    AOTFLine2 = dict (ex = ('AOTF:Line2:%(status)s:%(freq)sMHz:%(power)sdBm', 0.96)),
    AOTFLine3 = dict (ex = ('AOTF:Line3:%(status)s:%(freq)sMHz:%(power)sdBm', 0.96)),
    AOTFLine4 = dict (ex = ('AOTF:Line4:%(status)s:%(freq)sMHz:%(power)sdBm', 0.96)),
    )

nikon_light_sources = dict (
    NikonArc = dict(man='Prior', type='Xe', model='L200CY', power='200000'),
    NikonArcTr = dict(man='Nikon', type='Other')
    )

confocal_light_sources = dict (
    Laser633 = dict(man='Melles Griot', model='05-LHP-151', type='Other', wl='633', power='5'),
    Laser473 = dict(man='Shanghai Dream Lasers Technology Co. Ltd', model='SDL-473-LN-O1OT', type='Other', wl='473', power='35'),
    ArcTr = dict (man='Olympus', model='TL4', type='Hg', power='30000')
    )

nikon_detectors = dict (
    CameraImperx = dict (man = 'Imperx', type='CCD'),
    CameraAndor = dict (man = 'Andor Technology Ltd', model = 'DV-885K-C00-#VP', type='EMCCD'),
    )

confocal_detectors = dict (
    PhotonCounter = dict (man='Perkin Elmer', model='SPCM-AQRH-13', type='APD'),
    )

nikon_objectives = dict (
    CFI_Plan_Apochromat_VC_60xW_NA_1__x20 = dict(man='Nikon',model='PlanApoVC60x/1.2WI',correction='PlanApo',
                                                 immersion='Water',NA='1.2', mag='60', wd='270'),
    CFI_Super_Plan_Fluor_ELWD_40xC_NA_0__x60 = dict (man='Nikon', model='SuperPlanFluorELWD40x/0.60',correction='PlanFluor',
                                                     immersion='Air',NA='0.6',mag='40'),
    CFI_Super_Plan_Fluor_ELWD_20xC_NA_0__x45 = dict (man='Nikon', model='SuperPlanFluorELWD20x/0.45',correction='PlanFluor',
                                                     immersion='Air',NA='0.45',mag='20')
    )

confocal_objectives = dict (
    UPLSAPO_60xW_NA_1__x20 = dict (man='Olympus',model='UPLSAPO60XW/1.2', correction = 'PlanApo',
                                   immersion='Water',NA='1.2',mag='60',wd='280'),
    UPlanFLN_10x_NA_0__x30 = dict (man='Olympus',model='UPLFLN10X', correction = 'PlanFluor',
                                   immersion='Air',NA='0.3',mag='10',wd='10500')
    )

def create_ome_filter_set(config, channel):
    s = ome.FilterSet (Manufacturer='SysBio', ID='FilterSet:%s' % (channel))
    if channel in ['CameraAndor', 'CameraImperx']:
        top = config['top_turret_cube']
        bottom = config['bottom_turret_cube']
        top_cube = nikon_filters['NikonTurretTopCube%s' % top[3]]
        bottom_cube = nikon_filters['NikonTurretBottomCube%s' % bottom[3]]        

        e = top_cube.get('ex')
        if e is not None:
            s.append(ome.ExcitationFilterRef(ID='Filter:NikonTurretTopCube%s:%s' % (top[3],e[0])))
        e = top_cube.get('di')
        if e is not None:
            s.append(ome.DichroicRef(ID='Dichroic:NikonTurretTopCube%s:%s' % (top[3],e[0])))
        e = top_cube.get('em')
        if e is not None:
            s.append(ome.EmissionFilterRef(ID='Filter:NikonTurretTopCube%s:%s' % (top[3],e[0])))

        e = bottom_cube.get('di')
        if e is not None:
            s.append(ome.EmissionFilterRef(ID='Filter:NikonTurretBottomCube%s:%s' % (bottom[3],e[0])))        
        if channel=='CameraAndor':
            e = bottom_cube.get('ex')
            if e is not None:
                s.append(ome.EmissionFilterRef(ID='Filter:NikonTurretBottomCube%s:%s' % (bottom[3],e[0])))        
        elif channel=='CameraImperx':
            e = bottom_cube.get('em')
            if e is not None:
                s.append(ome.EmissionFilterRef(ID='Filter:NikonTurretBottomCube%s:%s' % (bottom[3],e[0])))        
        return s
    if channel in ['PhotonDetector']:
        pass

aotf_freq_to_wavelength_table ='''450:146.7,454.5:144.4,457.9:142.8,458.5:142.5,465.8:139.1,468:138.4,472.7:136.3,476.5:134.7,482.0:132.5,488:130.1,496.5:127,
          501.7:125.2,511:122,514.5:120.8,520.8:118.8,528.7:116.5,530.9:115.8,543.5:112.3,560.0:107.8,568.2:106,575.3:104.2,578:103.6,
          594:100.1,627:93.6,632.8:92.5,647.1:90.1,654.0:88.9,668.2:86.7,676.4:85.4,694:82.6,700:81.7
       '''

def aotf_freq_to_wavelength(freq):
    if not freq:
        return
    wl_table = []
    fr_table = []
    j = None
    for i,item in enumerate(aotf_freq_to_wavelength_table.split(',')):
        item = item.strip()
        wl,fr = map (float,item.split (':'))
        wl_table.append (wl)
        fr_table.append (fr)
        if i>=1 and fr_table[-1] <= freq and freq <= fr_table[-2]:
            d = (freq - fr_table[-1])/(fr_table[-2] - fr_table[-1])
            wavelength = wl_table[-1] + d * (wl_table[-2] - wl_table[-1])
            return int(wavelength)
    #from matplotlib import pyplot as plt
    #plt.plot(fr_table, wl_table); plt.show ()
    raise ValueError ('argument %s is out of range [%s, %s]' % (freq, fr_table[-1], fr_table[0]))

def get_aotf_filter_name(name, config):
    line = int(name[9])
    freq = config.get('AOTF_Line%sAcousticFrequency' % (line))
    if freq is None:
        return
    power = config['AOTF_Line%sAcousticPower' % (line)]
    enabled = int(config['AOTF_Line%sEnable' % (line)])
    status = 'ON' if enabled else 'OFF'
    return name % locals()

def get_aotf_filter_wavelength(name, config):
    line = int(name[9])
    freq = float(config['AOTF_Line%sAcousticFrequency' % (line)])
    return aotf_freq_to_wavelength(freq)    

def create_ome_filter(part_no, wheel, transmittance, config=None):
    if part_no=='Mirror':
        man = 'None'
        t = 'Other'
        r = ome.TransmittanceRange(Transmittance=str(transmittance))
    elif part_no.startswith ('AOTF'):
        man = 'AA Sa Opto-Electronic Division'
        t = 'BandPass'
        wl = get_aotf_filter_wavelength(part_no, config)
        if wl:
            r = ome.TransmittanceRange(CutIn=str(wl), Transmittance=str(transmittance))
        else:
            r = ome.TransmittanceRange(Transmittance=str(transmittance))
    elif part_no.startswith ('LM'):
        man = 'Semrock'
        t = 'Dichroic'
        r = ome.TransmittanceRange(Transmittance=str(transmittance))
    elif part_no.startswith ('FF'):
        man = 'Semrock'
        if 'Di' in part_no:
            t = 'Dichroic'
            wl = part_no[2:5]
            r = ome.TransmittanceRange (CutIn=wl, Transmittance=str(transmittance))
        else:
            wl_bw,sz = part_no.split('-')[1:]
            wl, bw = map (int, wl_bw.split('/'))
            t = 'BandPass'
            r = ome.TransmittanceRange (CutIn=str (wl-bw//2), CutOut=str (wl+bw//2), Transmittance=str(transmittance))
    else:
        man = 'Chroma'
        if part_no[0]=='Z':
            t = 'Dichroic'
            r = ome.TransmittanceRange(Transmittance=str(transmittance))
        else:
            wl = int(part_no[:3])
            if part_no.endswith ('XR'):
                t = 'Dichroic'
            elif part_no.endswith ('LP'):
                t = 'LongPass'
            else:
                raise NotImplementedError (`part_no`)
            r = ome.TransmittanceRange(CutIn=str(wl), Transmittance=str(transmittance))

    return ome.Filter (r,
                       Manufacturer=man,
                       Model=part_no, 
                       #SerialNumber='', 
                       #LotNumber='',
                       Type = t,
                       FilterWheel=wheel,
                       ID = 'Filter:%s:%s' % (wheel,part_no),
                       )

def create_ome_dichroic(part_no, wheel, transmittance):
    if part_no.startswith ('FF') or part_no.startswith ('LM'):
        man = 'Semrock'
        assert 'Di' in part_no or part_no[:2]=='LM',`part_no`
    else:
        man = 'Chroma'
        assert part_no.endswith ('XR') or 'DC' in part_no or part_no[0]=='Z', `part_no`
    return ome.Dichroic(
        Manufacturer=man,
        Model=part_no, 
        #SerialNumber='', 
        #LotNumber='',
        ID = 'Dichroic:%s:%s' % (wheel,part_no),
        )

class OMEConfiguration(OMEBase):

    _detectors = ['Imperx', 'Andor', 'Confocal'] # prefixes of tiff file names

    prefix = 'ome'
    def __init__(self, path):
        OMEBase.__init__(self)
        dir_path = self.dir_path = os.path.dirname(path)
        self.config_path = path
        self.info_path = os.path.join(dir_path,'info.txt')
        
        dpath = os.path.dirname(os.path.abspath(path))
        self.file_prefix += '_'.join([p for p in dpath[len(self.cwd)+1:].split(os.sep) if p]) + '_'

        config = self.config = pathinfo.get_tag_from_configuration(self.config_path, None) or {}
        info = self.info = pathinfo.get_tag_from_configuration(self.info_path, None) or {}

        data = self.data = {}

        for d in self._detectors:
            d_path = os.path.join(dir_path, '%s_index.txt' % (d))
            if not os.path.isfile (d_path): # mari-ism support
                d_path = os.path.join(dir_path, d, '%s_index.txt' % (d))
            if os.path.isfile (d_path):
                d_index = {}
                for index, line in enumerate(open (d_path).readlines ()):
                    t, fn = line.strip().split()
                    t = float(t)
                    d_index[t, index] = os.path.join(os.path.dirname(d_path), fn)
                data[d] = d_index

        import getpass
        current_user = getpass.getuser()
        if current_user not in people:
            import pwd
            fname, lname = pwd.getpwnam(current_user).pw_gecos.split(',')[0].split()
            people[current_user] = dict(
                first_name = fname, last_name = lname,
                email='%s@sysbio.ioc.ee' % (current_user),
                institution='Laboratory of Systems Biology',
                groups = ['SysBio'],
                )
        self.current_user = current_user

        if 0:
            for k in sorted (config):
                print '%s: %r' % (k,config[k])
    
    def get_AcquiredDate (self):
        year, month, day, hour, minute, second, ns = map(int, self.config['m_time'].split())
        year += 1900
        month += 1
        return '%s-%02d-%02dT%02d:%02d:%s.%s' % (year,month, day, hour, minute, second, ns)

    def iter_Experiment(self, func):
        mode = self.config['main_protocol_mode']
        descr = self.info.get('DESCRIPTION')
        descr = 'Experiment type: %s\n%s' % (mode, descr or '')
        e = func(Type="Other", ID='Experiment:%s' % (mode))
        if descr is not None:
            e.append(ome.Description(descr))
        yield e
        return

    def iter_Image(self, func):
        sys.stdout.write('iter_Image: reading image data from TIFF files\n')
        for detector in self.data:
            sys.stdout.write('  detector: %s\n' % (detector))
            d_index = self.data[detector]

            # write the content of tiff files to a single raw files
            f,fn,dtype = None, None, None
            time_set = set()
            mn, mx = None, None
            mnz = float(self.config['PROTOCOL_Z_STACKER_Minimum'])
            mxz = float(self.config['PROTOCOL_Z_STACKER_Maximum'])
            nz = int(self.config['PROTOCOL_Z_STACKER_NumberOfFrames'])
            if nz > 1:
                dz = (mxz-mnz)/(nz-1)
            else:
                dz = 0
            plane_l = []
            ti = -1

            exptime = '0'
            if detector=='Confocal':
                exptime = float(self.config['CONFOCAL_PixelAcqusitionTime']) * 1e-6
            elif detector=='Andor':
                exptime = self.config['CAMERA_ANDOR_ExposureTime']
            elif detector=='Imperx':
                for line in  self.config['CAMERA_IMPERX_HardwareInformation'].split('\n'):
                    if line.startswith ('Exposure time:'):
                        v,u = line[14:].lstrip().split()
                        v = v.strip (); u = u.strip ()
                        if u=='usec': exptime = float(v)*1e-6
                        elif u=='msec': exptime = float(v)*1e-3
                        elif u=='sec': exptime = float(v)
                        else:
                            raise NotImplementedError (`v,u,line`)
            else:
                raise NotImplementedError(`detector`)

            for t, index in sorted(d_index):
                if t not in time_set:
                    time_set.add(t)
                    ti += 1
                    zi = 0
                else:
                    zi += 1
                z = mnz + dz * zi
                d = dict(DeltaT=str(t), TheT=str(ti), TheZ = str(zi), PositionZ=str(z), TheC='0', ExposureTime=str(exptime))
                plane_l.append(d)

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

                sys.stdout.write('\r  copying TIFF image data to RAW file: %5s%% done' % (int(100.0*(index+1)/len(d_index))))
                sys.stdout.flush()

            if f is None:
                continue
            f.close ()
            shape = tuple (shape)

            xsz = shape[2]
            ysz = shape[1]
            tsz = len(time_set)
            zsz = shape[0] // tsz
            order = 'XYZTC'
            sys.stdout.write("\n  RAW file contains %sx%sx%sx%sx%s [%s] array, dtype=%s, MIN/MAX=%s/%s\n" \
                                 % (xsz, ysz,zsz,tsz,1,order, dtype, mn,mx))
            assert zsz*tsz==shape[0],`zsz,tsz,shape`

            tif_filename = '%s%s.ome.tif' % (self.file_prefix, detector)
            sys.stdout.write("  creating memmap image for OME-TIF file %r..." % (tif_filename))
            sys.stdout.flush()
            mmap = numpy.memmap(fn, dtype=dtype, mode='r', shape=shape)
            tif_image = TIFFimage(mmap)
            atexit.register(os.remove, fn)
            tif_uuid = self._mk_uuid()
            self.tif_images[detector, tif_filename, tif_uuid] = tif_image
            sys.stdout.write (' done\n')
            sys.stdout.flush()


            pixels_d = {}
            channel_d = dict(SamplesPerPixel='1')
            lpath_l = []
            #channel_d todo: ExcitationWavelength, EmissionWavelength, Fluor, NDFilter, PockelCellSetting, Color 
            if detector in ['Confocal']:
                objective = ome.ObjectiveSettings(ID='Objective:%s' % (self.config['olympus_optics_objective']))
                instrument_id = 'Instrument:Airy'
                pixels_d['PhysicalSizeX'] = str(self.config['CONFOCAL_PixelSizeX'])
                pixels_d['PhysicalSizeY'] = str(self.config['CONFOCAL_PixelSizeY'])
                pixels_d['TimeIncrement'] = str(self.config['CONFOCAL_TimeBetweenFrames'])
                channel_d['Name'] = 'Confocal'
                channel_d['IlluminationType'] = 'Epifluorescence'
                channel_d['PinholeSize'] = '180'
                # todo: FluorescenceCorrelationSpectroscopy
                channel_d['AcquisitionMode'] = 'LaserScanningConfocalMicroscopy'

                for i in range (1,5):
                    d1 = 'AOTFLine%s' % i
                    ft = confocal_filters.get(d1)
                    if ft is None:
                        continue
                    fn = ft['ex'][0]
                    fn = get_aotf_filter_name (fn, self.config)
                    if 'OFF' in fn:
                        continue
                    lpath_l.append(ome.ExcitationFilterRef(ID='Filter:%s:%s' % (d1,fn)))
                fn = confocal_filters['OpticalTableSplitter']['di'][0]
                lpath_l.append(ome.DichroicRef(ID='Dichroic:OpticalTableSplitter:%s' % (fn)))
                d1 = 'ThorlabsWheelPosition%s' % (self.config['thorlabs_filter_wheel_position'][3])
                ft = confocal_filters.get(d1)
                if ft is not None:
                    fn = ft['em'][0]
                    lpath_l.append(ome.EmissionFilterRef (ID='Filter:%s:%s' % (d1,fn)))
            elif detector in ['Andor', 'Imperx']:
                objective = ome.ObjectiveSettings(ID='Objective:%s' % (self.config['optics_objective']))
                instrument_id = 'Instrument:Suga'
                channel_d['Name'] = '%s camera' % (detector)
                channel_d['AcquisitionMode'] = 'WideField'
                pixels_d['PhysicalSizeX'] = pixels_d['PhysicalSizeY'] = str(self.config['CAMERA_%s_PixelSize' % (detector.upper ())])
                tbf = float(self.config['CAMERA_%s_TimeBetweenFrames' % (detector.upper ())])
                d1 = 'NikonTurretTopCube%s' % (self.config['top_turret_cube'][3])
                d2 = 'NikonTurretBottomCube%s' % (self.config['bottom_turret_cube'][3])
                top_cube = nikon_filters[d1]
                bottom_cube = nikon_filters[d2]
                if detector=='Andor':
                    channel_d['IlluminationType'] = 'Epifluorescence'
                    if self.config['CAMERA_ANDOR_FrameTransferMode']=='1':
                        m = re.search(r'Kinetic cycle time:\s*(?P<time>\d+[.]\d*)\s*sec', self.config['CAMERA_ANDOR_HardwareInformation'], re.M)
                        pixels_d['TimeIncrement'] = str(m.group('time'))
                    else:
                        pixels_d['TimeIncrement'] = str(tbf)
                    if 'ex' in top_cube:
                        fn = top_cube['ex'][0]
                        lpath_l.append(ome.ExcitationFilterRef(ID='Filter:%s:%s' % (d1,fn)))
                    if 'di' in top_cube:
                        fn = top_cube['di'][0]
                        lpath_l.append(ome.DichroicRef(ID='Dichroic:%s:%s' % (d1,fn)))
                    if 'em' in top_cube:
                        fn = top_cube['em'][0]
                        lpath_l.append(ome.EmissionFilterRef(ID='Filter:%s:%s' % (d1,fn)))
                    if 'ex' in bottom_cube:
                        fn = bottom_cube['ex'][0]
                        lpath_l.append(ome.EmissionFilterRef(ID='Filter:%s:%s' % (d2,fn)))
                else:
                    #m = re.search(r'Exposure time:\s*(?P<time>\d+[.]\d*)\s*msec', self.config['CAMERA_IMPERX_HardwareInformation'], re.M)
                    #exp_time = float (m.group ('time'))
                    if self.config['main_protocol_mode'].startswith('MyocyteMechanicsFluorescence'):
                        tbf = float(self.config['PROTOCOL_MYOCYTE_MECHANICS_TimeBetweenSavedFrames'])
                    pixels_d['TimeIncrement'] = str(tbf)
                    channel_d['IlluminationType'] = 'Transmitted'
                    if self.config['optics_transmission_light_filter']!='Empty':
                        fn = nikon_filters['NikonIllumination']['ex'][0]
                        lpath_l.append(ome.ExcitationFilterRef(ID='Filter:NikonIllumination:%s' % (fn)))
                    if 'di' in top_cube:
                        fn = top_cube['di'][0]
                        lpath_l.append(ome.EmissionFilterRef(ID='Filter:%s:%s' % (d1, fn)))
                    if 'em' in top_cube:
                        fn = top_cube['em'][0]
                        lpath_l.append(ome.EmissionFilterRef(ID='Filter:%s:%s' % (d1,fn)))
                    if 'em' in bottom_cube:
                        fn = bottom_cube['em'][0]
                        lpath_l.append(ome.EmissionFilterRef(ID='Filter:%s:%s' % (d2,fn)))
                    if 'di' in bottom_cube:
                        fn = bottom_cube['di'][0]
                        lpath_l.append(ome.EmissionFilterRef(ID='Filter:%s:%s' % (d2,fn)))
            else:
                raise NotImplementedError (`detector`)

            if zsz>1:
                pixels_d['PhysicalSizeZ'] = str(dz)
            channel = ome.Channel(ID='Channel:%s' % (detector),
                                  **channel_d)

            lpath = ome.LightPath(*lpath_l)
            channel.append (lpath)

            #todo attributes: 
            #todo elements: BIN:BinData, MetadataOnly, SA:AnnotationRef
            tiffdata = ome.TiffData(ome.UUID (tif_uuid, FileName=tif_filename))
            pixels = ome.Pixels(channel,
                                tiffdata, 
                                DimensionOrder=order, ID='Pixels:%s' % (detector),
                                SizeX = str(xsz), SizeY = str(ysz), SizeZ = str(zsz), SizeT=str(tsz), SizeC = str(1),
                                Type = self.dtype2PixelIType (dtype),
                                **pixels_d
                                )
            for d in plane_l:
                pixels.append(ome.Plane(**d))
            #todo attributes: Name
            #todo elements: Description, ExperimentRef, DatasetRef ,
            #               ImagingEnvironment, StageLabel, ROIRef, MicrobeamManipulationRef, AnnotationRef

            image = ome.Image (ome.AcquiredDate (self.get_AcquiredDate()),
                               ome.ExperimenterRef(ID='Experimenter:%s' % (self.current_user)),
                               ome.GroupRef(ID='Group:SysBio'),
                               ome.InstrumentRef(ID=instrument_id),
                               objective,
                               pixels, 
                               ID='Image:%s' % (detector))

            if 0:
                image.append(sa.AnnotationRef (ID='Annotation:configuration.txt'))
            yield image
        return

    def iter_Experimenter(self, func):
        for id, d in people.items():
            d1 = dict (Email=d['email'], Institution=d['institution'], UserName=id)
            if 'first_name' in d:
                d1['FirstName'] = d['first_name']
                d1['LastName'] = d['last_name']
                d1['DisplayName'] = '%(first_name)s %(last_name)s' % d
            else:
                d1['DisplayName'] = id
            e = func (ID='Experimenter:%s' % (id), **d1)
            for g in d['groups']:
                e.append (ome.GroupRef(ID='Group:%s' % g))
            yield e

    def iter_Group(self, func):
        for id, d in groups.items():
            e = func(ID='Group:%s' % (id), Name=d['name'])
            if 'descr' in d:
                e.append (ome.Description (d['descr']))
            if 'leader_id' in d:
                e.append (ome.Leader(ID='Experimenter:%s' % (d['leader_id'])))
            if 'contact_id' in d:
                e.append (ome.Contact(ID='Experimenter:%s' % (d['contact_id'])))
            yield e

    def iter_Instrument(self, func):
        yield  self.get_Instrument_Suga(func)
        yield  self.get_Instrument_Airy(func)

    def get_Instrument_Suga(self, func):
        e = func(ID='Instrument:Suga')
        e.append(ome.Microscope(Manufacturer='Nikon', 
                                Model='Eclipse Ti-U', 
                                #SerialNumber='', 
                                #LotNumber='',
                                Type = 'Inverted'))

        for n, d in nikon_light_sources.items():
            if 'Laser' in n:
                s = ome.Laser(Type=d['type'], Wavelength=d['wl'])
            elif 'Arc' in n:
                s = ome.Arc (Type=d['type'])
            else:
                raise NotImplementedError (`n,d`)
            d1 = {}
            for k1,k2 in dict(man='Manufacturer', model='Model', power='Power').items ():
                if k1 in d: d1[k2] = d[k1]
            e.append(ome.LightSource(s,ID='LightSource:%s' % n, **d1))

        for n, d in nikon_detectors.items():
            d1 = {}
            for k1,k2 in dict(man='Manufacturer', model='Model', type='Type').items ():
                if k1 in d: d1[k2] = d[k1]
            e.append(ome.Detector(ID='Detector:%s' % n, **d1))

        for n, d in nikon_objectives.items():
            d1 = {}
            for k1,k2 in dict(man='Manufacturer', model='Model', correction='Correction', immersion='Immersion', NA='LensNA',
                              mag = 'NominalMagnification', wd='WorkingDistance').items ():
                if k1 in d: d1[k2] = d[k1]
            e.append(ome.Objective(ID='Objective:%s' % n, **d1))

        for cube in sorted (nikon_filters):
            for f in ['ex', 'di', 'em']:
                cube_data = nikon_filters[cube]
                d = cube_data.get(f)
                if d is None: continue
                fn, tr = d
                e.append(create_ome_filter(fn, cube, str(tr)))

        for cube in sorted (nikon_filters):
            cube_data = nikon_filters[cube]
            d = cube_data.get('di')
            if d is None: continue
            fn, tr = d
            if fn=='Mirror': continue
            e.append(create_ome_dichroic(fn, cube, str(tr)))

        return e

    def get_Instrument_Airy(self, func):

        e = func(ID='Instrument:Airy')
        e.append (ome.Microscope(Manufacturer='Olympus', 
                                 Model='IX71', 
                                 #SerialNumber='', 
                                 #LotNumber='',
                                 Type = 'Inverted'))

        for n, d in confocal_light_sources.items():
            if 'Laser' in n:
                s = ome.Laser(Type=d['type'], Wavelength=d['wl'])
            elif 'Arc' in n:
                s = ome.Arc (Type=d['type'])
            else:
                raise NotImplementedError (`n,d`)
            d1 = {}
            for k1,k2 in dict(man='Manufacturer', model='Model', power='Power').items ():
                if k1 in d: d1[k2] = d[k1]
            e.append(ome.LightSource(s, ID='LightSource:%s' % n, **d1))

        for n, d in confocal_detectors.items():
            d1 = {}
            for k1,k2 in dict(man='Manufacturer', model='Model', type='Type').items ():
                if k1 in d: d1[k2] = d[k1]
            e.append(ome.Detector(ID='Detector:%s' % n, **d1))

        for n, d in confocal_objectives.items():
            d1 = {}
            for k1,k2 in dict(man='Manufacturer', model='Model', correction='Correction', immersion='Immersion', NA='LensNA',
                              mag = 'NominalMagnification', wd='WorkingDistance').items ():
                if k1 in d: d1[k2] = d[k1]
            e.append(ome.Objective(ID='Objective:%s' % n, **d1))

        for detector in ['PhotonCounter']:
            f = create_ome_filter_set(self.config, detector)
            if f is not None:
                e.append(f)

        for f in ['ex', 'di', 'em']:
            for cube in sorted (confocal_filters):
                cube_data = confocal_filters[cube]
                d = cube_data.get(f)
                if d is None: continue
                fn, tr = d
                if fn.startswith ('AOTF'):
                    fn = get_aotf_filter_name (fn, self.config)
                    if fn is not None:
                        e.append(create_ome_filter(fn, cube, str(tr), self.config))
                else:
                    e.append(create_ome_filter(fn, cube, str(tr), self.config))

        for cube in sorted (confocal_filters):
            cube_data = confocal_filters[cube]
            d = cube_data.get('di')
            if d is None: continue
            fn, tr = d
            e.append(create_ome_dichroic(fn, cube, str(tr)))        
        
        return e

    def iter_SA_StructuredAnnotations(self, func):

        e = func()

        if 0:
            filesize = os.stat (self.config_path).st_size
            filecontent = open(self.config_path).read ()
            assert filesize==len(filecontent), `filesize, len(filecontent)`
            base64content = base64.encodestring(filecontent)
            f = sa.FileAnnotation (
                bf.BinaryFile(
                    bf.BinData (base64content, Compression='none', BigEndian='false', Length=str(len (base64content))),
                    FileName=os.path.basename(self.config_path), Size=str (filesize), MIMEType='text/plain'),
                ID='Annotation:configuration.txt')
            e.append (f)
        yield e
        return

        if 0:
            c = sa.CommentAnnotation(sa.Description('info.txt'),ID='Annotation:info.txt')
            c.append(sa.Value(open(self.info_path).read()))
            e.append(c)
            
        c = sa.CommentAnnotation(ID='Annotation:filepath-configuration.txt')
        c.append(sa.Value(self.config_path))
        e.append(c)

        #c = sa.CommentAnnotation(ID='Annotation:test')
        #c.append(sa.Value('This is test comment'))
        #e.append(c)
        
        l = sa.ListAnnotation(sa.Description('configuration.txt'), #sa.Value ('Content of %s' % (self.config_path)), 
                                 ID='Annotation:configuration.txt')# Namespace='configuration.txt')

        l1 = []

        for key in sorted(self.config):
            value = self.config[key]
            l1.append ('%s=%r' % (key, value))
            continue
            if isinstance (value, str):
                value = value.strip ()
                if not value: continue
                ID='Annotation:string-%s' % (key)
                e.append(sa.CommentAnnotation(sa.Value(value), ID=ID))#, Namespace='configuration.txt'))
            elif 1:
                ID='Annotation:%s' % (key)
                e.append(sa.CommentAnnotation(sa.Value('%s=%s' % (key, value)), ID=ID, Namespace='configuration.txt'))
            elif isinstance (value, bool):
                ID='Annotation:bool-%s' % (key)
                e.append(sa.BooleanAnnotation(sa.Value(str(value).lower()), ID=ID, Namespace='configuration.txt'))
            elif isinstance (value, int):
                ID='Annotation:int-%s' % (key)
                e.append(sa.LongAnnotation(sa.Value(str(value)), ID=ID, Namespace='configuration.txt'))
            elif isinstance (value, float):
                ID='Annotation:float-%s' % (key)
                e.append(sa.DoubleAnnotation(sa.Value(str(value)), ID=ID, Namespace='configuration.txt'))
            else:
                raise NotImplementedError (`value, type(value)`)

            l.append(sa.AnnotationRef(ID=ID))

        c = sa.CommentAnnotation(ID='Annotation:configuration.txt')
        c.append(sa.Value('\n'.join(l1)))
        e.append(c)

        #e.append(l)


        yield e

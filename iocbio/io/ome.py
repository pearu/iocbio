""" Utilities for OME-XML converters.

References
----------
http://www.openmicroscopy.org/Schemas/

Module content
--------------
"""

__all__ = ['ome', 'ATTR', 'validate_xml', 'OMEBase']

import os
import sys
from uuid import uuid1 as uuid
from lxml import etree
from lxml.builder import ElementMaker

namespace_map=dict(bf = "http://www.openmicroscopy.org/Schemas/BinaryFile/2010-06",
                   ome = "http://www.openmicroscopy.org/Schemas/OME/2010-06",
                   xsi = "http://www.w3.org/2001/XMLSchema-instance",
                   sa = "http://www.openmicroscopy.org/Schemas/SA/2010-06",
                   spw = "http://www.openmicroscopy.org/Schemas/SPW/2010-06")

# create element makers: bf, ome, xsi
ome = ElementMaker (namespace = namespace_map['ome'], nsmap = namespace_map)
sa = ElementMaker (namespace = namespace_map['sa'], nsmap = namespace_map)
spw = ElementMaker (namespace = namespace_map['spw'], nsmap = namespace_map)

def ATTR(namespace, name, value):
    return {'{%s}%s' % (namespace_map[namespace], name): value}

def validate_xml(xml):
    ome_xsd = os.path.join(os.path.dirname(__file__), 'ome.xsd')
    if os.path.isfile (ome_xsd):
        f = open (ome_xsd)
    else:
        import urllib2
        ome_xsd = os.path.join(namespace_map['ome'],'ome.xsd')
        f = urllib2.urlopen(ome_xsd)
    sys.stdout.write('Validating XML content against %r...' % (ome_xsd))
    xmlschema_doc = etree.parse(f)
    xmlschema = etree.XMLSchema(xmlschema_doc)
    if isinstance (xml, basestring):
        xml = etree.parse(xml)
    result = xmlschema.validate(xml)
    if not result:
        sys.stdout.write('FAILED:\n')
        for error in xmlschema.error_log:
            s = str (error)
            for k,v in namespace_map.items():
                s = s.replace ('{%s}' % v, '%s:' % k)
            print(s)
        sys.stdout.write('-----\n')
    else:
        sys.stdout.write('SUCCESS!\n')
    return result

class ElementBase:

    def __init__ (self, parent, root):
        self.parent = parent
        self.root = root
        
        n = self.__class__.__name__
        iter_mth = getattr(parent, 'iter_%s' % (n), None)
        nsn = 'ome'
        nm = n
        if '_' in n:
            nsn, nm = n.split('_',1)
            nsn = nsn.lower()
        ns = eval(nsn)    
        ome_el = getattr (ns, n, None)
        if iter_mth is not None:
            for element in iter_mth(ome_el):
                root.append(element)
        else:
            print 'NotImplemented: %s.iter_%s(<%s.%s callable>)' % (parent.__class__.__name__, n, nsn, nm)

class Project(ElementBase): pass
class Dataset(ElementBase): pass
class SPW_Plate(ElementBase): pass
class SPW_Screen(ElementBase): pass
class Experiment(ElementBase): pass
class Experimenter(ElementBase): pass
class Group(ElementBase): pass
class Instrument(ElementBase): pass
class Image(ElementBase): pass
class SA_StructuredAnnotations(ElementBase): pass

class OMEBase:
    """ Base class for OME-XML writers.
    """

    _subelement_classes = [Project, Dataset, Experiment, SPW_Plate, SPW_Screen, 
                   Experimenter, Group, Instrument, Image, SA_StructuredAnnotations]

    def __init__(self):
        self.tif_images = {}

    def process(self, validate=True):
        template_xml = self.make_xml()
        s = None
        for (fn, uuid), tif_image in self.tif_images.items ():
            xml= ome.OME(ATTR('xsi','schemaLocation',"%s %s/ome.xsd" % ((namespace_map['ome'],)*2)),
                          UUID = uuid)
            map(xml.append,template_xml)
            if s is None:
                validate_xml(xml)
            s = etree.tostring(xml, pretty_print = True, xml_declaration=True)
            tif_image.description = s
            tif_image.write_file(fn, compression='lzw')
            # todo: load image and compare it to tif_image.data, especially when compressing
        return s

    def _mk_uuid(self):
        return 'urn:uuid:%s' % (uuid())

    def make_xml(self):
        self.temp_uuid = self._mk_uuid()
        xml = ome.OME(ATTR('xsi','schemaLocation',"%s %s/ome.xsd" % ((namespace_map['ome'],)*2)),
                       UUID = self.temp_uuid)
        for element_cls in self._subelement_classes:
            element_cls(self, xml) # element_cls should append elements to root
        return xml
    
    def get_AcquiredDate(self):
        return None

    @staticmethod
    def dtype2PixelIType(dtype):
        return dict (int8='int8',int16='int16',int32='int32',
                     uint8='uint8',uint16='uint16',uint32='uint32',
                     complex128='double-complex', complex64='complex',
                     float64='double', float32='float',
                     ).get(dtype.name, dtype.name)

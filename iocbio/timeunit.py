"""Time units.

Provides number like classes to manipulate time quantity with various
units (Seconds, Minutes, and Hours).

Examples
--------

::

  >>> from iocbio.timeunit import *
  >>> Seconds(10)
  Seconds(10.0)
  >>> print Seconds(10)+Minutes(1)
  70.0sec
  >>> print Hours(1) - Minutes(20)
  0.666667h
  >>> print Hours(1) - Minutes(30)
  0.5h
  >>> print Seconds(Hours(1) - Minutes(30))
  1800.0sec

The base class, ``Time``, of ``Seconds``, ``Minutes``, and ``Hours`` classes can be used
to construct time instances from strings and to convert between different
units. For example::

  >>> Time(20, unit='min')
  Minutes(20.0)
  >>> Time('60 seconds', unit='min')
  Minutes(1.0)
  >>> Time(Hours(1), unit='min')
  Minutes(60.0)

"""

# Author: Pearu Peterson
# Created: September 2010

from __future__ import division
import re

__all__ = ['Time', 'Seconds', 'Minutes', 'Hours']

re_time = re.compile (r'(?P<data>[+-]?(\d+(\s*[.]\s*\d+)?|[.]\s*\d+))\s*(?P<unit>(s|sec|seconds|second|m|min|minutes|minute|h|hour|hours|))')

class Time (object):
    """ Base class to time quantity with units.

    See also
    --------
    iocbio.timeunit
    """
    def __new__(cls, data, unit=None):
        """ Construct a Time instance.

        Parameters
        ----------
        data : {float, int, long, Seconds, Minutes, Hours}
        unit : {None, 's', 'm', 'h',...}
        """
        if isinstance(data, basestring):
            m = re_time.match(data.lower())
            if m is None:
                raise ValueError (`cls, data`)
            data = float(m.group('data'))
            u = m.group ('unit')
            if u: 
                data = Time(data, unit = u)

        if cls is Time:
            if isinstance(data, Time) and unit is None:
                return data
            objcls = dict(s=Seconds, sec=Seconds, secs = Seconds, seconds = Seconds, second=Seconds,
                          m=Minutes, min=Minutes, minutes=Minutes, minute=Minutes,
                          h=Hours, hour = Hours, hours = Hours).get(str(unit).lower(), None)
            if objcls is None:
                raise NotImplementedError(`cls, unit, data`)
            if isinstance(data, Time):
                data = objcls(data).data
            if isinstance (data, (float, int, long)):
                obj = object.__new__ (objcls)
                obj.data = objcls.round(data)
                return obj
            raise NotImplementedError (`cls, objcls, data`)

        assert unit is None,`unit`
        if isinstance(data, Time):
            if data.__class__ is cls:
                return data # assume that Time is immutable
            else:
                to_other_name = 'to_%s' % (cls.__name__)
                to_other = getattr (data, to_other_name, None)
                if to_other_name is None:
                    raise NotImplementedError ('%s class needs %s member' % (data.__class__.__name__, to_other_name))
                data = to_other * data.data
        if isinstance (data, (float, int, long)):
            obj = object.__new__(cls)
            obj.data = cls.round(data)
            return obj
        raise NotImplementedError(`cls, data`)

    def __str__ (self):
        return '%s%s' % (self.data, self.unit_label)

    def __repr__ (self):
        return '%s(%r)' % (self.__class__.__name__, self.data)

    def __abs__ (self):
        return self.__class__(abs(self.data))

    def __pos__ (self): return self
    def __neg__ (self): return self.__class__(-self.data)

    def __add__ (self, other):
        if isinstance (other, Time):
            return self.__class__ (self.data + self.__class__(other).data)
        return NotImplemented
    __radd__ = __add__

    def __sub__ (self, other):
        return self + (-other)

    def __rsub__ (self, other):
        return other + (-self)

    def __mul__ (self, other):
        if isinstance (other, (float, int, long)):
            return self.__class__ (self.data * other)
        return NotImplemented

    __rmul__ = __mul__

    def __floordiv__(self, other):
        if isinstance (other, (float, int, long)):
            return self.__class__(self.data // other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance (other, (float, int, long)):
            return self.__class__(self.data / other)
        return NotImplemented

    __div__ = __truediv__

    def __float__(self):
        return float(self.data)

    def __int__ (self):
        return int(self.data)

    def __hash__ (self):
        return hash((self.__class__.__name__, self.data))

    def __eq__ (self, other):
        if isinstance (other, Time):
            return self.data == self.__class__(other).data
        return NotImplemented
    def __ne__ (self, other):
        if isinstance (other, Time):
            return self.data != self.__class__(other).data
        return NotImplemented
    def __lt__ (self, other):
        if isinstance (other, Time):
            return self.data < self.__class__(other).data
        return NotImplemented
    def __le__ (self, other):
        if isinstance (other, Time):
            return self.data <= self.__class__(other).data
        return NotImplemented
    def __gt__ (self, other):
        if isinstance (other, Time):
            return self.data > self.__class__(other).data
        return NotImplemented
    def __ge__ (self, other):
        if isinstance (other, Time):
            return self.data >= self.__class__(other).data
        return NotImplemented

class Seconds(Time):
    """ Time in seconds.

    See also
    --------
    iocbio.timeunit
    """

    unit_label = 'sec'
    to_Minutes = 1/60
    to_Hours = to_Minutes/60

    @classmethod
    def round(cls, data):
        return round (data, 2)

class Minutes(Time):
    """ Time in minutes.

    See also
    --------
    iocbio.timeunit
    """

    unit_label = 'min'
    to_Seconds = 60
    to_Hours = 1/60

    @classmethod
    def round(cls, data):
        return round (data, 4)

class Hours(Time):
    """ Time in hours.

    See also
    --------
    iocbio.timeunit
    """

    unit_label = 'h'
    to_Minutes = 60
    to_Seconds = to_Minutes * 60

    @classmethod
    def round(cls, data):
        return round (data, 6)

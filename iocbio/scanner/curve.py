
from __future__ import division
import time

VERBOSE = 1

from ioc.microscope.utils import argument_string

def time_it(func):
    """ Print how long calling given function took.
    """
    def new_func(*args, **kws):
        t = time.time()
        r = func (*args, **kws)
        dt = time.time() - t
        if VERBOSE:
            print 'Calling %s(%s) -> %s took %s seconds' % \
                (func.__name__, ', '.join(map(argument_string, args)), argument_string(r), dt)
        return r
    return new_func

import numpy

def closest(t1, t2, t0):
    if t1 > t2:
        return closest(t2, t1, t0)
    if t0 <= t1:
        return t1
    if t2 <= t0:
        return t2
    if t0 - t1 < t2 - t0:
        return t1
    return t2

class Point:

    _T = None

    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __abs__ (self):
        return numpy.hypot(self.x, self.y)
    def __neg__ (self):
        return Point(-self.x, -self.y)
    def __add__ (self, other):
        if isinstance (other, Point):
            return Point(self.x + other.x, self.y + other.y)
        if isinstance (other, (int, float, long)):
            return Point(self.x + other, self.y + other)
        return NotImplemented
    __radd__ = __add__
    def __sub__ (self, other):
        return self + (-other)
    def __rsub__ (self, other):
        return (-self) + other
    def __mul__ (self, other):
        return Point(self.x * other, self.y * other)
    __rmul__ = __mul__

    def __truediv__ (self, other):
        return self * (1.0/other)

    def __repr__(self):
        return 'Point(%r, %r)' % (self.x, self.y)

    @property
    def T(self):
        r = self._T
        if r is None:
            r = self._T = Point (self.y, self.x)
            r._T = self
        return r

    def dot (self, other):
        return self.x * other.x + self.y * other.y

    def cross(self, other):
        return self.x * other.y - self.y * other.x

    def distance(self, other):
        d = self - other
        return numpy.sqrt(d.dot(d))

    def normal(self):
        l = abs(self)
        return Point(self.x/l, self.y/l)

class Curve:

    _T = None

    def __init__(self):
        self.points = []

    @property
    def T (self):
        r = self._T
        if r is None:
            r = self._T = self.__class__(*[p.T for p in self.points])
            r._T = self
        return r

    def __repr__ (self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join (map (repr, self.points)))

    @property
    def points_x (self):
        return (p.x for p in self.points)
    @property
    def points_y (self):
        return (p.y for p in self.points)

    def at_x (self, x):
        return self.T.at_y(x)

    def diff(self, t):
        delta = 1e-10
        return (self(t + delta) - self(t - delta)) / (2*delta)

    def tangent(self, t):
        delta = 1e-10
        return Line(self(t), self (t + delta))

    def __sub__ (self, other):
        return self + (-other)
    def __truediv__ (self, other):
        return self * (1/other)

    _max_x = None
    _min_x = None
    _max_y = None
    _min_y = None

    @property
    def min_x(self):
        """ Returns min(Curve(t).x, t=0..1) """
        r = self._min_x
        if r is None:
            self._min_x = r = self.T.min_y
        return r

    @property
    def min_y(self):
        """ Returns min(Curve(t).y, t=0..1) """
        r = self._min_y
        if r is None:
            self._min_y = r = min(self[0], self[1])
        return r

    @property
    def max_x(self):
        """ Returns max(Curve(t).x, t=0..1) """
        r = self._max_x
        if r is None:
            self._max_x = r = self.T.max_y
        return r

    @property
    def max_y(self):
        """ Returns max(Curve(t).y, t=0..1) """
        r = self._max_y
        if r is None:
            self._max_y = r = max(self[0], self[1])
        return r

    @property
    def max_y_arg(self):
        """ Returns t_max such that Curve.max_y == Curve(t_max).y. """
        return max((self(0).y,0), (self(1).y,1))[1]

    @property
    def min_y_arg(self):
        """ Returns t_min such that Curve.min_y == Curve(t_min).y. """
        return min((self(0).y,0), (self(1).y,1))[1]

    @property
    def max_x_arg(self):
        """ Returns t_max such that Curve.max_x == Curve(t_max).x. """
        return self.T.max_y_arg

    @property
    def min_x_arg(self):
        """ Returns t_min such that Curve.min_x == Curve(t_min).x. """
        return self.T.min_y_arg

class Line(Curve):
    
    def __init__(self, origin, end=None, direction=None, beta_exp=1):
        self.origin = origin
        if direction is None:
            if end is not None:
                direction = end - origin
            else:
                raise NotImplementedError (`origin, end, direction`)
        elif end is None:
            if direction is not None:
                end = origin + direction
            else:
                raise NotImplementedError (`origin, end, direction`)
        self.end = end
        self.direction = direction
        self.points = [origin, end]
        self.beta_exp = beta_exp

    def set_beta_exp (self, beta_exp):
        self.beta_exp = beta_exp
        return self

    def __repr__ (self):
        return 'Line(%r, %r)' % (self.origin, self.end)

    def __getitem__(self, t):
        return self.origin.y + t * self.direction.y

    def __call__(self, t):
        return self.origin + t * self.direction

    def __neg__(self):
        return Line(self.origin, direction=-self.direction)

    def __add__ (self, other):
        if isinstance(other, Point):
            return Line(self.origin + other, direction=self.direction)
        if isinstance(other, Line):
            return Line(self.origin + other.origin, direction=self.direction + other.direction)
        return NotImplemented

    def __mul__ (self, other):
        return Line(self.origin, direction=self.direction * other)
    __rmul__ = __mul__

    def intersection (self, other):
        c = other.origin - self.origin
        d = self.direction.cross(other.direction)
        if d==0:
            return None
        return self.origin + self.direction * c.cross(other.direction) / d

    def at(self, point):
        r = (point - self.origin)
        return r.dot (self.direction) / self.direction.dot(self.direction)

    def projection (self, other):
        if isinstance(other, Point):
            return self(self.at(other))
        raise NotImplementedError(`type (other)`)

    def middle (self, other):
        origin = self.intersection(other)
        end = ((self(1) + other(0))/2 + origin)/2
        return Line (origin, end = end)

    def length (self):
        return self(0).distance(self(1))

    def joint(self, other, alpha=1.0, beta=1.0, method='cubic'):
        # method in 'quadric', 'cubic'
        if isinstance(other, Line):
            P1 = self(1)
            P2 = other(0)

            if method=='cubic':
                return Cubic.to_minimal_dot_length(self.direction, other.direction, self(1), other(0),
                                                   self.beta_exp)

                d = alpha * abs(P1 - P2)
                left_t = (d/self.length()) * beta
                right_t = (d/other.length()) / beta
                                
                P0 = self(1-left_t)
                P3 = other(right_t)
                
                PC3 = P1 - P0
                PC4 = P1
                PC2 = 8*P2 - 10*P1 + 4*P0 - 2*P3
                PC1 = -10*P2 + 8*P1 - 2*P0 + 4*P3
            
                return Cubic(PC1, PC2, PC3, PC4)

            elif method=='quadric':
                dP1 = self.direction.normal()
                dP2 = other.direction.normal()
                d = dP1.dot (dP2)
                P12 = P2 - P1
                dP12 = dP1.dot(P12)
                dP21 = dP2.dot(P12)
                alpha = (dP12 - dP21*d)*2.0/(1-d*d)
                beta = (-dP12*d + dP21)*2.0/(1-d*d)
                curve = Quadric(dP1*alpha, dP2*beta, P1)
                print alpha, beta
                print curve
                print curve (0), P1
                print curve (1), P2
                return curve
            else:
                raise NotImplementedError(`method`)

        return NotImplemented

    def at_x(self, x):
        return (x - self.origin.x) / self.direction.x

    def at_y (self, y):
        return (y - self.origin.y) / self.direction.y

    def stretch_start_y(self, y):
        return Line(self(self.at_y(self[0] - y)), self.end)
    def stretch_start_x(self, x):
        return Line(self(self.at_x(self(0).x - x)), self.end)

    def stretch_end_x(self, x):
        return Line(self(0), self(self.at_x(self(1).x + x)))

    def stretch_end_to_y(self, y):
        return Line(self(0), self(self.at_y(y)))

class Quadric(Curve):

    def __init__ (self, PC1, PC2, PC3):
        self.points = (PC1, PC2, PC3)

    def __getitem__ (self, t):
        PC1, PC2, PC3 = self.points
        return PC1.y * (t-t*t/2) + PC2.y * (t*t/2) + PC3.y

    def __call__ (self, t):
        PC1, PC2, PC3 = self.points
        return PC1 * (t-t*t/2) + PC2 * (t*t/2.0) + PC3

    def at_y(self, y):
        PC1, PC2, PC3 = self.points_y
        if PC1==PC2:
            return (y-PC3)/PC1
        t2 = 1/(PC2-PC1)
        t13 = numpy.sqrt(PC1*PC1+2.0*(PC2-PC1)*(y-PC3))
        s1 = t2*(-PC1+t13)
        s2 = t2*(-PC1-t13)

        y0 = self[0]
        y1 = self[1]
        t0 = (y-y0)/(y1-y0)
        return closest (s1, s2, t0)

class Cubic (Curve):
    
    def __init__ (self, dP1, dP2, P1, P2):
        self.points = (dP1, dP2, P1, P2)

    def __getitem__(self, t):
        dP1, dP2, P1, P2 = self.points_y
        return (dP1*(t-1) + dP2*t)*t*(t-1) + (P2 - P1)*(-2*t+3)*t*t + P1

    def __call__(self, t):
        dP1, dP2, P1, P2 = self.points
        return (dP1*(t-1) + dP2*t)*t*(t-1) + (P2 - P1)*(-2*t+3)*t*t + P1

    def at_y(self, y):
        dP1, dP2, P1, P2 = self.points_y
        from solve_cubic_ext import solve_cubic
        y0 = self[0]
        y1 = self[1]
        t0 = (y-y0)/(y1-y0)
        tc = solve_cubic(t0, y, *self.points_y)
        return tc

    @classmethod
    def to_minimal_length(cls, dP1, dP2, P1, P2):
        tP1 = dP1.normal()
        tP2 = dP2.normal()
        P21 = P2 - P1
        dP12 = tP1.dot (tP2)
        det = 16-dP12*dP12
        dtP1 = tP1.dot (P21)
        dtP2 = tP2.dot (P21)
        alpha = (4*dtP1 + dP12*dtP2)/det
        beta = (dP12*dtP1 + 4*dtP2)/det
        if alpha<0 or beta<0:
            pass
            #print 'Warning: Cubic will not be minimal length (forcing alpha=%r, beta=%r to be positive)' % (alpha, beta)
        return Cubic (tP1*abs(alpha), tP2*abs(beta), P1, P2)

    @classmethod
    def to_minimal_dot_length(cls, dP1, dP2, P1, P2, beta_exp=1):
        tP1 = dP1.normal()
        tP2 = dP2.normal()
        P21 = P2 - P1
        dP12 = tP1.dot (tP2)
        det = 4-dP12*dP12
        dtP1 = tP1.dot (P21)
        dtP2 = tP2.dot (P21)
        alpha = 3*(2*dtP1 - dP12*dtP2)/det
        beta = 3*(-dP12*dtP1 + 2*dtP2)/det
        if alpha<0 or beta<0:
            pass
            #print 'Warning: Cubic will not be minimal length (forcing alpha=%r, beta=%r to be positive)' % (alpha, beta)
        return Cubic (tP1*abs(alpha), tP2*abs(beta)**beta_exp, P1, P2)

class Cubic2 (Curve):

    def __init__ (self, PC1, PC2, PC3, PC4):
        self.points = (PC1, PC2, PC3, PC4)

    def __getitem__(self, t):
        if t==0:
            return self.points[-1].y
        (PC1, PC2, PC3, PC4) = self.points
        return PC1.y*t**3/6 + PC2.y*t*t*(1-t/3)/2+PC3.y*t+PC4.y

    def __call__(self, t):
        if t==0:
            return self.points[-1]
        (PC1, PC2, PC3, PC4) = self.points
        return PC1*t**3/6 + PC2*t*t*(1-t/3)/2+PC3*t+PC4

    def _solve_cubic(self, x, x1, x2, x3, x4):
        sqrt = numpy.lib.scimath.sqrt
        pow = numpy.lib.scimath.power
        if x2==x1:
            if x1==0:
                return (-x4+x)/x3,
            t1 = 1/x1
            t2 = x3*x3
            t8 = sqrt(t2-2.0*x1*x4+2.0*x1*x)
            return t1*(-x3+t8), t1*(-x3-t8)
        t2 = 1/(-x2+x1)
        t3 = x2*x2
        t4 = x3*t3
        t6 = x3*x2
        t9 = x4*t3
        t11 = x2*x4
        t14 = x1*x1
        t15 = x4*t14
        t19 = x2*x
        t24 = t3*x2
        t27 = x*x
        t30 = x4*x4
        t36 = x3*x1
        t44 = x3*x3
        t45 = t44*x3
        t69 = -18.0*t15*x+9.0*t14*t27+9.0*t14*t30-18.0*x1*t27*x2+18.0*t36*t19 \
            -18.0*x1*t30*x2-18.0*t36*t11+8.0*x1*t45+36.0*x1*x4*t19+9.0*t30*t3-3.0*t44*t3 \
            -8.0*t45*x2-18.0*t4*x-6.0*x*t24+6.0*x4*t24+18.0*t4*x4-18.0*t9*x+9.0*t27*t3
        t70 = sqrt(t69)
        t73 = -3.0*t4+3.0*t6*x1-3.0*t9+6.0*t11*x1-3.0*t15+3.0*x*t3-6.0*t19*x1+3.0 \
            *x*t14-t24-t70*x2+t70*x1
        t74 = pow(t73,1/3)
        t75 = t2*t74
        t81 = (-2.0*t6+2.0*t36-t3)*t2/t74
        t82 = x2*t2
        t85 = t75/2.0
        t86 = t81/2.0
        t89 = t75+t81
        s1 = -t85+t86-t82
        s2 = sqrt(-0.75)*t89
        return (t75-t81-t82), (s1+s2), (s1-s2)

    def _solve_quadratic (self, x, x1, x2, x3, x4):
        sqrt = numpy.lib.scimath.sqrt
        if x2==x1:
            if x1==0:
                return ()
            return (x-x3)/x1,
        t2 = 1/(-x2+x1)
        t3 = x2*x2
        t13 = sqrt(t3-2.0*x2*x+2.0*x3*x2+2.0*x1*x-2.0*x3*x1)
        return t2*(-x2+t13), t2*(-x2-t13)

    def _solve_linear(self, x, x1, x2, x3, x4):
        if x2==x1:
            return ()
        return (-x2+x)/(-x2+x1),

    def at_y(self, y):
        v0 = self[0]
        v1 = self[1]
        if v1==v0:
            return 0.5

        t0 = (y-v0)/(v1-v0)
        if 1:
            from solve_cubic_ext import solve_cubic
            tc = solve_cubic(t0, y, *self.points_y)
            #assert abs(self[tc]-y)<1e-10,`abs(self[tc]-y),tc,self[tc],y,t0,self[t0]`
        else:
            r = numpy.real_if_close(self._solve_cubic(y, *self.points_y), tol=1e5)
            tc = r[abs(r-t0).argmin()]
            if isinstance (tc, complex):
                tc = tc.real
            elif isinstance (tc, float):
                pass
            else:
                raise NotImplementedError (`tc, type(tc)`)

            #assert abs(tc-t)<1e-10,`t0,tc,t,self[tc], self[t],y`

        return tc


    _max_y = None
    _min_y = None

    @property
    def max_y(self):
        r = self._max_y
        if r is None:
            t_lst = [0, 1]
            for t in numpy.real_if_close(self._solve_quadratic(0, *self.points_y), tol=1e5):
                if 0<t and t<1:
                    t_lst.append(t)
            r = max([self[t] for t in t_lst])
            self._max_y = r
        return r

    @property
    def min_y(self):
        r = self._min_y
        if r is None:
            t_lst = [0, 1]
            for t in numpy.real_if_close(self._solve_quadratic(0, *self.points_y), tol=1e5):
                if 0<t and t<1:
                    t_lst.append(t)
            r = self._min_y = min([self[t] for t in t_lst])
        return r

    @property
    def max_y_arg(self):
        t_lst = [0, 1]
        for t in numpy.real_if_close(self._solve_quadratic(0, *self.points_y), tol=1e5):
            if 0<t and t<1:
                t_lst.append(t)
        return max([(self[t], t) for t in t_lst])[1]

    @property
    def min_y_arg(self):
        t_lst = [0, 1]
        for t in numpy.real_if_close(self._solve_quadratic(0, *self.points_y), tol=1e5):
            if 0<t and t<1:
                t_lst.append(t)
        return min([(self[t], t) for t in t_lst])[1]

    def length(self):
        raise NotImplementedError('length of a cubic spline')

class Curves:
    
    def __init__ (self, *curves):
        self.curves = list(curves)

    def __len__ (self):
        return len(self.curves)

    def __repr__ (self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join (map (repr, self.curves)))

    @property
    def T(self):
        return self.__class__(*[p.T for p in self.curves])        

    def __getitem__(self, t):
        n = len (self.curves)
        if isinstance(t, (numpy.ndarray, list, tuple)):
            return numpy.array([self[tt] for tt in t])
        k = min(max(0,int(t)),n-1)
        return self.curves[k][t - k]

    def __call__(self, t):
        n = len (self.curves)
        if isinstance(t, (numpy.ndarray, list, tuple)):
            return numpy.array([self(tt) for tt in t])
        k = min(max(0,int(t)),n-1)
        return self.curves[k](t - k)

    def graph(self, x):
        if isinstance(x, (numpy.ndarray, list, tuple)):
            return numpy.array([self.graph(xx) for xx in x])

        for curve in self.curves:
            if curve.min_x <= x and x <= curve.max_x:
                t = curve.at_x(x)
                r = curve[t]
                #assert abs(curve(t).x-x)<1e-11,` abs(curve(t).x-x),t,x`
                assert not numpy.isnan (r),`t,r,x`
                return r

    @property
    def min_x(self):
        return min([curve.min_x for curve in self.curves])
    @property
    def min_y(self):
        return min([curve.min_y for curve in self.curves])
    @property
    def max_x(self):
        return max([curve.max_x for curve in self.curves])
    @property
    def max_y(self):
        return max([curve.max_y for curve in self.curves])

    def add_line(self, line, alpha=1.0, beta=1.0):
        if self.curves:
            prev_line = self.curves[-1]
            if prev_line (1) != line (1):
                curve = prev_line.joint(line, alpha=alpha, beta=beta)
                self.curves.append(curve)
        self.curves.append(line)

if __name__ == '__main__':

    line1 = Line(Point(-1,0), direction=Point(1,0.1))
    line2 = Line(Point(2,2), direction=Point(1,0.1))
    line3 = Line(Point(4,1), direction=Point(1,0))
    
    line12 = Line(line1(1), direction=line2(0) - line1(1))

    #print line1.intersection (line2)
    #j = line1.joint(line2)#, 0.1, 4)
    #print j.max_y, j.min_y
    #print j.max_x, j.min_x

    import matplotlib.pyplot as plt



    c = Curves()
    c.add_line (line1)

    c.add_line (line2, alpha=1, beta=2)
    #c.add_line (line3)
    
    #c = Curves(line1, j, line2)

    t = numpy.arange(0,len (c),0.01)
    plt.plot(c.T[t], c[t])

    x = numpy.arange(c.min_x,c.max_x,0.05)
    plt.plot(x, c.graph(x), '-')

    t = numpy.arange(0,1.01,0.01)
    #plt.plot (line1.T[t], line1[t], line2.T[t], line2[t],
    #          line12.T[t], line12[t])

    plt.show ()

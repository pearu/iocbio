
from __future__ import division

import numpy

from configuration import image_area_width, image_area_height, \
    camera_area_width, camera_area_height, mirror_max_angle, \
    mirror_x_offset, mirror_y_offset, mirror_max_user_angle, \
    get_mirror_settling_time, mirror_max_overshoot

pi = numpy.pi

sin_deg = lambda x: numpy.sin(x * numpy.pi/180) # argument is in degrees
cos_deg = lambda x: numpy.cos(x * numpy.pi/180)

def xya_xy(alpha, x, y):
    ca = cos_deg(alpha)
    sa = sin_deg(alpha)
    return ca * x + sa*y, -sa*x + ca*y

def xy_xya(alpha, xa, ya): 
    return xya_xy(-alpha, xz, ya)

def xya_pq(p, q, 
           WC=camera_area_width, W=image_area_width,
           HC=camera_area_height, H=image_area_height):
    return (-1/2 + (p-1)/WC)*W, (-1/2 + (q-1)/HC)*H

def pq_xya(xa,ya,
           WC=camera_area_width, W=image_area_width,
           HC=camera_area_height, H=image_area_height):
    return 1+(xa/W+1/2)*WC, 1+(ya/H+1/2)*HC

def xy_pq(alpha, p, q):
    return xy_xya(alpha, *xya_pq (p, q))

class MirrorDriver:

    i_offset = None
    j_offset = None
    t_offset = None

    def set_params(self, 
                   i_offset = None, j_offset = None,
                   t_offset = None,
                   flyback_alpha = 1.0, flyback_beta = 1.0):
        if i_offset is not None:
            self.i_offset = i_offset
        elif self.i_offset is None:
            self.i_offset = 0

        if j_offset is not None:
            self.j_offset = j_offset
        elif self.j_offset is None:
            self.j_offset = 0

        if t_offset is not None:
            self.t_offset = t_offset
        elif self.t_offset is None:
            self.t_offset = 0

        self.flyback_alpha = flyback_alpha
        self.flyback_beta = flyback_beta

        self.compute_curve()

    def get_params (self):
        return self.i_offset, self.j_offset, self.t_offset #, self.flyback_alpha, self.flyback_beta

    def __init__ (self, image_size=(4,3), T=3, 
                  flyback_ratio = 1.2, flystart_ratio = 0, alpha=0, roi = None,
                  clock_rate = None, params={},
                  verbose = True):
        """
        T - samples per pixel
        roi - (P0, Q0, P1, Q1)
        """
        self.clock_rate = clock_rate
        W = self.W = image_area_width
        H = self.H = image_area_height
        WC = self.WC = camera_area_width
        HC = self.HC = camera_area_height
        self.Vmax = mirror_max_user_angle
        self.Vx0, self.Vy0 = mirror_x_offset, mirror_y_offset

        self.sin = lambda x: numpy.sin(x * numpy.pi/180) # argument is in degrees
        self.cos = lambda x: numpy.cos(x * numpy.pi/180)
        self.arcsin = lambda y: 180/pi*numpy.arcsin(y)
        self.pi = pi

        if roi is None:
            roi = (1,1,WC,HC)
        P0, Q0, P1, Q1 = self.P0, self.Q0, self.P1, self.Q1 = roi

        assert 1<=P0<=WC,`P0, WC`
        assert 1<=P1<=WC,`P1, WC`
        assert 1<=Q0<=HC,`Q0, HC`
        assert 1<=Q1<=HC,`Q1, HC`
        assert P0<=P1,`P0, P1`
        assert Q0<=Q1,`Q0, Q1`

        if Q1 == Q0:
            self.Q1 = Q1 = Q0 + 0.001

        if P1 == P0:
            self.P1 = P1 = P0 + 0.001

        (N, M) = (self.N, self.M) = image_size
        self.T = T
        assert 1<=N,`N`
        assert 1<=M,`M`
        assert 1<=T,`T`
        
        assert 0<=flyback_ratio,`flyback_ratio`

        self.alpha = alpha

        io = self.io = self.i_v(0, 0)
        jo = self.jo = self.j_v(0, 0)
        i_min = self.i_v(-mirror_max_angle*self.cos(alpha), -mirror_max_angle*self.sin (alpha))
        j_min = self.j_v(-mirror_max_angle*self.cos(alpha), -mirror_max_angle*self.sin (alpha))
        #i_min = self.i_min = self.i_xa(-W/2*(1+self.sin (alpha)**2))

        i_min = max(i_min, -20)
        j_min = max(j_min, -20)

        self.i_min = i_min
        self.j_min = j_min

        self.i_user_min = max(self.i_v(self.vx_i(0) - mirror_max_overshoot, 0), i_min)

        K, K0, K1, S = self.init_curve(flyback_ratio, flystart_ratio)

        self.K = K
        self.K0 = K0
        self.K1 = K1
        self.S = S

        self.set_params(**params)

        if not verbose:
            return

        print 'Scan area: %s x %s mm' % (1e3*W, 1e3*H)
        print 'Camera size: %s x %s au' % (WC, HC)
        print 'Camera orientation: %s deg' % (alpha)
        print
        print 'ROI: P_ll=(%s, %s) au, P_ur=(%s, %s) au' % tuple (roi)
        print 'ROI: P_ll=(%.3f, %.3f) mm, P_ur=(%.3f, %.3f) mm' % (1e3*self.x_pq (P0, Q0), 1e3*self.y_pq (P0, Q0),
                                             1e3*self.x_pq (P1, Q1), 1e3*self.y_pq (P1, Q1))
        print
        print 'Image size: %s x %s px, %s x %s au, %s x %s mm' \
            % (N, M, P1-P0+1, Q1-Q0+1, 1e3*(self.xa_p (P1+1) - self.xa_p(P0)), 1e3*(self.ya_q(Q1+1) - self.ya_q(Q0)))
        print 'Pixel size: %.2f x %.2f um' % (1e6*self.get_pixel_size_xa(), 1e6*self.get_pixel_size_ya())
        print 'Pixel step (line scan): %.2f x %.2f um' % (1e6*self.get_pixel_step_x(), 1e6*self.get_pixel_step_y())
        print 'Pixel step (flyback): %.2f x %.2f um' % (1e6*self.get_pixel_step_x1(), 1e6*self.get_pixel_step_y1())
        print

        print 'Pixels from origin to image start: %s' % (self.K0)
        print 'Pixels from image end to origin: %s' % (self.K1)
        print 'Pixels from line end to next line start: %s' % (K)
        print 'Ticks per pixel: %s' % (T)
        print 'Total number of pixels:', self.get_pixels()
        print 'Total number of ticks:', self.get_ticks()
        print 'Origin: (%.3f, %.3f) px' % (io, jo)
        print
        print 'Flyback/line ratio: %s/%s=%.3f actual, %.3f given' % (K, N, K/N, flyback_ratio)
        print
        return
        vx = self.vx_t_array ()[K0*T:-K1*T-1]
        vy = self.vy_t_array ()[K0*T:-K1*T-1]
        print 'Flyback extremes: min=%.2fpx, %.3fmm; max=%.2fpx, %.3fmm' \
            % (imin, 1e3*self.xa_t((K0+N)*T+tmin), imax, 1e3*self.xa_t((K0+N)*T+tmax))
        print 'Mirror X extremes: min=%.2fV; max=%.2fV, max diff/on line=%.4fV/%.4fV' \
            % (vx.min(), vx.max(), abs(numpy.diff (vx)).max(), abs(numpy.diff(vx[:T*N])).max())
        print 'Mirror Y extremes: min=%.2fV; max=%.2fV, max diff/on line=%.4fV/%.4fV' \
            % (vy.min(), vy.max(), abs(numpy.diff (vy)).max (), abs(numpy.diff(vy[:T*N])).max())
        print

    def get_key(self):
        return self.__class__.__name__,\
            self.N, self.M, self.K, self.K0, self.K1, self.S, self.alpha,\
            (self.P0, self.Q0, self.P1, self.Q1), self.get_params()

    def __hash__ (self):
        return hash(self.get_key())

    def init_curve (self, flyback_ratio, flystart_ratio):
        """
        Returns K, K0, K1.
        """
        alpha = self.alpha
        N, T = self.N, self.T
        K = self.K = max(int(N*flyback_ratio), 1)

        io, jo = self.io, self.jo
        i_min, j_min = self.i_min, self.j_min

        # bounding conditions
        tmax = numpy.arccos(N/(K+N))*K*T/2/pi
        tmin = K * T - tmax
        imin = self.i_t_flyback(tmin)
        imax = self.i_t_flyback(tmax)

        # starting parameters
        i0 = io
        i1 = -1/2+1/2/T
        K0 = self.K0 = 4*int(((1-pi/2)*(i1-i0) - min(imin, i0)*pi)) or 1
        assert K0>0, `K0, imin, i1,i0`

        K0 = self.K0 = K + int(abs(2*i_min))

        # ending parameters
        i0 = N-1/2+1/2/T
        i1 = io
        K1 = self.K1 = int(-( pi*i1 - pi*max(imax,i1) + (1+pi/2)* (i0 - i1))) or 1

        assert K1>0, `K1, imax, i1, i0`        
        
        S = 0

        return K, K0, K1, S

    def compute_curve(self):
        pass

    def get_pixels (self):
        return self.K0 + self.N*self.M + self.K*(self.M-1) + self.K1

    def get_ticks(self):
        return self.get_pixels()*self.T + 1

    def get_pixel_size_xa(self):
        return self.xa_i(1) - self.xa_i(0)

    def get_pixel_size_ya(self):
        return self.ya_j(1) - self.ya_j(0)

    def get_pixel_step_x(self):
        return self.x_ij(1, 0) - self.x_ij(0, 0)

    def get_pixel_step_y(self):
        return self.y_ij(1, 0) - self.y_ij(0, 0)

    def get_pixel_step_x1(self):
        return self.x_ij(0, 1) - self.x_ij(0, 0)

    def get_pixel_step_y1(self):
        return self.y_ij(0, 1) - self.y_ij(0, 0)

    def i_t_start (self, t):
        assert isinstance (t, numpy.ndarray)
        r = 0*t
        T, K0, K = self.T, self.K0, self.K
        t0 = 0
        tb = T*K
        tm = T*(K0+self.i_min +1/2 -1/2/T)
        #tm = 2*T*K0/3
        t1 = T*K0
        indices_b = numpy.where (t < tb)

        indices_m = numpy.where ((tb <=t) * (t < tm))
        indices_1 = numpy.where ((tm <=t) * (t <= t1))


        i0 = self.io
        i1 = self.i_min + self.i_offset

        t_b = t[indices_b]
        r[indices_b] = i0 + (i1-i0)/(tb-t0)*t_b - (i1-i0)/2/pi*numpy.sin(2*pi/(tb-t0)*t_b)

        r[indices_m] = i1
        

        i0 = self.i_min
        i1 = -1/2+1/2/T + self.i_offset
        t_1 = t[indices_1]
        r[indices_1] = -(K0-i1) + t_1/T

        #r[indices_1] = i0 + (i1-i0)/t1*t_1 + ((i1-i0)*T - t1)/pi/T*numpy.sin(pi/(t1)*t_1)

        return r

        t1 = K0*T
        i0 = self.io
        i1 = -1/2+1/2/T + self.i_offset
        r = i0 + (i1-i0)/t1*t + ((i1-i0)*T - t1)/pi/T*numpy.sin(pi/t1*t)
        return r


        t12 = t1*0.5
        if isinstance (t, (int, float)):
            if 0<t<t12:
                r = (r - i0) * numpy.sin(t/t12*numpy.pi/2) + i0
        elif isinstance (t, numpy.ndarray):
            indices = numpy.where (t < t12)
            r[indices] = (r[indices] - i0) * numpy.sin(t[indices]/t12*numpy.pi/2) + i0
        else:
            raise NotImplementedError (`type (t)`)
        return r

    def di_t_start (self, t):
        T, K0 = self.T, self.K0
        t1 = K0*T
        i0 = self.io
        i1 = -1/2+1/2/T
        return (i1-i0)/t1 + ((i1-i0)/K0 - 1)/T*numpy.cos(pi/t1*t)
        

    def j_t_start (self, t):
        T, K0, K = self.T, self.K0, self.K
        j0 = self.jo
        j1 = 0 + self.j_offset
        tb = K * T
        t1 = K0 * T
        
        indices_b = numpy.where (t < tb)
        indices_1 = numpy.where ((tb <=t) * (t <= t1))        

        r = 0*t + j1
        r[indices_b] = j0 + (j1-j0)/tb*t - (j1-j0)/2/pi*numpy.sin (2*pi/tb*t)
        return r
        return j0 + (j1-j0)/t1*t - (j1-j0)/2/pi*numpy.sin (2*pi/t1*t)

    def i_t_line (self, t):
        T, N = self.T, self.N
        i0 = -1/2 + 1/2/T + self.i_offset
        i1 = N - 1/2 + 1/2/T + self.i_offset
        t1 = N * T
        return i0 + (i1 - i0)/t1 * t

    def j_t_line (self, t, j):
        return j + 0*t + self.j_offset

    def i_t_flyback(self, t):
        T, N, K = self.T, self.N, self.K
        i0 = N - 1/2 + 1/2/T + self.i_offset
        i1 = -1/2 + 1/2/T + self.i_offset
        t1 = K * T
        return i0 + (i1 - i0)/t1*t + (t1 - (i1 - i0)*T)/T/2/pi*numpy.sin(2*pi/t1*t)

    def j_t_flyback(self, t, j):
        T, K = self.T, self.K
        j0 = j + self.j_offset
        j1 = j + 1 + self.j_offset
        t1 = K * T
        return j0 + (j1-j0)/t1*t - (j1-j0)/2/pi*numpy.sin (2*pi/t1*t)

    def i_t_end(self, t):
        T, N, K1 = self.T, self.N, self.K1
        t1 = K1 * T
        i0 = N - 1/2 + 1/2/T + self.i_offset
        i1 = self.io
        return i1 + (i0 - i1)/t1*(t1 - t) + ((i0 - i1)*T + t1)/pi/T*numpy.sin(pi/t1*(t1-t))

    def j_t_end (self, t):
        T, M, K1 = self.T, self.M, self.K1
        j0 = M - 1 + self.j_offset
        j1 = self.jo
        t1 = K1 * T
        return j0 + (j1-j0)/t1*t - (j1-j0)/2/pi*numpy.sin (2*pi/t1*t)


    def i_t(self, t): 
        if t < 0:
            # normally one should not start up here
            return self.io
        T, N, M, K, K0, K1 = self.T, self.N, self.M, self.K, self.K0, self.K1
        if t < K0 * T:
            return self.i_t_start(t)
        t -= K0 * T

        if t < N * T:
            return self.i_t_line(t)
        t -= N * T

        for j in xrange(1, M):
            if t < K * T:
                return self.i_t_flyback(t)
            t -= K * T

            if t < N * T:
                return self.i_t_line(t)
            t -= N * T

        if t < K1 * T:
            return self.i_t_end(t)

        # normally one should end up here only once
        return self.io

    def j_t(self, t):
        if t < 0:
            # normally one should not start up here
            return self.jo
        T, N, M, K, K0, K1 = self.T, self.N, self.M, self.K, self.K0, self.K1
        if t < K0 * T:
            return self.j_t_start(t)
        t -= K0 * T

        if t < N * T:
            return self.j_t_line(t, 0)
        t -= N * T

        for j in xrange(1, M):
            if t < K * T:
                return self.j_t_flyback(t, j - 1)
            t -= K * T

            if t < N * T:
                return self.j_t_line(t, j)
            t -= N * T

        if t < K1 * T:
            return self.j_t_end(t)

        # normally one should end up here only once
        return self.jo

    def i_t_iter(self): # 5x faster than i_t(t) for 100x100@3 image
        T, N, M, K, K0, K1 = self.T, self.N, self.M, self.K, self.K0, self.K1
        for t in range (K0 * T):
            yield self.i_t_start(t)
        for j in xrange(M):
            if j:
                for t in range (K * T):
                    yield self.i_t_flyback(t)
            for t in range (N * T):
                yield self.i_t_line(t)
        for t in range (K1 * T):
            yield self.i_t_end(t)
        yield self.io
        return

    def i_t_array(self): # 500x faster than i_t_iter() for 100x100@3 image
        T, N, M, K, K0, K1 = self.T, self.N, self.M, self.K, self.K0, self.K1
        result = numpy.zeros ((self.get_ticks(),), float)

        t_K0 = numpy.arange(K0*T, dtype=float)
        t_K = numpy.arange(K*T, dtype=float)
        t_N = numpy.arange(N*T, dtype=float)
        t_K1 = numpy.arange(K1*T, dtype=float)

        i_start = self.i_t_start(t_K0)
        i_line = self.i_t_line(t_N)
        i_flyback = self.i_t_flyback(t_K)
        i_end = self.i_t_end(t_K1)

        start_index = 0

        end_index = K0*T
        result[start_index:end_index] = i_start
        start_index = end_index

        for j in xrange(M):
            if j:
                end_index += K * T
                result[start_index:end_index] = i_flyback
                start_index = end_index

            end_index += N * T
            result[start_index:end_index] = i_line
            start_index = end_index

        end_index += K1 * T
        result[start_index:end_index] = i_end
        start_index = end_index

        result[start_index] = self.io

        return result

    def j_t_array(self):
        T, N, M, K, K0, K1 = self.T, self.N, self.M, self.K, self.K0, self.K1
        result = numpy.zeros ((self.get_ticks(),), float)

        t_K0 = numpy.arange(K0*T, dtype=float)
        t_K = numpy.arange(K*T, dtype=float)
        t_N = numpy.arange(N*T, dtype=float)
        t_K1 = numpy.arange(K1*T, dtype=float)

        j_start = self.j_t_start(t_K0)
        j_flyback = self.j_t_flyback(t_K, -1)
        j_end = self.j_t_end(t_K1)

        start_index = 0

        end_index = K0*T
        result[start_index:end_index] = j_start
        start_index = end_index

        for j in xrange(M):
            if j:
                end_index += K * T
                result[start_index:end_index] = j + j_flyback
                start_index = end_index

            end_index += N * T
            result[start_index:end_index] = self.j_t_line(t_N, j)
            start_index = end_index

        end_index += K1 * T
        result[start_index:end_index] = j_end
        start_index = end_index

        result[start_index] = self.jo

        return result

    def i_p(self, p): 
        return eval ('-1/2 + (p - P0)/(-P0 + P1)*N', self.__dict__, dict (p=p))
    def j_q(self, q): 
        return eval ('-1/2 + (q - Q0)*M/(Q1 - Q0)', self.__dict__, dict (q=q))

    def i_xa(self, xa): return self.i_p(self.p_xa(xa))
    def j_ya(self, ya): 
        return self.j_q(self.q_ya(ya))

    def i_xy(self, x, y): return self.i_xa(self.xa_xy(x, y))
    def j_xy(self, x, y): 
        return self.j_ya(self.ya_xy(x, y))

    def i_v(self, vx, vy): return self.i_xy(self.x_v(vx), self.y_v(vy))
    def j_v(self, vx, vy): 
        return self.j_xy(self.x_v(vx), self.y_v(vy))

    @classmethod
    def compute_pixel_size_x(cls, P0, P1, N):
        W = image_area_width
        WC = camera_area_width
        p0 = P0 + (1/2)*(P1-P0)/N
        p1 = P0 + (1+1/2)*(P1-P0)/N
        x0 = (-1/2 + (p0 - 1)/WC)*W
        x1 = (-1/2 + (p1 - 1)/WC)*W
        # x1 - x0 = (p1 - p0)/WC*W = (P1 - P0)/N/WC*W
        return x1 - x0

    @classmethod
    def compute_image_width(cls, P0, P1, pixel_size):
        W = image_area_width
        WC = camera_area_width
        return max(1, int((P1 - P0)/WC*W/pixel_size))

    def p_i(self, i): return eval ('P0 + (i + 1/2)*(-P0 + P1)/N', self.__dict__, dict (i=i))
    def q_j(self, j): return eval ('Q0 + (j + 1/2)*(-Q0 + Q1)/M', self.__dict__, dict (j=j))

    def p_xa(self, x): return eval ('(x/W + 1/2)*WC + 1', self.__dict__, dict (x=x))
    def q_ya(self, y): return eval ('(y/H + 1/2)*HC + 1', self.__dict__, dict (y=y))

    def p_xy(self, x, y): return self.p_xa(self.xa_xy(x, y))
    def q_xy(self, x, y): return self.q_ya(self.ya_xy(x, y))

    def p_v(self, vx, vy): return self.p_xy(self.x_v(vx), self.y_v(vy))
    def q_v(self, vx, vy): return self.q_xy(self.x_v(vx), self.y_v(vy))
    
    def p_t(self, t): return self.p_i(self.i_t(t))
    def q_t(self, t): return self.q_j(self.j_t(t))

    def xa_i(self, i): return self.xa_p(self.p_i(i))
    def ya_j(self, j): return self.ya_q(self.q_j(j))


    def xa_p(self, p): return eval ('(-1/2 + (p - 1)/WC)*W', self.__dict__, dict (p=p))
    def ya_q(self, q): return eval ('(-1/2 + (q - 1)/HC)*H', self.__dict__, dict (q=q))

    def xa_v(self, vx, vy): return self.xa_xy(self.x_v(vx), self.y_v(vy))
    def ya_v(self, vx, vy): return self.ya_xy(self.x_v(vx), self.y_v(vy))

    def xa_xy(self, x, y): return eval ('cos(alpha) * x + sin(alpha) * y', self.__dict__, dict (x=x, y=y))
    def ya_xy(self, x, y): return eval ('-sin(alpha) * x + cos(alpha) * y', self.__dict__, dict (x=x, y=y))

    def xa_t(self, t): return self.xa_i(self.i_t(t))
    def ya_t(self, t): return self.ya_j(self.j_t(t))

    def x_ij (self, i, j): return self.x_a(self.xa_i(i), self.ya_j(j))
    def y_ij (self, i, j): return self.y_a(self.xa_i(i), self.ya_j(j))

    def x_pq (self, p, q): return self.x_a(self.xa_p(p), self.ya_q(q))
    def y_pq (self, p, q): return self.y_a(self.xa_p(p), self.ya_q(q))

    def x_a(self, xa, ya): return eval ('cos(alpha) * xa - sin(alpha) * ya', self.__dict__, dict (xa=xa, ya=ya))
    def y_a(self, xa, ya): 
        return eval ('sin(alpha) * xa + cos(alpha) * ya', self.__dict__, dict (xa=xa, ya=ya))

    def x_v(self, v): return eval ('1/sin(Vmax)*sin(v - Vx0)*W/2', self.__dict__, dict (v=v))
    def y_v(self, v): return eval ('1/sin(Vmax)*sin(v - Vy0)*H/2', self.__dict__, dict (v=v))

    def x_t(self, t): return self.x_ij(self.i_t(t), self.j_t(t))
    def y_t(self, t): return self.y_ij(self.i_t(t), self.j_t(t))

    def v_x(self, x):
        r = eval ('arcsin(2*x*sin(Vmax)/W) + Vx0', self.__dict__, dict (x=x))
        if numpy.isnan(r).any ():
            print x[:3],self.Vmax,self.W,self.Vx0,eval('2*x*sin(Vmax)/W', self.__dict__, dict (x=x))[:3]
            raise
        return r
    def v_y(self, y): return eval ('arcsin(2*y*sin(Vmax)/H) + Vy0', self.__dict__, dict (y=y))

    def vx_a(self, xa, ya): return self.v_x(self.x_a(xa, ya))
    def vy_a(self, xa, ya): return self.v_y(self.y_a(xa, ya))

    def vx_pq (self, p, q): return self.v_x(self.x_pq(p, q))
    def vy_pq (self, p, q): return self.v_y(self.y_pq(p, q))

    def vx_ij(self, i, j): return self.v_x(self.x_ij(i, j))
    def vy_ij(self, i, j): return self.v_y(self.y_ij(i, j))

    def vx_i(self, i): return self.v_x(self.xa_i(i))
    def vy_j(self, j): return self.v_y(self.ya_j(j))

    def vx_t(self, t): return self.vx_ij(self.i_t(t), self.j_t(t))
    def vy_t(self, t): return self.vy_ij(self.i_t(t), self.j_t(t))

    def p_t_array(self): return self.p_i(self.i_t_array ())
    def q_t_array(self): return self.q_j(self.j_t_array ())

    def xa_t_array(self): return self.xa_i(self.i_t_array ())
    def ya_t_array(self): return self.ya_j(self.j_t_array ())

    def x_t_array(self): return self.x_ij(self.i_t_array(), self.j_t_array())
    def y_t_array(self): 
        return self.y_ij(self.i_t_array(), self.j_t_array())

    def vx_t_array(self):
        xt = self.x_t_array()
        v = self.v_x(xt)
        #return v
        restrict(v, mirror_max_angle*0.8, mirror_max_angle, v.max())
        v = -restrict(-v, mirror_max_angle*0.8, mirror_max_angle, -v.min())
        return v
        #return numpy.minimum(numpy.maximum(vx, -mirror_max_angle), mirror_max_angle)

    def vy_t_array(self):
        v = self.v_y(self.y_t_array())
        return v
        restrict(v, mirror_max_angle*0.8, mirror_max_angle, v.max())
        v = -restrict(-v, mirror_max_angle*0.8, mirror_max_angle, -v.min())
        return v
        #return numpy.minimum(numpy.maximum(vy, -mirror_max_angle), mirror_max_angle)

    def estimate_offsets (self, i_target, j_target, i_response, j_response):
        N, M, T = self.N, self.M, self.T
        K, K0, K1 = self.K, self.K0, self.K1

        ioffset = 0
        joffset = 0


        M1 = M if M == 1 else M-1
        #M1 = M
        for j in range(M):
            if M1<M and not j:
                continue
            t0 = K0*T + j*(N+K)*T
            t1 = t0 + N*T

            err = i_target[t0:t1] - i_response[t0:t1]
            ioffset += err / M1

            err = j_target[t0:t1] - j_response[t0:t1]
            joffset += err / M1

        ioffset_std = ioffset.std()
        ioffset_max = abs(ioffset).mean()
        ioffset = ioffset.mean()

        joffset_std = joffset.std()
        joffset_max = abs(joffset).mean()
        joffset = joffset.mean()

        toffset = -ioffset * T

        return toffset, ioffset, joffset, ioffset_std, joffset_std, ioffset_max, joffset_max

    def line_average (self, data):
        result = 0
        N,M,K,K0,T = self.N, self.M, self.K, self.K0, self.T
        
        M1 = M if M == 1 else M-1
        for j in range (M):
            if M1<M and not j:
                continue
            t0 = K0*T + j*(N+K)*T
            t1 = t0 + N*T
            result += data[t0:t1]
        return result / M1

class MirrorDriverCubic (MirrorDriver):
    
    def __init__(self, *args, **kws):
        MirrorDriver.__init__ (self, *args, **kws)

    def init_curve (self, flyback_ratio, flystart_ratio):
        N, M, T = self.N, self.M, self.T

        Kx = 0*int(3*get_mirror_settling_time(self.vx_ij(N,0) - self.vx_ij(0,0)) * self.clock_rate / T)
        Ky = 0*int(3*get_mirror_settling_time(self.vy_ij(N,0) - self.vy_ij(0,0)) * self.clock_rate / T)
        Kx1 = 0*int(3*get_mirror_settling_time(self.vx_ij(N,M) - self.vx_ij(0,M)) * self.clock_rate / T)
        Ky1 = 0*int(3*get_mirror_settling_time(self.vy_ij(N,M) - self.vy_ij(0,M)) * self.clock_rate / T)
        print Kx, Ky, Kx1, Ky1, int(N*flyback_ratio)
        K = max(Kx, Ky, Kx1, Ky1, int(N*flyback_ratio), 1)

        Kx = int(3*get_mirror_settling_time(self.vx_ij(0,0)) * self.clock_rate / T)
        Ky = int(3*get_mirror_settling_time(self.vy_ij(0,0)) * self.clock_rate / T)
        K0 = max (Kx, Ky, 1, K)

        Kx = int(3*get_mirror_settling_time(self.vx_ij(N,M)) * self.clock_rate / T)
        Ky = int(3*get_mirror_settling_time(self.vy_ij(N,M)) * self.clock_rate / T)
        K1 = max (Kx, Ky, 1, K)

        S = max(1, int(K*flystart_ratio))

        return K, K0, K1, S

    def compute_curve(self):
        from curve import Line, Point, Curves        
        N, M, T = self.N, self.M, self.T
        K, K0, K1 = self.K, self.K0, self.K1
        S = self.S
        S0 = S/K * K0

        toffset = self.t_offset
        ioffset = Point(0, self.i_offset)
        joffset = Point(0, self.j_offset)

        icurve = Curves()
        jcurve = Curves()

        icurve.add_line(Line(Point(-(4*K1*T)//5,self.io), Point(0, self.io), beta_exp=1.2))
        jcurve.add_line(Line(Point(-(4*K1*T)//5,self.jo), Point(0, self.jo)))

        for j in range(M):
            if j:
                start_t = (K0 + (j-1) * (N+K))*T + T*N + (T*K)//2
            else:
                start_t = (T*K0)//2
            end_t = start_t + 1
            level_i = self.i_offset
            #icurve.add_line(Line (Point(start_t, level_i), Point(end_t, level_i)))

            start_i = 0
            end_i = N
            start_j = j
            end_j = j
            start_t = toffset + (K0 + j * (N+K))*T
            end_t = start_t + T*N
            iline = Line(Point(start_t, start_i), Point (end_t, end_i)\
                             ).stretch_end_x(-toffset + T*S/10).stretch_start_x(T*S) + ioffset
            if j<M-1:
                iline.set_beta_exp(1.3)
            icurve.add_line(iline)
            if j:
                jcurve.add_line(Line(Point(start_t, start_j), Point (end_t, end_j)).stretch_start_x(T*S) + joffset)
            else:
                jcurve.add_line(Line(Point(start_t, start_j), Point (end_t, end_j)).stretch_start_x(T*S0) + joffset)

        start_i = self.io
        end_i = self.io
        start_j = self.jo
        end_j = self.jo
        start_t = (K0 + (M-1)*(N+K) + N + K1)*T
        end_t = start_t + (K1)*T
        icurve.add_line(Line(Point(start_t, start_i), Point (end_t, end_i)))
        jcurve.add_line(Line(Point(start_t, start_j), Point (end_t, end_j)))

        self.icurve = icurve
        self.jcurve = jcurve

        return K, K0, K1

    def i_t_array(self):
        r = self.icurve.graph(range(self.get_ticks()))
        return r

    def j_t_array(self):
        r = self.jcurve.graph(range(self.get_ticks()))
        return r

def restrict(v, v0, v1, vmax):
    return numpy.minimum(numpy.maximum(v, -mirror_max_angle), mirror_max_angle)
    return v
    if vmax <= v1:
        return v
    indices = numpy.where (v>v0)
    dv = (v[indices] - v0)

    if 0:
        A = ((v1-v0)/(vmax-v0)-2/numpy.pi)/(1-2/numpy.pi)
        B = 1 - A
        v[indices] = v0 + A*dv + B*2/numpy.pi*(vmax-v0)*numpy.sin(numpy.pi/2/(vmax-v0)*dv)
    if 0:
        v[indices] = v0 + (v1-v0)*numpy.sin(numpy.pi/2/(vmax - v0)*dv)
    if 0:
        d = (v0 - vmax)**3
        a = -vmax*(4*v0**2-6*v0*v1+v0*vmax+vmax**2)/d
        b = 2*(2*v0**2-3*v0*v1+2*v0*vmax-3*vmax*v1+2*vmax**2)/d/2
        c = -3*(v0+vmax - 2*v1)/d/3
        v[indices] = v0 + a*(v[indices] - v0) + b*(v[indices]**2 - v0**2) + c*(v[indices]**3-v0**3)
    if 1:
        d = 4*(2*vmax+v0-3*v1)/(vmax**4-4*v0*vmax**3-4*vmax*v0**3+6*vmax**2*v0**2+v0**4)/4
        c = -3*(3*v0**2-8*v0*v1+6*v0*vmax+3*vmax**2-4*vmax*v1)/(vmax**4-4*v0*vmax**3-4*vmax*v0**3+6*vmax**2*v0**2+v0**4)/3
        a = -vmax*(6*v0**3+3*vmax*v0**2-12*v0**2*v1-vmax**3+4*v0*vmax**2)/(vmax**4-4*v0*vmax**3-4*vmax*v0**3+6*vmax**2*v0**2+v0**4)
        b = 6*v0*(v0**2-2*v0*v1+2*v0*vmax+3*vmax**2-4*vmax*v1)/(vmax**4-4*v0*vmax**3-4*vmax*v0**3+6*vmax**2*v0**2+v0**4)/2
        v[indices] = v0 + a*(v[indices] - v0) + b*(v[indices]**2 - v0**2) + c*(v[indices]**3-v0**3) + d*(v[indices]**4-v0**4)
    return v

if __name__ == '__main__':
    s = MirrorDriver(image_size=(5,2), T=3, alpha=0) #, roi = (200,300,400,800))#, alpha=180)
    import time

    if 1:
        st = time.time ()
        i = map(s.i_t, range(s.get_ticks()))
        j = map(s.j_t, range(s.get_ticks()))
        print time.time () - st
        #st = time.time ()
        #it = list (s.i_t_iter())
        #print time.time () - st

    st = time.time ()
    iarr = s.i_t_array()
    print 'i_t_array took %s seconds' % (time.time () - st)
    st = time.time ()
    jarr = s.j_t_array()
    print time.time () - st
    print 'j_t_array took %s seconds' % (time.time () - st)


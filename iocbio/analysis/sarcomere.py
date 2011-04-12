
# Author: Pearu Peterson
# Created: June 2010

from __future__ import division

import sys
import numpy
import fftw3

numpy.seterr("raise")


def interpolate_bilinear(line, image, (i0, j0), (di, dj)):
    """ Compute image values on a line starting at (i0,j0)
    and ending at (i0+(N-1)*di, j0+(N-1)*dj) using bilinear
    interpolation scheme.

    Parameters
    ----------
    line : numpy.ndarray
      An array with length N where computed line values will be stored.

    image: numpy.ndarray
      An array with rank 2 defining image.

    (i0, j0) : (float, float)
      Specify starting point of the line.

    (di, dj) : (float, float)
      Specify the direction of the line.

    Returns
    -------
    None

    Notes
    -----
    It is assumed that line points will not go over image borders.
    """
    N = len(line)
    for n in range (N):
        ir = i0 + n * di
        jr = j0 + n * dj
        i, j = int (ir), int (jr)            
        v = 0
        for (a,b,c,d,s) in [(i,j,i+1,j+1,1), (i,j+1,i+1,j,-1), (i+1,j,i,j+1,-1), (i+1,j+1,i,j,1)]:
            v += image[a,b] * (ir - c) * (jr - d) * s
        line[n] = v

from .lineinterp import interpolate_bilinear, interpolate_bicubic, acf, interpolate_bilinear_at, interpolate_bilinear_at_point, acf2

def _calc_params(pixel_size, roi_center_line, N):
    """ Calculate parameters to estimate_period_* functions.
    """
    j0,i0,j1,i1 = roi_center_line
    l_px = ((i0-i1)**2 + (j0-j1)**2)**0.5
    i2 = (j1 - j0)/l_px
    j2 = -(i1 - i0)/l_px

    l_um = (((i0-i1)*pixel_size[0])**2 + ((j0-j1)*pixel_size[1])**2)**0.5 * 1e6
    imin = int(1*N/l_um) # corresponds to 1um period
    imax = int(3*N/l_um) # corresponds to 3um period
    kmin = int(N / imax)
    kmax = int(N / imin)

    return (i0,j0,i1,j1,i2,j2), (imin,imax,kmin,kmax), l_um

def _alloc_fft(N, _cache={}):
    if N not in _cache:
        line = numpy.empty((N,), dtype=complex)
        fline = numpy.empty ((N,), dtype=complex)
        fft = fftw3.Plan (line, fline, direction='forward', flags=['estimate'])
        ifft = fftw3.Plan (fline, line, direction='backward', flags=['estimate'])
        _cache[N] = (line, fline), (fft, ifft)
    return _cache[N]

def estimate_period_fft(image, pixel_size, roi_center_line, roi_width, N, filter_low=False, filter_high=False):
    (i0,j0,i1,j1,i2,j2), (imin,imax,kmin,kmax), l_um = _calc_params(pixel_size, roi_center_line, N)
    (line, fline), (fft, ifft) = _alloc_fft(N)
    pline = numpy.zeros ((N,), dtype=float)
    tmpline = numpy.zeros ((N,), dtype=float)
    tmpimage = image.astype (float)
    for k in range (-roi_width//2, roi_width//2+1):
        interpolate_bilinear(tmpline, tmpimage, (i0+k*i2, j0+k*j2), ((i1-i0)/N, (j1-j0)/N))
        line[:] = tmpline
        fft.execute()
        fline[0] = 0
        fline *= fline.conjugate ()
        pline += fline.real
    peak_index = pline[kmin:kmax].argmax() + kmin
    period_px = N / peak_index
    return period_px * l_um / N

def estimate_period_acf1(image, pixel_size, roi_center_line, roi_width, N, filter_low=False, filter_high=False):
    (i0,j0,i1,j1,i2,j2), (imin,imax,kmin,kmax), l_um = _calc_params(pixel_size, roi_center_line, N)
    (line, fline), (fft, ifft) = _alloc_fft(N)
    pline = numpy.zeros ((N,), dtype=float)
    tmpline = numpy.zeros ((N,), dtype=float)
    tmpimage = image.astype (float)
    for k in range (-roi_width//2, roi_width//2+1):
        interpolate_bilinear(tmpline, tmpimage, (i0+k*i2, j0+k*j2), ((i1-i0)/N, (j1-j0)/N))
        line[:] = tmpline
        fft.execute()
        fline[0] = 0
        fline *= fline.conjugate ()
        if filter_low:
            fline[:kmin] = 0
            fline[len (fline)- kmin+1:] = 0
        if filter_high:
            fline[kmax:-kmax+1] = 0            
        ifft.execute()
        pline += line.real
    period_px = pline[imin:imax].argmax() + imin
    return period_px * l_um / N

def estimate_period_acf2(image, pixel_size, roi_center_line, roi_width, N, filter_low=False, filter_high=False):
    (i0,j0,i1,j1,i2,j2), (imin,imax,kmin,kmax), l_um = _calc_params(pixel_size, roi_center_line, N)
    (line, fline), (fft, ifft) = _alloc_fft(N)
    pline = numpy.zeros ((N,), dtype=float)
    l = []
    tmpline = numpy.zeros ((N,), dtype=float)
    tmpimage = image.astype (float)
    import matplotlib.pyplot as plt
    for k in range (-roi_width//2, roi_width//2+1):
        interpolate_bilinear(tmpline, tmpimage, (i0+k*i2, j0+k*j2), ((i1-i0)/N, (j1-j0)/N))
        line[:] = tmpline
        fft.execute()
        fline[0] = 0
        fline *= fline.conjugate()
        if filter_low:
            fline[:kmin] = 0
            fline[len (fline)- kmin+1:] = 0
        if filter_high:
            fline[kmax:-kmax+1] = 0            
        ifft.execute()
        period_px = line.real[imin:imax].argmax() + imin
        if period_px in [imin,imax-1]:
            #print 'Skipping', k, period_px
            continue
        l.append(period_px)
    period_px = numpy.mean (l)
    return period_px * l_um / N

def estimate_period_acf3(image, pixel_size, roi_center_line, roi_width, N, filter_low=False, filter_high=False):
    (i0,j0,i1,j1,i2,j2), (imin,imax,kmin,kmax), l_um = _calc_params(pixel_size, roi_center_line, N)
    (line, fline), (fft, ifft) = _alloc_fft(N)
    pline = numpy.zeros ((N,), dtype=float)
    l = []
    w = []
    tmpline = numpy.zeros ((N,), dtype=float)
    tmpimage = image.astype (float)
    for k in range (-roi_width//2, roi_width//2+1):
        interpolate_bilinear(tmpline, tmpimage, (i0+k*i2, j0+k*j2), ((i1-i0)/N, (j1-j0)/N))
        line[:] = tmpline
        fft.execute()
        fline[0] = 0
        fline *= fline.conjugate ()
        if filter_low:
            fline[:kmin] = 0
            fline[len (fline)- kmin+1:] = 0
        if filter_high:
            fline[kmax:-kmax+1] = 0            
        ifft.execute()
        period_px = line.real[imin:imax].argmax() + imin
        if period_px in [imin,imax-1]:
            #print 'Skipping', k, period_px
            continue
        w.append(line.real[period_px]/line.real[0])
        l.append(period_px * w[-1])

    period_px = numpy.sum(l) / numpy.sum(w)
    return period_px * l_um / N

def estimate_period_fft4(image, pixel_size, roi_center_line, roi_width, N, filter_low=False, filter_high=False):
    (i0,j0,i1,j1,i2,j2), (imin,imax,kmin,kmax), l_um = _calc_params(pixel_size, roi_center_line, N)
    (line, fline), (fft, ifft) = _alloc_fft(N)
    pline = numpy.zeros ((N,), dtype=float)
    tmpline = numpy.zeros ((N,), dtype=float)
    tmpimage = image.astype (float)
    for k in range (-roi_width//2, roi_width//2+1):
        interpolate_bilinear(tmpline, tmpimage, (i0+k*i2, j0+k*j2), ((i1-i0)/N, (j1-j0)/N))
        line[:] = tmpline
        fft.execute()
        fline[0] = 0
        if filter_high:
            fline[kmax:-kmax+1] = 0
        fline *= fline.conjugate ()
        ifft.execute()
        period_px = line.real[imin:imax].argmax() + imin
        if period_px in [imin,imax-1]:
            #print 'Skipping', k, period_px
            continue
        pline += fline.real
    peak_index = pline[kmin:kmax].argmax() + kmin
    kmin = max (peak_index-10, kmin)
    kmax = min(peak_index+10, kmax)
    peak_index2 = (pline[kmin:kmax]**4 * numpy.arange(kmin,kmax)).sum()/(pline[kmin:kmax]**4).sum()
    #print peak_index, peak_index2
    period_px = N / peak_index
    return period_px * l_um / N

def estimate_period_acf5(image, pixel_size, roi_center_line, roi_width, N, filter_low=False, filter_high=False):
    (i0,j0,i1,j1,i2,j2), (imin,imax,kmin,kmax), l_um = _calc_params(pixel_size, roi_center_line, N)
    (line, fline), (fft, ifft) = _alloc_fft(N)
    pline = numpy.zeros ((N,), dtype=float)
    l = []
    tmpline = numpy.zeros ((N,), dtype=float)
    tmpimage = image.astype(float)
    import matplotlib.pyplot as plt
    period_estimate = None
    for k in range (-roi_width//2, roi_width//2+1):
        interpolate_bilinear(tmpline, tmpimage, (i0+k*i2, j0+k*j2), ((i1-i0)/N, (j1-j0)/N))
        if period_estimate is None:
            line[:] = tmpline
            fft.execute()
            fline[0] = 0
            fline[1] = fline[-1] = 0
            fline[2] = fline[-2] = 0
            fline *= fline.conjugate()
            if filter_high:
                fline[kmax:-kmax+1] = 0            
            ifft.execute()
            period_estimate = line.real[imin:imax].argmax() + imin
            #print line.real[period_estimate]/N
        
        p = orig_estimate = period_estimate
        dp = 0.05 
        amx = 0
        la = []
        for i, dp in enumerate(numpy.arange (-1,1+dp,dp)):
            a = acf2(p+dp, N, tmpimage, (i0+k*i2,j0+k*j2),((i1-i0)/N, (j1-j0)/N))
            la.append (a)
            if a>amx:
                period_estimate = p+dp 
                amx = a
        #print numpy.diff(la), orig_estimate - period_estimate
        l.append (period_estimate)
        continue
        print l
        sys.exit ()
        while a0 < a1:
            a0 = a1
            p0 = p0-dp
            a1 = acf2(p0-dp, N, tmpimage, (i0+k*i2,j0+k*j2),((i1-i0)/N, (j1-j0)/N))

        a1 = acf2(p0+dp, N, tmpimage, (i0+k*i2,j0+k*j2),((i1-i0)/N, (j1-j0)/N))
        while a0 < a1:
            a0 = a1
            p0 = p0+dp
            a1 = acf2(p0+dp, N, tmpimage, (i0+k*i2,j0+k*j2),((i1-i0)/N, (j1-j0)/N))
            #print '>>', p0+dp, a1


            #print '>>>', p0-dp, a1
        #print p0, a0, a1
        print p0 - period_estimate
        l.append(p0); continue

        sys.exit ()
        for dp in [-1.5, -1, -0.5, 0, 0.5, 1]:
            p = period_estimate + dp
            r2 = acf2(p, N, tmpimage, (i0+k*i2,j0+k*j2),((i1-i0)/N, (j1-j0)/N))
            floor_p = int(p)
            ceil_p = floor_p + 1
            dp1 = ceil_p - p
            dp2 = p - floor_p

            v1, w1 = None, None
            r = 0
            for i in range (N):
                if i+ceil_p >= N:
                    break
                v2 = interpolate_bilinear_at_point(tmpimage, i0+k*i2+i*(i1-i0)/N, j0+k*j2+i*(j1-j0)/N)
                w2 = interpolate_bilinear_at_point(tmpimage, i0+k*i2+(i+p)*(i1-i0)/N, j0+k*j2+(i+p)*(j1-j0)/N)
                if dp2!=0:
                    if v1 is not None:
                        r += dp2*(2*(v1*w1+v2*w2)+v1*w2+v2*w1)/6
                    v1, w1 = v2, w2

                    v2 = interpolate_bilinear_at_point (tmpimage, i0+k*i2+(i+dp1)*(i1-i0)/N, j0+k*j2+(i+dp1)*(j1-j0)/N)
                    w2 = interpolate_bilinear_at_point (tmpimage, i0+k*i2+(i+ceil_p)*(i1-i0)/N, j0+k*j2+(i+ceil_p)*(j1-j0)/N)
                    r += dp1*(2*(v1*w1+v2*w2)+v1*w2+v2*w1)/6
                else:
                    if v1 is not None:
                        r += dp1*(2*(v1*w1+v2*w2)+v1*w2+v2*w1)/6
                v1, w1 = v2, w2
            print 'p=%s, r=%s, r2=%s' % (p,r, r2)
            continue
            print r
            if dp2==0.0:
                dp2 = 1.0
                assert dp1==1.0
            M = len (icoords1)
            #values1 = numpy.zeros((M,), dtype=float)
            #values2 = numpy.zeros((M,), dtype=float)
            values1 = numpy.array (values1, dtype=float)
            values2 = numpy.array (values2, dtype=float)
            icoords1 = numpy.array (icoords1, dtype=float)
            jcoords1 = numpy.array (jcoords1, dtype=float)
            icoords2 = numpy.array (icoords2, dtype=float)
            jcoords2 = numpy.array (jcoords2, dtype=float)
            #interpolate_bilinear_at(values1, tmpimage, icoords1, jcoords1)
            #interpolate_bilinear_at(values2, tmpimage, icoords2, jcoords2)
            #print nodes1[:12]
            #print values1[:12]
            #print nodes2[:12]
            #print values2[:12]
            r = 0
            for i in range (len (values1)-1):
                dp12 = dp1 if (i%2==0) else dp2
                v1,v2 = values1[i:i+2]
                w1,w2 = values2[i:i+2]
                r += dp12*(2*(v1*w1+v2*w2)+v1*w2+v2*w1)/6
            print 'p=%s, r=%s, r2=%s' % (p,r,r2)
        sys.exit ()

        #if period_px in [imin,imax-1]:
            #print 'Skipping', k, period_px
        #    continue
        l.append(period_px); continue


        if 0:
            plt.plot(line.real)
            plt.show ()
            sys.exit ()

        rline = line.real.copy()
        count = ((numpy.diff((rline > 0).astype(int))) == 1).sum ()
        assert count
        period_px = N / count
        l.append(period_px); continue
        sys.exit ()


        dk = 1
        left_acf = acf(rline, period_px-dk)
        max_acf = acf(rline, period_px)
        right_acf = acf(rline, period_px+dk)
        k = period_px
        if left_acf > max_acf:
            k = period_px-dk
            while left_acf > max_acf:
                k -= dk
                max_acf = left_acf
                left_acf = acf(rline, k)
            period_px = k + dk
        elif right_acf > max_acf:
            k = period_px+dk
            while right_acf > max_acf:
                k += dk
                max_acf = right_acf
                right_acf = acf(rline, k)
            period_px = k - dk
        #print period_px, k
        l.append(period_px)

        left_acf = acf(rline, period_px-dk)
        max_acf = acf(rline, period_px)
        right_acf = acf(rline, period_px+dk)

        assert left_acf < max_acf and max_acf > right_acf,`left_acf, max_acf, right_acf`
        continue
        ym2, ym1, y0, y1, y2 = line.real[period_px-2:period_px+3]
        dym1 = (y0 - ym2)/2
        dy0 = (y1 - ym1)/2
        dy1 = (y2 - y0)/2
        if dy0==dym1:
            period_px = period_px - 0.5
        elif dy0==dy1:
            period_px = period_px + 0.5
        elif dym1 > 0 and dy0 < 0:
            period_px = period_px - dy0 / (dy0 - dym1)
        elif dy0 > 0 and dy1 < 0:
            period_px = period_px + dy1 / (dy1 - dy0)
        else:
            assert 0,`dym1,dy0,dy1`
        #print period_px
        l.append(period_px)
    period_px = numpy.mean (l)
    return period_px * l_um / N


def sarcomere_length(image, pixel_size, roi_center_line, roi_width, nof_interpolation_points=256):
    """
    Compute sacromere length.
    """
    #import matplotlib.pyplot as plt


    N = nof_interpolation_points
    r = []
    l = []
    for func, method in [#(estimate_period_fft, 'max mean spectrum'),
                         #(estimate_period_acf1, '1st max mean ACF'),
        (estimate_period_acf2, 'mean 1st max ACF'),
        #(estimate_period_acf3, 'wmean 1st max ACF'),
        #(estimate_period_fft4, 'max mean spectrum^2'),
        (estimate_period_acf5, 'experimental'),
                         ]:
        for (filter_low,filter_high,comment) in [(False, False,''),
                                                 #(True, False,'filter low'),
                                                 #(False, True, 'filter high'),
                                                 #(True, True, 'filter low & high'),
                                                 ]:
            p = func(image, pixel_size, roi_center_line, roi_width, N, filter_low=filter_low, filter_high=filter_high)
            l.append ('%s (%s)' % (method, comment))
            r.append(p)
    return r, l

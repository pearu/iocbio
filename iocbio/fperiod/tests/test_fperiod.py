from __future__ import division

import numpy
from iocbio import fperiod #dp, ipwf

def show_measure (f):
    from matplotlib import pyplot as plt
    n = f.shape[-1]
    y = numpy.arange(0,n+n/100, n/100)
    o = fperiod.objective (y, f, order=1, method=0)
    plt.subplot (211)
    plt.plot (f.T)
    plt.subplot (212)
    plt.plot(y, o)
    plt.axhline(0)
    plt.axvline(1)
    plt.axvline(2)
    plt.axvline(3)
    plt.axvline(4)
    plt.show ()

def test_find_zero ():
    # FE(f) has minimum and maximum within the interval [2,3]
    x = numpy.arange(10)
    f = numpy.sin(4.9*x/x[-1]*numpy.pi*2)/(1+x)
    status, zero, slope = fperiod.ipwf.e11_find_zero(2,3,f,order=1)
    assert status==0,`status`
    assert 2.06<=zero<=2.07,`zero` # first nonzero minimum
    assert slope>0, `slope`
    fp = fperiod.fperiod(f)
    assert fp==zero,`fp, zero`
    

    status, zero, slope = fperiod.ipwf.e11_find_zero(3,2,f,order=1)
    assert status==0,`status`
    assert 2.98<=zero<=2.99,`zero`
    assert slope<0, `slope`

    for initial_period in [0, 0.5, 1, 1.5, 2, 2.5, 2.9]:
        assert fperiod.fperiod(f, initial_period=initial_period)==fp,`initial_period, fperiod.fperiod(f, initial_period=initial_period)`
    #show_measure (f)

    # FE(f) has maximum and minimum within the interval [1,2]
    x = numpy.arange(20)
    w = 12.57
    f = numpy.sin(w*(x/x[-1])**2*numpy.pi*2)
    status, zero, slope = fperiod.ipwf.e11_find_zero(1,2,f,order=1)
    assert status==0,`status`
    assert 1.65<=zero<=1.66,`zero`
    assert slope<0, `slope`
    status, zero, slope = fperiod.ipwf.e11_find_zero(2,1,f,order=1)
    assert status==0,`status`
    assert 1.94<=zero<=1.95,`zero` # first nonzero minimum
    assert slope>0, `slope`
    fp = fperiod.fperiod(f)
    assert fp==zero,`fp, zero`

    for initial_period in [0, 0.5, 1, 1.5, 1.8, 2, 2.5, 2.9, 3]:
        assert fperiod.fperiod(f, initial_period=initial_period)==fp,`initial_period, fperiod.fperiod(f, initial_period=initial_period)`

    #show_measure (f)

    # FE(f) is constant
    x = numpy.arange(20)
    f = 0*x+1
    fp = fperiod.fperiod(f)
    assert fp==-2,`fp`
    fp = fperiod.fperiod(f, initial_period = 2)
    assert fp==-2,`fp`

    # FE(f) has no nonzero minimum
    x = numpy.arange(0,1,0.1)
    f = x*(1-x)
    fp = fperiod.fperiod(f)
    assert fp==-2,`fp`
    fp = fperiod.fperiod(f, initial_period = 2)
    assert fp==-2,`fp`
    #show_measure (f)
    fp = fperiod.fperiod(f, initial_period = 6)
    assert fp==-2,`fp`

    # FE(f) has exactly one non-zero minimum
    x = numpy.arange(0,1,0.1)
    f = x*(1-x)*(0.25-x)*(0.5-x)*(0.75-x)
    fp = fperiod.fperiod(f)
    for initial_period in [0, 0.5, 1, 4, 5, 6, 7, 7.5, 8.5]:
        assert fperiod.fperiod(f, initial_period=initial_period)==fp,`initial_period, fperiod.fperiod(f, initial_period=initial_period)`

    #show_measure (f)

if __name__=='__main__':
    test_find_zero ()

from __future__ import division
from matplotlib import pyplot as plt
from numpy import *
from iocbio.analysis import cf

def show_cf ():
    x = arange (50, dtype=float)
    xx = arange(0,len (x), 0.01)
    y = arange(-len (x)*2,len (x)+10, 0.01)
    P = (len(x)-1)*0.4
    f = sin(2*pi*(x/P-0.3)) + 0#.5
    print 'P=',P
    
    def f1(x, order, f=f):
        return array([cf.e11_f1_evaluate(x0, f, order=order) for x0 in x])
    def f2(x, order, f=f):
        return array([cf.a22_f1_evaluate(x0, f, order=order) for x0 in x])
    def f3(x, order, f=f):
        return array([cf.a33_f1_evaluate(x0, f, order=order) for x0 in x])

    def a00(y, order=0, f=f): return array([cf.a00_evaluate(y0, f, order=order) for y0 in y])
    def b00(y, order=0, f=f): return array([cf.b00_evaluate(y0, f, order=order) for y0 in y])
    def c00(y, order=0, f=f): return array([cf.c00_evaluate(y0, f, order=order) for y0 in y])
    def a11(y, order=0, f=f): return array([cf.a11_evaluate(y0, f, order=order) for y0 in y])
    def b11(y, order=0, f=f): return array([cf.b11_evaluate(y0, f, order=order) for y0 in y])
    def c11(y, order=0, f=f): return array([cf.c11_evaluate(y0, f, order=order) for y0 in y])
    def a22(y, order=0, f=f): return array([cf.a22_evaluate(y0, f, order=order) for y0 in y])
    def b22(y, order=0, f=f): return array([cf.b22_evaluate(y0, f, order=order) for y0 in y])
    def c22(y, order=0, f=f): return array([cf.c22_evaluate(y0, f, order=order) for y0 in y])
    def a33(y, order=0, f=f): return array([cf.a33_evaluate(y0, f, order=order) for y0 in y])
    def b33(y, order=0, f=f): return array([cf.b33_evaluate(y0, f, order=order) for y0 in y])
    def c33(y, order=0, f=f): return array([cf.c33_evaluate(y0, f, order=order) for y0 in y])
    def e11(y, order=0, f=f): return array([cf.e11_evaluate(y0, f, order=order) for y0 in y])

    def make_extremes(name):
        func = getattr (cf, '%s_find_extreme' % name, None)
        def tmpl_extremes(order=0, f=f, func=func):
            y = 0
            status=0
            l = []
            while status==0:
                status, y = func(int(y+1),len(x)-1, f, order=order)
                if status==0:
                    l.append(y)
            return l
        return tmpl_extremes

    def make_zeros(name):
        func = getattr (cf, '%s_find_zero' % name, None)
        def tmpl_zeros(order=0, f=f, func=func):
            y = 0
            status=0
            l = []
            while status==0:
                status, y = func(int(y+1),len(x)-1, f, order=order)
                if status==0:
                    l.append(y)
            return l
        return tmpl_zeros

    a00_extremes = make_extremes('a00')
    b00_extremes = make_extremes('b00')
    c00_extremes = make_extremes('c00')
    a00_zeros = make_zeros('a00')
    b00_zeros = make_zeros('b00')
    c00_zeros = make_zeros('c00')

    a11_extremes = make_extremes('a11')
    b11_extremes = make_extremes('b11')
    c11_extremes = make_extremes('c11')
    e11_extremes = make_extremes('e11')
    a11_zeros = make_zeros('a11')
    b11_zeros = make_zeros('b11')
    c11_zeros = make_zeros('c11')
    e11_zeros = make_zeros('e11')

    a22_extremes = make_extremes('a22')
    b22_extremes = make_extremes('b22')
    c22_extremes = make_extremes('c22')
    a22_zeros = make_zeros('a22')
    b22_zeros = make_zeros('b22')
    c22_zeros = make_zeros('c22')

    a33_extremes = make_extremes('a33')
    b33_extremes = make_extremes('b33')
    c33_extremes = make_extremes('c33')
    a33_zeros = make_zeros('a33')
    b33_zeros = make_zeros('b33')
    c33_zeros = make_zeros('c33')

    plt.figure(figsize=(8,12))

    plt.subplot(211)
    styles = ['-', '--', ':']
    colors = ['b', 'r', 'g', 'k', 'c']
    for i,ff in enumerate([f1,f2,f3][:1]):
        for order in range (2):
            plt.plot(xx, ff(xx, order), colors[order]+styles[i], label='%s_%s' % (ff.__name__,order))
    plt.plot(x, f, 'x',label='f')
    plt.xlabel ('x')
    plt.legend ()

    plt.subplot(212)
    styles = ['-', '--', ':','-.']
    colors = ['b', 'r', 'g', 'k', 'c']
    i = -1
    plt.axvline (x=P, color='y')
    for a in [a00, b00, c00,
              a11, b11, c11, e11,
              a22, b22, c22,
              a33, b33, c33,
              ]:
        extremes = eval ('%s_extremes' % (a.__name__))
        zeros = eval ('%s_zeros' % (a.__name__))
        for order in range (4):
            if (a.__name__, order) not in [
                #('a00',0),
                #('a33',0),
                #('a33',1),
                #('a33',2),
                ('e11',0),
                ('e11',1),
                ('e11',2),
                #('a11',1),
                #('a11',1),
                #('b11',0),
                #('b11',1),
                #('c11',2),
                #('c11',0),
                #('c11',3),
                ]:
                continue
            print a
            i += 1
            color = colors[i]
            style = styles[i]
            plt.plot(y, a(y, order), color+style, label='%s_%s' % (a.__name__,order))
            print (a.__name__, order), extremes(order=order), zeros(order=order)
            for y1 in extremes(order=order):
                plt.axvline(x=y1, color=color, linestyle=style, marker='^')

            for y1 in zeros(order=order):
                plt.axvline(x=y1, color=color, linestyle=style, marker='o')


        
    plt.xlabel ('y')
    plt.legend ()

    plt.show ()

    pass

if __name__=='__main__':
    show_cf()

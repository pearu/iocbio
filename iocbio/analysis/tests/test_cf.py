
from matplotlib import pyplot as plt
from numpy import *
from iocbio.analysis import cf

def show_cf ():
    x = arange (10)
    xx = arange(0,len (x), 0.01)
    y = arange(0,len (x), 0.01)
    f = sin(x)
    
    def f1(x, order, f=f):
        return array([cf.a11_f1_evaluate(x0, f, order=order) for x0 in x])
    def f2(x, order, f=f):
        return array([cf.a22_f1_evaluate(x0, f, order=order) for x0 in x])
    def f3(x, order, f=f):
        return array([cf.a33_f1_evaluate(x0, f, order=order) for x0 in x])

    def a11(y, order=0, f=f):
        return array([cf.a11_evaluate(y0, f, order=order) for y0 in y])
    def a22(y, order=0, f=f):
        return array([cf.a22_evaluate(y0, f, order=order) for y0 in y])
    def a33(y, order=0, f=f):
        return array([cf.a33_evaluate(y0, f, order=order) for y0 in y])

    plt.figure(figsize=(8,12))

    plt.subplot(211)
    styles = ['-', '--', ':']
    colors = ['b', 'r', 'g', 'k', 'c']
    for i,ff in enumerate([f1,f2,f3]):
        for order in range (2):
            plt.plot(xx, ff(xx, order), colors[order]+styles[i], label='%s_%s' % (ff.__name__,order))
    plt.plot(x, f, 'x',label='f')
    plt.xlabel ('x')
    plt.legend ()

    plt.subplot(212)
    styles = ['-', '--', ':']
    colors = ['b', 'r', 'g', 'k', 'c']
    for i,a in enumerate([a11, a22, a33]):
        for order in range (4):
            plt.plot(y, a(y, order), colors[order]+styles[i], label='%s_%s' % (a.__name__,order))
    plt.xlabel ('y')
    plt.legend ()

    plt.show ()

    pass

if __name__=='__main__':
    show_cf()

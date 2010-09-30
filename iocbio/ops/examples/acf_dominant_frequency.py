from numpy import *
from iocbio.ops.autocorrelation import acf, acf_argmax, acf_sinefit
from matplotlib import pyplot as plt

dx = 0.05
x = arange(0,2*pi,dx)
N = len(x)
y = arange(0,N,0.1)

fig = plt.figure (0, (16*1.5,8*1.5))
fig.text(.5,.95, 'Finding dominant frequency via autocorrelation function analysis', ha='center')

for f_expr, sp1, sp2 in [('7*sin(5*x)+4*sin(9*x)', 221,223),
                         ('7*sin(5*x)+6*sin(9*x)', 222,224),
                         ]:
    f = eval(f_expr)
    acf_data = acf(f, y)
    omega = acf_sinefit(f, start_j=1)
    omega2 = acf_sinefit(f, start_j=int(2*pi/omega)+2)
    a = acf(f, 0.0)
    sinefit = a*cos(omega*y)*(N-y)/N
    sinefit2 = a*cos(omega2*y)*(N-y)/N

    plt.subplot(sp1)
    plt.plot(x, f, label='$f(x) = %s$' % (f_expr.replace('*','').replace ('sin','\sin')))
    plt.plot(x, (a/N)**0.5 *sin(omega/dx*x), label='$f(x)=\sqrt{A/N}\sin(\omega_{sine_1}x)$')
    plt.plot(x, (a/N)**0.5 *sin(omega2/dx*x), label='$f(x)=\sqrt{A/N}\sin(\omega_{sine_2}x)$')
    plt.legend ()
    plt.xlabel ('x')
    plt.ylabel ('f(x)')
    plt.title ('Test functions')

    plt.subplot (sp2)
    plt.plot(y, acf_data, label='ACF(f(x))')
    plt.plot(y, sinefit, label=r'$A\cos (\omega_{sine_1}y) \frac{N-y}{N}$, $\omega_{sine_1}$=%.3f' % (omega/dx))
    plt.plot(y, sinefit2, label=r'$A\cos (\omega_{sine_2}y) \frac{N-y}{N}$, $\omega_{sine_2}$=%.3f' % (omega2/dx))

    start_j = 1
    for n in range(1,4):
        y_max = acf_argmax(f, start_j)
        start_j = y_max + 1
        est_period = dx * y_max/n
    
        plt.axvline(y_max, label='$%s\cdot2\pi/(\Delta x\cdot y_{max}^{(%s)})$=%.3f' % (n, n, 2*pi/est_period),
                    color='k')
    plt.legend ()
    plt.xlabel('y')
    plt.ylabel ('acf')
    plt.title ('Autocorrelation functions')

plt.savefig('acf_dominant_frequency.png')
#plt.savefig('acf_data_6.png')
plt.show ()


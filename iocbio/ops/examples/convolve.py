
from numpy import *
from scipy.stats import poisson
from iocbio.ops import convolve
kernel = array([0,1,2,2,2,1,0])
x = arange(0,2*pi,0.1)
data = 50+7*sin(x)+5*sin(2*x)
data_with_noise = poisson.rvs(data)
data_convolved = convolve(kernel, data_with_noise)


from matplotlib import pyplot as plt

plt.plot (x,data,label='data')
plt.plot (x,data_with_noise,'o',label='data with poisson noise')
plt.plot (x,data_convolved,label='convolved noisy data')
plt.legend ()
plt.xlabel ('x')
plt.ylabel ('data')
plt.title ('Convolution with %s for recovering data=50+7*sin(x)+5*sin(2*x)' % (kernel))

plt.savefig('convolve_1d.png')

plt.show ()

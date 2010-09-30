
from numpy import *
from iocbio.ops.autocorrelation import acf, acf_argmax, acf_sinefit
# Define a signal:
dx = 0.05
x = arange(0,2*pi,dx)
N = len(x)
f = 7*sin(5*x)+6*sin(9*x)
# f is shown in the upper right plot below
# Calculate autocorrelation function:
y = arange(0,N,0.1)
af = acf (f, y, method='linear')
# af is shown in the lower right plot below
# Find the first maximum of the autocorrelation function:
y_max1 = acf_argmax(f, method='linear')
# The first maximum gives period estimate for f
print 'period=',dx*y_max1
print 'frequency=',2*pi/(y_max1*dx)
# Find the second maximum of the autocorrelation function:
y_max2 = acf_argmax(f, start_j=y_max1+1, method='linear')
print y_max1, y_max2
# Find sine-fit of the autocorrelation function:
omega = acf_sinefit(f, method='linear')
# The parameter omega in A*cos (omega*y)*(N-y)/N gives
# another period estimate for f:
print 'period=',2*pi/(omega/dx)
print 'frequency=', omega/dx


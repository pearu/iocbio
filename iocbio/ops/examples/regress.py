from numpy import *
from scipy.stats import poisson
from iocbio.ops.regress import regress
x = arange(0,2*pi,0.1)
data = 50+7*sin(x)+5*sin(2*x)
data_with_noise = poisson.rvs (data)
data_estimate, data_estimate_grad = regress(data_with_noise, (0.1, ), 
                                            kernel='tricube', 
                                            boundary='periodic', 
                                            method='average')


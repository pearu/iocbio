
import numpy
from iocbio.ops import convolve
from iocbio.microscope.deconvolution import deconvolve
from iocbio.io import ImageStack

from matplotlib import pyplot as plt

kernel = numpy.array([0,1,3,1,0])
test_data = numpy.array([0, 0,0,0,2, 0,0,0,0, 0,1,0,1, 0,0,0])
data =convolve(kernel, test_data)

plt.plot(test_data)
plt.plot (data)
plt.show ()

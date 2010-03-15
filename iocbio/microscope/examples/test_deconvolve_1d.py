
import numpy
from iocbio.ops import convolve
from iocbio.microscope.deconvolution import deconvolve
from iocbio.io import ImageStack
import scipy.stats

from matplotlib import pyplot as plt

kernel = numpy.array([0,1,3,1,0])
test_data = numpy.array([0, 0,0,0,2, 0,0,0,0, 0,1,0,1, 0,0,0])*50
data =convolve(kernel, test_data)
degraded_data = scipy.stats.poisson.rvs(numpy.where(data<=0, 1e-16, data)).astype(data.dtype)

psf = ImageStack(kernel, voxel_sizes = (1,))
stack = ImageStack(degraded_data, voxel_sizes = (1,))

deconvolved_data = deconvolve(psf, stack).images

plt.plot(test_data, label='test')
plt.plot (data, label='convolved')
plt.plot (degraded_data, label='degraded')
plt.plot (deconvolved_data, label='deconvolved')
plt.legend()
plt.ylabel('data')
plt.xlabel('index')
plt.title('Deconvolving degraded test data.')
plt.savefig('deconvolve_poisson_1d.png')
plt.show ()

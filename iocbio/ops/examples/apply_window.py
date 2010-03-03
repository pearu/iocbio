
from numpy import *
from iocbio.ops import apply_window

data = ones (50)
window0 = apply_window(data, (1/20., ), smoothness=0, background=0.2)
window1 = apply_window(data, (1/20., ), smoothness=1, background=0)

from matplotlib import pyplot as plt

plt.plot (data,label='data')
plt.plot (window0,'o',label='smoothness=0, bg=0.2')
plt.plot (window1,'o',label='smoothness=1, bg=0')
plt.legend ()
plt.xlabel ('x')
plt.ylabel ('data')
plt.title ('Applying window to constant data=1')

plt.savefig('apply_window_1d.png')

plt.show ()

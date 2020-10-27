import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np

x = np.arange(0, 20)
y = np.exp(-x/3.0)
f = interpolate.interp1d(x, y)  #  kind=‘cubic’

xnew = np.arange(0, 19, 0.1)
ynew = f(xnew)   # use interpolation function returned by `interp1d`
plt.plot(x, y, 'o', xnew, ynew, '--')
plt.show()

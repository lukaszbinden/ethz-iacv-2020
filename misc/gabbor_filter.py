import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# credits: https://moonbooks.org/Articles/How-to-plot-a-Gabor-filter-using-python-and-matplotlib-/

f = 0.1 
theta = math.radians(0.0) # Converts angle x from degrees to radians.
sigma_x = 7.0
sigma_y = 7.0
radius = 20

M = np.zeros((radius*2,radius*2))

def ChangeBase(x,y,theta):
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = y * math.cos(theta) - x * math.sin(theta)
    return x_theta, y_theta

def GaborFunction(x,y,theta,f,sigma_x,sigma_y):
    r1 = ChangeBase(x,y,theta)[0] / sigma_x
    r2 = ChangeBase(x,y,theta)[1] / sigma_y
    arg = - 0.5 * ( r1**2 + r2**2 )
    return math.exp(arg) * math.cos(2*math.pi*f*ChangeBase(x,y,theta)[0])

x = -float(radius)
for i in range(radius*2):
    y = -float(radius)
    for j in range(radius*2):
        M[i,j] = GaborFunction(x,y,theta,f,sigma_x,sigma_y)
        y = y + 1
    x = x + 1

fig = plt.figure(figsize=(10, 8), dpi=120)
ax = fig.gca(projection='3d')
X = np.arange(-20, 20, 1.0)
Y = np.arange(-20, 20, 1.0)
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X, Y, M, rstride=1, cstride=2, cmap=cm.Greys_r,
        linewidth=1, antialiased=False)
ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig('GaborFilter_3d.png')
plt.show()
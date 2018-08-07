import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 15, 0.01)
y = -3*(x-30)**2*np.sin(x) 
plt.title("-3*(x-30)**2*sin(x) ")
plt.plot(x, y)
plt.show()

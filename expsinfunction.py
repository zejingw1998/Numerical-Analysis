import numpy as np
from math import sin, pi
import matplotlib.pyplot as plt


x = np.linspace(0, 2*pi, 1000)


y = np.exp(2*x) * np.sin(2*x)

plt.plot(x, y)

plt.title("exp(2x) * sin(2x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)        
plt.legend(["f(x)"]) 
plt.axhline(0, color='black', linestyle='--')
plt.show()

import numpy as np
from math import sin, pi
import matplotlib.pyplot as plt

# ❶ linspace 的括号写错了，应该是：
x = np.linspace(0, 2*pi, 1000)

# ❷ np.exp() 自动处理数组，不要用 math.exp()
y = np.exp(2*x) * np.sin(2*x)

# ❸ plt.plot() 写成了 plt.plt()
plt.plot(x, y)

plt.title("exp(2x) * sin(2x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)        # grid 拼写错了
plt.legend(["f(x)"])  # legend 拼写错了
plt.axhline(0, color='black', linestyle='--')
plt.show()

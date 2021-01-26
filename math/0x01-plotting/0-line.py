#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

"""To plot y against x with a solid red line"""
plt.xlim(0, 10)
plt.plot(y, '-r')
plt.show()

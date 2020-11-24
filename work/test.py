from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.interpolate import make_interp_spline
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

arr = np.array([1,2,3,4,5,6])
print(arr)
print(arr[:2])

rr = np.hstack((arr[:2],arr[2:]))
print(rr)

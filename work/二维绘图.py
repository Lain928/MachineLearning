import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import random
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 读取txt文件数据 以空格分隔 无头标签
dataset = pd.read_csv('D:/Work/仿真数据/速度2ma/h-25km/SM6防空导弹_6_驱逐舰_2_297km.csv', sep=',',encoding='gb18030')
x = dataset.loc[:,'导弹经度']
y = dataset.loc[:,'导弹纬度']

from scipy.interpolate import make_interp_spline



plt.plot(x,y)
# plt.xlabel('导弹距离(m)')
# plt.ylabel('导弹高度(m)')

plt.show()
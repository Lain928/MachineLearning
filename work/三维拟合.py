from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.interpolate import make_interp_spline
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False


# 列出实验数据
dataset = pd.read_csv('D:/Work/仿真数据/速度2ma/h-25km/SM6防空导弹_6_驱逐舰_2_297km.csv', sep=',',encoding='gb18030')
x = dataset['导弹经度'].values
y = dataset['导弹纬度'].values
z = dataset['导弹高度'].values
rows = dataset.shape[0]

z_smooth = np.linspace(z.min(), z.max(), rows)
y_smooth = make_interp_spline(z, y)(z_smooth) # x->y x对应一个y y可以对应多个x
x_smooth = make_interp_spline(z, x)(z_smooth)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(z_smooth, y_smooth, x_smooth, color='red')
ax.set_title("数据分析")
plt.show()
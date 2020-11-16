import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import random
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
#
# # 读取txt文件数据 以空格分隔 无头标签
# dataset = pd.read_csv('D:/Work/仿真数据/速度2ma/h-25km/SM6防空导弹_6_驱逐舰_2_297km.csv', sep=',',encoding='gb18030')
# x = dataset.loc[:,'导弹经度'].values
# y = dataset.loc[:,'导弹纬度'].values
#
# from scipy.interpolate import make_interp_spline
#
# x_smooth = np.linspace(x.min(), x.max(), 20)
# y_smooth = make_interp_spline(x, y)(x_smooth)

# plt.plot(x,y)
# plt.xlabel('导弹距离(m)')
# plt.ylabel('导弹高度(m)')
# plt.plot(x_smooth, y_smooth)
# plt.show()


#读取csv文件
df = pd.read_csv('D:/Work/仿真数据/速度2ma/h-25km/SM6防空导弹_6_驱逐舰_2_297km.csv', sep=',',encoding='gb18030')

#取表中的第4列的所有值
x=df.iloc[:,4]
y = df.iloc[:,5]


#取表中的第3列的所有值
arrs_x = x.values
arrs_y = y.values

from scipy.interpolate import spline

xnew = np.linspace(arrs_x.min(),arrs_x.max(),20) #300 represents number of points to make between T.min and T.max
ynew = np.linspace(arrs_y.min(),arrs_y.max(),20) #300 represents number of points to make between T.min and T.max
plt.plot(xnew,ynew)
plt.show()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import random
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 读取txt文件数据 以空格分隔 无头标签
dataset = pd.read_csv('C:/Users/Dell/Desktop/200/SM6防空导弹_1_驱逐舰_2_200km.csv', sep=',',encoding='gb18030')
zdata1 = dataset.loc[:,'导弹高度']
misdis = dataset.loc[:,'导弹距离']

plt.plot(misdis,zdata1)
plt.xlabel('导弹距离(m)')
plt.ylabel('导弹高度(m)')

plt.show()
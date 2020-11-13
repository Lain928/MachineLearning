import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import random
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 读取txt文件数据 以空格分隔 无头标签
dataset = pd.read_csv('./SM6防空导弹_1_驱逐舰_2_200km.csv', sep=',',encoding='gb18030')


def tettt(dataset):
    xdata1 = dataset.loc[:,'导弹经度']
    ydata1 = dataset.loc[:,'导弹纬度']
    zdata1 = dataset.loc[:,'导弹高度']
    print(xdata1[1])
    print("导弹飞行的最大高度为：",ydata1.max())

    xdata2 = dataset.loc[:,'目标经度']
    ydata2 = dataset.loc[:,'目标纬度']
    zdata2 = dataset.loc[:,'目标高度']

    rows = dataset.shape[0]
    print(rows)
    # 交汇点 导弹坐标信息
    last_mis_x = xdata1[rows-1]
    last_mis_y = ydata1[rows-1]
    last_mis_z = zdata1[rows-1]
    print(last_mis_x,last_mis_y,last_mis_z)
    # 交汇点 目标坐标信息
    last_tar_x = xdata2[rows-1]
    last_tar_y = ydata2[rows-1]
    last_tar_z = zdata2[rows-1]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xdata1,ydata1,zdata1,"r",label="SM6")
    ax.plot(xdata2,ydata2,zdata2,"g",label="Target")
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    ax.set_zlabel('高度')
    ax.set_title("导弹弹道数据分析")

    # 对交汇点设置文本标注信息
    axis_mis = "mis_lon:"+str(round(last_mis_x / 3.14 * 180,2))+"\n"\
               + "mis_lat:" + str(round(last_mis_y / 3.14 * 180,2))\
               +"\n"+ "mis_alt:"+str(round(last_mis_z))
    ax.text(last_mis_x,last_mis_y,last_mis_z+2000.0,axis_mis,color='blue')
    ax.scatter(last_mis_x,last_tar_y,last_tar_z,marker="v",c="blue")

    axis_tar = "tar_lon:"+str(round(last_tar_x / 3.14 * 180,2))+"\n"\
               + "tar_lat:" + str(round(last_tar_y / 3.14 * 180,2))\
               +"\n"+ "tar_alt:"+str(round(last_tar_z))
    ax.text(last_tar_x,last_tar_y,last_tar_z-6000.0,axis_tar,color='blue')
    ax.scatter(last_tar_x,last_tar_y,last_tar_z,marker="v",c="blue")

    print("目标经度：",dataset.tail(1).loc[:,'目标经度'])
    plt.legend()
    plt.show()

tettt(dataset)
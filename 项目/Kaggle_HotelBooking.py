'''
kaggle 酒单预定需求
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文使用
plt.rcParams['axes.unicode_minus'] = False # 用于显示负号


# 载入数据
data = pd.read_csv('Resources/Kaggle_Hotelbooking/hotel_bookings.csv')
# pandas查看行数和列数 shape[0] 行数 shape[1]列数
print(data.shape)
print(data.info())
# 先查看是否有缺失值
test_data= data.isnull().sum()[data.isnull().sum()!=0]
print(test_data)
# 确定填充规则
# 删除company列
data_new = data.copy(deep = True)
data_new.drop("company", axis=1, inplace=True)
# 其他的填充agent 用0填充 children用众数填充 country用众数填充
data_new["agent"].fillna(0, inplace=True)
data_new["children"].fillna(data_new["children"].mode()[0], inplace=True)
data_new["country"].fillna(data_new["country"].mode()[0], inplace=True)


'''
这里还需要数据异常值的处理：为什么知道这个异常值呢，可以通过后面的计算错误得到这个东西。在后面计算人均价格的时候，如果总人数和为0的情况，则会有异常，所以需要处理异常值
需要对此数据集中异常值为那些总人数（adults+children+babies)为0的记录，同时，因为先前已指名“meal”中“SC”和“Undefined”为同一类别，因此也需要处理一下。
'''
data_new["children"] = data_new["children"].astype(int)
data_new["agent"] = data_new["agent"].astype(int)

data_new["meal"].replace("Undefined", "SC", inplace=True)
# 处理异常值
# 将 变量 adults + children + babies == 0 的数据删除
zero_guests = list(data_new["adults"] +
                  data_new["children"] +
                  data_new["babies"] == 0)
# hb_new.info()
data_new.drop(data_new.index[zero_guests], inplace=True)

# 入住率 和 退房率
def Is_canceled():
    # 数据可视化
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数
    data_new.is_canceled.value_counts().plot(kind='bar')# 柱状图
    plt.title(u"取消预订情况 (1为取消预订)") # 标题
    plt.ylabel(u"酒店数")
    # data_new.is_canceled 是读取数据的is_canceled 列的相关数据
    cancel = data_new.is_canceled.value_counts()
    Sum=cancel.sum()
    count=0
    for i in cancel:   # 显示百分比
        plt.text(count,i+0.5, str('{:.2f}'.format(cancel[count]/Sum *100)) +'%', ha='center') #位置，高度，内容，居中
        count= count + 1
    plt.show()


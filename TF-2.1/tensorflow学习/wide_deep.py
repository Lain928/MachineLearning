'''
实现wide&deep模型

多输入多输出模型
函数式API
子类API
'''
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #取消AVX2的警告

# 绘图
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

# 1 加载数据 房价预测数据集 用于做回归模型
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
# print(housing.data.shape)   # (20640, 8)
# print(housing.target.shape) # (20640, )
print(housing.data[0])

x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state=7)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state=11)
# 归一化操作
scaler = StandardScaler()
x_train_scal = scaler.fit_transform(x_train)
x_valid_scal = scaler.transform(x_valid)
x_test_scal = scaler.transform(x_test)

# 2 模型搭建
# 模型搭建使用函数式API
# wide层：输入后直接输出
# x_train.shape[1:]:表示x_train的第一行有多少列，也就是有每条数据有多少特征
def Create_Model():
    input = tf.keras.layers.Input(shape=x_train.shape[1:])
    # deep层
    hidden1 = tf.keras.layers.Dense(30, activation='relu')(input)
    hidden2 = tf.keras.layers.Dense(30, activation='relu')(hidden1)
    # 用concatenate实现wide层和deep层的拼接
    concat = tf.keras.layers.concatenate([input, hidden2])
    # 输出
    output = tf.keras.layers.Dense(1)(concat)
    # 将model固化下来
    model = tf.keras.models.Model(inputs = [input],
                              outputs = [output])
    return model


# 2 子类API
class WideDeepModel(tf.keras.models.Model):
    def __init__(self):
        super(WideDeepModel, self).__init__()
        """定义模型层次"""
        self.hidden1_layer = tf.keras.layers.Dense(30, activation='relu')
        self.hidden2_layer = tf.keras.layers.Dense(30, activation='relu')
        self.output_layer = tf.keras.layers.Dense(30)
    def call(self, intput):
        """完成模型的正向运算"""
        hidden1 = self.hidden1_layer(input)
        hidden2 = self.hidden2_layer(hidden1)
        concat = tf.keras.layers.concatenate([input, hidden2])
        output = self.output_layer(concat)
        return output

# 两种实例化模型的方式
# model = tf.keras.models.Sequential([WideDeepModel(),]) # 将WideDeepModel看作一个层
# model = WideDeepModel()
# model.build(input_shape=(None, 8))

# model = Create_Model()
# 2 多输入多输出模型的搭建
def Moreinput_MoreOutput_Model():
    input_wide = tf.keras.layers.Input(shape=[5])
    input_deep = tf.keras.layers.Input(shape=[6])
    # deep层
    hidden1 = tf.keras.layers.Dense(30, activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Dense(30, activation='relu')(hidden1)
    # 用concatenate实现wide层和deep层的拼接
    concat = tf.keras.layers.concatenate([input_wide, hidden2])
    # 输出
    output = tf.keras.layers.Dense(1)(concat)
    output2 = tf.keras.layers.Dense(1)(hidden2)
    # 将model固化
    model = tf.keras.models.Model(inputs = [input_wide, input_deep],
                              outputs = [output, output2])
    return model

model = Moreinput_MoreOutput_Model()
# 显示模型状态
# model.summary()
# 3 模型编译
model.compile(optimizer='adam', loss='mse', metrics=['acc'])

# 4 训练模型
# histroy = model.fit(x_train_scal, y_train, epochs=10, validation_data=(x_valid_scal, y_valid))
# plot_learning_curves(histroy)

# 整理数据
x_train_scal_wide = x_train_scal[:, :5]  # 前五个特征作为第一个输入 一共有八个特征
x_train_scal_deep = x_train_scal[:, 2:]  # 后六个特征作为第二个输入
x_valid_scal_wide = x_valid_scal[:, :5]
x_valid_scal_deep = x_valid_scal[:, 2:]
x_test_scal_wide = x_test_scal[:, :5]
x_test_scal_deep = x_test_scal[:, 2:]

# validation_data：形式为（X，y）的tuple，是指定的验证集
histroy_Moreinput_More_ouput = model.fit([x_train_scal_wide, x_train_scal_deep],[y_train, y_train], epochs=1,
                                         validation_data=([x_valid_scal_wide, x_valid_scal_deep], [y_valid, y_valid]))
# 5 多输入的评价模型
sorces = model.evaluate([x_test_scal_wide, x_test_scal_deep], [y_test, y_test])
print(sorces)
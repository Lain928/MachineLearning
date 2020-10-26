# 知识点总结
# 知识点：
#     1.pd中 index_col 的作用
#     pd.read_csv(r"Resources\Narrativedata.csv",index_col=0)
#     2.归一化
#     3.标准化
#     4.缺失值处理

import pandas as pd

import numpy as np
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
# print(data)
# print(np.array(data).max(axis=0))
#不太熟悉numpy的小伙伴，能够判断data的结构吗？

'''
##################
无量纲化
#1 实现归一化 y = (x - min) / (max - min)
##################
'''
from sklearn.preprocessing import MinMaxScaler
#实例化
scaler = MinMaxScaler()
scaler = scaler.fit(data)
result = scaler.transform(data)
# result_ = scaler.fit_transform(data) #训练和导出结果一步达成
# scaler.inverse_transform(result)#将归一化后的结果逆转

# 使用参数输出特定范围内的归一化数据
scaler = MinMaxScaler(feature_range=[5,10]) # 生成5-10 之间的数

# 当数据量太大时使用fit会报错，使用partial_fit
scaler = scaler.partial_fit(data)

'''
# numpy 实现归一化
import numpy as np
X = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
#归一化
# axis=0 代表对列进行求取最大值 array下进行求取
X_nor = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#逆转归一化
X_returned = X_nor * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0)
'''



'''
##################
2 实现标准化   均值和方差
##################
'''
from sklearn.preprocessing import StandardScaler
#实例化
scaler = StandardScaler()
scaler.fit(data)

#fit，本质是生成均值和方差
#查看均值的属性mean_
#查看方差的属性var_
# print(scaler.mean_)
# print(scaler.var_)

x_std = scaler.transform(data)
#通过接口导出结果
#导出的结果是一个数组，用mean()查看均值
#用std()查看方差
# x_std.mean()
# x_std.std()
#使用fit_transform(data)一步达成结果
scaler.fit_transform(data)
#使用inverse_transform逆转标准化
scaler.inverse_transform(x_std)


'''
##################
3 缺失值
使用泰坦尼克号数据集
##################
'''
from sklearn.impute import SimpleImputer
data = pd.read_csv(r"../Resources/Narrativedata.csv", index_col=0)
# print(data.info()) #代表统计的信息值 缺失值不进行统计
# print(data.shape[0])

#sklearn当中特征矩阵必须是二维
Age = data.loc[:,"Age"].values.reshape(-1,1) # 去除年龄列 并将数据变为二维
# print(Age[:20])

#实例化
# 1 默认均值填补
imp_mean = SimpleImputer()
# 2 用中位数填补
imp_median = SimpleImputer(strategy="median")
# 3 用0填补
# imp_0 = SimpleImputer(strategy="constant",fill_value=0)
# imp_mean = imp_mean.fit_transform(Age)
imp_median = imp_median.fit_transform(Age)
# print(imp_median)
# 填充
#在这里我们使用中位数填补Age
data.loc[:,"Age"] = imp_median
# print(data.loc[:,"Age"])


"""
# 使用pandas和numpy填充数据

import pandas as pd
data = pd.read_csv(r"Narrativedata.csv",index_col=0)
data.head()
data.loc[:,"Age"] = data.loc[:,"Age"].fillna(data.loc[:,"Age"].median())
#.fillna 在DataFrame里面直接进行填补
#.dropna(axis=0)删除所有有缺失值的行，.dropna(axis=1)删除所有有缺失值的列
#参数inplace，为True表示在原数据集上进行修改，为False表示生成一个复制对象，不修改原数据，默认False
data.dropna(axis=0,inplace=True)
"""


"""
4 处理分类型数据
    编码和哑变量
    将文字转换为数值
"""
from sklearn.preprocessing import OneHotEncoder # 独热编码
from sklearn.preprocessing import OrdinalEncoder # 特征专用
from sklearn.preprocessing import LabelEncoder #标签专用
# 导入数据
y = data.iloc[:, -1]
# 要输入的是标签，不是特征矩阵，所以允许一维
# 实例化
le = LabelEncoder()
le = le.fit(y)
label = le.transform(y)
# transform接口调取结果
# 属性.classes_查看标签中究竟有多少类别
# le.classes_ # 类别
# print(le.classes_)
# print(label)

le.fit_transform(y)
# 也可以直接fit_transform一步到位
# 使用inverse_transform可以逆转
le.inverse_transform(label)

# 让标签等于我们运行出来的结果
data.iloc[:, -1] = label

# # 如果不需要教学展示的话我会这么写：
# from sklearn.preprocessing import LabelEncoder
#
# data.iloc[:, -1] = LabelEncoder().fit_transform(data.iloc[:, -1])



'''
5 连续值的处理
    1 二值化
    2 分段 （分箱操作）
'''

data_2 = data.copy()
from sklearn.preprocessing import Binarizer #二值化
print(data_2.info())
X = data_2.iloc[:,0].values.reshape(-1,1)
#类为特征专用，所以不能使用一维数组
transformer = Binarizer(threshold=30).fit_transform(X)

# 分段 / 分箱操作
from sklearn.preprocessing import KBinsDiscretizer
X = data.iloc[:,0].values.reshape(-1,1)
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
est.fit_transform(X)
#查看转换后分的箱：变成了一列中的三箱
# set表示集合，revel表示降维，做集合处理时，不接受二维
set(est.fit_transform(X).ravel())
# 渡人编码形式的分箱操作
est = KBinsDiscretizer(n_bins=3, encode='onehot', strategy='uniform')
#查看转换后分的箱：变成了哑变量
est.fit_transform(X).toarray()










from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.datasets import fetch_california_housing as fch #加利福尼亚房屋价值数据集

housevalue = fch()
X = pd.DataFrame(housevalue.data) #放入DataFrame中便于查看
y = housevalue.target
print(X.shape)
print(X.head())
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
# 对训练集和测试集从新进行标号
for i in [Xtrain, Xtest]:
    i.index = range(i.shape[0])

# 构建模型
reg = LR().fit(Xtrain, Ytrain)
yhat = reg.predict(Xtest)
print(yhat)
'''
coef_:
数组，形状为 (n_features, )或者(n_targets, n_features)
线性回归方程中估计出的系数。如果在ﬁt中传递多个标签（当y为二维或以上的时候），则返
回的系数是形状为（n_targets，n_features）的二维数组，而如果仅传递一个标签，则返回
的系数是长度为n_features的一维数组
intercept_
数组，线性回归中的截距项。
'''

# 多元线性回归的模型评价指标
# 1 是否能够正确预测数值
from sklearn.metrics import mean_squared_error as MSE
print(MSE(yhat,Ytest))
# neg_mean_squared_error 均方值误差 计算出来的均方误差为负
mse = cross_val_score(reg,X,y,cv=10,scoring="neg_mean_squared_error")
print(mse)

# 2 是否拟合了足够的信息 可解释性方差分数
#调用R2
from sklearn.metrics import r2_score
r2_score(yhat,Ytest)
r2 = reg.score(Xtest,Ytest)
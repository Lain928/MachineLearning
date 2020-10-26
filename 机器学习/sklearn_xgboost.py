# -*- coding: utf-8 -*-
### load module
from sklearn import datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

### 1 加载数据load datasets
# 手写数字识别数据集
digits = datasets.load_digits()

### 2 数据分析 data analysis
print(digits.data.shape)   # 输入空间维度
print(digits.target.shape) # 输出空间维度

### 3 数据集切分data split
x_train,x_test,y_train,y_test = train_test_split(digits.data,
                                                 digits.target,
                                                 test_size = 0.3,
                                                 random_state = 33)
### 4 训练模型 fit model for train data
### 模型的参数设置
model = XGBClassifier()
model.fit(x_train,y_train)

### 5 测试集预测 make prediction for test data
y_pred = model.predict(x_test)

### 6 模型评价 model evaluate 使用sklearn自带的评价效果估计准确率
accuracy = accuracy_score(y_test,y_pred)
print("accuarcy: %.2f%%" % (accuracy*100.0))

### 7 可视化结果 用于分类的
### 特征重要性 特征值越大 表明该特征越重要
import matplotlib.pyplot as plt
from xgboost import plot_importance
fig,ax = plt.subplots(figsize=(10,15))
plot_importance(model,height=0.5,max_num_features=64,ax=ax)
plt.show()
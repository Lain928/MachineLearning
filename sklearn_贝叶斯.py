import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import datasets

# 贝叶斯鸢尾花数据集预测
def iristest():
    # 载入数据集
    iris = datasets.load_iris()
    iris_data = iris['data']
    iris_label = iris['target']
    iris_target_name = iris['target_names']
    X = np.array(iris_data)
    Y = np.array(iris_label)
    # 数据集划分
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=420)

    # 训练
    gnb = GaussianNB().fit(Xtrain, Ytrain)
    # 查看分数
    acc_score = gnb.score(Xtest, Ytest)
    print(acc_score)

    # 这里预测当前输入的值的所属分类
    print('类别是', iris_target_name[gnb.predict([[7, 7, 7, 7]])[0]])
iristest()



# digits = load_digits()
# X, y = digits.data, digits.target
#
# #数据集划分
# Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
# # 2. 建模，探索建模结果
# gnb = GaussianNB().fit(Xtrain,Ytrain)
# #查看分数
# acc_score = gnb.score(Xtest,Ytest)
# print(acc_score)
# #查看预测结果n
# Y_pred = gnb.predict(Xtest)
#
# #查看预测的概率结果
# prob = gnb.predict_proba(Xtest)
#
# # 3. 使用混淆矩阵来查看贝叶斯的分类结果
# from sklearn.metrics import confusion_matrix as CM
#
# classresult = CM(Ytest,Y_pred)
# print(classresult)
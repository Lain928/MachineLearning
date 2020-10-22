'''
装袋法 bagging
随机森林

知识点：
    1. sklearn中的model_selection选择数据切分方式
            train_test_split 切分数据
            参数：
                数据，标签，切分概率
                train_test_split(wine.data, wine.target, test_size=0.3)
            cross_val_score 交叉验证
            参数：
            rfc = RandomForestClassifier(n_estimators=25)
            rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10)
            rfc：使用模型，wine.data:数据集的数据，  wine.target：数据集的标签 cv=10： 交叉验证迭代次数
    2. .mean()求取平均值
    3. 网格搜索？？？ 实现多个参数一起调
    4. 输出表示中的format使用
        print("Single Tree:{}".format(score_c))
        占位符的使用：print("{}{}{}".format(0，1，2))

        In [1]: '{0},{1}'.format('kzc',18)
        Out[1]: 'kzc,18'
参数：
    n_estimators  随机森林中树的个数（默认为10 0.22版本后默认是100）

'''
#  AdaBoostClassifier adboost
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine, load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV# 数据集划分 交叉验证
import numpy as np

def dessiontree():
    # 2.导入需要的数据集
    wine = load_wine()
    # 划分数据集
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)

    # 树的实例化
    # 决策树
    clf = DecisionTreeClassifier(random_state=0)
    clf = clf.fit(Xtrain, Ytrain)
    score_c = clf.score(Xtest, Ytest)
    # 随机森林
    rfc = RandomForestClassifier(random_state=0)
    rfc = rfc.fit(Xtrain, Ytrain)
    score_r = rfc.score(Xtest, Ytest)
    print("Single Tree:{}".format(score_c)
          , "Random Forest:{}".format(score_r)
          )
    # 4.画出随机森林和决策树在十组交叉验证下的效果对比
    # 带大家复习一下交叉验证
    # 交叉验证：是数据集划分为n分，依次取每一份做测试集，每n-1份做训练集，多次训练模型以观测模型稳定性的方法
    rfc_l = []
    clf_l = []
    for i in range(10): #运行10次
        # 随机森林
        rfc = RandomForestClassifier(n_estimators=25)
        rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10).mean() # 10次交叉验证求取平均值
        rfc_l.append(rfc_s)

        # 决策树
        clf = DecisionTreeClassifier()
        clf_s = cross_val_score(clf, wine.data, wine.target, cv=10).mean()
        clf_l.append(clf_s)

    plt.plot(range(1, 11), rfc_l, label="Random Forest")
    plt.plot(range(1, 11), clf_l, label="Decision Tree")
    plt.legend()
    plt.show()


def tiaocan_test():
    data = load_breast_cancer()
    # sklearn自带数据据的使用规范，读取相关数据
    rfc = RandomForestClassifier(n_estimators=100, random_state=90)
    score_pre = cross_val_score(rfc, data.data, data.target, cv=10).mean()
    print(score_pre)

    # 调参
    """
    在这里我们选择学习曲线，可以使用网格搜索吗？可以，但是只有学习曲线，才能看见趋势
    我个人的倾向是，要看见n_estimators在什么取值开始变得平稳，是否一直推动模型整体准确率的上升等信息
    第一次的学习曲线，可以先用来帮助我们划定范围，我们取每十个数作为一个阶段，来观察n_estimators的变化如何
    引起模型整体准确率的变化
    """
    # # 决策树个数
    # scorel = []
    # for i in range(0, 200, 10):
    #     rfc = RandomForestClassifier(n_estimators=i + 1,
    #                                  n_jobs=-1,
    #                                  random_state=90)
    #     score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
    #     scorel.append(score)
    # print(max(scorel), (scorel.index(max(scorel)) * 10) + 1)
    # plt.figure(figsize=[20, 5])
    # plt.plot(range(1, 201, 10), scorel)
    # plt.show()


    # list.index([object])
    # 返回这个object在列表list中的索引

    # 网格搜素
    # 调整max_depth
    param_grid = {'max_depth': np.arange(1, 20, 1)}
    # 一般根据数据的大小来进行一个试探，乳腺癌数据很小，所以可以采用1~10，或者1~20这样的试探
    # 但对于像digit recognition那样的大型数据来说，我们应该尝试30~50层深度（或许还不足够
    # 更应该画出学习曲线，来观察深度对模型的影响
    rfc = RandomForestClassifier(n_estimators=39
                                 , random_state=90
                                 )
    GS = GridSearchCV(rfc, param_grid, cv=10)
    GS.fit(data.data, data.target)
    print(GS.best_params_)
    print(GS.best_score_)


tiaocan_test()




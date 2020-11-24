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
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score # 数据集划分 交叉验证

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




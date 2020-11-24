'''
步骤：
    1，打印加载的数据集的信息 info 查看现在相关的数据集
    2，进行数据的预处理（缺失值、归一化、标准化）

'''

import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

data_train = pd.read_csv("Resources/Kaggle_Titanic/Kaggle_Titanic_train.csv")
data_test = pd.read_csv("Resources/Kaggle_Titanic/Kaggle_Titanic_test.csv")
# print(data_test.info())
# print(data_train.info())
# print(data_train.columns)
# print(data_test.columns)
# print(data_train.head()) # 默认输出前五行数据
# print(data_train.columns) # 输出数据集的列标签
# print(data_train.shape) # 输出数据的行和列shape[0] 行 / shape[1] 列
# print(data_train.info()) # 每一列类别的个数（无缺失值的个数）
# print(data_train.describe()) # 每一列的数据信息


# 在数据集中筛选中需要的类别
# df = data_train[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
# print(df)

# def DealTest(data_test, rfr, ):
#     # data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
#     # # 接着我们对test_data做和train_data中一致的特征变换
#     # # 首先用同样的RandomForestRegressor模型填上丢失的年龄
#     # tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
#     # null_age = tmp_df[data_test.Age.isnull()].as_matrix()
#     # # 根据特征属性X预测年龄并补上
#     # X = null_age[:, 1:]
#     # predictedAges = rfr.predict(X)
#     # data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges
#     #
#     # data_test = set_Cabin_type(data_test)
#     # dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
#     # dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
#     # dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
#     # dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')
#     #
#     # df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
#     # df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
#     # df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'], age_scale_param)
#     # df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'], fare_scale_param)
#     return df_test


'''
数据预处理
特征工程
'''
# 使用随机森林来拟合缺失值的年龄数据
from sklearn.ensemble import RandomForestRegressor
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']] #提取出多个类别列
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values
    # print(known_age)
    # print(unknown_age)
    # y即目标年龄
    y = known_age[:, 0] # 第0列 表示年龄列 代表标签
    # X即特征属性值
    X = known_age[:, 1:] # 第一列之后的数据为数据列
    '''
    在随机森林上进行训练
    sklearn中三部曲
    1 加载模型
    2 fit训练模型
    3 score预测模型准确度 / predect预测模型
    '''
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1:]) # 用剩下的数据进行预测，年龄看作预测的标签
    # print(predictedAges)
    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges
    return df
    # return df, rfr # 当返回多个值时返回的形式为元组 （df, rfr）

data_new = set_missing_ages(data_train)


# cabin类别填充为 为空则no（为存活） 不为空则yes
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

data_new = set_Cabin_type(data_new)



# data_train, rfr = set_missing_ages(data_train)
# data_train = set_Cabin_type(data_train)

'''
在进行逻辑回归建模时，采用数值型特征，所以应该将不为数值型的数据变为数值型
### 采用特征因子化
以Cabin为例，原本一个属性维度，因为其取值可以是[‘yes’,‘no’]，而将其平展开为’Cabin_yes’,'Cabin_no’两个属性

原本Cabin取值为yes的，在此处的"Cabin_yes"下取值为1，在"Cabin_no"下取值为0
原本Cabin取值为no的，在此处的"Cabin_yes"下取值为0，在"Cabin_no"下取值为1
'''
'''
对非数值型的数据进行数值化，哑变量（[0 1 0 ]）
'''
def FatureToNum(data_train):
    # 对各个非数值型的类别进行数值化处理
    # 使用pd的函数进行数值化
    dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
    # 采用拷贝的形式，保存原有的数据集
    # 将处理之后的值添加到数据集中 列添加
    df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    # 将原有的数据集中的非数值型类别删除
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    return df

data_new = FatureToNum(data_new)

'''
归一化 / 标准化
对数值变化较大的类别，我们采用，归一化或者标准化来处理
各属性值之间scale差距太大，将对收敛速度造成几万点伤害值！甚至不收敛
'''

import sklearn.preprocessing as preprocessing
def Standard(df):
    scaler = preprocessing.StandardScaler() # 加载模型
    df_age = df['Age'].values.reshape(-1, 1) #使用sklearn库进行标准化时应将标准化的列 从一维转换成二维
    df_Fare = df['Fare'].values.reshape(-1, 1)
    # 训练转化一步搞定
    df['Age_scaled'] = scaler.fit_transform(df_age)
    df['Fare_scaled'] = scaler.fit_transform(df_Fare)
    df.drop(['Age', 'Fare'], axis=1, inplace=True) # 删除年龄和船票列
    return df
data_new = Standard(data_new)

'''
使用逻辑回归进行结果预测
把需要的feature字段取出来，转成numpy格式
使用scikit-learn中的LogisticRegression建模。
'''
from sklearn.linear_model import LogisticRegression
def MyLogisticRegression(df):
    # 用正则取出我们要的属性值
    # train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    # train_np = train_df.values
    # y即Survival结果
    y = df.iloc[:, 0].values
    # X即特征属性值
    X = df.iloc[:, 1:].values
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf
clf = MyLogisticRegression(data_new)

# 处理测试数据
def DealTest(data_test):
    # print(data_test)
    data_test_new = data_test.copy()
    # 打印test数据集信息时发现船票有缺失值 进行填补
    data_test_new.loc[(data_test.Fare.isnull()), 'Fare'] = 0
    data_test_new = set_missing_ages(data_test_new)
    data_test_new = set_Cabin_type(data_test_new)
    data_test_new = FatureToNum(data_test_new)
    data_test_new = Standard(data_test_new)
    return data_test_new

# 进行预测 并输出文件， 形式为csv
def Pridect(test, clf):
    predictions = clf.predict(test)
    result = pd.DataFrame({'PassengerId': data_test['PassengerId'].values, 'Survived': predictions.astype(np.int32)})
    result.to_csv("Resources/Kaggle_Titanic/logistic_regression_predictions.csv", index=False)

data_test_new = DealTest(data_test)
Pridect(data_test_new, clf)

'''
系统性能的优化
1，模型系数关联分析
    是否欠拟合或者过拟合
    首先，大家回去前两篇文章里瞄一眼公式就知道，这些系数为正的特征，和最后结果是一个正相关，反之为负相关
2，交叉验证 cross_val_score
3，特征工程 
    我们可以打印出分错的数据集，观察其中有什么联系，以此我们可以构造更多的特征来优化我们的模型
4，模型融合 （可以缓解过拟合的问题）
    也就是说的 集成学习 （bagging boosting）
'''

'''
对数据的认识
看看各个属性和多个属性对最后是否能够存活有什么关系

'''
def Drawing(data_train):
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
    data_train.Survived.value_counts().plot(kind='bar')# 柱状图
    plt.title(u"获救情况 (1为获救)") # 标题
    plt.ylabel(u"人数")

    plt.subplot2grid((2,3),(0,1))
    data_train.Pclass.value_counts().plot(kind="bar")
    plt.ylabel(u"人数")
    plt.title(u"乘客等级分布")

    plt.subplot2grid((2,3),(0,2))
    plt.scatter(data_train.Survived, data_train.Age)
    plt.ylabel(u"年龄")                         # 设定纵坐标名称
    plt.grid(b=True, which='major', axis='y')
    plt.title(u"按年龄看获救分布 (1为获救)")


    plt.subplot2grid((2,3),(1,0), colspan=2)
    data_train.Age[data_train.Pclass == 1].plot(kind='kde')
    data_train.Age[data_train.Pclass == 2].plot(kind='kde')
    data_train.Age[data_train.Pclass == 3].plot(kind='kde')
    plt.xlabel(u"年龄")# plots an axis lable
    plt.ylabel(u"密度")
    plt.title(u"各等级的乘客年龄分布")
    plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.


    plt.subplot2grid((2,3),(1,2))
    data_train.Embarked.value_counts().plot(kind='bar')
    plt.title(u"各登船口岸上船人数")
    plt.ylabel(u"人数")
    plt.show()


def DrawingFormLevel(data_train):
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title(u"各乘客等级的获救情况")
    plt.xlabel(u"乘客等级")
    plt.ylabel(u"人数")
    plt.show()
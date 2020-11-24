import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

'''
交叉验证的作用用于测试模型 用于验证模型的好坏，同时针对性选择模型
'''

# 1 加载数据
train_df = pd.read_csv('Resources/Kaggle_HousePrices/train.csv', index_col=0)
test_df = pd.read_csv('Resources/Kaggle_HousePrices/test.csv', index_col=0)
# print(train_df.head())
# print(train_df.shape)
# print(train_df.columns) # 打印出所有的列名

# 展示当前价格的分布情况
def showHist(train_df):
    # 2 数据平滑及合并
    prices = pd.DataFrame({'price': train_df['SalePrice'], 'log(price+1)': np.log1p(train_df['SalePrice'])})
    prices.hist()  # 画图 看一下标签是否平滑 python的可视化，直方图hist
    plt.show()


# 将训练集的数据删除标签列，并将训练集和测试集进行合并，然后做统一的预处理，与处理完成后再进行分开
y_train = np.log1p(train_df.pop('SalePrice'))  # 将原来的标签删除 剩下log(price+1)列的数据
all_df = pd.concat((train_df, test_df), axis=0)  # 将train_df, test_df合并 按行合并

# 3 特征工程
# 有些数据的取值只有可数个。这类数据我们转为one_hot编码
# 发现MSSubClass值应该是分类值
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)

# 我们将category的变量转变为numerical表达形式
# 当我们用numerical来表达categorical的时候，要注意，数字本身有大小。
# 不能乱指定大小，我们采用one_hot编码
# pandas自带的get_dummies方法可以一键做到one_hot
# 此刻MSSubClass被我们分成了12个column，每一个代表一个category。是就是1，不是就是0
pd.get_dummies(all_df['MSSubClass'], prefix='MSSubClass')
# 同理,我们把所有的category数据都转化为One_hot
all_dummy_df = pd.get_dummies(all_df)

# 缺失值处理
# 统计每列缺失值情况
# print(all_dummy_df.isnull().sum().sort_values(ascending=False).head())

# 我们用均值填充
mean_cols = all_dummy_df.mean()
all_dummy_df = all_dummy_df.fillna(mean_cols)
# 再检查一下是否有缺失值
# print(all_dummy_df.isnull().sum().sum())  # 0

# 先找出数字型数据
numeric_cols = all_df.columns[all_df.dtypes != 'object']
# 对其标准化
numeric_col_mean = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols]-numeric_col_mean) / numeric_col_std

# 将合并的数据此时进行拆分  分为训练数据和测试数据
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]

X_train = dummy_train_df.values
X_test = dummy_test_df.values
# 4 模型训练
from sklearn.model_selection import cross_val_score

# 模型1 岭回归
def Ridge_Regression(X_train, y_train):
    from sklearn.linear_model import Ridge
    # 用sklearn自带的交叉验证方法来测试模型：
    # 创建等比数列 开始为10的-3方幂 结束为10的2次方幂 个数为50个
    alphas = np.logspace(-3, 2, 50)
    test_scores = []
    for alpha in alphas:
        clf = Ridge(alpha)
        # cv：选择每次测试折数  scoring: 评价指标是什么 （accuracy：评价指标是准确度）（neg_mean_squared_error：评价指标是损失函数）
        # test_score 存储的为十次交叉验证的结果 test_score.mean()是对这十次的结果求平均值 然后存储起来
        test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
        test_scores.append(np.mean(test_score))

    # 看哪个alpha下 模型预测的更好
    plt.plot(alphas, test_scores)
    plt.title('Alpha vs CV Error')
    plt.show()
# Ridge_Regression(X_train, y_train)

# 模型2 随机森林
def RandomforestRegression(X_train, y_train):
    from sklearn.ensemble import RandomForestRegressor

    max_features = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    test_scores = []
    for max_feat in max_features:
        clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
        test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
        test_scores.append(np.mean(test_score))
    plt.plot(max_features, test_scores)
    plt.title('Max Features vs CV Error')
    plt.show()
# RandomforestRegression(X_train, y_train)
# 模型3 xgboost
def XgboostTest(X_train, y_train):
    from xgboost import XGBRegressor
    # 用sklearn自带的cross validation方法来测试模型
    params = [1, 2, 3, 4, 5, 6]
    test_scores = []
    for param in params:
        clf = XGBRegressor(max_depth=param)
        test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
        test_scores.append(np.mean(test_score))
    plt.plot(params, test_scores)
    plt.title("max_depth vs CV Error")
    plt.show()
# XgboostTest(X_train, y_train)
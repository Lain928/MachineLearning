import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

# 1 加载数据
train_df = pd.read_csv('Resources/Kaggle_HousePrices/train.csv', index_col=0)
test_df = pd.read_csv('Resources/Kaggle_HousePrices/test.csv', index_col=0)
print(train_df.head())
print(train_df.shape)


def showHist():
    # 2 数据平滑及合并
    '''
    1 画图如何去画
    画出房价图，可以看出房价的分布集中再某一个区域，为了使最后的结果更为准确
    对价格进行平滑处理， log（price + 1）
    log1p(x) 对应log(x + 1)
    log1p 对应的反运算expm1()
    '''
    prices = pd.DataFrame({'price': train_df['SalePrice'], 'log(price+1)': np.log1p(train_df['SalePrice'])})
    prices.hist()  # 画图 看一下标签是否平滑 python的可视化，直方图hist
    plt.show()

# 将训练集的数据删除标签列，并将训练集和测试集进行合并，然后做统一的预处理，与处理完成后再进行分开
y_train = np.log1p(train_df.pop('SalePrice'))  # 将原来的标签删除 剩下log(price+1)列的数据
all_df = pd.concat((train_df, test_df), axis=0)  # 将train_df, test_df合并


# 3 特征工程
'''
1 缺失值的处理
2 变量类型的变化（如MSSubClass是一个类别型而不是数值型，用数字表示
    在用pandas处理时，将被当作数字处理，导致训练不准确，所以应该修改为string型）
    把它变回string类型之后，我们的工作并没有完成。
    机器学习的模型处理类别型数据还是比较麻烦，我们还得将它变成数值型变量。
    通过One-Hot（独热码）编码进行数据变换。
    在pandas中，可以使用get_dummies实现这一转换
3 ont-hot编码 数的取值有数个时可以考虑独热编码
4 数据标准化处理
'''

# 有些数据的取值只有四五，或者可数个。这类数据我们转为one_hot编码
# 发现MSSubClass值应该是分类值
print(all_df['MSSubClass'].dtypes)  # int64
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
print(all_df['MSSubClass'].value_counts())

# 我们将category的变量转变为numerical表达形式
# 当我们用numerical来表达categorical的时候，要注意，数字本身有大小。
# 不能乱指定大小，我们采用one_hot编码
# pandas自带的get_dummies方法可以一键做到one_hot
print(pd.get_dummies(all_df['MSSubClass'], prefix='MSSubClass').head())
# 此刻MSSubClass被我们分成了12个column，每一个代表一个category。是就是1，不是就是0

# 同理,我们把所有的category数据都转化为One_hot
all_dummy_df = pd.get_dummies(all_df)

# 缺失值处理
# 统计每列缺失值情况
print(all_dummy_df.isnull().sum().sort_values(ascending=False).head())

# 我们用均值填充
mean_cols = all_dummy_df.mean()
all_dummy_df = all_dummy_df.fillna(mean_cols)
# 再检查一下是否有缺失值
print(all_dummy_df.isnull().sum().sum())  # 0

# 先找出数字型数据
numeric_cols = all_df.columns[all_df.dtypes != 'object']
print(numeric_cols)

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
def Ridge_Regression():
    from sklearn.linear_model import Ridge
    # 用sklearn自带的交叉验证方法来测试模型：
    alphas = np.logspace(-3, 2, 50)
    test_scores = []
    for alpha in alphas:
        clf = Ridge(alpha)
        test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
        test_scores.append(np.mean(test_score))

    # 看那个alpha下 模型预测的更好
    plt.plot(alphas, test_scores)
    plt.title('Alpha vs CV Error')
    plt.show()
Ridge_Regression()

# 模型2 随机森林
def RandomforestRegression(dummy_train_df, dummy_test_df):
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

# 模型3 xgboost
def XgboostTest():
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
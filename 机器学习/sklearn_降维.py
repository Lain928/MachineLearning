import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


'''
    pca中参数设置实现svd
    svd_solver的4个选项：
    “auto” 通常选用
    “full”
    “arpack” 适合特征矩阵稀疏
    “randomized” 适合特征矩阵巨大，计算量大
'''

# 2. 提取数据集
iris = load_iris()
y = iris.target
X = iris.data
print(iris.target_names)
# data = pd.DataFrame(X)
# print(data.head())

# 3 建模
# 调用PCA
# 实例化
# 拟合模型
# 获取新矩阵
pca = PCA(n_components=2)# 降为2维
pca = pca.fit(X)
X_dr = pca.transform(X)
# 也可以fit_transform一步到位
# X_dr = PCA(2).fit_transform(X)


# 4.可视化
# 要将三种鸢尾花的数据分布显示在二维平面坐标系中，对应的两个坐标（两个特征向量）应该是三种鸢尾花降维后的
# x1和x2，怎样才能取出三种鸢尾花下不同的x1和x2呢？
# X_dr[y == 0, 0]  # 这里是布尔索引，看出来了么？
# 要展示三中分类的分布，需要对三种鸢尾花分别绘图
# 可以写成三行代码，也可以写成for循环

colors = ['red', 'black', 'orange']

plt.figure()
for i in [0, 1, 2]: #将不同的类别标记维 0 1 2
    plt.scatter(X_dr[y == i, 0]
                , X_dr[y == i, 1] #使用索引值进行匹配
                , alpha=.7
                , c=colors[i]
                , label=iris.target_names[i] # iris.target_names存放着标签名称
                )
plt.legend()
plt.title('PCA of IRIS dataset')
# x y轴的名称
# plt.xlabel("number of components after dimension reduction")
# plt.ylabel("cumulative explained variance")
plt.show()

# # 6.探索降维后的数据
# # 属性explained_variance，查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
# pca.explained_variance_
# # 属性explained_variance_ratio，查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比
# # 又叫做可解释方差贡献率
# pca.explained_variance_ratio_
# # 大部分信息都被有效地集中在了第一个特征上
# pca.explained_variance_ratio_.sum()




# pca_line = PCA().fit(X)
# plt.plot([1, 2, 3, 4], np.cumsum(pca_line.explained_variance_ratio_))
# plt.xticks([1, 2, 3, 4])  # 这是为了限制坐标轴显示为整数
# plt.xlabel("number of components after dimension reduction")
# plt.ylabel("cumulative explained variance")
# plt.show()

# 使用最大似然估计进行降维参数选择
pca_mle = PCA(n_components="mle")
pca_mle = pca_mle.fit(X)
X_mle = pca_mle.transform(X)

print(X_mle)
#可以发现，mle为我们自动选择了3个特征

pca_mle.explained_variance_ratio_.sum()
#得到了比设定2个特征时更高的信息含量

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

# 用人来能识别看降维后数据保存量 从而确定降维维数
def test1():
    faces = fetch_lfw_people(min_faces_per_person=60)
    X = faces.data
    pca = PCA(150)#实例化
    X_dr = pca.fit_transform(X)
    print(X_dr.shape)
    X_inverse = pca.inverse_transform(X_dr)
    print(X_inverse.shape)

    #可视化
    fig,ax = plt.subplots(2,10,figsize=(10,2.5),subplot_kw={"xticks":[],"yticks":[]})

    for i in range(10):
        ax[0,i].imshow(faces.images[i,:,:],cmap="binary_r")
        ax[1,i].imshow(X_inverse[i].reshape(62,47),cmap="binary_r")


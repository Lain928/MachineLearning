from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


# kmeans聚类
def kMeansClustering():
    # 自己创建数据集 有数据 有标签 x代表数据 y代表标签
    X, y = make_blobs(n_samples=500,n_features=2,centers=4,random_state=1)
    color = ["red","pink","orange","gray"] # 颜色

    # 使用k_means聚类
    from sklearn.cluster import KMeans
    # k为3类
    n_clusters = 3
    # 训练好的模型
    cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    # 训练出来的模型 都可以查看哪些参数如何看？？？
    #分出的簇的标签
    y_pred = cluster.labels_
    # 簇的质心
    centroid = cluster.cluster_centers_
    # 总距离的平方和
    inertia = cluster.inertia_

    # 建立子图 1 代表一个子图
    # 绘图的方式
    fig, ax1 = plt.subplots(1)
    for i in range(n_clusters):  # 画多少类
        ax1.scatter(X[y_pred == i, 0], X[y_pred == i, 1] # 画点
                    , marker='o'
                    , s=8
                    , c=color[i]
                    )
    ax1.scatter(centroid[:, 0], centroid[:, 1] # 画质心
                , marker="x"
                , s=15
                , c="black")
    plt.show()
    n_clusters = 4
    cluster_ = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    inertia_ = cluster_.inertia_


    # 聚类算法模型的评估指标
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import silhouette_samples

    silhouette_score(X,y_pred)
    silhouette_samples(X,y_pred)

    # 基于轮廓系数选取k值

'''
密度聚类 DBSCAN
'''
from sklearn.datasets import make_circles #使用sklearn创建环形数据
import time
from sklearn.cluster import DBSCAN
def DensityClustering():
    fig, ax = plt.subplots(1, 2) # 创建一个一行两列的图
    axes = ax.flatten()
    # 自己创建的数据 有数据 有标签
    X, y_true = make_circles(n_samples=1000, noise=0.15)  # 这是一个圆环形状的
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true)

    # DBSCAN 算法
    t0 = time.time() # 用于计算算法所用时间
    dbscan = DBSCAN(eps=.1, min_samples=6).fit(X)  # 该算法对应的两个参数
    t = time.time() - t0
    axes[1].scatter(X[:, 0], X[:, 1], c=dbscan.labels_)
    # axes[1].title('time : %f' % t)
    plt.show()

'''
层次聚类
'''
def HierarchicalClustering():
    from sklearn import datasets
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import preprocessing
    from sklearn.cluster import AgglomerativeClustering # 层次聚类

    # 导入数据集，鸢尾花数据集
    iris = datasets.load_iris()
    iris_data = iris.data

    # 数据预处理
    data = np.array(iris_data[:50, 1:-1]) # 只取前五十行，取出第一列到倒数第一列
    min_max_scaler = preprocessing.MinMaxScaler() # 归一化处理
    data_M = min_max_scaler.fit_transform(data) # 训练数据

    from scipy.cluster.hierarchy import linkage, dendrogram
    # 绘制树状图
    plt.figure(figsize=(20, 6))
    Z = linkage(data_M, method='ward', metric='euclidean')
    p = dendrogram(Z, 0)
    plt.show()

    # 模型训练 层次聚类
    ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    ac.fit(data_M)
    # 预测标签
    labels = ac.fit_predict(data_M)
    # 聚类结果可视化
    plt.scatter(data_M[:, 0], data_M[:, 1], c=labels)
    plt.show()
HierarchicalClustering()


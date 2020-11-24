'''
sklearn 实现对决策树的绘制 显示
graphviz 图形可视化软件包的使用

'''
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz
import pandas as pd

# 构建数据集
def creat_data():
    row_data = {'no surfacing':[1,1,1,0,0],
                'flippers':[1,1,0,1,1],
                'fish':['yes','yes','no','no','no']}
    data = pd.DataFrame(row_data)
    return data
dataset = creat_data()
# 特征
xtrain = dataset.iloc[:, :-1]
# 标签 将yes no 转化为数字型
ytrain = dataset.iloc[:,-1]
labels = ytrain.unique().tolist()
ytrain = ytrain.apply(lambda x: labels.index(x))

# 绘制树模型
clf = DecisionTreeClassifier()
clf = clf.fit(xtrain, ytrain)
tree.export_graphviz(clf)
dot_data = tree.export_graphviz(clf,out_file=None)

# 给图形增加颜色
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=['no surfacing', 'flippers'],
                                class_names=['fish', 'no fish'],
                                filled=True,
                                rounded=True,
                                special_characters=True)
graphviz.Source(dot_data)

# 利用render方法生成图形 并显示
graph = graphviz.Source(dot_data)
graph.render("fish")
graph.view()

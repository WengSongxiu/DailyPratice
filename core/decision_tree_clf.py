# -*- coding:utf-8 -*-

# 导入依赖
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import os

print(os.environ['PATH'])

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# 读取数据
x_features = np.array([[-1, -2], [-2, -1], [-3, -2], [1, 3], [2, 1], [3, 2]])
y_label = np.array([0, 1, 0, 1, 0, 1])

# 训练模型
tree_clf = DecisionTreeClassifier()
tree_clf = tree_clf.fit(x_features, y_label)

# 模型可视化
plt.figure()
plt.scatter(x_features[:,0],x_features[:,1],c=y_label,s=50,cmap='viridis')
plt.title('Dataset')
# plt.show()

dot_data = tree.export_graphviz(tree_clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("pengunis")
# 模型预测

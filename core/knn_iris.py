# -*- coding:utf-8 -*-

# 导入依赖
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 读取数据
iris = datasets.load_iris()

feature = iris.data
target = iris.target

# 训练预测
x_train, x_test, y_train, y_test = train_test_split(
    feature, target, test_size=0.2, random_state=2020)
clf = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# 模型评估
print(metrics.accuracy_score(y_test, y_pred))

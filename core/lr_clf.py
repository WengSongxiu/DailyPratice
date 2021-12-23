# -*- coding:utf-8 -*-

# 导入依赖
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 读取数据
x_features = np.array([[-1, -2], [-2, -1], [-3, -2], [1, 3], [2, 1], [3, 2]])
y_label = np.array([0, 0, 0, 1, 1, 1])

# 训练模型
lr_clf = LogisticRegression()
lr_clf = lr_clf.fit(x_features, y_label)

# 查看模型参数
print('the weight of Logistic Regression:', lr_clf.coef_)
print('the intercept(w0) of Logistic Regression:', lr_clf.intercept_)


plt.figure()
plt.scatter(x_features[:, 0], x_features[:, 1],
            c=y_label, s=50, cmap='viridis')
# plt.show()

# 模型预测
x_1 = np.array([[0, -1]])
x_2 = np.array([[1, 2]])
y_predict1 = lr_clf.predict(x_1)
y_predict2 = lr_clf.predict(x_2)
print('The New point 1 predict class:\n', y_predict1)
print('The New point 1 predict class:\n', y_predict2)

y_predict_pro1 = lr_clf.predict_proba(x_1)
y_predict_pro2 = lr_clf.predict_proba(x_2)
print('The New point 1 predict class:\n', y_predict_pro1)
print('The New point 1 predict class:\n', y_predict_pro2)

# 读取数据
iris_data = load_iris()
# print(iris_data)
iris_target = iris_data.target
iris_features = pd.DataFrame(
    data=iris_data.data,
    columns=iris_data.feature_names)

# 数据探索
print(iris_features.info)
print(iris_target)
print(iris_target.shape)
print(iris_features.shape)
print(iris_features.head())
print(iris_features.tail())
print(pd.Series(iris_target).value_counts())
print(iris_features.describe())

# 可视化探索
iris_all = iris_features.copy()
iris_all['target'] = iris_target
sns.pairplot(data=iris_all, diag_kind='hist', hue='target')
# plt.show()

for col in iris_features.columns:
    sns.boxplot(
        x='target',
        y=col,
        saturation=0.5,
        palette='pastel',
        data=iris_all)
    plt.title(col)
    # plt.show()

# 训练预测
x_train, x_test, y_train, y_test = train_test_split(
    iris_features, iris_target, test_size=0.2, random_state=2020)
clf = LogisticRegression(random_state=0, solver='lbfgs')
clf.fit(x_train, y_train)
print(clf.coef_)
print(clf.intercept_)

y_predict = clf.predict(x_test)
y_predict_pro = clf.predict_proba(x_test)
print(y_predict)
print(y_test)
print(metrics.accuracy_score(y_test, y_predict))
confusion_result = metrics.confusion_matrix(y_test, y_predict)
print(confusion_result)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_result, annot=True, cmap='Blues')
plt.show()

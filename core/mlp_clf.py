# -*- coding:utf-8 -*-

# 基础数组运算库导入
import numpy as np
# 画图库导入
import matplotlib.pyplot as plt
# 导入三维显示工具
from mpl_toolkits.mplot3d import Axes3D
# 导入BP模型
from sklearn.neural_network import MLPClassifier
# 导入demo数据制作方法
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
import warnings
from sklearn.exceptions import ConvergenceWarning

# 制作五个类别的数据，每个类别1000个样本
train_samples, train_labels = make_classification(n_samples=1000, n_features=3, n_redundant=0,
                                                  n_classes=5, n_informative=3, n_clusters_per_class=1,
                                                  class_sep=3, random_state=10)
# 将五个类别的数据进行三维显示
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=20, azim=20)
ax.scatter(train_samples[:, 0], train_samples[:, 1],
           train_samples[:, 2], marker='o', c=train_labels)
plt.title('Demo Data Map')
# plt.show()

# 建立 BP 模型, 采用sgd优化器，relu非线性映射函数
BP = MLPClassifier(
    solver='sgd',
    activation='relu',
    max_iter=500,
    alpha=1e-3,
    hidden_layer_sizes=(32, 32),
    random_state=1)
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=ConvergenceWarning,
        module="sklearn")
    BP.fit(train_samples, train_labels)

print(BP)

# 模型预测
predict_labels = BP.predict(train_samples)
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=20, azim=20)
ax.scatter(train_samples[:, 0], train_samples[:, 1],
           train_samples[:, 2], marker='o', c=predict_labels)
plt.title('Demo Data Predict Map with BP Model')
# plt.show()
print("预测准确率：{:.4f}".format(BP.score(train_samples, train_labels)))
print("真实类别：", train_labels[:10])
print("预测类别：", predict_labels[:10])
print(classification_report(train_labels, predict_labels))

test_sample = np.array([[-1, 0.1, 0.1], [-1.2, 10, -91],
                       [-12, -0.1, -0.1], [100, -90.1, -9.1]])
print(f"{test_sample} 类别是: ", BP.predict(test_sample))
print(f"{test_sample} 类别概率分别是: ", BP.predict_proba(test_sample))

test_samples, test_labels = make_classification(n_samples=200, n_features=3, n_redundant=0,
                                                  n_classes=5, n_informative=3, n_clusters_per_class=1,
                                                  class_sep=3, random_state=2020)

test_predict = BP.predict(test_samples)
print(f"{test_sample} 类别是: ", test_predict)

# 将五个类别的数据进行三维显示
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=20, azim=20)
ax.scatter(np.append(train_samples[:, 0],test_samples[:, 0]), np.append(train_samples[:, 1],test_samples[:, 1]),
           np.append(train_samples[:, 2],test_samples[:, 2]), marker='o', c=np.append(train_labels,test_predict))
plt.title('Demo Data Map predicted.')
plt.show()

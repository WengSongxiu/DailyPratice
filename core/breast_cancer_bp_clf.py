# -*- coding:utf-8 -*-
# 导入依赖
# 导入乳腺癌数据集
from sklearn.datasets import load_breast_cancer
# 导入BP模型
from sklearn.neural_network import MLPClassifier
# 导入训练集分割方法
from sklearn.model_selection import train_test_split
# 导入预测指标计算函数和混淆矩阵计算函数
from sklearn.metrics import classification_report
# 导入绘图包
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# 读取数据
cancer = load_breast_cancer()
print(cancer)

# 数据探索
# 查看数据集信息
print('breast_cancer数据集的长度为：', len(cancer))
print('breast_cancer数据集的类型为：', type(cancer))
# 分割数据为训练集和测试集
cancer_data = cancer['data']
print('cancer_data数据维度为：', cancer_data.shape)
print('cancer_data数据类型为：', type(cancer_data))
cancer_target = cancer['target']
print('cancer_target标签维度为：', cancer_target.shape)
print('cancer_target标签类型为：', type(cancer_target))
feature_name = cancer['feature_names']
cancer_feature = pd.DataFrame(cancer_data, columns=feature_name)
x_train, x_test, y_train, y_test = train_test_split(
    cancer_feature, cancer_target, test_size=0.2, random_state=2020)

# 特征工程
# 模型训练
# 建立 BP 模型, 采用Adam优化器，relu非线性映射函数
BP = MLPClassifier(
    solver='adam',
    activation='relu',
    max_iter=1000,
    alpha=1e-3,
    hidden_layer_sizes=(64, 32, 32),
    random_state=1)
# 进行模型训练
BP.fit(x_train, y_train)
# 模型评估
# 进行模型预测
y_predict = BP.predict(x_test)
# 可视化真实数据
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=20, azim=20)
ax.scatter(x_train.values[:, 0], x_train.values[:, 1],
           x_train.values[:, 2], marker='o', c=y_train)
plt.title('True Label Map')

# 可视化预测数据
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=20, azim=20)
ax.scatter(x_test.values[:, 0], x_test.values[:, 1],
           x_test.values[:, 2], marker='o', c=y_test)
plt.title('Cancer with BP Model')


# 进行测试集数据的类别预测
print("测试集的真实标签:\n", y_test)
print("测试集的预测标签:\n", y_predict)
print(classification_report(y_test, y_predict))

plt.show()

# -*- coding:utf-8 -*-
"""
Part 1. 莺尾花数据集--贝叶斯分类
    Step1: 库函数导入
    Step2: 数据导入&分析
    Step3: 模型训练
    Step4: 模型预测
    Step5: 原理简析
"""

# 导入依赖
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

# 读取数据
features, target = datasets.load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=0, shuffle=True)

# 模型训练
clf = GaussianNB(var_smoothing=1e-8)
clf.fit(x_train, y_train)

# 预测评估
y_pre = clf.predict(x_test)
y_pre_proba = clf.predict_proba(x_test)
print(metrics.accuracy_score(y_test, y_pre))
print(y_pre)
print(y_pre_proba)

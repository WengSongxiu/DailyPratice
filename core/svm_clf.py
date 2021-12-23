# -*- coding:utf-8 -*-

"""
Demo实践
    Step1:库函数导入
    Step2:构建数据集
    Step3:模型训练
    Step4:模型预测
    Step5:模型可视化
"""

# 库函数导入
import pandas as pd
import numpy as np
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt

# 构建数据
x_fearures = np.array([[-1, -2], [-2, -1], [-3, -2], [1, 3], [2, 1], [3, 2]])
y_label = np.array([0, 0, 0, 1, 1, 1])

# 模型训练
clf = svm.SVC(kernel='linear')
clf = clf.fit(x_fearures, y_label)
print('the weight of Logistic Regression:', clf.coef_)
print('the intercept(w0) of Logistic Regression:', clf.intercept_)

# 模型预测
y_train_pred = clf.predict(x_fearures)
print('The predction result:', y_train_pred)

# 模型可视化
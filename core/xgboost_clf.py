# -*- coding:utf-8 -*-
# 导入依赖
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from core.xgboost_importance import *
from sklearn.model_selection import GridSearchCV


def get_mapfunction(x):
    mapp = dict(zip(x.unique().tolist(),
                    range(len(x.unique().tolist()))))

    def mapfunction(y):
        if y in mapp:
            return mapp[y]
        else:
            return -1
    return mapfunction


# 读取数据
data = pd.read_csv("../data/input/train.csv")

# 数据探索
print(data.info())
print(data.head())
print(data.isnull().sum() / data.isnull().count())
print(pd.Series(data['RainTomorrow']).value_counts())

# 特征工程
data = data.fillna(-1)
print(data.isnull().sum() / data.isnull().count())
numerical_features = [x for x in data.columns if data[x].dtype == np.float]
category_features = [
    x for x in data.columns if data[x].dtype != np.float and x != 'RainTomorrow']
for i in category_features:
    data[i] = data[i].apply(get_mapfunction(data[i]))
# 训练预测
data_target_part = data['RainTomorrow']
data_features_part = data[[x for x in data.columns if x != 'RainTomorrow']]
x_train, x_test, y_train, y_test = train_test_split(
    data_features_part, data_target_part, test_size=0.2, random_state=2020)
clf = XGBClassifier(learning_rate=0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)
# clf = XGBClassifier(learning_rate=0.1,
#                     n_estimators=1000,
#                     max_depth=5,
#                     min_child_weight=1,
#                     gamma=0,
#                     subsample=0.8,
#                     colsample_bytree=0.8,
#                     objective='binary:logistic',
#                     nthread=4,
#                     scale_pos_weight=1,
#                     seed=27)
clf = clf.fit(x_train, y_train)
# ans = classes(x_train, y_train, x_test)
# pre = accuracy_score(y_test, ans)
# print('acc=', accuracy_score(y_test, ans))

# 模型评估
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)
confusion_matrix_train = metrics.confusion_matrix(y_train, train_predict)
confusion_matrix_test = metrics.confusion_matrix(y_test, test_predict)
# 准确度
print('The accuracy of the Logistic Regression is(train):',
      metrics.accuracy_score(y_train, train_predict))
print('The accuracy of the Logistic Regression is(test):',
      metrics.accuracy_score(y_test, test_predict))

# 混淆矩阵
print('The confusion matrix train:\n', confusion_matrix_train)
print('The confusion matrix test:\n', confusion_matrix_test)

# 模型优化
"""
XGBoost中包括但不限于下列对模型影响较大的参数：
    learning_rate: 有时也叫作eta，系统默认值为0.3。每一步迭代的步长，很重要。太大了运行准确率不高，太小了运行速度慢。
    subsample：系统默认为1。这个参数控制对于每棵树，随机采样的比例。减小这个参数的值，算法会更加保守，避免过拟合, 取值范围零到一。
    colsample_bytree：系统默认值为1。我们一般设置成0.8左右。用来控制每棵随机采样的列数的占比(每一列是一个特征)。
    max_depth： 系统默认值为6，我们常用3-10之间的数字。这个值为树的最大深度。这个值是用来控制过拟合的。max_depth越大，模型学习的更加具体。
调节模型参数的方法有贪心算法、网格调参、贝叶斯调参等。这里我们采用网格调参，它的基本思想是穷举搜索：在所有候选的参数选择中，通过循环遍历，尝试每一种可能性，表现最好的参数就是最终的结果
"""

learning_rate = [0.1, 0.3, 0.6]
subsample = [0.8, 0.9]
colsample_bytree = [0.6, 0.8]
max_depth = [3, 5, 8]

parameters = {'learning_rate': learning_rate,
              'subsample': subsample,
              'colsample_bytree': colsample_bytree,
              'max_depth': max_depth}
clf = XGBClassifier(learning_rate=0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)


model = XGBClassifier(n_estimators=1000,
                      min_child_weight=1,
                      gamma=0,
                      objective='binary:logistic',
                      nthread=4,
                      scale_pos_weight=1,
                      seed=27)
# 网格搜索
clf = GridSearchCV(
    model,
    parameters,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1)
# clf = clf.fit(x_train, y_train)
print("Best score: %f using %s" % (clf.best_score_, clf.best_params_))

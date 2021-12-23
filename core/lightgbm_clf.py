# -*- coding:utf-8 -*-

# 导入依赖
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMClassifier
from sklearn import metrics
from core.lightgbm_importance import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *

# 读取数据
data = pd.read_csv('../data/input/high_diamond_ranked_10min.csv')
print(data.head())
target = data['blueWins']
feature = data.drop(['blueWins', 'gameId'], axis=1)

# 数据探索
print(data.info())
print(data.head())
print(target.value_counts())
print(feature.describe())

# 特征工程
drop_cols = [
    'redFirstBlood',
    'redKills',
    'redDeaths',
    'redGoldDiff',
    'redExperienceDiff',
    'blueCSPerMin',
    'blueGoldPerMin',
    'redCSPerMin',
    'redGoldPerMin',
    'redAvgLevel',
    'blueAvgLevel']
feature.drop(drop_cols, axis=1, inplace=True)

# 训练预测
x_train, x_test, y_train, y_test = train_test_split(
    feature, target, test_size=0.2, random_state=2020)
clf = LGBMClassifier(
    n_estimators=60,
    feature_fraction=1,
    learning_rate=0.1,
    max_depth=3,
    num_leaves=16)
clf = clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print('The accuracy of the LGB Regression is(test):',
      metrics.accuracy_score(y_test, y_predict))

# 特征选择
ans = classes(x_train, y_train, x_test)
pre = accuracy_score(y_test, ans)
print('acc=', accuracy_score(y_test, ans))

# 参数调优
parameters = {'learning_rate': [0.1, 0.3, 0.6],
              'feature_fraction': [0.5, 0.8, 1],
              'num_leaves': [16, 32, 64],
              'max_depth': [-1, 3, 5, 8]}

model = LGBMClassifier(n_estimators=50)
clf = GridSearchCV(
    model,
    parameters,
    cv=3,
    scoring='accuracy',
    verbose=3,
    n_jobs=-1)
# clf = clf.fit(x_train, y_train)
# print("Best score: %f using %s" % (clf.best_score_, clf.best_params_))

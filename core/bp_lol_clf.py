# -*- coding:utf-8 -*-
# 导入依赖
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# 读取数据
data = pd.read_csv("../data/input/high_diamond_ranked_10min.csv")
target = data['blueWins']
feature = data.drop(['blueWins'], axis=1)
print(target.value_counts())

# 训练模型
x_train, x_test, y_train, y_test = train_test_split(
    feature, target, test_size=0.2)
BP = MLPClassifier(
    solver='adam',
    alpha=1e-5,
    hidden_layer_sizes=(3,3),
    random_state=1,
    max_iter=100000
)
BP.fit(x_train, y_train)
y_predict = BP.predict(x_test)

# 模型评估
print("测试集的真实标签:\n", y_test)
print("测试集的预测标签:\n", y_predict)
print(accuracy_score(y_test, y_predict))

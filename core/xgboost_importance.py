# -*- coding:utf-8 -*-
# 导入依赖
from xgboost import plot_importance
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt


# 画图函数
def estimate(model, data):
    ax1 = plot_importance(model, importance_type='gain')
    ax1.set_title("gain")
    ax2 = plot_importance(model, importance_type='weight')
    ax2.set_title("weight")
    ax3 = plot_importance(model, importance_type='cover')
    ax3.set_title("cover")
    plt.show()


# 实现
def classes(data, label, test):
    model = XGBClassifier()
    model.fit(data, label)
    ans = model.predict(test)
    estimate(model, data)
    return ans

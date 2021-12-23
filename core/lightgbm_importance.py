# -*- coding:utf-8 -*-
from lightgbm import plot_importance
import matplotlib.pyplot as plt
from lightgbm.sklearn import LGBMClassifier


def estimate(model):
    ax1 = plot_importance(model, importance_type="gain")
    ax1.set_title('gain')
    ax2 = plot_importance(model, importance_type="split")
    ax2.set_title('split')
    plt.show()


def classes(data, label, test):
    model = LGBMClassifier()
    model.fit(data, label)
    ans = model.predict(test)
    estimate(model)
    return ans

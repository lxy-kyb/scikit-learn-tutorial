# -*- coding:utf-8 -*-
# from sklearn import preprocessing
# import numpy as np

# a = np.array([[10, 2.7, 3.6],
#               [-100, 5, -2],
#               [120, 20, 40]], dtype=np.float64)

# print preprocessing.scale(a)

# 数据预处理模块
from sklearn import preprocessing
import numpy as np
# 用来将样本分割成train和test
from sklearn.cross_validation import train_test_split
# 生成用来识别的样本
from sklearn.datasets.samples_generator import make_classification
# Support Vertor Machine 中的 Support Vertor Classifier 
from sklearn.svm import SVC
# 数据可视化模块
import matplotlib.pyplot as plt


X, y = make_classification(
    n_samples=300, n_features=2,
    n_redundant=0, n_informative=2,
    random_state=22, n_clusters_per_class=1,
    scale=100
)

X=preprocessing.scale(X)

plt.scatter(X[:, 0], X[:,1], c=y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
clf = SVC()
clf.fit(X_train, y_train)
print clf.score(X_test, y_test)
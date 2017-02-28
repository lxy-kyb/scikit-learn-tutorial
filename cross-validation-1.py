# -*- coding:utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier #K近邻分类
import matplotlib.pyplot as plt

#加载iris数据集

iris = load_iris()
X = iris.data
y = iris.target

# # #分割数据集 random_state 为随机数种子保证每次分割的数据一致
# # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 4)

# knn = KNeighborsClassifier()

# # knn.fit(X_train, y_train)

# # print knn.score(X_test, y_test)

# scores = cross_val_score(knn, X, y, cv = 5, scoring='accuracy')

# print scores

# print scores.mean()

k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
    loss = -cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error')
    k_scores.append(loss.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Vlidated MSE')
plt.show()
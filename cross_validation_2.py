# -*- coding:utf-8 -*-
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target

# train_size, train_loss, test_lost = learning_curve(
#     SVC(gamma=0.001), X, y, cv=10, scoring='mean_squared_error',
#     train_sizes=[0.1,0.25,0.5,0.75,1]
# )

# train_loss_mean = -np.mean(train_loss, axis=1)
# test_loss_mean = -np.mean(test_lost, axis=1)

# plt.plot(train_size, train_loss_mean, 'o-', color='r', label='Training')
# plt.plot(train_size,test_loss_mean, 'o-', color='g', label='Cross-Validation')

# plt.xlabel('Training examples')
# plt.ylabel('Loss')
# plt.legend(loc='best')
# plt.show()

#取1.00000000e-06 ~ 5.01187234e-03 之间10个数字
param_range = np.logspace(-6, -2.3, 20)

train_loss, test_loss = validation_curve(
    SVC(), X, y, param_name='gamma', param_range=param_range, cv=10, scoring='mean_squared_error'
)

train_loss_mean = -np.mean(train_loss, axis=1)
test_lost_mean = -np.mean(test_loss, axis=1)

plt.plot(param_range, train_loss_mean, color='r', label='Training')
plt.plot(param_range, test_lost_mean, color='b', label='Cross-Validation')

plt.xlabel('gamma')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()


import numpy as np
from keras.src.datasets import boston_housing

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

print(X_train.shape, X_train.ndim)

print(y_train.shape, y_train.ndim)

# 练习四 对波士顿房价数据集的数据张量进行切片操作，输出其中 第101~200个数据样本。
# (提示:注意Python的数据索引是从0开始的。)

print(X_train[100:200], X_train[100:200].shape, X_train[100:200].ndim)

# 练习五 用Python生成形状如下的两个张量，确定其阶的个数，并进行点积操作，最后输出结果。
# A = [1，2，3，4，5]
# B = [[5]，[4]，[3]，[2]，[1]]

array_a = np.array([1,2,3,4,5])
array_b = np.array([[5], [4], [3], [2], [1]])

print('shape of array_a ', array_a.shape, ' rank of array_a ', array_a.ndim)
print('shape of array_b ', array_b.shape, ' rank of array_b ', array_b.ndim)

print('array_a dot array_b: ', np.dot(array_a, array_b))
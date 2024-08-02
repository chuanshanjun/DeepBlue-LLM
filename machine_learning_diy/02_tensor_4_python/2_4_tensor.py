import numpy as np

# 2.4.2 标量-0D(阶)向量
# Num Py 标量
X = np.array(5)
print('X的值:      ', X)
print('X的阶:      ', X.ndim)
print('X的数据类型: ', X.dtype)
print('X的形状:    ', X.shape)


# 2.4.3 向量-1D(阶)向量
X = np.array([5,6,7,8,9])
print('X的值:      ', X)
print('X的阶:      ', X.ndim)
print('X的数据类型: ', X.dtype)
print('X的形状:    ', X.shape)


# 通过一个机器学习数据集了解数据
from keras.datasets import boston_housing


(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
print('X_train的形状:           ', X_train.shape)
print('X_train中第一个样本的形状: ', X_train[0].shape)
print('X_train中第一个样本的内容: ', X_train[0])
print('y_train的形状:           ', y_train.shape)


# 两个相同维度向量点积
weight = np.array([1, -1.8, 1, 1, 2]) # 权重向量(也就是多项式的参数)
X = np.array([1, 6, 7, 8, 9]) # 特征向量

print('权重向量的形状:      ', weight.shape)
print('特征向量的形状:      ', X.shape)

y_hat = np.dot(weight, X) # 通过点积构建预测函数

print('点积返回结果:        ', y_hat)
print('点积返回结果的形状:   ', y_hat.shape) # 返回的是标量，所以是0阶张量

y_hat = weight.dot(X)
print('weight.dot(X)实现同样的点积效果： ', y_hat)


# 2.4.4 矩阵-2D(阶)向量
print("X_train的内容: ", X_train)

# 矩阵点积


# 2.4.5 张量-3D(阶)张量


# 2.4.6 4D(阶)张量


# 2.4.7 5D

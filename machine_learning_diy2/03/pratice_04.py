import numpy as np
import pandas as pd
from keras.src.datasets import boston_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

print('X_train shape: ', X_train.shape, '\nrank: ', X_train.ndim, '\ndata: ', X_train)
print('y_train shape: ', y_train.shape, '\nrank: ', y_train.ndim, '\ndata: ', y_train)

# y 转换成矩阵
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print('after transfer y from 1D to 2D')
print('y_train shape: ', y_train.shape, '\nrank: ', y_train.ndim, '\ndata: ', y_train)

# 数据归一化
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

print('after MinMaxScaler() operation')
print('X_train shape: ', X_train.shape, '\nrank: ', X_train.ndim, '\ndata: ', X_train)
print('y_train shape: ', y_train.shape, '\nrank: ', y_train.ndim, '\ndata: ', y_train)

# 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 反归一化预测结果
y_pred_original_scale = y_scaler.inverse_transform(y_pred)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
mse_original_scale = mean_squared_error(y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(y_pred))

print('预测的销售架构(测试集):', y_pred_original_scale)
print('给预测评分 (MSE):', mse)
print('给预测评分 (MSE, 原始尺度):', mse_original_scale)


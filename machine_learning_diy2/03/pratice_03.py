import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# 加载数据
df_ads = pd.read_csv('./data/advertising.csv')

# 构建特征集与标签集
X = df_ads.drop('sales', axis=1)
y = df_ads['sales'].values.reshape(-1, 1)

# 切分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 数据归一化
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 归一化 y_train 和 y_test
y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

# 创建并训练模型
ridge_model = Ridge(alpha=1.0)  # alpha 参数控制正则化的强度, L1 正则化
ridge_model.fit(X_train, y_train)

# 预测
y_pred = ridge_model.predict(X_test)

# 反归一化预测结果
y_pred_original_scale = y_scaler.inverse_transform(y_pred)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
mse_original_scale = mean_squared_error(y_scaler.inverse_transform(y_test), y_pred_original_scale)

print('预测的销售架构(测试集):', y_pred_original_scale)
print('给预测评分 (MSE):', mse)
print('给预测评分 (MSE, 原始尺度):', mse_original_scale)

# 预测计划的数据
X_plan = np.array([250, 50, 50])
X_plan = scaler.transform(X_plan.reshape(1, -1))

y_pred_plan = ridge_model.predict(X_plan)
y_pred_plan_original_scale = y_scaler.inverse_transform(y_pred_plan)

print('预测的销售额:', y_pred_plan_original_scale)
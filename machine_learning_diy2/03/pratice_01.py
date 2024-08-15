import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

df_ads = pd.read_csv('./data/advertising.csv')

print(df_ads.head())

# 构建特征集与标签集
X = df_ads.drop('sales', axis=1)
y = df_ads['sales']

print("张量X的阶:", X.ndim)
print("张量X的形状:", X.shape)
print(X)
print("张量y的阶:", y.ndim)
print("张量y的形状:", y.shape)
print(y)

# 因为线性回归要求数据集都是2D张量，X已经是了，y还需要手工变
y = y.values.reshape(len(y), 1)

# 切分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 数据归一化
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print ('预测的销售架构(测试集)', y_pred)

print('给预测评分: ', model.score(X_test, y_test))

X_plan = np.array([250, 50, 50])
X_plan = scaler.transform(X_plan.reshape(1,3))

y_pred_plan = model.predict(X_plan)
y_pred_plan_original_scale = y_scaler.inverse_transform(y_pred_plan)  # 反归一化预测结果

print('预测的销售额:', y_pred_plan_original_scale)
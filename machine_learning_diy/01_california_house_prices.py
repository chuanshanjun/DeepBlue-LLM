import pandas as pd

df_housing = pd.read_csv("https://raw.githubusercontent.com/huangjia2019/house/master/house.csv")

# 显示加州房价数据
df_housing.head

print(df_housing.head)

# 构建特征集X
X = df_housing.drop("median_house_value", axis = 1)

# 构建标签集y
y = df_housing.median_house_value


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# 导入线性回归算法模型
from sklearn.linear_model import LinearRegression

# 确定线性回归算法
model = LinearRegression()

# 根据训练数据集，训练机器，拟合函数
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("房价的真值(测试集)", y_test)

print("预测的房价(测试集)", y_pred)

# 评估预测结果
print("给预测评分: ", model.score(X_test, y_test))


import matplotlib.pyplot as plt

#用散点图显示家庭收入中位数和房价中位数的分布
plt.scatter(X_test.median_income, y_test, color='brown')

#画出回归函数(从特征到预测标签)
plt.plot(X_test.median_income, y_pred, color='green', linewidth=1)

plt.xlabel('Median Income') #x轴:家庭收入中位数
plt.ylabel('Median House Value') #y轴:房价中位数
plt.show() #显示房价分布和机器学习到的函数模型


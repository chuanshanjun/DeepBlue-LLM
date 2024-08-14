import pandas as pd


df_housing = pd.read_csv('./data/house.csv')

print(df_housing.head()) # 显示房价

# 构建特征集与标签集
X = df_housing.drop('median_house_value', axis=1)

y = df_housing['median_house_value']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print ('预测的房价(测试集)', y_pred)

print('给预测评分: ', model.score(X_test, y_test))


import matplotlib.pyplot as plt
#用散点图显示家庭收入中位数和房价中位数的分布
plt.scatter(X_test.median_income, y_test, color='brown')
#画出回归函数(从特征到预测标签)
plt.plot(X_test.median_income, y_pred, color='green', linewidth=1)
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.legend(['src_data', 'pred_data'])
plt.show()


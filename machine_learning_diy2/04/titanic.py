import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

df_titanic = pd.read_csv('./data/titanic.csv')

print(df_titanic.head())

# 数据处理
# PassengerId 不用
# Survived -> target
# Pclass -> 票务等级 1,2,3
# Name -> 去掉
# sex -> male,female 改成0,1
# Parch -> 同在船上的父母/小孩数量
# ticket -> 感觉没有啥用 要不扔掉
# fare -> 费用


# 构建哑变量
a = pd.get_dummies(df_titanic['Pclass'], prefix='Pclass')
b = pd.get_dummies(df_titanic['Embarked'], prefix='Embarked')

# 把哑变量添加进dataframe
frame = [df_titanic, a, b]
df_titanic = pd.concat(frame, axis=1)
df_titanic['Sex'] = df_titanic['Sex'].map({'male': 0, 'female': 1})

df_titanic = df_titanic.drop(columns=['Name', 'Ticket', 'PassengerId', 'Pclass', 'Embarked', 'Cabin'])

# 缺失的年龄数据补全 使用平均年龄填充
mean_age = df_titanic['Age'].mean()
df_titanic['Age'].fillna(mean_age, inplace=True)

# 缺失数据fare 使用平均值填充
mean_fare = df_titanic['Fare'].mean()
df_titanic['Fare'].fillna(mean_fare, inplace=True)

print(df_titanic.head())

y = df_titanic['Survived'] # 标签集
X = df_titanic.drop('Survived', axis=1)

y = y.values.reshape(-1, 1)

print("张量X的形状:", X.shape)
print("张量y的形状:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 1 先不对数据做标准化和归一化
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("SK learn 逻 辑 回 归 测 试 准 确 率{:.2f}%".format(lr.score(X_test, y_test)*100))






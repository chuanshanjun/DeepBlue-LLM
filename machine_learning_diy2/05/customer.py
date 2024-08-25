import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df_bank = pd.read_csv('./data/BankCustomer.csv')

print(df_bank.head())

features = ['City', 'Gender', 'Age', 'Tenure', 'ProductsNo', 'HasCard', 'ActiveMember', 'Exited']

fig = plt.subplots(figsize=(15, 15))

for i, j in enumerate(features):
    plt.subplot(4, 2, i + 1)
    plt.subplots_adjust(hspace=1.0)
    sns.countplot(x=j, data=df_bank)
    plt.title("No.of costumers")
    plt.show()

# 数据处理
# 1 性别 二元类别特征 转换成 0/1
# 2 城市 多元类别特征 转换为 多个二元哑变量
# 把二元类别文本数字化
df_bank['Gender'].replace('Female', 0, inplace=True)
df_bank['Gender'].replace('Male', 1, inplace=True)

# 显示数字类别
print("Gender unique values: ", df_bank['Gender'].unique())

d_city = pd.get_dummies(df_bank['City'])
df_bank = [df_bank, d_city]
df_bank = pd.concat(df_bank, axis=1)

# 构建标签
y = df_bank['Exited']
X = df_bank.drop(['Exited', 'Name', 'City'], axis=1)

# 显示X的头部
print(X.head())

# 拆分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LogisticRegression()

history = lr.fit(X_train, y_train)
print('逻辑回归准确率 {:2f}%'.format(lr.score(X_test, y_test) * 100))
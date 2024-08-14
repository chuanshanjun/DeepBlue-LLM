import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df_ads = pd.read_csv('./data/advertising.csv')

print(df_ads.head())

sns.heatmap(df_ads.corr(), cmap="YlGnBu", annot=True)
plt.show()

sns.pairplot(df_ads,
             x_vars=['wechat', 'weibo', 'others'],
             y_vars='sales',
             height=4, aspect=1, kind='scatter')
plt.show()

X = np.array(df_ads['wechat'])
y = np.array(df_ads['sales'])

print('shape of X: ', X.shape, ' rank of X: ', X.ndim, '\nand content of X: ', X)

X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)
print('After reshape shape of X: ', X.shape, ' rank of X: ', X.ndim, '\nand content of X: ', X)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=0)

def scaler(train, test):
    max = np.max(train)
    min = np.min(train)
    temp = max - min
    train -= min
    test -= min
    return train/temp, test/temp

X_train, X_test = scaler(X_train, X_test)
y_train, y_test = scaler(y_train, y_test)

plt.plot(X_train, y_train, 'r.', label='Training data')
plt.xlabel('wechat')
plt.ylabel('sales')
plt.legend()
plt.show()


def loss_function(X, y, w, b):
    y_hat = w*X + b
    loss = y - y_hat
    return (np.sum(loss**2))/(2*len(X))

print ("当权重为5, 偏置为3时, 损失为：",
loss_function(X_train, y_train, w=5, b=3))
print ("当权重为100, 偏置为1时, 损失为：",
loss_function(X_train, y_train, w=100, b=1))


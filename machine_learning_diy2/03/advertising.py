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

# 梯度下降
def gradient_descent(X, y, w, b, lr, iter):
    l_history = np.zeros(iter)
    w_history = np.zeros(iter)
    b_history = np.zeros(iter)
    for i in range(iter):
        y_hat = w*X + b
        loss = y_hat - y
        d_w = np.dot(X.T, loss) / len(X)
        d_b = np.sum(loss) / len(X)

        w = w - lr*d_w
        b = b - lr*d_b

        l_history[i] = loss_function(X, y, w, b)
        w_history[i] = w
        b_history[i] = b

    return l_history, w_history, b_history

# 首先确定参数的初始值
iterations = 200 # 迭代100次
alpha = 0.5 # 初始学习速率设为1
weight = -5 # 权重
bias = 3 # 偏置
# 计算一下初始权重和偏置值所带来的损失
print (' 当 前 损 失 : ', loss_function(X_train, y_train, weight, bias))

# 绘制当前的函数模型
plt.plot(X_train, y_train, 'r.', label='Training Data') # 显示训练数据
line_X = np.linspace(np.min(X_train), np.max(X_train), 500)
line_y = [weight*x + bias for x in line_X]
plt.plot(line_X, line_y, 'b--', label='Current hypothesis')
plt.xlabel('wechat')
plt.ylabel('sales')
plt.legend(['Training Data', 'Current hypothesis'])
plt.show()

loss_history, weight_history, bias_history = gradient_descent(X_train, y_train, weight, bias, alpha, iterations)

# 画出损失大小，和迭代次数 X-迭代次数 y-损失
plt.plot(loss_history, 'g--', label='Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend(['Loss Curve'])
plt.show()

# 绘制当前的函数模型
plt.plot(X_train, y_train, 'r.', label='Training data')
line_X = np.linspace(np.min(X_train), np.max(X_train), 500)
line_y = [weight_history[-1]*x + bias_history[-1] for x in line_X]
plt.plot(line_X, line_y, 'b--', label='Current hypothesis')
plt.xlabel('wechat')
plt.ylabel('sales')
plt.legend(['Training Data', 'Current hypothesis'])
plt.show()

print ('当前损失:', loss_function(X_train, y_train, weight_history[-1], bias_history[-1]))
print ('当前权重:', weight_history[-1])
print ('当前偏置:', bias_history[-1])

print ('当前损失:', loss_function(X_test, y_test, weight_history[-1], bias_history[-1]))


loss_history_test, weight_history_test, bias_history_test = gradient_descent(X_test, y_test, weight, bias, alpha, iterations)


# 画出损失大小，和迭代次数 X-迭代次数 y-损失
plt.plot(loss_history, 'g--', label='Loss Curve')
plt.plot(loss_history_test, 'r-', label='Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend(['Train Date Loss Curve', 'Test Date Loss Curve'])
plt.show()
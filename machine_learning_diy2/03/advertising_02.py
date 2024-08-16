import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df_ads = pd.read_csv('./data/advertising.csv')

print(df_ads.head())

X = np.array(df_ads.drop('sales', axis=1))
y = np.array(df_ads['sales'])

print("张量X的阶:", X.ndim)
print("张量X的维度:", X.shape)
print(X)

y = y.reshape(len(y), 1)

X_train_origin, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=0)
def scaler(train, test):
    max = np.max(train)
    min = np.min(train)
    temp = max - min
    train -= min
    test -= min
    return train/temp, test/temp

X_train, X_test = scaler(X_train_origin, X_test)
y_train, y_test = scaler(y_train, y_test)

# 为X训练集添加 0维特征的代码
x0_train = np.ones((len(X_train), 1))
X_train = np.append(x0_train, X_train, axis=1)
print ("张量X的形状:", X_train.shape)
print (X_train)


def loss_function(X, y, weight):
    y_hat = np.dot(X, weight.reshape(-1,1)) # 点积运算h(x)=w0x0+w1x1+w2x2+w3x3
    loss = y_hat - y # 中间过程, 求 出当前W和真值的差值
    cost = np.sum((loss**2))/(2*len(X)) # 这是平方求和过程, 均方误差函数的代码实现
    return cost

#首先确定参数的初始值
iterations = 300 # 迭代300次
alpha = 0.15 #学习速率设为0.15
weight = np.array([0.5, 1, 1, 1]) # 权重向量, w[0] = bias #计算一下初始值的损失

print (' 当 前 损 失 : ', loss_function(X_train, y_train, weight))

def gradient_descent(X, y, weight, lr, iter):
    l_history = np.zeros(iter)
    w_history = np.zeros((iter, len(weight)))
    for i in range(iter):
        y_hat = np.dot(X, weight.reshape(-1, 1))
        loss = y_hat - y # 此时 loss 形状(140,1), X 形状(140,4)
        d_w = np.dot(loss.T, X)/len(X)
        weight = weight - lr*d_w

        l_history[i] = loss_function(X,y,weight)
        w_history[i] = weight

    return l_history, w_history

# 定义线性回归模型
def linear_regression(X, y, weight, lr, iter):
    l_history, w_history = gradient_descent(X, y, weight, lr, iter)
    print("训练最终损失:", l_history[-1])
    y_pred = X.dot(w_history[-1])
    traning_acc = 100 - np.mean(np.abs(y_pred.reshape(-1,1) - y))*100
    print(" 线 性 回 归 训 练 准 确 率 : {:.2f}%".format(traning_acc)) # 输出准确率
    return l_history, w_history # 返回训练历史记录

#首先确定参数的初始值
iterations = 300 # 迭代300次
alpha = 0.15 #学习速率设为0.15
weight = np.array([0.5, 1, 1, 1]) # 权重向量, w[0] = bias #计算一下初始值的损失
print (' 当 前 损 失 : ', loss_function(X_train, y_train, weight))

# 调用刚才定义的线性回归模型
loss_history, weight_history = linear_regression(X_train, y_train, weight, alpha, iterations) #训练机器

print("权重历史记录:", weight_history)
print("损失历史记录:", loss_history)

X_plan = [250,50,50] # 要预测的X特征数据
X_train,X_plan = scaler(X_train_origin, X_plan) # 对预测数据也要归一化缩放
X_plan = np.append([1], X_plan)
y_plan = X_plan.dot(weight_history[-1])
y_value = y_plan*23.8 + 3.2
print ("预计商品销售额:",y_value, "千元")

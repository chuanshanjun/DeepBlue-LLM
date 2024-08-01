import pandas as pd
import numpy as np

df_ads = pd.read_csv('./data/advertising.csv')


# 多元回归
X = np.array(df_ads) # 构建特征集，包含全部特征
X = np.delete(X, [3], axis=1) # 删除标签
y = np.array(df_ads.sales) # 构建标签集，销售额
print('张量X的阶: ', X.ndim)
print('张量X的形状（维度）: ', X.shape)
print('张量X的内容: ', X)

# 数据分割成训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 数据归一化
def scaler(train, test):  # 定义归一化函数，进行数据压缩
    min = train.min(axis=0)  # 训练集最小值
    max = train.max(axis=0)  # 训练集最大值
    gap = max - min  # 最大值与最小值的差
    train -= min  # 所有数据减去最小值
    train /= gap  # 所有数据除以最大值和最小值的差
    test -= min  # 把训练集最小值应用于测试集
    test /= gap  # 把训练集最大值和最小值的差应用于测试集
    return train, test  # 返回压缩后的数据

X_train, X_test = scaler(X_train, X_test)  # 对特征进行归一化

y_train, t_test = scaler(y_train, y_test)  # 对标签也归一化

print("当X 及 y 都归一化后 X_train: ", X_train, " y_train: ", y_train)

# y 转换从向量变为矩阵
y = y.reshape(-1, 1) # 通过reshape方法把向量转换为举证， -1等价于len(y)

# 构造X长度的全1数组配合对偏置的点积
x0_train = np.ones((len(X_train), 1))

# 把X增加一系列的1
X_train = np.append(x0_train, X_train, axis=1)

print("张量的形状: ", X_train.shape)
print("增加一列后的张量： ", X_train)


# 多变量的损失函数和梯度下降

# 损失函数
def loss_function(X, y, W): # 手工定义一个均方误差函数，此时W是一个向量，所以大写
    y_hat = X.dot(W.T) # 点积运算 h(x) = w0x0 +w1x1 + w2x2 +w3x3
    loss = y_hat.reshape((len(y_hat), 1)) -y # 中间过程，求出当前W和真值的差异
    cost = np.sum(loss**2)/(2*len(X)) # 这是平方求和过程，均方误差函数的代码实现
    return cost # 返回当前模型的均方误差值

# 梯度下降函数
def gradient_descent(X, y, W, lr, iter): # 定义梯度下降函数
    l_history = np.zeros(iter) # 初始化记录梯度下降中损失的数组
    W_history = np.zeros(iter) # 初始化记录梯度下降过程中权重的数组
    for i in range(iter): # 进行梯度下降的迭代，就是下多少级台阶
        y_hat = X.dot(W) # 这是向量化运算实现的假函数
        loss = y_hat.reshape((len(y_hat), 1)) - y # 中间过程，求出y_hat和y真值的差值
        derivative_W = X.T.dot(loss)/(2*len(X)) # 求出多项式的梯度向量
        derivative_W = derivative_W.reshape(len(W))
        W = W - lr*derivative_W # 结合学习率更新权重
        l_history[i] = loss_function(X, y, W)
        W_history[i] = W
    return l_history, W_history

# 定义线性回归函数模型
def linear_regression(X, y, weight, alpha, iter):
    loss_history, weight_history = gradient_descent(X, y, weight, alpha, iter)
    print("训练最终损失: ", loss_history[-1]) # 输出最终损失
    y_pred = X.dot(weight_history[-1].T) # 进行预测
    traning_acc = 100 - np.mean(np.abs(y_pred - y))*100 # 计算准确率
    print("线性回归训练准确率:  {:.2f}%".format(traning_acc)) # 输出准确率
    return loss_history, weight_history # 返回训练历史记录

# 初始化权重
#
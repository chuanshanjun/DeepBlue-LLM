import numpy as np
import pandas as pd

df_ads = pd.read_csv('../data/advertising.csv')

X = np.array(df_ads)               # 构建特征集，包含全部特征
X = np.delete(X, [3], axis=1)  # 删除标签
y = np.array(df_ads.sales)         # 构建标签集合

from sklearn.model_selection import train_test_split

print('X_train 的形状: ', X.shape, ' 及阶: ', X.ndim)

# X 已经是向量了， 此时将 y reshape 成矩阵
# -1 等价于 len(y)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# 目前输入数据集是一个2D矩阵，含有两个轴，一个轴是样本轴，数量200
# 另外一个轴是特征轴，数量是3

# 多变量的线性回归的关键在于，w已经从标量变成一个向量了

# 数据归一化
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


# 把偏置当作w0
x0_train = np.ones((len(X_train),1))
X_train = np.append(x0_train, X_train, axis=1) # 把X增加一系列的1

print('X_train 增加一列后X的形状 ', X_train.shape, ' 阶: ', X_train.ndim)

# 测试数据集也增加x0
x0_test = np.ones((len(X_test),1))
X_test = np.append(x0_test, X_test, axis=1) # 把X增加一系列的1

print('X_test 增加一列后X的形状 ', X_test.shape, ' 阶: ', X_test.ndim)

# 损失函数
def loss_function(X, y, W):
    # 将 bais -> 作为w0
    y_hat = np.dot(X, W.T)
    y_hat = y_hat.reshape(-1, 1)
    loss = y_hat - y
    cost = np.sum(loss**2)/(2*len(X))
    return cost

weight = np.array([0.5, 1, 1, 1]) # 0.5 就 w0
print('当前损失: ', loss_function(X_train, y_train, weight))


# 多元的梯度下降
def gradient_descent(X, y, W, lr, iter):
    l_history = np.zeros(iter)
    w_history = np.zeros((iter, len(W)))
    w_history.reshape(-1, 1)
    for i in range(iter):
        y_hat = np.dot(X, W.T).reshape(-1, 1)
        loss = y_hat - y
        d_w = X.T.dot(loss)/(len(X))
        d_w = d_w.reshape(len(W))
        W = W - lr*d_w
        l_history[i] = loss_function(X, y, W)
        w_history[i] = W
    return l_history, w_history


weight = np.array([0.5, 1, 1, 1]) # 0.5 就 w0
lr = 1
iter = 100

l_history, w_history = gradient_descent(X_train, y_train, weight, lr, iter)

print('当前损失: ', l_history[-1])
print('当前权重: ', w_history[-1])

def linear_regression(X, y, weight, alpha, iternations):
    l_history, w_history = gradient_descent(X_train, y_train, weight, alpha, iternations)
    print('训练最终损失: ', l_history[-1])
    y_pred = X.dot(w_history[-1]) # 进行预测
    training_acc = 100 - np.mean(np.abs(y_pred - y.reshape(len(y))))*100 # 计算准确率，注意这边的y计算的话，要从矩阵变成向量
    print("线性回归准确率: {:.2f}%".format(training_acc))
    return l_history, w_history # 返回训练历史记录

iter = 300
alpha = 0.1
weight = np.array([0.5, 1, 1, 1]) # 权重向量 w[0] = bias
# 计算一下初始损失
print('当前损失: ', loss_function(X_train, y_train, weight))

# 调用线性回归模型
l_history, w_history = linear_regression(X_train, y_train, weight, alpha, iter)

print('权重历史: ', w_history)
print('损失历史: ', l_history)

# 预测环节，当使用250元，50元，50元 进行一周的广告投放
X_plan = [250, 50, 50]

X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(X, y, test_size=0.2, random_state=0)

X_train, X_plan = scaler(X_train_original, X_plan)

X_plan = np.append([1], X_plan) # 加一个哑特征x0=1
y_plan = np.dot(w_history[-1], X_plan) # [-1] 即模型收敛时的权重

# 23.8是当前y_train中的最大值和最小值的差，3.2是最小值
y_val = y_plan*23.8 + 3.2
print("预计商品销售： ", y_val, "千元")
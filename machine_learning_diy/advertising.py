import numpy as np

import pandas as pd

df_ads = pd.read_csv('./data/advertising.csv')

# 首先 查看数据头，了解下数据内容
df_ads.head()

# 其次 了解数据相关性
import matplotlib.pyplot as plt
import seaborn as sns

# 关键 df_ads.corr() -> 计算的是皮尔逊相关系数，一种广泛用于度量两个变量之间线性关系强度的指标。
sns.heatmap(df_ads.corr(), cmap="YlGnBu", annot=True)

plt.show()

# 从热力图可知 微信的广告投放与销售的相关性最大

# 通过散点图两两一组显示商品销售额和各种广告投放金额的对应关系
sns.pairplot(df_ads,
             x_vars=['wechat', 'weibo', 'others'],
             y_vars='sales',
             height=4, aspect=1, kind='scatter')

plt.show()

# 数据清洗
# 因为微信的广告投放金额和商品销售额的相关性比较高，为了简化模型只留下微信的数据

X = np.array(df_ads.wechat)  # 构建特征集，只含有微信公众号广告投放金额一个特征

y = np.array(df_ads.sales)  # 构建标签集合，销售额

print("张量X的阶: ", X.ndim)

print("张量X的形状: ", X.shape)

print("张量X的内容: ", X)

# 注意 回归类的数值类型数据集，机器学习模型所读入的规范格式应该是2D张量，也就是矩阵，其形状为(样本数, 标签数)
# 所以对于现在的特征张量X 需reshap 从 （200，） -> (200,1)

X = X.reshape(len(X), 1)  # 向量 -> 矩阵

y = y.reshape(len(y), 1)

print("张量X的阶: ", X.ndim)

print("张量X的形状: ", X.shape)

print("张量X的内容: ", X)

print("张量y的阶: ", y.ndim)

print("张量y的形状: ", y.shape)

print("张量y的内容: ", y)

# 拆分数据集为训练集和测试集
# 上述两个数据集合需要随机分配， 拆分之前要注意数据是否已被排序或者分类，如果是，还要先进行打乱

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 使用print打印数据，发现已经打乱数据了， 因为上述 函数的 参数 shuffle = true 是默认的
print("X_train 数据集: ", X_train)


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


# 数据归一化 伪代码
# x_norm = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

# 数据归一化很重要的点在于，最大值(max)，最小值(min)，以及最大值与最小值之间的差(gap) 均来自训练集
# 不能使用测试集中的数据信息进行特征缩放中间步骤中任何值的计算


X_train, X_test = scaler(X_train, X_test)  # 对特征进行归一化

y_train, t_test = scaler(y_train, y_test)  # 对标签也归一化

print("当X 及 y 都归一化后 X_train: ", X_train, " y_train: ", y_train)

# 下面的散点图，形状一样，只是数值已被限制在一个较小的区间
plt.plot(X_train, y_train, 'r.', label='Training data')  # 显示训练数据
plt.xlabel('wechat')  # x轴标签
plt.ylabel('sales')  # y轴标签
plt.legend()  # 显示图例
plt.show()


# 选择机器模型
# 数学中      y = ax + b  a 为斜率, b 为截距
# 机器学习中  y = wx + b w 为权重, b 为偏置


# 假设(预测)函数 - h(x)

# 定义损失函数
def loss_function(X, y, weight, bias):
    y_hat = weight * X + bias  # 假设函数，其中已经应用了Python的广播功能
    loss = y_hat - y  # 求出每个y'和训练集中真实y的差值
    cost = np.sum(loss ** 2) / (2 * len(X))  # 这是均方误差函数的代码实现
    return cost  # 返回当前模型的均方误差值


# 随意设置参数查看均方误差大小：
print("当权重为5，偏置为3时，损失为：", loss_function(X_train, y_train, weight=5, bias=3))

print("当权重为100，偏置为1时，损失为：", loss_function(X_train, y_train, weight=100, bias=1))


def gradient_descent(X, y, w, b, lr, iter):  # 定义一个梯度下降函数
    l_history = np.zeros(iter)  # 初始化 记录梯度下降过程中损失的数组
    w_history = np.zeros(iter)  # 初始化 记录梯度下降过程中权重的数组
    b_history = np.zeros(iter)  # 初始化 记录梯度下降过程中偏置的数组
    for i in range(iter):  # 进行梯度下降的迭代，就是下多少级台阶
        y_hat = w*X + b  # 向量化运算的假设函数
        loss = y_hat - y  # 中间过程，求得的是假设函数y'和真正的y的差值
        derivative_w = X.T.dot(loss) / len(X)  # 对权重的求导
        derivative_b = np.sum(loss)*1 / len(X)  # 对偏置的求导
        w = w - (lr*derivative_w)  # 结合学习速率alpha更新权重
        b = b - (lr*derivative_b)  # 结合学习速率alpha更新偏置
        l_history[i] = loss_function(X, y, w, b)  # 梯度下降过程中的损失的历史记录
        w_history[i] = w  # 梯度下降过程中的权重的历史记录
        b_history[i] = b  # 梯度下降过程中的偏置的历史记录
    return l_history, w_history, b_history  # 返回梯度下降过程中的损失、权重、偏置的历史记录


# 3.5 实现一元线性回归模型并调试超参数

# 首先确定参数的初始值
# 迭代100次
iterations = 100
# 初始化学习速率设置为1
alpha = 1
weight = -5
bias = 3

# 计算一下初始权重和偏执带来的损失
print('当前损失: ', loss_function(X_train, y_train, weight, bias))

# 绘制当前的函数模型
plt.plot(X_train, y_train, 'r.', label='Training data')  # 显示训练数据

line_X = np.linspace(X_train.min(), X_train.max(), 500)  # X 值域

line_y = [weight * xx + bias for xx in line_X]  # 假设函数y_hat

plt.plot(line_X, line_y, 'b--', label='Current hypothesis')  # 显示当前假设函数

plt.xlabel('wechat')
plt.ylabel('sales')
plt.legend()  # 显示图例
plt.show()

# 梯度下降
loss_history, weight_history, bias_history = gradient_descent(X_train, y_train, weight,
                                                              bias, alpha, iterations)

# 画图 - 查看损失是否随着梯度下降而逐渐减小
plt.plot(loss_history, 'g--', label='Loss Curve') # 显示损失曲线
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend() # 显示图例
plt.show()

# 绘制当前的函数模型
plt.plot(X_train, y_train, 'r.', label='Training data')  # 显示训练数据
line_X = np.linspace(X_train.min(), X_train.max(), 500)
line_y = [weight_history[-1]*xx + bias_history[-1] for xx in line_X] # 假设函数
plt.plot(line_X, line_y, 'b--', label='Current hypothesis') # 显示当前假设函数
plt.xlabel('wechat')
plt.ylabel('sales')
plt.legend() # 显示图例
plt.show()

# 这个例子中
# 学习率 = 1 ， 迭代100次，损失函数的值已经收敛
print('当前损失: ', loss_function(X_train, y_train,
                                  weight_history[-1], bias_history[-1]))
print('当前权重: ', weight_history[-1])
print('当前偏置: ', bias_history[-1])

# 输出内容见下
# 当前损失:  0.00465780405531404
# 当前权重:  0.6552253409192806
# 当前偏置:  0.17690341009472493

# 在测试集上验证
print('测试集损失: ', loss_function(X_test, y_test, weight_history[-1], bias_history[-1]))

# 当前测试集损失值比训练集还要好
# 测试集损失:  0.004581809380247212

# 同时打印训练集损失函数和测试集损失函数
# 调用梯度下降函数，获取在测试集上的相应数据
loss_history_test, weight_history_test, bias_history_test = gradient_descent(X_test, y_test, w = -5 , b = 3, lr = 1, iter = 100)
plt.plot(loss_history, 'g--', label='Loss Curve Train') # 显示损失曲线
plt.plot(loss_history_test, 'r-', label='Loss Curve Test') # 显示损失曲线
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend() # 显示图例
plt.show()






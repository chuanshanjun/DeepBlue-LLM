import numpy as np

import pandas as pd

df_ads = pd.read_csv('../data/advertising.csv')

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


# 损失函数 - 此处使用的均方误差函数
def loss_function(X, y, w, b): # 手工定义一个损失函数
    y_hat = X*w + b
    loss = y - y_hat
    cost = np.sum(loss**2)/(2*len(X))
    return cost

# 测试 当w=5, b=3 在数据集X的损失值（均方误差） 是多少？
print('当w=5, b=3 在数据集X的损失值（均方误差）: ',
      loss_function(X_train, y_train, 5, 3))

# 测试 当w=100, b=1 在数据集X的损失值（均方误差） 是多少？
print('当w=100, b=1 在数据集X的损失值（均方误差）: ',
      loss_function(X_train, y_train, 10, 3))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# 1.数据读取
df_heart = pd.read_csv('./data/heart.csv')

print(df_heart.head())

# 用value_counts方法输出数据集中患心脏病和没有患心脏病的人数
val_counts = df_heart['output'].value_counts()
print(val_counts)

# 数据相关性分析例如可以显示年龄/最大心率这两个特征与是否患病之间的关系：
# 以年龄+最大心率作为输入, 查看分类结果散点图
plt.scatter(x=df_heart.age[df_heart.output==1], y=df_heart.thalachh[df_heart.output==1], c='red')
plt.scatter(x=df_heart.age[df_heart.output==0], y=df_heart.thalachh[df_heart.output==0], marker='^')
plt.legend(["Disease", "No Disease"]) # 显示图例
plt.xlabel("Age") # x轴标签
plt.ylabel("Heart Rate") # y轴标签
plt.show()

# 2.构建特征集和标签集
X = df_heart.drop(['output'], axis=1)
y = df_heart['output'].values
y = y.reshape(-1, 1)
print("张量X的形状:", X.shape)
print("张量y的形状:", y.shape)

# 3.拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 4.数据特征缩放
# 这里有一个很值得注意的地方，就是对数据缩放器要进行两次调用。针对X_train和X_test，要使用不同的方法，一个是fit_transform（先
# 拟合再应用），一个是transform（直接应用）。这是因为，所有的最大值、最小值、均值、标准差等数据缩放的中间值，都要从训练集得来，
# 然后同样的值应用到训练集和测试集。
# 本例中当然不需要对标签集进行归一化，因为标签集所有数据已经在［0，1］区间了。
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 1.逻辑函数的定义
# 首先定义一个Sigmoid函数, 输入Z, 返回y'
# 这函数接收中间变量z（线性回归函数的输出结果）
def sigmoid(z):
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

# 2.损失函数的定义
# 自己根据公式写的
# def loss_function(X, y, w, b): # 注 为了与X进行矩阵点积操作，把W直接构建成2D矩阵。
    # 1 先写出假设函数
    # y_hat = sigmoid(np.dot(X, w) + b) # Sigmoid逻辑函数 +线性函数(w X+b)得到y'
    # return (-1*(np.sum((y*np.log(y_hat) + (1-y)*np.log(1-y_hat))))/len(X)) # 注意此时X是2D使用len(X) 可能有问题
    # 因为我们要的是样本数量

# 书上的
def loss_function(X, y, w, b): # 注 为了与X进行矩阵点积操作，把W直接构建成2D矩阵。
    y_hat = sigmoid(np.dot(X, w) + b) # Sigmoid逻辑函数 +线性函数(w X+b)得到y'
    loss = -((y*np.log(y_hat) + (1-y)*np.log(1-y_hat))) #计算损失
    cost = np.sum(loss) / X.shape[0] # 整个数据集的平均损失
    return cost

# 3.梯度下降的实现
def gradient_descent(X, y, w, b, lr, iter):
    l_history = np.zeros(iter)
    w_history = np.zeros((iter, w.shape[0], w.shape[1])) # w_history是一个3D张量，因为w已经是一个2D张量了，因此语句
    # w_history［i］ = w，就是把权重赋值给w_history的后两个轴。而w_history的第一个轴则是迭代次数轴。
    b_history = np.zeros(iter)
    for i in range(iter):
        y_hat = sigmoid(np.dot(X, w) + b)
        # loss =  -((y*np.log(y_hat) + (1-y)*np.log(1-y_hat)))
        # 一定要注意里面是y_hat - y
        d_w = (np.dot(X.T, (y_hat - y)))/X.shape[0]    # 还要注意权重的梯度是一个形状为（13，1）的张量，其维度和特征轴维度相同，而偏置的梯度则是一个值。
        d_b = np.sum(y_hat - y)/X.shape[0]

        w = w - lr*d_w
        b = b - lr*d_b

        l_history[i] = loss_function(X, y, w, b)
        print(" 轮 次 ", i + 1, " 当 前 轮 训 练 集 损 失 ： ",l_history[i])
        w_history[i] = w
        b_history[i] = b
    return l_history, w_history, b_history

# 4.分类预测的实现

# 定义一个负责分类预测的函数
def predict(X, w, b): # 定义预测函数
    z = np.dot(X, w) + b # 线性函数
    y_hat = sigmoid(z) # 逻辑函数转换
    y_pred = np.zeros((y_hat.shape[0], 1)) # 初始化预测结果变量
    for i in range(y_hat.shape[0]):
        if y_hat[i, 0] < 0.5:
            y_pred[i, 0] = 0 # 如果预测概率小于0.5, 输出分类0
        else:
            y_pred[i, 0] = 1 # 如果预测概率大于等于0.5, 输出分类0
    return y_pred # 返回预测分类的结

def logistic_regression(X, y, w, b, lr, iter):
    l_history, w_history, b_history = gradient_descent(X,y, w, b, lr, iter)#梯度下降
    print("训练最终损失:", l_history[-1])  # 输出最终损失
    y_pred = predict(X, w_history[-1], b_history[-1])  # 进行预测
    traning_acc = 100 - np.mean(np.abs(y_pred - y_train))*100
    print(" 逻 辑 回 归 训 练 准 确 率 :{:.2f}%".format(traning_acc)) # 输出准确率
    return l_history, w_history, b_history  # 返回训练历史记录

#初始化参数
dimension = X.shape[1] # 这里的维度len(X)是矩阵的行的数目,维度是列的数目
weight = np.full((dimension, 1), 0.1) # 权重向量, 向量一般是1D, 但这里实际上创建了2D张量
bias = 0 # 偏置值
#初始化超参数
alpha = 1 # 学习速率
iterations = 500 # 迭代次数

# 用逻辑回归函数训练机器
loss_history, weight_history, bias_history = logistic_regression(X_train, y_train, weight,bias, alpha, iterations)

# 上面的只是在训练集上的预测，下面使用测试数据进行，模型泛化能力的验证
y_pred = predict(X_test, weight_history[-1], bias_history[-1])
traning_acc = 100 - np.mean(np.abs(y_pred - y_test))*100
print(" 测 试 集 上 的 逻 辑 回 归 训 练 准 确 率 :{:.2f}%".format(traning_acc)) # 输出准确率

print (" 测 试 集 上 逻 辑 回 归 预 测 分 类 值 :", predict(X_test,weight_history[-1], bias_history[-1]))

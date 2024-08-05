import numpy as np
import pandas as pd


# 使用sigmoid函数作为y_hat
# y_hat = 1/(1 + np.exp(-z))
def sigmoid(z):
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

def loss_function(X, y, w, b):
    y_hat = sigmoid(np.dot(X, w) + b) # Sigmoid逻辑函数 + 线性函数(wX +b) 得到y_hat
    loss = - (y*np.log(y_hat) + (1-y)*np.log(1 - y_hat)) # 计算损失
    cost = np.sum(loss)/X.shape[0] # 整个数据集的平均损失
    return cost # 返回整个数据集的平均损失

# loss = - (y_train*np.log(y_hat) + (1-y_train)*np.log(1-y_hat))

# 逻辑回归的梯度下降过程
def gradient_descent(X, y, w, b, lr, iter): # 定义逻辑回归梯度下降
    l_history = np.zeros(iter)
    w_history = np.zeros((iter, w.shape[0], w.shape[1]))
    b_history = np.zeros(iter)
    for i in range(iter):
        y_hat = sigmoid(np.dot(X, w) + b) # sigmoid函数 + 线性函数(wX +b)得到y_hat
        loss = -(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) # 计算损失
        d_w = np.dot(X.T, (y_hat - y))/X.shape[0]
        d_b = np.sum(y_hat - y)/X.shape[0]
        w = w - lr*d_w
        b = b - lr*d_b
        l_history[i] = loss_function(X,y,w,b)
        print('轮次', i+1, '当前训练集损失: ', l_history[i])
        w_history[i] = w # 注意w_history和w的形状
        b_history[i] = b
    return l_history, w_history, b_history


# 1 数据读取
df_heart = pd.read_csv('./data/heart.csv')

print('显示数据前5行: ', df_heart.head())

print('整个数据集总量: ', len(df_heart))

# 这是个一个必要步骤，如果300个数据中，只有3个人患病，那么这样的数据集直接通过
# 逻辑回归的方法做分类，可能不合适
# 本例中患病和没患病的人数接近
print('集中输出分类值， 以及各个类别的数量: ', df_heart.output.value_counts())


# 对数据进行相关性分析，例如 年龄/最大心率这两个特征与患病的关系
import matplotlib.pyplot as plt

# 以年龄+最大心率作为输入，查看分类结果
plt.scatter(x=df_heart.age[df_heart.output==1],
            y=df_heart.thalachh[df_heart.output==1], c='red')

plt.scatter(x=df_heart.age[df_heart.output==0],
            y=df_heart.thalachh[df_heart.output==0], marker='^')

plt.legend(['Disease', 'No Disease'])
plt.xlabel('Age')
plt.ylabel('Heart Rate')
plt.show()

# 2 构建特征集和标签集合
X = df_heart.drop(['output'], axis=1) #构建特征集
y = df_heart.output.values # 构建标签集
y = y.reshape(-1,1) # -1 是相对索引，等价于len(y) 因为y是向量，需要把它转换为矩阵
print('X的形状: ', X.shape, 'X的阶: ', X.ndim)
print('y的形状: ', y.shape, 'y的阶: ', y.ndim)

# 3 拆分数据集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 4 数据集的缩放
# 这次使用Sklearn中内置的数据缩放器
from sklearn.preprocessing import MinMaxScaler #导入数据缩放器
scaler = MinMaxScaler()

# 这是因为，所有的最大值、最小值、均值、标准差等数据缩放的中间值，都要从训练集得来， 然后同样的值应用到训练集和测试集。
X_train = scaler.fit_transform(X_train) # 特征归一化训练集 fit_transform(先拟合再应用)
X_test = scaler.transform(X_test)   # 特征归一化测试集     transform(直接应用) 注意上下两个不一样

# 本例中不需要对标签集进行归一化，因为标签集数据已经在[0,1]之间了


# 建立逻辑回归模型

# 定义分类预测函数
# 通过预测概率阈值0.5, 把y_hat 转换成 y_pred, 也就是把一个概率值转换成0,1分类
# y_pred与标签集y具有同样的维度的向量, 通过比较y_pred和真值，就可以看出多少个预测正确，多少个预测错误
def predict(X, w, b):
    z = np.dot(X, w) + b # 线性函数
    y_hat = sigmoid(z)   # 逻辑转换函数
    y_pred = np.zeros((y_hat.shape[0], 1)) # 初始化预测结果变量
    for i in range(y_hat.shape[0]):
        if y_hat[i,0] < 0.5:
            y_pred[i, 0] = 0 # 如果预测概率小于0.5，输出分类0
        else:
            y_pred[i, 0] = 1 # 如果预测概率大于等于0.5，输出分类0
    return y_pred # 返回预测结果分类

# 封装一个逻辑回归函数
def logistic_regression(X, y, w, b, lr, iter): # 定义一个逻辑回归模型
    l_history, w_history, b_history = gradient_descent(X, y, w, b, lr, iter) # 梯度下降，找到本轮训练最后的w和b
    print('训练最终损失: ', l_history[-1]) # 输出最终损失
    y_pred = predict(X, w_history[-1], b_history[-1])
    training_acc = 100 - np.mean(np.abs(y_pred - y))*100 # 计算准确率，注意这边y其实是y_train , *100 是因为上面对数据做过归一化处理
    print('逻辑回归训练准确率: {:.2f}%'.format(training_acc)) # 输出准确率
    return l_history, w_history, b_history

# 准备参数初始值
dimension = X.shape[1] # 这里的维度len(X)是矩阵的行的数目，维度是列的数据
weight = np.full((dimension, 1), 0.1) # 权重向量，向量一般是1D，这里创建了2D 张量
bias = 0

# 初始化超参数
alpha = 1 # 学习速率
iteration = 700 # 迭代次数

# 用逻辑回归训练机器
l_history, w_history, b_history = logistic_regression(X_train, y_train, weight, bias, alpha, iteration)

# 测试分类结构
# 用训练好的回归模型，对测试集进行分类测试
y_pred = predict(X_test, w_history[-1], b_history[-1]) # 预测测试集
testing_acc = 100 - np.mean(np.abs(y_pred - y_test))*100
print('逻辑回归测试准确率: {:.2f}%'.format(testing_acc)) # 输出准确率

print('逻辑回归预测分类值: ', predict(X_test, w_history[-1], b_history[-1]))

# 绘制损失曲线
# 绘制训练和测试的损失函数
loss_history_test = np.zeros(iteration) #初始化历史损失
for i in range(iteration): # 求训练过程中不同参数带来的测试集损失
    loss_history_test[i] = loss_function(X_test, y_test,
                                         w_history[i], b_history[i])
index = np.arange(0, iteration, 1)
plt.plot(index, l_history, c='blue', linestyle='solid')
plt.plot(index, loss_history_test, c='red', linestyle='dashed')
plt.legend(["Training Loss", "Test Loss"])
plt.xlabel("Number of Iteration")
plt.ylabel("Cost")
plt.show()

# 直接调用Sklearn库
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train) # fit 相当于梯度下降
print("Sklearn 逻辑回归测试准确率 {:.2f}%".format(lr.score(X_test, y_test)*100))


# 特征工程
# 把3个文本型变量转换成哑变量
a = pd.get_dummies(df_heart['cp'], prefix='cp')
b = pd.get_dummies(df_heart['thall'], prefix='thall')
c = pd.get_dummies(df_heart['slp'], prefix='slp')

# 把哑变量添加进dataframe
frames = [df_heart, a, b, c]
df_heart = pd.concat(frames, axis=1)
df_heart = df_heart.drop(columns = ['cp', 'thall', 'slp'])
print('显示新的dataframe', df_heart.head())

# 新数据重新训练-----------

X = df_heart.drop(['output'], axis=1) #构建特征集
y = df_heart.output.values # 构建标签集
y = y.reshape(-1,1) # -1 是相对索引，等价于len(y) 因为y是向量，需要把它转换为矩阵
print('X的形状: ', X.shape, 'X的阶: ', X.ndim)
print('y的形状: ', y.shape, 'y的阶: ', y.ndim)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 4 数据集的缩放
# 这次使用Sklearn中内置的数据缩放器
from sklearn.preprocessing import MinMaxScaler #导入数据缩放器
scaler = MinMaxScaler()

# 这是因为，所有的最大值、最小值、均值、标准差等数据缩放的中间值，都要从训练集得来， 然后同样的值应用到训练集和测试集。
X_train = scaler.fit_transform(X_train) # 特征归一化训练集 fit_transform(先拟合再应用)
X_test = scaler.transform(X_test)   # 特征归一化测试集     transform(直接应用) 注意上下两个不一样

dimension = X.shape[1] # 这里的维度len(X)是矩阵的行的数目，维度是列的数据
weight = np.full((dimension, 1), 0.1) # 权重向量，向量一般是1D，这里创建了2D 张量
bias = 0

# 用逻辑回归训练机器
l_history, w_history, b_history = logistic_regression(X_train, y_train, weight, bias, alpha, iteration)

# 测试分类结构
# 用训练好的回归模型，对测试集进行分类测试
y_pred = predict(X_test, w_history[-1], b_history[-1]) # 预测测试集
testing_acc = 100 - np.mean(np.abs(y_pred - y_test))*100
print('逻辑回归测试准确率: {:.2f}%'.format(testing_acc)) # 输出准确率
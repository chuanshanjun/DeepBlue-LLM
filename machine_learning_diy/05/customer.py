import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

df_bank = pd.read_csv('./data/BankCustomer.csv')

print(df_bank.head())  # 显示数据前5行

# 显示不同特征的分布情况
features = ['City', 'Gender', 'Age', 'Tenure', 'ProductsNo', 'HasCard', 'ActiveMember', 'Exited']

fig = plt.subplots(figsize=(15, 15))

for i, j in enumerate(features):
    plt.subplot(4, 2, i + 1)
    plt.subplots_adjust(hspace=1.0)
    sns.countplot(x=j, data=df_bank)
    plt.title("No.of costumers")
    plt.show()

# 数据处理
# 把二元类别文本数字化
df_bank['Gender'].replace("Female", 0, inplace=True)
df_bank['Gender'].replace("Male", 1, inplace=True)

# 显示数字类别
print("Gender unique values", df_bank['Gender'].unique())

# 把多元类别转换成多个二元类别哑变量，然后放回原始数据集
df_city = pd.get_dummies(df_bank['City'], prefix='City')
df_bank = [df_bank, df_city]
df_bank = pd.concat(df_bank, axis=1)

# 构建特征集和标签集
y = df_bank['Exited']
X = df_bank.drop(['Exited', 'City', 'Name'], axis=1)
print(X.head())  # 显示新的特征集

# 使用标砖方法拆分训练集和测试集
from sklearn.model_selection import train_test_split  # 拆分数据集

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0)

# 在不使用特征工程的前提下，先使用逻辑回归直接进行机器学习
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()  # 逻辑回归模型
history = lr.fit(X_train, y_train)  # 训练机器
print("在未做特征工程的前提下，"
      "直接使用逻辑回归后的正确率:"
      "{:.2f}%".format(lr.score(X_test, y_test) * 100))  # 在验证集上评分

# 单隐层神经网络Keras实现
# 使用序惯模型构建网络
import keras
from keras.src.models import Sequential
from keras.src.layers import Dense  # 导入Keras 全连接层

ann = Sequential()  # 创建一个序贯ANN模型
ann.add(Dense(units=12, input_dim=12, activation='relu'))  # 添加输入层
ann.add(Dense(units=24, activation='relu'))  # 添加隐层
ann.add(Dense(units=1, activation='sigmoid'))  # 添加输出层
ann.summary()  # 显示网络模型(这个语句不是必须的)

# 展示神经网络的形状结构
from IPython.display import SVG  # 实现神经网络结构的图形化显示
from keras.src.utils.model_visualization import model_to_dot

svg = SVG(model_to_dot(ann, show_shapes=True).create(prog='dot', format='svg'))

with open('ann.svg', 'w') as f:
    f.write(svg.data)

# 编译神经网络，指定优化器，损失函数，以及评估指标
ann.compile(optimizer='adam',  # 优化器
            loss='binary_crossentropy',  # 损失函数
            metrics=['acc'])  # 评估指标

history = ann.fit(X_train, y_train,  # 指定训练集
                  epochs=30,  # 指定轮次
                  batch_size=64,  # 指定批量大小
                  validation_data=(X_test, y_test))  # 指定验证集


def show_history(history):  # 显示训练过程中的学习曲线
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.subplot(1, 2, 2, )
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


show_history(history)

# 使用分类报告和混淆矩阵
from sklearn.metrics import classification_report  # 导入分类报告

# 此时未做特征处理，所以X_test 是 pandas 的 DataFrame 格式
# 但y_pred 是神经网络输出的已经是 NumPy array 格式
y_pred = ann.predict(X_test, batch_size=10)  # 预测测试集的标签

y_pred = np.round(y_pred)  # 四舍五入，将分类概率值转换为0/1整数值

# y_test 的格式与上面的 X_test 同理
y_test = y_test.values  # 把 Pandas series 转换为 NumPy array

y_test = y_test.reshape(len(y_test), 1)  # 转换成与y_pred相同形状

print(classification_report(y_test, y_pred, labels=[0, 1]))  # 调用分类报告

# 上面代码需要注意2点
# 1 神经网络模型的predict方法给出的预测结果也是一个概率，需要基于0.5的阈值进行转换，舍入成0、1整数值
# 2 y_test一直是一个pandas_series的数据格式并没有被转换为 NumPy数组。神经网络模型是可以接收Series和Dataframe格式的数据的， 但是此时为了和 进行比较，需要用values方法进行格式转换。
# 3 转换成NumPy数组后，需要再转换为与 形状一致的张 量，才输入classification_ report函数进行评估。

# 画出此时的混淆矩阵
# 还出混淆矩阵
from sklearn.metrics import confusion_matrix  # 导入混淆矩阵

cm = confusion_matrix(y_test, y_pred)  # 调用混淆矩阵
plt.title('ANN Confusion Matrix')  # 标题：人工神经网络混淆矩阵
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)  # 热力图设定
plt.show()

# 对于神经网络而言，特征缩放(feature scaling) 极为重要

# 此处先使用自己的代码进行数据标准化
# X' = (x - mean(x))/std(x)

mean = X_train.mean(axis=0)  # 计算训练集均值
X_train -= mean  # 训练集减去其均值
std = X_train.std(axis=0)  # 计算训练集标准差
X_train /= std  # 训练集/其标准差
X_test -= mean  # 测试集合-训练集均值
X_test /= std  # 测试集/训练集标准差

# 也可以直接使用Sklearn工具
# from sklearn.preprocessing import StandardScaler # 导入特征缩放器
# sc = StandardScaler() # 特征缩放器
# X_train = sc.fit_transform(X_train) # 拟合并应用于训练集
# X_test = sc.transform(X_test) # 训练集结果用于测试集

# 使用特征化（标准化）后的数据，进行逻辑回归模型的处理
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()  # 逻辑回归模型
history = lr.fit(X_train, y_train)  # 训练机器
print("做过特征工程后的，"
      "逻辑回归的正确率:"
      "{:.2f}%".format(lr.score(X_test, y_test) * 100))  # 在验证集上评分

# 使用特征化（标准化）后的数据，进行神经网络的处理
history = ann.fit(X_train, y_train,  # 指定训练集
        epochs=30,  # 指定轮次
        batch_size=64,  # 指定批量大小
        validation_data=(X_test, y_test))  # 指定验证集

# 做完特征工程后，再次查看随着训练次数的增加，训练集与测试集的损失及准确率的变化
show_history(history)

y_pred = ann.predict(X_test, batch_size=10)  # 预测测试集的标签

y_pred = np.round(y_pred)  # 四舍五入，将分类概率值转换为0/1整数值

# 上面的y_test 已经把格式转换过来了，所以此处不需要再转换了
# y_test = y_test.values  # 把 Pandas series 转换为 NumPy array
#
# y_test = y_test.reshape(len(y_test), 1)  # 转换成与y_pred相同形状

print(classification_report(y_test, y_pred, labels=[0, 1]))  # 调用分类报告

# 做完特征工程后，再次查看混淆矩阵
cm = confusion_matrix(y_test, y_pred)  # 调用混淆矩阵
plt.title('ANN Confusion Matrix')  # 标题：人工神经网络混淆矩阵
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)  # 热力图设定
plt.show()

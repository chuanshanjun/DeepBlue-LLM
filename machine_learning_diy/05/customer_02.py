from keras.src.models import Sequential
from keras.src.layers import Dense

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

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

df_bank = pd.read_csv('./data/BankCustomer.csv')

# 数据处理
# 把二元类别文本数字化
df_bank['Gender'].replace("Female", 0, inplace=True)
df_bank['Gender'].replace("Male", 1, inplace=True)

# 把多元类别转换成多个二元类别哑变量，然后放回原始数据集
df_city = pd.get_dummies(df_bank['City'], prefix='City')
df_bank = [df_bank, df_city]
df_bank = pd.concat(df_bank, axis=1)

# 构建特征集和标签集
y = df_bank['Exited']
X = df_bank.drop(['Exited', 'City', 'Name'], axis=1)

# 使用标砖方法拆分训练集和测试集
from sklearn.model_selection import train_test_split  # 拆分数据集

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0)

from sklearn.preprocessing import StandardScaler  # 导入特征缩放器

sc = StandardScaler()  # 特征缩放器
X_train = sc.fit_transform(X_train)  # 拟合并应用于训练集
X_test = sc.transform(X_test)  # 训练集结果用于测试集

# ann = Sequential()  # 创建序贯ANN模型
# ann.add(Dense(units=12, input_dim=12, activation='relu'))  # 添加输入层
# ann.add(Dense(units=24, activation='relu'))  # 添加隐层
# ann.add(Dense(units=48, activation='relu'))  # 添加隐层
# ann.add(Dense(units=96, activation='relu'))  # 添加隐层
# ann.add(Dense(units=192, activation='relu'))  # 添加隐层
# ann.add(Dense(units=1, activation='sigmoid'))  # 添加输出层
#
# # 编译神经网络，指定优化器，损失函数，以及评估指标
# history = ann.compile(optimizer='rmsprop',  # 使用RMSP优化器
#                       loss='binary_crossentropy',  # 损失函数
#                       metrics=['acc'])  # 评估指标
#
# history = ann.fit(X_train, y_train,  # 指定训练集
#                   epochs=30,  # 指定轮次
#                   batch_size=64,  # 指定批量大小
#                   validation_data=(X_test, y_test))  # 指定验证集
#
# show_history(history)
#
# y_pred = ann.predict(X_test, batch_size=10)  # 预测测试集的标签
# y_pred = np.round(y_pred)  # 四舍五入，将分类概率值转换为0/1整数值
# print(classification_report(y_test, y_pred, labels=[0, 1]))  # 调用分类报告
#
#
# print("更换优化器，将优化器rmsprop -> adam")
#
# # 更换优化器
# # 将优化器rmsprop -> adam
# history = ann.compile(optimizer='adam',  # 使用RMSP优化器
#                       loss='binary_crossentropy',  # 损失函数
#                       metrics=['acc'])  # 评估指标
#
# history = ann.fit(X_train, y_train,  # 指定训练集
#                   epochs=30,  # 指定轮次
#                   batch_size=64,  # 指定批量大小
#                   validation_data=(X_test, y_test))  # 指定验证集
#
# show_history(history)
#
# y_pred = ann.predict(X_test, batch_size=10)  # 预测测试集的标签
# y_pred = np.round(y_pred)  # 四舍五入，将分类概率值转换为0/1整数值
# print(classification_report(y_test, y_pred, labels=[0, 1]))  # 调用分类报告

# 神经网络正则化，添加Dropout层
from keras.src.layers import Dropout # 导入Dropout
ann = Sequential()  # 创建序贯ANN模型
ann.add(Dense(units=12, input_dim=12, activation='relu'))  # 添加输入层
ann.add(Dense(units=24, activation='relu'))  # 添加隐层
ann.add(Dropout(0.5)) # 添加Dropout层
ann.add(Dense(units=48, activation='relu'))  # 添加隐层
ann.add(Dropout(0.5)) # 添加Dropout层
ann.add(Dense(units=96, activation='relu'))  # 添加隐层
ann.add(Dropout(0.5)) # 添加Dropout层
ann.add(Dense(units=192, activation='relu'))  # 添加隐层
ann.add(Dropout(0.5)) # 添加Dropout层
ann.add(Dense(units=1, activation='sigmoid'))  # 添加输出层

history = ann.compile(optimizer='adam',  # 使用RMSP优化器
                      loss='binary_crossentropy',  # 损失函数
                      metrics=['acc'])  # 评估指标

history = ann.fit(X_train, y_train,  # 指定训练集
                  epochs=30,  # 指定轮次
                  batch_size=64,  # 指定批量大小
                  validation_data=(X_test, y_test))  # 指定验证集

show_history(history)

y_pred = ann.predict(X_test, batch_size=10)  # 预测测试集的标签
y_pred = np.round(y_pred)  # 四舍五入，将分类概率值转换为0/1整数值
print(classification_report(y_test, y_pred, labels=[0, 1]))  # 调用分类报告

cm = confusion_matrix(y_test, y_pred)  # 调用混淆矩阵
plt.title('ANN Confusion Matrix')  # 标题：人工神经网络混淆矩阵
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)  # 热力图设定
plt.show()
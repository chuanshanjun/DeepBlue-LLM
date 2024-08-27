from keras.src.models import Sequential
from keras.src.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


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

df_bank['Gender'].replace('Female', 0, inplace=True)
df_bank['Gender'].replace('Male', 1, inplace=True)

d_city = pd.get_dummies(df_bank['City'])
df_bank = [df_bank, d_city]
df_bank = pd.concat(df_bank, axis=1)

# 构建标签
y = df_bank['Exited']
X = df_bank.drop(['Exited', 'Name', 'City'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mean = X_train.mean(axis=0) # 计算训练集均值
X_train -= mean  # 训练集减去训练集均值
std = X_train.std(axis=0) # 计算训练集标准差
X_train /= std # 训练集除以训练集标准差
X_test -= mean # 测试集减去训练集均值
X_test /= std # 测试集除以训练集标准差

ann = Sequential()
ann.add(Dense(units=12, input_dim=12, activation='relu')) # 添加输入层
ann.add(Dense(units=24, activation='relu')) # 添加隐层
ann.add(Dense(units=48, activation='relu'))
ann.add(Dense(units=96, activation='relu'))
ann.add(Dense(units=192, activation='relu'))
ann.add(Dense(units=1, activation='sigmoid')) # 添加输出层

# 编译神经网络, 指定优化器、损失函数, 以及评估指标
ann.compile(optimizer='rmsprop',  # 此处我们先试试RMSP优化器
            loss='binary_crossentropy', # 损失函数
            metrics=['acc']) # 评估指标

history = ann.fit(X_train, y_train, # 指定训练集
        epochs=30, # 训练30轮
        batch_size=64, # 指定批量大小
        validation_data=(X_test, y_test)) # 训练验证集


y_pred = ann.predict(X_test, batch_size=10) # 预测测试集的标签

y_pred = np.round(y_pred) # 四舍五入，将分类值转换为0,1

y_test = y_test.values # 将y_test 从 Pandas series 转换为 numpy array

Y_test = y_test.reshape(-1,1) # 转换为与y_pred 相同的形状

print(classification_report(y_test, y_pred, labels=[0,1]))


show_history(history)



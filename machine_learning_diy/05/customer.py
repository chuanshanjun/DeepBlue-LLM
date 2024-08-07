import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


df_bank = pd.read_csv('./data/BankCustomer.csv')

print(df_bank.head()) # 显示数据前5行


# 显示不同特征的分布情况
features = ['City', 'Gender', 'Age', 'Tenure', 'ProductsNo', 'HasCard','ActiveMember','Exited']

fig = plt.subplots(figsize=(15,15))

for i, j in enumerate(features):
    plt.subplot(4, 2, i+1)
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
print(X.head()) # 显示新的特征集

# 使用标砖方法拆分训练集和测试集
from sklearn.model_selection import train_test_split # 拆分数据集

X_train, X_test, y_train, y_test = train_test_split(X, y,
                 test_size=0.2,
                 random_state=0)

# 在不使用特征工程的前提下，先使用逻辑回归直接进行机器学习
from sklearn.linear_model import LogisticRegression

lr =  LogisticRegression()         # 逻辑回归模型
history = lr.fit(X_train, y_train) # 训练机器
print("在未做特征工程的前提下，"
      "直接使用逻辑回归后的正确率:"
      "{:.2f}%".format(lr.score(X_test, y_test)*100)) # 在验证集上评分

# 单隐层神经网络Keras实现
# 使用序惯模型构建网络
import keras
from keras.src.models import Sequential
from keras.src.layers import Dense # 导入Keras 全连接层

ann = Sequential() # 创建一个序贯ANN模型
ann.add(Dense(units=12, input_dim=12, activation='relu')) # 添加输入层
ann.add(Dense(units=24, activation='relu'))               # 添加隐层
ann.add(Dense(units=1, activation='sigmoid'))             # 添加输出层
ann.summary() # 显示网络模型(这个语句不是必须的)

# 展示神经网络的形状结构
from IPython.display import SVG # 实现神经网络结构的图形化显示
from keras.src.utils.model_visualization import model_to_dot

svg = SVG(model_to_dot(ann, show_shapes=True).create(prog='dot', format='svg'))

with open('ann.svg', 'w') as f:
    f.write(svg.data)

# 编译神经网络，指定优化器，损失函数，以及评估指标
ann.compile(optimizer='adam',             # 优化器
            loss='binary_crossentropy',   # 损失函数
            metrics=['acc'])              # 评估指标

history = ann.fit(X_train, y_train,                 # 指定训练集
                  epochs=30,                        # 指定轮次
                  batch_size=64,                    # 指定批量大小
                  validation_data=(X_test, y_test)) # 指定验证集

def show_history(history): # 显示训练过程中的学习曲线
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.subplot(1,2,2,)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

show_history(history)

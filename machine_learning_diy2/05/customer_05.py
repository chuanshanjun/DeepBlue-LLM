from keras.src.models import Sequential
from keras.src.layers import Dense, BatchNormalization
from keras.src.layers import Dropout
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
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
ann.add(BatchNormalization()) # 添加批标准化层
ann.add(Dense(units=24, activation='relu')) # 添加隐层
ann.add(Dropout(0.5)) # 添加Dropout层
ann.add(BatchNormalization())
ann.add(Dense(units=48, activation='relu'))
ann.add(Dropout(0.5))
ann.add(BatchNormalization())
ann.add(Dense(units=96, activation='relu'))
ann.add(Dropout(0.5))
ann.add(BatchNormalization())
ann.add(Dense(units=192, activation='relu')) # 减少一层看效果,效果一般，验证集准确率略微提高，但F1下降
ann.add(Dropout(0.5))
ann.add(BatchNormalization())
ann.add(Dense(units=1, activation='sigmoid')) # 添加输出层

# 编译神经网络, 指定优化器、损失函数, 以及评估指标
ann.compile(optimizer='adam',  # 此处我们先试试RMSP优化器
            loss='binary_crossentropy', # 损失函数
            metrics=['acc']) # 评估指标

history = ann.fit(X_train, y_train, # 指定训练集
        epochs=30, # 训练30轮
        # epochs=22, # 减少训练轮次观察效果，效果略微变差，具体见下
        # epochs=35, # 增加训练轮次观察效果，效果一般，验证集准确率略微提高，但F1下降
        batch_size=64, # 指定批量大小
        validation_data=(X_test, y_test)) # 训练验证集


y_pred = ann.predict(X_test, batch_size=10) # 预测测试集的标签

y_pred = np.round(y_pred) # 四舍五入，将分类值转换为0,1

y_test = y_test.values # 将y_test 从 Pandas series 转换为 numpy array

Y_test = y_test.reshape(-1,1) # 转换为与y_pred 相同的形状

print(classification_report(y_test, y_pred, labels=[0,1]))


show_history(history)

cm = confusion_matrix(y_test, y_pred)
plt.title("ANN Confusion Matrix")

sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', cbar=False)

plt.show()


# Epoch 30/30
# 125/125 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.8567 - loss: 0.3677 - val_acc: 0.8625 - val_loss: 0.4458
# 200/200 ━━━━━━━━━━━━━━━━━━━━ 0s 701us/step
#
#               precision    recall  f1-score   support
#
#            0       0.91      0.93      0.92      1606
#            1       0.67      0.61      0.63       394
#
#     accuracy                           0.86      2000
#    macro avg       0.79      0.77      0.78      2000
# weighted avg       0.86      0.86      0.86      2000


# Epoch 22/22
# 125/125 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.8454 - loss: 0.3820 - val_acc: 0.8600 - val_loss: 0.4503
# 200/200 ━━━━━━━━━━━━━━━━━━━━ 0s 739us/step
#               precision    recall  f1-score   support
#
#            0       0.90      0.93      0.91      1606
#            1       0.67      0.57      0.62       394
#
#     accuracy                           0.86      2000
#    macro avg       0.78      0.75      0.77      2000
# weighted avg       0.85      0.86      0.86      2000


# Epoch 22/22
# 125/125 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.8468 - loss: 0.3908 - val_acc: 0.8595 - val_loss: 0.3981
# 200/200 ━━━━━━━━━━━━━━━━━━━━ 0s 684us/step
#               precision    recall  f1-score   support
#
#            0       0.89      0.95      0.92      1606
#            1       0.70      0.50      0.58       394
#
#     accuracy                           0.86      2000
#    macro avg       0.79      0.72      0.75      2000
# weighted avg       0.85      0.86      0.85      2000


# Epoch 35/35
# 125/125 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.8593 - loss: 0.3588 - val_acc: 0.8640 - val_loss: 0.3584
# 200/200 ━━━━━━━━━━━━━━━━━━━━ 0s 674us/step
#               precision    recall  f1-score   support
#
#            0       0.89      0.95      0.92      1606
#            1       0.73      0.49      0.59       394
#
#     accuracy                           0.86      2000
#    macro avg       0.81      0.72      0.75      2000
# weighted avg       0.85      0.86      0.85      2000


# 减少一层"ann.add(Dense(units=192, activation='relu'))"
# Epoch 30/30
# 125/125 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - acc: 0.8460 - loss: 0.3664 - val_acc: 0.8635 - val_loss: 0.3531
# 200/200 ━━━━━━━━━━━━━━━━━━━━ 0s 647us/step
#               precision    recall  f1-score   support
#
#            0       0.88      0.97      0.92      1606
#            1       0.77      0.44      0.56       394
#
#     accuracy                           0.86      2000
#    macro avg       0.82      0.70      0.74      2000
# weighted avg       0.85      0.86      0.85      2000


# 添加一层 批量标准化后的效果
# Epoch 30/30
# 125/125 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.8526 - loss: 0.3725 - val_acc: 0.8665 - val_loss: 0.4156
# 200/200 ━━━━━━━━━━━━━━━━━━━━ 0s 694us/step
#               precision    recall  f1-score   support
#
#            0       0.89      0.96      0.92      1606
#            1       0.74      0.50      0.60       394
#
#     accuracy                           0.87      2000
#    macro avg       0.81      0.73      0.76      2000
# weighted avg       0.86      0.87      0.86      2000

# 每层都加 批量标准化后的效果 验证集准确率略微下降，F1下降，但整个损失曲线更加光滑
# Epoch 30/30
# 125/125 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - acc: 0.8170 - loss: 0.4005 - val_acc: 0.8615 - val_loss: 0.3520
# 200/200 ━━━━━━━━━━━━━━━━━━━━ 0s 945us/step
#               precision    recall  f1-score   support
#
#            0       0.87      0.98      0.92      1606
#            1       0.80      0.40      0.53       394
#
#     accuracy                           0.86      2000
#    macro avg       0.83      0.69      0.72      2000
# weighted avg       0.85      0.86      0.84      2000
import numpy as np
import pandas as pd
from keras.src.ops import shape
from tensorflow.python.framework.test_ops import out_t

df_train = pd.read_csv('./data/exoTrain.csv')
df_test = pd.read_csv('./data/exoTest.csv')

print(df_train.head())
print(df_test.head())

# 因为数据是预先排序过的，所以先将其乱序排列
from sklearn.utils import shuffle
df_train = shuffle(df_train) # 乱序训练集
df_test = shuffle(df_test) # 乱序测试集

# 构建特征集合与标签集
X_train = df_train.iloc[:, 1:].values
y_train = df_train.iloc[:, 0].values
X_test = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:, 0].values

y_train -= 1
y_test -= 1

print(X_train)
print(y_train)

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

print(X_train.shape)
print(X_test.shape)

from keras.src.models import Sequential
from keras import layers
from keras.src.optimizers import Adam
from keras.src import Model
# model = Sequential()
# model.add(layers.Conv1D(32, 10, strides=4, input_shape=(3197, 1)))
# model.add(layers.MaxPool1D(pool_size=4, strides=2))
# model.add(layers.GRU(256, return_sequences=True))
# model.add(layers.Flatten()) # 展平层
# model.add(layers.Dropout(0.5))
# model.add(layers.BatchNormalization())
# model.add(layers.Dense(1, activation='sigmoid'))
# opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
# model.compile(
#     optimizer=opt,
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# history = model.fit(X_train, y_train,
#                     epochs=4,
#                     batch_size=128,
#                     validation_split=0.2,
#                     shuffle=True)

# 输出阈值的调整
from sklearn.metrics import classification_report # 分类报告
from sklearn.metrics import confusion_matrix # 混淆矩阵

# y_prob = model.predict(X_test)
# y_pred = np.where(y_prob > 0.2, 1, 0)
# cm = confusion_matrix(y_pred, y_test)
# print('Confusion matrix:\n', cm, '\n')
# print(classification_report(y_pred, y_test))

# 输出阈值调整
# y_prob = model.predict(X_test) # 对测试集进行预测
# y_pred = np.where(y_prob > 0.2, 1.0, 0.0) # 将概率值转换为真值
# cm = confusion_matrix(y_pred, y_test)
# print('Confusion matrix:\n', cm, '\n')
# print(classification_report(y_pred, y_test))

# 构建双向网络
X_train_rev = [X[::-1] for X in X_train]
X_test_rev = [X[::-1] for X in X_test]
X_train_rev = np.expand_dims(X_train_rev, axis=2)
X_test_rev = np.expand_dims(X_test_rev, axis=2)

# 再构建多头网络
# 构建正向网络
input_1 = layers.Input(shape=(3197,1))
x = layers.GRU(32, return_sequences=True)(input_1) # 改成return_sequences=True精确度暴跌
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
# 构建逆向网络
input_2 = layers.Input(shape=(3197,1))
y = layers.GRU(32, return_sequences=True)(input_2)
y = layers.Flatten()(y)
y = layers.Dropout(0.5)(y)
# 连接两个网络
z = layers.concatenate([x, y])
output = layers.Dense(1, activation='sigmoid')(z)
model = Model(inputs=[input_1, input_2], outputs=output)
model.summary()

opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01) # 设置优化器
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit([X_train, X_train_rev], y_train, validation_split=0.2, epochs=1, batch_size=128, shuffle=True)

# 32/32 ━━━━━━━━━━━━━━━━━━━━ 231s 7s/step - accuracy: 0.9469 - loss: 0.2350 - val_accuracy: 0.9931 - val_loss: 0.2107

# 32/32 ━━━━━━━━━━━━━━━━━━━━ 69s 2s/step - accuracy: 0.4613 - loss: 0.9705 - val_accuracy: 0.5000 - val_loss: 0.7375
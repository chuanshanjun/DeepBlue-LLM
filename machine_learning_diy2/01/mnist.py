import numpy as np
import pandas as pd

from keras.src.datasets import mnist

(X_train_image, y_train_lable), (X_test_image, y_test_lable) = mnist.load_data()

print ("数据集张量形状:", X_train_image.shape)
print ("第一个数据样本:\n", X_train_image[0])

print ("第一个数据样本的标签:", y_train_lable[0])
print ("第二个数据样本的标签:", y_train_lable[1])

from keras.src.utils import to_categorical # 导入keras.utils 工具库的类别转换工具

X_train = X_train_image.reshape(-1, 28, 28, 1)
X_test = X_test_image.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train_lable, 10) # 标签转换为one-hot编码
y_test = to_categorical(y_test_lable, 10)

print ("训练集张量形状:", X_train.shape)
print ("第一个数据标签:", y_train[0])
print ("第二个数据标签:", y_train[1])

from keras.src import models
from keras import layers

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train,validation_split = 0.3,epochs=5,batch_size=128)

score = model.evaluate(X_test, y_test)
print('测试集预测准确率:', score[1])

y_pred = model.predict(X_test[0].reshape(1,28,28,1)) # 预测测试集第一个数据
print(y_pred[0], "转换一下格式得到:", y_pred.argmax()) # 把 one-hot编码转换为数字

import matplotlib.pyplot as plt
plt.imshow(X_test[0].reshape(28, 28), cmap='Greys')
plt.show()
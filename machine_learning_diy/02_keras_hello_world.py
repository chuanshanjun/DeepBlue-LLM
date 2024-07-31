import numpy as np # 导入Num Py 库
import pandas as pd

from keras.datasets import mnist #从Keras中导入MNIST数据集

#读入训练集和测试集
(X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()

# 用shape方法显示张量形状
print("训练数据集张量形状: ", X_train_image.shape)

# 用shape方法显示张量形状
print("测试数据集张量形状: ", X_test_image.shape)

# 打印第一个数据样本
print("第一个数据样本: \n", X_train_image[0])

# 打印第一个数据样本标签
print("第一个数据样本标签: ", y_train_label[0])


# 导入keras.utils 工具库 的类别转换工具
from keras.utils import to_categorical

# 给标签增加一个纬度
X_train = X_train_image.reshape(60000, 28, 28, 1)

X_test = X_test_image.reshape(10000,28,28, 1)

# 特征转换为one-hot 编码
y_train = to_categorical(y_train_label, 10)

y_test = to_categorical(y_test_label, 10)


# 更新后的训练集张量形状
print("更新后的训练集张量形状: ", X_train.shape)

# 显示标签集的第一个数据
print("标签集的第一个数据: ", y_train[0])

print("标签集的第二个数据: ", y_train[1])



# 导入Keras模型, 以及各种神经网 络的层
from keras import models

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# 用序贯方式建立模型
model = models.Sequential()

# 添加 Conv2D 层
model.add(Conv2D(32, (3, 3), activation = 'relu',
                 input_shape = (28, 28, 1))) # 制定输入数据样本张量类型




# 添加Max Pooling2D 层
model.add(MaxPooling2D(pool_size = (2, 2)))

# 添加dropout层
model.add(Dropout(0.25))

# 展平
model.add(Flatten())

# 添加全连接层
model.add(Dense(128, activation = 'relu'))

# 添加dropout层
model.add(Dropout(0.5))

# Softmax 分类激活， 输出10纬分类码
model.add(Dense(10, activation = 'softmax'))

# 编译模型
model.compile(optimizer = 'rmsprop', # 指定优化器
              loss = 'categorical_crossentropy', # 指定损失函数
              metrics = ['accuracy'])  # 指定验证过程中的评估指标



model.fit(X_train, y_train, # 指定训练特征集和训练标签集
          validation_split = 0.3, # 部分训练集数据拆分成验证集
          epochs = 5, # 训练次数5轮
          batch_size = 128 # 以128为批量进行训练
          )

# 在验证集上进行模型评估
score = model.evaluate(X_test, y_test)

# 输出测试集上的预测准确率
print('测试集预测准确率： ', score[1])

# 预测测试集第一个数据
pred = model.predict(X_test[0].reshape(1, 28, 28, 1))

# 把one-hot编码转换为数字
print(pred[0], "转换一下格式得到： ", pred.argmax())

import matplotlib.pyplot as plt # 导入绘图工具包

# 输出图片
plt.imshow(X_test[0].reshape(28, 28), cmap = 'Greys')




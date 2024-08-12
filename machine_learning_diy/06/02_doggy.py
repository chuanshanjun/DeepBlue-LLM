import numpy as np
import pandas as pd
import os


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


print(os.listdir("../data/images"))

dir = "../data/images/"

chihuahua_dir = dir + 'n02085620-Chihuahua'  # 吉娃娃
japanese_spaniel_dir = dir + 'n02085782-Japanese_spaniel'  # 日本狆
maltese_dir = dir + 'n02085936-Maltese_dog'  # 马尔济斯犬
pekinese_dir = dir + 'n02086079-Pekinese'  # 狮子狗
shitzu_dir = dir + 'n02086240-Shih-Tzu'  # 西施犬
blenheim_spaniel_dir = dir + 'n02086646-Blenheim_spaniel'  # 英国可卡犬
papillon_dir = dir + 'n02086910-papillon'  # 蝴蝶犬
toy_terrier_dir = dir + 'n02087046-toy_terrier'  # 玩具猎狐梗
afghan_hound_dir = dir + 'n02088094-Afghan_hound'  # 阿富汗猎犬
basset_dir = dir + 'n02088238-basset'  # 巴吉度猎犬

import cv2  # 导入Open CV工具库

X = []
y_label = []
imgsize = 150


# 定义一个函数读入狗狗图像
def training_data(label, data_dir):
    print("正在读入:", data_dir)
    for img in os.listdir(data_dir):
        path = os.path.join(data_dir, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (imgsize, imgsize))
        X.append(np.array(img))
        y_label.append(str(label))


training_data('chihuahua', chihuahua_dir)
training_data('japanese_spaniel', japanese_spaniel_dir)
training_data('maltese', maltese_dir)
training_data('pekinese', pekinese_dir)
training_data('shitzu', shitzu_dir)
training_data('blenheim_spaniel', blenheim_spaniel_dir)
training_data('papillon', papillon_dir)
training_data('toy_terrier', toy_terrier_dir)
training_data('afghan_hound', afghan_hound_dir)
training_data('basset', basset_dir)

# 构建X y 张量
from sklearn.preprocessing import LabelEncoder  # 导入标签编码工具
from keras.src.utils import to_categorical  # 导入one-hot 编码

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_label)  # 标签编码
y = to_categorical(y, 10)  # 将标签转换为One-hot 编码

X = np.array(X)  # 将X从列表转换为张量数组
X = X / 255  # 将X张量归一化

print('X张量的形状:')
print("X张量中的第一个数据", X[1])
print('y张量中第一个数据', y[1])


# 将已经缩放到[0，1]区间之后的张量重新以图像的形式显示出来
import matplotlib.pyplot as plt
import random as rdm

# 随机显示狗狗图片
fig,ax = plt.subplots(5 ,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range(2):
        r = rdm.randint(0, len(X))
        ax[i,j].imshow(X[r])
        ax[i,j].set_title('Dog: '+y_label[r])
plt.tight_layout()

# 随机地乱序并拆分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 1 构建简单的卷积网络
# from keras import layers # 导入所有层
# from keras import models # 导入所有模型
# cnn = models.Sequential() # 序贯模型
# cnn.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3))) # 卷积层
# cnn.add(layers.MaxPooling2D((2,2))) # 最大池化层
# cnn.add(layers.Conv2D(64, (3,3), activation='relu')) # 卷积层
# cnn.add(layers.MaxPooling2D((2,2))) # 最大池化层
# cnn.add(layers.Conv2D(128, (3,3), activation='relu')) # 卷积层
# cnn.add(layers.MaxPooling2D((2,2))) # 最大池化层
# cnn.add(layers.Flatten()) # 展平层
# cnn.add(layers.Dense(512, activation='relu')) # 全连接层
# cnn.add(layers.Dense(10, activation='softmax')) # 分类输出
# cnn.compile(loss='categorical_crossentropy', # 损失函数
#             optimizer='rmsprop', # 优化器
#             metrics=['acc']) # 评估指标
#
# history = cnn.fit(X_train, y_train, # 指定训练集
#         epochs=50,        # 指定轮次
#         batch_size=256,   # 指定批量大小
#         validation_data=(X_test, y_test)) # 指定验证集
#
# show_history(history)

# 2 通过更换优化器提升网络的水平
# from keras import layers # 导入所有层
# from keras import models # 导入所有模型

# cnn = models.Sequential() # 序贯模型
# cnn.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3))) # 卷积层
# cnn.add(layers.MaxPooling2D((2,2))) # 最大池化层
# cnn.add(layers.Conv2D(64, (3,3), activation='relu')) # 卷积层
# cnn.add(layers.MaxPooling2D((2,2))) # 最大池化层
# cnn.add(layers.Conv2D(128, (3,3), activation='relu')) # 卷积层
# cnn.add(layers.MaxPooling2D((2,2))) # 最大池化层
# cnn.add(layers.Flatten()) # 展平层
# cnn.add(layers.Dense(512, activation='relu')) # 全连接层
# cnn.add(layers.Dense(10, activation='softmax')) # 分类输出
# cnn.compile(loss='categorical_crossentropy', # 损失函数
#             optimizer=optimizers.Adam(learning_rate=1e-4), # 优化器
#             metrics=['acc']) # 评估指标
#
# history = cnn.fit(X_train, y_train, # 指定训练集
#         epochs=50,        # 指定轮次
#         batch_size=256,   # 指定批量大小
#         validation_data=(X_test, y_test)) # 指定验证集
#
# show_history(history)

# 通过添加dropout层提升网络水平
from keras import layers # 导入所有层
from keras import models # 导入所有模型
from keras import optimizers # 导入优化器

cnn = models.Sequential() # 序贯模型
cnn.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3))) # 卷积层
cnn.add(layers.MaxPooling2D((2,2))) # 最大池化层
cnn.add(layers.Conv2D(64, (3,3), activation='relu')) # 卷积层
cnn.add(layers.Dropout(0.5)) # Dropout层
cnn.add(layers.MaxPooling2D((2,2))) # 最大池化层
cnn.add(layers.Conv2D(128, (3,3), activation='relu')) # 卷积层
cnn.add(layers.Dropout(0.5)) # Dropout层
cnn.add(layers.MaxPooling2D((2,2))) # 最大池化层
cnn.add(layers.Flatten()) # 展平层
cnn.add(layers.Dropout(0.5)) # Dropout层
cnn.add(layers.Dense(512, activation='relu')) # 全连接层
cnn.add(layers.Dense(10, activation='softmax')) # 分类输出
cnn.compile(loss='categorical_crossentropy', # 损失函数
            optimizer=optimizers.Adam(learning_rate=1e-4), # 优化器
            metrics=['acc']) # 评估指标

history = cnn.fit(X_train, y_train, # 指定训练集
        epochs=50,        # 指定轮次
        batch_size=256,   # 指定批量大小
        validation_data=(X_test, y_test)) # 指定验证集

show_history(history)


# 4 通过对原数据的变化，添加数据集，提升网络效果
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# augs_gen = ImageDataGenerator(
#         featurewise_center=False,
#         samplewise_center=False,
#         featurewise_std_normalization=False,
#         samplewise_std_normalization=False,
#         zca_whitening=False,
#         rotation_range=10,
#         zoom_range = 0.1,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         horizontal_flip=True,
#         vertical_flip=False)
# augs_gen.fit(X_train) # 针对训练集拟合数据增强器
#
# history = cnn.fit_generator( # 使用fit_generator
# augs_gen.flow(X_train, y_train, batch_size=16), # 增强后 的训练集
# validation_data = (X_test, y_test), # 指定验证集 validation_steps = 100, # 指定验证步长 steps_per_epoch = 100, # 指定每轮步长
# epochs = 50, # 指定轮次
# verbose = 1) # 指定是否显示训练过程中的信息
#
# show_history(history)
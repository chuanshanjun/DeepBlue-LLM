import os
import cv2
import numpy as np # 导入Numpy
from sklearn.preprocessing import LabelEncoder # 导入标签编码工具
from keras.src.utils import to_categorical # 导入one-hot 编码
import keras.src.models
import random as rdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# 获取目录
print(os.listdir('./data/images'))

dir = "./data/images/"

# 本示例只处理10种狗
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

# 将10个子目录中的图像和标签值读入X、y数据集
X = []
y_label = []
imgsize = 150

# 定义一个函数读入狗狗图像
def training_data(label, data_dir):
    print('正在读入: ', data_dir)
    for img in os.listdir(data_dir):
        path = os.path.join(data_dir, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (imgsize, imgsize))
        X.append(np.array(img))
        y_label.append(str(label))

# 读入10个目录中的狗狗图像
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

# 构建X和y张量
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_label) # 标签编码
y = to_categorical(y, 10) # 将标签转换为One-hot编码, 注意此处是10个标签所以是10，如果有更多label，那么数字大于10
X = np.array(X)  # 将X从列表转换为张量数组
# 相当于是手工将图像的像素值进行简单的压缩，也就是将X张量进行归一化，以利于神经网络处理它。
X = X/255 # 将X张量归一化， 像素值通常表示为0到255之间的整数，其中0表示黑色，255表示白色 在深度学习模型中，特别是在卷积神经网络（CNN）中，输入数据的尺度对于模型的表现非常重要。将图像像素值归一化到0到1之间（通常是除以255）有以下几个好处：

# 显示向量化之后的图像
print('X张量的形状：', X.shape)
print ('X张量中的第一个数据', X[1])

print ('y张量的形状：', y.shape)
print ('y张量中的第一个数据', y[1])

# 将已经缩放到[0，1]区间之后的张量重新以图像的形式显示出来
# 随机显示狗狗图片
fig, ax = plt.subplots(5 ,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range(2):
        r = rdm.randint(0, len(X))
        ax[i,j].imshow(X[r])
        ax[i,j].set_title('Dog: '+y_label[r])
plt.tight_layout()

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


model = keras.models.load_model('./data/models/doggy_04.h5')

# 绘制特征通道
layer_outputs = [layer.output for layer in model.layers[:16]]
image = X_train[0]
image = image.reshape(1, 150, 150, 3)
activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(image)
first_layer = activations[0]
plt.matshow(first_layer[0, :, :, 2], cmap='viridis')
plt.matshow(first_layer[0, :, :, 3], cmap='viridis')
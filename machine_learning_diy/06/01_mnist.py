from keras import models
from keras.src.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = models.Sequential() # 创建序贯模型
model.add(Conv2D(filters=32, # 添加Conv2D层，指定过滤器个数，即通道数量
                 kernel_size=(3,3), # 指定卷积核大小
                 activation='relu', # 指定激活函数
                 input_shape=(28,28,1))) # 指定输入数据样本张量的形状
model.add(MaxPooling2D(pool_size=(2,2))) # 添加MaxPolling2D层
model.add(Conv2D(64,(3,3), activation='relu')) # 添加Conv2D层
model.add(MaxPooling2D(pool_size=(2,2))) # 添加MaxPolling2D层
model.add(Dropout(0.25)) # 添加Dropout层
model.add(Flatten()) # 添加展平层
model.add(Dense(128, activation='relu')) # 添加全连接层，输出128
model.add(Dropout(0.5)) # 添加Dropout层
model.add(Dense(10, activation='softmax')) # Softmax分类 激活, 输出10维分类码

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary() # 显示网络模型

from IPython.display import SVG
from keras.src.utils.model_visualization import model_to_dot

svg = SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

with open('ann.svg', 'w') as f:
    f.write(svg.data)
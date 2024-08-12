import numpy as np
import pandas as pd

df_train = pd.read_csv('./data/exoTrain.csv')
df_test = pd.read_csv('./data/exoTest.csv')

print(df_train.head())
print(df_train.info()) # 输出训练前信息

# 数据集是预先排序过的，所以需要先打乱
from sklearn.utils import shuffle # 导入乱序工具
df_train = shuffle(df_train) # 乱序训练集
df_test = shuffle(df_test) # 乱序测试集

# 构建特征集与标签集
# 注意标签集分类 2-有行星，1-没有行星，所以要将标签值减1，将(1,2)转换成惯用的(0,1)分类
X_train = df_train.iloc[:, 1:].values # 构建训练-特征集
y_train = df_train.iloc[:, 0].values # 构建训练-标签集
X_test = df_test.iloc[:, 1:].values # 构建验证-特征集
y_test = df_test.iloc[:, 0].values # 构建验证-标签集
y_train = y_train - 1 # 标签转换成惯用的(0, 1)分类值
y_test = y_test - 1 # 标签转换成惯用的(0, 1)分类值
print(X_train, ' X_train 形状: ', X_train.shape, ' 阶: ', X_train.ndim)
print(y_train, ' y_train 形状: ', y_train.shape, ' 阶: ', y_train.ndim)

# 一定要注意张量的形状
# 此处为时序数据，其张量一定要是3阶
X_train = np.expand_dims(X_train, axis=2) # 张量升阶，已满足序列数据集的要求
X_test = np.expand_dims(X_test, axis=2) # 张量升阶，已满足序列数据集的要求

# 5 087个样本，3 197个时戳，1维的特征(光线的强度) ? 光线强度？
print(X_train, ' X_train 形状: ', X_train.shape, ' 阶: ', X_train.ndim)
print(X_test, ' X_test 形状: ', X_test.shape, ' 阶: ', X_test.ndim)

# 这个案例不需要对特征进行标准化（归一化）的原因，可能本身数据分布就比较好，或者说冥冥中自然带有它的规律
# 此时有一个同学举手发问:“咖哥，不需要进行特征缩放吗?”
# 咖哥的脸上露出一丝痛苦的神色，说:“原则上，我们应进行数据 的标准化，
# 再把数据输入神经网络。但是，就这个特定的问题而言，我 经过无数次的调试后发现，
# 这个例子中不进行数据的缩放，就我目前的 模型来说反而能够起到更好的效果。
# 毕竟，这是一个很不寻常的问题， 涉及系外行星的寻找......”

# 使用一维卷积网络作为预处理步骤，把长序 列提取成短序列，并把有用的特征交给循环神经网络来继续处理。
from keras.src.models import Sequential # 导入序贯模型
from keras.src import layers # 导入所有类型的层
from keras.src.optimizers import Adam # 导入优化器

model = Sequential() # 序贯模型
model.add(layers.Conv1D(32,
                        kernel_size=10,
                        strides=4,
                        input_shape=(3197,1))) # 1D CNN层
model.add(layers.MaxPooling1D(pool_size=4, strides=2)) # 池化层
model.add(layers.GRU(256, return_sequences=True)) # GRU层要足够大
model.add(layers.Flatten()) # 展平层
model.add(layers.Dropout(0.5)) # Dropout层
model.add(layers.BatchNormalization()) # 批标准化，防止过拟合
model.add(layers.Dense(1, activation='sigmoid')) # 分类输出层
opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01) # 设置优化器
model.compile(optimizer=opt, # 优化器
              loss='binary_crossentropy', # 交叉熵
              metrics=['accuracy']) # 准确率

history = model.fit(X_train, y_train, # 指定训练集
          validation_split=0.2, # 部分训练集数据拆分成验证集
          batch_size=128, # 指定批量大小
          epochs=4, # 指定轮次
          shuffle=True) # 乱序？ 上面不是已经将训练集数据做了乱序处理了吗？


# 输出阈值的调整
from sklearn.metrics import classification_report # 分类报告
from sklearn.metrics import confusion_matrix # 混淆矩阵

y_prob = model.predict(X_test) # 对测试集进行预测
y_pred = np.where(y_prob > 0.5, 1.0, 0.0) # 将概率值转换为真值
cm = confusion_matrix(y_pred, y_test)
print('Confusion matrix:\n', cm, '\n')
print(classification_report(y_pred, y_test))

# 通过观察y_prod的数据集，将阈值从0.5调整到0.3，再调整到0.2 此时，5颗近日行星找到了3颗
# 输出阈值的调整
from sklearn.metrics import classification_report # 分类报告
from sklearn.metrics import confusion_matrix # 混淆矩阵

y_prob = model.predict(X_test) # 对测试集进行预测
y_pred = np.where(y_prob > 0.2, 1.0, 0.0) # 将概率值转换为真值
cm = confusion_matrix(y_pred, y_test)
print('Confusion matrix:\n', cm, '\n')
print(classification_report(y_pred, y_test))


# 使用函数式API


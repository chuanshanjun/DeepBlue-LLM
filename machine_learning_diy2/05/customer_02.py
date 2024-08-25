import keras
from keras.src.models import Sequential
from keras.src.layers import Dense

ann = Sequential() # 创建一个序贯ANN模型
ann.add(Dense(units=12, input_dim=12, activation='relu')) # 添加一个输入层
ann.add(Dense(units=24, activation='relu')) # 添加一个隐藏层
ann.add(Dense(units=1, activation='sigmoid')) # 添加一个输出层
ann.summary() # 显示网络模型

# 编译神经网络, 指定优化器、损失函数, 以及评估指标

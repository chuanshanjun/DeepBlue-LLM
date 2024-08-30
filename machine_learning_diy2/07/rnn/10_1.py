import os

# 10_1 查看耶纳天气数据集

fname = os.path.join('./data/jena_climate_2009_2016.csv')

with open(fname) as f:
    data = f.read()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
print(header)
print(len(lines))

# 10_2 解析数据
# 构造特征集，温度为标签，其他为特征值，注意去除Date Time
import numpy as np

temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]  #
    temperature[i] = values[1]
    raw_data[i, :] = values[:]

# 10_3 绘制温度时间序列
import matplotlib.pyplot as plt

plt.plot(range(len(temperature)), temperature)
plt.show()

# 绘制前10天的温度时间序列
# 数据10分钟记录一次, 一天= 24*6 = 144
plt.plot(range(1440), temperature[:1440])
plt.show()

# 注：始终在数据中寻找周期性
# 10-5 计算用于训练、验证和测试的样本数
num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples
print("num_train_samples: ", num_train_samples)
print("num_val_samples: ", num_val_samples)
print("num_test_samples: ", num_test_samples)

# 数据规范化
mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std

# 问题描述： 每小时采样一次数据，给定前5天的数据，我们能否预测24小时之后的温度？
# 理解timeseries_dataset_from_array()
import numpy as np
from tensorflow import keras

int_sequence = np.arange(10)
dummy_dataset = keras.utils.timeseries_dataset_from_array(
    data=int_sequence[:-3],
    targets=int_sequence[3:],
    sequence_length=3,
    batch_size=2,
)

for inputs, targets in dummy_dataset:
    for i in range(inputs.shape[0]):
        print([int(x) for x in inputs[i]], int(targets[i]))

# 上面这个，关键是理解，创建一个数据集，并预测时间序列的下一份数据
# 如 [0, 1, 2] 3 ,[0, 1, 2]即为数据集，而3则是 预测时间序列的下一份数据

# 使用timeseries_dataset_from_array()b创建3个数据集，分别用来训练、验证和测试

# 10_7 创建3个数据集，分别用于训练、验证和测试
sampling_rate = 6  # 观测数据的采样频率是每小时一个数据点，也就是说，每6个数据点保留1个
sequence_length = 120  # 给定过去5天（120小时）的观测数据
delay = sampling_rate * (sequence_length + 24 - 1)  # 序列的目标是序列结束24小时之后的温度
batch_size = 256

train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples
)

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples
)

test_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples
)

# 查看一个数据集的输出
for samples, targets in train_dataset:
    print('samples shape: ', samples.shape)
    print('targets shape: ', targets.shape)
    break

# 使用平均绝对误差MAE 作为损失函数
# 10_9 计算基于常识的基准的MAE
def evaluate_native_method(dataset):
    total_abs_err = 0
    samples_seen = 0
    for samples, targets in dataset:
        preds = samples[:, -1, 1] * std[1] + mean[1]
        total_abs_err += np.sum(np.abs(preds - targets))
        samples_seen += samples.shape[0]
    return total_abs_err / samples_seen

print(f"Validation MAE: {evaluate_native_method(val_dataset):.2f}")
print(f"Test MAE: {evaluate_native_method(test_dataset):.2f}")

# 10-10 训练并评估一个密集连接模型
from tensorflow.keras import layers
# inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
# x = layers.Flatten()(inputs)
# x = layers.Dense(16, activation='relu')(x)
# outputs = layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint('jena_dense.keras', save_best_only=True)
# ]
# model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
# history = model.fit(train_dataset,
#                     epochs=10,
#                     validation_data=val_dataset,
#                     callbacks=callbacks)
#
# model = keras.models.load_model('jena_dense.keras')
# print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")

# 10-11 绘制结果
# import matplotlib.pyplot as plt
# loss = history.history['mae']
# val_loss = history.history['val_mae']
# epochs = range(1, len(loss) + 1)
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training MAE')
# plt.plot(epochs, val_loss, 'b', label='Validation MAE')
# plt.title('Training and validation MAE')
# plt.legend()
# plt.show()


# 10-12 基于LSTM的简单模型
# inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
# x = layers.LSTM(16)(inputs)
# outputs = layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint('jena_dense_lstm.keras', save_best_only=True)
# ]
#
# model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
# history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=callbacks)
#
# print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")
#
# import matplotlib.pyplot as plt
# loss = history.history['mae']
# val_loss = history.history['val_mae']
# epochs = range(1, len(loss) + 1)
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training MAE')
# plt.plot(epochs, val_loss, 'b', label='Validation MAE')
# plt.title('Training and validation MAE')
# plt.legend()
# plt.show()

# 10-13 RNN 伪代码
# state_t = 0 # t的状态
# for input_t in input_sequence:
#     output_t = f(input, state_t)
#     state_t = output_t

# 10-14 更详细的RNN伪代码
# state_t = 0
# for input_t in input_sequence:
#     output_t = actiation(dot(W, input) + dot(U, state_t), b)
#     state_t = output_t

# 10-15 简单RNN的NumPy实现
# timesteps = 100
# input_features = 32
# output_features = 64
# inputs = np.random.random((timesteps, input_features))
# state_t = np.zeros((output_features),)
# W = np.random.random((output_features, input_features))
# U = np.random.random((output_features, output_features))
# b = np.random.random((output_features),)
# successive_outputs = []
# for input_t in inputs:
#     output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
#     successive_outputs.append(output_t)
#     state_t = output_t
# final_output_sequence = np.stack(successive_outputs, axis=0)

# 10-16 能够处理任意长度序列的RNN
# num_features = 14
# inputs = keras.Input(shape=(None, num_features))
# out = layers.SimpleRNN(16) (inputs)

# 10-17 只返回最后一个时间步输出的RNN层
# num_features = 14
# steps = 120
# inputs = keras.Input(shape=(steps, num_features))
# outputs = layers.SimpleRNN(16, return_sequences=False)(inputs)
# model = keras.Model(inputs, outputs)
# model.summary()
# print(outputs.shape)

# 10-18 返回完整输出序列的RNN层
# num_features = 14
# steps = 120
# inputs = keras.Input(shape=(steps, num_features))
# outputs = layers.SimpleRNN(16, return_sequences=True)(inputs)
# model = keras.Model(inputs, outputs)
# model.summary()
# print(outputs.shape)

# 10-19 RNN层堆叠
# inputs = keras.Input(shape=(steps, num_features))
# x = layers.SimpleRNN(16, return_sequences=True)(inputs)
# x = layers.SimpleRNN(16, return_sequences=True)(x)
# outputs = layers.SimpleRNN(16)(x)
# model = keras.Model(inputs, outputs)
# model.summary()
# print(outputs.shape)

# 10-22 训练并评估一个使用dropout正则化的LSTM模型
# inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
# x = layers.LSTM(32, recurrent_dropout=0.25)(inputs)
# x = layers.Dropout(0.5)(x) # 这里在LSTM层之后还添加一个Dropout层，对Dense层进行正则化
# outputs = layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)
# model.summary()
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint('jena_lstm_dropout.keras', save_best_only=True)
# ]
#
# model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
# history = model.fit(train_dataset,
#                     epochs=50,
#                     validation_data=val_dataset,
#                     callbacks=callbacks)
#
# print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")
#
# import matplotlib.pyplot as plt
# loss = history.history['mae']
# val_loss = history.history['val_mae']
# epochs = range(1, len(loss) + 1)
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training MAE')
# plt.plot(epochs, val_loss, 'b', label='Validation MAE')
# plt.title('Training and validation MAE')
# plt.legend()
# plt.show()

# 10-23 训练并评估一个使用dropout 正则化的堆叠GRU模型
inputs = keras.Input(shape=(sequence_length, raw_data[-1]))
x = keras.GRU(32, recurrent_dropout=0.5, return_sequences=True)(inputs)
x = layers.GRU(32, recurrent_dropout=0.5) (x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('jena_stacked_gru_dropout.keras', save_best_only=True)
]

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=val_dataset,
                    callbacks=callbacks)
model = keras.models.load_model('jena_stacked_gru_dropout.keras')
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")

import matplotlib.pyplot as plt
loss = history.history['mae']
val_loss = history.history['val_mae']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training MAE')
plt.plot(epochs, val_loss, 'b', label='Validation MAE')
plt.title('Training and validation MAE')
plt.legend()
plt.show()

# 10-24 训练并评估双向LSTM
inputs = keras.Input(shape=(sequence_length, raw_data[-1]))
x = layers.Bidirectional(layers.LSTM(16))(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint('jena_bidirectional_lstm.keras', save_best_only=True)
]

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=val_dataset,
                    callbacks=callbacks)

model = keras.models.load_model('jena_bidirectional_lstm.keras')
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")

import matplotlib.pyplot as plt
loss = history.history['mae']
val_loss = history.history['val_mae']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training MAE')
plt.plot(epochs, val_loss, 'b', label='Validation MAE')
plt.title('Training and validation MAE')
plt.legend()
plt.show()
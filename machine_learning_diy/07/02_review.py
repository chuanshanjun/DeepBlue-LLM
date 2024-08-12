import pandas as pd
import numpy as np

dir = './data/'

dir_train = dir + 'Clothing Reviews.csv'

df_train = pd.read_csv(dir_train)  # 读入训练集

print(df_train.head()) # 输出部分数据

# 对数据集进行分词工作，词典的大小设置2万
from tensorflow.keras.preprocessing.text import Tokenizer # 导入分词工具

X_train_lst = df_train["Review Text"].values # 将评论读入张量（训练）

y_train = df_train["Rating"].values # 构建标签集

dictionary_size = 20000 # 设定词典大小

tokenizer = Tokenizer(num_words=dictionary_size) # 初始化词典

tokenizer.fit_on_texts(X_train_lst) # 使用训练集创建词典索引

X_train_tokenized_lst = tokenizer.texts_to_sequences(X_train_lst) # 为所有的单词分配索引值，完成分词工作

print(X_train_tokenized_lst[1])

print('y_train shape: ', y_train.shape, ' 阶: ', y_train.ndim)

import matplotlib.pyplot as plt

word_per_comment = [len(comment) for comment in X_train_tokenized_lst]

plt.hist(word_per_comment, bins=np.arange(0, 500, 10)) # 显示评论长度分布
plt.show()

from tensorflow.keras.preprocessing.sequence import pad_sequences

max_comment_length = 120 # 设定评论输入长度为120，并填充默认值（如字数少于120）

X_train = pad_sequences(X_train_tokenized_lst, max_comment_length)

# 构建包含词潜入的SimpleRNN
from keras.src.models import Sequential       # 导入序贯模型
from tensorflow.keras.layers import Embedding # 导入词嵌入层
from keras.src.layers import Dense            # 导入全连接层
from keras.src.layers import SimpleRNN        # 导入Simple RNN层

# embedding_vector_length = 60 # 设定词嵌入向量长度为60
# rnn = Sequential() # 序贯模型
# rnn.add(Embedding(dictionary_size,
#                   embedding_vector_length,
#                   input_length=max_comment_length)) # 加入词嵌入层
#
# rnn.add(SimpleRNN(100))                       # 加入Simple RNN层
# rnn.add(Dense(10, activation='relu'))   # 加入全连接层
# rnn.add(Dense(6, activation='softmax')) # 加入分类输出层
# rnn.compile(loss='sparse_categorical_crossentropy', # 损失函数
#             optimizer='adam', # 优化器
#             metrics=['acc']) # 评估指标
# print(rnn.summary()) # 输出网络
#
# # 构建网络，开始训练
# history = rnn.fit(X_train, y_train, validation_split=0.3, epochs=10, batch_size=64)


# 从SimpleRNN到LSTM
# 使用LSTM鉴定评论文本
# 不改变任何其他网络参数，仅是使用LSTM层替换SimpleRNN层
from keras.src.layers import LSTM

embedding_vector_length = 60 # 设定词嵌入向量长度为60
rnn = Sequential() # 序贯模型
rnn.add(Embedding(dictionary_size,
                  embedding_vector_length,
                  input_length=max_comment_length)) # 加入词嵌入层

rnn.add(LSTM(100))                       # 加入Simple RNN层
rnn.add(Dense(10, activation='relu'))   # 加入全连接层
rnn.add(Dense(6, activation='softmax')) # 加入分类输出层
rnn.compile(loss='sparse_categorical_crossentropy', # 损失函数
            optimizer='adam', # 优化器
            metrics=['acc']) # 评估指标
print(rnn.summary()) # 输出网络

# 构建网络，开始训练
history = rnn.fit(X_train, y_train, validation_split=0.3, epochs=10, batch_size=64)



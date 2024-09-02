import pandas as pd
import numpy as np
from keras import Model

dir = './data/'
dir_train = dir + 'Clothing Reviews.csv'
df_train = pd.read_csv(dir_train)
print(df_train.head())

from tensorflow.keras.preprocessing.text import Tokenizer

X_train_lst = df_train['Review Text'].values
y_train = df_train['Rating'].values
dictionary_size = 20000 # 设定词典大小
tokenizer = Tokenizer(num_words=dictionary_size) # 初始化词典
tokenizer.fit_on_texts(X_train_lst) # 使用训练集创建词典索引

# 为所有的单词分配索引值
X_train_tokenized_lst = tokenizer.texts_to_sequences(X_train_lst)

print(X_train_tokenized_lst[110:115])

# 通过直方图显示各个评论中单词个数的分布情况
import matplotlib.pyplot as plt
word_per_comment = [len(comment) for comment in X_train_tokenized_lst]
plt.hist(word_per_comment, bins=np.arange(0,500,10)) # 显示评论长度分布
plt.show()

from tensorflow.keras.preprocessing.sequence import pad_sequences
max_comment_length = 120 # 设定评论输入长度为120，并填充默认值(如字数少于120)
X_train = pad_sequences(X_train_tokenized_lst, max_comment_length)

# 构建包含词嵌入的SimpleRNN
from keras.src.models import Sequential
from tensorflow.keras.layers import Embedding # 导入词嵌入层
from keras.src.layers import SimpleRNN,LSTM,Dense         # 导入Simple RNN层
import keras

embedding_vector_length = 60 # 设定词嵌入向量长度为60
rnn = Sequential() # 序贯模型
rnn.add(Embedding(dictionary_size,  # 20000
        embedding_vector_length,  # 60
        input_length=max_comment_length # 120
                  ))  # 加入词嵌入层
# 一种堆叠方法 -> SimpleRNN
# rnn.add(SimpleRNN(100)) # 加入simpleRNN层，100是神经元个数(也可以认为是输出)
# rnn.add(Dense(10, activation='relu')) # 加入全连接层
# rnn.add(Dense(6, activation='softmax')) # 因为评论类型1-5， 其实只要5个神经元就可以了
# rnn.compile(
#         loss='sparse_categorical_crossentropy',
#         optimizer='adam', # 优化器
#         metrics=['acc'] # 评估指标
# )
#
# print(rnn.summary())
#
# history = rnn.fit(X_train, y_train,
#                   validation_split=0.3,
#                   epochs=10,
#                   batch_size=64)

# 使用lambda堆叠 -> SimpleRNN
# inputs = keras.Input(shape=(max_comment_length,))
# embedded = Embedding(dictionary_size, embedding_vector_length, input_length=max_comment_length)(inputs)
# rnn_output = SimpleRNN(100)(embedded)
# dense1 = Dense(10, activation='relu')(rnn_output)
# outputs = Dense(6, activation='softmax')(dense1)
# model = Model(inputs=inputs, outputs=outputs)
# model.compile(
#         loss='sparse_categorical_crossentropy',
#         optimizer='adam',
#         metrics=['acc']
# )
#
# print(model.summary)
# history = model.fit(
#         X_train, y_train,
#         validation_split=0.3,
#         epochs=10,
#         batch_size=64
# )

# 使用LSTM
lstm = Sequential()
lstm.add(Embedding(dictionary_size, embedding_vector_length, input_length=max_comment_length))
lstm.add(LSTM(100))
lstm.add(Dense(10, activation='relu'))
lstm.add(Dense(6, activation='softmax'))
lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = lstm.fit(X_train, y_train, validation_split=0.3, epochs=10, batch_size=64)

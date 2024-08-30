import pandas as pd
import numpy as np

dir = './data/'
dir_train = dir + 'Clothing Reviews.csv'
df_train = pd.read_csv(dir_train)
print(df_train.head())

from keras.preprocessing.text import Tokenizer

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

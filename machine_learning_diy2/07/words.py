from keras.preprocessing.text import Tokenizer

words = ['Lao Wang has a Wechat account.', 'He is not a nice person.', 'Be careful.']
tokenizer = Tokenizer(num_words=30) # 词典大小只设定30个词(因为句子数量少)
tokenizer.fit_on_texts(words) # 根据3个句子编辑词典
sequences = tokenizer.texts_to_sequences(words) # 为3个句子根据词典里面的索引进行序号编码
one_hot_metrix = tokenizer.texts_to_matrix(words, mode='binary') # 进行one-hot 编码
word_index = tokenizer.word_index # 词典中的单词索引总数
print('找到了%s个单词: ', len(word_index))
print('这3句话(单词)的序号编码: ', sequences)
print('这3句话(单词)的one-hot编码: ', one_hot_metrix)
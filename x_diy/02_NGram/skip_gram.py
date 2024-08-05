# 创建实验语料库
# 定义一个句子列表
sentences = ["Kage is Teacher", "Mazong is Boss", "Niuzong is Boss",
            "Xiaobing is Student", "Xiaoxue is Student"]

# 将所有句子连接在一起，然后用空格分隔成多个单词
words = ' '.join(sentences).split()

# 构建词汇表，去除重复的词
word_list = list(set(words))

# 创建一个字典，将每个词映射到一个唯一的索引
word_to_idx = {word: idx for idx, word in enumerate(word_list)}

# 创建一个字典，将每个索引映射到对应的词汇
idx_to_word = {idx: word for idx, word in enumerate(word_list)}

# 计算词汇表大小
voc_size = len(word_list)

# 输出词汇表
print("词汇表: ", word_list)

# 输出词汇到索引的字典
print("词汇到索引的字典: ", word_to_idx)

# 输出索引到词汇的字典
print("索引到词汇的字典: ", idx_to_word)

# 词汇表大小
print("词汇表大小: ", voc_size)


# 生成skip-gram数据
# 生成skip-gram训练数据
def create_skipgram_dataset(sentences, window_size=2):
    # 初始化数据集
    data = []
    # 遍历句子
    for sentence in sentences:
        # 将句子分割成单词列表
        sentence = sentence.split()
        # 遍历单词及索引
        for idx, word in enumerate(sentence):
            # 获取相邻的单词，将当前单词前后各N个单词作为相邻单词
            for neighor in sentence[max(idx - window_size, 0):
                min(idx + window_size + 1, len(sentence))]:
                # 排除当前单词本身
                if neighor != word:
                    # 将相邻单词与当前单词作为一组训练数据
                    data.append((neighor, word))
    return data

# 使用函数创建skip-gram训练数据
skipgram_data = create_skipgram_dataset(sentences)
# 打印未编码的skip-gram数据样本
print("Skip-Gram数据样例（未编码）： ", skipgram_data)

# one-hot编码
# 把上面的skip-gram训练数据转换成skip-gram可以读入的one-hot编码后的向量
# 定义One-Hot编码函数

# 导入torch库
import torch
def one_hot_encoding(word, word_to_idx):
    # 创建一个长度与词汇表相同的全0张量
    tensor = torch.zeros(len(word_to_idx))

import pandas as pd

df_housing = pd.read_csv("https://raw.githubusercontent.com/huangjia2019/house/master/house.csv")

# 显示加州房价数据


print(df_housing.head)

# 构建特征集X
df_housing.drop("median_house_value", axis = 1)

# 构建特征集X
X = df_housing.drop("median_house_value", axis = 1)

# 构建标签集y
y = df_housing.median_house_value

from sklearn.model_selection import train_test_split

train_test_split(X, y, test_size=0.2 random_stat=0)


# 构建一个玩具数据集
corpus = ["我喜欢吃苹果",
          "我喜欢吃香蕉",
          "她喜欢吃葡萄",
          "他不喜欢吃香蕉",
          "他喜欢吃苹果",
          "他喜欢吃草莓"]

# 把句子分为N个"Gram"(分词)

# 将句子转换为单个汉字
# 定义一个分词函数，将文本转换为单字的列表
def tokenize(text):
    return [char for char in text]  # 将文本拆分为单字列表

# 对每个文本进行分词，并打印出对应的单字列表
print("单字列表:")
for text in corpus:
    tokens = tokenize(text)
    print(tokens)

# 计算每个Bigram在语料库中的词频
from collections import defaultdict, Counter

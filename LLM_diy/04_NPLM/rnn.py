# 第1步
# 构建一个非常简单的数据集
from tensorflow_estimator.python.estimator.util import parse_input_fn_result

sentences = ["我 喜欢 玩具 ", "我 爱 爸爸 ", "我 讨厌 挨打"]
# 将所有句子连接在一起，用空格分隔成多个词，再将重复的词去除，构建词汇表
word_list = list(set("".join(sentences).split()))
# 创建一个字典，将每个词映射到一个唯一的索引
word_to_idx = {word: idx for idx, word in enumerate(word_list)}
# 创建一个字典，将每个索引映射到对应的词
idx_to_word = {idx: word for idx, word in enumerate(word_list)}
# 计算词汇表大小
voc_size = len(word_list)
# 打印词汇表到索引的映射字典
print('词汇表: ', word_to_idx)
# 打印词汇表大小
print('词汇表大小: ', voc_size)

#第2步 生成NPLM训练数据
# 构建批处理数据
import torch
import random
batch_size = 2
def make_batch():
    input_batch = []
    target_batch = []
    selected_sentences = random.sample(sentences, batch_size)
    for sen in selected_sentences:
        word = sen.split()
        # 将除最后一个词以外的所有词的索引作为输入
        input = [word_to_idx[n] for n in word[:-1]]
        # 将最后一个词的索引作为目标
        target = word_to_idx[word[-1]]
        input_batch.append(input)
        target_batch.append(target)
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)
    return input_batch, target_batch

input_batch, target_batch = make_batch()
print('输入批处理数据：', input_batch) # 打印输入批处理数据
# 将输入批处理数据中的每个索引值转换为对应的原始词
input_words = []
for input_idx in input_batch:
    input_words.append([idx_to_word[idx.item()] for idx in input_idx])
print('输入批处理数据对应的原始词: ', input_words)
print('目标批处理数据：', target_batch)
# 将目标批处理数据中的每个索引值转换为对应的原始词
target_words = [idx_to_word[idx.item()] for idx in target_batch]
print('目标批处理数据对应的原始词: ', target_words)

# 第3步 定义NPLM
import torch.nn as nn
# 定义神经概率语言模型
class NPLM(nn.Module):
    def __init__(self):
        super(NPLM, self).__init__()
        self.C = nn.Embedding(voc_size, embedding_size) # 定义一个词嵌入层
        # 用LSTM层替代第一个线性层，其输入大小为embedding_size,隐藏层大小为n_hidden
        # 使用lstm的时候，只要指定embedding_size 尺寸的大小 根本不需要像 nplm 那样根据
        # n_step（输入序列的长度）指定网络结构
        self.lstm = nn.LSTM(embedding_size, n_hidden, batch_first=True)
        # 第二个线性层，其输入大小为n_hidden，输出大小为voc_size，即词汇表大小
        self.linear = nn.Linear(n_hidden, voc_size)

    def forward(self, X):
        # 输入数据X张量的形状为[batch_size, n_step]
        X = self.C(X) # 将X通过词嵌入层，形状变为[batch_size, n_step, embedding_size]
        # 通过LSTM层
        lstm_out, _ = self.lstm(X) # lstm_out 形状变为 [batch_size, n_step, n_hidden]
        # 只选择最后一个时间步的输出作为全连接层的输入，通过第二个线性层得到输出
        # -1 可能就是指最后一个时间步
        output = self.linear(lstm_out[:,-1,:]) # output 形状为 [batch_size, voc_size]
        return output

# 第4步 实例化NPLM
n_step = 2 # 时间步数，表示每个输入序列的长度，也就是上下文长度
n_hidden = 2 # 隐藏层大小
embedding_size = 2 # 词嵌入大小
model = NPLM() # 创建神经概率语言模型实例
print('RNN模型结构： ', model)

# 第5步 训练NPLM
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
# 训练模型
for epoch in range(5000):
    optimizer.zero_grad() # 清除优化器的梯度，梯度！
    input_batch, target_batch = make_batch()
    output = model(input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step() # 更新模型参数 -> 更新权重

# 第6步 用NPLM预测新词
# 进行预测
input_strs = [['我', '讨厌'], ['我', '喜欢']] # 需要预测的输入序列
# 将输入序列转换为对应的索引
input_indices = [[word_to_idx[word] for word in seq ] for seq in input_strs]
# 将输入序列的索引转换为张量
input_batch = torch.LongTensor(input_indices)
# 对输入序列进行预测，取输出中概率最大的类别
predict = model(input_batch).data.max(1)[1]
# 将预测结果的索引转换为对应的词
predict_strs = [idx_to_word[n.item()] for n in predict.squeeze()]
for input_seq, pred in zip(input_strs, predict_strs):
    print(input_seq, '->', pred)


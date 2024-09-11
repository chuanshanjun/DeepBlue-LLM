# 第1步 构建实验语料库和词汇表
# 构建语料库，每行包含中文、英文（解码器输入）和翻译成英文后的目标输出3个句子
sentences = [
    ['咖哥 喜欢 小冰', '<sos> KaGe likes XiaoBing', 'KaGe likes XiaoBing <eos>'],
    ['我 爱 学习 人工智能', '<sos> I love studying AI', 'I love studying AI <eos>'],
    ['深度 学习 改变 世界', '<sos> Deep learning YYDS', 'Deep learning YYDS <eos>'],
    ['自然 语言 处理 很 强大', '<sos> NLP is so powerful', 'NLP is so powerful <eos>'],
    ['神经网络 非常 复杂', '<sos> Neural-Net are complex', 'Neural-Net are complex <eos>']]

word_list_cn, word_list_en = [], [] # 初始化中英文词汇表
# 遍历每一个句子并将单词添加到词汇表中
for s in sentences:
    word_list_cn.extend(s[0].split())
    word_list_en.extend(s[1].split())
    word_list_en.extend(s[2].split())

# 去重，得到没有重复单词的词汇表
word_list_cn = list(set(word_list_cn))
word_list_en = list(set(word_list_en))

# 构建单词到索引的映射
word2idx_cn = {w: i for i, w in enumerate(word_list_cn)}
word2idx_en = {w: i for i, w in enumerate(word_list_en)}

# 构建索引到单词的映射
idx2word_cn = {i: w for i, w in enumerate(word_list_cn)}
idx2word_en = {i: w for i, w in enumerate(word_list_en)}

# 计算词汇表的大小
voc_size_cn = len(word_list_cn)
voc_size_en = len(word_list_en)

print('句子数量: ', len(sentences))
print('中文词汇表大小: ', voc_size_cn)
print('英文词汇表大小：', voc_size_en)
print('中文词汇表到索引的字典：', word2idx_cn)
print('英文词汇表到索引的字典：', word2idx_en)

# 第2步 生成Seq2Seq训练数据
import numpy as np
import torch
import random
# 定义一个函数，随机选择一个句子和词汇表生成输入、输出和目标数据
def make_data(sentences):
    # 随机选择一个句子进行训练
    random_sentence = random.choice(sentences)
    # 将输入句子中的单词转换为对应的索引(np array 类型)
    encoder_input = np.array([[word2idx_cn[w] for w in random_sentence[0].split()]])
    # 将输出句子中的单词转换为对应的索引
    decoder_input = np.array([[word2idx_en[w] for w in random_sentence[1].split()]])
    # 将目标句子中的单词转换为对应的索引
    target = np.array([[word2idx_en[w] for w in random_sentence[2].split()]])

    # 将输入、输出和目标批次转换为LongTensor
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input)
    target = torch.LongTensor(target)
    return encoder_input,decoder_input,target

# 使用make_data 函数生成输入、输出和目标张量
encoder_input, decoder_input, target = make_data(sentences)
for s in sentences: # 获取原始句子
    if all([word2idx_cn[w] in encoder_input[0] for w in s[0].split()]):
        original_sentence = s
        break

print("原始句子: ", original_sentence) # 打印原始句子
print("编码器输入张量的形状: ", encoder_input.shape)
print("解码器输入张量的形状: ", decoder_input.shape)
print("目标张量的形状: ", target.shape)
print("编码器输入张量: ", encoder_input)
print("解码器输入张量: ", decoder_input)
print("目标张量: ", target)

# 第3步 定义编码器和解码器类
import torch.nn as nn
# 定义编码器类，继承自nn.Module
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs) # 将输入转换为嵌入向量
        output, hidden = self.rnn(embedded, hidden) # 将嵌入向量输入RNN层并获取输出
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        output = self.out(output)
        return output, hidden

n_hidden = 128
# 创建编码器和解码器
encoder = Encoder(voc_size_cn, n_hidden)
decoder = Decoder(n_hidden, voc_size_en)
print('编码器结构: ', encoder)
print('解码器结构: ', decoder)

# 第4步 定义Seq2Seq架构
# 组合编码器和解码器，形成Seq2Seq架构

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        # 初始化编码器和解码器
        self.encoder = encoder
        self.decoder = decoder
    # 定义前向传播函数
    def forward(self, encoder_input, hidden, decoder_input):
        # 将输入序列通过编码器并获取输出和隐藏状态
        encoder_output, encoder_hidden = self.encoder(encoder_input, hidden)
        # 将编码器的隐藏状态传递给解码器作为初始隐藏状态
        decoder_hidden = encoder_hidden
        # 将目标序列通过解码器并获取输出
        decoder_output, _ = self.decoder(decoder_input, decoder_hidden)
        return decoder_output

# 创建Seq2Seq模型
model = Seq2Seq(encoder, decoder)
print('S2S模型结构：', model)  # 打印模型的结构


# 定义训练函数
def train_seq2seq(model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        encoder_input, decoder_input, target = make_data(sentences) # 训练数据的创建
        hidden = torch.zeros(1, encoder_input.size(0), n_hidden) # 初始化隐藏状态
        optimizer.zero_grad()# 梯度清零
        output = model(encoder_input, hidden, decoder_input) # 获取模型输出
        loss = criterion(output.view(-1, voc_size_en), target.view(-1)) # 计算损失
        if (epoch + 1) % 100 == 0: # 打印损失
            print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
        loss.backward()# 反向传播
        optimizer.step()# 更新参数

# 训练模型
epochs = 1000 # 训练轮次
criterion = nn.CrossEntropyLoss() # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 优化器
train_seq2seq(model, criterion, optimizer, epochs) # 调用函数训练模型

# 定义测试函数
def test_seq2seq(model, source_sentence, word_dict, number_dict):
    # 将输入句子转换为索引
    encoder_input = np.array([[word2idx_cn[n] for n in source_sentence.split()]])
    # 构建输出句子的索引，以'<sos>'开始，后面跟'<eos>'，长度与输入句子相同
    decoder_input = np.array([word2idx_en['<sos>']] + [word_dict['<eos>']]*(len(encoder_input[0])-1))
    # 转换为LongTensor
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input).unsqueeze(0) # 增加一维
    hidden = torch.zeros(1, encoder_input.size(0), n_hidden) # 初始化隐藏状态
    predict = model(encoder_input, hidden, decoder_input) # 获取模型输出
    predict = predict.data.max(2, keepdim=True)[1] # 获取最大概率的索引
    # 打印输入句子和预测的句子
    print(source_sentence, '->', [number_dict[n.item()] for n in predict.squeeze()])

# 测试模型
test_seq2seq(model, '咖哥 喜欢 小冰', word2idx_en, idx2word_en)
test_seq2seq(model, '自然 语言 处理 很 强大', word2idx_en, idx2word_en)
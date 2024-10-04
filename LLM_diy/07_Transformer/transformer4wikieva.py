import numpy as np
import torch
import torch.nn as nn

from Transformer_Model import device

d_k = 64
d_v = 64

# 缩放点积注意力类
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V, attn_mask):
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-1, -2))/np.sqrt(d_k)

        # 使用注意力掩码，将attn_mask中值为1的位置权重替换为极小值
        # 自注：对attn_mask中为True的位置，对应的scores中的元素会被设置为-1e9
        scores.masked_fill_(attn_mask, -1e9)

        # 使用softmax函数对注意力分数进行归一化
        weights = nn.Softmax(dim=-1)(scores)

        # 计算上下文向量（也就是注意力的输出），是上下文信息的紧凑表示
        context = torch.matmul(weights, V)

        return context, weights


# 多头自注意力
d_embedding = 512
n_headers = 8
batch_size = 3

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(d_embedding, d_k * n_headers)
        self.W_K = nn.Linear(d_embedding, d_k * n_headers)
        self.W_V = nn.Linear(d_embedding, d_v * n_headers)
        self.linear = nn.Linear(n_headers * d_v, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)

    def forward(self, Q, K, V, attn_mask):
        # 保留残差连接
        residual, batch_size = Q, Q.size(0)

        # 将输入进行线性变换和重塑，以便后续处理
        # 自注：我这要重塑的话，每个seq中的每个词(token原来是512维度的现在我只拿到其中的一部分就是64维)
        q_s = self.W_Q(Q).view(batch_size, -1, n_headers, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_headers, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_headers, d_v).transpose(1, 2)

        # 将注意力掩码复制到多头
        # 自注：说白了，我现在的维度是，batch_size, n_headers, seq_len, d_k
        # 因为n_headers在前面，但seq_len在后面，所以我按照n_headers维度填充，知道不同seq_len的填充不同的内容
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_headers, 1, 1)

        # 使用缩放点积注意力计算上下文和注意力权重
        # 自注：此处的ScaledDotProductAttention是实例化出来的
        context, weight = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)

        # 通过调整维度将多个头的上下文向量连接在一起
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_headers * d_v)

        # 用一个线性层把连接后的多头自注意力结果转换为原始的嵌入维度
        # 自注：这么做的一个可能的原因是，我的context的维度是dim_v与dim_q不同，所以可能要转回去?
        # 还有个可能，线性变换后，张量的形状与Q相同，我就可以做下面的残差连接操作了
        output = self.linear(context)

        # 与输入(Q)进行残差连接，进行层归一化的操作
        # 在每个样本的特征维上进行归一化，也就是对每个token的embedding维度进行归一化
        output = self.layer_norm(output + residual)

        # 返回层归一化的输出和注意力权重
        return output, weight

# 定义逐位置前馈网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_ff=2048):
        super().__init__()
        # 定义一维卷积层1，用于将输入映射到更高维度
        self.conv1 = nn.Conv1d(in_channels=d_embedding, out_channels=d_ff, kernel_size=1)
        # 定义一维卷积层2，用于将输入映射回原始维度
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_embedding, kernel_size=1)
        # 定义层归一化
        self.layer_norm = nn.LayerNorm(d_embedding)

    def forward(self, inputs):
        # 保留残差连接
        residual = inputs

        # 在卷积层1后使用Relu函数
        # 自注：这边的转置后inputs 的维度 [batch_size, embedding_dim, len_q]
        # 网络沿着len_q的维度对每个token(d_embedding)做卷积(这边就是全连接了)
        output = nn.ReLU()(self.conv1(inputs.transpose(1,2)))

        # 维度 [batch_size, d_ff, len_q]
        # 使用卷积层2进行降维,再将维度
        output = self.conv2(output).transpose(1,2)
        # 维度 [batch_size, len_q, embedding_dim]

        # 与输入进行残差连接，并进行层归一化
        output = self.layer_norm(output + residual)

        # 返回加入残差连接后的层归一化的结果
        return output

# 自注：目的是为了在不同位置和维度之间产生独特的角度值，以便在生成位置嵌入向量时
# 捕获不同序列中不同位置的信息
# 生成正弦位置编码表的函数，用于在Transformer中引入位置信息
def get_sin_enc_table(n_position, embedding_dim):
    # n_position：输入序列的最大长度，范围[0, max_seq_len -1]
    # 根据位置和维度信息，初始化正弦位置编码表
    sinusoid_table = np.zeros((n_position, embedding_dim))
    # 遍历所有位置和维度，计算角度值
    for pos_i in range(n_position):
        for hid_j in range(embedding_dim):
            angle = pos_i / np.power(10000, 2*(hid_j // 2)/embedding_dim)
            sinusoid_table[pos_i, hid_j] = angle

    # 计算正弦和余弦值
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 偶数维度
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 奇数维度

    return torch.FloatTensor(sinusoid_table) # 返回正弦位置编码表

# 定义填充注意力掩码函数
def get_attn_pad_mask(seq_q, seq_k):

    # seq_q 维度 [batch_size, len_q]
    # seq_k 维度 [batch_size, len_k]

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # 生成布尔类型的张量
    # 自注: pad索引为0的地方填充
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # <PAD>token的编码值为0
    # pad_attn_mask 维度 [batch_size, 1, len_k]

    # 变形为与注意力分数相同形状的张量
    # 自注：expand 扩展操作仅复制已有的数据，不会引入新的信息
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
    # pad_attn_mask 维度 [batch_size, len_q, len_k]

    return pad_attn_mask # 返回填充位置的注意力掩码

# 生成后续注意力掩码的函数，用于在多头注意力计算中忽略未来信息
def get_attn_subsequent_mask(seq):
    # ------------------维度信息------------------
    # seq 的维度: [batch_size, seq_len(Q), seq_len(K)] ，形状与多头自注意力中的注意力权重矩阵相匹配
    # -------------------------------------------

    # 获取输入序列的形状
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]

    # ------------------维度信息------------------
    # attn_shape 是一个一维张量 [batch_size, seq_len(Q), seq_len(K)]
    # -------------------------------------------

    # 使用numpy创建一个上三角矩阵 (triu = triangle upper)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)

    # ------------------维度信息------------------
    # subsequent_mask 的维度: [batch_size, seq_len(Q), seq_len(K)]
    # -------------------------------------------

    # 将numpy数组转换为PyTorch张量，并将数据类型设置为byte(布尔值)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()

    # ------------------维度信息------------------
    # 返回的subsequent_mask 的维度： [batch_size, seq_len(Q), seq_len(K)]
    # -------------------------------------------

    return subsequent_mask # 返回后续位置的注意力掩码

# 定义解码器层类
class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention() # 多头自注意力
        self.feed_forward = PoswiseFeedForwardNet() # 逐位置前馈网络层
        self.norm1 = nn.LayerNorm(d_embedding) # 第一个层归一化
        self.norm2 = nn.LayerNorm(d_embedding) # 第二个层归一化

    def forward(self, dec_inputs, attn_mask=None):
        # 使用多头自注意力处理输入
        attn_output, _ = self.self_attn(dec_inputs, dec_inputs, dec_inputs, attn_mask)
        # 将注意力输出与输入相加并进行第一个层归一化
        norm1_outputs = self.norm1(dec_inputs + attn_output)
        # 将归一后的输出 输入逐位置前馈神经网络
        ff_outputs = self.feed_forward(norm1_outputs)
        # 将前馈神经网络输出与第一次归一化后的输出相加并进行第二个层归一化
        dec_outputs = self.norm2(norm1_outputs + ff_outputs)
        return dec_outputs # 返回解码器输出

# 定义解码器类
n_layers = 6 # 设置Decoder的层数
class Decoder(nn.Module): #
    def __init__(self, vocab_size, max_seq_len):
        super().__init__()
        # 词嵌入层(参数为词典维度)
        self.src_emb = nn.Embedding(vocab_size, d_embedding)
        # 位置编码层(参数为序列长度)
        self.pos_emb = nn.Embedding(max_seq_len, d_embedding)
        # 初始化N个解码器层
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs):
        # 创建位置信息 ?
        positions = torch.arange(len(dec_inputs), device=dec_inputs.device).unsqueeze(-1)
        # 将词嵌入与位置编码相加
        inputs_embedding = self.src_emb(dec_inputs) + self.pos_emb(positions)
        # 生成自注意力掩码 ?
        attn_mask = get_attn_subsequent_mask(inputs_embedding).to(device).bool()
        # 初始化话解码器输入，这是第一个解码器层的输入
        dec_outputs = inputs_embedding
        for layer in self.layers:
            # 将输入数据传递给解码器层，并返回解码器层的输出，作为下一层的输入
            dec_outputs = layer(dec_outputs, attn_mask)
        return dec_outputs # 返回解码器输出

# 定义GPT模型
class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len):
        super().__init__()
        self.decoder = Decoder(vocab_size, max_seq_len) # 解码器，用于学习文本生成能力
        self.projection = nn.Linear(d_embedding, vocab_size) # 全连接层，输出预测结果

    def forward(self, dec_inputs):
        dec_outputs = self.decoder(dec_inputs) # 将输入数据传递给解码器
        logits = self.projection(dec_outputs) # 传递给全连接层以生成预测
        return logits # 返回预测结果


# from torchtext.datasets import WikiText2 # 导入WikiText2
from tensorflow.python.data.experimental.ops.distribute import batch_sizes_for_worker
from torchtext.data.utils import get_tokenizer # 导入Tokenizer 分词工具
from torchtext.vocab import build_vocab_from_iterator # 导入Vocabulary 工具
from torch.utils.data import DataLoader, Dataset # 导入Pytorch 的 DataLoader 和 Dataset

tokenizer = get_tokenizer('basic_english') # 定义数据预处理所需的Tokenizer
# train_iter = WikiText2(split='train') # 加载WikiText2 数据集的训练部分

from Utilities import read_data
from CorpusLoader import WikiCorpus
# corpus = WikiCorpus(read_data('../../01_Data/wikitext-103/wiki.train.txt'))
train_iter = read_data('../../01_Data/wikitext-103/wiki.train.txt')
# vocab_size = len(corpus.vocab)

valid_iter = read_data('../../01_Data/wikitext-103/wiki.valid.txt')

# 定义一个生成器函数，用于将数据集中的文本转换为tokens
def yield_token(data_iter):
    for item in data_iter:
        yield tokenizer(item)

# 创建词汇表，包括特殊 tokens: '<pad>' '<sos>' '<eos>'
vocab = build_vocab_from_iterator(yield_token(train_iter),
                                  specials=['<pad>', '<sos>', '<eos>'])
vocab.set_default_index(vocab['<pad>'])

# 打印词汇表信息
print('词汇表大小: ', len(vocab))
print('词汇表示例(word to index): ', {word: vocab[word] for word in ['<pad>', '<sos>', '<eos>', 'the', 'apple']})

import torch
# 实现WikiDataset类
from torch.utils.data import Dataset # 导入 Dataset 类
max_seq_len = 256 # 设置序列的最大长度

# 定义一个处理WikiText2数据集的自定义数据类型
class WikiDataset(Dataset):
    def __init__(self, data_iter, vocab, max_len=max_seq_len):
        self.data = []
        for sentence in data_iter: # 遍历数据集，将文本转换为tokens
            # 对每个句子进行Tokenization, 截取长度为max_len-2, 为<sos>和<eos>留出空间
            tokens = tokenizer(sentence)[:max_len-2]
            tokens = [vocab['<sos>']] + vocab(tokens) + [vocab['eos']] # 添加<sos>和<eos>
            self.data.append(tokens) # 将处理好的tokens添加到数据集中

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): # 定义数据集的索引方法（即抽取数据条目）
        source = self.data[idx][:-1] # 获取当前数据，并将<eos>移除，作为源(source)数据
        target = self.data[idx][1:] # 获取当前数据，并将<sos>移除，作为目标(target)数据
        return torch.tensor(source), torch.tensor(target) # 转换为tensor并返回

# 注意可创建新的conda环境 python3.8 torch==2.2.2 torchaudio==2.2.2 torchdata==0.7.1 torchtext==0.17.2 torchvision==0.17.2
train_dataset = WikiDataset(train_iter, vocab) # 创建训练数据集

valid_dataset = WikiDataset(valid_iter, vocab) # 创建验证数据集

print(f'Dataset 数据条目数:{len(train_dataset)}')
sample_source, sample_target = train_dataset[100]
print(f'输入序列张量样例: {sample_source}')
print(f'目标序列样例文本: {sample_target}')
decoded_source = ' '.join(vocab.lookup_tokens(sample_source.tolist()))
decoded_target = ' '.join(vocab.lookup_tokens(sample_target.tolist()))
print(f'输入序列样例文本: {decoded_source}')
print(f'目标序列样例文本: {decoded_target}')

# 第3步 构建DataLoader类
# 定义pad_sequence函数，用于将一批序列补齐到相同长度
def pad_sequence(sequences, padding_value=0, length=None):
    # 计算最大序列长度，如果length参数未提供，则使用输入序列中的最大长度
    max_length = max(len(seq) for seq in sequences) if length is None else length
    # 创建一个具有适当形状的全零张量，用于存储补齐后的序列
    result = torch.full((len(sequences), max_length), padding_value, dtype=torch.long)
    # 遍历序列，将每个序列的内容复制到张量result中
    for i, seq in enumerate(sequences):
        end = len(seq)
        result[i, :end] = seq[:end]
    return result

# 定义collate_fn函数,用于将一个批次的数据整理成适当的形状
def collate_fn(batch):
    # 从批次中分离源序列和目标序列
    sources, targets = zip(*batch)
    # 计算批次中的最大序列长度
    max_length = max(max(len(s) for s in sources), max(len(t) for t in targets))
    # 使用pad_sequence 函数补齐源序列和目标序列
    sources = pad_sequence(sources, padding_value=vocab['<pad>'], length=max_length)
    targets = pad_sequence(targets, padding_value=vocab['<pad>'], length=max_length)
    # 返回补齐后的源序列和目标序列
    return sources, targets

batch_size= 6

# 创建一个训练数据加载器，使用自定义的collate_fn函数
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, collate_fn=collate_fn)

# 创建一个验证数据加载器，使用自定义的collate_fn函数
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)


# 使用DataLoader提供的数据进行训练
import torch.nn as nn
import torch.optim as optim # 导入优化器
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 设置设备
model = GPT(len(vocab), max_seq_len).to(device) # 创建GPT模型实例
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=0.0001) # 优化器啊
epochs = 2 # 训练轮次

import os
min_valid_loss = float("inf") # 初始化最低验证损失为无穷大
save_path = "best_model.pth" # 设置模型保存路径


for epoch in range(epochs):
    epoch_loss = 0
    for batch_idx, (source, targe) in enumerate(train_dataloader): # 用dataloader 加载数据
        inputs, targets = source.to(device), targe.to(device)
        optimizer.zero_grad() # 梯度清零
        outputs = model(inputs) # 获取模型输出
        loss = criterion(outputs.view(-1, len(vocab)), targets.view(-1)) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        epoch_loss += loss.item() # 积累每轮损失
        if (batch_idx + 1) % 1000 == 0: # 每 1000 个批次打印一次数据
            print(f'Batch {batch_idx + 1}/{len(train_dataloader)}, Loss: {loss.item()}')
    epoch_loss /= len(train_dataloader) # 每轮打印一次损失
    print(f'Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss}')

    # 评估模型
    model.eval() # 将模型设置为评估模式
    valid_loss = 0
    with torch.no_grad(): # 禁用梯度计算
        for source, target in valid_dataloader:
            inputs, targets = source.to(device), target.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, len(vocab)), targets.view(-1))
            valid_loss += loss.item()
        valid_loss /= len(valid_dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {valid_loss}')
        # 保存损失最小的模型
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            print(f'New best model saved at epoch {epoch+1} with Validation Loss: {valid_loss}')
        model.train() # 将模型设置为训练模式

# 使用集束搜索的函数
def generate_text_beam_search(model, input_str, max_len=50, beam_width=5):
    model.eval() # 将模型设置为评估模式，关闭dropout 和 batch normalizaion 等与训练相关的层
    # 将输入字符串中的每个token转换为其在词汇表中的索引
    input_tokens = [vocab[token] for token in input_str.split()]
    # 创建一个列表，用于存储候选序列
    candidates = [(input_tokens, 0.0)]
    with torch.no_grad(): # 禁用梯度计算，以节省内存并加速测试过程
        for _ in range(max_len): # 生成最多max_len个token
            new_candidates = []
            for candidate, candidate_score in candidates:
                inputs = torch.LongTensor(candidate).unsqueeze(0).to(device)
                outputs = model(inputs) # 输出 logits 形状为[1, len(output_tokens), vocab_size]
                logits = outputs[:, -1, :] # 只关心最后一个时间步(即最新生成的token)的logits
                # 找到具有最高分数的前beam_width 个 token
                scores, next_tokens = torch.topk(logits, beam_width, dim=-1)
                final_results = [] # 初始化输出序列
                for score, next_token in zip(scores.squeeze(), next_tokens.squeeze()):
                    new_candidate = candidate + [next_token.item()]
                    new_score = candidate_score - score.item() # 使用负数，因为我们需要降序排序
                    if next_token.item() == vocab['<eos>']:
                        # 如果生成的token是EOS(结束符)，将其添加到最终结果中
                        final_results.append((new_candidate, new_score))
                    else:
                        # 将新生成的候选序列添加到新候选列表中
                        new_candidates.append((new_candidate, new_score))
                # 从新候选列表中选择得分最高的beam_width个序列
                candidates = sorted(new_candidates, key=lambda x: x[1][:beam_width])
            # 选择得分最高的候选序列
            best_candidate, _ = sorted(candidates, key=lambda x: x[1])[0]
            # 将输出的token转换回文本字符串
            output_str = ' '.join([vocab.get_itos()[token] for token in best_candidate if vocab.get_itos()[token] != '<pad>'])
            return output_str

model.load_state_dict(torch.load('best_model.pth')) # 加载模型
input_str = "my name" # 输入几个词
generated_text = generate_text_beam_search(model, input_str) # 模型根据这些词生成后续文本
print('生成的文本: ', generated_text) # 打印生成的文本
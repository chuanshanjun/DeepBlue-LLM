import numpy as np
import torch
import torch.nn as nn
from tensorflow import resource

from Transformer_Model import n_layers

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
        context.trnaspose(1, 2).contiguous().view(batch_size, -1, n_headers * d_v)

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
        self.layer_norm(output, residual)

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

# 定义编码器层类
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention() # 多头自注意力层
        self.pos_ffn = PoswiseFeedForwardNet() # 逐位置前馈网络层

    def forward(self, enc_inputs, enc_self_attn_mask):
        # 将相同的Q, K, V 输入多头自注意力层，返回attn_weights 增加了头数
        enc_outpus, attn_weight = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)

        # 将多头自注意力outputs输入逐位置前馈网络层
        enc_outpus = self.pos_ffn(enc_outpus)

        return enc_outpus, attn_weight # 返回编码器输出和每层编码器的注意力权重

n_layers = 6 # 设置Encoder的层数
# 定义编码器类
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_emb = nn.Embedding(len(corpus.src_vocab), d_embedding) # 词嵌入层
        self.pop_emb = nn.Embedding.from_pretrained(get_sin_enc_table(corpus.src_len+1, d_embedding), freeze=True) # 位置嵌入层
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]) # 编码器层数

    def forward(self, enc_inputs):
        # enc_inputs的维度: [natch_size, source_len]
        # 创建一个从1到source_len的位置索引序列
        # 自注：这里+1是因为在位置编码中，0位置被保留为填充位置，所以从1开始 - 这段解释是模型生成的
        # 自注：又arange(1,6)方法生成的张量不包含6，所以这边还要再加1位
        pos_indices = torch.arange(1, enc_inputs.size(1) + 1).unsqueeze(0).to(enc_inputs)
        # pos_indices 维度 [1, source_len]

        # 对输入进行词嵌入和位置嵌入相加 [batch_size, source_len, embedding_dim]
        enc_outputs = self.src_emb(enc_inputs) + self.pop_emb(pos_indices)

        # 生成自注意力掩码
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

        enc_self_attn_weigths = []

        # 通过编码器层[batch_size, seq_len, embedding_dim]
        for layer in self.layers:
            enc_outputs, enc_self_attn_weigth = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attn_weigths.append(enc_self_attn_weigth)

        return enc_outputs, enc_self_attn_weigths # 返回编码器输出和注意力权重


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

# 定义解码器层
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention() # 多头自注意力层
        self.dec_enc_attn = MultiHeadAttention() # 多头自注意力层，连接编码器和解码器
        self.pos_ffn = PoswiseFeedForwardNet() # 逐位置前馈网络层

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # ------------------维度信息------------------
        # dec_inputs [batch_size, target_len, embedding_dim]
        # enc_outputs [batch_size, source_len, embedding_dim]
        # dec_self_attn_mask [batch_size, target_len, target_len]
        # dec_enc_attn_mask [batch_size, target_len, source_len]
        # -------------------------------------------

        # 将相同的 Q K V 输入多头自注意力层
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        # ------------------维度信息------------------
        # dec_outputs [batch_size, target_len, embedding_dim]
        # dec_self_attn_mask [batch_size, n_headers, target_len, target_len]
        # -------------------------------------------

        # 将解码器输出和编码器输出输入多头自注意力层
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)

        # ------------------维度信息------------------
        # dec_outputs [batch_size, target_len, embedding_dim]
        # dec_enc_attn_mask [batch_size, n_headers, target_len, source_len]
        # -------------------------------------------

        # 输入逐位置前馈网络层
        dec_outputs = self.pos_ffn(dec_outputs)

        # ------------------维度信息------------------
        # dec_outputs [batch_size, target_len, embedding_dim]
        # dec_self_attn [batch_size, n_headers, target_len, target_len]
        # dec_enc_attn [batch_size, n_headers, target_len, source_len]
        # -------------------------------------------

        # 返回解码器层输出，每层的自注意力和解码器-编码器注意力权重

        return dec_outputs, dec_self_attn, dec_enc_attn

# 定义解码器类
n_layers = 6 # 设置Decoder的层数
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tgt_emb = nn.Embedding(len(corpus.tgt_vocab), d_embedding) # 词嵌入层
        self.pos_emb = nn.Embedding.from_pretrained(get_sin_enc_table(corpus.tgt_len+1, d_embedding), freeze=True) # 位置嵌入层
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        # 创建一个从1到source_len的位置索引序列
        pos_indices = torch.arange(1, dec_inputs.size(1) + 1).unsqueeze(0).to(dec_inputs)

        # pos_indices 维度 [1, target_len] , arange中的因为后面是开区间，所以 dec_inputs.size(1) 需要加1

        # 对输入进行词嵌入和位置嵌入
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos_indices)

        # 生成解码器自注意力掩码和解码器-编码器注意力掩码
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) # 填充掩码
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs) # 后续掩码
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # 解码器-编码器掩码

        dec_self_attns, dec_enc_attns = [], [] # 初始化
        # 通过解码器层 [batch_size, seq_len, embedding_dim]
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns
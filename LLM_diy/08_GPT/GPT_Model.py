import numpy as np
import torch
import torch.nn as nn
from fsspec.core import OpenFile
from keras.backend import dtype
from torch.fx.experimental.unification.multipledispatch.dispatcher import source

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
# n_layers = 6 # 设置Decoder的层数
class Decoder(nn.Module): #
    def __init__(self, vocab_size, max_seq_len, n_layers):
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
    def __init__(self, vocab_size, max_seq_len, n_layers=6):
        super().__init__()
        self.decoder = Decoder(vocab_size, max_seq_len, n_layers) # 解码器，用于学习文本生成能力
        self.projection = nn.Linear(d_embedding, vocab_size) # 全连接层，输出预测结果

    def forward(self, dec_inputs):
        dec_outputs = self.decoder(dec_inputs) # 将输入数据传递给解码器
        logits = self.projection(dec_outputs) # 传递给全连接层以生成预测
        return logits # 返回预测结果

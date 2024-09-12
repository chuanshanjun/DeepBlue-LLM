import numpy as np
import torch
import torch.nn as nn

from GPT_Model_with_Decode import d_embedding

d_k = 64 # K(=Q)维度
d_v = 64 # V 维度

# 定义缩放点积注意力类
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # ------------------维度信息------------------
        # Q K V [batch_size, n_headers, len_q/k/v, dim_q=k/v] (dim_q=dim_k)
        # attn_mask [batch_size, n_headers, len_q, lenk]
        # -------------------------------------------

        # 计算注意力分数项（原始权重） [batch_size, n_headers, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2))/np.sqrt(d_k)
        # ------------------维度信息------------------
        # scores = [batch_size, n_headers, len_q, len_k]
        # -------------------------------------------

        # 自注：使用缩放因子后，使得分数高的不会太高，低的不会太低，
        # 这样计算SoftMax的时候输出更“不尖锐-平滑”，避免在梯度更新的时候出现梯度消失或梯度爆炸的问题

        # 使用注意力掩码，将attn_mask中值为1的位置的权重替换为极小值
        # ------------------维度信息------------------
        # attn_mask [batch_size, n_headers, len_q, len_k] 形状和scores相同
        # -------------------------------------------

        scores.masked_fill(attn_mask, -1e9)

        # 用softmax函数对注意力分数进行归一化
        weights = nn.Softmax(dim=-1)(scores)

        # ------------------维度信息------------------
        # weights [batch_size, n_headers, len_q, len_k]
        # -------------------------------------------

        # 计算上下文向量(也就是注意力输出)，是上下文信息的紧凑表示
        context = torch.matmul(weights, V)

        # ------------------维度信息------------------
        # context [batch_size, n_headers, len_q, dim_v]
        # -------------------------------------------

        return context, weights # 返回上下文向量和注意力分数

        # ------------------维度信息------------------
        # -------------------------------------------

# 定义多头自注意类
d_embedding = 512 # Embedding 维度
n_headers = 8 # Multi-Head Attention 中头的个数
batch_size = 3 # 每一批的数据大小

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_embedding, d_k * n_headers) # Q 的线性变换层
        self.W_K = nn.Linear(d_embedding, d_k * n_headers) # K 的线性变换层
        self.W_V = nn.Linear(d_embedding, d_v * n_headers) # V 的线性变换层
        self.linear = nn.Linear(d_v * n_headers, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)

    def forward(self, Q, K, V, attn_mask):
        # ------------------维度信息------------------
        # Q, K, V [batch_size, len_q/k/v, embedding_dim]
        # -------------------------------------------
        residual, batch_size = Q, Q.size(0) # 保留残差连接

        # 将输入进行线性变换和重塑，以便后续处理
        q_s = self.W_Q(Q).view(batch_size, -1, n_headers, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_headers, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_headers, d_v).transpose(1,2)

        # ------------------维度信息------------------
        # q_s k_s v_s: [batch_size, n_headers, len_q/k/v, d_q=k/v]
        # -------------------------------------------

        # 将注意力掩码复制到多头 attn_mask:[batch_size, n_headers, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_headers, 1, 1)

        # ------------------维度信息------------------
        # attn_mask [batch_size, n_headers, len_q, len_k]
        # -------------------------------------------

        # 使用缩放点积注意力计算上下文和注意力权重
        context, weights = ScaledDotProductAttention(q_s, k_s, v_s, attn_mask)

        # ------------------维度信息------------------
        # context [batch_size, n_headers, len_q, dim_v]
        # context [batch_size, n_headers, len_q, len_k]
        # -------------------------------------------

        # 通过调整维度将多个头的上下文向量连接在一起
        context = context.transpose(1,2).contiguous().view(batch_size, -1, n_headers * d_v)
        # ------------------维度信息------------------
        # context [batch_size, len_q, n_headers * dim_v]
        # -------------------------------------------

        # 用一个线性层把连接后的多头自注意力结果转换，原始地嵌入维度
        output = self.linear(context)

        # ------------------维度信息------------------
        # output [batch_size, len_q, embedding_dim]
        # -------------------------------------------

        # 与输入(Q)进行残差连接，并进行层归一化后的输出
        output = self.layer_norm(output + residual)

        # ------------------维度信息------------------
        # output [batch_size, len_q, embedding_dim]
        # -------------------------------------------

        return output, weights # 返回层归一化的输出和注意力权重

# 定义逐位置前馈网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_ff=2048):
        super(PoswiseFeedForwardNet, self).__init__()
        # 定义一维卷积层1，用于将输入映射到更高维度
        self.conv1 = nn.Conv1d(in_channels=d_embedding, out_channels=d_ff, kernel_size=1)
        # 定义一维卷积层2，用于将输入映射回原始维度
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_embedding, kernel_size=1)
        # 定义层归一化
        self.layer_norm = nn.LayerNorm(d_embedding)

    def forward(self, inputs):
        # ------------------维度信息------------------
        # inputs [batch_size, len_q, embedding_dim]
        # -------------------------------------------

        residual = inputs # 保留残差连接

        # 在卷积层1后使用ReLU函数
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))

        # ------------------维度信息------------------
        # output [batch_size, d_ff, len_q]
        # -------------------------------------------

        # 使用卷积层2进行降维
        output = self.conv2(output).transpose(1, 2)
        # ------------------维度信息------------------
        # output [batch_size, len_q, embedding_dim]
        # -------------------------------------------

        # 与输入进行残差连接，并进行层归一化
        output = self.layer_norm(output + residual)

        # ------------------维度信息------------------
        # output [batch_size, len_q, embedding_dim]
        # -------------------------------------------

        return output # 返回加入残差连接后层归一化的结果

# 生成正弦位置编码表的函数，用于Transformer中引入位置信息
def get_sin_enc_table(n_position, embedding_dim):
    # ------------------维度信息------------------
    # n_position: 输入序列的最大长度
    # embedding_dim: 词嵌入向量的维度
    # -------------------------------------------

    # 根据位置和维度信息，初始化正弦位置编码表
    sinusoid_table = np.zeros((n_position, embedding_dim))

    # 遍历所有位置和维度，计算角度值
    for pos_i in range(n_position):
        for hid_j in range(embedding_dim):
            angle = pos_i/np.power(10000, 2*(hid_j//2)/embedding_dim)
            sinusoid_table[pos_i, hid_j] = angle

    # 计算正弦和余弦值
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 偶数维
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 0::2]) # dim 2i+1 奇数维

    # ------------------维度信息------------------
    # signusoid_table 维度：[n_position, embedding_dim]
    # -------------------------------------------

    return torch.FloatTensor(sinusoid_table) # 返回正弦位置编码表

    # ------------------维度信息------------------
    # -------------------------------------------

# 定义填充注意力掩码函数
def get_attn_pad_mask(seq_q, seq_k):
    # ------------------维度信息------------------
    # seq_q 的维度： [batch_size, len_q]
    # seq_k 的维度： [batch_size, len_k]
    # -------------------------------------------

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # 生成布尔类型张量
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # <PAD> token 的编码值为0

    # ------------------维度信息------------------
    # pad_attn_mask 的维度: [batch_size, 1, len_k]
    # -------------------------------------------

    # 变形为与注意力分数相同形状的张量
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)

    # ------------------维度信息------------------
    # pad_attn_mask 的维度: [batch_size, len_q, len_k]
    # -------------------------------------------

    # ------------------维度信息------------------
    # -------------------------------------------

    return pad_attn_mask # 返回填充位置的注意力掩码

# 定义编码器层类
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention() # 多头自注意力层
        self.pos_ffn = PoswiseFeedForwardNet() # 逐位置前馈网络层

    def forward(self, enc_inputs, enc_self_attn_mask):
        # ------------------维度信息------------------
        # enc_inputs 的维度：[batch_size, seq_len, embedding_dim]
        # enc_self_attn_mask 的维度: [batch_size, seq_len, seq_len]
        # -------------------------------------------

        # 将相同的Q K V 输入多头自注意力层，返回的attn_weights增加了头数
        enc_outputs, attn_weights = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)

        # ------------------维度信息------------------
        # enc_outputs 的维度: [batch_size, seq_len, embedding_dim]
        # attn_weights 的维度：[batch_size, n_headers, seq_len, seq_len]
        # -------------------------------------------

        # 将多头自注意力outputs输入逐位置前馈网络层
        enc_outputs = self.pos_ffn(enc_outputs) # 维度与 enc_inputs 相同

        # ------------------维度信息------------------
        # enc_outputs 的维度 [batch_size, seq_len, embedding_dim]
        # -------------------------------------------

        return enc_outputs, attn_weights # 返回编码器输出和每层编码器的注意力权重

# 定义编码器类
n_layers = 6 # 设置 Encoder 的层数
class Encoder(nn.Module):
    def __init__(self, corpus):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(len(corpus.src_vocab), d_embedding) # 词嵌入层
        self.pos_emb = nn.Embedding.from_pretrained(get_sin_enc_table(corpus.src_len+1, d_embedding), freeze=True) # 位置嵌入层
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(n_layers)) # 编码器层数

    def forward(self, enc_inputs):
        # ------------------维度信息------------------
        # enc_input [batch_size, source_len]
        # -------------------------------------------

        # 创建一个1到source_len的位置索引序列
        pos_indices = torch.arange(1, enc_inputs.size(1) +1).unsqueeze(0).to(enc_inputs)

        # ------------------维度信息------------------
        # pos_indices [1, source_len]
        # -------------------------------------------

        # 对输入进行词嵌入和位置嵌入相加[batch_size, source_len, embedding_dim]
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(pos_indices)

        # ------------------维度信息------------------
        # enc_outputs [batch_size, seq_len, embedding_dim]
        # -------------------------------------------

        # 生成自注意力掩码
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

        # ------------------维度信息------------------
        # enc_self_attn_mask [batch_size, len_q, len_k]
        # -------------------------------------------

        enc_self_attn_weights = [] # 初始化 enc_self_attn_weights

        # 通过编码器层[batch_size, seq_len, embedding_dim]
        for layer in self.layers:
            # 自注：输出的enc_outpus 作为下一层的输入
            enc_outputs, enc_self_attn_weight = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attn_weights.append(enc_self_attn_weight)

        # ------------------维度信息------------------
        # -------------------------------------------

        return enc_outputs, enc_self_attn_weights # 返回编码器输出和编码器注意力权重
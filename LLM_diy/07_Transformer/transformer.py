import numpy as np
import torch
import torch.nn as nn

d_k = 64
d_v = 64

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # 维度信息
        # Q K V [batch_size, n_headers, len_q/k/v, dim_q=k/v](dim_q = dim_k)
        # attn_mask[batch_size, n_headers, len_q, len_k]

        # 计算注意力分数(原始权重)[batch_size, n_headers, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2))/np.sqrt(d_k)

        # 使用注意力掩码，将atten_mask中值为1的位置的权重替换为极小值
        scores.masked_fill_(attn_mask, -1e9)

        # 用softmax函数对注意力分数进行归一化
        weights = nn.Softmax(dim=-1)(scores)

        # 计算上下文向量(也就是注意力输出)，是上下文信息的紧凑表示
        context = torch.matmul(weights, V)

        # context[batch_size, n_headers, len_q, dim_v]

        return context, weights # 返回上下文向量和注意力分数

    d_embedding = 512
    n_headers = 8
    batch_size = 3
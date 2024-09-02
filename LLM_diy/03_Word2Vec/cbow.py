# 定义一个句子列表，后面会用这些句子来训练CBOW和Skip-Gram模型
sentences = ["Kage is Teacher", "Mazong is Boss", "Niuzong is Boss",
             "Xiaobing is Student", "Xiaoxue is Student",]
# 将所有句子连接在一起，然后用空格分隔成词汇
words = ' '.join(sentences).split()
# 构建词汇表，去除重复的词
word_list = list(set(words))
# 创建一个字典，将每个词汇映射到一个唯一的索引
word_to_idx = {word: idx for idx, word in enumerate(word_list)}
# 创建一个字典，将每个索引映射到对应的词汇
idx_to_word = {idx: word for idx, word in enumerate(word_list)}
voc_size = len(word_list) # 计算词汇表的大小
print("词汇表：", word_list) # 输出词汇表
print("词汇到索引的字典：", word_to_idx) # 输出词汇到索引的字典
print("索引到词汇的字典：", idx_to_word) # 输出索引到词汇的字典
print("词汇表大小：", voc_size) # 输出词汇表大小

# 生成CBOW训练数据
def create_cbow_dataset(sentences, window_size=2):
    data = []
    for sentence in sentences:
        sentence = sentence.split()  # 将句子分割成单词列表
        for idx, word in enumerate(sentence):  # 遍历单词及其索引
            # 获取上下文词汇，将当前单词前后各window_size个单词作为上下文词汇
            context_words = sentence[max(idx - window_size, 0):idx] \
                + sentence[idx + 1:min(idx + window_size + 1, len(sentence))]
            # 将当前单词与上下文词汇作为一组训练数据
            data.append((word, context_words))
    return data

# 使用函数创建CBOW训练数据
cbow_data = create_cbow_dataset(sentences)
# 打印未编码的CBOW数据样例（前三个）
print("CBOW数据样例（未编码）：", cbow_data[:3])

# 定义One-Hot编码函数
import torch # 导入torch库
def one_hot_encoding(word, word_to_idx):
    # 创建一个全为0的张量，长度与词汇表大小相同
    tensor = torch.zeros(len(word_to_idx))
    tensor[word_to_idx[word]] = 1  # 将对应词汇的索引位置置为1
    return tensor  # 返回生成的One-Hot向量

# 展示One-Hot编码前后的数据
word_example = "Teacher"
print("One-Hot编码前的单词：", word_example)
print("One-Hot编码后的向量：", one_hot_encoding(word_example, word_to_idx))

# 展示编码后的CBOW数据样例
# 展示编码后的CBOW数据样例
print("CBOW数据样例（已编码）：", [(one_hot_encoding(target, word_to_idx),
                          [one_hot_encoding(context, word_to_idx) for context in context_words])
                         for target, context_words in cbow_data[:3]])

# 定义CBOW模型
import torch.nn as nn
class CBOW(nn.Module):
    def __init__(self, voc_size, embedding_size):
        super(CBOW, self).__init__()
        self.input_to_hidden = nn.Linear(voc_size, embedding_size, bias=False)
        self.hidden_to_output = nn.Linear(embedding_size, voc_size, bias=False)

    def forward(self, X): # X:[num_context_words, vocsize]
        embeddings = self.input_to_hidden(X)
        hidden_layers = torch.mean(embeddings, dim=0)
        output_layer = self.hidden_to_output(hidden_layers.unsqueeze(0))
        return output_layer

embeddings_size = 2
cbow_model = CBOW(voc_size, embeddings_size)
print('CBOW模型: ', cbow_model)

# 训练Skip-Gram模型
learning_rate = 0.001 # 设置学习速率
epochs = 1000 # 设置训练轮次
criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
import torch.optim as optim # 导入随机梯度下降优化器
optimizer = optim.SGD(cbow_model.parameters(), lr=learning_rate)


loss_values = []  # 用于存储每轮的平均损失值
for epoch in range(epochs):
    loss_sum = 0 # 初始化损失值
    for target, context_words in cbow_data:
        X = torch.stack([one_hot_encoding(word, word_to_idx) for word in context_words]).float()
        y_true = torch.tensor([word_to_idx[target]], dtype=torch.long) # 将周围词转换为索引值
        y_pred = cbow_model(X)  # 计算预测值
        loss = criterion(y_pred, y_true)  # 计算损失
        loss_sum += loss.item() # 累积损失
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
    if (epoch+1) % 100 == 0: # 输出每100轮的损失，并记录损失
        print(f"Epoch: {epoch+1}, Loss: {loss_sum/len(cbow_data)}")
        loss_values.append(loss_sum / len(cbow_data))

import matplotlib.pyplot as plt # 导入matplotlib
plt.plot(range(1, epochs//100 + 1), loss_values) # 绘图
plt.title('Training Loss') # 图题
plt.xlabel('Epochs') # X轴Label
plt.ylabel('Loss') # Y轴Label
plt.show() # 显示图

# 输出Skip-Gram习得的词嵌入
print("Skip-Gram词嵌入:")
for word, idx in word_to_idx.items(): # 输出每个单词的嵌入向量
    print(f"{word}: \
    {cbow_model.input_to_hidden.weight[:, idx].detach().numpy()}")

plt.rcParams["font.family"]=['SimHei'] # 用来设定字体样式
plt.rcParams['font.sans-serif']=['SimHei'] # 用来设定无衬线字体样式
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
fig, ax = plt.subplots()
for word, idx in word_to_idx.items():
    vec = cbow_model.input_to_hidden.weight[:,idx].detach().numpy()
    ax.scatter(vec[0], vec[1])
    ax.annotate(word, (vec[0], vec[1]), fontsize=12)
plt.title('二维词嵌入')
plt.xlabel('向量维度1')
plt.ylabel('向量维度2')
plt.show()

# 第1步 聊天数据的构建

from torchtext.data.utils import get_tokenizer # 导入Tokenizer分词工具
from torchtext.vocab import build_vocab_from_iterator # 导入Vocabulary 工具
tokenizer = get_tokenizer('basic_english') # 定义数据预处理所需的tokenizer

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        # 读取文件内容
        content = file.read().split('\n')
    return content

train_iter = read_file('../../01_Data/wikitext-103/wiki.train02.txt')

# 定义一个生成器函数
def yield_tokens(data_iter):
    for item in data_iter:
        yield tokenizer(item)

# 创建词汇表，包括特殊token: '<pad>', '<sos>', '<eos>'
vocab = build_vocab_from_iterator(yield_tokens(train_iter),
                                  specials=['<pad>', '<sos>', '<eos>'])

vocab.set_default_index(vocab['<pad>'])

# 有了和基础模型一致的词汇表后，可以基于Tokenizer和词汇表创建聊天数据集
import torch
from torch.utils.data import Dataset # 导入Dataset

class ChatDataset(Dataset):
    def __init__(self, file_path, tokenizer, vocab):
        self.tokenizer = tokenizer # 分词器
        self.vocab = vocab # 词汇表
        self.input_data, self.target_data = self.load_and_process_data(file_path)

    def load_and_process_data(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines() # 打开文件，读取每一个行数据
        input_data, target_data = [], []
        for i, line in enumerate(lines):
            if line.startswith('User:'): # 移除'User' 前缀，构建输入序列
                tokens = self.tokenizer(line.strip()[6:])
                tokens = ['<sos>'] + tokens + ['<eos>']
                indices = [self.vocab[token] for token in tokens]
                input_data.append(torch.tensor(indices, dtype=torch.long))
            elif line.startswith('AI:'): # 移除‘AI：’ 前缀，构建目标序列
                tokens = self.tokenizer(line.strip()[4:])
                tokens = ['<sos>'] + tokens + ['<eos>']
                indices = [self.vocab[token] for token in tokens]
                target_data.append(torch.tensor(indices, dtype=torch.long))
        return input_data, target_data

    def __len__(self): # 数据集的长度
        return len(self.input_data)

    def __getitem__(self, idx): # 获取指定索引的数据
        return self.input_data[idx], self.target_data[idx]

file_path = 'chat.txt' # 加载chat.txt 语料库
chat_dataset = ChatDataset(file_path, tokenizer, vocab)

for i in range(3): # 打印几个样本数据
    input_sample, target_sample = chat_dataset[i]
    print(f'Sample{i+1}:')
    print('Input Data: ', input_sample)
    print('Target Data: ', target_sample)
    print('-'*50)

from torch.utils.data import DataLoader # 导入DataLoader
# 定义 pad_sequence 函数，用于将一批序列补齐到相同长度
def pad_sequence(sequences, padding_value=0, length=None):
    # 计算最大序列长度，如果Length参数未提供，则使用输入序列中的最大长度
    max_length = max(len(seq) for seq in sequences) if length is None else length
    # 创建一个具有适当形状的全零张量，用于存储补齐后的序列
    result = torch.full((len(sequences), max_length), padding_value, dtype=torch.long)
    # 遍历序列，将每个序列的内容复制到张量result中
    for i, seq in enumerate(sequences):
        end = len(seq)
        result[i, :end] = seq[:end]
    return result

# 定义collate_fn函数，用于将一个批次的数据整理成适当的形状
def collate_fn(batch):
    # 从批次中分离源序列和目标序列
    sources, targets = zip(*batch)
    # 计算批次中的最大序列长度
    max_length = max(max(len(s) for s in sources), max(len(t) for t in targets))
    # 使用pad_sequence函数补齐源序列和目标序列
    sources = pad_sequence(sources, padding_value=vocab['<pad>'], length=max_length)
    targets = pad_sequence(targets, padding_value=vocab['<pad>'], length=max_length)
    # 返回补齐后的源序列和目标序列
    return sources, targets

# 创建DataLoader
batch_size = 2
chat_dataloader = DataLoader(chat_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# 第2步 微调Wiki-GPT
from GPT_Model import GPT # 导入GPT模型的类(这是我们自己制作的)
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 确定设备(CPU 或 GPU)
model = GPT(28785, 256, n_layers=6) # 创建模型示例
model.load_state_dict(torch.load('best_model.pt')) # 加载模型

# 加载Wiki-GPT之后，就使用和训练模型相同的方法对模型进行微调
import torch.nn as nn # 导入 mm
import torch.optim as optim # 导入优化器
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>']) # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0001) # 优化器
for epoch in range(100): # 开始训练
    for batch_idx, (input_batch, target_batch) in enumerate(chat_dataloader):
        optimizer.zero_grad() # 梯度清零
        input_batch, target_batch = input_batch.to(device), target_batch.to(device) # 移动到设备
        outputs = model(input_batch) # 前向传播， 计算模型输出
        loss = criterion(outputs.view(-1, len(vocab)), target_batch.view(-1)) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
    if (epoch + 1) % 20 == 0: # 每20个epoch打印一次损失值
        print(f'Epoch: {epoch+1:04d}, Loss: {loss:6f}')

# 微调一个常见的做法是，可以仅微调顶层，即模型的头部层，以适应特定任务，即节省计算资源
# 又能保留预训练模型的底层已习得得通用特征
# 关键可以在构建优化器时，冻结部分层次，指定需要更新的参数
# 下面这段代码可选
def freeze_layers(model, n):
    params_to_update = [] # 获取模型的参数
    for name, param in model.named_parameters():
        if int(name.split('.')[1] >= n): # 冻结前n层
            params_to_update.append(param)
        return params_to_update
params_to_update = freeze_layers(GPT, n=2) # 冻结前2层(底层)参数
optimizer = optim.Adam(params_to_update, lr=0.0001) # 仅更新未冻结的参数

# 第3步 与简版ChatGPT对话
# 此处仍然使用集束算法来生成对话结果
def generate_text_beam_search(model, input_str, max_len=50, beam_width=5):
    model.eval() # 将模型设置为评估模式，关闭dropout和batch normalization 等与训练相关的层
    # 将输入字符串中的每个token转换为其在词汇表中的索引
    input_tokens = [vocab[token] for token in input_str.split()]
    # 创建一个列表，用于存储候选序列
    candidates = [(input_tokens, 0.0)]
    with torch.no_grad(): # 禁用梯度计算，以节省内存并加速测试过程
        for _ in range(max_len): # 生成最多个max_len个token
            new_candidates = []
            for candidate, candidate_score in candidates:
                inputs = torch.LongTensor(candidate).unsqueeze(0).to(device)
                outputs = model(inputs) # 输出logitis 形状为[1, len(output_tokens), vocab_size]
                logits = outputs[:, -1, :] # 只关心最后一个时间步（即最新生成的token） 的logits
                # 找到具有最高分数的前beam_width个token
                scores, next_tokens = torch.topk(logits, beam_width, dim=-1)
                final_results = [] # 初始化输出序列
                for score, next_token in zip(scores.squeeze(), next_tokens.squeeze()):
                    new_candidate = candidate+[next_token.item()]
                    new_score = candidate_score - score.item() # 使用负数，因为我们需要降序排列
                    if next_token.item() == vocab['<eos>']:
                        # 如果生成的token是EOS(结束符),将其添加到最终结果中
                        final_results.append((new_candidate, new_score))
                    else:
                        # 将新生成的候选序列添加到新候选列表中
                        new_candidates.append((new_candidate, new_score))
            # 从新候选列表中选择得分最高的beam_width个序列
            candidates = sorted(new_candidates, key=lambda x: x[1][:beam_width])
        # 选择得分最高的候选序列
        best_candidate, _  = sorted(candidates, key=lambda x:x[1][0])
        # 将输出的token转换回文本字符串
        output_str = ' '.join([vocab.get_itos()[token] for token in best_candidate if vocab.get_itos()[token]!='<pad>'])
        return output_str

# input_str = 'what is the weather like today ?'
input_str = 'hi, how are you ?'
generated_text = generate_text_beam_search(model, input_str) # 模型根据这些词生成后续文本
print('生成的文本: ', generated_text) # 打印生成的文本






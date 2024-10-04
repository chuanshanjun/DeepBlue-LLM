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

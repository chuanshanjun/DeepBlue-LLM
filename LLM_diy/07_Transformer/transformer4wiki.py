# from torchtext.datasets import WikiText2 # 导入WikiText2
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

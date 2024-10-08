from torchtext.datasets import WikiText2 # 导入 WikiText2
from torchtext.data.utils import get_tokenizer # 导入Tokenizer分词工具
from torchtext.vocab import build_vocab_from_iterator # 导入 Vocabulary 工具
from torch.utils.data import DataLoader, Dataset # 导入Pytorch的DataLoader 和 Dataset

tokenizer = get_tokenizer('basic_english') # 定义数据预处理所需的Tokenizer
train_iter = WikiText2(split='train') # 加载WikiText2数据集的训练部分
# 定义一个生成器函数，用于将数据集中的文本转换为tokens
def yield_tokens(data_iter):
    for item in data_iter:
        yield tokenizer(item)

# 创建词汇表，包括特殊tokens: '<pad>', '<sos>', '<eos>'
vocab = build_vocab_from_iterator(yield_tokens(train_iter),
                                  specials=['<pad>', '<sos>', '<eos>'])
vocab.set_default_index(vocab['<pad>'])
# 打印词汇表信息
print('词汇表大小: ', len(vocab))
print('词汇示例(word to index): ', {word: vocab[word] for word in ['<pad>', '<sos>', '<eos>', 'the', 'apple']})
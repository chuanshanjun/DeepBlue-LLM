from weakref import finalize

import torch # 导入 torch
from requests.packages import target
from transformers import GPT2Tokenizer # 导入GPT-2分词器
from transformers import GPT2LMHeadModel # 导入 GPT-2语言模型
model_name = 'gpt2' # 也可以选择其他模型，如'gpt2-medium' 'gpt2-large' 等
tokenizer = GPT2Tokenizer.from_pretrained(model_name) # 加载分词器
tokenizer.pad_token = '<pad>' # 为分词器添加 pad token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPT2LMHeadModel.from_pretrained(model_name).to(device) # 将模型加载到设备上(CPU或GPU)
vocab = tokenizer.get_vocab() # 获取词汇表
print('模型信息: ', model)
print('分词信息: ', tokenizer)
print('词汇表大小: ', len(vocab))
print('部分词汇示例: ', (list(vocab.keys())[8000:8005]))


# 准备微调数据集
from torch.utils.data import Dataset # 导入Pytorh 的 Dataset
# 自定义ChatData 类， 继承自PyTorch的Dataset类
class ChatDataset(Dataset):
    def __init__(self, file_path, tokenizer, vocab):
        self.tokenizer = tokenizer # 分词器
        self.vocab = vocab # 词汇表
        # 加载数据并处理，将处理后的输入数据和目标数据赋值给input_data 和 target_data
        self.input_data, self.target_data = self.load_and_process_data(file_path)
        # 定义加载和处理数据的方法
    def load_and_process_data(self, file_path):
        with open(file_path) as f: # 读取文件内容
            lines = f.readlines()
        input_data, target_data = [], []
        for i, line in enumerate(lines): # 遍历文件的每一行
            if line.startswith('User:'): # 如果'User'开头，移除'User'前缀，并将张量转换为列表
                tokens = self.tokenizer(line.strip()[6:], return_tensors='pt')['input_ids'].tolist()[0]
                tokens = tokens + [tokenizer.eos_token_id] # 添加结束符
                input_data.append(torch.tensor(tokens, dtype=torch.long)) # 添加 input_data
            elif line.startswith('AI:'): # AI 开头，移除'AI'前缀，并将张量转换为列表
                tokens = self.tokenizer(line.strip()[4:], return_tensors='pt')['input_ids'].tolist()[0]
                tokens = tokens + [tokenizer.eos_token_id] # 添加结束符
                target_data.append(torch.tensor(tokens, dtype=torch.long)) # 添加target_data
        return input_data, target_data

    # 定义数据集长度，即input_data的长度
    def __len__(self):
        return len(self.input_data)

    # 定义获取数据集中指定索引的数据的方法
    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]

file_path = 'chat.txt' # 记载chat.txt 数据集
chat_dataset = ChatDataset(file_path, tokenizer, vocab) # 创建ChatDataset 对象, 传入文件、分词器和词汇表
for i in range(2): # 打印数据集中前2个数据示例
    input_example, target_example = chat_dataset[i]
    print(f'示例[i+1]: ')
    print(f'输入: {tokenizer.decode(input_example)}')
    print(f'输出: {tokenizer.decode(target_example)}')

# 第4步 准备微调数据加载器
from torch.utils.data import DataLoader # 导入DataLoader
tokenizer.pad_token = '<pad>' # 为分词器添加 pad token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')
# 定义 pad_sequence 函数，用于将一批序列补齐到相同长度
def pad_sequence(sequences, padding_value=0, length=None):
    # 计算最大序列长度，如果length参数未提供，则使用输入序列中的最大长度
    max_length = max(len(seq) for seq in sequences) if torch is None else length
    # 创建一个具有适当形状的全零张量，用于存储补齐后的序列
    result = torch.full((len(sequences), max_length), padding_value, dtype=torch.long)
    # 遍历序列，将每个序列的内容复制到张量result中
    for i, seq in enumerate(sequences):
        end = len(seq)
        result[i, :end] = seq[:end]
    return result

# 定义collate_fn 函数，用于将一个批次的数据整理成适当的形状
def collate_fn(batch):
    # 从批粗中分离源序列和目标序列
    sources, targets = zip(*batch)
    # 计算批次中的最大序列长度
    max_length = max(max(len(s) for s in sources), max(len(t) for t in targets))
    # 使用pad_sequence 函数补齐源序列和目标序列
    sources = pad_sequence(sources, padding_value=tokenizer.pad_token_id, length=max_length)
    targets = pad_sequence(targets, padding_value=tokenizer.pad_token_id, length=max_length)
    # 返回补齐后的源序列和目标序列
    return sources, targets

# 创建DataLoader
chat_dataloader = DataLoader(chat_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
# 检查Dataloader 输出
for input_batch, target_batch in chat_dataloader:
    print('Input batch tensor size:', input_batch.size())
    print('Target batch tensor size:', target_batch.size())
    break
for input_batch, target_batch in chat_dataloader:
    print('Input batch tensor:')
    print(input_batch)
    print('Target batch tensor:')
    print(target_batch)
    break

# 对GPT-2进行微调
import torch.nn as nn
import torch.optim as optim
# 定义损失函数，忽略pad_token_id对应的损失值
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
# 定义优化器
optimizer = optim.Adam(model.parameters())
# 进行100个epoch的训练
for epoch in range(500):
    for batch_idx, (input_batch, target_batch) in enumerate(chat_dataloader): # 遍历数据加载器中的批次
        optimizer.zero_grad() # 梯度清零
        input_batch, target_batch = input_batch.to(device), target_batch.to(device) # 将输入和目标批次移至设备
        outputs = model(input_batch) # 前向传播
        logits = outputs.logits # 获取logits
        loss = criterion(logits.view(-1, logits.size(-1)), target_batch.view(-1)) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
    if (epoch + 1)%100 == 0: # 每100个epoch打印一次损失值
        print(f'Epoch: {epoch+1:04d}, cost={loss:6f}')

# 定义集束解码函数
def generate_text_beam_search(model, input_str, max_len=50, beam_width=5):
    model.eval() # 将模型设置为评估模式(不计算梯度)
    # 对输入字符串进行编码，并将其转换为张量，然后将其移动到相应的设备上
    input_tokens = tokenizer.encode(input_str, return_tensors='pt').to(device)
    # 初始化候选序列表，包含当前输入序列和其对数概率得分（我们从0开始）
    candidate = [(input_tokens, 0.0)]
    # 禁用梯度计算，以加速预测过程
    with torch.no_grad():
        # 迭代生成最大长度的序列
        for _ in range(max_len):
            new_candidates = []
            # 对于每个候选序列
            for candidate, candidate_score in candidate:
                # 使用模型进行预测
                outputs = model(candidate)
                # 获取输出 logits
                logits = outputs.logits[:, -1, :]
                # 获取对数概率得分top-k值(即beam_width)及其对应的token
                scores, next_token = torch.topk(logits, beam_width, dim=-1)
                final_results = []
                # 遍历 top-k token 及其对应的得分
                for score, next_token in zip(scores.squeeze(), next_token.squeeze()):
                    # 在当前候选序列中添加新的token
                    torch.cat((candidate, next_token.unsqueeze(0).unsqueeze(0)), dim=-1)
                    # 更新候选序列的得分
                    new_score = candidate_score - score.item()
                    # 如果新的token是结束符(eos_token), 则将该候选序列添加到最终结果中
                    if next_token.item() == tokenizer.eos_token_id:
                        final_results.append((new_candidates, new_score))
                    # 否则，将新的候选序列添加到新候选序列列表中
                    else:
                        new_candidates.append((new_candidates, new_score))
            # 从新候选序列列表中选择得分最高的top-k个序列
            best_candidate = sorted(new_candidates, key=lambda x: x[1])[0]
            # 将输出token转换回文本字符串
            output_str = tokenizer.decode(best_candidate[0])
            # 移除输入字符串并修复空格问题
            input

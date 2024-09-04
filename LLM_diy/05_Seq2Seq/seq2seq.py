# 第1步 构建实验语料库和词汇表
# 构建语料库，每行包含中文、英文（解码器输入）和翻译成英文后的目标输出3个句子
sentences = [
    ['咖哥 喜欢 小冰', '<sos> KaGe likes XiaoBing', 'KaGe likes XiaoBing <eos>'],
    ['我 爱 学习 人工智能', '<sos> I love studying AI', 'I love studying AI <eos>'],
    ['深度 学习 改变 世界', '<sos> Deep learning YYDS', 'Deep learning YYDS <eos>'],
    ['自然 语言 处理 很 强大', '<sos> NLP is so powerful', 'NLP is so powerful <eos>'],
    ['神经网络 非常 复杂', '<sos> Neural-Net are complex', 'Neural-Net are complex <eos>']]

word_list_cn, word_list_en = [], [] # 初始化中英文词汇表
# 遍历每一个句子并将单词添加到词汇表中
for s in sentences:
    word_list_cn.extend(s[0].split())
    word_list_en.extend(s[1].split())
    word_list_en.extend(s[2].split())

# 去重，得到没有重复单词的词汇表
word_list_cn = list(set(word_list_cn))
word_list_en = list(set(word_list_en))

# 构建单词到索引的映射
word2idx_cn = {w: i for i, w in enumerate(word_list_cn)}
word2idx_en = {w: i for i, w in enumerate(word_list_en)}

# 构建索引到单词的映射
idx2word_cn = {i: w for i, w in enumerate(word_list_cn)}
idx2word_en = {i: w for i, w in enumerate(word_list_en)}

# 计算词汇表的大小
voc_size_cn = len(word_list_cn)
voc_size_en = len(word_list_en)

print('句子数量: ', len(sentences))
print('中文词汇表大小: ', voc_size_cn)
print('英文词汇表大小：', voc_size_en)
print('中文词汇表到索引的字典：', word2idx_cn)
print('英文词汇表到索引的字典：', word2idx_cn)
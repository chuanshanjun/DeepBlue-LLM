第1课  
1.1 N-Gram模型  
语言模型：就是一个用来估计文本概率分布的数学模型，它可以帮助我们了解某个文本序列在
自然语言中出现的概率，因此也就能够根据给定的文本，预测下一个最可能出现的单词。

奇普夫定律:在任何给定的语料库中，一个词出现的频率与其排名成反比。

N-Gram:N代表什么？ 在英语中可以是“单词”、“字符”、“子词”；在中文中可以指词或者短语，也可以是字

1.3 创建一个Bigram模型

1.4 词袋模型  
词袋模型：它将文本中的词看作一个个独立的个体，忽略文本中词与词之间的顺序关系，只考虑词频。  
N-Gram模型考虑了词与词之前的顺序关系，而词袋模型则忽略了这个信息  
词袋在文本分析和情感分析等更加简单高效。

余弦相似度和向量距离(Vector Distance)都可以衡量两个向量之间的相似性。
余弦相似度关注向量之间的角度，而不是它们之间的距离，其取值范围在-1(完全相反)到1(完全相同)
之间。向量距离关注向量之间的实际距离。通常使用欧几里德距离来计算。两个向量越接近，它们
的距离越小。

如果要和量两个向量的相似性，而不关心它们的大小，那么余弦相似度会更合适。
因此，余弦相似度通常用于衡量文本、图像等高维数据的相似性，因为在这些场景下，关注向量的方向
关系通常比关注距离更有意义。而一些需计算实际距离的应用场景，如聚类分析、推荐系统等，
向量距离会更合适


第2课  
2.1 词向量 约等于 词嵌入
词嵌入(Wrod Embedding),是一种寻找词和词之间相似性的NLP技术，它把词汇各个维度上的
特征用数值向量进行表示，利用这些**维度上特征的相似程度**，就可以判断出哪些词和哪些词语义更接近。  

在实际应用中，词向量和词嵌入这两个重要的NLP术语通常可以互换使用。它们都表示将词汇表中的
单词映射到固定大小的连续向量空间中的过程。这些向量可以捕捉词汇的语义信息，例如：相似
语义的词在向量空间中余弦相似度高，距离也较近，而不同语义的词余弦相似度低，距离也较远。

两个术语可互换但在某些场合有细微差别。

* 词向量：用于描述具体的向量表示，即一个词对应的实际数值向量。例如，我们可以说'cat'
这个词的词向量是一个300维的向量
* 词嵌入：用于描述将词映射到向量空间的过程或表示方法。它通常包括训练算法和生成的词向量空间。
例如，我们可以说“我们使用Word2Vec算法来生成词嵌入”

Word2Vec: 在将词映射到向量空间时，会将这个词和它周围的一些词语一起学习，这就使得
具有相似语义的词在向量空间中靠得更近。这样，我们就可以通过向量之间的距离来度量词之间的
相似度了。

2.2 Word2Vec
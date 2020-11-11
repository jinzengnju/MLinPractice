# 基于预训练Bert的下游任务

## 1. 提取动态句向量

### 1.1 原理

​		提到提取句向量，传统的做法应该是利用word embedding技术例如word2vec，获取每个词对应的词向量，将每个词对应的词向量进行element-wise add，得到句子的向量。这种做法，将词向量看成是一个静态的向量，不同的句子中对于词A都是相同的embedding。实际上，在不同的句子场景中，相同的词也应该有不同的embedding表示，这就涉及到动态词向量。

​		最早的动态词向量起源于ELMO模型，对于Bert而言，我们也可以使用动态词向量。比如，已经预先训练好一个bert模型，将输入的句子先经过bert重新计算编码，得到每一个时间步的动态词向量。源码extract_features.py正好体现了这一点。

### 1.2 工程实现

​		使用bert-as-server，就可以使用预训练好的bert模型生成动态句向量以及ELMO风格的词向量。

​		[bert-as-server地址](https://github.com/hanxiao/bert-as-service)

## 2. 提取静态句向量和词向量

这个方法就比较简单了，直接使用Tensorflow将参数矩阵导出，用于下游的任务

## 3.finetune+知识蒸馏

​		有了pretrain的bert模型，可以使用finetune的方法训练一个比较精准的ctr预估模型，再用这个很复杂的模型蒸馏出一个简单的模型上线。
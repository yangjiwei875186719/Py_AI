# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
"""
建立网络模型结构
"""

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    #输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)
        #使用lstm
        # x, _ = self.lstm(x)
        #使用线性层
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        # 输出一个文本向量
        return x

#
class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)  # 建立孪生网络
        self.loss = nn.CosineEmbeddingLoss()  # 孪生网络的交叉熵

    # 计算余弦距离  1-cos(a,b)      cosine-sim(u,v)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)  # 两边都做归一化
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        # 计算了两个张量对应元素相乘后的和，这在数学上被称为点积（dot product）或内积（inner product），但仅当这两个张量是一维的（即向量）时，这个操作才直接对应于点积
        #  是一个逐元素相乘（element-wise multiplication）的操作，它接受两个张量（tensor1 和 tensor2）作为输入，并返回一个新的张量，其中每个元素都是输入张量对应元素相乘的结果。这里的 mul 是乘法（multiplication）的缩写，但它特指逐元素乘法，而不是矩阵乘法
        #  torch.mul(tensor1, tensor2) 计算 tensor1 和 tensor2 的逐元素乘积。
        #  torch.sum(..., axis=-1) 对上一步得到的结果张量沿着最后一个轴（axis=-1）进行求和。在PyTorch中，轴（axis）的编号从0开始，其中0是第一个轴（通常是批处理维度），-1是最后一个轴（在二维张量中通常是特征或列维度）。
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1) # 内积 cos值
        return 1 - cosine

    # Triplet Loss
    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)]) #greater than ,.gt(0)取出比0大的值

    # sentence : (batch_size, max_length)   有三种选择，传入两个句子，计算两个句子距离，传入layber就是计算损失函数，  传入一个句子，句子规划成向量
    # 如果使用triplet loss，要传入三个句子,a,p,n，留个作业 ，还有一处需要改造是数据上面loader.py random__train_sample,从train抽一行文本，正样本，从train抽两行，各一条形成负样本，三个样本
    # 训练集和测试集不一样，格式不一样
    def forward(self, sentence1, sentence2=None, target=None):
        #同时传入两个句子
        if sentence2 is not None:
            vector1 = self.sentence_encoder(sentence1)  # vec:(batch_size, hidden_size)
            vector2 = self.sentence_encoder(sentence2)
            #如果有标签，则计算loss
            if target is not None:
                return self.loss(vector1, vector2, target.squeeze())  # target 是1 还是 -1
            #如果无标签，计算余弦距离
            else:
                return self.cosine_distance(vector1, vector2)
        #单独传入一个能句子时，认为正在使用向量化力
        else:
            return self.sentence_encoder(sentence1)


# 选择优化器和学习率
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    s2 = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    l = torch.LongTensor([[1],[0]])
    y = model(s1, s2, l)
    print(y)
    # print(model.state_dict())
#coding:utf8

import torch
import torch.nn as nn
import jieba
import numpy as np
import random
import json
from torch.utils.data import DataLoader

"""
基于pytorch的网络编写一个分词模型
我们使用jieba分词的结果作为训练数据
看看是否可以得到一个效果接近的神经网络模型
不依赖词表
"""

class TorchModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_rnn_layers, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim, padding_idx=0) #shape=(vocab_size, dim)  len(vocab) + 1是因为此表没有给pad位置,
        self.rnn_layer = nn.RNN(input_size=input_dim,
                            hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=num_rnn_layers,
                            )
        self.classify = nn.Linear(hidden_size, 2)  # w = hidden_size * 2 映射成2维 因为要2分类，所以映射到2维
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  #input shape: (batch_size, sen_len), output shape:(batch_size, sen_len, input_dim)
        x, _ = self.rnn_layer(x)  #output shape:(batch_size, sen_len, hidden_size)  
        y_pred = self.classify(x)   #output shape:(batch_size, sen_len, 2) -> y_pred.view(-1, 2) (batch_size*sen_len, 2)

        """
        y_pred.view(-1, 2)
        三句话
        [[[0.2,0.8],[0.3,0.9]...] 第一个样本的值
        [[0.3,0.9],[0.3,0.9]...]
        [[0.2,0.8],[0.3,0.9]...]
        ]
        变成 一句话
              [[0.2,0.8],[0.3,0.9]... ,[0.3,0.9],[0.3,0.9]...,[0.2,0.8],[0.3,0.9]...]
        """
        """
        两层RNN
        x = (bs,length,dim)
        rnn1 = (bs,length,hidden_size)
        rnn2 = (hidden_size,hidden_size) # rnn2 =   (bs,length,hidden_size) 
        
        双向RNN 设置参数bidirecttional
        """
        # 交叉熵运算要求 y_pred(battch_size ,class_num) y(batch_size)
        # y.view(-1) [[0,1,0,1],[0,1,0,1]] -->[0,1,0,1,0,1,0,1]
        if y is not None:
            return self.loss_func(y_pred.reshape(-1, 2), y.view(-1)) # -1帮你自动生成,y.view(-1)展开一整个序列
        else:
            return y_pred

# 读取训练数据，加载数据
class Dataset:
    def __init__(self, corpus_path, vocab, max_length):
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.max_length = max_length
        self.load()

    def load(self):
        self.data = []
        with open(self.corpus_path, encoding="utf8") as f:
            for line in f:
                sequence = sentence_to_sequence(line, self.vocab)
                label = sequence_to_label(line)
                sequence, label = self.padding(sequence, label)
                sequence = torch.LongTensor(sequence)
                label = torch.LongTensor(label)
                self.data.append([sequence, label])
                #使用部分数据做展示，使用全部数据训练时间会相应变长
                if len(self.data) > 10000:
                    break

    #将文本截断或补齐到固定长度
    def padding(self, sequence, label):
        sequence = sequence[:self.max_length]  # 截断
        sequence += [0] * (self.max_length - len(sequence))  # 补齐
        label = label[:self.max_length]  # 截断
        label += [-100] * (self.max_length - len(label))  # 补齐 -100 是因为上面的交叉熵nn.CrossEntropyLoss(ignore_index=-100) 是不算的
        return sequence, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

#文本转化为数字序列，为embedding做准备
def sentence_to_sequence(sentence, vocab):
    sequence = [vocab.get(char, vocab['unk']) for char in sentence]
    return sequence

#基于结巴生成分级结果的标注
def sequence_to_label(sentence):
    words = jieba.lcut(sentence) # 结巴分词，正常是人标的，数据标注  [abc,d]
    label = [0] * len(sentence)  # [0,0,1,1]
    pointer = 0
    for word in words:
        pointer += len(word)
        label[pointer - 1] = 1
    return label

#加载字表  # 读取词表
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            vocab[char] = index + 1   #每个字对应一个序号
    vocab['unk'] = len(vocab) + 1
    return vocab

#建立数据集
def build_dataset(corpus_path, vocab, max_length, batch_size):
    dataset = Dataset(corpus_path, vocab, max_length) #diy __len__长度 __getitem__根据index获取值  Dataset自动湖北分层数据
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size) #torch  # DataLoader 数据集封装
    return data_loader


def main():
    epoch_num = 5        #训练轮数
    batch_size = 20       #每次训练样本个数
    char_dim = 50         #每个字的维度
    hidden_size = 100     #隐含层维度
    num_rnn_layers = 1    #rnn层数
    max_length = 20       #样本最大长度
    learning_rate = 1e-3  #学习率
    vocab_path = "chars.txt"  #字表文件路径
    corpus_path = "../corpus.txt"  #语料文件路径
    vocab = build_vocab(vocab_path)       #建立字表
    data_loader = build_dataset(corpus_path, vocab, max_length, batch_size)  #建立数据集
    model = TorchModel(char_dim, hidden_size, num_rnn_layers, vocab)   #建立模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)     #建立优化器
    #训练开始
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y in data_loader:
            optim.zero_grad()    #梯度归零
            loss = model.forward(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    return

#最终预测
def predict(model_path, vocab_path, input_strings):
    #配置保持和训练时一致
    char_dim = 50  # 每个字的维度
    hidden_size = 100  # 隐含层维度
    num_rnn_layers = 1  # rnn层数
    vocab = build_vocab(vocab_path)       #建立字表
    model = TorchModel(char_dim, hidden_size, num_rnn_layers, vocab)   #建立模型
    model.load_state_dict(torch.load(model_path))   #加载训练好的模型权重
    model.eval()
    for input_string in input_strings:
        #逐条预测
        x = sentence_to_sequence(input_string, vocab) #  转换成数据序列
        with torch.no_grad():
            result = model.forward(torch.LongTensor([x]))[0]
            result = torch.argmax(result, dim=-1)  #预测出的01序列  ，argmax 自从转换获取最大的,返回最大值的索引，2维向量，取出最大的以为
            #在预测为1的地方切分，将切分后文本打印出来
            for index, p in enumerate(result):
                if p == 1:
                    print(input_string[index], end=" ") # 1的序列加一个空格，不是1的序列不加空格
                else:
                    print(input_string[index], end="")
            print()



if __name__ == "__main__":
    main()
    input_strings = ["同时国内有望出台新汽车刺激方案",
                     "沪胶后市有望延续强势",
                     "经过两个交易日的强势调整后",
                     "昨日上海天然橡胶期货价格再度大幅上扬"]
    
    predict("model.pth", "chars.txt", input_strings)


#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现


“abcd”        ----每个字符转化成向量---->    4 * 5矩阵
4 * 5矩阵   ----向量求平均---->                     1 * 5向量
1 * 5向量   ----w*x + b线性公式 --->           实数
实数            ----sigmoid归一化函数--->        0-1之间实数



"""
"""
self.embedding = nn.Embedding(len(vocab), vector_dim)
在这行代码中，len(vocab) 表示词汇表的大小，用于指定嵌入层（embedding layer）的输入维度。具体来说，nn.Embedding 的第一个参数是一个整数，表示词汇表的大小，而第二个参数 vector_dim 表示词向量的维度。
词汇表是用来存储文本数据中所有不同词语的集合。每个词语都会被分配一个唯一的索引，作为其在词汇表中的位置。因此，词汇表的大小等于词汇表中不同词语的数量。
在自然语言处理任务中，词向量是一种常用的表示词语的方式。词向量将每个词语映射到一个固定长度的实数向量，以便能够在计算机上进行处理和表示。词向量的维度 vector_dim 决定了每个词语在嵌入层中的表示维度。
因此，为了创建一个合适大小的嵌入层，需要知道词汇表的大小（即不同词语的数量）和词向量的维度。在这里，len(vocab) 是通过 vocab 变量获取词汇表的大小，确保嵌入层能够适应输入数据的词汇表范围。
"""
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):  # vocab字典表  vector_dim每个字的维度 sentence_length样本文本长度
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层 将文本转成数字  len(vocab)文本长度，vector_dim每个字符向量化后的向量维度
        self.pool = nn.AvgPool1d(sentence_length)   #池化层，文本的长度向量求平均很常用， 当有很长的文本，对文本分类，将矩阵压缩成向量，数值相加，纵向求平均 4*5 矩阵 ---> 1*向量长度
        self.classify = nn.Linear(vector_dim, 1)     #线性层 向量层变成变成一个实数，因为y_pred也是一个真实值
        self.activation = torch.sigmoid     #sigmoid归一化函数 映射到0-1之间
        self.loss = nn.functional.mse_loss  #loss函数采用均方差损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值 batch_size,多少个样本，sen_len一个样本的长度， vector_dim一个样本中的一个字的 向量
    def forward(self, x, y=None):                  # batch_size个,sen_len * vector_dim 的矩阵
        print("X:",x.shape)
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        print("X.embedding",x.shape)
        x = x.transpose(1, 2)                      #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)   #1,2时索引的向量进行交换， 做一个转换，对[3, 4, 5]位置做索引，对4,5做转换
        print("x.transpose",x.shape)
        x = self.pool(x)                           #(batch_size, vector_dim,sen_len  )->(batch_size, vector_dim, 1)，例3个5*4矩阵转化成3个5*1矩阵
        print("x.pool",x.shape)
        x = x.squeeze()                            #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)  得到3*5张量
        print("x.squeeze", x.shape)
        x = self.classify(x)                       #(batch_size, vector_dim) -> (batch_size, 1) 3*5 5*1 -> 3*1
        print("x.classify", x.shape)
        y_pred = self.activation(x)                #(batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #指定哪些字出现时为正样本
    if set("abc") & set(x): # 出现a 或b 或c
        y = 1
    #指定字都未出现，则为负样本
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding，字典取值带默认值的情况
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):  # vocab字典表  char_dim每个字的维度 sentence_length样本文本长度
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1   #负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1   #正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 30         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    # plt.legend()
    # plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

# #使用训练好的模型做预测
# def predict(model_path, vocab_path, input_strings):
#     char_dim = 20  # 每个字的维度
#     sentence_length = 6  # 样本文本长度
#     vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
#     model = build_model(vocab, char_dim, sentence_length)     #建立模型
#     model.load_state_dict(torch.load(model_path))             #加载训练好的权重
#     x = []
#     for input_string in input_strings:
#         x.append([vocab[char] for char in input_string])  #将输入序列化
#     model.eval()   #测试模式
#     with torch.no_grad():  #不计算梯度
#         result = model.forward(torch.LongTensor(x))  #模型预测
#     for i, input_string in enumerate(input_strings):
#         print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i])) #打印结果
#


if __name__ == "__main__":
    main()
    # test_strings = ["fnvfee", "wzsdfa", "rqwdeg", "nakwww"]
    # predict("model.pth", "vocab.json", test_strings)

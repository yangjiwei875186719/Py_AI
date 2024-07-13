
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from torch import optim

"""
构建一个nlp任务，不用pooling层，用rnn做一个多分类，结合交叉熵
特定字符，“你"保证一定出现在字符串，“你”在文本里面第几位，就属于第几类
"""
# 生成字典：
def build_vocab():
    chars = "你我他好坏上中下" # 字符集
    vocab = {"pad":0}
    for index,char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab) + 1
    return vocab
print("build_vocab",build_vocab())
# 找 字符在列表中在第几位
def find_indices(lst, value):
    return [index for index, item in enumerate(lst) if item == value]


#随机生成一个样本
def build_sample(vocab,sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # print("x",x)
    if x.count("你") == 1:
        indices = find_indices(x, "你")
        y = [idx + 1 for idx in indices]
        # print("y",y)
    else:
        y = list([0])
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x,y
# 生成一个样本
vocab= build_vocab()
print("build_sample",build_sample(vocab,2))
# 生成多组样本
def build_dataset(vocab,sample_length,sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x),  torch.LongTensor(np.array(dataset_y).flatten())  # 扁平化

vocab = build_vocab()
sentence_length = len(vocab) - 1  # 几列
sample_num = 3  # 几行
dataset_x, dataset_y = build_dataset(vocab,sample_num,sentence_length) # sample_num几行，  sentence_length 几列   sample_num几行文本，sentence_length几列
print("dataset_x",dataset_x)
print("dataset_y",dataset_y)  # 一维  sample_num列
print('创建数据完成')
class RNNModel(nn.Module):
    def __init__(self, vocab, char_dim, sentence_length, hidden_size,num_class):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, char_dim, padding_idx=0)
        self.nb = nn.LayerNorm(char_dim)
        self.dropout = nn.Dropout(0.3)
        self.rnn_layer = nn.LSTM(input_size=char_dim, hidden_size=hidden_size, batch_first=True)
        self.classify = nn.Linear(hidden_size,num_class)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x) # batch_size,sen_len ----- batch_size,sen_len,char_dim
        x = self.nb(x)
        x = self.dropout(x)
        x, (ht,ct) = self.rnn_layer(x) # batch_size,sen_len,char_dim ----- batch_size,sen_len,hidden_size
        y_pred = self.classify(ht.squeeze()) # batch_size,sen_len,hidden_size ---- batch_size,sen_len,num_class
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred
def main():
    epoch_num = 10
    batch_size = 20
    char_dim = 50
    hidden_size = 100
    vocab = build_vocab()
    sentence_length = len(vocab) - 1
    sample_num = 5000
    num_class = sentence_length
    model = RNNModel(vocab, char_dim, sentence_length, hidden_size,num_class)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset_x, dataset_y = build_dataset(vocab,sample_num,sentence_length)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(sample_num // batch_size):
            x = dataset_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = dataset_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        print("第%d轮,loss=%f" % (epoch + 1,np.mean(watch_loss)))
    torch.save(model.state_dict(), 'model.pth')

def predict(model_path,input_str):
    vocab = build_vocab()
    char_dim = 50
    hidden_size = 100
    sentence_length = len(vocab) - 1
    num_class = sentence_length
    model = RNNModel(vocab, char_dim, sentence_length, hidden_size,num_class)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    X = []
    for sentence in input_str:
        x = [vocab.get(char) for char in sentence]
        X.append(x)
    X = torch.LongTensor(X)
    with torch.no_grad():
        y_pred = model(X)
    for y_p, y_t in zip(y_pred, input_str):
        i = y_t.index('你')
        print("正确位置:%d,预测位置:%d,是否正确:%s" % (i, torch.argmax(y_p), (torch.argmax(y_p) == i)))
"""
错误是因为生成样本时，y 的取值范围应该是 0 到 sentence_length-1，而不是 1 到 sentence_length。这是因为索引从 0 开始计数，所以类别标签应该从 0 开始。
"""

if __name__ == '__main__':
    main()
    print('模型创建成功')
#
# class RNNModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, vocab):
#         super(RNNModel, self).__init__()
#         self.embedding = nn.Embedding(len(vocab), hidden_size)
#         self.rnn = nn.RNN(input_size, hidden_size , batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#         self.criterion = nn.CrossEntropyLoss()
#
#     def forward(self, input_seq, labels=None):
#         embedded = self.embedding(input_seq)
#         output, _ = self.rnn(embedded)
#         output = self.fc(output[:, -1, :])  # 使用最后一个时间步的输出作为分类器的输入
#
#         if labels is not None:
#             loss = self.criterion(output, labels)
#             return loss
#         else:
#             output = self.softmax(output)
#             return output
#
# #建立模型
# def build_model(vocab, char_dim, sentence_length,num_class):  # vocab字典表  char_dim每个字的维度 sentence_length样本文本长度
#     model = RNNModel(char_dim, sentence_length,num_class, vocab)
#     return model
# #测试代码
# #用来测试每轮模型的准确率
# def evaluate(model, vocab, sample_length):
#     model.eval()
#     x,y = build_dataset(200,vocab,sample_length)
#     print("本次预测集中共有200个样本")
#     correct, wrong = 0, 0
#     with torch.no_grad():
#         y_pred = model(x)      #模型预测
#         for y_p,y_t  in zip(y_pred, y):
#             if torch.argmax(y_p) == int(y_t):
#                 correct += 1
#             else:
#                 wrong += 1
#     print("正确预测个数：%d,正确率:%f" % (correct,correct/(correct+wrong)))
#
# # 主程序
# def main():
#     epoch_num = 20        # 训练轮数
#     batch_size = 20       # 每次训练样本个数
#     train_sample = 500    # 每轮训练总共训练的样本个数
#     char_dim = 7         # 每个字的维度
#     sentence_length = 5  # 样本文本长度 10
#     output_size = 6
#     learing_rate = 0.00005 # 学习率
#     # 建立字表
#     vocab1 = build_vocab()
#     # 建立模型
#     model =  RNNModel(char_dim, sentence_length, 2, vocab1)
#     # 选择优化器
#     optim = torch.optim.Adam(model.parameters(), lr=learing_rate)
#     log = []
#     for epoch in range(epoch_num):
#         model.train()
#         watch_loss = []
#         for batch in range(int(train_sample/batch_size)):
#             x, y = build_dataset(batch_size,vocab1,sentence_length) # 构造一组训练样本
#             print("x",x)
#             print("y",y)
#             optim.zero_grad() # 梯度归0
#             loss = model(x,y) # 计算loss
#             loss.backward()  # 计算梯度
#             optim.step()  # 更新权重
#             watch_loss.append(loss.item())
#             print("======第/%d轮平均loss：%f" % (epoch + 1, np.mean(watch_loss)))
#         # acc = evaluate(model,vocab1,sentence_length)
#         # log.append([acc,np.mean(watch_loss)])
#         # print(acc)
#     # #画图
#     # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
#     # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
#     # plt.legend()
#     # plt.show()
#     # #保存模型
#     # torch.save(model.state_dict(), "model.pth")
#     # # 保存词表
#     # writer = open("vocab111.json", "w", encoding="utf8")
#     # writer.write(json.dumps(vocab1, ensure_ascii=False, indent=2))
#     # writer.close()
#     return
#
# if __name__ == "__main__":
#     main()
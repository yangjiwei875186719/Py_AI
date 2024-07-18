# coding:utf8
import random
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
输入一个字符串，根据字符a所在位置进行分类
对比rnn和pooling做法

"""
"""
rnn_out 是RNN层的输出，它的形状是 (batch_size, sequence_length, vector_dim)，其中：
batch_size 表示批次中样本的数量。
sequence_length 表示输入序列的长度，即句子中的词数量。
vector_dim 表示隐藏状态的维度，也是词向量的维度。
在这行代码中，我们使用 [:, -1, :] 进行索引操作，具体含义如下：

: 表示在第一个维度（batch_size）上选择所有的元素，即保持批次维度不变。
-1 表示在第二个维度（sequence_length）上选择最后一个时间步的隐藏状态。
: 表示在第三个维度（vector_dim）上选择所有的元素，即保持隐藏状态的维度不变。
这样，x 就表示输入序列经过RNN层处理后的最后一个时间步的隐藏状态，它的形状是 (batch_size, vector_dim)。在文本分类任务中，这个隐藏状态可以看作是整个句子的固定长度表示，用于进行后续的分类操作。
通过选择最后一个时间步的隐藏状态作为句子表示，模型能够捕捉输入序列的整体语义信息，并将其编码为一个固定维度的向量，以便进行分类或其他下游任务。

在循环神经网络（RNN）中，隐藏状态（hidden state）是指在每个时间步产生的中间状态。隐藏状态可以看作是模型对过去观察的记忆或表示。
在文本分类任务中，RNN被用来处理输入序列，其中每个时间步对应输入序列中的一个词或字符。在每个时间步，RNN模型会接收当前时间步的输入以及前一个时间步的隐藏状态，并产生当前时间步的输出和下一个时间步的隐藏状态。
在这个过程中，每个时间步的隐藏状态是根据当前时间步的输入、前一个时间步的隐藏状态以及模型的参数计算得到的。隐藏状态可以看作是模型对过去输入的编码或表示，它携带了过去输入序列的信息。
在文本分类任务中，我们通常只关注最后一个时间步的隐藏状态，因为它包含了整个输入序列的总结和编码信息。这个最后时间步的隐藏状态可以被视为整个句子的固定长度表示，用于进行分类或其他后续任务。
总结起来，时间步的隐藏状态是RNN模型在每个时间步计算得到的中间状态，它携带了过去输入序列的信息，并用于建模序列中的依赖关系和提取序列的特征。在文本分类任务中，我们常常使用最后一个时间步的隐藏状态作为整个句子的表示。
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):  # __init__方法是类的构造函数
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
        #可以自行尝试切换使用rnn或pooling
        # self.pool = nn.AvgPool1d(sentence_length)   #池化层
        self.rnn = nn.RNN(vector_dim, vector_dim+ 1, batch_first=True)   # self.rnn是一个nn.RNN层，它接受词向量作为输入，并在序列维度上进行逐步处理。该层的输出是每个时间步的隐藏状态

        # vector_dim + 1是根据RNN的输出层作为linear的输入层     +1的原因是可能出现 a 不存在的情况，那时的真实label在构造数据时设为了sentence_length
        self.classify = nn.Linear(vector_dim + 1, sentence_length + 1) # 输出映射到相同的维度 # self.classify是一个线性层 (nn.Linear)，它将RNN的输出映射到与句子长度加1相同的维度。这是为了适应可能出现的"a不存在"的情况，其中句子长度被设为了sentence_length
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        print("x",x.shape)        # （batch_size ,sentenc_length）
        x = self.embedding(x)      # （batch_size ,sentenc_length） -->（batch_size ,sentenc_length ,vecor_dim）
        print("x.embedding", x.shape)
        #使用pooling的情况
        # x = x.transpose(1, 2)
        # x = self.pool(x)
        # x = x.squeeze()
        #使用rnn的情况
        rnn_out, hidden = self.rnn(x) # （batch_size ,sentenc_length ,vecor_dim）-->（batch_size ,sentenc_length ,vecor_dim + 1）
        print("x.rnn", rnn_out.shape)
        # batch,max_len,vector_dim
        print("hidden",hidden.shape)
        print("hidden.squeeze", hidden.squeeze().shape)
        x = rnn_out[:, -1, :]  #或者写hidden.squeeze()也是可以的，因为rnn的hidden就是最后一个位置的输出 x = rnn_out[:, -1] x = rnn_out[:, sequence_length - 1, :] # （batch_size ,sentenc_length ,vecor_dim + 1）--->（batch_size  ,vecor_dim + 1）
        print("x.rnn_out",x.shape)
        #接线性层做分类
        y_pred = self.classify(x)  # （batch_size  ,vecor_dim + 1） --->（batch_size  ,sentence_length + 1）
        print("y_pred:",y_pred.shape)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijk"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

#随机生成一个样本
def build_sample(vocab, sentence_length):
    #注意这里用sample，是不放回的采样，每个字母不会重复出现，但是要求字符串长度要小于词表长度
    x = random.sample(list(vocab.keys()), sentence_length)
    #指定哪些字出现时为正样本
    if "a" in x:
        y = x.index("a")  # 0-9
    else:
        y = sentence_length # y = 句子长度 也就是10
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    print("本次预测集中共有%d个样本"%(len(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 40       #每次训练样本个数
    train_sample = 1000    #每轮训练总共训练的样本总数
    char_dim = 30         #每个字的维度
    sentence_length = 10   #样本文本长度
    learning_rate = 0.001 #学习率
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
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 30  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i])) #打印结果



if __name__ == "__main__":
    main()
    test_strings = ["kijabcdefh", "gijkbcdeaf", "gkijadfbec", "kijhdefacb"]
    predict("model.pth", "vocab.json", test_strings)
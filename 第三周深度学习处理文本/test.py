import torch
import torch.nn as nn
import torch.optim as optim

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_seq, labels=None):
        embedded = self.embedding(input_seq)
        output, _ = self.rnn(embedded)
        output = self.fc(output[:, -1, :])  # 使用最后一个时间步的输出作为分类器的输入

        if labels is not None:
            loss = self.criterion(output, labels)
            return output, loss
        else:
            output = self.softmax(output)
            return output

# 数据准备
input_texts = ["你好吗", "你在哪里", "你喜欢什么", "你喜欢吃什么"]
labels = [0, 1, 2, 3]  # 标签对应的类别

# 构建词表
word2index = {"你": 0, "好": 1, "吗": 2, "在": 3, "哪": 4, "里": 5, "喜": 6, "欢": 7, "什": 8, "么": 9, "吃": 10}
index2word = {index: word for word, index in word2index.items()}
vocab_size = len(word2index)

# 将文本转换为张量
input_seqs = [[word2index[word] for word in text] for text in input_texts]
input_tensors = [torch.tensor(seq) for seq in input_seqs]
print("input_tensors",input_tensors)
label_tensor = torch.tensor(labels)
print("label_tensor",label_tensor)


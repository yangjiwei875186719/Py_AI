# -*- coding: utf-8 -*-
from torch import nn
import torch
from transformers import BertModel,BertTokenizer
from torch.optim import Adam,SGD
class SftModel(nn.Module):
    def __init__(self,config,vocab_size):
        super(SftModel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config["bert_path"], return_dict=False)
        hidden_size = self.bert.config.hidden_size
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy
    def forward(self,x,y=None):
        if y is not None:
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1,y_pred.shape[-1]),y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)
def choose_optimizer(self,config,model):
    optimizer = config["optimizer"]
    learning_rate = config["learing_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr = learning_rate)
    elif optimizer == "SGB" :
        return SGD(model.parameters(), lr = learning_rate)

if __name__ == '__main__':
    from config import Config
    model = SftModel(Config,21128)

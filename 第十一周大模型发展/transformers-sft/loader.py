# -*- coding: utf-8 -*-

import json
import re
import os

import tokenizers
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import BertModel ,BertTokenizer
import torch
"""
数据加载
"""


class DataGenerator:
    def __init__(self,data_path, config, logger):
        self.config = config
        self.tokenizer = load_vocab(config["bert_path"])
        # self.token = BertModel.from_pretrained(config["bert_path"],return_dict=False)
        self.logger = logger
        self.path = data_path
        self.config["vocab_size"] = self.tokenizer.vocab_size
        print(self.config["vocab_size"])
        self.data = []
        self.load()
    def load(self):
        titles=[]
        contents=[]
        with open(self.path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                titles.append(title)
                contents.append(content)
            # print("titles:",titles)
            # print(f"contents:{contents}")
                self.encode_sentence(self.tokenizer,title, content)
        return
    def encode_sentence(self,tokenizer,title,content):
        input = tokenizer.encode(title, max_length = self.config["input_max_length"],add_special_tokens=False,padding='max_length',truncation=True)
        # mask = torch.tril(torch.ones(self.config["output_max_length"],self.config["output_max_length"]))
        mask = create_lower_triangular_mask(self.config["output_max_length"])
        # print(f"mask:{mask} \n mask_type:{mask.shape}")
        output = tokenizer.encode(content, max_length = self.config["output_max_length"], add_special_tokens=False, padding='max_length',truncation=True)
        # 将mask与output_ids合并
        # output_with_mask = output * mask.view(-1)
        # print("mask.shape",mask.shape)
        # print("torch.LongTensor(input):",torch.LongTensor(input).shape,",torch.LongTensor(output).shape,",torch.LongTensor(output).shape,"torch.LongTensor(output).masked_fill(mask,0)",torch.LongTensor(output).masked_fill(mask,0).shape)
        self.data.append([torch.LongTensor(input),torch.LongTensor(output)])  # torch.LongTensor(output).masked_fill(mask,0)
        # print("(self.data",self.data)
        return

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
def create_lower_triangular_mask(seq_length):
    """创建下三角形的mask张量"""
    mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(mask)
def load_vocab(vocab_path):
    vocab = BertTokenizer.from_pretrained(vocab_path)
    return vocab
# #用torch自带的DataLoader类封装数据
def load_data(data_path, config, logger, shuffle=True):
    dg = DataGenerator(data_path, config, logger)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
if __name__ == "__main__":
    from config import Config
    # DataGenerator(Config,1)
    dl = load_data(Config["train_data_path"],Config,1)
    # dl = dict([(y, x) for x, y in dl.dataset.vocab.items()])
    # print("self.reverse_vocab",dl)
    # print(dl[1])
    # 遍历 DataLoader 中的数据
    for index,data in enumerate(dl):
        input, output = data
        print(f"index:{index},input:{input.shape},output:{output.shape}")
        break



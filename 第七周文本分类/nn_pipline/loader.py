# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config  # 参数
        self.path = data_path
        self.index_to_label = {0: '家居', 1: '房产', 2: '股票', 3: '社会', 4: '文化',
                               5: '国际', 6: '教育', 7: '军事', 8: '彩票', 9: '旅游',
                               10: '体育', 11: '科技', 12: '汽车', 13: '健康',
                               14: '娱乐', 15: '财经', 16: '时尚', 17: '游戏'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())  # 标签字段存储到字段中
        self.config["class_num"] = len(self.index_to_label)  # 标签数量
        if self.config["model_type"] == "bert":    # 如果模型为bert 就加载预训练文件中的权重
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])    # 加载字典表，形成一个字典
        self.config["vocab_size"] = len(self.vocab)     # 字典的长度，放到config配置文件里面
        self.load()  # 加载的方法


    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:  # 对文件每行遍历
                line = json.loads(line)
                tag = line["tag"]
                # print("tag",tag)
                label = self.label_to_index[tag] # 从健里面寻找对应的值
                # print(label)
                title = line["title"]
                # print("title:",title)
                if self.config["model_type"] == "bert":
                    # 如果 model_type 为 "bert"，则使用 self.tokenizer 对标题 title 进行编码。这里的 encode 方法会将输入的文本转换为模型可接受的输入格式，其中 max_length 参数指定最大长度，pad_to_max_length=True 则表示填充到最大长度。
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
                    print("input_id",input_id)
                else:
                    # 调用了一个名为 encode_sentence 的方法来处理标题 title。这个方法可能是根据不同的模型类型（非 BERT 类型）编码标题的自定义函数。
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return
    # 对文本进行处理
    def encode_sentence(self, text):  # 这是一个方法定义，接受两个参数 self 和 text，其中 self 是类的实例，text 是要编码的文本。
        input_id = []
        for char in text:  # 遍历文本中的每个字符。
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))  # 对于每个字符，使用 self.vocab 字典来获取其对应的编码。如果字符不在 self.vocab 中，就使用 [UNK] 的编码（表示未知字符）
        input_id = self.padding(input_id)  # 调用 self.padding 方法对 input_id 进行填充操作。这里假设 padding 方法是另一个方法，用于填充序列到特定长度，确保输入的长度是相同的。
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):  # 这是一个填充方法的定义，接受两个参数 self 和 input_id，其中 self 是类的实例，input_id 是要进行填充操作的列表。
        input_id = input_id[:self.config["max_length"]]  # 将输入的 input_id 列表截取至最大长度 self.config["max_length"]。如果列表长度超过了最大长度，则截取到最大长度；如果列表长度小于最大长度，则保持不变。
        input_id += [0] * (self.config["max_length"] - len(input_id))  # 计算需要填充的数量，即最大长度减去当前列表的长度。然后将这个数量的 0 添加到列表的末尾，以达到最大长度。
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):  # 加载词典形成有一个字典
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):  # 这是一个函数定义，接受三个参数 data_path（数据路径）、config（配置信息）、shuffle（是否对数据进行打乱，默认为 True）。
    dg = DataGenerator(data_path, config)  # 通过 DataGenerator 类（假设是自定义的数据生成器类）创建一个数据生成器 dg，并传入数据路径和配置信息
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)  # 使用 PyTorch 的 DataLoader 类来封装数据。指定数据生成器 dg 作为数据源，指定批处理大小为配置中指定的 batch_size，同时根据 shuffle 参数来决定是否对数据进行打乱。
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../data/valid_tag_news.json", Config)
    print("dg[1]:",dg[0])

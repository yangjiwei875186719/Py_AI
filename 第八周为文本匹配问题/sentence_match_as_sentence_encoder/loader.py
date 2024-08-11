# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"]) # 加载字词典
        self.config["vocab_size"] = len(self.vocab)  # 词典长度，不能超过这个数
        self.schema = load_schema(config["schema_path"])  # 加载label, 将label中文和整数对应
        self.train_data_size = config["epoch_data_size"]  # 由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.data_type = None  # 用来标识加载的是训练集还是测试集 "train" or "test"
        self.load()

    # 加载数据，把数据形成一个标签，对应张量的训练集或者测试集
    def load(self):
        self.data = []  # 测试的格式
        self.knwb = defaultdict(list)   #训练的格式 defaultdict 是 collections 模块中的一个类，它提供了一个类似字典的对象，但是它会为任何尚未在字典中存在的键自动提供一个默认值。这意呀着，当你尝试访问一个不存在的键时，defaultdict 会自动为该键创建一个条目，并将默认值赋给它，而不是抛出一个 KeyError 异常
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                #加载训练集  格式不一样
                """
                  `isinstance()` 函数是 Python 内置的函数，用于判断一个对象是否为指定类型或指定类型的子类。其语法如下：
                isinstance(object, classinfo)
                """
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    # print("questions形成的是一个列表：",questions)
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)

                        # self.schema[label] 从schema字典中获取key对应的value值作为knwb的key。append将获取到的张量作为knwb字典里的value
                        self.knwb[self.schema[label]].append(input_id)  # # self.knwb[keyword].append(word)  将单词添加到与关键词相关联的列表中   # # 如果关键词不存在，会自动创建一个新的列表
                        # print("self.knwb", self.knwb) # key为标签张量，value为每个文本的张量
                #加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)  # 列表
                    question, label = line  # 列表中两个值，一个是问题，一个是测试的结果
                    input_id = self.encode_sentence(question)  # 对测试集的文本进行遍历
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])  # 获取schema的value转化成张量
                    # print("self.data.append([input_id, label_index]):",self.data.append([input_id, label_index])) #举例 [tensor([1548, ... ]),tensor([-1])]
                    self.data.append([input_id, label_index])
        return

    # 考虑多种情况，利用结巴分词遍历和字的遍历，从字典中找相应的字或者词，如果找不到就赋值为UNK
    # 文本转化成数字，为转化成深度可识别的张量做准备
    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":  # 如果配置（self.config["vocab_path"]）中的 "vocab_path" 等于 "words.txt"，则使用 jieba 分词库对文本进行分词。jieba 是一个流行的中文分词库，能够将文本分割成一系列的词语
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:   # 如果配置不是 "words.txt"，则直接按字符处理文本。对于文本中的每个字符（char），同样使用 self.vocab.get(char, self.vocab["[UNK]"]) 查找其索引
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        # print("input_id:",input_id)   # 索引当成整数
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):  # 最总返回值
        if self.data_type == "train":
            return self.random_train_sample() #随机生成一个训练样本
        else:
            return self.data[index]

    #依照一定概率“生成”负样本或正样本
    #负样本从随机两个不同的标准问题中各随机选取一个
    #正样本从随机一个标准问题中随机选取两个
    # 一般会设置小数，大于这个小数是正样本，小于这个小数 时候负样本，控制正负样本的比例

    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())  # 得到有多少个 label
        # print("standard_question_index",standard_question_index)
        #随机正样本
        if random.random() <= self.config["positive_sample_rate"]:
            p = random.choice(standard_question_index)
            #如果选取到的标准问下不足两个问题，则无法选取，所以重新随机一次
            if len(self.knwb[p]) < 2:
                return self.random_train_sample()
            else:
                # random.sample() 是 Python 中 random 模块的一个函数，用于从指定的序列中随机获取指定长度的片段。这个函数不会修改原有序列，而是返回一个新的列表，包含从原序列中随机选择的指定数量的元素。这些元素以被选中的顺序返回，且每个元素只能被选中一次（即无重复
                # 第一个参数 一个序列（如列表、元组、字符串等），表示可以从中抽取样本的总体 。第二个参数：一个整数，表示要抽取的样本数量
                s1, s2 = random.sample(self.knwb[p], 2)
                return [s1, s2, torch.LongTensor([1])]
        #随机负样本
        else:
            p, n = random.sample(standard_question_index, 2) # 先选择两标签，从标签里面选择
            s1 = random.choice(self.knwb[p])  # choice() 函数从一个与某个键 p 相关联的列表中随机选择一个元素
            s2 = random.choice(self.knwb[n])
            return [s1, s2, torch.LongTensor([-1])]



#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始

    return token_dict

#加载schema 形成 形成字典
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())
# print("load_schema",load_schema('../data/schema.json'))

"""
用torch自带的DataLoader类封装数据，
在 PyTorch 中，DataLoader 类是一个非常强大的工具，用于封装数据集并提供批量加载、打乱数据、多进程数据加载等功能。
然而，DataLoader 预期接收的是一个实现了 __getitem__ 和 __len__ 方法的对象，这通常是 Dataset 类的一个实例。在你的代码中，
你尝试直接使用一个名为 DataGenerator 的类作为 DataLoader 的输入。
如果 DataGenerator 类没有实现 __getitem__ 和 __len__ 方法，那么这将导致错误。
"""

def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)  # dg是个数据集dataset
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    # from config import Config
    # dg = DataGenerator("valid_tag_news.json", Config)
    # print(dg[1])

    from config import Config
    dg = DataGenerator("../data/train.json", Config)
    print(dg.__len__())
    print(dg[1])  # 调用的就是get方法取出一条，返回的是两个文本向量和一个label

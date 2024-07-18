
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
一个文本，一个结果，比值 NLPDemo.py
"""
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
vocab = build_vocab()
print(build_dataset(2,vocab,4))



"""
分类 第三周参考作业
"""
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

vocab = build_vocab()
print(build_dataset(2,vocab,4))
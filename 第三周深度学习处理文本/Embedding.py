#coding:utf8
import torch
import torch.nn as nn

'''
embedding层的处理
就像是一本巨大的字典，其中每个单词都对应一个数字列表（向量）
Embedding可以将简单的单词转换成计算机能够处理的数值形式，让计算机能够更好地学习和理解自然语言
将离散值转化为向量
'''

num_embeddings = 6  #通常对于nlp任务，此参数为字符集字符总数，必须是已知的
embedding_dim = 5   #每个字符向量化后的向量维度
embedding_layer = nn.Embedding(num_embeddings, embedding_dim) # 6个5维向量  padding_idx=0参数不会对计算造成影响，随机生成也是有区间的，在代码可以找到
print("随机初始化权重")
print(embedding_layer.weight)
print("################")

#构造字符表
vocab = {
     "[pad]":0,
    "a" : 0,
    "b" : 1,
    "c" : 2,
    "d" : 3,
    "e" : 4,
    "f" : 5,
    # "你好":6, str_to_sequence就不能用了，需要按词区分
}
# padding 补齐
# [1,2,3,0,0] 后面两位补成0
# 截断 一共是1,2,3,4,5,6，会丢失信息，无奈的选择
#[1,2,3,4,5]
# unk 未知字符都使用统一的向量，预测会出现问题,没有办法的办法，退而求其次 例‘&!@#$%^’

# def str_to_sequence(string, vocab):
#     return [vocab[s] for s in string]
def str_to_sequence(string, vocab):
    result = []
    for s in string:
        result.append(vocab[s]) # 字典获取
    return result
string1 = "abcde"
string2 = "ddccb"
string3 = "fedab"

sequence1 = str_to_sequence(string1, vocab)

sequence2 = str_to_sequence(string2, vocab)
sequence3 = str_to_sequence(string3, vocab)

print("sequence1",sequence1)
print("sequence2",sequence2)
print("sequence3",sequence3)

x = torch.LongTensor([sequence1, sequence2, sequence3]) # Long整型，不能有小数点，将三个文本放到一个张量里面
print('x',x)
embedding_out = embedding_layer(x)
print("embedding_out",embedding_out)


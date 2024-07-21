#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from collections import defaultdict

"""
支持三种分词模式

1、精确模式，试图将句子最精确地切开，适合文本分析；
        result1 = jieba.cut(str2)
2、全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；
        result2 =jieba.cut(str1,cut_all = True)
3、搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
        result3 = jieba.cut_for_search(str3)
"""
#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    sentences = set()  # 创建空的集合  { 1,2,3,4 }
    with open(path, encoding="utf8") as f:
        for line in f:  # 对每一行进行遍历
            sentence = line.strip()  # 空格切分  '新增资金入场 沪胶强势创年内新高'
            sentences.add(" ".join(jieba.cut(sentence)))  # jieba分词,对一行数据进行切分
    print("获取句子数量：", len(sentences))
    return sentences  # {'卡未 离身 钱 却 被盗   江苏 ATM 出现 冒牌 读卡器', '梁博 20 日 央视 直播 再现 live 实力   冲击 冠军',...}
# # 举例
# sentences = load_sentence("titles.txt")
# print(sentences)

#将文本向量化 词向量到文本向量的转化
def sentences_to_vectors(sentences, model):
    vectors = []   # 创建空的列表
    for sentence in sentences: # 集合 对每一个进行遍历
        words = sentence.split()  #sentence是分好词的，空格分开  将当前句子按空格分割成单词，并将结果存储在words列表中。
        vector = np.zeros(model.vector_size)  # vector_size向量  # 创建一个全零向量，大小与词向量模型的维度一致。
        #所有词的向量相加求平均，作为句子向量
        for word in words: # 对集合中的一个句子 进行遍历
            try:
                vector += model.wv[word]  #  尝试从词向量模型中获取单词的词向量，并将结果累加到向量中
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size) #  # 如果单词不在词向量模型中，则将全零向量累加到向量中。
        vectors.append(vector / len(words))  # 平均整句话的向量 # 将句子向量除以句子长度（单词数量），得到平均向量。
    return np.array(vectors) # # 将句子向量列表转换为NumPy数组，并返回结果。
#  测试
model = load_word2vec_model(r"D:\\appdev\PyProject\Py_AI\第五周词向量及文本向量\model.w2v") #加载词向量模型
sentences = load_sentence(r"D:\\appdev\PyProject\Py_AI\第五周词向量及文本向量\\titles.txt")  #加载所有标题
vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

def main():
    model = load_word2vec_model(r"D:\appdev\PyProject\Py_AI\第五周词向量及文本向量\model.w2v") #加载词向量模型
    sentences = load_sentence(r"D:\appdev\PyProject\Py_AI\第五周词向量及文本向量\titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量 42 该行代码用于确定 K-means 算法的聚类数量。它计算句子数量的平方根，并将其转换为整数。平方根值被用作启发式方法，用于估计合理的聚类数量。
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类 ：该行代码使用 scikit-learn 的 sklearn.cluster 模块创建了一个 KMeans 类的实例，并指定了聚类数量。
    kmeans.fit(vectors)          #进行聚类计算 该行代码通过将 K-means 算法应用于句子向量，执行聚类计算。fit() 方法计算聚类中心，并将每个句子分配到一个聚类中心。

    sentence_label_dict = defaultdict(list) # 该行代码创建了一个 defaultdict 对象，用于按照句子的聚类标签将句子进行分组。
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签 循环遍历每个句子及其对应的聚类标签，使用 zip() 函数同时处理句子和标签。
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    for label, sentences in sentence_label_dict.items(): #  遍历每个聚类标签及其对应的句子。
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()


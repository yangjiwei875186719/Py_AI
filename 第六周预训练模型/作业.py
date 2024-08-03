import torch
import math
import numpy as np
from transformers import BertModel

'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''

bert = BertModel.from_pretrained(r"D:\appdev\PyProject\Py_AI\第六周预训练模型\bert-base-chinese", return_dict=False) # return_dict 老版本和新版本不一样
state_dict = bert.state_dict()  # 查看权重字典

bert.eval()
x = np.array([2450, 15486, 102, 2110])   #假想成4个字的句子
torch_x = torch.LongTensor([x])          #pytorch形式输入
## seqence_output 过了12层的输出
seqence_output, pooler_output = bert(torch_x)  # seqence_output 每个字对应的向量 4 * 768   pooler_output 1*768向量 pooler_output可做分类任务 seqence_output可做分词的任务 现在演变成都用 seqence_output

print(seqence_output.shape, pooler_output.shape)
# print(seqence_output, pooler_output)

print("bert.state_dict().keys():",bert.state_dict().keys())  #查看所有的权值矩阵名称   例：word_embeddings  position_embeddings  token_type_embeddings  LayerNorm


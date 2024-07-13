#coding:utf8
import torch
import torch.nn as nn

'''
pooling层的处理
降低后续网络层的输入维度，缩减模型大小，提高计算速度
图搞了特征的鲁棒性（健壮性），防止过拟合
最大，平均等
'''

#pooling操作默认对于输入张量的最后一维进行
#入参5，代表把五维池化为一维
layer = nn.AvgPool1d(4)  # 1d 一维池化层 2d二维池化层， 一般用1维的，因为文本长度为4，所以用4
#随机生成一个维度为3x4x5的张量
#可以想象成3条,文本长度为4,向量长度为5的样本
x = torch.rand([3, 4, 5])
print(x)
print(x.shape)  # x.shape 对象尺度，对于矩阵就是n行m列
x = x.transpose(1,2)   # 做一个转换，对[3, 4, 5]位置做索引，对4,5做转换
print(x.shape, "交换后")
print("交换",x)
#经过pooling层
y = layer(x)
print(y)
print(y.shape)
#squeeze方法去掉值为1的维度
y = y.squeeze()
print(y)
print(y.shape)

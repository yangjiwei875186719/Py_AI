#coding:utf8

import torch
import torch.nn as nn
import numpy as np


"""
基于pytorch的网络编写
测试dropout层
作用：减少过拟合
按照指定概率，随机丢弃一些神经元（将其化为零）
其余元素乘以 1 / (1 – p)进行放大
因为比例数据缩小，加大其他的值，保持值和没有放大的值和不会差异太大

"""

import torch

x = torch.Tensor([1,2,3,4,5,6,7,8,9])
dp_layer = torch.nn.Dropout(0.1)  # 10%概率置为0
dp_x = dp_layer(x)
print(dp_x)



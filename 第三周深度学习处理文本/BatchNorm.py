import torch
import numpy as np
# 约束，规范，拉回来
"""
规划层 Normalization 
数字过大或过小，影响梯度，权重过大，整个就会爆炸，规划层的作用是把分布不太好的值，降到正常值，防止梯度爆炸
缩小波动
横向batch normalization,多在图像里面用
纵向layer normalization 多在NLP用
需要学习的参数是 gama 白塔
不改变维度，只是改变值
"""
x = np.random.random((4, 5))

bn = torch.nn.BatchNorm1d(5) # 5维度，规划的维度

y = bn(torch.from_numpy(x).float())
y=bn(torch.FloatTensor(x))

print(x)
print("torch y:", y)
print(bn.state_dict())

#numpy实现
# 得到的数值 权重  和 b
gamma = bn.state_dict()["weight"].numpy()
beta = bn.state_dict()["bias"].numpy()

num_features = 5
eps = 1e-05
momentum = 0.1


# initialize the running mean and variance to zero
running_mean = np.zeros(num_features)
running_var = np.zeros(num_features)

mean = np.mean(x, axis=0)
var = np.var(x, axis=0)

# update the running mean and variance with momentum
running_mean = momentum * running_mean + (1 - momentum) * mean
running_var = momentum * running_var + (1 - momentum) * var

# normalize the input with the mean and variance
x_norm = (x - mean) / np.sqrt(var + eps)

# scale and shift the normalized input with gamma and beta
y = gamma * x_norm + beta
print("ours y:", y)
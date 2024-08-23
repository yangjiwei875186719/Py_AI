# -*- coding: utf-8 -*-
import torch
x = [1,2,3,4]
x_1 = [[1,2,3,4],[2,3,5,6]]
# torch实现的softmax
"""
第二个参数是 dim 参数，用于指定在哪个维度上进行 softmax 操作。
当 dim=0 时，表示在第一个维度上进行 softmax 操作。这通常用于对每一列进行 softmax 操作，即在输入张量的列维度上应用 softmax 函数
当 dim=1 时，表示在第一个维度上进行 softmax 操作。这通常用于对每一行进行 softmax 操作，即在输入张量的列维度上应用 softmax 函数
"""
print(torch.softmax(torch.Tensor(x),0))
print(torch.softmax(torch.Tensor(x),0))
print("竖向softmax:",torch.softmax(torch.Tensor(x_1),0))
print("横向softmax:",torch.softmax(torch.Tensor(x_1),1))

h = torch.rand(2,3,4)
h= torch.tril(h, diagonal=-1)
print("h",h)

print("torch.Tensor(x)", torch.Tensor(x))
print("torch.FloatTensor(x)", torch.FloatTensor(x))

"""
argmax 返回最大张量的索引位置
"""
arg_max = torch.argmax(torch.Tensor(x))
print("arg_max", arg_max)


"""
Tensor和NumPy相互转换
"""
a = torch.ones(5)
b = a.numpy()
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)
"""
NumPy数组转Tensor
"""
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)

"""
Tensor on GPU
"""
# 以下代码只有在PyTorch GPU版本上才会执行
if torch.cuda.is_available():
    device = torch.device("cuda")          # GPU
    y = torch.ones_like(x, device=device)  # 直接创建一个在GPU上的Tensor
    x = x.to(device)                       # 等价于 .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # to()还可以同时更改数据类型

"""
ones(*sizes)	全1Tensor
zeros(*sizes)	全0Tensor
"""

"""
view使用
view(-1)的作用是将张量重新reshape为一个维度未知的形状，其中-1表示PyTorch会根据张量的总元素数量和其他维度的大小来自动确定这个未知维度的大小
通常用于将一个多维张量重新整形为一个一维张量，或者在不知道具体维度大小的情况下，让PyTorch自动计算这个维度的大小
"""
x = torch.rand(5, 3)
print("随机张量：",x)  # 形状5*3
y = x.view(-1)
print("转换形状：",y)   # 形状1*15  顺序不变
y = x.view(-1,3) # 将多维向量转化成一位向量 ，例如一维向量15个值，后面一个3 就形成了一个5* 3 的向量
print("转换形状：",y)  # 形状 5 * 3  顺序不变
z = x.T              # 形状 5 * 3 竖变行，行变竖
print("转置：",z)

x = torch.rand(5, 3,2)
print("随机张量：",x)  # 5*3*2
y = x.view(-1)
print("转换形状：",y)  # 1*30
y = x.view(-1,3,5) # 将多维向量转化成一位向量 ，在用30/3/5 得到一个 2*3*5的张量
print("转换形状：",y) # 2*3*5

"""
在PyTorch中，对于高于2维的张量，直接使用.T进行转置操作已被弃用，并在未来的版本中会导致错误。替代方法是使用.permute()，.transpose()或者.mT等操作来实现相同的功能。
"""
x = torch.rand(5, 3, 2)
# z = x.T # 转置报错
# print("转置",z)
z = x.permute(0, 2, 1)  # 转置操作
print("z_permute",z)  # 5*2*3
# 或者使用 transpose
z = x.transpose(0, 2)
print("z_transpose",z)  # 2*3*5

y = torch.rand(2, 3)
z = x.view(-1,y.shape[-1])
print("y.shape[-1]",y.shape[-1])   # y.shape[-1]就是去y形状，最会一位
print("z",z)



attention_mask = torch.tril(10)
print("attention_mask:",attention_mask)
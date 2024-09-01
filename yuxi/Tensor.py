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



# attention_mask = torch.tril(10)
# print("attention_mask:",attention_mask)




"""
加载模型
在🤗 Transformers库中，return_dict是from_pretrained方法中的一个参数，用于控制是否返回一个字典对象而不是多个输出。当return_dict=True时，模型的输出以字典的形式返回，其中包含模型的所有输出。这个参数通常用于方便地访问模型输出的不同部分，而不必通过索引来获取。
当return_dict=False时，模型将返回一个包含多个输出元素的元组，通常按照模型的输出顺序排列。这种方式可能需要额外的索引操作来获取所需的输出。
使用return_dict=True时，可以通过字典键来访问不同的模型输出，这样可以更容易地理解和处理模型的输出结果。这对于在训练、评估或推理过程中需要同时处理多个输出的情况特别有帮助。
"""
from transformers import BertModel,BertTokenizer
from config import Config
bert = BertModel.from_pretrained(Config["bert_path"], return_dict=False)
# print("bert:", bert)


# 加载bert词表
"""
text:
类型: Union[TextInput, PreTokenizedInput, EncodedInput]
描述: 输入的文本，可以是原始文本字符串、预分词后的文本或者编码后的文本。
text_pair:
类型: 可选的 Union[TextInput, PreTokenizedInput, EncodedInput]
描述: 可选的第二个句子，用于处理句对任务（例如文本分类中的文本对）。
add_special_tokens:
类型: 布尔值
描述: 是否添加特殊标记（如[CLS]和[SEP]）到编码序列中。
padding:
类型: Union[bool, str, PaddingStrategy]
描述: 是否对输入进行填充，可以是布尔值或填充策略，用于确保输入序列达到指定的最大长度。
truncation:
类型: Union[bool, str, TruncationStrategy]
描述: 是否对输入进行截断，可以是布尔值或截断策略，用于处理超过最大长度的输入。
max_length:
类型: 可选的整数
描述: 指定处理后的文本序列的最大长度，通常用于截断和填充操作。
stride:
类型: 整数
描述: 在截断时，指定截断窗口的滑动距离。
return_tensors:
类型: 可选的字符串或张量类型
描述: 指定返回的张量类型，如'pt'表示返回PyTorch张量。
**kwargs:
类型: 关键字参数
描述: 其余未列出的关键字参数将被传递给内部调用的方法。


用到的参数
add_special_tokens=False:去掉前缀
max_length=Config["max_length"]:截取文本最大长度
truncation=True:参数告诉tokenizer.encode方法在处理输入文本时要进行截断操作
padding="max_length:padding='max_length'参数告诉tokenizer.encode方法在处理输入文本时要进行填充操作，并且填充到指定的最大长度。这样可以确保输入序列达到指定的长度，以符合BERT模型对输入序列长度的要求
"""

def load_vocab(vocab_path):
    return BertTokenizer.from_pretrained(vocab_path)
tokenizer = load_vocab(Config["bert_path"])  # 这个是bert路径，不是bert的字典路径
print("tokenizer:",tokenizer)

vocab_index = tokenizer.encode("你好呀！",padding="max_length",max_length=Config["max_length"],truncation=True,) ##  ，
print("vocab_index:",vocab_index)
print(f"vocab_index:{vocab_index}结束")
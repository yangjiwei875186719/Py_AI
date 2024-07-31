import numpy as np



"""
，两种输出都包含了相同的数据，只是表示方式稍有不同。你可以根据具体的需求选择使用其中一种表示形式。
如果你需要进行数值计算和操作，使用 NumPy 数组可能更方便；如果你需要处理一般的列表操作，那么使用嵌套列表可能更适合。
"""
print("-----------------两种列表嵌套方法-----------------")
def build_sample():
    x = np.random.random(5)   # 随机一个5位向量
    print('###############')
    print(x)
    print('###############')
    if x[0] > x[4]:
        return x, 1
    else:
        return x, 0

def build_dataset(total_sample_num): # 传的是几行 5列的数据
    X = []
    Y = []
    for i in range(total_sample_num):
        print(i)  # range输出结果是 0,1,3,4,5...
        x, y = build_sample() # 一个样本一个样本执行
        X.append(x)   # 1 * 5  因为默认 x = 5
        Y.append([y]) # 1 * 1
    print("X",X)
    print("Y",Y)
    # 输出结果是一个包含两个 NumPy 数组的 Python 列表。每个 NumPy 数组都是一个一维数组，其中包含了一组浮点数
    # X [array([0.44541908, 0.35554432, 0.32272479, 0.12980671, 0.62663108]), array([0.1380557 , 0.02755651, 0.77031392, 0.26681412, 0.00235754]), array([0.45561876, 0.62357479, 0.77525069, 0.91026681, 0.42842542])]
    # Y [[0], [1], [1]]

print(build_dataset(3))
x = [0.31823829, 0.9867538 , 0.66611314, 0.75575286, 0.90822339]
x1 = [0.0662027 , 0.15525149, 0.99548496, 0.0983143 , 0.95267728]
x2 = []
x2.append(x)
# 是一个嵌套列表（list of lists）。每个内部列表都包含一组浮点数
print(x2)   # [[0.31823829, 0.9867538, 0.66611314, 0.75575286, 0.90822339]]


"""
zip() 用法
zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。
我们可以使用 list() 转换来输出列表。
如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
"""
print("-----------------生成深度学习可以使识别的张量-----------------")
a = [1,2,3]
b = [4,5,6]
c = [4,5,6,7,8]
zipped = zip(a,b)
for i ,j in zip(a,b):
    print('i',i)
    print('j',j)
print(zipped)            # 返回一个对象 <zip object at 0x000001C5E10F9A80>
print(list(zipped))  # list() 转换为列表 列表的形式，(tuple to list)
#
print(list(zip(a,c)))              # 元素个数与最短的列表一致 # [(1, 4), (2, 5), (3, 6)]

a1, a2 = zip(*zip(a,b))          # 与 zip 相反，zip(*) 可理解为解压，返回二维矩阵式
print(list(a1))  # [1, 2, 3]
print(list(a2))  # [4, 5, 6]


"""

"""
print("-----------------生成深度学习可以使识别的张量-----------------")
import torch
# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    x = np.random.random(5)   # 随机一个5位向量
    print('###############')
    print(x)
    print('###############')
    if x[0] > x[4]:
        return x, 1  # 返回元组([],[])
    else:
        return x, 0


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num): # 传的是几行 5列的数据
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample() # 一个样本一个样本执行
        X.append(x)   # 1 * 5  因为默认 x = 5
        Y.append([y]) # 1 * 1
    print("X",X)
    print("Y",Y)
    # 当函数参数为整形时，表示生成矩阵的维度，此时参数可以为多个变量
    return torch.FloatTensor(X), torch.IntTensor(Y) ## 返回元组([],[])  整数张量 (torch.IntTensor)、长整数张量 (torch.LongTensor)、布尔张量 (torch.BoolTensor) 、浮点张量(torch/FloatTensor)
xx = build_dataset(3)
print("xx",xx)  # 返回元组([],[])



"""
enumerate 用法
enumerate函数是Python的一个内置函数，用于将一个可迭代对象转换成一个枚举对象。这个枚举对象包含了对可迭代对象中的元素进行标号的功能。
 函数的基本语法是 enumerate(iterable, start=0)，其中 iterable 是必填参数，表示要枚举的可迭代对象；start 是可选参数，表示枚举的起始值，默认为0。
 返回的枚举对象是一个迭代器，可以通过for循环遍历，也可以使用 list() 函数将其转换为列表。
 
参数解释:
iterable：任何可迭代对象，如列表、元组、字符串等。
start：可选参数，指定索引的起始值，默认为0。

返回值：返回的枚举对象中的每个元素都是一个包含两个元素的元组，第一个元素是元素的索引，第二个元素是元素的值。
使用场景：在处理列表或其他可迭代对象时，经常需要同时访问元素的值和索引。例如，在遍历列表的同时打印出索引和对应的元素值，这时就可以使用 enumerate 函数

"""
print("-----------------enumerate 用法-----------------")
import random
def build_vocab(): # vocab词汇
    chars = "你我他好坏上中下asdfghjk" # 字符集
    vocab = {}     # 定义字典类型
    for index, char in enumerate(chars,start=0):  # index 索引，char字符  start是索引从哪个开始
        vocab[char] = index + 1  # char变成了键   index + 1 变成了值
    vocab['unk'] = len(vocab) + 1
    return vocab
vocab = build_vocab()
print(vocab)


"""
字典遍历，以及get用法
"""
print("-----------------字典遍历，以及get用法-----------------")
# 遍历字典
for index,value in vocab.items():  # set(vocab.items())将字典中的值转化为集合，避免重复值
    print("index",index,"value",value) #
    yy = vocab.get("hh","78") # 根据健，获取字典中对应的值，如果没有找到对应的健，就会返回第二值
    # print("yy",yy)

chars = "下asdfghjk"
for i in enumerate(chars):
    print('i',i)  # 元组结果(0, '你')

"""
random.choice() 用法 index
choice函数的语法很简单，只需要传入一个序列（如列表、元组等）作为参数，即可从该序列中随机选择一个元素

"""
print("-----------------random.choice() 用法-----------------")
# 位置
def find_indices(lst, value):  # 判断某个字符在文本里面的索引位置
    return [index for index, item in enumerate(lst) if item == value]

def build_sample(vocab, sentence_length): # sample样本
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)] # vocab.keys()获取 将样本转换成字典里面对应的value,random.choice(list(vocab.keys()))将键值对中的键转化成列表， random.choice()是Python中random模块的函数，用于从序列中随机选择一个元素
    #print(x) # 一个文本
    if x.count("你") == 1:
        indices = find_indices(x, "你")
        y1 = x.index("你")
        # print("y1",y1)
        y = [idx for idx in indices]
    else:
        return build_sample(vocab, sentence_length) # 不满足条件，重新调用样本
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将输入序列化,获取索引的位置
    return x, y # 元组
xx = build_sample(vocab,3)
print("xx",xx) #



"""

张量返回 分别用 x, y接收
"""
print("-----------------return x y 张量接收方式-----------------")
def build_dataset(vocab, sample_length, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(np.array(dataset_y).flatten())
vocab = build_vocab()
sentence_length = len(vocab) - 1  # 几列(有多少个字符)
sample_num = 3  # 几行（几个样本）
dataset_x, dataset_y = build_dataset(vocab,sample_num,sentence_length) # sample_num几行，  sentence_length 几列   sample_num（几行文本)，sentence_length几列(文本长度)sentence句子长度
print("dataset_x:",dataset_x)
print("dataset_y:",dataset_y)  # 一维  sample_num列
vocab = build_vocab()
x1,y1 = build_dataset(vocab,3,5)
print("x1:",x1)
print("y1:",y1)


"""

defaultdict
range(n) 生成了一个从 0 到 n-1 的数字序列。
对于范围内的每个数字 x，x + 1 被用作字典中的键。
每个键对应的值是一个新的 defaultdict 对象，其值被初始化为整数。这意味着如果你访问字典中不存在的键，它会自动创建该键，并将其默认值设为 0。
"""
print("----------------- defaultdict-----------------")

from collections import defaultdict

n = 5  # 或者其他任何n的值
ngram_count_dict = dict((x + 1, defaultdict(int)) for x in range(n))

print(ngram_count_dict)
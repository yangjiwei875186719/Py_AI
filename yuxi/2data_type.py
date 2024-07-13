# 字符串
my_str = 'hello'
# 获取字符串中的指定元素[0]:代表索引或者下标
print(my_str[4])

# 获取最后一个元素
print(my_str[4])
print(my_str[-1])
# 获取字符串的长度
print(len(my_str))
print([len(my_str)-1])
# 字符串切片
my_str = 'hello'
print(my_str[1:5]) # 左闭右开区间
# 从某个位置接截取到最后
print(my_str[1:])
print(my_str[:3])
# 字符串有哪些常见的内置函数
# find:检测str是否包含在my_str字符串中，如果是返回开始的索引值，否则返回-1
my_str = 'hello world'
my_str1 = 'hello world'
my_str1.find('hello')
my_str1.find('ho')
# 将字母全部转换成大写
print(my_str.upper())
# 转换成小写
print(my_str.lower())
# 字符串的起始位置是否包含指定字符串
my_str = 'http://wwww.BAIDU.COM'
print(my_str.startswith('http'))
print(my_str.endswith('com'))

# 统计字符串出现的次数
print(my_str.count('s'))
# 判断字符串是不是全是数字
print('123'.isdigit())
# 判断字符串是不是全是字母
print('abc1'.isalpha())
# 检测字符串的时候包含数字和字母
print('abs111'.isalnum())
# 字符串的遍历
my_str = 'hello'
# 字符串遍历
for value in my_str:
    print(value)
# enumerate 将一个可遍历的数据对象(如列表，字符串，元组)组合为一个索引序列额，同时返回数据和数据的下标
for index, value in enumerate(my_str):
    print(index,value)



# 列表
# 定义:是一种有序的集合，可以进行添加或者删除，定义的时候写在方括号之内的，元素之间使用逗号分割
# 列表中的数据类型未必是相同的

# 下标是从0开始的
my_str = [1,1.2,'abc',True]
print(my_str)
# 根据索引获取值
print(my_str[3])
print(len(my_str))
# 返回最后一个元素
print(my_str[-1])
print(type(my_str))
# 切片
# 注意：使用切片前后数据类型不改变
print(my_str[0:2]) # 左闭右开
print(type(my_str[0:2]))

# 列表的正删改查
my_list=list()
my_list=[1,2]
print(my_list)
# 元素的添加(追加)
my_list.append(3)
print(my_list)
my_list.append(4)
print(my_list)
# 插入（指定位置插入元素）
my_list.insert(0,"a")
print(my_list)

# 列表合并
a = [11,2]
b = [3,"a"]
a.extend(b)
print(a)
# 修改
a[0] = 'a'
print(a)
# 删除指定位置的数据
a.remove(2)  # 2指定数据
print(a)
# 删除数据制定位置
del a[0]  # 0 代表下标
print(a)
# 清空列表
print(a)
a.clear()
print(a)
# 列表遍历
a = [1,2,3,4,5]
for value in a:
    print(value)
for index,value in enumerate(a ):
    print(index,value)
# 判断某个元素是否属于该里诶博爱】
# in 如果存在哪些结果为true 否则为false
# not in 如果不存在那么结果为true,否则为false
print( 6 in a)  #判断6是否在里面

# 列表嵌套列表
my_list = [1,2,3]
my_list1 = [1,2]
my_list.extend(my_list1)
print(my_list)
my_list.append(my_list1)
print(my_list)

# 元组
# 元组是不能修改的 元组使用的是小括号，其他的和列表相同
my_tuple = (1,2,'abc',True)
print(my_tuple)
print(type(my_tuple))
# 获取 下标是从0 开始的
print(my_tuple[2])  # 2是索引
# 修改
# my_tuple[2] = 'c'   # 不支持修改
print(my_tuple)
# del my_tuple[2] 不支持删除操作
print(my_tuple)
my_tuple = (1,2 ,[1,2])
print(my_tuple[2])  # 返回是一个列表
print(type(my_tuple[2]))
print(my_tuple[2][1])
# 修改tuple中列表的值
my_tuple[2][0] = 5
print(my_tuple[2][0])
print(my_tuple)
# 遍历
tuple = ('a','b','c')
for value in tuple:
    print(value)
for index,value in enumerate(tuple):
    print(index,value)
# 注意 如果只有一个元素，需要在末尾加上逗号
tuple = (1)
print(tuple)
print(type(tuple))  # 返回int
tuple = (1,)
print(tuple)
print(type(tuple))  # 返回数组
# 两个数字相加
# num1 = input('请输入第一个数字：')
# num2 = input('请输入第一个数字：')
# sum = float(num1) + float(num2)

# 生成随机数
# 随机模块
import random
print(random.randint(0,9))
# 生成一个0-1之间的浮点随机数
print(random.random())
# 九九乘法表
for i in range(1,10):  #   random.randint(1,10) 左闭右开
    for j in range(1,i+1):
        print('{}*{}={}\t'.format(i,j,i*j),end='')
    print()
# 字典 -->键值对 key:value
# {'key1':'value','key2':'value2'..}
my_dict={'name':'zs','age':'23','height':'175'}
# 获取新字典中的value
print(my_dict['name'])
print(my_dict.get('name'))
# 定义空字典
my_dict = {}
print(my_dict)
# 字典添加元素
my_dict['name'] = 'ls'
my_dict['age'] = '18'
print(my_dict)
# 修改
my_dict['age'] = '19'
print(my_dict)
# 删除
my_dict.pop('age')
# del my_dict['age']
print(my_dict)

# 清除
my_dict.clear()
print(my_dict)
# 判断key是否在字典中
print('age' in my_dict)
# 获取字典中所有的值
print(my_dict.values())
# 获取字典中所有的key
print(my_dict.keys())
my_dict={'name':'zs','age':'23','height':'175'}

for key in my_dict:
    print(key)
    print(my_dict[key])

for key, value in my_dict.items():
    print(key,value)

# 集合 set 特点：无序的不重合的元素系列
my_set = {11,4,'abc','hello'}
print(type(my_set))
# 定义空集合
my_set= set()
print(type(my_set))

for index ,value in enumerate(my_set):
    print(index,value)

# 集合去重
my_list = [1,3,4,5,3]
print(type(my_list))
# 将列表转换成集合，达到数据去重的目的
my_set = set(my_list)
print(my_set)
print(type(my_set))

# 将数据转换为列表
my_list = list(my_set)
print(my_list)

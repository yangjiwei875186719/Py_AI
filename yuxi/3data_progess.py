# 函数的定义 将独立功能的代码块组织为一个小模块，这就是函数print(a) python内置函数
# 语法
# def 函数名(参数[可选])：
# 功能代码
# pass 表示该函数什么也不做

# 函数的定义
def show():
    pass
    print('hello word')
# 函数的调用
show()

# 函数的文档说明
def show2(num1):
    '''

    :param num1:
    :return:
    '''
    print('hello word')
# 调用当前函数的文档说明
help(show2)
# 函数的几种类型
# 无参数，没有返回值
def show3():
    print('hello badou')
# 有参数，没有返回值
# name,age 形式采纳数
def show(name,age):
    print(name,age)
# 'likui','28' 实际参数
show('likui','28')

# 无参数，有返回值
# rreturn 区分函数真正是否具有返回值，通过return修饰
def show3():
    str1='hello word'
    return str1
print(show3())

# 有参数，有返回值
def show4(name,age):
    strr = 'hello %s word %d' % (name,age)
    return strr
print(show4('玩儿',23))
# 函数注意点
# 当两个函数名称相同时，下面的函数会覆盖上面函数的结果
# 函数名不能相同，也不要和变量相同
def show5():
    print("aa")

def show5():
    print("bbb")
show5()
print(show5()) #  函数的地址
show5 = 10
# show5()

# 缺省参数
# 针对两个数字进行求和
# 必须两个都要设置参数
def sum_num(num1=1,num2=2):
    result = num1 + num2
    return result
re =sum_num(4)
print(re)

# 函数的不定长（我我也不知道你传多少参数）参数
# 不定长参数包括两个：不定长必修参数，不定长关键字参数
# 不定长必修参数（*args）
def sum_num(*args):
    print(args,type(args))
    result = 0
    for value in args: # 遍历元组
        result +=value
    return result
re = sum_num(1,2,3,4)
print(re)

# 不定长关键字参数（**kargs）
def sum_num(**kargs):
    print(kargs,type(kargs))  # 返回字典
    for key,value in kargs.items():  # 遍历字典
        print(key,value)
    show(**kargs)
# sum_num(1,2) # *krags
# sum_num(1,2) # **krags
# 该show函数如何在另一个函数中进行调用：
def show(**kargs):
    print(kargs)
show()

# 递归函数： 如果一个函数在函数内部调用本身， 这个函数叫做递归函数
# 注意：递归函数容易造成死循环，优点：递归使代码看起来很整洁，方便讲复杂的函数拆分为简单的
# 原因： 缺少结束递归条件
# def show6():
#     print(1)
#     show6()
#     print('end')
# # show6()


# 计算5的阶乘
def num_car(num):
    # 先计算1 的阶乘
    if num == 1:
        return 1
    else:
        return num * num_car(num - 1) # num_car
result = num_car(2)
print(result)
# 注意：在函数中遇见return 代表函数有返回值，如果在一个函数中有多个return,其余后面不会执行
# 局部变量：函数定义在作用域内的变量
# 和全局的区别是：作用域不同
def show():
# 局部变量
    num = 100
    print(num)
show()
# print(num)
print('------')
# def func():
#     print(num)
# 全局变量 在函数外面定义的变量
# 全局变量命名时：尽量使用大写NUM或者g_num
num = 100
def show():
# 修改全局变量 通过global关键字
    global num
    num = 10
    print(num)
show()  # 10
print(num)
# 针对多个值进行直接赋值
a,b,c = 1,2,3
print(a,b,c)

# 面向对象
# 匿名函数 没有名字，用lambda关键字定义的
# 优点：对代码进行简化，增加运行效率，特点：只适用于简单的操作，返回值不加上return
# 普通函数
def fync(a,b,c):
    return a+b+c
print(fync(1,2,3))
# 匿名函数的创建和调用
# a,b,c实际参数
result = (lambda a,b,c:a+b+c)(1,2,3)
print(result)

# 面向对象 是一种程序设计的思想 将对象作为程序的基本单元
# 面向过程： 就是一系列的函数，将这些功能组合在一起
# 注意：在python中，所有的数据类型都可以视为对象
# 面向对象汇总的类和对象
# 类：一类事物 是一个概念
# 对象 在一类事物中具体到某一个东西 水杉，向日葵  一个类中有多个对象
# 类是对象的概括，对象是类的具体体现
# 对象可以有属性和方法（功能） dog:属性大小，品种，颜色，毛发，耳朵   方法：吃，跑，卖萌等等
# 类和对象的关系：一个类中有多个对象，一对多关系

# 面向对象三个特征： 封装，继承 多态
# 类的定义
class Teacher(object):
    # d定义类的属性
    country = '中国'
    # 方法
    def show(self):
        print("老师")

# 根据类来创建对象
teacher = Teacher()
# 获取对象的属性
print(teacher.country)
teacher.show()
# 查看Teacher类继承的父类
print(Teacher.__base__)

teacher2 = Teacher()
print(teacher2.country)
# 对象中的属性和方法
class Teacher(object):
    def show(self):
        print("老师")
# 通过对象进行动态的属性添加
teacher = Teacher()
teacher.name = 'zs'
teacher.age = 20
# 获取属性
print(teacher.name,teacher.age)
# 修改属性 前提：该对象必须首先要有这个属性
teacher.name ='lisi'
print(teacher.name)

# self?
class Cat(object):
    def eat(self):
        pass
        print('%s在吃' %self.name)
# 创建对象
cl = Cat()
cl.name ='c1'
print(cl)
c2 = Cat()
c2.name = 'c2'
c2.eat()

# __init__ 初始化
class Teacher(object):
    # 魔法方法
    def __init__(self,name,age):
        self.name = name
        self.age =age
    def show(self):
        print('老师')
t1 = Teacher('zs',19)
print(t1.name,t1.age)
# 继承定义：描述多个类之间之间的关系
# 注意：
# 1.如果一个类A里面的属性和方法可以服用，则可以通过继承的方式，传递到类B里面
# 2.类A 就是基类，即父类，类B 就是派生类，即子类
# 3.子类可以继承父类的属性和方法
# 在继承中有单继承和多继承
# 单继承：子类可以直接使用父类的属性和方法
# 优点：子类可以直接使用父类的属性和方法
class Person(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def show(self):
        print(self.name, self.age)
# 子类
class Student(Person):
    pass

stu = Student('zs', 23)
print(stu.name, stu.age)

# 多继承
class A(object):
    def show(self):
        print('A类')
class B(object):
    def show(self):
        print('B类')
class C(A,B):
    pass

c = C()
c.show()
print(c) # 输出A类
print(C.mro())   # 顺序执行找 找到一个就返回

# 重写定义：子类可以继承父类，父类中的方法满足不了子类的需要可以对父类的方法进行重写
# 重写特点：1：继承 2：方法名相同
class Person(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def show(self):
        print(self.name, self.age)
# 子类
class Student(Person):
    def __init__(self,name,age,sex):
        self.name = name
        self.age = age
        self.sex = sex
    def show(self):
        print('我的名字%s，我的年龄%d,我的性别是：%s' %(self.name ,self.age,self.sex))
    pass
stu = Student('zs',30,'男')
print(stu.show())

# 私有方法和私有属性
class Student(object):
    def __init__(self, name, age,height):
        # 公共属性
        self.name = name
        self.age = age
        # 私有属性 是以__开始的属性，我只能在本类的内部进行使用，在类外面不能使用
        self.__height = height
    def show(self):
        print('我的名字%s，我的年龄%d,我的身高是：%d' % (self.name, self.age, self.__height))
stu = Student('zs',30,175)
print(stu.show())
stu.__height ='195'
print(stu.show())    # 不能修改私有属性


# 多态：多态在python中表现不是很明显
# 多态：不同的子类对象调用相同的父类方法，产生不同的执行结果。优点：增加代码外部的灵活性
# 多态形成的前提，1、继承 2、重写父类的方法
# 定义Animal
class Animal(object):
    def run(self):
        print("Animal rnu")
class Dog(Animal):
    def run(self):
        print('dog run')
class Cat(Animal):
    def run(self):
        print('Cat run')
def run_two(animal):
    animal.run()
    animal.run()
dog = Dog()
cat = Cat()
run_two(dog)
run_two(cat)


# 异常： 程序运行出错，就会向上抛出，直到某个函数可以处理该异常 ，每种语言都有try cache 错误处理机制
try:
    print(num11)
except NameError as e:
    print("有异常出现的时候会执行里面的代码")
    print(e)
print('有异常出现的时候会执行日志')

# 异常使用类是Exception类 通用的一个模块
try:
    print(num11)
except Exception as e:
    print("有异常出现的时候会执行里面的代码")
    print(e)
print('有异常出现的时候会执行日志')

# while True:
#     my_str = input("请输入数字：")
#     try:
#         num = int(my_str)+ 1
#     except Exception as e :
#         print('有异常会执行')
#         print("重新输入数字")
#
#     else:
#         print("没有异常会执行else")
#     finally:
#         print("有没有异常都会执行里面的代码")


# 模块： 通俗的来讲就是.py文件，就是一个模块，作用：管理功能代码，在代码里面可以定义变量 函数  类
import time

time.sleep(3)
help('modules')  # 内置模块
print('123')


# 自定义模块 import 导入别人活着自己封装好的功能py文件
# g_num = 100
# def show ():
#     print("函数")
# class Student(object):
#     def __init__(self,name,age):
#         self.name = name
#         self.age =age
#     def show_msg(self):
#         print(self.name,self.age)
# if __name__ == '__main__':
#     show()


import first_modual
print(first_modual.g_num)
first_modual.show()
stu = first_modual.Student('zs', 20)
stu.show_msg()

# 模块导入方式
# for 模块名 import 功能代码 推荐使用
from first_modual import show
# show()
# Studdent('zsss',20)
# from 模块名 import * 导入模块下面所有的功能代码 不推荐使用
# from first_modual import *
# 给模块起别名
from first_modual import show as show_msg
show_msg()

def show():
    print('hekllo word')
show()


# 文件读写
# 1.打开文件
# 2.对文件内容进行操作（读，写）
# 3.关闭文件
# 打开方式
# r:只读 ，文文件不存在程序会崩溃，出现文件不存在的异常
# w:只写，会将原来的内容进行覆盖的掉，如果文件不存在，会创建一个文件
# a:追加写入
# b：表示二进制的形式，常用rb:以二进制方式打开一个文件用于只读。wb:以二进制方式打开一个文件用于只写

# w 模式
# 打开文件
f = open('badou.txt', 'w', encoding='utf-8')
# print(f.encoding)
# 操作文件
# 注意：文件被打开之后，多次邪恶如数据不会覆盖之前的数据
f.write('八斗')
f.write('大数据')  # 追加形式
# 关闭文件
f.close()


# r 模式
f = open('badou.txt', 'r', encoding='utf-8')
# d 读取数据
context = f.read()
print(context)
# 关闭
f.close()

# a 模式
f = open('badou.txt', 'a', encoding='utf-8')
f.write("ABC")
# 关闭
f.close()

# rb 模式二进制方式读取
# 注意点 带有B的模式都是二进制模式，在这种模式下， 不管什么系统都不要添加encoding参数
f = open('badou.txt', 'rb')
context = f.read()
print(context, type(context))

# 对二进制的数据进行utf-8解码操作，将bytes类型额数据转换成str类型
result = context.decode('utf-8')
print(result)
# 将str类型转换额为bytes类型，编码
print(result.encode('utf-8'))
# 关闭
f.close()

# 1.定义一个列表，并按照降序排序
my_list = [1,2,3,7,5,9]
# reverse=True :表示降序，默认不写的话表示升序
print(my_list.sort(reverse=True))
# 2 判断是否为偶数（分别用普通函数和匿名函数实现）
def is_os(num):
    if num % 2 == 0:
        return True
    else:
        return False
print(is_os(8))

# 匿名函数
f =lambda num:True if num % 2 ==0 else False
re =f(2)
print(re)
# 3.如何使用匿名函数对字典中的列表进行排序
my_list = [{'name':'lis','age':19},{'name':'lisi','age':31}]
my_list.sort(key=lambda item:item['age'],reverse=True)
print(my_list)

def get_value(item):
    return item['age']

my_list.sort(key=get_value,reverse=True)
print(my_list)

# 4.利用Python进行文件拷贝
old_file = open('badou.txt', 'rb')
new_file = open('target.txt', 'wb')

# 文件操作
while True:
# 1024 :读取1024字节的数据
    file_data = old_file.read(1024)
# 判断数据是否读取完成
    if len(file_data) == 0:
        break
    new_file.write(file_data)
# 关闭文件
old_file.close()
new_file.close()

# 6.定义类class为BOOK，定义init函数和自定义函数举例如：you() ,info()
# 1) __init__中，需要默认书籍名称，价格，作者例如： name='爵迹' price='39',auth='郭敬明'
# 2) 定义实例方法you(),使用输出以下字样%s为书籍名称，“努力学习%s图书”
# 3）定义实例方法info（）：打印书的详细信息 “书籍名称：%s”,价格：%s,作者：%S“
# 4）定义一个子类，继承Book类，类名BookZi,BookzI中不定义任何函数，pass
# 5)定义父类创建对象，并调用you(),___init__()方法
# 6）定义子类创建的对象并调用info方法
class Book():
    def __init__(self,name='爵迹',price='39',author='郭敬明'):
        self.name = name
        self.price = price
        self.author = author
    def you(self):
        print('努力学习%s图书'%( self.name ))
    def info(self):
        print('书籍名称：%s”,价格：%s,作者：%s' %(self.name,self.price,self.author))
class BookZi(Book):
    pass
book = Book()
book.you()
book.__init__()

bookzi = BookZi()
bookzi.info()
# 7.使用正则表达式匹配全部字符串进行输出
# 源数据： abc 123 def
# 结果： abc 123 def
import re
str2 = 'abc 123 def'
print(re.match('^abc\s\d\d\d\sdef$',str2).group())
print(re.match('^abc\s\d{3}\sdef$',str2).group())
print(re.match('^abc\s(.*)\sdef$',str2).group(0))  # 默认是0 匹配出所有

# 8.使用正则表达式中sub实现获取我们匹配的字符串，然后追加执行字符
# 源数据 hello 7709 badou
# 结果 hello 7709 789 badou
import re
context = 'hello 7709 badou'
context = re.sub('(\d+)', r'\1 789',context)
print(context)

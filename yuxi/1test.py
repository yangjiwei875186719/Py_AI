# 单行注释
# 注释 标准代码功能能说明之类的
print("hello wprd")
# 多行注释
print("""
多行注释内容
可以翰皇进行输入2
""")
print('''
多行注释内容
可以翰皇进行输入2
''')
# 文本注释
print('这里是文本注释')
# 变量定义
# 定义一个变量，存储的数据
name = 'YJW'
print(name)
age = 22
print(age)
# 变量类型
# 可以通过type 查询数据的类型
print(type(name))  # str字符串类型
print(type(age))  # int 整数类型
# 标识符 只能有字母，下划线和数字组成额，并且数字不呢我那个作为开头，注意在标识符中是区分大小写name和Name是不一样的
# 2age= 11
# print(2age)
# 关键字 注意：在命名变量的时候不能和关键字相同
import keyword
keyword.kwlist

#输出
# 输出变量
print("python程序中输出")
# 输入
password = input("请输入密码：")
print(password)
# 格式化输出 %s是对字符串进行格式化输出 %d:对整型进行输出
password = input("请输入密码：")
print("你的密码是： %s" % password)
# 查询学生的个人信息
t_id=input("学号：")
name=input("姓名：")
print("该学生的学号:%s,姓名%s" % (t_id, name))

number = 10010
print("数字： %d" % number)

# 单个变量赋值
name = '刘备'
print(name)
id,name,age=1001,"刘贝尔",45
print('id:%d,姓名:%s,年龄：%d' % (id, name, age))
# 判断
# if else
# 用1代表有车票 0代表没有车票
chePiao= 1
if chePiao == 1:
    print("有车票，可以回家过年")
else :
    print("没有车票，不可以回家过年")

# elif
score = 70
if score>=90 and score<=100:
    print("本次考试，等级为A")
elif score >=80 and score <90:
    print('本次考试，等级为B')
elif score >=70 and score <80:
    print('本次考试，等级为C')
elif score >=60 and score <70:
    print('本次考试，等级为D')
elif score >=0 and score <600:
    print('本次考试，等级为E')

# while循环 字符串进行遍历
# 在 while 中 如果不设置终止条件，会一直执行
i=0
while i<5:
    print("当前是抄写第%d遍笔记" %(i+1))
    print("i=%d" % i)
    i += 1

# for循环
# 需求: 将1234 进行求和
sum = 0
for i in [1,2,3,4]:
    sum +=i
    print(sum)
print(sum)

# python3 中提供了range(),返回的是一个可迭代对象（类型是对象） 不是列表类型
# range 一般集合for 进行使用
for i in range(5):
    print(i)
# break
# 作用:提前退出循环语句
# [1,2,3,4,5] list
for i in [1, 2, 3, 4, 5]:
    print('------')
    if i == 3:
        break
    print(i)

i = 0
while i < 10:
    i+=1
    if i % 2 ==0:
        break
    print(i)
# continue
# 作用：介素本次循环，继续下一次循环
for i in range(10):
    if i == 2:
        continue
    print(i)





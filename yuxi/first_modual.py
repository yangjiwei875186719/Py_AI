g_num = 100
def show ():
    print("函数")
class Student(object):
    def __init__(self,name,age):
        self.name = name
        self.age =age
    def show_msg(self):
        print(self.name,self.age)
if __name__ == '__main__':
    show()

index_to_label = {0: '家居', 1: '房产', 2: '股票', 3: '社会', 4: '文化',
                               5: '国际', 6: '教育', 7: '军事', 8: '彩票', 9: '旅游',
                               10: '体育', 11: '科技', 12: '汽车', 13: '健康',
                               14: '娱乐', 15: '财经', 16: '时尚', 17: '游戏'}
label_to_index = dict((y, x) for x, y in index_to_label.items())
print(label_to_index)

from config import Config
from loader import load_data
print(Config)  # 字典
# 增加一个配置采参数
Config['test'] ='test'
# 取字典
a = Config["test"]
print('-'*5,"a",'-'*5)
print(a)



"""
1.使用json字符串生成python对象（load）
2.由python对象格式化成json字符串*（dump）
"""
import json
#1、 从python对象格式化一个json string
person = {"name":"sniper","age":30,"tel":["12312312","123123123"],"isniper":True}
print(person)   # 双引号变单引号，python中的字典是单引号,大写的True
print(type(person))
jsonStr = json.dumps(person)   # 格式化代码
print("jsonStr类型：",type(jsonStr))
print("jsonStr:",jsonStr)


jsonStr = json.dumps(person,indent=4,sort_keys=True)  # 默认按照key排序
print("jsonStr格式化：",jsonStr)
#dumps  将python数据类型转换并保存到jjson格式文件内
json.dump(person,open('date.json','w'))
json.dump(person,open('date1.json','w'),indent=4)  # 格式化保存
json.dump(person,open('date2.json','w'),indent=4,sort_keys=True)  # 格式化保存


pythonOjb = json.loads(jsonStr)
age = pythonOjb["age"]
print("pythonOjb加载loads：",pythonOjb)
print(age)

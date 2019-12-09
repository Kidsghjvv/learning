---
title: Python学习笔记
date: 2019-11-20
tags: 
- python
- note
categories:
- python学习
mathjax: true
---

# Python学习笔记

## 基本语法

### 打印输出

```python
print("hello world")
print("-"*10)
print("hello","world")
#打印变量值
print("width : %s, height : %s channels : %s" % (width, height, channels))
#打印矩阵
print(image)
```

### 代码注释

代码缩进：python对代码缩进的要求非常严格

### 多行语句的分割

使用`\`将代码分割成多行

```python
print("Hello,World.Hello,World.\
Hello,World.Hello,World.")
```

## 变量

### 赋值方式

```python
string1 = string2 = string3 = "Hello,World"
string1, string2, string3 = "Hello", "World","Hello,World"
```

### 数据类型及其索引

#### 数字

```python
float_num = 10.000

print(float_num)
print("%f" % float_num)
print("%.2f" % float_num)
print("%.4f" % float_num)
#ouput
10.0
10.000000
10.00
10.0000
```

#### 字符串

```python
string = 'Hello,World' #双引号也可以
string1 = string[0:11]
string2 = string[0:5]
string6 = string[:5] #注意第五个字符不显示
string3 = string[-1]
string4 = string[-5:-1]
#ouput
Hello,World
Hello
Hello
d
worl
```

#### 列表

列表是一种容器型数据类型，可以实现多种数据类型的嵌套，元素可重新赋值

```python
list1 = [ "Hello,World", 100 , 10.00 ]
list2 = [123, 'Hi']

print(list1) # 输出整个list1 列表元素
print(list1[0]) # 输出列表的第1 个元素
print(list1[1:]) # 输出从第1 个索引开始至列表末尾的所有元素
print(list1[-1]) # 输出列表的最后一个元素
print(list1 + list2) # 输出列表的组合
list1[0] = "0"
print(list1)
#output
['Hello,World', 100, 10.0]
Hello,World
[100, 10.0]
10.0
['Hello,World', 100, 10.0, 123, 'Hi']
['0', 100, 10.0]
```

#### 元组

另一种容器型数据类型，基本性质、索引值操作与列表相同，但其元素不能重新赋值

```python
tuple1 = ( "Hello,World", 100 , 10.00 )
tuple2 = (123, 'Hi')
print(list1) # 输出整个tuple1 列表元素
```

#### 字典

列表与元组为有序的元素组合，字典通过键值来操控元素

```python
dict_info = {"name": "Tang", "num":7272, "city": "GL"}
dict_info["city"] = "changsha"

print (dict_info) # 输出整个dict_info 字典
print(dict_info["city"])
print (dict_info.keys()) # 输出dict_info 的所有键值
print (dict_info.values()) # 输出dict_info 的所有值

#output
{'name': 'Tang', 'num': 7272, 'city': 'changsha'}
changsha
dict_keys(['name', 'num', 'city'])
dict_values(['Tang', 7272, 'changsha'])
```

#### tuple

构造一个元组

## 面向对象的方法-类

类是用来描述具有相同属性和方法的对象的集合，定义了该集合中每个对象的共有属性和方法，对象是类的实例

### 类的创建、继承与重写

```python
class People:
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def dis_name(self):
        print("name is:",self.name)

    def set_age(self, age):
        self.age = age
        
    def dis_age(self):
        print("age is:",self.age)
        
class Student(People): #继承父类
    def __init__(self, name, age, school_name):
        self.name = name
        self.age = age
        self.school_name = school_name

    def dis_student(self):
        print("school name is:",self.school_name)
    
    def dis_name(self): #子类中对父类进行重写
        print("名字：",self.name)
        
student = Student("Wu", "20", "GLDZ") #创建一个Student 对象
student.dis_student() #调用子类的方法
student.dis_name() #调用子类的方法，已重写
student.dis_age() #调用父类的方法
student.set_age(22) #调用父类的方法
student.dis_age() #调用父类的方法
```

`next()`返回迭代器的下一个值

`iter()`生成迭代器

## Matplotlib

### 画函数图像

```python
import matplotlib.pyplot as plt
import numpy as nps
%matplotlib inline
x = np.arange(-10,10,0.01)
# y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
y = np.where(x<0,0,x)
# y = np.sin(x)
plt.plot(x, y)
plt.title("ReLU",fontsize = 10)
# plt.xlabel("horizontal axis", fontsize = 10)
# plt.ylabel("vertical axis",fontsize = 10)
plt.tick_params(axis="both", labelsize = 10)
ax = plt.gca()                                            # get current axis 获得坐标轴对象
ax.spines['right'].set_color('none') 
ax.spines['top'].set_color('none')         # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
ax.xaxis.set_ticks_position('bottom')   
ax.yaxis.set_ticks_position('left')          # 指定下边的边作为 x 轴   指定左边的边为 y 轴
ax.spines['bottom'].set_position(('data', 0))   #指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
ax.spines['left'].set_position(('data', 0))
plt.show()
```


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
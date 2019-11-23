---
title: PyTorch笔记
date: 2019-11-23
tags: 
- python
- pytorch
categories:
- deeplearning
mathjax: true
---

# PyTorch笔记

## 数据类型Tensor

浮点型

```python
import torch
a = torch.FloatTensor(2,3) #按照指定维度随机生成浮点型tensor
b = torch.FloatTensor([2,3,4,5]) #按照给定列表生成浮点型tensor
a = torch.IntTensor(2,3) #整型同上
b = torch.IntTensor([2,3,4,5])
```

数据生成

```python
a = torch.rand(2,3) #生成0到1之间的随机浮点型tensor
a = torch.randn(2,3) #生成0到1之间的随机浮点型tensor，满足均值为0，方差为1
a = torch.arange(1,20,1) #生成一个等差数组，输入为起始值、结束值、步长
a = torch.zeros(2,3) #生成全0 tesnor
x = torch.zeros(5, 3, dtype=torch.long)
x = torch.empty(5, 3) #创建一个5x3的未初始化的Tensor
print(x.size()) #获取tensor的形状
print(x.shape)
```

## tensor运算

```python
b = torch.abs(a) #取绝对值
c = torch.add(a,b) #两tensor相加
e = torch.add(c,10) #tensor与scalar相加

```


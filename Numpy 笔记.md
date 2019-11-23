---
title: Numpy 笔记
date: 2019-11-22
tags: 
- 笔记
- 编程
- numpy
categories:
- 编程笔记
mathjax: true
---

# Numpy 笔记

## transpose

```
numpy.transpose(a, axes=None)
#修改张量的维度
>>> x = np.arange(4).reshape((2,2))
>>> x
array([[0, 1],
       [2, 3]])
>>> np.transpose(x)
array([[0, 2],
       [1, 3]])
>>> x = np.ones((1, 2, 3))
>>> np.transpose(x, (1, 0, 2)).shape
(2, 1, 3)
```

## nonzero

找到非零元素的下标




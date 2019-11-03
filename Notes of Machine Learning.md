# Notes of Machine Learning 

Min Chen Yang,  Hunan University,  2019.11

## 初识机器学习

进入21世纪，纵观机器学习发展历程，研究热点可以简单总结为2000-2006年的流形学习、2006年-2011年的稀疏学习、2012年至今的深度学习 、未来迁移学习？

人工智能的真正挑战在于解决对人来说很容易执行，但很难形式化描述的任务。

 常用的10大机器学习算法有：决策树、随机森林、逻辑回归、SVM、朴素贝叶斯、K最近邻算法、K均值算法、Adaboost算法、神经网络、马尔科夫 

大致分类：

监督式学习：决策树、KNN（K邻近）、朴素贝叶斯、逻辑回归、支持向量机

非监督式学习：聚类、主成分分析PCA

深度学习：卷积神经网络、自编码器、循环神经网络

## Deep Learning

### 初识

让计算机从经验中学习，根据层次化的概念体系来理解世界，每个概念通过与相对简单的概念之间的关系定义。

### 数学基础

一维数组，二维矩阵，三维张量

#### 雅可比Jacobian矩阵 

<img src="Notes of Machine Learning.assets/jacobian-1572707333030.PNG" style="zoom:80%;" />

#### 海森Hessian矩阵

<img src="Notes of Machine Learning.assets/hessian.PNG" style="zoom: 67%;" />

#### Logistic sigmoid 函数

$$
f(x)=\frac{1}{1+e^{-x}}
$$

<img src="Notes of Machine Learning.assets/logistic-1572783311445.PNG" style="zoom:50%;" />

输出类似sigmoid曲线
---
title: Notes of Machine Learning
date: 2019-11-16
tags: 
- ML
- DL
categories:
- ML
mathjax: true
---



# Notes of Machine Learning 

Min Chen Yang,  Hunan University,  2019.11

## 初识机器学习

**从原始数据中提取模式的能力。**

进入21世纪，纵观机器学习发展历程，研究热点可以简单总结为2000-2006年的流形学习、2006年-2011年的稀疏学习、2012年至今的深度学习 、未来迁移学习？

人工智能的真正挑战在于解决对人来说很容易执行，但很难形式化描述的任务。

 常用的10大机器学习算法有：决策树、随机森林、逻辑回归、SVM、朴素贝叶斯、K最近邻算法、K均值算法、Adaboost算法、神经网络、马尔科夫 

大致分类：

监督式学习：决策树、KNN（K邻近）、朴素贝叶斯(贝叶斯分类器）、逻辑回归、支持向量机

非监督式学习：聚类、主成分分析PCA

深度学习：卷积神经网络、自编码器、循环神经网络

## Deep Learning

### 初识

**让计算机从经验中学习，根据层次化的概念体系来理解世界，每个概念通过与相对简单的概念之间的关系定义。**

### 数学基础

一维数组，二维矩阵，三维张量

#### 雅可比Jacobian矩阵 

<img src="Notes of Machine Learning/jacobian-1572707333030.PNG" style="zoom:80%;" />

#### 海森Hessian矩阵

<img src="Notes of Machine Learning/hessian.PNG" style="zoom: 67%;" />

#### Logistic sigmoid 函数

$$
f(x)=\frac{1}{1+e^{-x}}
$$

<img src="Notes of Machine Learning/logistic-1572783311445.PNG" style="zoom:50%;" />

输出类似sigmoid曲线，常用作激活函数

#### softmax函数（归一化指数函数）

$$
\sigma(\mathbf{z})_{j}=\frac{e^{z_{j}}}{\sum_{k=1}^{K} e^{z_{k}}} \quad \text { for } j=1, \ldots, K
$$

观察到的数据属于某个类的概率,经常将其作为神经网络的输出层

#### linear algebra

标量、向量、矩阵、张量

矩阵乘积、矩阵点积

单位矩阵、逆矩阵

生成子空间（列空间、值域）、线性相关（无关）、矩阵的奇异性

范数、欧几里得范数、最大范数

对角矩阵、正交矩阵、实对称矩阵

特征值、特征向量、矩阵的特征分解、实对称矩阵特征分解来优化二次方程

### 李宏毅note

#### 一天学会深度学习

DNN:深度神经网络

CNN：卷积神经网络

RNN：递归（循环）神经网络

LSTM：长短期记忆

##### 完全连接前馈网络

定义一组函数：

<img src="Notes of Machine Learning/3-1573908546270.PNG" alt="3" style="zoom:50%;" />

- 给定参数 𝜃, 定义一个函数；给定网络结构，定义一个函数集

- <img src="Notes of Machine Learning/4-1573908942871.PNG" alt="4" style="zoom:67%;" />

- $$
  f\left(\left[\begin{array}{c}{1} \\ {-1}\end{array}\right]\right)=\left[\begin{array}{l}{0.62} \\ {0.83}\end{array}\right]
  $$

深度学习的三个步骤：

<img src="Notes of Machine Learning/frame.PNG" style="zoom:67%;" />

模型和数据的拟合：

- 准备训练数据—图片及其标签
- softmax层作为输出层，正确结果应对应输出层的最大值

最优函数的选择：

- 神经网络的两个参数：权重和偏差
- 损失是神经网络输出和目标之间的距离

- 最优的函数：找到使总损失最小的参数 𝜽

- 关键计算算是关于参数的偏导

<u>选择的方法：梯度下降</u>       Backpropagation？

- 选取W的初始值,计算$$w \leftarrow w-\eta \partial L/ \partial w$$,直到$$\partial L/ \partial w$$足够小
- 不能保证全局最小值，不同的初始点到达不同的初始值
- 反向传播是一种有效的计算$\partial L/ \partial w$的方法—深度学习框架
- <img src="Notes of Machine Learning/15.PNG" style="zoom:50%;" />
- $$\eta$$可认为其参数值更新的快慢，即学习率

##### 为什么使用深度神经网络

对于同样个数的神经元数量，浅胖或者深瘦？

**EX**：深瘦：长短发，男孩女孩，浅胖：长发男孩、长发女孩、短法男孩、短法女孩

深瘦: 每个基础分类器都可以有足够的训练实例   浅胖：长发男孩样本少，没有足够的训练实例

<img src="Notes of Machine Learning/2.PNG" style="zoom: 50%;" />

##### 神经网络变体

###### 卷积神经网络（CNN)

**为什么CNN用于图像识别**

1. 当处理图像时，全连接网络的第一层将会非常大  
2. 一些模式比整张图片小得多（如：识别鸟嘴即可识别出鸟），神经元通过较少的参数连接到小区域去发现模式
3. 同样的模式可能出现在图像的不同区域，不同区域的神经元可能相同
4. 对图像进行二次采样不会改变图像中的物体、使图像变小（获取图像压缩比例、加载图像缩略图、避免图片加载时的OOM异常 ）
5. <img src="Notes of Machine Learning/7.PNG" alt="7" style="zoom:50%;" />

**滤波器**

给定宽度和高度的滤波器，不同滤波器提取一个 patch 的不同特性。例如，一个滤波器寻找特定颜色，另一个寻找特定物体的特定形状。卷积层滤波器的数量被称为滤波器深度。

**图像的最大池化（Max Pooling）**

最大池化是基于样本的离散化过程。目的是对输入表示（图像，隐藏层输出矩阵等）进行下采样，以减小其尺寸，并允许对合并的子区域中包含的特征进行假设.

 通过提供表示形式的抽象形式来帮助过度拟合。同样，它通过减少学习参数的数量来减少计算成本，并为内部表示提供基本的平移不变性 。

<img src="Notes of Machine Learning/5.PNG" alt="5" style="zoom:30%;" />

**CNN整体结构**

<img src="Notes of Machine Learning/6.PNG" alt="6" style="zoom:40%;" />

###### 循环神经网络（RNN)

**插槽填充**

输入序列向量化，将每一个输入的单词用向量表示，可以使用 One-of-N Encoding 或者是 Word hashing 等编码方法，输出预测槽位的概率分布

**1.1-of-N encoding**

向量是词典大小；每个维度对应于词典中的一个单词；这个词的维数是1，其他的是0；为了表示一些不知道的词汇，加入other这个维度

```
词典 = {apple, bag, cat, dog, elephant}
```

**2.Word hashing**（字串比对）

<img src="Notes of Machine Learning/8.PNG" alt="8" style="zoom: 50%;" />

不同的序列如：arrive Taipei和leave Taipei，此时Taipei应该被放入不同的slot

因此引入具有记忆属性的RNN

**3.RNN（Elman Network）**

<img src="Notes of Machine Learning/9.PNG" alt="9" style="zoom:50%;" />

变式：深层、双向（可以考虑整个sequence的input）、Jordan Network

<img src="Notes of Machine Learning/10.PNG" alt="10" style="zoom:50%;" />

输出是有目标的，因此Jordan Network表现更好

**4.LSTM（长短期记忆）**

一种特殊神经元机构：4 inputs，1 output，memory cell

<img src="Notes of Machine Learning/11.PNG" alt="11" style="zoom:50%;" />

<img src="Notes of Machine Learning/12.PNG" alt="12" style="zoom:50%;" />

图中$$c^{\prime}=g(z) f\left(z_{\mathrm{i}}\right)+c f\left(z_{f}\right)$$,为memory cell的记忆更新值，激活函数使用sigmoid function，输出表示门的打开和关闭。输入向量点乘相应的权重向量得到输入、输出、以及遗忘“门”的输入通过激活函数的输出来决定“门”的开闭。

**5.Multiple-layer  LSTM**

<img src="Notes of Machine Learning/13.png" style="zoom: 50%;" />

一般的LSTM还靖上一层LSTM的memory cell的值、output与当前时刻的input合并为一个新的向量，再点乘相应权重向量，来控制该层LSTM。

Keras 支持LSTM、GRU（Gated Recurrent Unit、two gates、参数量少1/3、旧的不去新的不来）、SimpleRNN（memory不断被洗掉）

**6.RNN 训练技巧**

<img src="Notes of Machine Learning/14.PNG" style="zoom: 67%;" />

原因：memory cell不断与权重乘积

导致的问题：

1. 当$\partial L/ \partial w$非常大时，梯度下降的参数更新会将参数抛出很远，导致lost异常增大，因此应选择衰减速度足够慢的学习率，避免上坡运动。
2. 会出现梯度消失

**解决方法：**

**截断梯度**：在参数更新前，逐元素地截断**小批量产生的参数梯度**或**截断梯度$$g$$的范数**
$$
\begin{array}{c}{\text { if }\|g\|>v} \\ {g \leftarrow \frac{g v}{\|g\|}}\end{array}
$$
**LSTM**:可以解决梯度消失 (不能解决梯度爆炸)

1. 当forget gate打开时梯度永远不会消失

GRU可将input gate与forget gate联动，只有一个gate可打开，只有将memory cell清掉，才能输入；或者没有输入时，memory cell才能更新

**clockwise RNN、Structurally Constrained Recurrent Network (SCRN)**

**7.应用**

1. many to one :setiment  analysis(情感分析)

2. many to many（output is shorter）:语音识别，通过Connectionist Temporal Classification (CTC)来剔除重复的声音vector对应的输出（添加null来隔开相同的输出）

3. many to many（No limition、sequence to sequence):机器翻译，EX：根据中文声音信号转成英文文字

4. beyond sequence: syntactic parsing(语法分析)

5. auto-encoder:**Text**:考虑文件或语句的顺序将其变为向量 **Speech**：将声音信号编码为向量

   <img src="Notes of Machine Learning/16.PNG" style="zoom:50%;" />

   此时编码器和解码器同时工作，目标是使经过编码和解码后的输出与输入尽量相似。

###### 注意力集中模型

Attention-based Model、Neural Turing machine

<img src="Notes of Machine Learning/17.PNG" style="zoom:60%;" />

**应用**

Reading Comprehension、Visual Question Answering

EX：Speech Question Answering

<img src="Notes of Machine Learning/18.png" alt="image-20191119214456154" style="zoom:50%;" />

#### RNN vs  Structured Learning

### 深度学习框架

theano: 蒙特利尔大学蒙特利尔学习算法研究所开发 

caffe: CAFFE是一个深度学习框架，最初开发于加利福尼亚大学伯克利分校。Caffe在BSD许可下开源，使用C++编写，带有Python接口 

tensorflow: TensorFlow是一个开源软件库，用于各种感知和语言理解任务的机器学习 

keras: Keras是一个用Python编写的开源神经网络库，能够在TensorFlow、Microsoft Cognitive Toolkit、Theano或PlaidML之上运行 